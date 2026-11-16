"""Answer generators: an offline extractive default plus optional Ollama / OpenAI-compatible LLM backends.

LLM backends use only ``urllib`` from the standard library. Credentials are read from environment variables at
call time and are never persisted or logged.
"""
from __future__ import annotations

import json
import logging
import os
import re
import time
import urllib.error
import urllib.request
from abc import ABC, abstractmethod
from typing import Optional

from raglogger.interaction_log import InteractionLogger
from raglogger.models import SearchHit
from raglogger.redaction import MASK

LOG = logging.getLogger(__name__)

SYSTEM_PROMPT = (
    "You are a senior SAP Commerce (Hybris) support engineer analysing a server console log. "
    "Answer ONLY from the provided context excerpts. Cite excerpts as [1], [2], ... "
    "Quote exact logger names, exception classes, timestamps and occurrence counts. "
    "If the context is insufficient, say so explicitly and suggest what to search for next. "
    "Finish with probable root cause and recommended next steps when the question is about a problem."
)


class GenerationError(RuntimeError):
    """Raised when an LLM backend cannot produce an answer."""


_TEMPERATURE_WORD = re.compile(r"temperature", re.IGNORECASE)
_REJECTION_HINTS = re.compile(r"unsupported|not support|only temperature|invalid|must be|not allowed|deprecated",
                              re.IGNORECASE)


def is_temperature_rejection_text(text: str) -> bool:
    """True for provider errors like "model X does not support temperature=0.1. Only temperature=1 is supported"."""
    return bool(_TEMPERATURE_WORD.search(text)) and bool(_REJECTION_HINTS.search(text))


def build_context(hits: list[SearchHit], max_chars: int) -> str:
    """Formats retrieved chunks into a numbered context block bounded by ``max_chars``."""
    parts: list[str] = []
    used = 0
    for number, hit in enumerate(hits, start=1):
        chunk = hit.chunk
        block = (f"[{number}] {chunk.kind.upper()} | {chunk.title}\n"
                 f"source {chunk.source or '-'} | time {chunk.start_time} .. {chunk.end_time} | "
                 f"lines {chunk.line_start}-{chunk.line_end}\n"
                 f"{chunk.text}\n")
        if used + len(block) > max_chars:
            remaining = max_chars - used
            if remaining > 200:
                parts.append(block[:remaining] + "\n...[truncated]\n")
            break
        parts.append(block)
        used += len(block)
    return "\n".join(parts)


class AnswerGenerator(ABC):
    name = "base"

    @abstractmethod
    def generate(self, question: str, hits: list[SearchHit]) -> str:
        """Produces an answer for ``question`` grounded in ``hits``."""

    def answer(self, question: str, hits: list[SearchHit]) -> tuple[str, str]:
        """Returns ``(answer_text, backend_label)``; chains override this to report which backend answered."""
        return self.generate(question, hits), self.name


class ExtractiveGenerator(AnswerGenerator):
    """Offline default: presents the most relevant evidence without calling any model."""

    name = "extractive"

    def __init__(self, max_lines_per_hit: int = 14) -> None:
        self._max_lines_per_hit = max_lines_per_hit

    def generate(self, question: str, hits: list[SearchHit]) -> str:
        if not hits:
            return ("No relevant log content was found. Try different keywords (logger name, exception class, "
                    "cart/order id) or relax the --level/--since/--until filters.")
        lines = [f"Top {len(hits)} relevant log excerpts for: \"{question}\"", ""]
        for number, hit in enumerate(hits, start=1):
            chunk = hit.chunk
            lines.append(f"[{number}] {chunk.title}")
            lines.append(f"    kind={chunk.kind} score={hit.score} occurrences={chunk.occurrences} "
                         f"source={chunk.source or '-'} time={chunk.start_time}..{chunk.end_time} "
                         f"lines={chunk.line_start}-{chunk.line_end}")
            body = chunk.text.splitlines()
            lines.extend(f"    {line}" for line in body[:self._max_lines_per_hit])
            if len(body) > self._max_lines_per_hit:
                lines.append(f"    ... ({len(body) - self._max_lines_per_hit} more lines)")
            lines.append("")
        lines.append("Tip: use --backend ollama or --backend openai for a synthesised natural-language answer.")
        return "\n".join(lines)


def _elapsed(started: float) -> int:
    return int((time.monotonic() - started) * 1000)

class _HttpLlmGenerator(AnswerGenerator):
    """Shared HTTP plumbing for chat-style LLM endpoints."""

    def __init__(self, model: str, base_url: str, interaction_logger: InteractionLogger, max_context_chars: int,
                 temperature: Optional[float], timeout_seconds: int) -> None:
        self._model = model
        self._base_url = base_url.rstrip("/")
        self._interaction_logger = interaction_logger
        self._max_context_chars = max_context_chars
        self._temperature = temperature  # None -> not sent; the provider's default applies
        self._timeout_seconds = timeout_seconds

    def generate(self, question: str, hits: list[SearchHit]) -> str:
        context = build_context(hits, self._max_context_chars)
        user_prompt = f"Context excerpts from the console log:\n\n{context}\n\nQuestion: {question}"
        messages = [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": user_prompt}]
        try:
            return self._logged_call(question, user_prompt, messages, hits, self._temperature)
        except GenerationError as error:
            if self._temperature is None or not is_temperature_rejection_text(str(error)):
                raise
            # e.g. Claude (extended thinking) via LiteLLM only accepts its default temperature.
            LOG.warning("Backend '%s' rejected temperature=%s; retrying once without it. Set %s_TEMPERATURE=omit "
                        "to skip this round-trip.", self.name, self._temperature, self.name.upper())
            return self._logged_call(question, user_prompt, messages, hits, None)

    def _logged_call(self, question: str, user_prompt: str, messages: list[dict], hits: list[SearchHit],
                     temperature: Optional[float]) -> str:
        request_id = self._interaction_logger.log_request(
            model=self._model, system_prompt=SYSTEM_PROMPT, user_prompt=question, assembled_prompt=user_prompt,
            parameters={"temperature": temperature if temperature is not None else "omitted", "backend": self.name},
            context=[hit.chunk.chunk_id for hit in hits])
        started = time.monotonic()
        try:
            answer, usage = self._call(messages, temperature)
        except GenerationError as error:
            self._interaction_logger.log_response(request_id, response="", duration_ms=_elapsed(started),
                                                  tokens={}, error=str(error))
            raise
        self._interaction_logger.log_response(request_id, response=answer, duration_ms=_elapsed(started),
                                              tokens=usage)
        return answer

    @abstractmethod
    def _call(self, messages: list[dict], temperature: Optional[float]) -> tuple[str, dict]:
        """Calls the endpoint and returns ``(answer, token_usage)``; ``temperature=None`` must omit the field."""

    def _post(self, url: str, payload: dict, headers: Optional[dict] = None) -> dict:
        if not url.lower().startswith(("http://", "https://")):
            raise GenerationError(f"Refusing non-HTTP endpoint: {url}")
        request = urllib.request.Request(url, data=json.dumps(payload).encode("utf-8"), method="POST",
                                         headers={"Content-Type": "application/json", **(headers or {})})
        try:
            with urllib.request.urlopen(request, timeout=self._timeout_seconds) as response:  # noqa: S310
                return json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as error:
            detail = error.read().decode("utf-8", errors="replace")[:300]
            raise GenerationError(f"{self.name} endpoint returned HTTP {error.code}: {detail}") from error
        except (urllib.error.URLError, TimeoutError, OSError) as error:
            raise GenerationError(f"Could not reach {self.name} endpoint at {self._base_url}: {error}") from error
        except json.JSONDecodeError as error:
            raise GenerationError(f"{self.name} endpoint returned invalid JSON") from error


class OllamaGenerator(_HttpLlmGenerator):
    """Local model served by Ollama (``ollama serve``). Data never leaves the machine."""

    name = "ollama"

    def _call(self, messages: list[dict], temperature: Optional[float]) -> tuple[str, dict]:
        payload = {"model": self._model, "messages": messages, "stream": False}
        if temperature is not None:
            payload["options"] = {"temperature": temperature}
        data = self._post(f"{self._base_url}/api/chat", payload)
        answer = (data.get("message") or {}).get("content", "")
        if not answer:
            raise GenerationError("Ollama returned an empty answer")
        prompt_tokens = data.get("prompt_eval_count")
        completion_tokens = data.get("eval_count")
        total = (prompt_tokens or 0) + (completion_tokens or 0) if (prompt_tokens or completion_tokens) else None
        return answer, {"prompt": prompt_tokens, "completion": completion_tokens, "total": total}


class OpenAiCompatibleGenerator(_HttpLlmGenerator):
    """Any OpenAI-compatible ``/chat/completions`` endpoint (OpenAI, Azure proxy, vLLM, LM Studio, ...)."""

    name = "openai"
    API_KEY_ENV = "RAGLOGGER_API_KEY"

    def __init__(self, model: str, base_url: str, interaction_logger: InteractionLogger, max_context_chars: int,
                 temperature: Optional[float], timeout_seconds: int, max_tokens: int = 2048,
                 api_key_env: Optional[str] = None) -> None:
        super().__init__(model, base_url, interaction_logger, max_context_chars, temperature, timeout_seconds)
        self._max_tokens = max_tokens
        self._api_key_env = api_key_env or self.API_KEY_ENV

    def _api_key(self) -> str:
        # Read lazily on every call so the key is never stored on the object or in settings.
        return os.environ.get(self._api_key_env, "").strip()

    def _auth_headers(self) -> dict:
        api_key = self._api_key()
        return {"Authorization": f"Bearer {api_key}"} if api_key else {}

    def _call(self, messages: list[dict], temperature: Optional[float]) -> tuple[str, dict]:
        payload = {"model": self._model, "messages": messages, "max_tokens": self._max_tokens}
        if temperature is not None:
            payload["temperature"] = temperature
        data = self._post(f"{self._base_url}/chat/completions", payload, self._auth_headers())
        try:
            answer = data["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as error:
            raise GenerationError(f"Unexpected response shape from {self.name} endpoint") from error
        if not answer:
            raise GenerationError(f"{self.name} returned an empty answer")
        usage = data.get("usage") or {}
        return answer, {"prompt": usage.get("prompt_tokens"), "completion": usage.get("completion_tokens"),
                        "total": usage.get("total_tokens")}

    def _post(self, url: str, payload: dict, headers: Optional[dict] = None) -> dict:
        try:
            return super()._post(url, payload, headers)
        except GenerationError as error:
            # Some gateways echo request headers in error bodies; never let the key reach logs or the console.
            raise GenerationError(self._scrub(str(error))) from error.__cause__

    def _scrub(self, text: str) -> str:
        api_key = self._api_key()
        return text.replace(api_key, MASK) if api_key else text

    def list_models(self) -> list[str]:
        """Returns model ids exposed by the endpoint (``GET /models``)."""
        url = f"{self._base_url}/models"
        if not url.lower().startswith(("http://", "https://")):
            raise GenerationError(f"Refusing non-HTTP endpoint: {url}")
        request = urllib.request.Request(url, method="GET", headers=self._auth_headers())
        try:
            with urllib.request.urlopen(request, timeout=self._timeout_seconds) as response:  # noqa: S310
                data = json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as error:
            detail = self._scrub(error.read().decode("utf-8", errors="replace")[:300])
            raise GenerationError(f"{self.name} /models returned HTTP {error.code}: {detail}") from error
        except (urllib.error.URLError, TimeoutError, OSError, json.JSONDecodeError) as error:
            raise GenerationError(self._scrub(f"Could not list models from {self._base_url}: {error}")) from error
        return sorted(item.get("id", "") for item in data.get("data", []) if item.get("id"))


class LiteLlmGenerator(OpenAiCompatibleGenerator):
    """LiteLLM proxy (OpenAI-compatible gateway in front of Claude, GPT, Bedrock, ...).

    The virtual key is read from ``LITELLM_API_KEY`` and sent as ``Authorization: Bearer <key>``.
    """

    name = "litellm"
    API_KEY_ENV = "LITELLM_API_KEY"

    def _call(self, messages: list[dict], temperature: Optional[float]) -> tuple[str, dict]:
        if not self._api_key():
            raise GenerationError(f"{self._api_key_env} is not set; export your LiteLLM virtual key first")
        return super()._call(messages, temperature)


class FallbackChainGenerator(AnswerGenerator):
    """Tries each backend in order (e.g. litellm -> ollama) and returns the first successful answer."""

    def __init__(self, generators: list[AnswerGenerator]) -> None:
        if not generators:
            raise ValueError("At least one backend is required")
        self._generators = generators
        self.name = "+".join(generator.name for generator in generators)

    def generate(self, question: str, hits: list[SearchHit]) -> str:
        return self.answer(question, hits)[0]

    def answer(self, question: str, hits: list[SearchHit]) -> tuple[str, str]:
        failures: list[str] = []
        for generator in self._generators:
            try:
                return generator.generate(question, hits), generator.name
            except GenerationError as error:
                LOG.warning("Backend '%s' failed: %s", generator.name, error)
                failures.append(f"{generator.name}: {error}")
        raise GenerationError("all backends failed -> " + " | ".join(failures))
