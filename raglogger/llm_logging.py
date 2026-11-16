"""Transcript log configuration (stdlib only) plus the LangChain callback handler that records every LLM call.

Two sinks are written for each call:

* a human-readable, size-rotated transcript (``<project>/.raglogger/logs/llm-transcript.log``) containing the retrieved
  context documents, every prompt message (system + user incl. the full context block) and the full response;
* the structured JSONL audit log (``llm-interactions.jsonl``) via :class:`InteractionLogger`.

All text is passed through :class:`Redactor` before it is written. Logging failures never break the request.
"""
from __future__ import annotations

import logging
import sys
import time
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Any, Optional
from uuid import UUID

from raglogger.interaction_log import InteractionLogger
from raglogger.paths import default_transcript_path, ensure_parent
from raglogger.redaction import Redactor

try:  # LangChain is optional for the native engine; the transcript logger itself has no dependencies.
    from langchain_core.callbacks import BaseCallbackHandler
    from langchain_core.documents import Document
    from langchain_core.messages import BaseMessage
    from langchain_core.outputs import LLMResult
except ImportError:  # pragma: no cover - exercised only without the LangChain requirements installed
    BaseCallbackHandler = object  # type: ignore[assignment,misc]
    Document = BaseMessage = LLMResult = Any  # type: ignore[assignment,misc]

DEFAULT_TRANSCRIPT_PATH = default_transcript_path()
TRANSCRIPT_LOGGER_NAME = "raglogger.llm.transcript"
_MAX_BYTES = 10 * 1024 * 1024
_BACKUP_COUNT = 5


def configure_transcript_logger(path: Optional[Path]) -> logging.Logger:
    """Returns the dedicated transcript logger with a single rotating file handler for ``path`` (idempotent)."""
    logger = logging.getLogger(TRANSCRIPT_LOGGER_NAME)
    logger.setLevel(logging.INFO)
    logger.propagate = False  # never duplicate full prompts onto the console / root logger
    target = str(path.resolve()) if path else None
    for handler in list(logger.handlers):
        if getattr(handler, "baseFilename", None) != target:
            logger.removeHandler(handler)
            handler.close()
    if path is not None and not logger.handlers:
        try:
            ensure_parent(path)
            handler = RotatingFileHandler(path, maxBytes=_MAX_BYTES, backupCount=_BACKUP_COUNT, encoding="utf-8")
            handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
            logger.addHandler(handler)
        except OSError as error:
            print(f"[raglogger] WARNING: transcript log unavailable ({error})", file=sys.stderr)
    if not logger.handlers:
        logger.addHandler(logging.NullHandler())
    return logger

class LlmInteractionCallback(BaseCallbackHandler):
    """Logs retriever results, full chat prompts, responses, token usage, latency and errors."""

    raise_error = False  # a logging problem must never abort the LLM call

    def __init__(self, transcript: logging.Logger, audit: InteractionLogger, redactor: Redactor) -> None:
        self._transcript = transcript
        self._audit = audit
        self._redactor = redactor
        self._started: dict[UUID, float] = {}

    # ------------------------------------------------------------------ retrieval context
    def on_retriever_end(self, documents: list[Document], *, run_id: UUID, **kwargs: Any) -> None:
        lines = [f"===== RETRIEVED CONTEXT run={run_id} documents={len(documents)}"]
        for number, document in enumerate(documents, start=1):
            meta = document.metadata
            lines.append(f"--- [{number}] id={meta.get('chunk_id')} kind={meta.get('kind')} "
                         f"source={meta.get('source')} score={meta.get('score')} "
                         f"retrieval={meta.get('retrieval')} ---")
            lines.append(document.page_content)
        self._emit("\n".join(lines), {"type": "retrieval", "runId": str(run_id),
                                      "documents": [{"metadata": doc.metadata, "content": doc.page_content}
                                                    for doc in documents]})

    # ------------------------------------------------------------------ LLM input
    def on_chat_model_start(self, serialized: dict[str, Any], messages: list[list[BaseMessage]], *, run_id: UUID,
                            metadata: Optional[dict[str, Any]] = None, **kwargs: Any) -> None:
        self._started[run_id] = time.monotonic()
        meta = metadata or {}
        params = kwargs.get("invocation_params") or {}
        flat = [message for batch in messages for message in batch]
        model = meta.get("model") or params.get("model") or params.get("model_name")
        max_tokens = (meta.get("max_tokens") or params.get("max_tokens") or params.get("max_completion_tokens")
                      or params.get("num_predict"))
        temperature = meta.get("temperature", params.get("temperature"))
        if temperature is None:
            temperature = "omitted"
        header = (f"===== LLM REQUEST run={run_id} session={self._audit.session_id} backend={meta.get('backend')} "
                  f"model={model} temperature={temperature} max_tokens={max_tokens}")
        body = [header] + [f"--- [{message.type}] ---\n{message.content}" for message in flat]
        self._emit("\n".join(body), {
            "type": "request", "runId": str(run_id), "backend": meta.get("backend"), "model": model,
            "parameters": {"temperature": temperature, "max_tokens": max_tokens},
            "question": meta.get("question"), "contextChunkIds": meta.get("chunk_ids"),
            "messages": [{"role": message.type, "content": message.content} for message in flat],
        })

    def on_llm_start(self, serialized: dict[str, Any], prompts: list[str], *, run_id: UUID,
                     metadata: Optional[dict[str, Any]] = None, **kwargs: Any) -> None:
        # Completion-style (non-chat) models.
        self._started[run_id] = time.monotonic()
        backend = (metadata or {}).get("backend")
        self._emit(f"===== LLM REQUEST run={run_id} backend={backend}\n" + "\n".join(prompts),
                   {"type": "request", "runId": str(run_id), "backend": backend, "prompts": prompts})

    # ------------------------------------------------------------------ LLM output
    def on_llm_end(self, response: LLMResult, *, run_id: UUID, **kwargs: Any) -> None:
        duration_ms = self._duration(run_id)
        texts = [generation.text for batch in response.generations for generation in batch]
        tokens = self._token_usage(response)
        header = (f"===== LLM RESPONSE run={run_id} duration_ms={duration_ms} tokens(prompt/completion/total)="
                  f"{tokens.get('prompt')}/{tokens.get('completion')}/{tokens.get('total')}")
        self._emit("\n".join([header, *texts]),
                   {"type": "response", "runId": str(run_id), "response": "\n".join(texts),
                    "durationMs": duration_ms, "tokens": tokens})

    def on_llm_error(self, error: BaseException, *, run_id: UUID, **kwargs: Any) -> None:
        duration_ms = self._duration(run_id)
        message = f"{type(error).__name__}: {error}"
        self._emit(f"===== LLM ERROR run={run_id} duration_ms={duration_ms}\n{message}",
                   {"type": "response", "runId": str(run_id), "response": "", "durationMs": duration_ms,
                    "tokens": {}, "error": message}, level=logging.ERROR)

    # ------------------------------------------------------------------ helpers
    def _duration(self, run_id: UUID) -> Optional[int]:
        started = self._started.pop(run_id, None)
        return int((time.monotonic() - started) * 1000) if started is not None else None

    @staticmethod
    def _token_usage(response: LLMResult) -> dict[str, Optional[int]]:
        for batch in response.generations:
            for generation in batch:
                usage = getattr(getattr(generation, "message", None), "usage_metadata", None)
                if usage:
                    return {"prompt": usage.get("input_tokens"), "completion": usage.get("output_tokens"),
                            "total": usage.get("total_tokens")}
        usage = (response.llm_output or {}).get("token_usage") or {}
        return {"prompt": usage.get("prompt_tokens"), "completion": usage.get("completion_tokens"),
                "total": usage.get("total_tokens")}

    def _emit(self, text: str, record: dict[str, Any], level: int = logging.INFO) -> None:
        try:
            self._transcript.log(level, self._redactor.redact(text))
        except Exception as error:  # noqa: BLE001 - logging must never break the workflow
            print(f"[raglogger] WARNING: could not write LLM transcript: {error}", file=sys.stderr)
        self._audit.write_record(record)  # redacts internally and reports its own failures
