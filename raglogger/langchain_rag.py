"""LangChain RAG engine: retriever -> prompt template -> chat model chain, with full interaction logging.

Every call runs with :class:`LlmInteractionCallback`, so the retrieved context, the complete prompt messages and the
model response are written to the transcript log and the JSONL audit log.
"""
from __future__ import annotations

import logging
import os
from typing import Any, Optional

from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.language_models import BaseChatModel
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

from raglogger.config import Settings
from raglogger.generators import (SYSTEM_PROMPT, AnswerGenerator, GenerationError, build_context,
                                  is_temperature_rejection_text)
from raglogger.index import Bm25Index
from raglogger.models import SearchFilters, SearchHit
from raglogger.redaction import MASK, Redactor
from raglogger.vectorstore import HybridLogRetriever

LOG = logging.getLogger(__name__)
_REDACTOR = Redactor()

PROMPT = ChatPromptTemplate.from_messages([
    ("system", SYSTEM_PROMPT),
    ("human", "Context excerpts from the console log:\n\n{context}\n\nQuestion: {question}"),
])


def create_chat_model(backend: str, settings: Settings) -> BaseChatModel:
    resolved = settings.resolve(backend)
    temperature = settings.temperature_for(backend)  # None -> parameter is not sent at all
    if not resolved.model:
        raise ValueError(f"No model configured for backend '{backend}'. Pass --model or set "
                         f"{backend.upper()}_MODEL (run 'raglogger models --backend {backend}' to list them).")
    if backend == "ollama":
        from langchain_ollama import ChatOllama
        return ChatOllama(model=resolved.model, base_url=resolved.base_url, temperature=temperature,
                          num_predict=settings.max_tokens, client_kwargs={"timeout": settings.timeout_seconds})
    if backend in ("litellm", "openai"):
        from langchain_openai import ChatOpenAI
        api_key = os.environ.get(resolved.api_key_env, "").strip()
        if backend == "litellm" and not api_key:
            raise ValueError(f"{resolved.api_key_env} is not set; export your LiteLLM virtual key first")
        # LiteLLM speaks the OpenAI protocol, so ChatOpenAI pointed at the gateway is the supported client.
        return ChatOpenAI(model_name=resolved.model, openai_api_base=resolved.base_url,
                          openai_api_key=api_key or "not-needed", temperature=temperature,
                          max_tokens=settings.max_tokens, request_timeout=settings.timeout_seconds, max_retries=1)
    raise ValueError(f"Backend '{backend}' is not an LLM backend")


def is_temperature_rejection(error: Exception) -> bool:
    """True for HTTP 400/422 errors that complain about the temperature value (see generators)."""
    status = getattr(error, "status_code", None)
    return status in (None, 400, 422) and is_temperature_rejection_text(str(error))


def without_temperature(model: BaseChatModel) -> Optional[BaseChatModel]:
    """Copy of ``model`` that omits temperature from requests, or None if it was not being sent anyway."""
    if getattr(model, "temperature", None) is None:
        return None
    return model.model_copy(update={"temperature": None})


def scrub_secrets(text: str) -> str:
    """Masks configured API keys plus anything matching the generic secret patterns (defence in depth)."""
    for env_name in ("LITELLM_API_KEY", "RAGLOGGER_API_KEY"):
        key = os.environ.get(env_name, "").strip()
        if key:
            text = text.replace(key, MASK)
    return _REDACTOR.redact(text)[:600]

class LangChainChainGenerator(AnswerGenerator):
    """Runs ``PROMPT | model | StrOutputParser`` for each configured backend in order until one succeeds."""

    def __init__(self, models: list[tuple[str, str, BaseChatModel]], callbacks: list[BaseCallbackHandler],
                 max_context_chars: int, model_parameters: Optional[dict[str, Any]] = None) -> None:
        if not models:
            raise ValueError("At least one LLM backend is required")
        self._models = models
        self._callbacks = callbacks
        self._max_context_chars = max_context_chars
        self._model_parameters = model_parameters or {}
        self.name = "+".join(name for name, _, _ in models)

    def generate(self, question: str, hits: list[SearchHit]) -> str:
        return self.answer(question, hits)[0]

    def answer(self, question: str, hits: list[SearchHit]) -> tuple[str, str]:
        context = build_context(hits, self._max_context_chars)
        failures: list[str] = []
        for backend, model_name, model in self._models:
            try:
                text = self._invoke(backend, model_name, model, context, question, hits)
            except Exception as error:  # noqa: BLE001 - provider SDKs raise many unrelated exception types
                message = scrub_secrets(f"{type(error).__name__}: {error}")
                LOG.warning("Backend '%s' failed: %s", backend, message)
                failures.append(f"{backend}: {message}")
                continue
            if text and text.strip():
                return text, backend
            failures.append(f"{backend}: empty answer")
        raise GenerationError("all backends failed -> " + " | ".join(failures))

    def _invoke(self, backend: str, model_name: str, model: BaseChatModel, context: str, question: str,
                hits: list[SearchHit]) -> str:
        try:
            return self._run_chain(backend, model_name, model, context, question, hits)
        except Exception as error:  # noqa: BLE001
            retry_model = without_temperature(model) if is_temperature_rejection(error) else None
            if retry_model is None:
                raise
            # e.g. Claude (extended thinking) via LiteLLM: "does not support temperature=0.1, only 1 is supported".
            LOG.warning("Backend '%s' rejected temperature=%s; retrying once without it. Set %s_TEMPERATURE=omit "
                        "to skip this round-trip.", backend, getattr(model, "temperature", None), backend.upper())
            return self._run_chain(backend, model_name, retry_model, context, question, hits,
                                   temperature_override="omitted (auto-retry)")

    def _run_chain(self, backend: str, model_name: str, model: BaseChatModel, context: str, question: str,
                   hits: list[SearchHit], temperature_override: Optional[str] = None) -> str:
        parameters = {**self._model_parameters, "temperature": getattr(model, "temperature", None)}
        if temperature_override:
            parameters["temperature"] = temperature_override
        config: dict[str, Any] = {
            "callbacks": self._callbacks, "run_name": f"raglogger-{backend}", "tags": ["raglogger", backend],
            "metadata": {"backend": backend, "model": model_name, "question": question,
                         "chunk_ids": [hit.chunk.chunk_id for hit in hits], **parameters},
        }
        chain = PROMPT | model | StrOutputParser()
        return chain.invoke({"context": context, "question": question}, config=config)


class LangChainRetrieverAdapter:
    """Exposes the LangChain hybrid retriever through the ``search(query, top_k, filters)`` contract."""

    def __init__(self, vector_store: Any, bm25: Bm25Index, callbacks: list[BaseCallbackHandler],
                 fetch_k: int = 30) -> None:
        self._vector_store = vector_store
        self._bm25 = bm25
        self._callbacks = callbacks
        self._fetch_k = fetch_k
        self._by_id = {chunk.chunk_id: chunk for chunk in bm25.chunks}

    def search(self, query: str, top_k: int = 6, filters: Optional[SearchFilters] = None) -> list[SearchHit]:
        retriever = HybridLogRetriever(vector_store=self._vector_store, bm25=self._bm25, k=top_k,
                                       fetch_k=max(self._fetch_k, top_k * 3), filters=filters or SearchFilters())
        documents = retriever.invoke(query, config={"callbacks": self._callbacks,
                                                    "run_name": "raglogger-retriever"})
        hits = []
        for document in documents:
            chunk = self._by_id.get(document.metadata.get("chunk_id"))
            if chunk is not None:
                hits.append(SearchHit(chunk=chunk, score=document.metadata.get("score", 0.0),
                                      retrieval=document.metadata.get("retrieval", "hybrid")))
        return hits
