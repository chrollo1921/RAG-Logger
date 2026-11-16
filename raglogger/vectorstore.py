"""Embeddings + Chroma vector store + hybrid (vector + BM25) LangChain retriever.

* ``local``   - HuggingFace sentence-transformers on CPU (default ``all-MiniLM-L6-v2``); nothing leaves the machine.
* ``litellm`` - an embedding model exposed by the LiteLLM gateway (OpenAI-compatible ``/embeddings``).

Retrieval fuses vector similarity and BM25 with Reciprocal Rank Fusion so exact identifiers (logger names,
exception classes, cart ids) keep working while paraphrased questions benefit from semantic matching.
"""
from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.retrievers import BaseRetriever
from pydantic import ConfigDict

from raglogger.index import Bm25Index
from raglogger.models import Chunk, SearchFilters, SearchHit
from raglogger.paths import ensure_dir

LOG = logging.getLogger(__name__)

# Chroma enables anonymous PostHog telemetry by default and its client is broken on chromadb 0.6 (noisy errors).
# Production log tooling must not phone home: disable it before chromadb is imported anywhere.
os.environ.setdefault("ANONYMIZED_TELEMETRY", "False")
logging.getLogger("chromadb.telemetry.product.posthog").setLevel(logging.CRITICAL)

COLLECTION_NAME = "console_log_chunks"
MANIFEST_NAME = "raglogger-manifest.json"
EMBEDDING_PROVIDERS = ("local", "litellm")
DEFAULT_LOCAL_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
RRF_K = 60  # standard Reciprocal Rank Fusion constant
_BATCH_SIZE = 256


class EmbeddingConfigError(ValueError):
    """Raised for missing/invalid embedding configuration or an index built with a different embedding model."""


@dataclass
class EmbeddingSettings:
    provider: str = "local"
    model: str = ""
    base_url: str = ""
    api_key_env: str = "LITELLM_API_KEY"
    timeout_seconds: int = 120

    @property
    def resolved_model(self) -> str:
        return self.model or (DEFAULT_LOCAL_MODEL if self.provider == "local" else "")


def create_embeddings(settings: EmbeddingSettings) -> Embeddings:
    if settings.provider not in EMBEDDING_PROVIDERS:
        raise EmbeddingConfigError(f"Unknown embeddings provider '{settings.provider}'. Use local or litellm.")
    if settings.provider == "local":
        try:
            from langchain_huggingface import HuggingFaceEmbeddings
        except ImportError as error:
            raise EmbeddingConfigError("Local embeddings need: .venv/bin/pip install -r "
                                       "requirements-local-embeddings.txt") from error
        LOG.info("Loading local embedding model %s (CPU)", settings.resolved_model)
        return HuggingFaceEmbeddings(model_name=settings.resolved_model, model_kwargs={"device": "cpu"},
                                     encode_kwargs={"normalize_embeddings": True, "batch_size": 64})

    from langchain_openai import OpenAIEmbeddings
    api_key = os.environ.get(settings.api_key_env, "").strip()
    if not api_key:
        raise EmbeddingConfigError(f"{settings.api_key_env} is not set; required for --embeddings litellm")
    if not settings.model:
        raise EmbeddingConfigError("Set LITELLM_EMBEDDING_MODEL (or --embedding-model) to an embedding alias "
                                   "available on your gateway, e.g. one listed by 'raglogger models'")
    # check_embedding_ctx_length=False: send plain strings (gateways/non-OpenAI models reject token-id arrays).
    return OpenAIEmbeddings(model=settings.model, openai_api_base=settings.base_url, openai_api_key=api_key,
                            check_embedding_ctx_length=False, request_timeout=settings.timeout_seconds,
                            max_retries=2, chunk_size=_BATCH_SIZE)


def chunk_metadata(chunk: Chunk) -> dict[str, Any]:
    # Chroma metadata values must be scalars; the level list is stored as a delimited string (",ERROR,WARN,").
    return {"chunk_id": chunk.chunk_id, "kind": chunk.kind, "title": chunk.title[:500],
            "start_time": chunk.start_time, "end_time": chunk.end_time, "line_start": chunk.line_start,
            "line_end": chunk.line_end, "occurrences": chunk.occurrences,
            "levels": "," + ",".join(chunk.levels) + ",", "source": chunk.source}


def open_chroma(persist_dir: Path, embeddings: Embeddings):
    import chromadb
    from chromadb.config import Settings as ChromaSettings
    from langchain_chroma import Chroma
    # Chroma ships anonymous PostHog telemetry enabled by default; production log tooling must not phone home.
    client = chromadb.PersistentClient(path=str(persist_dir),
                                       settings=ChromaSettings(anonymized_telemetry=False, allow_reset=False))
    return Chroma(collection_name=COLLECTION_NAME, embedding_function=embeddings, client=client,
                  collection_metadata={"hnsw:space": "cosine"})

class VectorStoreService:
    """Builds and opens the persistent Chroma collection; guards against embedding-model mismatches."""

    def __init__(self, persist_dir: Path, embeddings: Embeddings, settings: EmbeddingSettings) -> None:
        self._persist_dir = persist_dir
        self._embeddings = embeddings
        self._settings = settings

    def build(self, chunks: list[Chunk]) -> int:
        """Full rebuild: drops the collection and embeds ``chunks``."""
        ensure_dir(self._persist_dir)
        store = open_chroma(self._persist_dir, self._embeddings)
        store.reset_collection()
        self._add(store, chunks)
        self._write_manifest(store)
        return len(chunks)

    def upsert_sources(self, chunks: list[Chunk], replace_sources: list[str],
                       remove_sources: Optional[list[str]] = None) -> int:
        """Incremental update: deletes vectors of ``replace_sources`` + ``remove_sources``, then embeds ``chunks``.

        Other sources' vectors are untouched, so adding one log never re-embeds the rest.
        """
        ensure_dir(self._persist_dir)
        if self.read_manifest(self._persist_dir) is not None:
            store = self.open()  # verifies the embedding model matches before mixing vectors
        else:
            store = open_chroma(self._persist_dir, self._embeddings)
        for label in [*replace_sources, *(remove_sources or [])]:
            store.delete(where={"source": {"$eq": label}})
            LOG.info("Removed vectors of source %s", label)
        self._add(store, chunks)
        self._write_manifest(store)
        return len(chunks)

    def _add(self, store, chunks: list[Chunk]) -> None:
        documents = [Document(page_content=f"{chunk.title}\n{chunk.text}", metadata=chunk_metadata(chunk))
                     for chunk in chunks]
        ids = [chunk.chunk_id for chunk in chunks]
        for start in range(0, len(documents), _BATCH_SIZE):
            store.add_documents(documents[start:start + _BATCH_SIZE], ids=ids[start:start + _BATCH_SIZE])
            LOG.info("Embedded %d/%d chunks", min(start + _BATCH_SIZE, len(documents)), len(documents))

    def _write_manifest(self, store) -> None:
        manifest = {"provider": self._settings.provider, "model": self._settings.resolved_model,
                    "chunks": store._collection.count()}
        (self._persist_dir / MANIFEST_NAME).write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    def open(self):
        manifest = self.read_manifest(self._persist_dir)
        if manifest is None:
            raise FileNotFoundError(f"Vector store not found at {self._persist_dir}. "
                                    "Run 'ingest' with --embeddings local|litellm first.")
        expected = (manifest.get("provider"), manifest.get("model"))
        actual = (self._settings.provider, self._settings.resolved_model)
        if expected != actual:
            raise EmbeddingConfigError(f"Vector store was built with embeddings {expected[0]}:{expected[1]} but "
                                       f"{actual[0]}:{actual[1]} is configured. Use the same --embeddings/"
                                       "--embedding-model or re-run 'ingest'.")
        return open_chroma(self._persist_dir, self._embeddings)

    @staticmethod
    def read_manifest(persist_dir: Path) -> Optional[dict]:
        path = persist_dir / MANIFEST_NAME
        if not path.is_file():
            return None
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return None


def chroma_where(filters: SearchFilters) -> Optional[dict]:
    """Pushes the filters Chroma can evaluate natively; the rest (incl. --source substring) are re-checked in Python."""
    clauses = []
    if filters.kind:
        clauses.append({"kind": {"$eq": filters.kind}})
    if not clauses:
        return None
    return clauses[0] if len(clauses) == 1 else {"$and": clauses}


class HybridLogRetriever(BaseRetriever):
    """LangChain retriever fusing Chroma similarity search and BM25 via Reciprocal Rank Fusion."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    vector_store: Any
    bm25: Bm25Index
    k: int = 6
    fetch_k: int = 30
    filters: SearchFilters = SearchFilters()

    def search_hits(self, query: str) -> list[SearchHit]:
        by_id = {chunk.chunk_id: chunk for chunk in self.bm25.chunks}
        fused: dict[str, float] = {}
        sources: dict[str, set] = {}

        for rank, hit in enumerate(self.bm25.search(query, top_k=self.fetch_k, filters=self.filters), start=1):
            fused[hit.chunk.chunk_id] = fused.get(hit.chunk.chunk_id, 0.0) + 1.0 / (RRF_K + rank)
            sources.setdefault(hit.chunk.chunk_id, set()).add("bm25")

        vector_results = self.vector_store.similarity_search_with_score(
            query, k=max(1, min(self.fetch_k, len(by_id))), filter=chroma_where(self.filters))
        rank = 0
        for document, _distance in vector_results:
            chunk = by_id.get(document.metadata.get("chunk_id"))
            if chunk is None or not self.filters.matches(chunk):
                continue  # level/logger/time filters are enforced here with the same semantics as BM25
            rank += 1
            fused[chunk.chunk_id] = fused.get(chunk.chunk_id, 0.0) + 1.0 / (RRF_K + rank)
            sources.setdefault(chunk.chunk_id, set()).add("vector")

        ranked = sorted(fused.items(), key=lambda item: item[1], reverse=True)[:self.k]
        hits = []
        for chunk_id, score in ranked:
            hit = SearchHit(chunk=by_id[chunk_id], score=round(score * 1000, 3),
                            retrieval="+".join(sorted(sources[chunk_id])))
            hits.append(hit)
        LOG.info("Hybrid retrieval: %d fused hits (bm25+vector) for '%s'", len(hits), query[:80])
        return hits

    def _get_relevant_documents(self, query: str, *,
                                run_manager: CallbackManagerForRetrieverRun) -> list[Document]:
        return [hit_to_document(hit) for hit in self.search_hits(query)]


def hit_to_document(hit: SearchHit) -> Document:
    metadata = chunk_metadata(hit.chunk)
    metadata.update({"score": hit.score, "retrieval": hit.retrieval})
    return Document(page_content=hit.chunk.text, metadata=metadata)
