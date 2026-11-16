"""Application service layer: orchestrates parse -> redact -> chunk -> index, and retrieve -> generate."""
from __future__ import annotations

import logging
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator, Optional, Protocol

from raglogger.chunker import LogChunker
from raglogger.generators import AnswerGenerator, ExtractiveGenerator, GenerationError
from raglogger.index import Bm25Index
from raglogger.models import Answer, Chunk, IngestStats, LogEntry, SearchFilters, SearchHit
from raglogger.parser import ConsoleLogParser
from raglogger.redaction import Redactor
from raglogger.sources import DEFAULT_PATTERN, SourceFile, assign_labels, discover

LOG = logging.getLogger(__name__)


class Retriever(Protocol):
    """Anything that can return ranked hits: the BM25 index or the LangChain hybrid retriever adapter."""

    def search(self, query: str, top_k: int = ..., filters: Optional[SearchFilters] = ...) -> list[SearchHit]:
        ...


class VectorStoreBuilder(Protocol):
    def build(self, chunks: list[Chunk]) -> int:
        ...

    def upsert_sources(self, chunks: list[Chunk], replace_sources: list[str],
                       remove_sources: Optional[list[str]] = None) -> int:
        ...


SOURCES_METADATA_KEY = "sources"


class IngestService:
    """Ingests one or more console logs into a single BM25 index (+ optional vector store).

    * ``append=False`` (default): the index is rebuilt from exactly the given files.
    * ``append=True``: given files are added; a file already in the index (same path) is replaced; files unchanged
      since the last ingest (same size + mtime) are skipped unless ``force=True``.
    * ``remove``: labels of sources to drop (implies an incremental update).

    Each file is parsed and chunked independently (memory stays bounded per file) and every chunk carries its
    source label, so results and filters can be scoped per file / node.
    """

    def __init__(self, parser: ConsoleLogParser, redactor: Redactor, chunker: LogChunker,
                 vector_store: Optional["VectorStoreBuilder"] = None) -> None:
        self._parser = parser
        self._redactor = redactor
        self._chunker = chunker
        self._vector_store = vector_store

    def ingest(self, log_path: Path, index_path: Path) -> IngestStats:
        """Single-file convenience wrapper (full rebuild), kept for backwards compatibility."""
        return self.ingest_many([log_path], index_path)

    def ingest_many(self, paths: list[Path], index_path: Path, append: bool = False, force: bool = False,
                    remove: Optional[list[str]] = None, label: Optional[str] = None,
                    pattern: str = DEFAULT_PATTERN) -> IngestStats:
        started = time.monotonic()
        remove = list(remove or [])
        incremental = append or bool(remove)
        existing_chunks, existing_sources = self._load_existing(index_path) if incremental else ([], {})
        unknown = [name for name in remove if name not in existing_sources]
        if unknown:
            raise ValueError(f"Cannot remove unknown source(s): {', '.join(unknown)}. "
                             f"Indexed sources: {', '.join(existing_sources) or 'none'}")
        files = assign_labels(discover(paths, pattern) if paths else [],
                              {name: meta["path"] for name, meta in existing_sources.items()}, label)
        if not files and not remove:
            raise ValueError("Nothing to ingest: pass at least one console log file or directory")

        result = IngestStats(source=", ".join(str(item.path) for item in files), lines=0, entries=0, templates=0,
                             chunks=0, duration_seconds=0.0, removed_sources=remove)
        sources = {name: meta for name, meta in existing_sources.items() if name not in remove}
        new_chunks, replaced = self._process_files(files, sources, result, skip_unchanged=incremental and not force)
        self._aggregate(result)

        dropped = set(replaced) | set(remove)
        all_chunks = [chunk for chunk in existing_chunks if chunk.source not in dropped] + new_chunks
        index = Bm25Index()
        index.build(all_chunks, metadata=self._index_metadata(sources))
        index.save(index_path)
        result.total_sources, result.total_chunks = len(sources), len(all_chunks)

        if self._vector_store is not None:
            # Chunks are already redacted, so no secrets are embedded or sent to a remote embedding model.
            if incremental:
                result.vectors = self._vector_store.upsert_sources(new_chunks, replaced, remove)
            else:
                result.vectors = self._vector_store.build(all_chunks)
        result.duration_seconds = round(time.monotonic() - started, 2)
        LOG.info("Ingest finished in %.2fs: %d source(s), %d chunks in index", result.duration_seconds,
                 result.total_sources, result.total_chunks)
        return result

    def _process_files(self, files: list[SourceFile], sources: dict[str, dict], result: IngestStats,
                       skip_unchanged: bool) -> tuple[list[Chunk], list[str]]:
        """Parses/chunks each file; updates ``sources`` in place. Returns ``(new_chunks, replaced_labels)``."""
        new_chunks: list[Chunk] = []
        replaced: list[str] = []
        for source in files:
            fingerprint = source.fingerprint()
            previous = sources.get(source.label)
            if skip_unchanged and previous and previous.get("fingerprint") == fingerprint:
                LOG.info("Skipping unchanged source %s", source.label)
                result.files.append(IngestStats(source=str(source.path), label=source.label, lines=0, entries=0,
                                                templates=0, chunks=0, duration_seconds=0.0, skipped=True))
                continue
            file_stats, chunks = self._ingest_file(source)
            new_chunks.extend(chunks)
            result.files.append(file_stats)
            if previous:
                replaced.append(source.label)
            sources[source.label] = {"path": str(source.path), "fingerprint": fingerprint,
                                     "lines": file_stats.lines, "entries": file_stats.entries,
                                     "chunks": file_stats.chunks, "level_counts": file_stats.level_counts,
                                     "ingested_at": datetime.now(timezone.utc).isoformat(timespec="seconds")}
        return new_chunks, replaced

    def _ingest_file(self, source: SourceFile) -> tuple[IngestStats, list[Chunk]]:
        started = time.monotonic()
        chunks = self._chunker.build_chunks(self._redacted_entries(source.path), source=source.label,
                                            source_key=source.key)
        stats = IngestStats(source=str(source.path), label=source.label, lines=self._parser.lines_read,
                            entries=self._chunker.entry_count, templates=self._chunker.template_count,
                            chunks=len(chunks), duration_seconds=round(time.monotonic() - started, 2),
                            level_counts=dict(self._chunker.level_counts.most_common()))
        LOG.info("Ingested %s: %d lines -> %d chunks", source.label, stats.lines, stats.chunks)
        return stats, chunks

    @staticmethod
    def _aggregate(result: IngestStats) -> None:
        levels: Counter = Counter()
        for item in result.files:
            result.lines += item.lines
            result.entries += item.entries
            result.templates += item.templates
            result.chunks += item.chunks
            levels.update(item.level_counts)
        result.level_counts = dict(levels.most_common())

    @staticmethod
    def _index_metadata(sources: dict[str, dict]) -> dict:
        levels: Counter = Counter()
        for meta in sources.values():
            levels.update(meta.get("level_counts", {}))
        return {"source": ", ".join(meta["path"] for meta in sources.values()),
                "lines": sum(meta.get("lines", 0) for meta in sources.values()),
                "entries": sum(meta.get("entries", 0) for meta in sources.values()),
                "level_counts": dict(levels.most_common()), SOURCES_METADATA_KEY: sources}

    @staticmethod
    def _load_existing(index_path: Path) -> tuple[list[Chunk], dict[str, dict]]:
        if not index_path.exists():
            return [], {}
        index = Bm25Index.load(index_path)
        sources = index.metadata.get(SOURCES_METADATA_KEY)
        if sources is None:
            raise ValueError(f"{index_path} was built by an older single-file version and cannot be appended to. "
                             "Re-run 'ingest' without --append once to rebuild it.")
        return index.chunks, sources


    def _redacted_entries(self, log_path: Path) -> Iterator[LogEntry]:
        for entry in self._parser.parse_file(log_path):
            yield self._redactor.redact_entry(entry)


class QueryService:
    def __init__(self, index: "Retriever", generator: AnswerGenerator,
                 fallback: Optional[AnswerGenerator] = None) -> None:
        self._index = index
        self._generator = generator
        self._fallback = fallback or ExtractiveGenerator()

    def search(self, question: str, top_k: int, filters: Optional[SearchFilters] = None) -> list[SearchHit]:
        if not question or not question.strip():
            raise ValueError("Question must not be empty")
        hits = self._index.search(question, top_k=top_k, filters=filters)
        LOG.info("Retrieved %d hits for question '%s'", len(hits), question[:80])
        return hits

    def ask(self, question: str, top_k: int = 6, filters: Optional[SearchFilters] = None) -> Answer:
        hits = self.search(question, top_k, filters)
        if not hits:
            return Answer(question, self._fallback.generate(question, hits), hits, self._fallback.name)
        try:
            text, backend = self._generator.answer(question, hits)
            return Answer(question, text, hits, backend)
        except GenerationError as error:
            LOG.warning("LLM backend '%s' failed (%s); falling back to extractive answer",
                        self._generator.name, error)
            text = f"[LLM backend unavailable: {error}]\n\n" + self._fallback.generate(question, hits)
            return Answer(question, text, hits, self._fallback.name)
