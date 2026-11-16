"""Data transfer objects shared across the parser, chunker, index, service and CLI layers."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Optional

EXCEPTION_HINTS = ("Exception", "Error", "Throwable", "Caused by")
UNTIL_INCLUSIVE_SUFFIX = "~"


@dataclass
class LogEntry:
    """A single logical log event: the header line plus continuation lines such as stack frames."""

    line_no: int
    timestamp: str
    level: str
    thread: str
    logger: str
    message: str
    correlation_id: str = ""
    session_id: str = ""
    trace_id: str = ""
    continuation: list[str] = field(default_factory=list)
    truncated_lines: int = 0
    last_line_no: int = 0

    @property
    def exception_line(self) -> str:
        """Returns the first continuation line that looks like an exception declaration."""
        for line in self.continuation:
            stripped = line.strip()
            if stripped and not stripped.startswith("at ") and any(hint in stripped for hint in EXCEPTION_HINTS):
                return stripped
        return ""

    def stack_preview(self, max_lines: int) -> list[str]:
        """Returns up to ``max_lines`` non-blank continuation lines."""
        preview = [line.strip() for line in self.continuation if line.strip()]
        return preview[:max_lines]


@dataclass
class Chunk:
    """A retrievable document stored in the index."""

    chunk_id: str
    kind: str
    title: str
    text: str
    start_time: str
    end_time: str
    line_start: int
    line_end: int
    levels: list[str] = field(default_factory=list)
    loggers: list[str] = field(default_factory=list)
    occurrences: int = 1
    source: str = ""  # display label of the console log file this chunk came from ("" for legacy indexes)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Chunk":
        known = {name for name in cls.__dataclass_fields__}
        return cls(**{key: value for key, value in data.items() if key in known})


@dataclass
class SearchFilters:
    """Optional metadata filters applied on top of lexical relevance."""

    level: Optional[str] = None
    logger: Optional[str] = None
    kind: Optional[str] = None
    since: Optional[str] = None
    until: Optional[str] = None
    source: Optional[str] = None  # case-insensitive substring of the source label, e.g. "node1" or "20260829"
    source_labels: Optional[frozenset] = None  # exact labels (resolved from label *or* path by the CLI)

    def __post_init__(self) -> None:
        self.level = self.level.upper() if self.level else None
        self.since = _normalize_time(self.since)
        self.until = _normalize_time(self.until)
        self.source = self.source.strip().lower() if self.source and self.source.strip() else None

    def matches_source(self, label: str) -> bool:
        if self.source_labels is not None:
            return label in self.source_labels
        return not self.source or self.source in label.lower()

    def matches(self, chunk: Chunk) -> bool:
        if not self.matches_source(chunk.source):
            return False
        if self.level and self.level not in chunk.levels:
            return False
        if self.logger and not any(self.logger.lower() in name.lower() for name in chunk.loggers):
            return False
        if self.kind and chunk.kind != self.kind:
            return False
        if self.since and chunk.end_time and chunk.end_time < self.since:
            return False
        if self.until and chunk.start_time and chunk.start_time > self.until + UNTIL_INCLUSIVE_SUFFIX:
            return False
        return True


@dataclass
class SearchHit:
    chunk: Chunk
    score: float
    retrieval: str = "bm25"


@dataclass
class Answer:
    question: str
    text: str
    hits: list[SearchHit]
    backend: str


@dataclass
class IngestStats:
    """Statistics for one ingested file, or (``files`` populated) the aggregate of a whole ingest run."""

    source: str
    lines: int
    entries: int
    templates: int
    chunks: int
    duration_seconds: float
    level_counts: dict[str, int] = field(default_factory=dict)
    vectors: int = 0
    label: str = ""
    files: list["IngestStats"] = field(default_factory=list)
    removed_sources: list[str] = field(default_factory=list)
    total_sources: int = 0
    total_chunks: int = 0
    skipped: bool = False  # unchanged since last ingest (append mode)


def _normalize_time(value: Optional[str]) -> Optional[str]:
    """Accepts ``2026-08-29 00:13`` or ``2026-08-29T00:13:22,631`` and returns the log's ISO-like form."""
    if not value:
        return None
    return value.strip().replace(" ", "T").replace(".", ",")
