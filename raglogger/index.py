"""Okapi BM25 lexical index persisted as JSON (no third-party dependencies)."""
from __future__ import annotations

import gzip
import json
import logging
import math
import re
from collections import Counter
from pathlib import Path
from typing import Optional

from raglogger.models import Chunk, SearchFilters, SearchHit
from raglogger.paths import ensure_parent

LOG = logging.getLogger(__name__)

INDEX_FORMAT_VERSION = 1
_TOKEN_RE = re.compile(r"[A-Za-z][A-Za-z0-9_$]*|\d+")
_CAMEL_RE = re.compile(r"[A-Z]+(?=[A-Z][a-z])|[A-Z]?[a-z]+|[A-Z]+|\d+")
STOPWORDS = frozenset(
    "a an and are as at be by did do does for from has have how i in is it its of on or show that the this to "
    "was were what when where which who why with me tell any there list give".split()
)
KIND_BOOST = {"incident": 1.15, "template": 1.0, "window": 0.9, "overview": 0.8}
_OVERVIEW_HINTS = frozenset({"overview", "summary", "summarize", "summarise", "overall", "most", "top", "frequent"})
# Maps natural-language variants onto the vocabulary used in log lines (levels, exception words, ...).
_SYNONYMS = {
    "errors": "error", "warning": "warn", "warnings": "warn", "warns": "warn", "exceptions": "exception",
    "failed": "fail", "failure": "fail", "failures": "fail", "failing": "fail", "fails": "fail",
    "timeouts": "timeout", "timed": "timeout", "fatals": "fatal", "issues": "issue", "problems": "problem",
}


def _normalize_token(token: str) -> str:
    return _SYNONYMS.get(token, token)


def tokenize(text: str) -> list[str]:
    """Lower-cased word tokens, with CamelCase/dotted identifiers additionally split into their parts."""
    tokens: list[str] = []
    for raw in _TOKEN_RE.findall(text):
        lowered = raw.lower()
        if lowered not in STOPWORDS:
            tokens.append(_normalize_token(lowered))
        parts = [part.lower() for part in _CAMEL_RE.findall(raw)]
        if len(parts) > 1:
            tokens.extend(_normalize_token(part) for part in parts if len(part) > 1 and part not in STOPWORDS)
    return tokens


class Bm25Index:
    def __init__(self, k1: float = 1.4, b: float = 0.6) -> None:
        self._k1 = k1
        self._b = b
        self.chunks: list[Chunk] = []
        self._term_freqs: list[dict[str, int]] = []
        self._doc_lengths: list[int] = []
        self._doc_freq: Counter = Counter()
        self._avg_length = 0.0
        self.metadata: dict = {}

    def build(self, chunks: list[Chunk], metadata: Optional[dict] = None) -> None:
        self.chunks = chunks
        self._term_freqs = []
        self._doc_lengths = []
        self._doc_freq = Counter()
        for chunk in chunks:
            counts = Counter(tokenize(chunk.title + "\n" + chunk.text))
            self._term_freqs.append(dict(counts))
            self._doc_lengths.append(sum(counts.values()))
            self._doc_freq.update(counts.keys())
        self._avg_length = (sum(self._doc_lengths) / len(self._doc_lengths)) if self._doc_lengths else 0.0
        self.metadata = metadata or {}
        LOG.info("Indexed %d chunks, vocabulary size %d", len(chunks), len(self._doc_freq))

    def search(self, query: str, top_k: int = 5, filters: Optional[SearchFilters] = None) -> list[SearchHit]:
        terms = list(dict.fromkeys(tokenize(query)))
        if not self.chunks:
            return []
        filters = filters or SearchFilters()
        total = len(self.chunks)
        wants_overview = bool(_OVERVIEW_HINTS.intersection(terms))
        hits: list[SearchHit] = []
        for position, chunk in enumerate(self.chunks):
            if not filters.matches(chunk):
                continue
            score = self._score(position, terms, total)
            if chunk.kind == "overview" and wants_overview:
                score += 5.0
            if score <= 0:
                continue
            # Mild boost for frequently recurring problems: log1p keeps it from dominating relevance.
            score *= KIND_BOOST.get(chunk.kind, 1.0) * (1.0 + 0.05 * math.log1p(chunk.occurrences))
            hits.append(SearchHit(chunk=chunk, score=round(score, 4)))
        hits.sort(key=lambda hit: hit.score, reverse=True)
        return hits[:top_k]

    def _score(self, position: int, terms: list[str], total: int) -> float:
        freqs = self._term_freqs[position]
        length_norm = self._k1 * (1 - self._b + self._b * self._doc_lengths[position] / (self._avg_length or 1.0))
        score = 0.0
        for term in terms:
            frequency = freqs.get(term)
            if not frequency:
                continue
            doc_freq = self._doc_freq[term]
            idf = math.log(1 + (total - doc_freq + 0.5) / (doc_freq + 0.5))
            score += idf * frequency * (self._k1 + 1) / (frequency + length_norm)
        return score

    # ------------------------------------------------------------------ persistence
    def save(self, path: Path) -> None:
        ensure_parent(path)
        payload = {
            "version": INDEX_FORMAT_VERSION,
            "metadata": self.metadata,
            "params": {"k1": self._k1, "b": self._b},
            "chunks": [chunk.to_dict() for chunk in self.chunks],
        }
        with gzip.open(path, "wt", encoding="utf-8") as handle:
            json.dump(payload, handle)
        LOG.info("Saved index with %d chunks to %s", len(self.chunks), path)

    @classmethod
    def load(cls, path: Path) -> "Bm25Index":
        if not path.exists():
            raise FileNotFoundError(f"Index not found at {path}. Run the 'ingest' command first.")
        try:
            with gzip.open(path, "rt", encoding="utf-8") as handle:
                payload = json.load(handle)
        except (OSError, json.JSONDecodeError) as error:
            raise ValueError(f"Index file {path} is unreadable or corrupt: {error}") from error
        if payload.get("version") != INDEX_FORMAT_VERSION:
            raise ValueError(f"Unsupported index version {payload.get('version')}; re-run 'ingest'.")
        index = cls(**payload.get("params", {}))
        # Term statistics are rebuilt on load; this keeps the file small and is fast for tens of thousands of chunks.
        index.build([Chunk.from_dict(item) for item in payload["chunks"]], payload.get("metadata"))
        return index
