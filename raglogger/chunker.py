"""Turns parsed log entries into retrievable chunks.

Console logs are dominated by repetitive lines, so instead of indexing raw lines the chunker builds:

* ``template``  - one chunk per (level, logger, normalised message) with occurrence counts and samples;
* ``incident``  - one chunk per distinct WARN/ERROR/FATAL signature including the exception and stack preview;
* ``window``    - one digest chunk per time window (level counts, busiest templates, problems);
* ``overview``  - a single chunk describing the whole file.
"""
from __future__ import annotations

import hashlib
import logging
import re
from collections import Counter
from dataclasses import dataclass, field
from typing import Iterable

from raglogger.models import Chunk, LogEntry

LOG = logging.getLogger(__name__)

PROBLEM_LEVELS = ("WARN", "ERROR", "FATAL")
_NORMALIZERS: list[tuple[re.Pattern, str]] = [
    (re.compile(r"\b[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}\b"), "<UUID>"),
    (re.compile(r"\b\d{1,3}(?:\.\d{1,3}){3}(?::\d+)?\b"), "<IP>"),
    (re.compile(r"\b(?=[0-9a-fA-F]*\d)(?=[0-9a-fA-F]*[a-fA-F])[0-9a-fA-F]{12,}\b"), "<HEX>"),
    (re.compile(r"(?<![A-Za-z])\d+(?:[.,]\d+)*"), "<NUM>"),
    (re.compile(r"'[^']{0,200}'"), "'<STR>'"),
]


def normalize_message(message: str) -> str:
    """Replaces variable tokens (ids, numbers, quoted values) so that similar messages collapse together."""
    text = message
    for pattern, replacement in _NORMALIZERS:
        text = pattern.sub(replacement, text)
    return re.sub(r"\s+", " ", text).strip()[:400]


@dataclass
class _TemplateStats:
    level: str
    logger: str
    template: str
    count: int = 0
    first_time: str = ""
    last_time: str = ""
    first_line: int = 0
    last_line: int = 0
    samples: list[str] = field(default_factory=list)
    threads: Counter = field(default_factory=Counter)
    exception: str = ""
    stack: list[str] = field(default_factory=list)
    correlation_ids: list[str] = field(default_factory=list)


@dataclass
class _WindowStats:
    key: str
    first_time: str = ""
    last_time: str = ""
    first_line: int = 0
    last_line: int = 0
    levels: Counter = field(default_factory=Counter)
    templates: Counter = field(default_factory=Counter)
    problems: Counter = field(default_factory=Counter)
    loggers: set = field(default_factory=set)


def _digest(*parts: str) -> str:
    return hashlib.sha1("|".join(parts).encode("utf-8")).hexdigest()[:16]

class LogChunker:
    """Aggregates entries in a single streaming pass; memory is bounded by the number of distinct templates."""

    def __init__(self, window_minutes: int = 5, max_samples: int = 3, stack_lines: int = 12,
                 max_window_items: int = 8) -> None:
        if window_minutes < 1 or window_minutes > 60:
            raise ValueError("window_minutes must be between 1 and 60")
        self._window_minutes = window_minutes
        self._max_samples = max_samples
        self._stack_lines = stack_lines
        self._max_window_items = max_window_items
        self.entry_count = 0
        self.template_count = 0
        self.level_counts: Counter = Counter()
        self._source = ""
        self._id_prefix = ""

    def build_chunks(self, entries: Iterable[LogEntry], source: str, source_key: str = "") -> list[Chunk]:
        """Builds chunks for one console log.

        ``source`` is the display label (stored on every chunk and mentioned in its text so both BM25 and vector
        search can match file/node names); ``source_key`` namespaces chunk ids so that identical templates from
        different files never overwrite each other. Empty key keeps the legacy single-file ids.
        """
        self._source = source
        self._id_prefix = f"{source_key}:" if source_key else ""
        templates: dict[tuple[str, str, str], _TemplateStats] = {}
        incidents: dict[str, _TemplateStats] = {}
        windows: dict[str, _WindowStats] = {}
        loggers: Counter = Counter()
        self.entry_count = 0
        self.level_counts = Counter()
        first_time, last_time = "", ""

        for entry in entries:
            self.entry_count += 1
            self.level_counts[entry.level] += 1
            loggers[entry.logger] += 1
            first_time = first_time or entry.timestamp
            last_time = entry.timestamp or last_time
            template = normalize_message(entry.message)
            label = f"{entry.level} [{entry.logger}] {template}"

            stats = templates.setdefault((entry.level, entry.logger, template),
                                         _TemplateStats(entry.level, entry.logger, template))
            self._accumulate(stats, entry)

            if entry.level in PROBLEM_LEVELS or entry.exception_line:
                signature = self._signature(entry, template)
                incident = incidents.setdefault(signature, _TemplateStats(entry.level, entry.logger, template))
                self._accumulate(incident, entry)

            key = self._window_key(entry.timestamp)
            self._accumulate_window(windows.setdefault(key, _WindowStats(key)), entry, label)

        self.template_count = len(templates)
        chunks = [self._template_chunk(stats) for stats in templates.values()]
        chunks.extend(self._incident_chunk(signature, stats) for signature, stats in incidents.items())
        chunks.extend(self._window_chunk(window) for window in windows.values())
        chunks.append(self._overview_chunk(source, first_time, last_time, templates, incidents, loggers))
        for chunk in chunks:
            chunk.chunk_id = self._id_prefix + chunk.chunk_id
            chunk.source = source
            if chunk.kind != "overview":
                chunk.text += f"\nSource log: {source}"
        LOG.info("Built %d chunks (%d templates, %d incidents, %d windows) from %d entries of %s",
                 len(chunks), len(templates), len(incidents), len(windows), self.entry_count, source)
        return chunks

    # ------------------------------------------------------------------ accumulation helpers
    def _accumulate(self, stats: _TemplateStats, entry: LogEntry) -> None:
        stats.count += 1
        if stats.count == 1:
            stats.first_time, stats.first_line = entry.timestamp, entry.line_no
            stats.exception = entry.exception_line
            stats.stack = entry.stack_preview(self._stack_lines)
        stats.last_time, stats.last_line = entry.timestamp, entry.last_line_no or entry.line_no
        if len(stats.samples) < self._max_samples and entry.message[:500] not in stats.samples:
            stats.samples.append(entry.message[:500])
        if entry.thread:
            stats.threads[entry.thread] += 1
        if entry.correlation_id and len(stats.correlation_ids) < self._max_samples \
                and entry.correlation_id not in stats.correlation_ids:
            stats.correlation_ids.append(entry.correlation_id)

    @staticmethod
    def _accumulate_window(window: _WindowStats, entry: LogEntry, label: str) -> None:
        window.first_time = window.first_time or entry.timestamp
        window.first_line = window.first_line or entry.line_no
        window.last_time = entry.timestamp or window.last_time
        window.last_line = entry.last_line_no or entry.line_no
        window.levels[entry.level] += 1
        window.templates[label] += 1
        window.loggers.add(entry.logger)
        if entry.level in PROBLEM_LEVELS or entry.exception_line:
            problem = label + (f" -> {entry.exception_line}" if entry.exception_line else "")
            window.problems[problem[:600]] += 1

    @staticmethod
    def _signature(entry: LogEntry, template: str) -> str:
        frames = [line.strip() for line in entry.continuation if line.strip().startswith("at ")][:3]
        return _digest(entry.level, entry.logger, template, normalize_message(entry.exception_line), *frames)

    def _window_key(self, timestamp: str) -> str:
        if len(timestamp) < 16:
            return "unknown"
        minute = int(timestamp[14:16]) // self._window_minutes * self._window_minutes
        return f"{timestamp[:14]}{minute:02d}"

    # ------------------------------------------------------------------ chunk builders
    def _template_chunk(self, stats: _TemplateStats) -> Chunk:
        lines = [
            f"Log message template from logger {stats.logger} at level {stats.level}.",
            f"Template: {stats.template}",
            f"Occurrences: {stats.count} between {stats.first_time} and {stats.last_time}.",
            "Threads: " + ", ".join(f"{name} ({count})" for name, count in stats.threads.most_common(3)),
            "Sample messages:",
            *[f"- {sample}" for sample in stats.samples],
        ]
        if stats.exception:
            lines.append(f"Exception: {stats.exception}")
        return Chunk(
            chunk_id="tpl-" + _digest(stats.level, stats.logger, stats.template), kind="template",
            title=f"{stats.level} {stats.logger}: {stats.template[:120]}", text="\n".join(lines),
            start_time=stats.first_time, end_time=stats.last_time, line_start=stats.first_line,
            line_end=stats.last_line, levels=[stats.level], loggers=[stats.logger], occurrences=stats.count,
        )

    def _incident_chunk(self, signature: str, stats: _TemplateStats) -> Chunk:
        lines = [
            f"{stats.level} incident reported by {stats.logger}.",
            f"Message: {stats.samples[0] if stats.samples else stats.template}",
            f"Template: {stats.template}",
            f"Occurrences: {stats.count}; first at {stats.first_time} (line {stats.first_line}), "
            f"last at {stats.last_time}.",
        ]
        if stats.exception:
            lines.append(f"Exception: {stats.exception}")
        if stats.stack:
            lines.append("Stack trace preview:")
            lines.extend(f"  {frame}" for frame in stats.stack)
        if stats.correlation_ids:
            lines.append("Sample correlation ids: " + ", ".join(stats.correlation_ids))
        if stats.threads:
            lines.append("Threads: " + ", ".join(name for name, _ in stats.threads.most_common(3)))
        title_suffix = f" ({stats.exception[:80]})" if stats.exception else ""
        return Chunk(
            chunk_id=f"inc-{signature}", kind="incident",
            title=f"{stats.level} {stats.logger}: {stats.template[:100]}{title_suffix}", text="\n".join(lines),
            start_time=stats.first_time, end_time=stats.last_time, line_start=stats.first_line,
            line_end=stats.last_line, levels=[stats.level], loggers=[stats.logger], occurrences=stats.count,
        )

    def _window_chunk(self, window: _WindowStats) -> Chunk:
        total = sum(window.levels.values())
        lines = [
            f"Activity digest for the {self._window_minutes}-minute window starting {window.key}.",
            f"Time range: {window.first_time} to {window.last_time}; {total} log events.",
            "Events per level: " + ", ".join(f"{level}={count}" for level, count in window.levels.most_common()),
            "Most frequent messages:",
            *[f"- {count}x {label[:300]}" for label, count in window.templates.most_common(self._max_window_items)],
        ]
        if window.problems:
            lines.append("Warnings and errors in this window:")
            lines.extend(f"- {count}x {label}" for label, count in window.problems.most_common(self._max_window_items))
        else:
            lines.append("No warnings or errors in this window.")
        return Chunk(
            chunk_id=f"win-{window.key}", kind="window", title=f"Window {window.key} ({total} events)",
            text="\n".join(lines), start_time=window.first_time, end_time=window.last_time,
            line_start=window.first_line, line_end=window.last_line, levels=sorted(window.levels),
            loggers=sorted(window.loggers), occurrences=total,
        )

    def _overview_chunk(self, source: str, first_time: str, last_time: str,
                        templates: dict[tuple[str, str, str], _TemplateStats],
                        incidents: dict[str, _TemplateStats], loggers: Counter) -> Chunk:
        top_templates = sorted(templates.values(), key=lambda item: item.count, reverse=True)[:10]
        top_incidents = sorted(incidents.values(), key=lambda item: item.count, reverse=True)[:10]
        lines = [
            f"Overview summary of console log {source}.",
            f"Covers {first_time} to {last_time}; {self.entry_count} log events, {len(templates)} distinct "
            f"message templates and {len(incidents)} distinct warning/error incidents.",
            "Events per level: " + ", ".join(f"{level}={count}" for level, count in self.level_counts.most_common()),
            "Most active loggers: " + ", ".join(f"{name} ({count})" for name, count in loggers.most_common(10)),
            "Most frequent messages:",
            *[f"- {item.count}x {item.level} [{item.logger}] {item.template[:200]}" for item in top_templates],
            "Most frequent warnings/errors:",
            *[f"- {item.count}x {item.level} [{item.logger}] {item.template[:200]}"
              + (f" -> {item.exception[:150]}" if item.exception else "") for item in top_incidents],
        ]
        return Chunk(
            chunk_id="overview", kind="overview", title=f"Overview of {source}", text="\n".join(lines),
            start_time=first_time, end_time=last_time, line_start=1, line_end=0,
            levels=sorted(self.level_counts), loggers=[name for name, _ in loggers.most_common(50)],
            occurrences=self.entry_count,
        )
