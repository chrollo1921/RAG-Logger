"""Parser for SAP Commerce (Tanuki wrapper + Log4j2) console logs."""
from __future__ import annotations

import gzip
import logging
import re
from pathlib import Path
from typing import Iterable, Iterator, Optional

from raglogger.models import LogEntry

LOG = logging.getLogger(__name__)

# INFO   | jvm 1    | main    | 2026/08/29 00:13:22.638 | <body>
# STATUS | wrapper  | 2026/08/29 00:13:22.638 | <body>
WRAPPER_RE = re.compile(
    r"^(?P<wlevel>[A-Z]+)\s*\|\s*(?P<src>[^|]*?)\s*\|(?:[^|]*\|)?\s*"
    r"(?P<wts>\d{4}/\d{2}/\d{2} \d{2}:\d{2}:\d{2}(?:\.\d+)?)\s*\|\s?(?P<body>.*)$"
)
# [2026-08-29T00:13:22,631] INFO  [task-scheduler-3] <rest>
HEADER_RE = re.compile(
    r"^\[(?P<ts>\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:[,.]\d+)?)\]\s+(?P<level>[A-Z]+)\s+"
    r"\[(?P<thread>[^\]]*)\]\s?(?P<rest>.*)$"
)
# [corId]-SF-1_2[sId]-SF-1_3[127.0.0.6] [LoggerName] {dt.trace_id=...} message
LOGGER_RE = re.compile(
    r"^(?P<ctx>(?:\[[^\]]*\][^\[\s]*\s*)*?)\[(?P<logger>[A-Za-z_$][\w$.-]*)\](?:\s+|$)"
    r"(?:\{(?P<mdc>[^}]*)\}\s*)?(?P<msg>.*)$"
)
CORRELATION_RE = re.compile(r"\[corId\](?P<value>[^\[\s]*)")
# Log4j2 %highlight / %style emit ANSI colour sequences such as ESC[32m ... ESC[m into the console log.
ANSI_RE = re.compile(r"\x1b\[[0-9;?]*[A-Za-z]")
SESSION_RE = re.compile(r"\[sId\](?P<value>[^\[\s]*)")
TRACE_RE = re.compile(r"dt\.trace_id=(?P<value>[0-9a-fA-F]+)")

WRAPPER_LEVEL_MAP = {"STATUS": "INFO", "FATAL": "FATAL", "ERROR": "ERROR", "WARN": "WARN", "DEBUG": "DEBUG"}


class ConsoleLogParser:
    """Streams a console log and groups physical lines into logical :class:`LogEntry` objects."""

    def __init__(self, max_continuation_lines: int = 60) -> None:
        self._max_continuation_lines = max_continuation_lines
        self.lines_read = 0

    def parse_file(self, path: Path) -> Iterator[LogEntry]:
        """Parses a plain or gzip-compressed (``*.gz``, e.g. rotated ``console.log.1.gz``) console log."""
        LOG.info("Parsing console log %s", path)
        if path.suffix.lower() == ".gz":
            with gzip.open(path, "rt", encoding="utf-8", errors="replace") as handle:
                yield from self.parse_lines(handle)
            return
        with open(path, "r", encoding="utf-8", errors="replace") as handle:
            yield from self.parse_lines(handle)

    def parse_lines(self, lines: Iterable[str]) -> Iterator[LogEntry]:
        current: Optional[LogEntry] = None
        self.lines_read = 0
        for line_no, raw in enumerate(lines, start=1):
            self.lines_read = line_no
            line = ANSI_RE.sub("", raw.rstrip("\r\n"))
            wrapper = WRAPPER_RE.match(line)
            body = wrapper.group("body") if wrapper else line

            header = HEADER_RE.match(body)
            if header:
                if current is not None:
                    yield current
                current = self._new_entry(line_no, header)
                continue

            text = body.rstrip()
            if not text.strip():
                continue
            if wrapper and wrapper.group("src").lower().startswith("wrapper"):
                # Messages emitted by the Tanuki wrapper itself (JVM restarts, pings, ...) are standalone events.
                if current is not None:
                    yield current
                current = self._wrapper_entry(line_no, wrapper, text)
                continue
            if current is None:
                current = LogEntry(line_no=line_no, timestamp=self._wrapper_time(wrapper), level="INFO",
                                   thread="", logger="console", message=text.strip(), last_line_no=line_no)
                continue
            current.last_line_no = line_no
            if len(current.continuation) < self._max_continuation_lines:
                current.continuation.append(text)
            else:
                current.truncated_lines += 1

        if current is not None:
            yield current
        LOG.info("Finished parsing %d physical lines", self.lines_read)

    @staticmethod
    def _new_entry(line_no: int, header: re.Match) -> LogEntry:
        rest = header.group("rest")
        logger, message, mdc, ctx = "unknown", rest, "", ""
        parsed = LOGGER_RE.match(rest)
        if parsed:
            logger = parsed.group("logger")
            message = parsed.group("msg")
            mdc = parsed.group("mdc") or ""
            ctx = parsed.group("ctx") or ""
        return LogEntry(
            line_no=line_no,
            timestamp=header.group("ts").replace(".", ","),
            level=header.group("level"),
            thread=header.group("thread"),
            logger=logger,
            message=message.strip(),
            correlation_id=_group(CORRELATION_RE, ctx),
            session_id=_group(SESSION_RE, ctx),
            trace_id=_group(TRACE_RE, mdc),
            last_line_no=line_no,
        )

    def _wrapper_entry(self, line_no: int, wrapper: re.Match, text: str) -> LogEntry:
        level = WRAPPER_LEVEL_MAP.get(wrapper.group("wlevel"), "INFO")
        return LogEntry(line_no=line_no, timestamp=self._wrapper_time(wrapper), level=level, thread="wrapper",
                        logger="TanukiWrapper", message=text.strip(), last_line_no=line_no)

    @staticmethod
    def _wrapper_time(wrapper: Optional[re.Match]) -> str:
        if not wrapper:
            return ""
        return wrapper.group("wts").replace("/", "-").replace(" ", "T").replace(".", ",")


def _group(pattern: re.Pattern, text: str) -> str:
    match = pattern.search(text) if text else None
    return match.group("value") if match else ""
