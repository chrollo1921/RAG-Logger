"""Masks secrets and personal data before anything is indexed, logged or sent to an LLM."""
from __future__ import annotations

import re
from typing import Any

from raglogger.models import LogEntry

MASK = "***REDACTED***"

_PATTERNS: list[tuple[re.Pattern, str]] = [
    (re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----.*?-----END [A-Z ]*PRIVATE KEY-----", re.S), MASK),
    (re.compile(r"(?i)\bbearer\s+[A-Za-z0-9._~+/=-]+"), "Bearer " + MASK),
    (re.compile(r"(?i)\bbasic\s+[A-Za-z0-9+/=]{8,}"), "Basic " + MASK),
    (re.compile(r"(?i)\b(password|passwd|pwd|secret|client_secret|access_token|refresh_token|token|api[_-]?key"
                r"|authorization|cookie|jsessionid)(\"?\s*[=:]\s*\"?)(?!(?:bearer|basic)\s)([^\s,;&\"']+)"), r"\1\2" + MASK),
    (re.compile(r"\b(?:sk|ghp|gho|ghs|glpat|xox[baprs])-[A-Za-z0-9_-]{16,}\b"), MASK),
    (re.compile(r"\bAKIA[0-9A-Z]{16}\b"), MASK),
    (re.compile(r"\beyJ[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}\b"), MASK),
    (re.compile(r"\b[\w.+-]+@[\w-]+(?:\.[\w-]+)+\b"), "***REDACTED_EMAIL***"),
]


class Redactor:
    """Applies a fixed set of masking rules; stateless and thread-safe."""

    def redact(self, text: str) -> str:
        if not text:
            return text
        for pattern, replacement in _PATTERNS:
            text = pattern.sub(replacement, text)
        return text

    def redact_entry(self, entry: LogEntry) -> LogEntry:
        entry.message = self.redact(entry.message)
        entry.continuation = [self.redact(line) for line in entry.continuation]
        return entry

    def redact_value(self, value: Any) -> Any:
        """Recursively redacts strings within JSON-like structures."""
        if isinstance(value, str):
            return self.redact(value)
        if isinstance(value, dict):
            return {key: self.redact_value(item) for key, item in value.items()}
        if isinstance(value, list):
            return [self.redact_value(item) for item in value]
        return value
