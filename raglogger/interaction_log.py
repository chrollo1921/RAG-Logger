"""Append-only JSONL audit log of every LLM request/response made by the application.

Entries are redacted before being written and the file handle is flushed and fsync'ed after each write.
Logging failures never interrupt the user workflow; they are reported on stderr instead.
"""
from __future__ import annotations

import json
import logging
import os
import sys
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from raglogger.paths import default_interaction_log_path, ensure_parent
from raglogger.redaction import Redactor

DEFAULT_LOG_PATH = default_interaction_log_path()


class InteractionLogger:
    def __init__(self, path: Optional[Path], redactor: Redactor,
                 transcript: Optional[logging.Logger] = None) -> None:
        self._path = path
        self._redactor = redactor
        self._transcript = transcript
        self._lock = threading.Lock()
        self.session_id = uuid.uuid4().hex[:12]

    def log_request(self, model: str, system_prompt: str, user_prompt: str, assembled_prompt: str,
                    parameters: dict, context: list[str]) -> str:
        request_id = uuid.uuid4().hex[:12]
        self._write({
            "type": "request", "requestId": request_id, "model": model, "parameters": parameters,
            "systemPrompt": system_prompt, "userPrompt": user_prompt, "assembledPrompt": assembled_prompt,
            "contextChunkIds": context,
        })
        self._log_transcript(f"===== LLM REQUEST id={request_id} session={self.session_id} model={model} "
                             f"params={parameters}\n--- [system] ---\n{system_prompt}\n--- [human] ---\n"
                             f"{assembled_prompt}")
        return request_id

    def log_response(self, request_id: str, response: str, duration_ms: int, tokens: dict,
                     error: Optional[str] = None) -> None:
        self._write({
            "type": "response", "requestId": request_id, "response": response, "durationMs": duration_ms,
            "tokens": tokens, "error": error,
        })
        status = f"ERROR {error}" if error else "OK"
        self._log_transcript(f"===== LLM RESPONSE id={request_id} status={status} duration_ms={duration_ms} "
                             f"tokens={tokens}\n{response}", logging.ERROR if error else logging.INFO)

    def _log_transcript(self, text: str, level: int = logging.INFO) -> None:
        if self._transcript is None:
            return
        try:
            self._transcript.log(level, self._redactor.redact(text))
        except Exception as error:  # noqa: BLE001 - logging must never break the workflow
            print(f"[raglogger] WARNING: could not write LLM transcript: {error}", file=sys.stderr)

    def write_record(self, record: dict[str, Any]) -> None:
        """Appends an arbitrary record (used by the LangChain callback handler)."""
        self._write(record)

    def _write(self, record: dict[str, Any]) -> None:
        if self._path is None:
            return
        record = {"timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z"),
                  "sessionId": self.session_id, **record}
        try:
            line = json.dumps(self._redactor.redact_value(record), ensure_ascii=False)
            with self._lock:
                ensure_parent(self._path)
                with open(self._path, "a", encoding="utf-8") as handle:
                    handle.write(line + "\n")
                    handle.flush()
                    os.fsync(handle.fileno())
        except (OSError, TypeError, ValueError) as error:
            print(f"[raglogger] WARNING: could not write LLM interaction log: {error}", file=sys.stderr)
