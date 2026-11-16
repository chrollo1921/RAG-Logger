"""Minimal ``.env`` loader (standard library only, so the native engine keeps working without the venv).

Supported syntax::

    # comment
    KEY=value
    export KEY=value              # 'export' prefix is ignored
    KEY="value with spaces"       # double quotes: \\n \\t \\" \\\\ escapes and ${OTHER} expansion
    KEY='literal ${not expanded}' # single quotes: taken literally
    KEY=value # inline comment    # unquoted: whitespace + '#' starts a comment
    KEY=${RAGLOGGER_HOME}/index   # ${VAR} expands from earlier lines or the environment

Precedence: variables already set in the process environment (shell exports, CI secrets) win over the file unless
``override=True``. Values are never logged; only variable *names* are reported at DEBUG level.
"""
from __future__ import annotations

import logging
import os
import re
import stat
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, MutableMapping, Optional

LOG = logging.getLogger(__name__)

ENV_FILE_VARIABLE = "RAGLOGGER_ENV_FILE"
DISABLED_VALUES = ("off", "none", "false", "0")
PROJECT_ROOT = Path(__file__).resolve().parent.parent  # same value as raglogger.paths.PROJECT_ROOT
_LINE_RE = re.compile(r"^\s*(?:export\s+)?(?P<key>[A-Za-z_][A-Za-z0-9_]*)\s*=\s*(?P<value>.*?)\s*$")
_EXPANSION_RE = re.compile(r"\$\{(?P<name>[A-Za-z_][A-Za-z0-9_]*)\}")
_ESCAPES = {"n": "\n", "t": "\t", "r": "\r", '"': '"', "\\": "\\", "$": "$"}


class EnvFileError(ValueError):
    """Raised when an explicitly requested env file is missing or unreadable."""


@dataclass
class EnvLoadResult:
    path: Optional[Path]
    loaded: list[str] = field(default_factory=list)
    skipped: list[str] = field(default_factory=list)  # already set in the environment, not overridden
    warnings: list[str] = field(default_factory=list)


def parse_env_text(text: str,
                   environ: Optional[MutableMapping[str, str]] = None) -> tuple[dict[str, str], list[str]]:
    """Parses ``.env`` content into ``(values, warnings)``; ``environ`` is consulted for ``${VAR}`` expansion."""
    environ = os.environ if environ is None else environ
    values: dict[str, str] = {}
    warnings: list[str] = []
    for number, raw in enumerate(text.splitlines(), start=1):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        match = _LINE_RE.match(raw)
        if not match:
            # Never echo the line itself: it may be a secret with a malformed key.
            warnings.append(f".env line {number}: ignored (expected KEY=value)")
            continue
        key = match.group("key")
        try:
            values[key] = _parse_value(match.group("value"), lambda name: values.get(name, environ.get(name, "")))
        except ValueError as error:
            warnings.append(f".env line {number} ({key}): {error}")
    return values, warnings


def _parse_value(value: str, lookup: Callable[[str], str]) -> str:
    if value[:1] in ("'", '"'):
        quote = value[0]
        end = _closing_quote(value, quote)
        if end < 0:
            raise ValueError(f"missing closing {quote}")
        trailing = value[end + 1:].strip()
        if trailing and not trailing.startswith("#"):
            raise ValueError("unexpected text after closing quote")
        inner = value[1:end]
        return inner if quote == "'" else _expand(_unescape(inner), lookup)
    comment = re.search(r"\s#", value)
    if comment:
        value = value[:comment.start()]
    return _expand(value.strip(), lookup)


def _closing_quote(value: str, quote: str) -> int:
    index = 1
    while index < len(value):
        if quote == '"' and value[index] == "\\":
            index += 2
            continue
        if value[index] == quote:
            return index
        index += 1
    return -1


def _unescape(text: str) -> str:
    return re.sub(r"\\(.)", lambda match: _ESCAPES.get(match.group(1), "\\" + match.group(1)), text)


def _expand(text: str, lookup: Callable[[str], str]) -> str:
    return _EXPANSION_RE.sub(lambda match: lookup(match.group("name")), text)

def resolve_env_file(explicit: Optional[str] = None,
                     environ: Optional[MutableMapping[str, str]] = None) -> tuple[Optional[Path], bool]:
    """Returns ``(path, required)``. Order: ``--env-file`` > ``$RAGLOGGER_ENV_FILE`` > ``./.env`` > project ``.env``."""
    environ = os.environ if environ is None else environ
    requested = explicit if explicit is not None else environ.get(ENV_FILE_VARIABLE)
    if requested is not None:
        if requested.strip().lower() in DISABLED_VALUES:
            return None, False
        return Path(requested).expanduser(), True
    for candidate in (Path.cwd() / ".env", PROJECT_ROOT / ".env"):
        if candidate.is_file():
            return candidate, False
    return None, False


def load_env_file(explicit: Optional[str] = None, override: bool = False,
                  environ: Optional[MutableMapping[str, str]] = None) -> EnvLoadResult:
    """Loads the resolved env file into ``environ`` (default ``os.environ``)."""
    environ = os.environ if environ is None else environ
    path, required = resolve_env_file(explicit, environ)
    if path is None:
        return EnvLoadResult(path=None)
    if not path.is_file():
        if required:
            raise EnvFileError(f"Env file not found: {path}")
        return EnvLoadResult(path=None)
    try:
        text = path.read_text(encoding="utf-8-sig")  # tolerate a BOM written by Windows editors
    except (OSError, UnicodeDecodeError) as error:
        raise EnvFileError(f"Could not read env file {path}: {error}") from error

    result = EnvLoadResult(path=path)
    values, result.warnings = parse_env_text(text, environ)
    for key, value in values.items():
        if value == "":
            # Blank placeholders (KEY=) mean "not configured": keep the built-in default instead of forcing "".
            continue
        if not override and key in environ:
            result.skipped.append(key)
            continue
        environ[key] = value
        result.loaded.append(key)
    if _is_group_or_world_readable(path):
        result.warnings.append(f"{path} is readable by other users; run: chmod 600 {path}")
    LOG.debug("Loaded %d variable(s) from %s: %s (kept from environment: %s)", len(result.loaded), path,
              ", ".join(result.loaded) or "-", ", ".join(result.skipped) or "-")
    return result


def _is_group_or_world_readable(path: Path) -> bool:
    if os.name != "posix":
        return False
    try:
        return bool(path.stat().st_mode & (stat.S_IRGRP | stat.S_IROTH))
    except OSError:
        return False
