"""Single place that decides where RAG-Logger keeps its runtime data (index, vector store, LLM logs).

Default: ``<project root>/.raglogger`` (the folder containing the ``raglogger`` package), so each checkout has its
own data. Override with ``RAGLOGGER_HOME``. Relative paths in settings are resolved against the project root, not
the current working directory, so commands behave the same from any directory. Standard library only.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Mapping, Optional

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR_NAME = ".raglogger"
HOME_VARIABLE = "RAGLOGGER_HOME"


def resolve_path(value: str | Path, base: Path = PROJECT_ROOT) -> Path:
    """Expands ``~`` and makes relative paths absolute against ``base`` (the project root by default)."""
    path = Path(value).expanduser()
    return path if path.is_absolute() else (base / path)


def data_dir(env: Optional[Mapping[str, str]] = None) -> Path:
    """``$RAGLOGGER_HOME`` if set, else ``<project root>/.raglogger``. Read at call time (after ``.env`` loading)."""
    env = os.environ if env is None else env
    configured = (env.get(HOME_VARIABLE) or "").strip()
    return resolve_path(configured) if configured else PROJECT_ROOT / DATA_DIR_NAME


def ensure_parent(path: Path) -> None:
    """Creates ``path``'s parent directories (see :func:`ensure_dir`)."""
    ensure_dir(path.parent)


def ensure_dir(directory: Path) -> None:
    """Creates ``directory``. When it lies inside the data dir, every created level (including the data dir itself)
    is owner-only (0700), because the data holds production log content and full LLM prompts."""
    if directory.exists():
        return
    root = data_dir()
    try:
        relative = directory.resolve().relative_to(root.resolve())
    except (OSError, ValueError):
        directory.mkdir(parents=True, exist_ok=True)  # explicit path outside the data dir: default permissions
        return
    root.parent.mkdir(parents=True, exist_ok=True)
    current = root
    for part in ("", *relative.parts):
        current = current / part if part else current
        if not current.exists():
            current.mkdir(mode=0o700, exist_ok=True)


def default_index_path(env: Optional[Mapping[str, str]] = None) -> Path:
    return data_dir(env) / "index.json.gz"


def default_transcript_path(env: Optional[Mapping[str, str]] = None) -> Path:
    return data_dir(env) / "logs" / "llm-transcript.log"


def default_interaction_log_path(env: Optional[Mapping[str, str]] = None) -> Path:
    return data_dir(env) / "logs" / "llm-interactions.jsonl"
