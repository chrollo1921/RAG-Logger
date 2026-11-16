"""Resolves ``ingest`` arguments (files, directories) into uniquely labelled console log sources (stdlib only)."""
from __future__ import annotations

import fnmatch
import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Optional

DEFAULT_PATTERN = "*.log*"  # console.log, console-20260829.log, console.log.1, console.log.2.gz, ...


@dataclass(frozen=True)
class SourceFile:
    path: Path
    label: str

    @property
    def key(self) -> str:
        """Short stable id used to namespace chunk ids, so identical chunks from different files never collide."""
        return source_key(self.label)

    def fingerprint(self) -> dict:
        stat = self.path.stat()
        return {"size": stat.st_size, "mtime_ns": stat.st_mtime_ns}


def source_key(label: str) -> str:
    return hashlib.sha1(label.encode("utf-8")).hexdigest()[:8]


def discover(paths: list[Path], pattern: str = DEFAULT_PATTERN) -> list[Path]:
    """Expands directories (recursively, matching ``pattern``) and de-duplicates; preserves argument order."""
    found: list[Path] = []
    seen: set[Path] = set()
    for raw in paths:
        path = raw.expanduser()
        if path.is_dir():
            candidates = sorted(item for item in path.rglob("*")
                                if item.is_file() and fnmatch.fnmatch(item.name, pattern))
            if not candidates:
                raise FileNotFoundError(f"No files matching '{pattern}' in directory {path}")
        elif path.is_file():
            candidates = [path]
        else:
            raise FileNotFoundError(f"Console log not found: {path}")
        for candidate in candidates:
            resolved = candidate.resolve()
            if resolved not in seen:
                seen.add(resolved)
                found.append(resolved)
    return found


def assign_labels(paths: list[Path], existing: Optional[Mapping[str, str]] = None,
                  explicit_label: Optional[str] = None) -> list[SourceFile]:
    """Gives each file the shortest unique label: ``console.log`` -> ``node1/console.log`` when names clash.

    ``existing`` maps labels already in the index to their absolute path: re-ingesting the same path reuses its
    label (replacing that source), while a *different* file with the same name gets a longer, distinct label.
    """
    existing = dict(existing or {})
    if explicit_label is not None:
        if len(paths) != 1:
            raise ValueError("--label can only be used when ingesting exactly one file")
        label = explicit_label.strip()
        if not label:
            raise ValueError("--label must not be empty")
        owner = existing.get(label)
        if owner and owner != str(paths[0]):
            raise ValueError(f"Label '{label}' is already used by {owner}; choose another label or remove it")
        return [SourceFile(paths[0], label)]

    by_path = {path: label for label, path in existing.items()}
    taken = set(existing)
    result: list[SourceFile] = []
    for path in paths:
        if str(path) in by_path:
            label = by_path[str(path)]
        else:
            label = _shortest_unique(path, paths, taken)
        taken.add(label)
        result.append(SourceFile(path, label))
    return result


def _shortest_unique(path: Path, batch: list[Path], taken: set[str]) -> str:
    parts = path.parts
    for depth in range(1, len(parts) + 1):
        label = "/".join(parts[-depth:]).lstrip("/")
        clashes_in_batch = any(other != path and "/".join(other.parts[-depth:]).lstrip("/") == label
                               for other in batch)
        if not clashes_in_batch and label not in taken:
            return label
    return str(path)
