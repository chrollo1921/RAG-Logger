#!/usr/bin/env python3
"""One-off migration: ~/.raglogger  ->  <project>/.raglogger, and clear old home paths from .env.

Usage:  python3 scripts/migrate_data_dir.py            # dry run (shows what would happen)
        python3 scripts/migrate_data_dir.py --apply    # do it

Safe by design: refuses to overwrite existing files in the target, only rewrites .env lines whose value points
into the old directory (prints variable NAMES only, never values), keeps a .env backup with mode 600.
"""
import os
import pathlib
import re
import shutil
import stat
import sys

PROJECT = pathlib.Path(__file__).resolve().parent.parent
OLD = pathlib.Path.home() / ".raglogger"
NEW = PROJECT / ".raglogger"
ENV_FILE = PROJECT / ".env"
PATH_KEYS = ("RAGLOGGER_INDEX", "RAGLOGGER_VECTOR_DIR", "RAGLOGGER_TRANSCRIPT_LOG", "RAGLOGGER_INTERACTION_LOG")
APPLY = "--apply" in sys.argv


def say(message: str) -> None:
    print(("" if APPLY else "[dry-run] ") + message)


def points_to_old_dir(value: str) -> bool:
    expanded = value.strip().strip("'\"").replace("${HOME}", str(pathlib.Path.home())).replace("$HOME", str(
        pathlib.Path.home()))
    expanded = os.path.expanduser(expanded)
    return expanded == str(OLD) or expanded.startswith(str(OLD) + os.sep)


def migrate_data() -> None:
    if not OLD.exists():
        say(f"nothing to move: {OLD} does not exist")
        return
    items = sorted(OLD.iterdir())
    clashes = [item.name for item in items if (NEW / item.name).exists()]
    if clashes:
        sys.exit(f"ABORT: {NEW} already contains {', '.join(clashes)}. Move or delete them first; nothing changed.")
    say(f"move {len(items)} item(s) from {OLD} to {NEW}: {', '.join(item.name for item in items)}")
    if APPLY:
        NEW.mkdir(mode=0o700, exist_ok=True)
        for item in items:
            shutil.move(str(item), str(NEW / item.name))
        OLD.rmdir()
        print(f"moved; removed empty {OLD}")


def migrate_env() -> None:
    if not ENV_FILE.is_file():
        say(".env not found; nothing to update")
        return
    lines = ENV_FILE.read_text(encoding="utf-8").splitlines(keepends=True)
    changed = []
    for number, line in enumerate(lines):
        match = re.match(r"^(\s*(?:export\s+)?)(\w+)(\s*=\s*)(.*?)(\s*)$", line.rstrip("\n"))
        if match and match.group(2) in PATH_KEYS and points_to_old_dir(match.group(4)):
            lines[number] = f"{match.group(1)}{match.group(2)}{match.group(3)}\n"
            changed.append(match.group(2))
    if not changed:
        say(".env has no path settings pointing at ~/.raglogger; unchanged")
        return
    say(f".env: clear {', '.join(changed)} (blank = default inside <project>/.raglogger)")
    if APPLY:
        backup = ENV_FILE.with_name(".env.bak-home-paths")
        shutil.copy2(ENV_FILE, backup)
        os.chmod(backup, stat.S_IRUSR | stat.S_IWUSR)
        ENV_FILE.write_text("".join(lines), encoding="utf-8")
        print(f"updated .env (backup: {backup.name}, mode 600)")


migrate_env()
migrate_data()
if not APPLY:
    print("Run again with --apply to perform these steps.")
