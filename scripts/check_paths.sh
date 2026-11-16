#!/usr/bin/env bash
# Verifies data lives in <project>/.raglogger, works from any working directory, and nothing is written to ~/.raglogger.
PROJECT="$(cd "$(dirname "$0")/.." && pwd)"
PY="$PROJECT/venv/bin/python"; test -x "$PY" || PY="$PROJECT/.venv/bin/python"; test -x "$PY" || PY=python3

echo "== resolved paths (from project dir)"
(cd "$PROJECT" && "$PY" -c 'from raglogger.config import Settings; from raglogger.cli import vector_dir_for
s = Settings.from_env(); print("index     ", s.index_path); print("vectors   ", vector_dir_for(s))
print("transcript", s.transcript_log_path); print("jsonl     ", s.interaction_log_path)')

echo "== same command from /tmp (cwd must not matter)"
(cd /tmp && PYTHONPATH="$PROJECT" "$PY" -m raglogger --engine native stats 2>&1 | grep -E '"index"|"chunks"|not found')

echo "== RAGLOGGER_HOME override"
(cd "$PROJECT" && RAGLOGGER_HOME=/tmp/rl-alt "$PY" -c 'from raglogger.config import Settings; print(Settings.from_env().index_path)')

echo "== nothing in the user home"
ls -d "$HOME/.raglogger" 2>/dev/null && echo "WARNING: $HOME/.raglogger exists" || echo "ok: $HOME/.raglogger absent"
