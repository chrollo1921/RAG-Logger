#!/usr/bin/env bash
# Verifies .env loading: real .env is picked up, shell overrides it, permission warning, no values printed.
cd "$(dirname "$0")/.." || exit 1

echo "== .env picked up (names only, at -v)"
.venv/bin/python -m raglogger -v --engine native stats 2>&1 | grep -E "Environment file|Loaded [0-9]+ variable" \
  | sed -E 's/^[0-9-]+ [0-9:,]+ //'

echo "== shell variable overrides .env"
RAGLOGGER_INDEX=/tmp/raglogger-does-not-exist.json.gz python3 -m raglogger stats
echo "exit=$? (2 expected: index from shell override was used)"

echo "== permission warning when .env is world-readable"
chmod 644 .env
python3 -m raglogger --engine native stats 2>&1 | grep WARNING
chmod 600 .env
stat -c '%a %n' .env

echo "== --env-file off ignores .env"
RAGLOGGER_TRANSCRIPT_LOG= python3 -m raglogger --env-file off -v --engine native stats 2>&1 \
  | grep -c "Environment file" | sed 's/^/env file log lines: /'
