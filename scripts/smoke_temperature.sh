#!/usr/bin/env bash
# Reproduces "does not support temperature=0.1. Only temperature=1 is supported" with a local mock gateway and
# shows both fixes: automatic retry without temperature, and LITELLM_TEMPERATURE=omit (no failed round-trip).
cd "$(dirname "$0")/.." || exit 1
PY=${PY:-venv/bin/python}
test -x "$PY" || PY=.venv/bin/python
WORK=$(mktemp -d)
MOCK_REJECT_TEMPERATURE=1 $PY scripts/mock_litellm.py 4012 & MOCK_PID=$!
trap 'kill $MOCK_PID 2>/dev/null; rm -rf "$WORK"' EXIT
sleep 1

export LITELLM_BASE_URL=http://127.0.0.1:4012 LITELLM_MODEL=claude-opus-5-5
export LITELLM_API_KEY=dummy-smoke-test-value-000000
export RAGLOGGER_TRANSCRIPT_LOG="$WORK/transcript.log" RAGLOGGER_INTERACTION_LOG="$WORK/interactions.jsonl"
RL="$PY -m raglogger --env-file off --embeddings none --index $WORK/idx.json.gz"
$RL ingest tests/fixtures/sample-console.log > /dev/null

echo "== 1) temperature 0.1 sent -> gateway rejects -> automatic retry without temperature"
RAGLOGGER_TEMPERATURE=0.1 $RL ask "summarize the errors" --backend=litellm -k 2
echo; echo "== 2) LITELLM_TEMPERATURE=omit -> single request, no warning"
LITELLM_TEMPERATURE=omit $RL ask "summarize the errors" --backend=litellm -k 2
echo; echo "== 3) temperature values logged per request"
grep -E "===== LLM (REQUEST|RESPONSE|ERROR)" "$RAGLOGGER_TRANSCRIPT_LOG" | sed -E 's/^[0-9-]+ [0-9:,]+ //; s/run=[0-9a-f-]+ //'
