#!/usr/bin/env bash
# End-to-end check of LangChain engine + hybrid retrieval + LLM logging against a local mock gateway.
cd "$(dirname "$0")/.." || exit 1
PY=.venv/bin/python
LOGDIR=$(mktemp -d)
$PY scripts/mock_litellm.py 4011 & MOCK_PID=$!
trap 'kill $MOCK_PID 2>/dev/null; rm -rf "$LOGDIR"' EXIT
sleep 1

export LITELLM_BASE_URL=http://127.0.0.1:4011 LITELLM_MODEL=claude-opus-mock
export LITELLM_API_KEY=dummy-smoke-test-value-000000
export RAGLOGGER_TRANSCRIPT_LOG="$LOGDIR/llm-transcript.log" RAGLOGGER_INTERACTION_LOG="$LOGDIR/llm-interactions.jsonl"

echo "== ask via litellm (mock) with hybrid retrieval"
$PY -m raglogger ask "why could the basket not be priced" --backend litellm,ollama -k 3 2>/dev/null

echo; echo "== transcript log (first 40 lines)"
head -40 "$RAGLOGGER_TRANSCRIPT_LOG"
echo "..."; grep -E "===== (RETRIEVED|LLM)" "$RAGLOGGER_TRANSCRIPT_LOG"

echo; echo "== JSONL audit record types / tokens"
$PY -c "import json,sys; [print(r['type'], r.get('backend'), r.get('tokens'), len(json.dumps(r))) for r in map(json.loads, open(sys.argv[1]))]" "$RAGLOGGER_INTERACTION_LOG"

echo; echo "== key leak check"
if grep -q "$LITELLM_API_KEY" "$LOGDIR"/*; then echo "LEAK: key found in logs"; exit 1; else echo "no key in logs"; fi
