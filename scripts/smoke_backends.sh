#!/usr/bin/env bash
# Manual smoke test of the LLM backend wiring (no real key or server required).
cd "$(dirname "$0")/.." || exit 1
unset LITELLM_API_KEY

echo "== litellm without model"
python3 -m raglogger ask why removing item failed --backend litellm
echo "exit=$?"

echo "== chain litellm,ollama (no key, no ollama) -> falls back to extractive"
LITELLM_MODEL=my-claude python3 -m raglogger ask why removing item failed --backend litellm,ollama -k 1 2>&1 | head -6

echo "== models against unreachable proxy"
LITELLM_API_KEY=dummy-value-123456 python3 -m raglogger models --base-url http://127.0.0.1:9
echo "exit=$?"
