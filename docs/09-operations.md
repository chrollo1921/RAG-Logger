# 9. Operations: commands, recipes, troubleshooting

Run everything from the project root inside WSL, with your virtual environment (`venv/bin/python` or
`source venv/bin/activate` and then `python`).

## 9.1 Typical workflows

```bash
# one-time setup
python3 -m venv --without-pip venv && curl -sS https://bootstrap.pypa.io/get-pip.py | venv/bin/python
venv/bin/pip install -r requirements.txt -r requirements-local-embeddings.txt
cp .env.example .env && chmod 600 .env       # then fill in LITELLM_* values

# index logs
python -m raglogger ingest /mnt/c/Users/<you>/Downloads/console-20260829.log
python -m raglogger ingest --append /mnt/c/logs/node2/console.log
python -m raglogger sources

# ask
python -m raglogger ask "summarize the errors" --backend litellm
python -m raglogger ask "why did removing item fail" --backend litellm,ollama --source node2 -k 8
python -m raglogger search "tax price user group" --level ERROR --since 2026-08-29T00:15
python -m raglogger chat --backend litellm
```

## 9.2 Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| `does not support temperature=0.1. Only temperature=1 is supported` | Model only accepts its default temperature | Already handled by auto-retry. To skip the extra round-trip, set `LITELLM_TEMPERATURE=omit` (already in your `.env`) or pass `--temperature omit` |
| `[LLM backend unavailable: …]` followed by excerpts | Every backend in the chain failed | Read the reasons in the message or in `llm-transcript.log` (`LLM ERROR`) |
| `LITELLM_API_KEY is not set` | Key missing from `.env` / environment | Add it to `.env` |
| `No model configured for backend 'litellm'` | `LITELLM_MODEL` empty | `python -m raglogger models`, then set `LITELLM_MODEL` |
| `Vector store was built with embeddings X but Y is configured` | `--embeddings` / model changed since ingest | Use the same setting, or re-run `ingest` |
| `Vector store not found` | Index built with `--embeddings none` or `--engine native` | Re-run `ingest` with the LangChain engine |
| `older single-file version and cannot be appended to` | Index predates multi-file support | Run one plain `ingest` (no `--append`) |
| `No indexed source matches --source 'x'` | Typo or file not ingested | The message lists the indexed sources |
| `Indexed … -> 3 events` on a real log | Unusual log format | Check that lines look like `INFO \| jvm 1 \| main \| … \| [timestamp] LEVEL [thread] [Logger] …` |
| Slow first question | Local embedding model loading (5–25 s) | Use `chat` for several questions, or `--embeddings none` |
| `.env is readable by other users` warning | File mode | `chmod 600 .env` |
| `$'\r': command not found` in a script | Windows line endings | `python3 scripts/dev_check.py --no-tests` |

## 9.3 Resetting

```bash
rm -rf .raglogger/index.json.gz .raglogger/index-chroma      # drop index + vectors (logs kept)
rm -f .raglogger/logs/llm-*.log* .raglogger/logs/*.jsonl       # drop LLM logs
```
