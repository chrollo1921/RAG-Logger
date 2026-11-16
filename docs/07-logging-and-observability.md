# 7. Logging and observability

## 7.1 Log files

| File | Format | Written by | Rotation |
|---|---|---|---|
| `.raglogger/logs/llm-transcript.log` | Readable text | `llm_logging.LlmInteractionCallback` (LangChain) / `interaction_log.InteractionLogger` (native) through the `raglogger.llm.transcript` logger | 10 MB × 5 backups (`RotatingFileHandler`) |
| `.raglogger/logs/llm-interactions.jsonl` | JSON Lines, one record per event | `interaction_log.InteractionLogger._write()` | none (append-only) |
| stderr | `logging` warnings/errors; `-v` for INFO/DEBUG | Python `logging` | – |

Paths: `RAGLOGGER_TRANSCRIPT_LOG` and `RAGLOGGER_INTERACTION_LOG` (`off` disables a file). The transcript logger
has `propagate=False`, so full prompts are never printed to the console.

## 7.2 What is logged for each question

| Event | Hook (LangChain engine) | Content |
|---|---|---|
| `RETRIEVED CONTEXT` | `on_retriever_end` | every document given to the LLM: id, kind, **source**, score, retrieval method (`bm25` / `vector` / `bm25+vector`), full text |
| `LLM REQUEST` | `on_chat_model_start` | session id, backend, model, temperature (number / `omitted` / `omitted (auto-retry)`), max_tokens, the **full system and user messages** (including the context block and question) |
| `LLM RESPONSE` | `on_llm_end` | full answer, duration (ms), prompt/completion/total tokens |
| `LLM ERROR` | `on_llm_error` | exception type and message (secrets masked), duration |

A fallback (for example litellm failing, then ollama) or a temperature retry produces **one REQUEST/ERROR pair
per attempt**, so you can see exactly what each backend received.

Transcript sample (from `scripts/smoke_temperature.sh`):

```
INFO ===== RETRIEVED CONTEXT run=… documents=2
--- [1] id=67d5831c:overview kind=overview source=sample-console.log score=32.787 retrieval=bm25 ---
…
INFO ===== LLM REQUEST run=… session=f407c015f950 backend=litellm model=claude-opus-5-5 temperature=0.1 max_tokens=2048
--- [system] ---
You are a senior SAP Commerce (Hybris) support engineer …
--- [human] ---
Context excerpts from the console log: …
ERROR ===== LLM ERROR run=… duration_ms=69
BadRequestError: … does not support temperature=0.1. Only temperature=1 is supported …
INFO ===== LLM REQUEST run=… backend=litellm model=claude-opus-5-5 temperature=omitted (auto-retry) max_tokens=2048
INFO ===== LLM RESPONSE run=… duration_ms=19 tokens(prompt/completion/total)=765/20/785
<answer text>
```

JSONL records (shortened):

```json
{"timestamp":"2026-11-16T14:10:20Z","sessionId":"f407c015f950","type":"retrieval","runId":"…","documents":[{"metadata":{…},"content":"…"}]}
{"timestamp":"…","sessionId":"…","type":"request","runId":"…","backend":"litellm","model":"claude-opus-5-5","parameters":{"temperature":"omitted","max_tokens":2048},"question":"summarize the errors","contextChunkIds":["…"],"messages":[{"role":"system","content":"…"},{"role":"human","content":"…"}]}
{"timestamp":"…","sessionId":"…","type":"response","runId":"…","response":"…","durationMs":19,"tokens":{"prompt":765,"completion":20,"total":785}}
```

The native engine writes `request`/`response` records with `systemPrompt`, `userPrompt`, `assembledPrompt` and
`contextChunkIds`.

## 7.3 Guarantees

* **Redacted:** every string goes through `Redactor` before it is written (`redact_value()` for JSONL,
  `redact()` for the transcript). API keys from the environment are also masked in error messages.
* **Durable:** JSONL is flushed and `fsync`ed per record, under a lock.
* **Non-blocking:** logging failures print a warning on stderr and never break the request
  (`raise_error = False` on the callback; `try/except` around every write).
* **Values never logged from `.env`:** only variable names appear, at `-v`.

## 7.4 Useful commands

```bash
tail -f .raglogger/logs/llm-transcript.log
grep -E "===== LLM (REQUEST|RESPONSE|ERROR)" .raglogger/logs/llm-transcript.log | tail
python3 -c "import json,os;[print(r['type'],r.get('backend'),r.get('tokens')) for r in map(json.loads,open('.raglogger/logs/llm-interactions.jsonl'))]" | tail
```
