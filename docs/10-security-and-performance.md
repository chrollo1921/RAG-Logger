# 10. Security, privacy and performance

## 10.1 Data flow and where data can leave the machine

| Data | Stays local | Leaves the machine when |
|---|---|---|
| Raw console log | ✔ never copied; read in place | – |
| Chunks (redacted) | `.raglogger/index.json.gz` | – |
| Vectors | `.raglogger/index-chroma/` | `--embeddings litellm`: **all chunks** are sent to the gateway at ingest |
| Retrieved context (top-k chunks, ≤ 12,000 chars) | – | every `ask` with `litellm`/`openai` (to the gateway / provider). `ollama` stays local. |
| Prompts + answers | `.raglogger/logs/*` | – |
| API key | `.env` (chmod 600) and process environment | only in the `Authorization` header to the configured endpoint |

## 10.2 Controls

| Control | Where |
|---|---|
| Secret masking before chunking (passwords, tokens, bearer/basic auth, API keys, JWTs, private keys, e-mails) | `redaction.py`, applied in `IngestService._redacted_entries()` |
| Masking again for everything logged | `InteractionLogger._write()`, `LlmInteractionCallback._emit()` |
| Keys never stored in settings / index / logs; scrubbed from error messages | `config.py` (only the variable *name*), `scrub_secrets()`, `_scrub()` |
| `.env` git-ignored, permission warning | `.gitignore`, `envfile._is_group_or_world_readable()` |
| Indexes, vector stores and logs git-ignored | `.gitignore` (`*.json.gz`, `*-chroma/`, `*.log`, `*.jsonl`) |
| Chroma telemetry disabled | `vectorstore.open_chroma()` |
| Only `http(s)` endpoints accepted | `generators._HttpLlmGenerator._post()` |
| Answers grounded in excerpts with citations | `SYSTEM_PROMPT` |

**Not masked:** cart/order/account ids, correlation ids, IPs and hostnames. They are useful for diagnosis but
may be personal or business data. Treat the index, the vector store and the logs as **confidential production
data**, and check your gateway's retention policy before using `litellm`.

## 10.3 Performance characteristics

| Operation | Cost | Notes |
|---|---|---|
| Parse + chunk | about 3.5 s per 70k-line / 18.5 MB log | streaming; memory grows with distinct templates, not file size |
| Local embedding | about 2 min per 1,800 chunks (CPU) | only new/changed files with `--append` |
| Index load per command | < 0.1 s per 2,000 chunks | whole `index.json.gz` is rewritten on every ingest; fine for dozens of logs |
| Local model load per command | 5–25 s | use `chat` for several questions |
| BM25 search | linear in chunk count | fine at about 10⁴ chunks; would need an inverted index at 10⁶ |
| LLM call | 1 request per question (+1 if temperature is rejected, +1 per fallback) | about 3k prompt tokens with defaults; lower `-k` or `RAGLOGGER_MAX_CONTEXT_CHARS` to save cost |
| Timeouts | 120 s per backend | a long chain can wait several timeouts |
