# 8. Configuration reference

## 8.1 Precedence

```
CLI flag  >  shell environment variable  >  .env file  >  built-in default
```

`.env` lookup: `--env-file <path>` → `$RAGLOGGER_ENV_FILE` → `./.env` → `<project>/.env` (`envfile.resolve_env_file()`).
`--env-file off` disables it. Blank values (`KEY=`) are ignored. `.env.example` lists every variable (a test
enforces this).

## 8.2 Where data lives

All runtime data defaults to **`<project>/.raglogger/`**, e.g. `/home/karthik/projects/RAG-Logger/.raglogger/`
(`raglogger/paths.py`). It is git-ignored and created with mode 700.

```
.raglogger/
├── index.json.gz              chunks + BM25 data + source registry
├── index-chroma/              Chroma vector store
└── logs/
    ├── llm-transcript.log     readable LLM transcript (rotating)
    └── llm-interactions.jsonl structured audit log
```

* `RAGLOGGER_HOME` moves the whole directory; the individual path variables override single files.
* **Relative paths** in these variables are resolved against the **project root**, not the shell's current
  directory, so `python -m raglogger …` behaves the same from any folder. (`--index` on the command line is
  relative to the current directory, like any CLI path.)
* `~` is expanded. Absolute paths are used as-is.
* Tests never touch this directory: `tests/_isolation.py` points `RAGLOGGER_HOME` at a temporary folder.
* Older versions stored data in `~/.raglogger`. `scripts/migrate_data_dir.py` moves it (dry run by default,
  `--apply` to perform it, never overwrites).

## 8.3 Environment variables

### LLM backends

| Variable | Default | Meaning |
|---|---|---|
| `RAGLOGGER_BACKEND` | `extractive` | default chain, e.g. `litellm,ollama,extractive` |
| `LITELLM_BASE_URL` | `http://localhost:4000` | gateway base URL (no `/chat/completions`) |
| `LITELLM_API_KEY` | – | virtual key (secret; `.env` only) |
| `LITELLM_MODEL` | – (required) | model alias on the gateway, see `raglogger models` |
| `LITELLM_TEMPERATURE` | inherit | `0`–`2` or `omit` (your `.env`: `omit`) |
| `LITELLM_EMBEDDING_MODEL` | – | embedding alias, for `RAGLOGGER_EMBEDDINGS=litellm` |
| `OLLAMA_BASE_URL` / `OLLAMA_MODEL` / `OLLAMA_TEMPERATURE` | `http://localhost:11434` / `llama3.1:8b` / inherit | local Ollama |
| `OPENAI_BASE_URL` / `OPENAI_MODEL` / `OPENAI_TEMPERATURE` | `https://api.openai.com/v1` / `gpt-4o-mini` / inherit | other OpenAI-compatible endpoints |
| `RAGLOGGER_API_KEY` | – | key for the `openai` backend |
| `RAGLOGGER_MODEL` / `RAGLOGGER_BASE_URL` | – | override model/URL of the **first** backend in the chain |
| `RAGLOGGER_TEMPERATURE` | `0.1` | default temperature (`0`–`2` or `omit`) |
| `RAGLOGGER_MAX_TOKENS` | `2048` | max answer tokens |
| `RAGLOGGER_TIMEOUT` | `120` | HTTP timeout in seconds (LLM and embeddings) |
| `RAGLOGGER_MAX_CONTEXT_CHARS` | `12000` | size limit of the context block sent to the LLM |

### Retrieval, storage, logs

| Variable | Default | Meaning |
|---|---|---|
| `RAGLOGGER_HOME` | `<project>/.raglogger` | **data directory**: index, vector store and LLM logs all default to paths inside it |
| `RAGLOGGER_ENGINE` | `auto` | `auto` / `langchain` / `native` |
| `RAGLOGGER_EMBEDDINGS` | `local` | `local` / `litellm` / `none` |
| `RAGLOGGER_EMBEDDING_MODEL` | `sentence-transformers/all-MiniLM-L6-v2` (local) | embedding model override |
| `RAGLOGGER_INDEX` | `$RAGLOGGER_HOME/index.json.gz` | chunk store / BM25 index |
| `RAGLOGGER_VECTOR_DIR` | `<index name>-chroma` next to the index | Chroma directory |
| `RAGLOGGER_TRANSCRIPT_LOG` | `$RAGLOGGER_HOME/logs/llm-transcript.log` | path or `off` |
| `RAGLOGGER_INTERACTION_LOG` | `$RAGLOGGER_HOME/logs/llm-interactions.jsonl` | path or `off` |
| `RAGLOGGER_ENV_FILE` | – | env file to load, or `off` |
| `ANONYMIZED_TELEMETRY` | `False` | Chroma telemetry (keep off) |

## 8.4 CLI

Global flags (before the command): `--index PATH`, `--env-file PATH|off`, `--engine auto|langchain|native`,
`--embeddings local|litellm|none`, `--embedding-model NAME`, `-v/--verbose`.

| Command | Flags |
|---|---|
| `ingest [FILES/DIRS…]` | `--append`, `--force`, `--remove LABEL` (repeatable), `--label NAME`, `--pattern GLOB` (default `*.log*`), `--window-minutes N` (default 5) |
| `ask QUESTION…` / `chat` | `-k/--top-k` (6), `--level`, `--logger`, `--kind`, `--source`, `--since`, `--until`, `--json`, `--backend`, `--model`, `--base-url`, `--temperature T\|omit` |
| `search QUESTION…` | same filters, `-k`, `--json` (retrieval only, no LLM) |
| `sources` | `--json` |
| `stats` | – |
| `models` | `--backend litellm\|openai`, `--base-url` |

Exit codes: `0` ok, `2` configuration/input error, `3` LLM error (e.g. `models` can't reach the gateway),
`130` Ctrl-C.
