# RAG-Logger

Retrieval-augmented question answering over **SAP Commerce (Hybris) console logs**
(Tanuki wrapper + Log4j2 format, including ANSI colour codes).

> 📚 **Full documentation: [`docs/`](docs/README.md)**: [file reference](docs/02-file-reference.md) ·
> [chunking & storage](docs/03-chunking-and-storage.md) · [embeddings & vector store](docs/04-embeddings-and-vector-store.md) ·
> [LLM requests & parameters](docs/06-llm-requests.md) · [logging](docs/07-logging-and-observability.md) ·
> [configuration](docs/08-configuration.md) · [troubleshooting](docs/09-operations.md)

```
console.log ─► Parser ─► Redactor ─► Chunker ─┬─► BM25 index (gzip JSON)
               (events,  (secrets,   (templates,└─► Chroma vector DB (embeddings: local HF | LiteLLM)
               stack      e-mails)    incidents,
               traces)                windows)
question ─► LangChain HybridLogRetriever (vector + BM25, Reciprocal Rank Fusion, filters)
         ─► ChatPromptTemplate | ChatModel (LiteLLM/Claude ▸ Ollama ▸ …fallback) | StrOutputParser
         ─► answer + sources            ▲ every retrieval, prompt and response logged (redacted)
```

Two engines, selected with `--engine` (default `auto`):

| Engine | Needs | Retrieval | LLM client |
|---|---|---|---|
| `langchain` (auto when installed) | `.venv` with `requirements*.txt` | **hybrid**: Chroma vectors + BM25 | LangChain `ChatOpenAI` (LiteLLM/OpenAI) / `ChatOllama` |
| `native` | Python 3.10 stdlib only | BM25 | built-in `urllib` clients |

## Setup (WSL)

```bash
cd ~/projects/RAG-Logger
python3 -m venv --without-pip .venv && curl -sS https://bootstrap.pypa.io/get-pip.py | .venv/bin/python
.venv/bin/pip install -r requirements.txt                      # LangChain, Chroma, OpenAI/Ollama clients
.venv/bin/pip install -r requirements-local-embeddings.txt     # optional: local CPU embeddings (~1 GB, torch CPU)
```

## Usage

```bash
PY=.venv/bin/python
$PY -m raglogger ingest /mnt/c/Users/<you>/Downloads/console-20260829.log          # BM25 + vectors (local)
$PY -m raglogger --embeddings litellm ingest <log>      # embeddings via gateway (needs LITELLM_EMBEDDING_MODEL)
$PY -m raglogger --embeddings none ingest <log>         # BM25 only
$PY -m raglogger stats
$PY -m raglogger search "why could the basket not be priced" -k 5     # shows bm25 / vector / bm25+vector
$PY -m raglogger ask "why did removing item fail" --backend litellm,ollama
$PY -m raglogger chat --backend litellm,ollama,extractive
$PY -m raglogger --engine native ask "..."               # stdlib-only path, no LangChain
```

Filters: `--level`, `--logger` (substring), `--kind incident|template|window|overview`, `--since`, `--until`, `-k`.
They apply to **both** BM25 and vector hits. Use the same `--embeddings`/`--embedding-model` for `ingest` and
queries; a mismatch is detected from the vector store manifest and reported.

## Multiple console logs

One index can hold any number of console logs: several nodes, several days, and rotated or `.gz` files.

```bash
PY=.venv/bin/python
$PY -m raglogger ingest node1/console.log node2/console.log      # rebuild the index from exactly these files
$PY -m raglogger ingest /mnt/c/logs/prod                         # a directory: searched recursively for *.log*
$PY -m raglogger ingest --append /mnt/c/logs/console-20260830.log  # add or replace a file, keep the rest
$PY -m raglogger ingest --append /mnt/c/logs/prod                # unchanged files are skipped (size + mtime)
$PY -m raglogger ingest --append --force node1/console.log       # re-ingest even if unchanged
$PY -m raglogger ingest --append --label app01 /var/log/console.log  # custom label (single file only)
$PY -m raglogger ingest --remove node2/console.log               # drop a source (repeatable)
$PY -m raglogger sources                                         # label, chunks, lines, ERROR/WARN, path
$PY -m raglogger ask "why did removing item fail" --source node2 # restrict to label/path substring
```

* **Labels:** each file gets the shortest unique label, e.g. `console.log`, or `node1/console.log` and
  `node2/console.log` when file names clash. Every hit, answer source and LLM context block shows its label.
* **Results per file:** the same error in two files gives two separate results, each with its own counts and
  timestamps, so you can compare nodes. Chunk ids are namespaced per file, so nothing is overwritten.
* **`--source`:** case-insensitive substring of the label *or* the original path, so `--source node2` also
  works when the label is just `console-node.log`. Works with every other filter.
* **Rebuild vs append:** without `--append` the index is rebuilt from exactly the given paths (old sources are
  dropped). With `--append` / `--remove`, only the affected sources are re-parsed and re-embedded. Other files'
  vectors are untouched, so adding one log to a large index is fast.
* **Older indexes:** an index built before multi-file support cannot be appended to. Run one plain `ingest`
  to rebuild it.

## Embeddings / vector DB

| `--embeddings` | Model (override `--embedding-model` / `RAGLOGGER_EMBEDDING_MODEL`) | Data leaves machine? |
|---|---|---|
| `local` (default) | `sentence-transformers/all-MiniLM-L6-v2` (384-d, CPU) | No (model downloaded once from Hugging Face) |
| `litellm` | `LITELLM_EMBEDDING_MODEL` alias on your gateway | Yes – all chunks are sent at ingest |
| `none` | – | – |

Vectors are stored in Chroma next to the index (`.raglogger/index-chroma/`, override `RAGLOGGER_VECTOR_DIR`),
cosine distance, Chroma telemetry disabled. Re-ingest replaces the collection.

## Configuration (.env)

All settings and API keys are read from environment variables, which can live in a `.env` file:

```bash
cp .env.example .env && chmod 600 .env     # already done once; then edit .env
```

* **Lookup order:** `--env-file <path>` → `$RAGLOGGER_ENV_FILE` → `./.env` → `<project>/.env`.
  `--env-file off` (or `RAGLOGGER_ENV_FILE=off`) disables loading.
* **Precedence:** variables already exported in the shell win over `.env`, so one-off overrides still work:
  `RAGLOGGER_BACKEND=ollama .venv/bin/python -m raglogger ask "..."`.
* **Blank values** (`KEY=`) are ignored and the built-in default applies.
* **Syntax:** `KEY=value`, `export KEY=value`, `"double quoted"` (escapes plus `${VAR}` expansion),
  `'single quoted'` (literal), `# comments`. A UTF-8 BOM or Windows CRLF line endings are fine.
* **Not committed:** `.env` and `.env.*` are git-ignored. `.env.example` is the committed template and must
  never contain real values. A warning is printed if `.env` is readable by other users.
* The loader is standard-library only, so it also works with the system `python3` (native engine).
* Values are never printed or logged. `-v` logs only the variable *names* that were loaded.

## LLM interaction logging

Every LLM call (both engines) writes two **redacted** logs:

| File | Format | Contents |
|---|---|---|
| `.raglogger/logs/llm-transcript.log` | human-readable, rotating 10 MB × 5 | `RETRIEVED CONTEXT` (every document fed to the LLM, with id/kind/score/retrieval source), `LLM REQUEST` (backend, model, temperature, max_tokens, full system + human messages incl. the context block and question), `LLM RESPONSE` (full text, duration, prompt/completion/total tokens) or `LLM ERROR` |
| `.raglogger/logs/llm-interactions.jsonl` | JSON Lines | the same data as structured `retrieval` / `request` / `response` records with session/run ids |

In the LangChain engine this is a `BaseCallbackHandler` (`raglogger/llm_logging.py`) attached to the retriever
and every chain run, so fallbacks (`litellm` failing → `ollama`) are logged per attempt.
Override with `RAGLOGGER_TRANSCRIPT_LOG` / `RAGLOGGER_INTERACTION_LOG` (path or `off`).

## LLM backends (optional)

| Backend | Key / config | Notes |
|---|---|---|
| `litellm` | `LITELLM_API_KEY`, `LITELLM_BASE_URL` (default `http://localhost:4000`), `LITELLM_MODEL` | LiteLLM proxy / gateway, e.g. Claude behind your company gateway |
| `ollama` | `OLLAMA_BASE_URL` (default `http://localhost:11434`), `OLLAMA_MODEL` (default `llama3.1:8b`) | local, data stays on machine |
| `openai` | `RAGLOGGER_API_KEY`, `OPENAI_BASE_URL`, `OPENAI_MODEL` | any other OpenAI-compatible endpoint (vLLM, LM Studio, ...) |

### LiteLLM (Claude) + Ollama fallback

```bash
# Put these in your shell session (or ~/.bashrc) – never in the repo.
export LITELLM_BASE_URL="https://<your-litellm-gateway>"     # no trailing /chat/completions
read -rs LITELLM_API_KEY && export LITELLM_API_KEY             # paste key, not echoed / not in history
python3 -m raglogger models                                    # lists exact model aliases your key may use
export LITELLM_MODEL="<alias from the list above>"

python3 -m raglogger ask "why did removing item fail" --backend litellm
python3 -m raglogger ask "summarise the errors" --backend litellm,ollama           # Claude, else local Ollama
python3 -m raglogger chat --backend litellm,ollama,extractive
```

* `--backend a,b,c` is a **fallback chain**: tried in order, the first success wins, the answering backend is shown.
  If all fail, the offline extractive answer is returned with the reasons.
* `--model` / `--base-url` override only the **first** backend in the chain; others use their env settings.
* **Temperature:** some gateway models (e.g. Claude Opus with extended thinking) reject any temperature except
  their default. The error looks like *"does not support temperature=0.1. Only temperature=1 is supported"*.
  * Set `LITELLM_TEMPERATURE=omit` in `.env` (or pass `--temperature omit`) so the parameter is not sent at all.
  * Without that, the app retries the request once without temperature when a backend rejects it, and logs a
    warning. Both attempts appear in the transcript log.
  * Per backend: `LITELLM_TEMPERATURE`, `OLLAMA_TEMPERATURE` and `OPENAI_TEMPERATURE` win over
    `RAGLOGGER_TEMPERATURE` (default `0.1`). Values: `0`–`2` or `omit`. `--temperature` overrides all of them.
  * Do **not** set `litellm.drop_params` on the gateway just for this tool: that silently drops parameters for
    every client.
* Use the model **alias exactly as configured in LiteLLM** (`raglogger models` shows it) – the alias decides
  which Claude version is used, not this tool.
* Other variables: `RAGLOGGER_BACKEND` (default chain), `RAGLOGGER_MAX_TOKENS` (default 2048),
  `RAGLOGGER_TEMPERATURE`, `RAGLOGGER_TIMEOUT`, `RAGLOGGER_MAX_CONTEXT_CHARS`, `RAGLOGGER_INDEX`,
  `RAGLOGGER_INTERACTION_LOG` (path or `off`).
* API keys are read from the environment at call time only; they are never stored in settings, the index, the
  audit log or error messages (keys echoed back by a gateway are masked).

Every LLM request/response is appended (redacted, fsync'ed) to `.raglogger/logs/llm-interactions.jsonl`
with timestamp, session id, prompts, model parameters, duration and token counts.

## Security & privacy

* Passwords, tokens, bearer/basic auth, API keys, JWTs, private keys and e-mail addresses are masked as
  `***REDACTED***` **before** indexing, so they never reach the index, the LLM or the audit log.
* Log files, indexes and JSONL logs are git-ignored – production logs may contain customer data
  (cart ids, account numbers, correlation ids remain in the index).
* Prefer the `ollama` backend for production logs; `litellm`/`openai` send retrieved excerpts to the gateway /
  provider (check your gateway's data-retention policy).

## Tests

```bash
python3 -m unittest discover -s tests -v
```
