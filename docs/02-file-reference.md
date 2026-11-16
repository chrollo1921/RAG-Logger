# 2. File reference: what each file does

Line numbers are approximate. Search for the function name if code has moved.

## 2.1 Project layout

```
RAG-Logger/
├── raglogger/                  application package (all runtime code)
├── tests/                      unittest suites + fixtures/sample-console.log
├── scripts/                    smoke tests, demos, developer helpers
├── docs/                       this documentation
├── .env                        your local settings and API key (git-ignored, chmod 600)
├── .env.example                committed template listing every variable
├── requirements.txt            LangChain + Chroma + LLM clients
├── requirements-local-embeddings.txt   CPU PyTorch + sentence-transformers (local embeddings)
├── .gitignore / .gitattributes keep secrets, indexes and logs out of git; force LF line endings
├── README.md                   quick start
├── venv/ or .venv/             Python virtual environment (not committed)
└── .raglogger/                 runtime data (git-ignored, mode 700; move with RAGLOGGER_HOME)
    ├── index.json.gz           chunks + BM25 data + source registry
    ├── index-chroma/           Chroma vector store
    └── logs/
        ├── llm-transcript.log      readable LLM transcript (rotating 10 MB × 5)
        └── llm-interactions.jsonl  structured JSON Lines audit log
```

All paths in these docs written as `.raglogger/…` are relative to the project root
(`/home/karthik/projects/RAG-Logger`). The location is decided in `raglogger/paths.py`.

## 2.2 `raglogger/` package

### Entry point and configuration

| File | What it does | Key functions / classes |
|---|---|---|
| `__init__.py` | Package marker, `__version__`. | – |
| `__main__.py` | Makes `python -m raglogger` work. Calls `cli.main()`. | – |
| `cli.py` | **Controller.** Defines every command (`ingest`, `ask`, `search`, `chat`, `sources`, `stats`, `models`) and flag. Loads `.env`, builds `Settings`, applies CLI overrides, picks the engine, wires the services, prints output. Maps errors to exit codes (2 = config/input error, 3 = LLM error). | `main()`, `_dispatch()`, `_ingest()`, `_build_query_components()`, `build_langchain_generator()`, `build_generator()` (native), `langchain_callbacks()`, `_resolve_sources()`, `vector_dir_for()`, `resolve_engine()` |
| `envfile.py` | Stdlib `.env` loader. Supports quotes, `${VAR}` expansion, `export`, comments, BOM and CRLF. Shell variables win over the file. Blank values are ignored. Warns if `.env` is readable by others. Never logs values. | `load_env_file()`, `parse_env_text()`, `resolve_env_file()` |
| `paths.py` | **Where runtime data lives.** `<project>/.raglogger` by default, or `$RAGLOGGER_HOME`. Resolves relative paths against the project root, so the current directory doesn't matter. Read at call time, after `.env` is loaded. | `data_dir()`, `resolve_path()`, `default_index_path()`, `default_transcript_path()`, `default_interaction_log_path()`, `PROJECT_ROOT` |
| `config.py` | The `Settings` DTO: every tunable value, read from environment variables. Per-backend `BackendSettings` (model, base URL, API-key *variable name*, temperature). API keys are **not** stored here. | `Settings.from_env()`, `Settings.resolve()`, `Settings.temperature_for()`, `Settings.backend_chain`, `Settings.embedding_settings`, `parse_temperature()`, `default_backends()` |

### Ingest pipeline

| File | What it does | Key functions / classes |
|---|---|---|
| `sources.py` | Turns `ingest` arguments into files: expands directories recursively (pattern `*.log*`), de-duplicates, gives each file the shortest unique **label**, and computes a size + mtime **fingerprint** for skipping unchanged files. | `discover()`, `assign_labels()`, `SourceFile.key` (8-char hash used to namespace chunk ids), `SourceFile.fingerprint()` |
| `parser.py` | Streams a console log (plain or `.gz`) line by line. Strips ANSI colour codes, unwraps the Tanuki wrapper prefix (`INFO \| jvm 1 \| main \| …`), parses the Log4j header (timestamp, level, thread, logger, correlation/session/trace ids), and attaches continuation lines (stack traces, capped at 60) to the right event. | `ConsoleLogParser.parse_file()`, `parse_lines()`, regexes `WRAPPER_RE`, `HEADER_RE`, `LOGGER_RE`, `ANSI_RE` |
| `redaction.py` | Masks passwords, tokens, bearer/basic auth, API keys (`sk-…`, `ghp_…`, AWS), JWTs, private keys and e-mails as `***REDACTED***`. Applied to every entry **before** chunking, and again to everything written to the logs. | `Redactor.redact()`, `redact_entry()`, `redact_value()` |
| `chunker.py` | **Chunking.** Aggregates events in one streaming pass into four chunk kinds: *template*, *incident*, *window* and *overview*. Normalises variable tokens (numbers, ids, UUIDs, IPs, quoted values) so repeated messages collapse. See [doc 03](03-chunking-and-storage.md). | `LogChunker.build_chunks()`, `normalize_message()`, `_template_chunk()`, `_incident_chunk()`, `_window_chunk()`, `_overview_chunk()` |
| `service.py` | **Application services.** `IngestService` runs discover → parse → redact → chunk → store, with rebuild/append/remove modes and per-file stats. `QueryService` runs retrieve → generate, with the extractive fallback. | `IngestService.ingest_many()`, `_process_files()`, `_ingest_file()`, `_load_existing()`, `QueryService.search()`, `QueryService.ask()` |

### Storage and retrieval

| File | What it does | Key functions / classes |
|---|---|---|
| `index.py` | **Chunk store + BM25 keyword index.** Tokenises chunks (lower-case, CamelCase split, synonyms like `errors→error`, `warnings→warn`), scores with Okapi BM25 (`k1=1.4`, `b=0.6`), applies filters and boosts, and saves/loads `index.json.gz`. | `Bm25Index.build()`, `search()`, `save()`, `load()`, `tokenize()`, `KIND_BOOST` |
| `vectorstore.py` | **Embedding + vector DB + hybrid retriever.** Creates the embedding model (local HuggingFace or LiteLLM), writes vectors to Chroma (full rebuild or per-source update), checks the embedding model hasn't changed, and fuses vector and BM25 rankings with RRF in a LangChain `BaseRetriever`. See [doc 04](04-embeddings-and-vector-store.md) and [doc 05](05-retrieval.md). | `create_embeddings()`, `open_chroma()`, `VectorStoreService.build()`, `upsert_sources()`, `_add()`, `open()`, `HybridLogRetriever.search_hits()`, `chunk_metadata()` |

### LLM integration

| File | What it does | Key functions / classes |
|---|---|---|
| `langchain_rag.py` | **LangChain engine.** Builds the chat model per backend (`ChatOpenAI` for LiteLLM/OpenAI, `ChatOllama`), defines the prompt template, runs `PROMPT \| model \| StrOutputParser` for each backend in fallback order, retries once without temperature when a model rejects it, and adapts the hybrid retriever for `QueryService`. See [doc 06](06-llm-requests.md). | `PROMPT`, `create_chat_model()`, `LangChainChainGenerator.answer()` / `_invoke()` / `_run_chain()`, `LangChainRetrieverAdapter.search()`, `is_temperature_rejection()`, `without_temperature()`, `scrub_secrets()` |
| `generators.py` | **Shared prompt pieces + native engine.** `SYSTEM_PROMPT` and `build_context()` are used by *both* engines. Also holds the offline `ExtractiveGenerator`, and stdlib `urllib` clients for Ollama, OpenAI-compatible endpoints and LiteLLM, plus `FallbackChainGenerator` and `list_models()`. | `SYSTEM_PROMPT`, `build_context()`, `ExtractiveGenerator`, `_HttpLlmGenerator._logged_call()` / `_post()`, `OllamaGenerator._call()`, `OpenAiCompatibleGenerator._call()`, `LiteLlmGenerator`, `FallbackChainGenerator`, `is_temperature_rejection_text()` |

### Observability and DTOs

| File | What it does | Key functions / classes |
|---|---|---|
| `llm_logging.py` | Sets up the rotating **transcript** logger. Defines the LangChain callback handler that logs retrieved documents, the full prompt messages, parameters, the response, token usage, latency and errors. | `configure_transcript_logger()`, `LlmInteractionCallback.on_retriever_end()`, `on_chat_model_start()`, `on_llm_end()`, `on_llm_error()` |
| `interaction_log.py` | Append-only **JSONL audit log** (`llm-interactions.jsonl`). Redacted, flushed and fsync'ed per record, thread-safe, never raises. The native engine also writes the transcript through it. | `InteractionLogger.log_request()`, `log_response()`, `write_record()` |
| `models.py` | DTO dataclasses passed between layers. `SearchFilters.matches()` holds the filter logic shared by BM25 and vector search. | `LogEntry`, `Chunk`, `SearchFilters`, `SearchHit`, `Answer`, `IngestStats` |

## 2.3 `tests/`

| File | Covers |
|---|---|
| `test_raglogger.py` | Parser, redaction, chunker, BM25 index, services, native LLM clients, LiteLLM auth and key scrubbing, fallback chain, settings, CLI |
| `test_langchain_rag.py` | Chroma vector store, hybrid retriever, LangChain chain + fallback, callback logging, model factories (skipped without LangChain) |
| `test_multifile.py` | Discovery, labels, rebuild/append/replace/skip/remove, `--source`, `.gz`, Chroma kept in sync with the index |
| `test_envfile.py` | `.env` syntax, precedence, permissions, CLI integration, `.env.example` completeness |
| `test_temperature.py` | `omit`, per-backend temperature, auto-retry against a gateway that rejects temperature (both engines) |
| `test_paths.py` | Project-local data dir, `RAGLOGGER_HOME`, relative-path resolution, cwd independence, test isolation |
| `_isolation.py` | Imported first by every test module: points `RAGLOGGER_HOME` at a temp dir so tests never touch `.raglogger/` |
| `fixtures/sample-console.log` | 13-line console log with an exception, an ERROR repeated twice, a WARN containing a fake password and e-mail, and a wrapper line |

## 2.4 `scripts/`

| Script | Purpose |
|---|---|
| `dev_check.py` | Converts CRLF → LF (files edited from Windows) and runs the full test suite |
| `mock_litellm.py` | Local fake OpenAI/LiteLLM gateway (`/chat/completions`, `/models`). `MOCK_REJECT_TEMPERATURE=1` copies the real "only temperature=1" error |
| `smoke_langchain.sh` | End to end: hybrid retrieval → mock gateway → checks both logs and that the key isn't leaked |
| `smoke_temperature.sh` | Reproduces the temperature 400 error and shows the auto-retry and the `omit` fix |
| `smoke_backends.sh` | Native backend wiring: missing model, fallback chain, unreachable gateway |
| `demo_multifile.sh` | Multi-file ingest, `sources`, append-skip, `--source`, `--remove` on the real log |
| `compare_retrieval.sh` | Side-by-side BM25-only vs hybrid results for sample questions |
| `check_env.sh` | `.env` loading, shell override, permission warning, `--env-file off` |
| `set_litellm_temperature.py` | One-off: adds `LITELLM_TEMPERATURE=omit` to `.env` (prints variable names only) |
| `migrate_data_dir.py` | One-off: moves `~/.raglogger` → `<project>/.raglogger` and clears old home paths in `.env` (dry run unless `--apply`; refuses to overwrite; keeps a `.env` backup) |
| `check_paths.sh` | Shows the resolved data paths, checks they don't depend on the current directory, and that `~/.raglogger` is unused |
| `check_docs.py` | Verifies these docs: every referenced function/class exists, links and paths resolve, no secrets |
| `_doc_samples.py`, `_doc_payloads.py` | Read-only helpers that produced the sample records and request bodies in these docs |

## 2.5 Dependencies

| Package | Used by | Why |
|---|---|---|
| *(stdlib only)* | everything in the native engine | Runs without `pip` |
| `langchain-core` | `langchain_rag.py`, `llm_logging.py`, `vectorstore.py` | Prompt templates, runnables, retriever base class, callbacks |
| `langchain-openai` | `create_chat_model()`, `create_embeddings()` | `ChatOpenAI` / `OpenAIEmbeddings` pointed at LiteLLM or OpenAI |
| `langchain-ollama` | `create_chat_model()` | `ChatOllama` |
| `langchain-chroma`, `chromadb` | `vectorstore.py` | Persistent vector store |
| `langchain-huggingface`, `sentence-transformers`, `torch` (CPU) | `create_embeddings()` with `local` | On-machine embeddings (`all-MiniLM-L6-v2`) |
