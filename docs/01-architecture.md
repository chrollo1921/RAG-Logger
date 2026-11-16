# 1. Architecture

## 1.1 Layers

The code follows a layered design (similar to Controller → Facade → Service → DAO). Every class gets its
collaborators through its constructor, so each layer can be tested in isolation.

| Layer | Module(s) | Responsibility |
|---|---|---|
| **Entry point / controller** | `__main__.py`, `cli.py` | Parse arguments, load `.env`, build objects, print results |
| **Configuration** | `envfile.py`, `config.py` | `.env` loading, the `Settings` DTO, per-backend settings |
| **Services** | `service.py` | `IngestService` (ingest pipeline) and `QueryService` (retrieve → generate) |
| **Domain processing** | `parser.py`, `redaction.py`, `chunker.py`, `sources.py` | Log parsing, secret masking, chunk building, file discovery and labels |
| **Storage / "DAO"** | `index.py`, `vectorstore.py` | BM25 index file, Chroma vector store |
| **LLM integration** | `langchain_rag.py`, `generators.py` | Prompt building, chat model clients, fallback chain, retries |
| **Observability** | `llm_logging.py`, `interaction_log.py` | Transcript log and JSONL audit log |
| **DTOs** | `models.py` | `LogEntry`, `Chunk`, `SearchFilters`, `SearchHit`, `Answer`, `IngestStats` |

## 1.2 Ingest flow (`raglogger ingest …`)

```
cli.main()                                   load .env  → Settings.from_env() → apply CLI flags
 └─ cli._ingest()
     ├─ cli._vector_service()                 only when engine=langchain and embeddings != none
     │    └─ vectorstore.create_embeddings()   loads the HF model or creates the LiteLLM embeddings client
     └─ service.IngestService.ingest_many(paths, index_path, append, force, remove, label, pattern)
         ├─ _load_existing()                  (append/remove only) read the current index + its sources
         ├─ sources.discover()                files and directories → list of files (recursive, *.log*)
         ├─ sources.assign_labels()           shortest unique label per file, e.g. node1/console.log
         ├─ _process_files()                  for every file (unchanged files are skipped in append mode):
         │    └─ _ingest_file(source)
         │         ├─ parser.ConsoleLogParser.parse_file()   streams LogEntry objects (.gz supported)
         │         ├─ redaction.Redactor.redact_entry()      masks secrets in every entry
         │         └─ chunker.LogChunker.build_chunks()      ◄── CHUNKING (see doc 03)
         ├─ index.Bm25Index.build() + .save()               ◄── CHUNK STORAGE → .raglogger/index.json.gz
         └─ vectorstore.VectorStoreService.build()          (rebuild)      ◄── EMBEDDING + VECTOR STORAGE
            or .upsert_sources()                            (append/remove)     → .raglogger/index-chroma/
```

## 1.3 Question flow (`raglogger ask …` / `chat`)

```
cli.main() → cli._dispatch()
 ├─ index.Bm25Index.load()                                  read chunks back from index.json.gz
 ├─ SearchFilters(level, logger, kind, since, until, source)
 ├─ cli._build_query_components()
 │    ├─ engine=langchain:
 │    │    retriever = langchain_rag.LangChainRetrieverAdapter(Chroma, bm25, callbacks)
 │    │    generator = langchain_rag.LangChainChainGenerator([(backend, model, ChatModel), …])
 │    └─ engine=native:
 │         retriever = Bm25Index
 │         generator = generators.LiteLlmGenerator / OllamaGenerator / … (or FallbackChainGenerator)
 └─ service.QueryService.ask(question, top_k, filters)
      ├─ retriever.search()            ◄── RETRIEVAL (doc 05): vectorstore.HybridLogRetriever.search_hits()
      ├─ generator.answer()
      │    ├─ generators.build_context()          numbered context block, at most 12,000 characters
      │    └─ _run_chain(): PROMPT | model | StrOutputParser → .invoke()   ◄── LLM REQUEST (doc 06)
      └─ on GenerationError → ExtractiveGenerator   (offline excerpts + the reason the LLM failed)
```

## 1.4 Two engines

| | `langchain` engine (default if installed) | `native` engine (`--engine native`) |
|---|---|---|
| Requirements | `venv` with `requirements.txt` (+ optional local embeddings) | Python 3.10 standard library only |
| Retrieval | Hybrid: Chroma vectors + BM25 (RRF) | BM25 only |
| LLM clients | `langchain_openai.ChatOpenAI` (LiteLLM, OpenAI), `langchain_ollama.ChatOllama` | `generators.py` classes over `urllib` |
| Logging | `llm_logging.LlmInteractionCallback` (LangChain callbacks) | `interaction_log.InteractionLogger` |
| Selection | `cli.resolve_engine()` | same |

Both engines read the same `index.json.gz`, use the same system prompt and context format, handle the
temperature retry the same way, and write to the same two log files.

## 1.5 Design decisions

| Decision | Why |
|---|---|
| Chunk by **message template**, not by raw lines or fixed-size windows | Console logs are extremely repetitive. 70,280 lines collapse to about 1,800 chunks with occurrence counts, so the LLM sees *patterns* instead of thousands of duplicates. |
| Separate **incident** chunks for WARN/ERROR/FATAL and exceptions | Problems are what people ask about. Each distinct error signature gets its own chunk with exception and stack preview. |
| **Redact before chunking** | Secrets never reach the index, the vector store, the LLM or the logs. |
| **Hybrid** BM25 + vectors | BM25 nails exact identifiers (`LynxItemRemoveLoggingInterceptor`, cart ids). Vectors catch paraphrases ("why couldn't the basket be priced"). |
| BM25 index is also the **chunk store** | One gzip JSON file is the source of truth for chunk text and metadata. Chroma only adds vectors on top. |
| **Fallback chain** of backends and the **extractive** fallback | A failing gateway never leaves you without an answer: you get the evidence plus the reason. |
| Stdlib-only **native** engine | Works on a machine where `pip` isn't available. |
