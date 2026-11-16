# RAG-Logger documentation

RAG-Logger answers questions about **SAP Commerce (Hybris) console logs**. It turns raw logs into searchable
chunks, finds the chunks relevant to your question with keyword (BM25) and vector search, and asks an LLM
(Claude via LiteLLM, Ollama, or any OpenAI-compatible endpoint) to answer **only from those chunks**. Every
retrieval, prompt and response is logged.

## Read this first

| If you want to know… | Read |
|---|---|
| How the pieces fit together, end to end | [01-architecture.md](01-architecture.md) |
| **What every file does** | [02-file-reference.md](02-file-reference.md) |
| **Where chunking happens and where chunks are stored** | [03-chunking-and-storage.md](03-chunking-and-storage.md) |
| **Where embedding happens and where vectors are stored** | [04-embeddings-and-vector-store.md](04-embeddings-and-vector-store.md) |
| How a question becomes a set of chunks (hybrid search) | [05-retrieval.md](05-retrieval.md) |
| **Where the LLM request is sent, with which parameters** | [06-llm-requests.md](06-llm-requests.md) |
| What is logged about each LLM call, and where | [07-logging-and-observability.md](07-logging-and-observability.md) |
| Every setting, `.env` variable and CLI flag | [08-configuration.md](08-configuration.md) |
| Commands, recipes, troubleshooting | [09-operations.md](09-operations.md) |
| Security, privacy and performance notes | [10-security-and-performance.md](10-security-and-performance.md) |
| Tests and helper scripts | [11-testing-and-development.md](11-testing-and-development.md) |

## One-minute summary

```
INGEST   console logs ─► parser.py ─► redaction.py ─► chunker.py ─┬─► index.py       .raglogger/index.json.gz
         (files/dirs)    (events)     (mask secrets)  (chunks)    └─► vectorstore.py .raglogger/index-chroma/
                                                                       (embeddings)

ASK      question ─► vectorstore.HybridLogRetriever (BM25 + vectors, fused) ─► top-k chunks
                  ─► generators.build_context() ─► langchain_rag.LangChainChainGenerator
                  ─► ChatOpenAI → LiteLLM /chat/completions   (or ChatOllama → /api/chat)
                  ─► answer + sources       every step logged to .raglogger/logs/
```

| Question | Short answer |
|---|---|
| Where is chunking done? | `raglogger/chunker.py` → `LogChunker.build_chunks()`, called from `service.IngestService._ingest_file()` |
| Where are chunks stored? | `.raglogger/index.json.gz` (gzip JSON, written by `index.Bm25Index.save()`) |
| Where is embedding done? | `raglogger/vectorstore.py` → `create_embeddings()` (model) and `VectorStoreService._add()` (embedding calls) |
| Where are vectors stored? | Chroma at `.raglogger/index-chroma/` (collection `console_log_chunks`, cosine) |
| Where is the LLM called? | LangChain engine: `langchain_rag.LangChainChainGenerator._run_chain()` → `chain.invoke()`. Native engine: `generators._HttpLlmGenerator._post()` |
| Which parameters are sent? | `model`, `messages` (system + user), `max_tokens` (default 2048), `temperature` (default 0.1, or omitted). Details in [06](06-llm-requests.md) |
| Where is it logged? | `.raglogger/logs/llm-transcript.log` (readable) and `.raglogger/logs/llm-interactions.jsonl` (JSON) |

## Glossary

| Term | Meaning |
|---|---|
| **Event / `LogEntry`** | One logical log statement: the header line plus its continuation lines (stack trace). |
| **Chunk** | A retrievable document built from many events (a message template, an incident, a time window, or an overview). |
| **Source** | One ingested console log file, identified by a short unique *label* such as `node1/console.log`. |
| **BM25** | Classic keyword ranking. Good for exact identifiers like logger names, exception classes and ids. |
| **Embedding / vector** | A list of numbers (384 for the default model) representing a chunk's meaning, used for similarity search. |
| **Hybrid retrieval** | BM25 and vector results merged with Reciprocal Rank Fusion (RRF). |
| **Engine** | `langchain` (default when installed: hybrid search plus LangChain LLM clients) or `native` (stdlib only: BM25 plus `urllib`). |
| **Backend** | Where answers come from: `extractive` (no LLM), `litellm`, `ollama`, `openai`. Can be chained: `litellm,ollama`. |
