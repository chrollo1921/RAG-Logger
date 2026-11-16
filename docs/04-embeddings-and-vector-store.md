# 4. Embeddings: where they are computed and where vectors are stored

Embeddings are used only by the **LangChain engine** with `--embeddings local` (default) or `--embeddings litellm`.
With `--embeddings none` or `--engine native`, nothing is embedded and retrieval is BM25 only.

## 4.1 Where embedding happens

| Step | File → function | What happens |
|---|---|---|
| 1. Decide | `cli.py` → `_uses_vectors(settings, engine)` | `True` when engine = `langchain` and embeddings ≠ `none` |
| 2. Settings | `config.py` → `Settings.embedding_settings` | Builds `EmbeddingSettings(provider, model, base_url, api_key_env, timeout_seconds)` |
| 3. **Create the model** | **`vectorstore.py` → `create_embeddings(settings)`** | Returns a LangChain `Embeddings` object (see 4.2) |
| 4. Wire | `cli.py` → `_vector_service()` | `VectorStoreService(vector_dir_for(settings), embeddings, embedding_settings)` |
| 5. Trigger | `service.py` → `IngestService.ingest_many()` | Rebuild → `VectorStoreService.build(all_chunks)`. Append/remove → `upsert_sources(new_chunks, replaced, removed)` |
| 6. **Embed + write** | **`vectorstore.py` → `VectorStoreService._add(store, chunks)`** | Each chunk becomes a LangChain `Document`. `store.add_documents()` runs in batches of 256, and Chroma calls `embeddings.embed_documents()` for each batch. |
| 7. Query time | `vectorstore.py` → `HybridLogRetriever.search_hits()` | `similarity_search_with_score(query, …)` calls `embeddings.embed_query(question)` |

What gets embedded for each chunk (`_add()`):

```python
Document(page_content=f"{chunk.title}\n{chunk.text}", metadata=chunk_metadata(chunk))   # id = chunk.chunk_id
```

The text is **already redacted** (redaction happens before chunking), so no secrets are embedded or sent to a
remote embedding service.

## 4.2 Embedding providers (`create_embeddings()`)

| `--embeddings` | Class | Model | Notes |
|---|---|---|---|
| `local` (default) | `langchain_huggingface.HuggingFaceEmbeddings` | `sentence-transformers/all-MiniLM-L6-v2` (override with `--embedding-model` / `RAGLOGGER_EMBEDDING_MODEL`) | CPU, `normalize_embeddings=True`, `batch_size=64`. **384** dimensions. Reads at most **256 tokens** per chunk (the vector ignores the rest; BM25 still sees all of it). Downloaded once into `~/.cache/huggingface/`. Needs `requirements-local-embeddings.txt`. |
| `litellm` | `langchain_openai.OpenAIEmbeddings` | `LITELLM_EMBEDDING_MODEL` (an alias on your gateway; required) | `POST {LITELLM_BASE_URL}/embeddings` with `LITELLM_API_KEY`. `check_embedding_ctx_length=False` (sends plain strings), `chunk_size=256`, `max_retries=2`, `request_timeout=RAGLOGGER_TIMEOUT`. **Every chunk is sent to the gateway at ingest.** |
| `none` | – | – | Vector store is neither built nor used |

Configuration problems raise `EmbeddingConfigError` (CLI exit code 2): unknown provider, local packages missing,
`LITELLM_API_KEY` missing, or no `LITELLM_EMBEDDING_MODEL`.

## 4.3 Where vectors are stored

| | |
|---|---|
| **Engine** | Chroma (`chromadb` 0.6 via `langchain_chroma.Chroma`), opened in `vectorstore.open_chroma()` |
| **Directory** | `<index dir>/<index name>-chroma/` = **`.raglogger/index-chroma/`** (`cli.vector_dir_for()`; override with `RAGLOGGER_VECTOR_DIR`) |
| **Collection** | `console_log_chunks`, `hnsw:space = cosine` |
| **Client** | `chromadb.PersistentClient(…, anonymized_telemetry=False, allow_reset=False)`. Telemetry is off. |
| **Size** | about 23 MB for 1,843 chunks with the local model |

```
.raglogger/index-chroma/
├── chroma.sqlite3                 documents, metadata, ids (SQLite)
├── <uuid>/                        HNSW vector index segment (data_level0.bin, link_lists.bin, …)
└── raglogger-manifest.json        {"provider": "local", "model": "sentence-transformers/all-MiniLM-L6-v2", "chunks": 1843}
```

One record per chunk (`chunk_metadata()`). Chroma metadata must be scalar, so `levels` is a delimited string:

| Field | Example |
|---|---|
| **id** | `67d5831c:tpl-c76c8d63ae0f061d` (= `chunk.chunk_id`) |
| **document** | `"<title>\n<text>"` |
| **embedding** | 384 floats |
| `chunk_id`, `kind`, `title` (max 500 chars) | `"template"`, `"INFO LynxRegexPatternFileListFilter: …"` |
| `start_time`, `end_time`, `line_start`, `line_end`, `occurrences` | `"2026-08-29T00:13:22,631"`, `1`, `1`, `1` |
| `levels` / `source` | `",INFO,"` / `"sample-console.log"` |

## 4.4 Keeping vectors and chunks in sync

| Operation | Vector-store call | Effect |
|---|---|---|
| `ingest <files>` (rebuild) | `build(all_chunks)` | `reset_collection()`, then embed everything |
| `ingest --append <file>` (new or changed) | `upsert_sources(new_chunks, replace_sources=[label])` | `store.delete(where={"source": label})` for the replaced file, then embed only its new chunks |
| `ingest --append` (unchanged file) | – | Skipped (size + mtime match); nothing is re-embedded |
| `ingest --remove <label>` | `upsert_sources([], [], remove_sources=[label])` | Deletes that file's vectors |

`test_multifile.ChromaIncrementalTest` checks that the set of Chroma ids always equals the set of chunk ids in
`index.json.gz`.

**Model change protection:** `raglogger-manifest.json` records the provider and model.
`VectorStoreService.open()` raises `EmbeddingConfigError` if the configured model differs ("Vector store was built
with embeddings local:… but litellm:… is configured"), because vectors from different models can't be compared.
`upsert_sources()` runs the same check before adding new vectors.

## 4.5 Performance

| | Local (`all-MiniLM-L6-v2`, CPU) | LiteLLM |
|---|---|---|
| Ingest 1,843 chunks | about 2 minutes (12 cores) | about 8 requests of 256 texts; depends on gateway latency |
| Per question | 5–25 s to load the model per process (`chat` loads it once), then about 10 ms per query | one `/embeddings` call |
| First run | downloads about 90 MB | – |

Use `--append` for new files so only those files are embedded.
