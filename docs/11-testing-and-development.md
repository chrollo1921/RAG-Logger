# 11. Testing and development

## 11.1 Running tests

```bash
python3 scripts/dev_check.py               # fix CRLF line endings, then run all tests with venv/.venv Python
venv/bin/python -m unittest discover -s tests -v
python3 -m unittest discover -s tests      # system Python: LangChain/Chroma tests are skipped
venv/bin/python scripts/check_docs.py      # docs/ references still match the code (run after refactoring)
```

All tests run offline. LLM gateways are faked with local HTTP servers or `unittest.mock`, and embeddings with a
deterministic bag-of-words `KeywordHashEmbedding` (`tests/test_langchain_rag.py`).

## 11.2 Smoke tests and demos

| Command | Shows |
|---|---|
| `bash scripts/smoke_langchain.sh` | Hybrid retrieval → mock LiteLLM → transcript + JSONL written, key not leaked |
| `bash scripts/smoke_temperature.sh` | Temperature rejection → auto-retry, and the `omit` fix |
| `bash scripts/smoke_backends.sh` | Native-engine fallback chain and error messages |
| `bash scripts/demo_multifile.sh [LOG]` | Multi-file ingest, skip on append, `--source`, `--remove` |
| `bash scripts/compare_retrieval.sh` | BM25-only vs hybrid ranking |
| `bash scripts/check_env.sh` | `.env` loading and precedence |

## 11.3 Extending

| Goal | Where to change |
|---|---|
| New log format | `parser.py`: add or adjust `WRAPPER_RE` / `HEADER_RE` / `LOGGER_RE`, and add a fixture plus a test |
| New chunk kind or different grouping | `chunker.py`: accumulate in `build_chunks()`, add a `_<kind>_chunk()` builder, add it to `KIND_BOOST` in `index.py` |
| New LLM backend | `config.default_backends()` + `SUPPORTED_BACKENDS`, `langchain_rag.create_chat_model()`, and optionally a native class in `generators.py` plus `cli.GENERATOR_CLASSES` |
| New embedding provider | `vectorstore.create_embeddings()` + `EMBEDDING_PROVIDERS` / `config.EMBEDDING_CHOICES` |
| Prompt wording | `generators.SYSTEM_PROMPT` and `langchain_rag.PROMPT` (keep the two in sync) |
| New redaction rule | `redaction._PATTERNS`, with a test in `RedactionTest` |
| New setting | `config.Settings.from_env()` **and** `.env.example` (`EnvExampleTest` fails otherwise) |

Conventions: constructor injection, DTOs from `models.py` between layers, `LOG = logging.getLogger(__name__)`
per module, errors as `ValueError` / `FileNotFoundError` (exit 2) or `GenerationError` (exit 3), LF line
endings, no new dependencies without approval.
