# 3. Chunking: where it happens and where chunks are stored

## 3.1 Where chunking happens

| Step | File → function | What happens |
|---|---|---|
| 1. Per-file loop | `service.py` → `IngestService._process_files()` → `_ingest_file(source)` | Called once per console log. Unchanged files are skipped in `--append` mode. |
| 2. Parse | `parser.py` → `ConsoleLogParser.parse_file()` | Streams `LogEntry` events (header + stack-trace lines). Memory stays constant regardless of file size. |
| 3. Redact | `service.py` → `_redacted_entries()` → `redaction.Redactor.redact_entry()` | Masks secrets in each event's message and continuation lines **before** chunking. |
| 4. **Chunk** | **`chunker.py` → `LogChunker.build_chunks(entries, source, source_key)`** | One pass over the events builds every chunk for the file. |
| 5. Collect | `service.py` → `ingest_many()` | Keeps the chunks of untouched sources, drops replaced/removed ones, adds the new ones. |
| 6. **Store** | **`index.py` → `Bm25Index.build()` + `Bm25Index.save(index_path)`** | Writes every chunk to `.raglogger/index.json.gz`. |
| 7. Embed (optional) | `vectorstore.py` → `VectorStoreService.build()` / `upsert_sources()` | The *same* chunk objects are embedded. See [doc 04](04-embeddings-and-vector-store.md). |

`LogChunker` is created in `cli._ingest()` as `LogChunker(window_minutes=args.window_minutes)`.

## 3.2 How `build_chunks()` works

Console logs repeat the same few messages thousands of times with different ids. Fixed-size text chunks would
fill the LLM context with duplicates. Instead, events are **aggregated**:

```
for each LogEntry:
    template = normalize_message(entry.message)       # ids/numbers → placeholders
    templates[(level, logger, template)]  += entry    # → template chunk
    if level in (WARN, ERROR, FATAL) or entry has an exception line:
        incidents[signature] += entry                 # → incident chunk
    windows[5-minute bucket of timestamp] += entry    # → window chunk
end → one overview chunk for the whole file
```

### Normalisation (`chunker.normalize_message()`)

Applied in this order (`_NORMALIZERS`, `chunker.py` around line 24):

| Pattern | Replaced with | Example |
|---|---|---|
| UUID | `<UUID>` | `0ba7af67-02cf-428e-8490-ff91c06fe8a0` |
| IPv4 (+port) | `<IP>` | `127.0.0.6` |
| Hex id of 12+ chars containing a digit and a letter | `<HEX>` | `f9919a08fd329a43e369835b5789045d` |
| Number (not preceded by a letter) | `<NUM>` | cart `735042471` → `<NUM>` |
| Single-quoted value up to 200 chars | `'<STR>'` | `paymentMode='iyzico-card'` |

Whitespace is collapsed and the template is capped at 400 characters. So
`Tax price user group wasn't found for cart 735042471` and `… cart 735099999` become **one** template.

### The four chunk kinds

| Kind | One chunk per | Text contains | Id format |
|---|---|---|---|
| `template` | distinct `(level, logger, normalised message)` | template, occurrence count, first/last time, top 3 threads, up to 3 sample messages, exception if any | `tpl-<sha1(level\|logger\|template)[:16]>` |
| `incident` | distinct **problem signature**: level + logger + template + normalised exception line + first 3 `at …` frames | message, template, count, first/last time and line, exception, stack preview (12 lines), up to 3 correlation ids, threads | `inc-<sha1(signature)[:16]>` |
| `window` | 5-minute time bucket (`--window-minutes`, 1–60) | event count, events per level, 8 most frequent messages, 8 most frequent warnings/errors | `win-<YYYY-MM-DDTHH:MM>` |
| `overview` | console log file | time range, total events, distinct templates/incidents, events per level, top 10 loggers, top 10 messages, top 10 problems | `overview` |

For every chunk, after building:

* `chunk_id` gets the per-file prefix **`<source_key>:`**. `source_key` is the first 8 hex chars of the SHA-1 of the
  source label (`sources.source_key()`), e.g. `67d5831c:inc-5b1c256a2f8e6bef`. The same error in two files
  therefore gets two distinct chunks.
* `chunk.source` = the label (e.g. `node2/console.log`).
* Except for the overview, `"\nSource log: <label>"` is appended to `chunk.text`, so both BM25 and vector search
  can match node or file names. (The overview already names the file in its first line.)

### Limits (constructor defaults)

| Parameter | Default | Effect |
|---|---|---|
| `LogChunker(window_minutes=…)` | 5 | width of window chunks (CLI `--window-minutes`) |
| `LogChunker(max_samples=…)` | 3 | sample messages and correlation ids per template/incident |
| `LogChunker(stack_lines=…)` | 12 | stack-trace lines kept in an incident chunk |
| `LogChunker(max_window_items=…)` | 8 | messages/problems listed per window |
| `ConsoleLogParser(max_continuation_lines=…)` | 60 | continuation lines kept per event (the rest are counted in `truncated_lines`) |

### Real numbers

`console-20260829.log` (18.5 MB, 70,280 lines) → 56,553 events → 1,821 templates → **1,843 chunks** in about
3.5 s.

## 3.3 Chunk record (the `Chunk` DTO, `models.py`)

A real incident chunk built from `tests/fixtures/sample-console.log`, with text shortened:

```json
{
  "chunk_id": "67d5831c:inc-5b1c256a2f8e6bef",
  "kind": "incident",
  "title": "ERROR LynxItemRemoveLoggingInterceptor: Error occurred while removing item[can be ignored] (java.lang.IllegalArgumentException: Exception occurred while removing item)",
  "text": "ERROR incident reported by LynxItemRemoveLoggingInterceptor.\nMessage: Error occurred while removing item[can be ignored]\nTemplate: …\nOccurrences: 1; first at 2026-08-29T00:13:31,876 (line 6), last at …\nException: java.lang.IllegalArgumentException: Exception occurred while removing item\nStack trace preview:\n  …\n  at com.amway.lynxcore.interceptors.LynxItemRemoveLoggingInterceptor.logRemovedItemInfo(LynxItemRemoveLoggingInterceptor.java:65)\n  …\nSample correlation ids: -SF-7025586804_457511\nThreads: hybrisHTTP13\nSource log: sample-console.log",
  "start_time": "2026-08-29T00:13:31,876",
  "end_time": "2026-08-29T00:13:31,876",
  "line_start": 6,
  "line_end": 10,
  "levels": ["ERROR"],
  "loggers": ["LynxItemRemoveLoggingInterceptor"],
  "occurrences": 1,
  "source": "sample-console.log"
}
```

`line_start`/`line_end` refer to physical lines in the original file, so you can open it at that spot.

## 3.4 Where chunks are stored

| | |
|---|---|
| **File** | `.raglogger/index.json.gz` (change with `RAGLOGGER_INDEX` or `--index`) |
| **Written by** | `index.Bm25Index.save()`, called from `IngestService.ingest_many()` |
| **Read by** | `index.Bm25Index.load()`, at the start of every `ask` / `search` / `chat` / `stats` / `sources` |
| **Format** | gzip-compressed JSON |
| **Size** | about 130–210 KB per 1,800 chunks |

```json
{
  "version": 1,
  "params":   {"k1": 1.4, "b": 0.6},
  "metadata": {
    "source": "<comma-separated paths>", "lines": 13, "entries": 9,
    "level_counts": {"INFO": 5, "ERROR": 3, "WARN": 1},
    "sources": {
      "sample-console.log": {
        "path": "/home/karthik/projects/RAG-Logger/tests/fixtures/sample-console.log",
        "fingerprint": {"size": 2907, "mtime_ns": 1794834688725389249},
        "lines": 13, "entries": 9, "chunks": 14,
        "level_counts": {"INFO": 5, "ERROR": 3, "WARN": 1},
        "ingested_at": "2026-11-16T14:47:50+00:00"
      }
    }
  },
  "chunks": [ { "…Chunk…" }, "…" ]
}
```

* `metadata.sources` is the **source registry** used by `--append` (fingerprint comparison), `--remove`,
  `--source` and the `sources` command.
* BM25 term statistics are **not** saved. `Bm25Index.load()` rebuilds them from the chunk text (under 0.1 s for
  about 2,000 chunks), which keeps the file small and the format simple.
* This file is the **single source of truth** for chunk text and metadata. The vector store holds a copy of the
  text and metadata per vector, but hits are always resolved back to the `Chunk` objects from this file.
* An index written before multi-file support has no `metadata.sources`, and `--append` refuses it with a clear
  message. Run one plain `ingest` to rebuild it. Your current `.raglogger/index.json.gz` is in this older
  format.

## 3.5 Inspecting the chunks yourself

```bash
venv/bin/python -m raglogger stats                       # totals, levels, sources, vector-store manifest
venv/bin/python -m raglogger sources                     # one row per console log
venv/bin/python -m raglogger search "removing item" --json -k 3     # full chunk records for the top hits
python3 -c "import gzip,json,os; d=json.load(gzip.open('.raglogger/index.json.gz','rt')); print(len(d['chunks']))"
```
