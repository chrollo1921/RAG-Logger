# 5. Retrieval: how a question becomes a set of chunks

## 5.1 Call path

```
service.QueryService.search(question, top_k, filters)
 └─ langchain engine: langchain_rag.LangChainRetrieverAdapter.search()
       └─ vectorstore.HybridLogRetriever.invoke(query, config={callbacks})   ← LangChain BaseRetriever
            └─ search_hits(query)
                 ├─ index.Bm25Index.search(query, top_k=fetch_k, filters)          keyword ranking
                 ├─ Chroma.similarity_search_with_score(query, k=fetch_k, filter)  vector ranking
                 └─ Reciprocal Rank Fusion → top k
       (LlmInteractionCallback.on_retriever_end logs the documents found)
 └─ native engine:    index.Bm25Index.search(query, top_k, filters)
```

Defaults: `k = -k/--top-k` (6), `fetch_k = max(30, 3 × k)` candidates from each method.

## 5.2 BM25 (`index.py`)

* **Tokenizer (`tokenize()`):** lower-case words and numbers, with CamelCase/dotted identifiers *also* split into
  parts (`LynxItemRemoveLoggingInterceptor` → `lynxitemremovelogginginterceptor`, `lynx`, `item`, `remove`,
  `logging`, `interceptor`). Stop words are dropped. Synonyms are mapped: `errors→error`, `warning(s)→warn`,
  `failed/failure→fail`, `exceptions→exception`, `timeouts/timed→timeout`, …
* **Score:** Okapi BM25, `k1 = 1.4`, `b = 0.6`, over `title + text`.
* **Adjustments:** `+5` for the overview chunk when the question contains *overview / summary / summarize /
  overall / most / top / frequent*. Then `× KIND_BOOST` (incident 1.15, template 1.0, window 0.9, overview 0.8)
  and `× (1 + 0.05·ln(1+occurrences))`, a mild boost for frequent problems.

## 5.3 Vector search (`vectorstore.py`)

* The question is embedded with the same model as the chunks, and Chroma returns the `fetch_k` nearest by cosine
  distance.
* `chroma_where()` pushes the `--kind` filter into Chroma. All other filters are re-checked in Python with
  `SearchFilters.matches()`, so BM25 and vector hits use exactly the same filter rules.

## 5.4 Fusion (RRF)

```
score(chunk) = Σ over methods that returned it:  1 / (60 + rank_in_that_method)
```

The constant 60 is `RRF_K`. The displayed score is `× 1000`. Each hit records which method(s) found it:
`bm25`, `vector` or `bm25+vector`. This is shown in `search` output, answer sources and the transcript log.

Example from the real log, *"why could the basket not be priced"*:

| Rank | Found by | Chunk |
|---|---|---|
| 1 | bm25 | WARN DefaultDeliveryFromAddressStrategy: Could not determine ship-from address … |
| 2 | **vector** | ERROR LynxFindPricingWithCurrentPriceFactoryStrategyImpl: Couldn't get base price … (CalculationException) |
| 3 | bm25 | WARN DefaultFindPaymentCostStrategy: Could not find paymentCost … |

Keyword search alone never found the pricing error. The vector side matched "priced" to "base price".

## 5.5 Filters (`models.SearchFilters`)

| Flag | Rule |
|---|---|
| `--level ERROR` | chunk's `levels` contains the level (upper-cased) |
| `--logger Lynx` | case-insensitive substring of any of the chunk's loggers |
| `--kind incident` | exact kind (`incident` / `template` / `window` / `overview`) |
| `--since 2026-08-29T00:15` | chunk `end_time` ≥ since (spaces → `T`, `.` → `,`) |
| `--until 2026-08-29T00:20` | chunk `start_time` ≤ until (inclusive) |
| `--source node2` | substring of the source **label or original path**. `cli._resolve_sources()` turns it into exact labels and errors out, listing the indexed sources, if nothing matches. |
