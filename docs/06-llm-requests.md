# 6. LLM requests: where they are sent, what is sent, with which parameters

## 6.1 Where the request is sent (call path)

### LangChain engine (default)

```
cli._dispatch() → cli._build_query_components()
   └─ cli.build_langchain_generator(settings, redactor, callbacks)
        └─ for each backend in settings.backend_chain (e.g. litellm,ollama):
             langchain_rag.create_chat_model(backend, settings)        ◄── client + parameters are fixed here
service.QueryService.ask()
   └─ langchain_rag.LangChainChainGenerator.answer(question, hits)
        ├─ generators.build_context(hits, max_context_chars)          ◄── context block
        └─ for each backend (fallback order):
             _invoke() → _run_chain()
                 chain = PROMPT | model | StrOutputParser()
                 chain.invoke({"context": …, "question": …}, config={callbacks, metadata, tags})
                                                                       ◄── THE HTTP REQUEST IS SENT HERE
```

| Backend | Client class (in `create_chat_model()`) | HTTP endpoint |
|---|---|---|
| `litellm` | `langchain_openai.ChatOpenAI` | `POST {LITELLM_BASE_URL}/chat/completions` |
| `openai` | `langchain_openai.ChatOpenAI` | `POST {OPENAI_BASE_URL}/chat/completions` |
| `ollama` | `langchain_ollama.ChatOllama` | `POST {OLLAMA_BASE_URL}/api/chat` |

### Native engine (`--engine native`)

```
cli.build_generator() → generators.LiteLlmGenerator / OpenAiCompatibleGenerator / OllamaGenerator
                        (FallbackChainGenerator for a chain)
_HttpLlmGenerator.generate() → _logged_call() → <Backend>._call() → _post()
                                                     └─ urllib.request.urlopen(…)   ◄── HTTP REQUEST SENT HERE
```

## 6.2 What is sent: the prompt

Two messages, built the same way by both engines.

**System message**: `generators.SYSTEM_PROMPT`:

> You are a senior SAP Commerce (Hybris) support engineer analysing a server console log. Answer ONLY from the
> provided context excerpts. Cite excerpts as [1], [2], ... Quote exact logger names, exception classes,
> timestamps and occurrence counts. If the context is insufficient, say so explicitly and suggest what to search
> for next. Finish with probable root cause and recommended next steps when the question is about a problem.

**User message**: `langchain_rag.PROMPT` (human template). The native engine builds the same string in
`_HttpLlmGenerator.generate()`:

```
Context excerpts from the console log:

{context}

Question: {question}
```

**`{context}`** comes from `generators.build_context(hits, max_chars)`. Each retrieved chunk becomes a numbered
block:

```
[1] INCIDENT | ERROR LynxItemRemoveLoggingInterceptor: Error occurred while removing item[can be ignored] (java.lang.IllegalArgumentException: …)
source sample-console.log | time 2026-08-29T00:13:31,876 .. 2026-08-29T00:13:31,876 | lines 6-10
ERROR incident reported by LynxItemRemoveLoggingInterceptor.
Message: …
Exception: java.lang.IllegalArgumentException: Exception occurred while removing item
Stack trace preview:
  at com.amway.lynxcore.interceptors.LynxItemRemoveLoggingInterceptor.logRemovedItemInfo(LynxItemRemoveLoggingInterceptor.java:65)
  …

[2] TEMPLATE | …
```

* Blocks are added in rank order until **`RAGLOGGER_MAX_CONTEXT_CHARS`** (default **12,000** characters, about
  3k tokens) is reached. A block that doesn't fit is cut (and marked `...[truncated]`) if more than 200
  characters remain, otherwise dropped.
* The number of candidate chunks is `-k/--top-k` (default **6**).
* All chunk text was redacted at ingest. The citation numbers `[1]`, `[2]` match the "Sources" list printed with
  the answer.

## 6.3 Parameters sent

| Parameter | LiteLLM / OpenAI (`ChatOpenAI`) | Ollama (`ChatOllama`) | Default | Set by |
|---|---|---|---|---|
| model | `model` | `model` | LiteLLM: *none, required*; Ollama `llama3.1:8b`; OpenAI `gpt-4o-mini` | `LITELLM_MODEL` / `OLLAMA_MODEL` / `OPENAI_MODEL`, or `--model` (first backend only) |
| endpoint | `openai_api_base` | `base_url` | `http://localhost:4000` / `http://localhost:11434` / `https://api.openai.com/v1` | `*_BASE_URL`, or `--base-url` (first backend only) |
| messages | system + user (6.2) | same | – | – |
| max tokens | `max_tokens` (langchain-openai 0.3 sends it as `max_completion_tokens`) | `num_predict` (in `options`) | **2048** | `RAGLOGGER_MAX_TOKENS` |
| temperature | `temperature`, **left out entirely when `omit`** | `temperature` (in `options`), left out when `omit` | **0.1**; your `.env` has `LITELLM_TEMPERATURE=omit` | `--temperature` > `<BACKEND>_TEMPERATURE` > `RAGLOGGER_TEMPERATURE` |
| timeout | `request_timeout` | `client_kwargs={"timeout": …}` | **120 s** | `RAGLOGGER_TIMEOUT` |
| retries | `max_retries=1` (SDK-level, for network/5xx errors) | – | 1 | fixed |
| auth | `Authorization: Bearer <key>` | none | key from `LITELLM_API_KEY` (LiteLLM) / `RAGLOGGER_API_KEY` (OpenAI) | `.env` / environment |
| streaming | `"stream": false` | `"stream": true` internally (langchain-ollama reads the stream and returns one complete answer) | – | fixed |

The API key is read from the environment **when the client is created** (`create_chat_model()`; the native
engine reads it on every call). It is never stored in `Settings`, the index or the logs.

### Exact request bodies

Captured from the real clients with `scripts/_doc_payloads.py` (no network). Message text is shortened.

**LiteLLM via LangChain `ChatOpenAI`** (your setup, `LITELLM_TEMPERATURE=omit`):

```http
POST https://<your-gateway>/chat/completions
Content-Type: application/json
Authorization: Bearer ***

{
  "model": "claude-opus-5-5",
  "stream": false,
  "max_completion_tokens": 2048,
  "messages": [
    {"role": "system", "content": "You are a senior SAP Commerce (Hybris) support engineer …"},
    {"role": "user",   "content": "Context excerpts from the console log:\n\n[1] INCIDENT | …\n\nQuestion: why did removing item fail"}
  ]
}
```

There is no `temperature` key because it is omitted. With `LITELLM_TEMPERATURE` unset, `"temperature": 0.1` is
added.

**Ollama via LangChain `ChatOllama`** (temperature 0.1):

```http
POST http://localhost:11434/api/chat

{
  "model": "llama3.1:8b",
  "messages": [ …same two messages… ],
  "stream": true,
  "options": {"num_predict": 2048, "temperature": 0.1}
}
```

**Native engine** (`--engine native`, `generators.OpenAiCompatibleGenerator._call()`): same as the first body,
but with `"max_tokens": 2048` instead of `max_completion_tokens` and no `stream` key.

## 6.4 Response handling

| Step | Where |
|---|---|
| Answer text | `StrOutputParser` (LangChain) / `choices[0].message.content` (native) / `message.content` (Ollama) |
| Token usage | `usage_metadata` or `llm_output.token_usage` (LangChain), `usage.*_tokens` (OpenAI/LiteLLM), `prompt_eval_count` / `eval_count` (Ollama). Logged; not shown on screen. |
| Empty answer | counts as a failure → next backend |
| Shown to user | answer text, then "Sources:" with `[n] (retrieval) [source] title (lines a-b)` |

## 6.5 Failure handling

```
for backend in chain (e.g. litellm → ollama):
    try request
      └─ HTTP 400/422 complaining about temperature?  → retry ONCE with temperature omitted
    success → return (answer, backend)
    failure → log warning, remember reason, try next backend
all failed → GenerationError
QueryService.ask(): GenerationError → ExtractiveGenerator (offline excerpts) prefixed with
                    "[LLM backend unavailable: <reasons>]"
```

| Situation | Behaviour | Code |
|---|---|---|
| Model rejects temperature (e.g. *"does not support temperature=0.1. Only temperature=1 is supported"*) | One retry without temperature; warning suggests `<BACKEND>_TEMPERATURE=omit` | `langchain_rag.is_temperature_rejection()` / `_invoke()`; native: `generators.is_temperature_rejection_text()` in `_HttpLlmGenerator.generate()` |
| Gateway unreachable, 4xx/5xx, timeout | Next backend in the chain | `LangChainChainGenerator.answer()` / `FallbackChainGenerator.answer()` |
| `LITELLM_API_KEY` missing | Fails before any network call | `create_chat_model()` / `LiteLlmGenerator._call()` |
| No model configured | Error, exit code 2 | `create_chat_model()` / `cli._build_single()` |
| Every backend failed | Offline extractive answer plus the reasons | `service.QueryService.ask()` |

API keys are removed from error messages: `langchain_rag.scrub_secrets()` and
`OpenAiCompatibleGenerator._scrub()` replace the configured key and anything matching the secret patterns.

## 6.6 Choosing backends

```bash
venv/bin/python -m raglogger ask "summarize the errors" --backend litellm              # Claude via gateway
venv/bin/python -m raglogger ask "…" --backend litellm,ollama,extractive               # fallback chain
venv/bin/python -m raglogger ask "…" --backend litellm --model claude-sonnet           # different alias
venv/bin/python -m raglogger ask "…" --backend litellm --temperature omit
venv/bin/python -m raglogger models                                                    # GET {LITELLM_BASE_URL}/models
```

`RAGLOGGER_BACKEND` sets the default chain (`extractive` in `.env.example`).
