#!/usr/bin/env python3
"""Captures the exact JSON bodies LangChain's ChatOpenAI and ChatOllama send (no network; read-only helper)."""
import json
import pathlib
import sys
from unittest import mock

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
from raglogger.config import Settings, default_backends  # noqa: E402
from raglogger.langchain_rag import PROMPT, create_chat_model  # noqa: E402

env = {"LITELLM_MODEL": "claude-opus-5-5", "LITELLM_BASE_URL": "https://gateway.example.com/v1",
       "LITELLM_TEMPERATURE": "omit", "OLLAMA_MODEL": "llama3.1:8b"}
settings = Settings(backend="litellm,ollama", temperature=0.1, max_tokens=2048, timeout_seconds=120,
                    backends=default_backends(env))
messages = PROMPT.format_messages(context="[1] INCIDENT | ...", question="why did removing item fail")

with mock.patch.dict("os.environ", {"LITELLM_API_KEY": "sk-doc-example"}):
    openai_model = create_chat_model("litellm", settings)
payload = openai_model._get_request_payload(messages)
payload["messages"] = [{"role": item["role"], "content": item["content"][:60] + "..."} for item in payload["messages"]]
print("ChatOpenAI (litellm) body:", json.dumps(payload, indent=2))
print("ChatOpenAI client base_url:", openai_model.openai_api_base, "timeout:", openai_model.request_timeout,
      "max_retries:", openai_model.max_retries)

ollama_model = create_chat_model("ollama", settings)
params = ollama_model._chat_params(messages)
params["messages"] = [{"role": item["role"], "content": item["content"][:60] + "..."} for item in params["messages"]]
print("ChatOllama body:", json.dumps(params, indent=2, default=str))
