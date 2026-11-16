"""Runtime configuration resolved from CLI arguments and environment variables.

API keys are deliberately NOT part of :class:`Settings`; generators read them from the environment at call time so
they can never be serialised, printed or logged together with the configuration.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Mapping, Optional

from raglogger.paths import (data_dir, default_index_path, default_interaction_log_path, default_transcript_path,
                             resolve_path)

# Module-level defaults reflect the environment at import time; Settings.from_env() re-resolves them after .env load.
DEFAULT_INDEX_PATH = default_index_path()
DEFAULT_TRANSCRIPT_PATH = default_transcript_path()
DEFAULT_LOG_PATH = default_interaction_log_path()
SUPPORTED_BACKENDS = ("extractive", "litellm", "ollama", "openai")
EMBEDDING_CHOICES = ("none", "local", "litellm")
TEMPERATURE_OMIT_VALUES = ("omit", "none", "default")


def parse_temperature(raw: str, setting: str = "RAGLOGGER_TEMPERATURE") -> Optional[float]:
    """``"0.1"`` -> 0.1; ``"omit"``/``"none"``/``"default"`` -> None (parameter is not sent, provider default applies).

    Some models (e.g. Claude with extended thinking behind LiteLLM) reject any temperature except their default.
    """
    value = raw.strip().lower()
    if value in TEMPERATURE_OMIT_VALUES:
        return None
    try:
        temperature = float(value)
    except ValueError as error:
        raise ValueError(f"{setting}={raw!r} is invalid: use a number between 0 and 2, or 'omit'") from error
    if not 0.0 <= temperature <= 2.0:
        raise ValueError(f"{setting}={raw!r} is out of range: use a number between 0 and 2, or 'omit'")
    return temperature


@dataclass
class BackendSettings:
    model: str
    base_url: str
    api_key_env: str = ""
    temperature: str = ""  # raw per-backend override ("" = inherit RAGLOGGER_TEMPERATURE)
    temperature_env: str = ""


def default_backends(env: Mapping[str, str]) -> dict[str, BackendSettings]:
    return {
        "litellm": BackendSettings(model=env.get("LITELLM_MODEL", ""),
                                   base_url=env.get("LITELLM_BASE_URL", "http://localhost:4000"),
                                   api_key_env="LITELLM_API_KEY",
                                   temperature=env.get("LITELLM_TEMPERATURE", ""),
                                   temperature_env="LITELLM_TEMPERATURE"),
        "ollama": BackendSettings(model=env.get("OLLAMA_MODEL", "llama3.1:8b"),
                                  base_url=env.get("OLLAMA_BASE_URL", "http://localhost:11434"),
                                  temperature=env.get("OLLAMA_TEMPERATURE", ""),
                                  temperature_env="OLLAMA_TEMPERATURE"),
        "openai": BackendSettings(model=env.get("OPENAI_MODEL", "gpt-4o-mini"),
                                  base_url=env.get("OPENAI_BASE_URL", "https://api.openai.com/v1"),
                                  api_key_env="RAGLOGGER_API_KEY",
                                  temperature=env.get("OPENAI_TEMPERATURE", ""),
                                  temperature_env="OPENAI_TEMPERATURE"),
    }


@dataclass
class Settings:
    index_path: Path = DEFAULT_INDEX_PATH
    backend: str = "extractive"
    model: str = ""
    base_url: str = ""
    temperature: Optional[float] = 0.1
    max_tokens: int = 2048
    timeout_seconds: int = 120
    max_context_chars: int = 12000
    interaction_log_path: Optional[Path] = DEFAULT_LOG_PATH
    transcript_log_path: Optional[Path] = DEFAULT_TRANSCRIPT_PATH
    vector_dir_override: Optional[Path] = None
    embeddings: str = "local"
    embedding_model: str = ""
    engine: str = "auto"
    backends: dict[str, BackendSettings] = field(default_factory=lambda: default_backends(os.environ))

    @property
    def embedding_settings(self):
        """Embedding configuration for the vector store (``embeddings == 'none'`` disables it)."""
        from raglogger.vectorstore import EmbeddingSettings  # lazy: keeps the BM25-only path dependency-free
        if self.embeddings not in EMBEDDING_CHOICES:
            raise ValueError(f"Unknown embeddings '{self.embeddings}'. Use one of: {', '.join(EMBEDDING_CHOICES)}")
        litellm = self.backends["litellm"]
        model = self.embedding_model
        if not model and self.embeddings == "litellm":
            model = os.environ.get("LITELLM_EMBEDDING_MODEL", "")
        return EmbeddingSettings(provider=self.embeddings, model=model,
                                 base_url=litellm.base_url, api_key_env=litellm.api_key_env,
                                 timeout_seconds=self.timeout_seconds)

    @property
    def backend_chain(self) -> list[str]:
        """``--backend litellm,ollama`` -> ``["litellm", "ollama"]`` (tried in order)."""
        chain = [name.strip().lower() for name in self.backend.split(",") if name.strip()]
        unknown = [name for name in chain if name not in SUPPORTED_BACKENDS]
        if unknown or not chain:
            raise ValueError(f"Unknown backend(s) {unknown or self.backend!r}. "
                             f"Use one or more of: {', '.join(SUPPORTED_BACKENDS)} (comma-separated)")
        return chain

    def resolve(self, backend: str) -> BackendSettings:
        """Per-backend settings; ``--model``/``--base-url`` override the FIRST backend in the chain only."""
        defaults = self.backends[backend]
        is_primary = self.backend_chain[0] == backend
        return BackendSettings(model=(self.model if is_primary and self.model else defaults.model),
                               base_url=(self.base_url if is_primary and self.base_url else defaults.base_url),
                               api_key_env=defaults.api_key_env, temperature=defaults.temperature,
                               temperature_env=defaults.temperature_env)

    def temperature_for(self, backend: str) -> Optional[float]:
        """``<BACKEND>_TEMPERATURE`` if set, else ``RAGLOGGER_TEMPERATURE``; ``None`` means "do not send"."""
        override = self.backends[backend].temperature if backend in self.backends else ""
        if override.strip():
            return parse_temperature(override, self.backends[backend].temperature_env)
        return self.temperature

    @classmethod
    def from_env(cls) -> "Settings":
        """Reads settings from ``os.environ``. Path defaults derive from ``RAGLOGGER_HOME`` (default
        ``<project>/.raglogger``); relative paths are resolved against the project root."""
        env = os.environ
        log_path = env.get("RAGLOGGER_INTERACTION_LOG") or str(default_interaction_log_path(env))
        transcript = env.get("RAGLOGGER_TRANSCRIPT_LOG") or str(default_transcript_path(env))
        embeddings = env.get("RAGLOGGER_EMBEDDINGS", "local").lower()
        embedding_model = env.get("RAGLOGGER_EMBEDDING_MODEL", "")
        vector_dir = (env.get("RAGLOGGER_VECTOR_DIR") or "").strip()
        return cls(
            index_path=resolve_path(env.get("RAGLOGGER_INDEX") or default_index_path(env)),
            backend=env.get("RAGLOGGER_BACKEND", "extractive").lower(),
            model=env.get("RAGLOGGER_MODEL", ""),
            base_url=env.get("RAGLOGGER_BASE_URL", ""),
            temperature=parse_temperature(env.get("RAGLOGGER_TEMPERATURE") or "0.1"),
            max_tokens=int(env.get("RAGLOGGER_MAX_TOKENS", "2048")),
            timeout_seconds=int(env.get("RAGLOGGER_TIMEOUT", "120")),
            max_context_chars=int(env.get("RAGLOGGER_MAX_CONTEXT_CHARS", "12000")),
            interaction_log_path=None if log_path.lower() == "off" else resolve_path(log_path),
            transcript_log_path=None if transcript.lower() == "off" else resolve_path(transcript),
            vector_dir_override=resolve_path(vector_dir) if vector_dir else None,
            embeddings=embeddings,
            embedding_model=embedding_model,
            engine=env.get("RAGLOGGER_ENGINE", "auto").lower(),
            backends=default_backends(env),
        )

    @property
    def data_dir(self) -> Path:
        return data_dir()
