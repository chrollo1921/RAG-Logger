"""Command-line interface: ``ingest``, ``ask``, ``search``, ``chat``, ``models`` and ``stats``."""
from __future__ import annotations

import argparse
import importlib.util
import json
import logging
import sys
from pathlib import Path
from typing import Optional

from raglogger.chunker import LogChunker
from raglogger.config import EMBEDDING_CHOICES, Settings, parse_temperature
from raglogger.envfile import load_env_file
from raglogger.generators import (AnswerGenerator, ExtractiveGenerator, FallbackChainGenerator, GenerationError,
                                  LiteLlmGenerator, OllamaGenerator, OpenAiCompatibleGenerator)
from raglogger.index import Bm25Index
from raglogger.interaction_log import InteractionLogger
from raglogger.llm_logging import configure_transcript_logger
from raglogger.models import SearchFilters
from raglogger.parser import ConsoleLogParser
from raglogger.redaction import Redactor
from raglogger.service import SOURCES_METADATA_KEY, IngestService, QueryService
from raglogger.sources import DEFAULT_PATTERN

LOG = logging.getLogger("raglogger")
GENERATOR_CLASSES = {"litellm": LiteLlmGenerator, "ollama": OllamaGenerator, "openai": OpenAiCompatibleGenerator}
ENGINES = ("auto", "langchain", "native")


def langchain_available() -> bool:
    return importlib.util.find_spec("langchain_core") is not None


def resolve_engine(requested: str) -> str:
    if requested == "native":
        return "native"
    if langchain_available():
        return "langchain"
    if requested == "langchain":
        raise ValueError("LangChain is not installed. Run: .venv/bin/pip install -r requirements.txt")
    return "native"


def vector_dir_for(settings: Settings) -> Path:
    """Vector store lives next to the BM25 index so ``--index`` keeps separate logs fully separate."""
    if settings.vector_dir_override:
        return settings.vector_dir_override
    return settings.index_path.parent / (settings.index_path.name.split(".")[0] + "-chroma")


def _interaction_logger(settings: Settings, redactor: Redactor) -> InteractionLogger:
    return InteractionLogger(settings.interaction_log_path, redactor,
                             transcript=configure_transcript_logger(settings.transcript_log_path))


def _build_single(backend: str, settings: Settings, interaction_logger: InteractionLogger) -> AnswerGenerator:
    if backend == "extractive":
        return ExtractiveGenerator()
    resolved = settings.resolve(backend)
    if not resolved.model:
        raise ValueError(f"No model configured for backend '{backend}'. Pass --model or set "
                         f"{backend.upper()}_MODEL (run 'raglogger models --backend {backend}' to list them).")
    common = dict(model=resolved.model, base_url=resolved.base_url, interaction_logger=interaction_logger,
                  max_context_chars=settings.max_context_chars, temperature=settings.temperature_for(backend),
                  timeout_seconds=settings.timeout_seconds)
    if backend == "ollama":
        return OllamaGenerator(**common)
    return GENERATOR_CLASSES[backend](**common, max_tokens=settings.max_tokens, api_key_env=resolved.api_key_env)


def build_generator(settings: Settings, redactor: Redactor) -> AnswerGenerator:
    """Native (urllib) engine: one backend, or a :class:`FallbackChainGenerator` for ``litellm,ollama``."""
    interaction_logger = _interaction_logger(settings, redactor)
    generators = [_build_single(name, settings, interaction_logger) for name in settings.backend_chain
                  if name != "extractive"]
    if not generators:
        return ExtractiveGenerator()
    return generators[0] if len(generators) == 1 else FallbackChainGenerator(generators)


def build_langchain_generator(settings: Settings, redactor: Redactor, callbacks: list) -> AnswerGenerator:
    """LangChain engine: ``ChatPromptTemplate | ChatModel | StrOutputParser`` per backend, logged via callbacks."""
    from raglogger.langchain_rag import LangChainChainGenerator, create_chat_model
    models = [(name, settings.resolve(name).model, create_chat_model(name, settings))
              for name in settings.backend_chain if name != "extractive"]
    if not models:
        return ExtractiveGenerator()
    return LangChainChainGenerator(models, callbacks, settings.max_context_chars,
                                   {"max_tokens": settings.max_tokens})


def langchain_callbacks(settings: Settings, redactor: Redactor) -> list:
    from raglogger.llm_logging import LlmInteractionCallback
    transcript = configure_transcript_logger(settings.transcript_log_path)
    audit = InteractionLogger(settings.interaction_log_path, redactor)  # callback writes its own transcript
    return [LlmInteractionCallback(transcript, audit, redactor)]


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="raglogger",
                                     description="RAG question answering over SAP Commerce console logs")
    parser.add_argument("--index", type=Path,
                        help="index file (default <project>/.raglogger/index.json.gz, see RAGLOGGER_HOME)")
    parser.add_argument("-v", "--verbose", action="store_true", help="enable debug logging")
    parser.add_argument("--env-file", help="env file to load (default: ./.env, then the project .env; "
                                           "'off' disables). Shell variables take precedence.")
    parser.add_argument("--engine", choices=ENGINES,
                        help="auto (LangChain if installed), langchain, or native (stdlib only, BM25)")
    parser.add_argument("--embeddings", choices=EMBEDDING_CHOICES,
                        help="vector embeddings: local (default, on-CPU), litellm (gateway), none (BM25 only)")
    parser.add_argument("--embedding-model", help="override embedding model (HF name or LiteLLM alias)")
    sub = parser.add_subparsers(dest="command", required=True)

    ingest = sub.add_parser("ingest", help="parse console logs (files and/or directories) into one index",
                            description="Without --append the index is rebuilt from exactly the given logs. "
                                        "With --append they are added; a log already indexed (same path) is "
                                        "replaced, unchanged ones are skipped.")
    ingest.add_argument("logfiles", type=Path, nargs="*", metavar="LOGFILE_OR_DIR",
                        help="console log files (plain or .gz) or directories (searched recursively)")
    ingest.add_argument("--append", action="store_true", help="add to / update the existing index")
    ingest.add_argument("--force", action="store_true", help="with --append: re-ingest even unchanged files")
    ingest.add_argument("--remove", action="append", default=[], metavar="LABEL",
                        help="drop a source from the index (repeatable; see 'sources')")
    ingest.add_argument("--label", help="custom source label, e.g. node1 (only with a single file)")
    ingest.add_argument("--pattern", default=DEFAULT_PATTERN,
                        help=f"filename pattern for directories (default {DEFAULT_PATTERN})")
    ingest.add_argument("--window-minutes", type=int, default=5)

    for name, help_text in (("ask", "answer a question"), ("search", "show raw retrieval hits"),
                            ("chat", "interactive question loop")):
        command = sub.add_parser(name, help=help_text)
        if name != "chat":
            command.add_argument("question", nargs="+")
        command.add_argument("-k", "--top-k", type=int, default=6)
        command.add_argument("--level", help="filter by level, e.g. ERROR")
        command.add_argument("--logger", help="filter by logger name substring")
        command.add_argument("--kind", choices=["incident", "template", "window", "overview"])
        command.add_argument("--source", help="only these console logs: substring of a label or file path "
                                              "(see 'sources'), e.g. node2 or 20260829")
        command.add_argument("--since", help="e.g. 2026-08-29T00:10")
        command.add_argument("--until", help="e.g. 2026-08-29T01:00")
        command.add_argument("--json", action="store_true", help="machine-readable output")
        if name != "search":
            command.add_argument("--backend", help="extractive | litellm | ollama | openai, or a fallback chain "
                                                   "such as litellm,ollama")
            command.add_argument("--model", help="model for the first backend, e.g. the LiteLLM model alias")
            command.add_argument("--base-url", help="endpoint for the first backend")
            command.add_argument("--temperature", metavar="T|omit",
                                 help="sampling temperature for all backends; 'omit' sends none (needed by models "
                                      "that only accept their default, e.g. some Claude models via LiteLLM)")

    models = sub.add_parser("models", help="list models available on an OpenAI-compatible backend (e.g. LiteLLM)")
    models.add_argument("--backend", default="litellm", choices=["litellm", "openai"])
    models.add_argument("--base-url")

    sub.add_parser("stats", help="show index statistics")
    sources = sub.add_parser("sources", help="list the console logs contained in the index")
    sources.add_argument("--json", action="store_true", help="machine-readable output")
    return parser


def main(argv: Optional[list[str]] = None) -> int:
    args = _parser().parse_args(argv)
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.WARNING, stream=sys.stderr,
                        format="%(asctime)s %(levelname)s [%(name)s] %(message)s")
    try:
        _load_env(args.env_file)
        settings = Settings.from_env()
        if args.index:
            settings.index_path = args.index.expanduser().absolute()  # CLI paths are relative to the shell's cwd
        for attribute in ("backend", "model", "base_url", "engine", "embeddings", "embedding_model"):
            if getattr(args, attribute, None):
                setattr(settings, attribute, getattr(args, attribute))
        if getattr(args, "temperature", None):
            # An explicit CLI value wins over RAGLOGGER_TEMPERATURE *and* the per-backend <BACKEND>_TEMPERATURE.
            settings.temperature = parse_temperature(args.temperature, "--temperature")
            for backend in settings.backends.values():
                backend.temperature = ""
        return _dispatch(args, settings)
    except (FileNotFoundError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2
    except GenerationError as error:
        print(f"error: {error}", file=sys.stderr)
        return 3
    except KeyboardInterrupt:
        return 130


def _load_env(explicit: Optional[str]) -> None:
    """Loads ``.env`` into the process environment before any setting or API key is read."""
    result = load_env_file(explicit)
    for warning in result.warnings:
        print(f"[raglogger] WARNING: {warning}", file=sys.stderr)
    if result.path:
        LOG.info("Environment file %s: %d loaded, %d kept from shell", result.path, len(result.loaded),
                 len(result.skipped))


def _list_models(args: argparse.Namespace, settings: Settings) -> int:
    resolved = settings.backends[args.backend]
    generator = GENERATOR_CLASSES[args.backend](
        model=resolved.model or "-", base_url=args.base_url or resolved.base_url,
        interaction_logger=InteractionLogger(None, Redactor()), max_context_chars=0,
        temperature=settings.temperature, timeout_seconds=settings.timeout_seconds,
        api_key_env=resolved.api_key_env)
    for model_id in generator.list_models():
        print(model_id)
    return 0


def _vector_service(settings: Settings):
    from raglogger.vectorstore import VectorStoreService, create_embeddings
    embedding_settings = settings.embedding_settings
    return VectorStoreService(vector_dir_for(settings), create_embeddings(embedding_settings), embedding_settings)


def _uses_vectors(settings: Settings, engine: str) -> bool:
    return engine == "langchain" and settings.embeddings != "none"


def _ingest(args: argparse.Namespace, settings: Settings, redactor: Redactor, engine: str) -> int:
    vector_service = _vector_service(settings) if _uses_vectors(settings, engine) else None
    service = IngestService(ConsoleLogParser(), redactor, LogChunker(window_minutes=args.window_minutes),
                            vector_service)
    mode = "append" if args.append or args.remove else "rebuild"
    print(f"Ingesting {', '.join(str(path) for path in args.logfiles) or '(no new files)'} "
          f"(mode={mode}, engine={engine}, embeddings={settings.embeddings if vector_service else 'none'}) ...")
    stats = service.ingest_many(args.logfiles, settings.index_path, append=args.append, force=args.force,
                                remove=args.remove, label=args.label, pattern=args.pattern)
    for item in stats.files:
        if item.skipped:
            print(f"  = {item.label}: unchanged, skipped (use --force to re-ingest)")
        else:
            print(f"  + {item.label}: {item.lines} lines -> {item.entries} events -> {item.chunks} chunks "
                  f"({item.duration_seconds}s)")
    for label in stats.removed_sources:
        print(f"  - {label}: removed")
    print(f"Processed {stats.lines} lines -> {stats.entries} events -> {stats.chunks} new chunks "
          f"in {stats.duration_seconds}s")
    if stats.level_counts:
        print("Levels: " + ", ".join(f"{level}={count}" for level, count in stats.level_counts.items()))
    print(f"Index {settings.index_path}: {stats.total_sources} source(s), {stats.total_chunks} chunks")
    if vector_service:
        print(f"Vector store: {stats.vectors} embeddings written to {vector_dir_for(settings)}")
    return 0


def _print_sources(index: Bm25Index, as_json: bool) -> int:
    sources = index.metadata.get(SOURCES_METADATA_KEY)
    if sources is None:  # legacy single-file index
        sources = {"(legacy index)": {"path": index.metadata.get("source", "?"), "chunks": len(index.chunks)}}
    if as_json:
        print(json.dumps(sources, indent=2))
        return 0
    print(f"{'LABEL':<40} {'CHUNKS':>7} {'LINES':>8} {'ERROR':>6} {'WARN':>6}  INGESTED             PATH")
    for label, meta in sorted(sources.items()):
        levels = meta.get("level_counts", {})
        print(f"{label[:40]:<40} {meta.get('chunks', 0):>7} {meta.get('lines', 0):>8} {levels.get('ERROR', 0):>6} "
              f"{levels.get('WARN', 0):>6}  {meta.get('ingested_at', '-'):<20} {meta.get('path')}")
    return 0


def _resolve_sources(index: Bm25Index, needle: str) -> frozenset:
    """Maps a ``--source`` substring to the exact labels whose label *or* original file path contains it."""
    sources = index.metadata.get(SOURCES_METADATA_KEY) or {}
    needle = needle.lower()
    matched = frozenset(label for label, meta in sources.items()
                        if needle in label.lower() or needle in str(meta.get("path", "")).lower())
    if not matched:
        known = ", ".join(sorted(sources)) or "none"
        raise ValueError(f"No indexed source matches --source '{needle}' (label or path). Indexed sources: {known}")
    LOG.info("--source '%s' matched %s", needle, ", ".join(sorted(matched)))
    return matched


def _dispatch(args: argparse.Namespace, settings: Settings) -> int:
    redactor = Redactor()
    if args.command == "models":
        return _list_models(args, settings)
    engine = resolve_engine(settings.engine)
    if args.command == "ingest":
        return _ingest(args, settings, redactor, engine)

    index = Bm25Index.load(settings.index_path)
    if args.command == "sources":
        return _print_sources(index, args.json)
    if args.command == "stats":
        from_manifest = None
        if langchain_available():
            from raglogger.vectorstore import VectorStoreService
            from_manifest = VectorStoreService.read_manifest(vector_dir_for(settings))
        metadata = {key: value for key, value in index.metadata.items() if key != SOURCES_METADATA_KEY}
        sources = index.metadata.get(SOURCES_METADATA_KEY) or {}
        print(json.dumps({"index": str(settings.index_path), "chunks": len(index.chunks), **metadata,
                          "sources": sorted(sources),
                          "vectorStore": {"path": str(vector_dir_for(settings)), **(from_manifest or {})}
                          if from_manifest else None}, indent=2))
        return 0

    filters = SearchFilters(level=args.level, logger=args.logger, kind=args.kind, since=args.since,
                            until=args.until, source=args.source)
    if filters.source:
        filters.source_labels = _resolve_sources(index, filters.source)
    retriever, generator = _build_query_components(args, settings, redactor, engine, index)
    service = QueryService(retriever, generator)
    if args.command == "search":
        hits = service.search(" ".join(args.question), args.top_k, filters)
        if args.json:
            print(json.dumps([{"score": hit.score, "retrieval": hit.retrieval, **hit.chunk.to_dict()}
                              for hit in hits], indent=2))
        else:
            for hit in hits:
                print(f"{hit.score:8.3f}  {hit.retrieval:<11} {hit.chunk.kind:<8} x{hit.chunk.occurrences:<6} "
                      f"[{hit.chunk.source or '-'}] {hit.chunk.title}")
        return 0
    if args.command == "ask":
        _print_answer(service, " ".join(args.question), args, filters)
        return 0

    print(f"raglogger chat - engine={engine}, backend={generator.name}, index={settings.index_path}. "
          "Type 'exit' to quit.")
    while True:
        try:
            question = input("\n? ").strip()
        except EOFError:
            return 0
        if question.lower() in ("exit", "quit", ":q"):
            return 0
        if question:
            _print_answer(service, question, args, filters)


def _build_query_components(args: argparse.Namespace, settings: Settings, redactor: Redactor, engine: str,
                            index: Bm25Index):
    """Returns ``(retriever, generator)`` for the selected engine."""
    is_search = args.command == "search"
    if engine == "native":
        return index, ExtractiveGenerator() if is_search else build_generator(settings, redactor)

    from raglogger.langchain_rag import LangChainRetrieverAdapter
    callbacks = langchain_callbacks(settings, redactor)
    retriever = index
    if _uses_vectors(settings, engine):
        retriever = LangChainRetrieverAdapter(_vector_service(settings).open(), index, callbacks)
    generator = ExtractiveGenerator() if is_search else build_langchain_generator(settings, redactor, callbacks)
    return retriever, generator


def _print_answer(service: QueryService, question: str, args: argparse.Namespace, filters: SearchFilters) -> None:
    answer = service.ask(question, args.top_k, filters)
    if args.json:
        sources = [{"score": hit.score, "retrieval": hit.retrieval, "id": hit.chunk.chunk_id,
                    "source": hit.chunk.source, "title": hit.chunk.title,
                    "lines": [hit.chunk.line_start, hit.chunk.line_end]}
                   for hit in answer.hits]
        print(json.dumps({"question": answer.question, "backend": answer.backend, "answer": answer.text,
                          "sources": sources}, indent=2))
        return
    print(answer.text)
    if answer.backend != "extractive" and answer.hits:
        print("\nSources:")
        for number, hit in enumerate(answer.hits, start=1):
            print(f"  [{number}] ({hit.retrieval}) [{hit.chunk.source or '-'}] {hit.chunk.title} "
                  f"(lines {hit.chunk.line_start}-{hit.chunk.line_end})")
