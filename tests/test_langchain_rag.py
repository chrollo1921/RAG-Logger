"""Tests for the LangChain engine: Chroma vector store, hybrid retrieval, chain fallback and LLM logging.

Skipped automatically when the LangChain requirements are not installed (system Python without the venv).
Uses deterministic fake embeddings and fake chat models, so no network, model download or API key is needed.
"""
import _isolation  # noqa: F401  (must be first: keeps tests out of <project>/.raglogger)
import json
import logging
import math
import os
import tempfile
import unittest
import zlib
from pathlib import Path
from unittest import mock

try:
    from langchain_core.embeddings import Embeddings
    from langchain_core.language_models import FakeListChatModel
    HAS_LANGCHAIN = True
except ImportError:  # pragma: no cover
    Embeddings = object
    HAS_LANGCHAIN = False

from raglogger.chunker import LogChunker
from raglogger.config import Settings, default_backends
from raglogger.generators import GenerationError
from raglogger.index import Bm25Index
from raglogger.interaction_log import InteractionLogger
from raglogger.models import SearchFilters
from raglogger.parser import ConsoleLogParser
from raglogger.redaction import Redactor
from raglogger.service import IngestService, QueryService

FIXTURE = Path(__file__).parent / "fixtures" / "sample-console.log"
FAKE_KEY = "sk-test-0123456789abcdefXYZ"


def _chunks():
    entries = [Redactor().redact_entry(entry) for entry in ConsoleLogParser().parse_file(FIXTURE)]
    return LogChunker().build_chunks(entries, "sample")


class KeywordHashEmbedding(Embeddings):
    """Deterministic bag-of-words embedding: texts sharing tokens are close, so ranking assertions are stable."""

    def __init__(self, size: int = 256) -> None:
        self._size = size

    def _embed(self, text: str) -> list:
        from raglogger.index import tokenize
        vector = [0.0] * self._size
        for token in tokenize(text):
            vector[zlib.crc32(token.encode()) % self._size] += 1.0
        norm = math.sqrt(sum(value * value for value in vector)) or 1.0
        return [value / norm for value in vector]

    def embed_documents(self, texts):
        return [self._embed(text) for text in texts]

    def embed_query(self, text):
        return self._embed(text)


class _BrokenChatModel(FakeListChatModel if HAS_LANGCHAIN else object):
    def _call(self, *args, **kwargs):
        raise ConnectionError(f"gateway refused key {FAKE_KEY}")


@unittest.skipUnless(HAS_LANGCHAIN, "LangChain requirements not installed")
class LangChainTestBase(unittest.TestCase):
    def setUp(self):
        from raglogger.llm_logging import LlmInteractionCallback, configure_transcript_logger
        from raglogger.vectorstore import EmbeddingSettings, VectorStoreService
        self.tmp = tempfile.TemporaryDirectory()
        root = Path(self.tmp.name)
        self.transcript_path = root / "llm-transcript.log"
        self.audit_path = root / "llm-interactions.jsonl"
        self.embedding_settings = EmbeddingSettings(provider="local", model="fake-embedding")
        self.vector_service = VectorStoreService(root / "chroma", KeywordHashEmbedding(), self.embedding_settings)
        self.chunks = _chunks()
        self.vector_service.build(self.chunks)
        self.bm25 = Bm25Index()
        self.bm25.build(self.chunks)
        self.callback = LlmInteractionCallback(configure_transcript_logger(self.transcript_path),
                                               InteractionLogger(self.audit_path, Redactor()), Redactor())

    def tearDown(self):
        from raglogger.llm_logging import configure_transcript_logger
        configure_transcript_logger(None)  # release the file handle before the temp dir is removed
        self.tmp.cleanup()

    def _retriever(self):
        from raglogger.langchain_rag import LangChainRetrieverAdapter
        return LangChainRetrieverAdapter(self.vector_service.open(), self.bm25, [self.callback])

    def _transcript(self):
        for handler in logging.getLogger("raglogger.llm.transcript").handlers:
            handler.flush()
        return self.transcript_path.read_text(encoding="utf-8")

    def _audit(self):
        return [json.loads(line) for line in self.audit_path.read_text(encoding="utf-8").splitlines()]

class VectorStoreTest(LangChainTestBase):
    def test_build_persists_all_chunks_with_manifest(self):
        from raglogger.vectorstore import VectorStoreService
        store = self.vector_service.open()
        self.assertEqual(len(self.chunks), store._collection.count())
        manifest = VectorStoreService.read_manifest(Path(self.tmp.name) / "chroma")
        self.assertEqual({"provider": "local", "model": "fake-embedding", "chunks": len(self.chunks)}, manifest)

    def test_rebuild_replaces_previous_vectors(self):
        self.vector_service.build(self.chunks[:3])
        self.assertEqual(3, self.vector_service.open()._collection.count())

    def test_open_rejects_different_embedding_model(self):
        from raglogger.vectorstore import EmbeddingConfigError, EmbeddingSettings, VectorStoreService
        other = VectorStoreService(Path(self.tmp.name) / "chroma", KeywordHashEmbedding(),
                                   EmbeddingSettings(provider="litellm", model="text-embedding-3-small"))
        with self.assertRaisesRegex(EmbeddingConfigError, "built with embeddings local:fake-embedding"):
            other.open()

    def test_no_secrets_embedded(self):
        joined = "\n".join(self.vector_service.open().get()["documents"])
        self.assertNotIn("Sup3rS3cret", joined)
        self.assertNotIn("test@example.com", joined)

    def test_ingest_builds_bm25_and_vectors(self):
        index_path = Path(self.tmp.name) / "idx.json.gz"
        stats = IngestService(ConsoleLogParser(), Redactor(), LogChunker(), self.vector_service).ingest(
            FIXTURE, index_path)
        self.assertTrue(index_path.is_file())
        self.assertEqual(stats.chunks, stats.vectors)


class HybridRetrieverTest(LangChainTestBase):
    def test_fuses_bm25_and_vector_results(self):
        hits = self._retriever().search("IllegalArgumentException removing item", top_k=4)
        self.assertIn("LynxItemRemoveLoggingInterceptor", hits[0].chunk.title)
        self.assertEqual("bm25+vector", hits[0].retrieval)
        self.assertTrue(all(hit.retrieval in ("bm25", "vector", "bm25+vector") for hit in hits))

    def test_filters_apply_to_vector_hits(self):
        hits = self._retriever().search("tax price user group", top_k=10,
                                        filters=SearchFilters(level="ERROR", kind="incident"))
        self.assertTrue(hits)
        self.assertTrue(all(hit.chunk.kind == "incident" and "ERROR" in hit.chunk.levels for hit in hits))

    def test_retrieval_context_is_logged(self):
        self._retriever().search("JVM hung", top_k=2)
        self.assertIn("===== RETRIEVED CONTEXT", self._transcript())
        self.assertIn("JVM appears hung", self._transcript())
        self.assertEqual("retrieval", self._audit()[0]["type"])

class LangChainGeneratorTest(LangChainTestBase):
    def _generator(self, *models):
        from raglogger.langchain_rag import LangChainChainGenerator
        return LangChainChainGenerator(list(models), [self.callback], max_context_chars=6000)

    def test_logs_full_input_context_and_output_response(self):
        hits = self._retriever().search("removing item exception", top_k=2)
        model = FakeListChatModel(responses=["Root cause: interceptor [1]"])
        text, backend = self._generator(("litellm", "claude-opus", model)).answer("why did removal fail?", hits)
        self.assertEqual(("Root cause: interceptor [1]", "litellm"), (text, backend))

        transcript = self._transcript()
        for expected in ("===== LLM REQUEST", "backend=litellm model=claude-opus", "--- [system] ---",
                         "Context excerpts from the console log", "java.lang.IllegalArgumentException",
                         "Question: why did removal fail?", "===== LLM RESPONSE", "Root cause: interceptor [1]",
                         *[hit.chunk.title for hit in hits]):
            self.assertIn(expected, transcript)

        records = {record["type"]: record for record in self._audit()}
        self.assertEqual("why did removal fail?", records["request"]["question"])
        self.assertEqual([hit.chunk.chunk_id for hit in hits], records["request"]["contextChunkIds"])
        self.assertEqual(["system", "human"], [message["role"] for message in records["request"]["messages"]])
        self.assertIn("LynxItemRemoveLoggingInterceptor", records["request"]["messages"][1]["content"])
        self.assertEqual("Root cause: interceptor [1]", records["response"]["response"])
        self.assertIsNotNone(records["response"]["durationMs"])

    @mock.patch.dict(os.environ, {"LITELLM_API_KEY": FAKE_KEY})
    def test_falls_back_and_logs_error_without_leaking_key(self):
        hits = self._retriever().search("tax price", top_k=2)
        generator = self._generator(("litellm", "claude-opus", _BrokenChatModel(responses=["unused"])),
                                    ("ollama", "llama3.1:8b", FakeListChatModel(responses=["local answer"])))
        self.assertEqual(("local answer", "ollama"), generator.answer("q", hits))
        transcript = self._transcript()
        self.assertIn("===== LLM ERROR", transcript)
        self.assertNotIn(FAKE_KEY, transcript)
        self.assertNotIn(FAKE_KEY, self.audit_path.read_text(encoding="utf-8"))

    def test_raises_when_all_backends_fail(self):
        generator = self._generator(("litellm", "m", _BrokenChatModel(responses=["unused"])))
        with self.assertRaisesRegex(GenerationError, "all backends failed") as raised:
            generator.answer("q", self._retriever().search("tax", top_k=1))
        # Key is not in the environment here; the generic secret patterns must still mask it.
        self.assertNotIn(FAKE_KEY, str(raised.exception))

    def test_query_service_end_to_end(self):
        generator = self._generator(("ollama", "llama3.1:8b", FakeListChatModel(responses=["answer"])))
        answer = QueryService(self._retriever(), generator).ask("why did removing item fail")
        self.assertEqual(("answer", "ollama"), (answer.text, answer.backend))
        self.assertEqual("bm25+vector", answer.hits[0].retrieval)


@unittest.skipUnless(HAS_LANGCHAIN, "LangChain requirements not installed")
class FactoryTest(unittest.TestCase):
    def _settings(self, backend):
        return Settings(backend=backend, max_tokens=777,
                        backends=default_backends({"LITELLM_MODEL": "claude-opus",
                                                   "LITELLM_BASE_URL": "https://gw.example/v1"}))

    @mock.patch.dict(os.environ, {"LITELLM_API_KEY": FAKE_KEY})
    def test_litellm_uses_chat_openai_against_gateway(self):
        from raglogger.langchain_rag import create_chat_model
        model = create_chat_model("litellm", self._settings("litellm"))
        self.assertEqual(("ChatOpenAI", "claude-opus", "https://gw.example/v1", 777),
                         (type(model).__name__, model.model_name, model.openai_api_base, model.max_tokens))

    @mock.patch.dict(os.environ, {"LITELLM_API_KEY": ""})
    def test_litellm_without_key_is_rejected(self):
        from raglogger.langchain_rag import create_chat_model
        with self.assertRaisesRegex(ValueError, "LITELLM_API_KEY is not set"):
            create_chat_model("litellm", self._settings("litellm"))

    def test_ollama_model(self):
        from raglogger.langchain_rag import create_chat_model
        model = create_chat_model("ollama", self._settings("ollama"))
        self.assertEqual(("ChatOllama", "llama3.1:8b", 777), (type(model).__name__, model.model, model.num_predict))

    @mock.patch.dict(os.environ, {"LITELLM_API_KEY": ""})
    def test_litellm_embeddings_require_key_and_model(self):
        from raglogger.vectorstore import EmbeddingConfigError, EmbeddingSettings, create_embeddings
        with self.assertRaisesRegex(EmbeddingConfigError, "LITELLM_API_KEY is not set"):
            create_embeddings(EmbeddingSettings(provider="litellm", model="emb"))
        with mock.patch.dict(os.environ, {"LITELLM_API_KEY": FAKE_KEY}):
            with self.assertRaisesRegex(EmbeddingConfigError, "LITELLM_EMBEDDING_MODEL"):
                create_embeddings(EmbeddingSettings(provider="litellm"))
            embeddings = create_embeddings(EmbeddingSettings(provider="litellm", model="emb",
                                                             base_url="https://gw.example/v1"))
            self.assertEqual(("emb", False), (embeddings.model, embeddings.check_embedding_ctx_length))


if __name__ == "__main__":
    unittest.main()
