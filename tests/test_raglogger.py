import _isolation  # noqa: F401  (must be first: keeps tests out of <project>/.raglogger)
import gzip
import io
import json
import os
import tempfile
import unittest
import urllib.error
from contextlib import redirect_stdout
from pathlib import Path
from unittest import mock

from raglogger.chunker import LogChunker, normalize_message
from raglogger.cli import build_generator, main
from raglogger.config import Settings, default_backends
from raglogger.generators import (ExtractiveGenerator, FallbackChainGenerator, GenerationError, LiteLlmGenerator,
                                  OllamaGenerator)
from raglogger.index import Bm25Index, tokenize
from raglogger.interaction_log import InteractionLogger
from raglogger.models import SearchFilters
from raglogger.parser import ConsoleLogParser
from raglogger.redaction import MASK, Redactor
from raglogger.service import IngestService, QueryService

FIXTURE = Path(__file__).parent / "fixtures" / "sample-console.log"


def _entries():
    return [Redactor().redact_entry(entry) for entry in ConsoleLogParser().parse_file(FIXTURE)]


class ParserTest(unittest.TestCase):
    def test_groups_header_and_stack_trace(self):
        entries = _entries()
        self.assertEqual(9, len(entries))
        removal = entries[5]
        self.assertEqual("ERROR", removal.level)
        self.assertEqual("LynxItemRemoveLoggingInterceptor", removal.logger)
        self.assertEqual("hybrisHTTP13", removal.thread)
        self.assertEqual("-SF-7025586804_457511", removal.correlation_id)
        self.assertEqual("f9919a08fd329a43e369835b5789045d", removal.trace_id)
        self.assertEqual("java.lang.IllegalArgumentException: Exception occurred while removing item",
                         removal.exception_line)
        self.assertEqual(3, len(removal.continuation))
        self.assertEqual(10, removal.last_line_no)

    def test_parses_logger_without_context_and_wrapper_lines(self):
        entries = _entries()
        self.assertEqual("LynxRegexPatternFileListFilter", entries[0].logger)
        self.assertEqual("2026-08-29T00:13:22,631", entries[0].timestamp)
        self.assertEqual("TanukiWrapper", entries[-1].logger)
        self.assertIn("JVM appears hung", entries[-1].message)

    def test_strips_ansi_colour_codes(self):
        line = ("INFO   | jvm 1    | main    | 2026/08/29 00:13:22.638 | \x1b[m[2026-08-29T00:13:22,631] "
                "\x1b[32mINFO \x1b[m [task-scheduler-3] [LynxRegexPatternFileListFilter] #cleanUpIfNeeded\n")
        entry = list(ConsoleLogParser().parse_lines([line]))[0]
        self.assertEqual("INFO", entry.level)
        self.assertEqual("task-scheduler-3", entry.thread)
        self.assertEqual("LynxRegexPatternFileListFilter", entry.logger)
        self.assertEqual("#cleanUpIfNeeded", entry.message)

    def test_continuation_lines_are_capped(self):
        lines = ["[2026-08-29T00:00:00,000] ERROR [t] [X] boom\n"] + [f"\tat a.b.C.m{i}(C.java:{i})\n" for i in range(10)]
        entry = list(ConsoleLogParser(max_continuation_lines=3).parse_lines(lines))[0]
        self.assertEqual(3, len(entry.continuation))
        self.assertEqual(7, entry.truncated_lines)


class RedactionTest(unittest.TestCase):
    def test_masks_secrets_and_emails(self):
        warn = _entries()[7]
        self.assertNotIn("Sup3rS3cret", warn.message)
        self.assertNotIn("test@example.com", warn.message)
        self.assertIn(MASK, warn.message)

    def test_masks_bearer_tokens(self):
        self.assertEqual("Authorization: Bearer " + MASK, Redactor().redact("Authorization: Bearer abc.def-123"))


class ChunkerTest(unittest.TestCase):
    def test_normalize_collapses_variable_tokens(self):
        first = normalize_message("Tax price user group wasn't found for cart 735042471, is personal: false")
        second = normalize_message("Tax price user group wasn't found for cart 735099999, is personal: false")
        self.assertEqual(first, second)

    def test_builds_all_chunk_kinds(self):
        chunker = LogChunker(window_minutes=5)
        chunks = chunker.build_chunks(_entries(), "sample")
        kinds = {chunk.kind for chunk in chunks}
        self.assertEqual({"template", "incident", "window", "overview"}, kinds)
        tax = [c for c in chunks if c.kind == "incident" and "LynxPersonalTaxPrice" in c.title]
        self.assertEqual(1, len(tax))
        self.assertEqual(2, tax[0].occurrences)
        removal = [c for c in chunks if c.kind == "incident" and "IllegalArgumentException" in c.text]
        self.assertIn("LynxItemRemoveLoggingInterceptor.java:65", removal[0].text)
        self.assertEqual(9, chunker.entry_count)

    def test_rejects_invalid_window(self):
        with self.assertRaises(ValueError):
            LogChunker(window_minutes=0)


class IndexTest(unittest.TestCase):
    def setUp(self):
        self.index = Bm25Index()
        self.index.build(LogChunker().build_chunks(_entries(), "sample"))

    def test_tokenize_splits_camel_case(self):
        tokens = tokenize("LynxItemRemoveLoggingInterceptor failed")
        self.assertIn("lynxitemremovelogginginterceptor", tokens)
        self.assertIn("interceptor", tokens)
        self.assertIn("remove", tokens)

    def test_tokenize_normalizes_plurals_and_levels(self):
        self.assertEqual(["error", "warn", "fail"], tokenize("errors warnings failed"))

    def test_search_ranks_matching_incident_first(self):
        hits = self.index.search("IllegalArgumentException removing item", top_k=3)
        self.assertEqual("incident", hits[0].chunk.kind)
        self.assertIn("LynxItemRemoveLoggingInterceptor", hits[0].chunk.title)

    def test_filters(self):
        hits = self.index.search("tax price user group", top_k=10, filters=SearchFilters(level="warn"))
        self.assertTrue(all("WARN" in hit.chunk.levels for hit in hits))
        late = self.index.search("tax price", top_k=10, filters=SearchFilters(since="2026-08-29 00:15", kind="incident"))
        self.assertTrue(late)
        self.assertTrue(all(hit.chunk.end_time >= "2026-08-29T00:15" for hit in late))

    def test_overview_for_summary_questions(self):
        self.assertEqual("overview", self.index.search("give me an overall summary", top_k=1)[0].chunk.kind)

    def test_save_and_load_roundtrip(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "idx.json.gz"
            self.index.save(path)
            loaded = Bm25Index.load(path)
            self.assertEqual(len(self.index.chunks), len(loaded.chunks))
            self.assertEqual(self.index.search("hung JVM")[0].chunk.chunk_id, loaded.search("hung JVM")[0].chunk.chunk_id)

    def test_load_missing_index(self):
        with self.assertRaises(FileNotFoundError):
            Bm25Index.load(Path("/nonexistent/idx.json.gz"))



class _FailingGenerator(ExtractiveGenerator):
    name = "failing"

    def generate(self, question, hits):
        raise GenerationError("connection refused")


class ServiceTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.index_path = Path(self.tmp.name) / "idx.json.gz"
        self.stats = IngestService(ConsoleLogParser(), Redactor(), LogChunker()).ingest(FIXTURE, self.index_path)

    def tearDown(self):
        self.tmp.cleanup()

    def test_ingest_stats(self):
        self.assertEqual(13, self.stats.lines)
        self.assertEqual(9, self.stats.entries)
        self.assertEqual(3, self.stats.level_counts["ERROR"])

    def test_ingest_missing_file(self):
        with self.assertRaises(FileNotFoundError):
            IngestService(ConsoleLogParser(), Redactor(), LogChunker()).ingest(Path("/nope.log"), self.index_path)

    def test_ask_falls_back_when_llm_fails(self):
        service = QueryService(Bm25Index.load(self.index_path), _FailingGenerator())
        answer = service.ask("tax price user group not found")
        self.assertEqual("extractive", answer.backend)
        self.assertIn("LLM backend unavailable", answer.text)
        self.assertIn("LynxPersonalTaxPriceUserGroupServiceImpl", answer.text)

    def test_empty_question_rejected(self):
        with self.assertRaises(ValueError):
            QueryService(Bm25Index.load(self.index_path), ExtractiveGenerator()).ask("  ")

    def test_no_secret_persisted_in_index(self):
        with gzip.open(self.index_path, "rt", encoding="utf-8") as handle:
            content = handle.read()
        self.assertNotIn("Sup3rS3cret", content)
        self.assertNotIn("test@example.com", content)


class OllamaGeneratorTest(unittest.TestCase):
    def test_calls_endpoint_and_logs_interaction(self):
        index = Bm25Index()
        index.build(LogChunker().build_chunks(_entries(), "sample"))
        hits = index.search("removing item exception", top_k=2)
        with tempfile.TemporaryDirectory() as tmp:
            log_path = Path(tmp) / "llm.jsonl"
            generator = OllamaGenerator("llama3.1:8b", "http://localhost:11434",
                                        InteractionLogger(log_path, Redactor()), 4000, 0.1, 5)
            body = json.dumps({"message": {"content": "Root cause [1]"}, "prompt_eval_count": 10,
                               "eval_count": 5}).encode()
            response = mock.MagicMock()
            response.read.return_value = body
            response.__enter__.return_value = response
            with mock.patch("urllib.request.urlopen", return_value=response) as urlopen:
                self.assertEqual("Root cause [1]", generator.generate("why?", hits))
            sent = json.loads(urlopen.call_args[0][0].data.decode())
            self.assertEqual("llama3.1:8b", sent["model"])
            self.assertIn("LynxItemRemoveLoggingInterceptor", sent["messages"][1]["content"])
            records = [json.loads(line) for line in log_path.read_text().splitlines()]
            self.assertEqual(["request", "response"], [record["type"] for record in records])
            self.assertEqual(15, records[1]["tokens"]["total"])

    def test_unreachable_endpoint_raises_generation_error(self):
        generator = OllamaGenerator("m", "http://127.0.0.1:9", InteractionLogger(None, Redactor()), 1000, 0.1, 2)
        with self.assertRaises(GenerationError):
            generator.generate("q", [])


FAKE_KEY = "sk-test-0123456789abcdefXYZ"


def _http_response(payload):
    response = mock.MagicMock()
    response.read.return_value = json.dumps(payload).encode()
    response.__enter__.return_value = response
    return response


def _hits():
    index = Bm25Index()
    index.build(LogChunker().build_chunks(_entries(), "sample"))
    return index.search("removing item exception", top_k=2)


def _litellm(log_path=None):
    return LiteLlmGenerator("claude-opus", "http://litellm.local:4000", InteractionLogger(log_path, Redactor()),
                            4000, 0.1, 5, max_tokens=1024)


@mock.patch.dict(os.environ, {"LITELLM_API_KEY": FAKE_KEY})
class LiteLlmGeneratorTest(unittest.TestCase):
    def test_sends_bearer_key_and_model_alias(self):
        reply = {"choices": [{"message": {"content": "Root cause [1]"}}],
                 "usage": {"prompt_tokens": 100, "completion_tokens": 20, "total_tokens": 120}}
        with tempfile.TemporaryDirectory() as tmp:
            log_path = Path(tmp) / "llm.jsonl"
            with mock.patch("urllib.request.urlopen", return_value=_http_response(reply)) as urlopen:
                self.assertEqual("Root cause [1]", _litellm(log_path).generate("why?", _hits()))
            request = urlopen.call_args[0][0]
            self.assertEqual("http://litellm.local:4000/chat/completions", request.full_url)
            self.assertEqual(f"Bearer {FAKE_KEY}", request.get_header("Authorization"))
            body = json.loads(request.data.decode())
            self.assertEqual("claude-opus", body["model"])
            self.assertEqual(1024, body["max_tokens"])
            log_text = log_path.read_text()
            self.assertNotIn(FAKE_KEY, log_text)
            self.assertEqual(120, json.loads(log_text.splitlines()[1])["tokens"]["total"])

    def test_missing_key_fails_fast_without_network(self):
        with mock.patch.dict(os.environ, {"LITELLM_API_KEY": ""}), mock.patch("urllib.request.urlopen") as urlopen:
            with self.assertRaisesRegex(GenerationError, "LITELLM_API_KEY is not set"):
                _litellm().generate("q", _hits())
            urlopen.assert_not_called()

    def test_key_is_scrubbed_from_http_errors(self):
        error = urllib.error.HTTPError("u", 401, "Unauthorized", {}, io.BytesIO(f"bad key {FAKE_KEY}".encode()))
        with mock.patch("urllib.request.urlopen", side_effect=error):
            with self.assertRaises(GenerationError) as raised:
                _litellm().generate("q", _hits())
        self.assertIn("HTTP 401", str(raised.exception))
        self.assertNotIn(FAKE_KEY, str(raised.exception))

    def test_list_models(self):
        reply = {"data": [{"id": "claude-opus"}, {"id": "gpt-4o"}]}
        with mock.patch("urllib.request.urlopen", return_value=_http_response(reply)) as urlopen:
            self.assertEqual(["claude-opus", "gpt-4o"], _litellm().list_models())
        self.assertEqual("http://litellm.local:4000/models", urlopen.call_args[0][0].full_url)


class FallbackChainTest(unittest.TestCase):
    def test_uses_next_backend_when_first_fails(self):
        chain = FallbackChainGenerator([_FailingGenerator(), ExtractiveGenerator()])
        text, backend = chain.answer("q", _hits())
        self.assertEqual("extractive", backend)
        self.assertIn("LynxItemRemoveLoggingInterceptor", text)

    def test_raises_when_all_fail(self):
        with self.assertRaisesRegex(GenerationError, "all backends failed"):
            FallbackChainGenerator([_FailingGenerator(), _FailingGenerator()]).answer("q", _hits())

    def test_query_service_reports_answering_backend(self):
        index = Bm25Index()
        index.build(LogChunker().build_chunks(_entries(), "sample"))
        chain = FallbackChainGenerator([_FailingGenerator(), ExtractiveGenerator()])
        self.assertEqual("extractive", QueryService(index, chain).ask("removing item").backend)


class SettingsTest(unittest.TestCase):
    def _settings(self, backend, model=""):
        return Settings(backend=backend, model=model,
                        backends=default_backends({"LITELLM_MODEL": "claude-opus", "OLLAMA_MODEL": "qwen2.5"}))

    def test_backend_chain_parsing(self):
        self.assertEqual(["litellm", "ollama"], self._settings("LiteLLM, ollama").backend_chain)
        with self.assertRaises(ValueError):
            _ = self._settings("litellm,bogus").backend_chain

    def test_cli_model_overrides_only_primary_backend(self):
        settings = self._settings("litellm,ollama", model="claude-sonnet")
        self.assertEqual("claude-sonnet", settings.resolve("litellm").model)
        self.assertEqual("qwen2.5", settings.resolve("ollama").model)
        self.assertEqual("LITELLM_API_KEY", settings.resolve("litellm").api_key_env)

    def test_build_generator_creates_chain(self):
        generator = build_generator(self._settings("litellm,ollama"), Redactor())
        self.assertIsInstance(generator, FallbackChainGenerator)
        self.assertEqual("litellm+ollama", generator.name)

    def test_litellm_requires_model(self):
        settings = Settings(backend="litellm", backends=default_backends({}))
        with self.assertRaisesRegex(ValueError, "No model configured"):
            build_generator(settings, Redactor())


class CliTest(unittest.TestCase):
    def test_ingest_ask_search_stats(self):
        with tempfile.TemporaryDirectory() as tmp:
            index = str(Path(tmp) / "idx.json.gz")
            output = io.StringIO()
            with redirect_stdout(output):
                self.assertEqual(0, main(["--env-file", "off", "--index", index, "ingest", str(FIXTURE)]))
                self.assertEqual(0, main(["--env-file", "off", "--index", index, "ask", "JVM", "hung",
                                          "--backend", "extractive"]))
                self.assertEqual(0, main(["--env-file", "off", "--index", index, "search", "tax",
                                          "--level", "ERROR"]))
                self.assertEqual(0, main(["--env-file", "off", "--index", index, "stats"]))
            text = output.getvalue()
            self.assertIn("Processed 13 lines", text)
            self.assertIn("JVM appears hung", text)
            self.assertIn("LynxPersonalTaxPriceUserGroupServiceImpl", text)

    def test_missing_index_returns_error_code(self):
        self.assertEqual(2, main(["--env-file", "off", "--index", "/nonexistent/idx.json.gz", "stats"]))


if __name__ == "__main__":
    unittest.main()
