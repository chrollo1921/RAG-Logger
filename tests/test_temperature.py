"""Tests for temperature handling: 'omit', per-backend overrides, and auto-retry when a model rejects it.

Reproduces the LiteLLM error for Claude models that only accept their default temperature:
"global.anthropic.claude-opus-5-5 does not support temperature=0.1. Only temperature=1 is supported."
"""
import _isolation  # noqa: F401  (must be first: keeps tests out of <project>/.raglogger)
import http.server
import io
import json
import os
import tempfile
import threading
import unittest
import urllib.error
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from unittest import mock

from raglogger.config import Settings, default_backends, parse_temperature
from raglogger.generators import LiteLlmGenerator, OllamaGenerator, is_temperature_rejection_text
from raglogger.interaction_log import InteractionLogger
from raglogger.redaction import Redactor

FIXTURE = Path(__file__).parent / "fixtures" / "sample-console.log"
FAKE_KEY = "sk-test-temperature-0123456789abcdef"
REJECTION = ("litellm.UnsupportedParamsError: global.anthropic.claude-opus-5-5 does not support temperature=0.1. "
             "Only temperature=1 is supported. To drop unsupported params, set `litellm.drop_params = True`.")


class _Gateway(http.server.BaseHTTPRequestHandler):
    """Minimal OpenAI-compatible gateway that rejects any temperature except 1 (like the real Claude model)."""

    requests: list = []

    def do_POST(self):  # noqa: N802
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        type(self).requests.append(body)
        if body.get("temperature") not in (None, 1, 1.0):
            status, payload = 400, {"error": {"message": REJECTION, "type": "invalid_request_error", "code": "400"}}
        else:
            status, payload = 200, {"choices": [{"message": {"role": "assistant", "content": "summary [1]"}}],
                                    "usage": {"prompt_tokens": 10, "completion_tokens": 2, "total_tokens": 12}}
        data = json.dumps(payload).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def log_message(self, *args):
        pass


class GatewayTestBase(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.server = http.server.HTTPServer(("127.0.0.1", 0), _Gateway)
        cls.url = f"http://127.0.0.1:{cls.server.server_address[1]}"
        threading.Thread(target=cls.server.serve_forever, daemon=True).start()

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown()
        cls.server.server_close()

    def setUp(self):
        _Gateway.requests = []
        self.environ = mock.patch.dict(os.environ, {"LITELLM_API_KEY": FAKE_KEY})
        self.environ.start()

    def tearDown(self):
        self.environ.stop()

    def sent_temperatures(self):
        return [request.get("temperature", "<omitted>") for request in _Gateway.requests]

class ParseTemperatureTest(unittest.TestCase):
    def test_values(self):
        self.assertEqual(0.1, parse_temperature("0.1"))
        self.assertEqual(1.0, parse_temperature(" 1 "))
        for omitted in ("omit", "OMIT", "none", "default"):
            self.assertIsNone(parse_temperature(omitted))

    def test_invalid_values_name_the_setting(self):
        with self.assertRaisesRegex(ValueError, "LITELLM_TEMPERATURE='hot' is invalid"):
            parse_temperature("hot", "LITELLM_TEMPERATURE")
        with self.assertRaisesRegex(ValueError, "out of range"):
            parse_temperature("3")

    def test_rejection_detection(self):
        self.assertTrue(is_temperature_rejection_text(REJECTION))
        self.assertTrue(is_temperature_rejection_text("temperature must be between 0 and 1"))
        self.assertFalse(is_temperature_rejection_text("rate limit exceeded"))
        self.assertFalse(is_temperature_rejection_text("temperature=0.1 accepted"))


class PerBackendTemperatureTest(unittest.TestCase):
    def settings(self, env, temperature=0.1):
        return Settings(backend="litellm,ollama", temperature=temperature, backends=default_backends(env))

    def test_backend_override_wins_and_others_inherit(self):
        settings = self.settings({"LITELLM_TEMPERATURE": "omit", "OLLAMA_TEMPERATURE": ""})
        self.assertIsNone(settings.temperature_for("litellm"))
        self.assertEqual(0.1, settings.temperature_for("ollama"))
        self.assertEqual(0.7, self.settings({"OLLAMA_TEMPERATURE": "0.7"}).temperature_for("ollama"))

    def test_global_omit(self):
        self.assertIsNone(self.settings({}, temperature=None).temperature_for("openai"))

    def test_invalid_backend_override(self):
        with self.assertRaisesRegex(ValueError, "LITELLM_TEMPERATURE"):
            self.settings({"LITELLM_TEMPERATURE": "warm"}).temperature_for("litellm")


class NativeEngineTest(GatewayTestBase):
    def generator(self, temperature, log_path=None):
        return LiteLlmGenerator("claude-opus-5-5", self.url, InteractionLogger(log_path, Redactor()), 4000,
                                temperature, 5, max_tokens=256)

    def test_omit_sends_no_temperature(self):
        self.assertEqual("summary [1]", self.generator(None).generate("summarize the errors", []))
        self.assertEqual(["<omitted>"], self.sent_temperatures())

    def test_rejected_temperature_is_retried_once_without_it_and_both_calls_are_logged(self):
        with tempfile.TemporaryDirectory() as tmp:
            log_path = Path(tmp) / "llm.jsonl"
            with self.assertLogs("raglogger.generators", level="WARNING") as captured:
                self.assertEqual("summary [1]", self.generator(0.1, log_path).generate("summarize the errors", []))
            records = [json.loads(line) for line in log_path.read_text().splitlines()]
        self.assertEqual([0.1, "<omitted>"], self.sent_temperatures())
        self.assertIn("retrying once without it", captured.output[0])
        self.assertEqual(["request", "response", "request", "response"], [record["type"] for record in records])
        self.assertIn("Only temperature=1", records[1]["error"])
        self.assertEqual("omitted", records[2]["parameters"]["temperature"])
        self.assertEqual("summary [1]", records[3]["response"])

    def test_accepted_temperature_is_sent_unchanged(self):
        self.generator(1.0).generate("q", [])
        self.assertEqual([1.0], self.sent_temperatures())

    def test_other_errors_are_not_retried(self):
        error = urllib.error.HTTPError(self.url, 429, "Too Many", {}, io.BytesIO(b"rate limit exceeded"))
        with mock.patch("urllib.request.urlopen", side_effect=error) as urlopen:
            with self.assertRaises(Exception):
                self.generator(0.1).generate("q", [])
        self.assertEqual(1, urlopen.call_count)

    def test_ollama_omits_options_when_temperature_is_none(self):
        response = mock.MagicMock()
        response.read.return_value = json.dumps({"message": {"content": "ok"}}).encode()
        response.__enter__.return_value = response
        with mock.patch("urllib.request.urlopen", return_value=response) as urlopen:
            OllamaGenerator("m", "http://localhost:11434", InteractionLogger(None, Redactor()), 1000, None, 5) \
                .generate("q", [])
        self.assertNotIn("options", json.loads(urlopen.call_args[0][0].data))

try:
    import langchain_core  # noqa: F401
    HAS_LANGCHAIN = True
except ImportError:  # pragma: no cover
    HAS_LANGCHAIN = False


@unittest.skipUnless(HAS_LANGCHAIN, "LangChain requirements not installed")
class LangChainEngineTest(GatewayTestBase):
    """Uses the real ChatOpenAI client against the rejecting gateway (same code path as the reported error)."""

    def settings(self, litellm_temperature=""):
        env = {"LITELLM_MODEL": "claude-opus-5-5", "LITELLM_BASE_URL": self.url,
               "LITELLM_TEMPERATURE": litellm_temperature}
        return Settings(backend="litellm", temperature=0.1, max_tokens=256, timeout_seconds=5,
                        backends=default_backends(env))

    def generator(self, settings):
        from raglogger.langchain_rag import LangChainChainGenerator, create_chat_model
        model = create_chat_model("litellm", settings)
        return LangChainChainGenerator([("litellm", "claude-opus-5-5", model)], [], 4000,
                                       {"max_tokens": settings.max_tokens})

    def test_reported_error_is_recovered_automatically(self):
        with self.assertLogs("raglogger.langchain_rag", level="WARNING") as captured:
            text, backend = self.generator(self.settings()).answer("summarize the errors", [])
        self.assertEqual(("summary [1]", "litellm"), (text, backend))
        self.assertEqual([0.1, "<omitted>"], self.sent_temperatures())
        self.assertIn("LITELLM_TEMPERATURE=omit", captured.output[0])

    def test_omit_setting_avoids_the_failed_round_trip(self):
        text, _ = self.generator(self.settings("omit")).answer("summarize the errors", [])
        self.assertEqual("summary [1]", text)
        self.assertEqual(["<omitted>"], self.sent_temperatures())

    def test_cli_temperature_flag_end_to_end(self):
        from raglogger.cli import main
        with tempfile.TemporaryDirectory() as tmp:
            index = str(Path(tmp) / "idx.json.gz")
            env = {"LITELLM_MODEL": "claude-opus-5-5", "LITELLM_BASE_URL": self.url,
                   "RAGLOGGER_TRANSCRIPT_LOG": "off", "RAGLOGGER_INTERACTION_LOG": "off"}
            out = io.StringIO()
            with mock.patch.dict(os.environ, env), redirect_stdout(out), redirect_stderr(io.StringIO()):
                base = ["--env-file", "off", "--embeddings", "none", "--index", index]
                self.assertEqual(0, main([*base, "ingest", str(FIXTURE)]))
                self.assertEqual(0, main([*base, "ask", "summarize the errors", "--backend", "litellm",
                                          "--temperature", "omit"]))
        self.assertIn("summary [1]", out.getvalue())
        self.assertEqual(["<omitted>"], self.sent_temperatures())


if __name__ == "__main__":
    unittest.main()
