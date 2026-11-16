"""Tests for the .env loader and its CLI integration (standard library only)."""
import _isolation  # noqa: F401  (must be first: keeps tests out of <project>/.raglogger)
import io
import logging
import os
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from unittest import mock

from raglogger.envfile import PROJECT_ROOT, EnvFileError, load_env_file, parse_env_text, resolve_env_file

SECRET = "sk-test-envfile-0123456789abcdef"


class ParseEnvTextTest(unittest.TestCase):
    def parse(self, text, environ=None):
        return parse_env_text(text, environ if environ is not None else {})

    def test_basic_syntax(self):
        values, warnings = self.parse(
            "# comment\n\nA=1\nexport B=two\n  C = spaced  \nD=value # trailing comment\nE=url#fragment\n")
        self.assertEqual({"A": "1", "B": "two", "C": "spaced", "D": "value", "E": "url#fragment"}, values)
        self.assertEqual([], warnings)

    def test_quotes_escapes_and_expansion(self):
        values, _ = self.parse('BASE=/opt\nDQ="a b\\n${BASE}/x \\"q\\""\nSQ=\'lit ${BASE} \\n\'\n'
                               'UNQ=${BASE}/y\nHOMEVAR=${HOME}/z\nEMPTY=\nQEMPTY=""\nHASH="x # not a comment"\n',
                               {"HOME": "/home/u"})
        self.assertEqual("a b\n/opt/x \"q\"", values["DQ"])
        self.assertEqual("lit ${BASE} \\n", values["SQ"])
        self.assertEqual("/opt/y", values["UNQ"])
        self.assertEqual("/home/u/z", values["HOMEVAR"])
        self.assertEqual(("", "", "x # not a comment"), (values["EMPTY"], values["QEMPTY"], values["HASH"]))

    def test_malformed_lines_warn_without_echoing_content(self):
        values, warnings = self.parse(f"not a pair {SECRET}\nBAD=\"unterminated {SECRET}\nOK=1\n")
        self.assertEqual({"OK": "1"}, values)
        self.assertEqual(2, len(warnings))
        self.assertTrue(all(SECRET not in warning for warning in warnings))
        self.assertIn("missing closing", warnings[1])


class LoadEnvFileTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.path = Path(self.tmp.name) / ".env"

    def tearDown(self):
        self.tmp.cleanup()

    def write(self, text, mode=0o600):
        self.path.write_text(text, encoding="utf-8")
        os.chmod(self.path, mode)

    def test_shell_environment_wins_unless_override(self):
        self.write("A=file\nB=file\n")
        environ = {"A": "shell"}
        result = load_env_file(str(self.path), environ=environ)
        self.assertEqual({"A": "shell", "B": "file"}, environ)
        self.assertEqual((["B"], ["A"]), (result.loaded, result.skipped))
        load_env_file(str(self.path), override=True, environ=environ)
        self.assertEqual("file", environ["A"])

    def test_blank_values_do_not_mask_defaults(self):
        self.write("LITELLM_MODEL=\nOLLAMA_MODEL=qwen2.5\n")
        environ = {}
        load_env_file(str(self.path), environ=environ)
        self.assertEqual({"OLLAMA_MODEL": "qwen2.5"}, environ)

    def test_utf8_bom_and_crlf_from_windows_editors(self):
        self.path.write_bytes("\ufeffA=1\r\nB=\"two\"\r\n".encode("utf-8"))
        environ = {}
        load_env_file(str(self.path), environ=environ)
        self.assertEqual({"A": "1", "B": "two"}, environ)

    @unittest.skipUnless(os.name == "posix", "permission bits are POSIX-only")
    def test_warns_when_file_is_readable_by_others(self):
        self.write("A=1\n", mode=0o644)
        self.assertIn("chmod 600", load_env_file(str(self.path), environ={}).warnings[0])
        self.write("A=1\n", mode=0o600)
        self.assertEqual([], load_env_file(str(self.path), environ={}).warnings)

    def test_explicit_missing_file_is_an_error_but_default_is_optional(self):
        with self.assertRaises(EnvFileError):
            load_env_file(str(Path(self.tmp.name) / "missing.env"), environ={})
        with mock.patch("raglogger.envfile.PROJECT_ROOT", Path(self.tmp.name) / "nope"), \
                mock.patch("pathlib.Path.cwd", return_value=Path(self.tmp.name) / "nope"):
            self.assertIsNone(load_env_file(environ={}).path)

    def test_resolution_order_and_disable(self):
        self.write("A=1\n")
        self.assertEqual((self.path, True), resolve_env_file(str(self.path), {}))
        self.assertEqual((self.path, True), resolve_env_file(None, {"RAGLOGGER_ENV_FILE": str(self.path)}))
        self.assertEqual((None, False), resolve_env_file("off", {}))
        with mock.patch("pathlib.Path.cwd", return_value=Path(self.tmp.name)):
            self.assertEqual((self.path, False), resolve_env_file(None, {}))
        self.assertEqual(Path(__file__).resolve().parent.parent, PROJECT_ROOT)

    def test_values_never_logged(self):
        self.write(f"LITELLM_API_KEY={SECRET}\n")
        with self.assertLogs("raglogger.envfile", level=logging.DEBUG) as captured:
            load_env_file(str(self.path), environ={})
        joined = "\n".join(captured.output)
        self.assertIn("LITELLM_API_KEY", joined)
        self.assertNotIn(SECRET, joined)

_MANAGED_KEYS = ("RAGLOGGER_INDEX", "RAGLOGGER_ENGINE", "RAGLOGGER_EMBEDDINGS", "LITELLM_API_KEY",
                 "RAGLOGGER_ENV_FILE")


class CliEnvFileTest(unittest.TestCase):
    """The CLI must load .env before Settings are built; --env-file and shell precedence must work end to end."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.env_path = Path(self.tmp.name) / "custom.env"
        self.index = Path(self.tmp.name) / "from-env-file.json.gz"
        self.fixture = Path(__file__).parent / "fixtures" / "sample-console.log"
        self.env_path.write_text(f"RAGLOGGER_INDEX={self.index}\nRAGLOGGER_ENGINE=native\n"
                                 f"RAGLOGGER_EMBEDDINGS=none\nLITELLM_API_KEY={SECRET}\n", encoding="utf-8")
        os.chmod(self.env_path, 0o600)
        # Isolate os.environ: the CLI writes loaded values into it and they must not leak into other tests.
        self.environ = mock.patch.dict(os.environ, {})
        self.environ.start()
        for key in _MANAGED_KEYS:
            os.environ.pop(key, None)

    def tearDown(self):
        self.environ.stop()
        self.tmp.cleanup()

    def run_cli(self, *argv):
        from raglogger.cli import main
        out, err = io.StringIO(), io.StringIO()
        with redirect_stdout(out), redirect_stderr(err):
            code = main(list(argv))
        return code, out.getvalue(), err.getvalue()

    def test_settings_come_from_env_file(self):
        code, out, err = self.run_cli("--env-file", str(self.env_path), "ingest", str(self.fixture))
        self.assertEqual(0, code, err)
        self.assertIn("engine=native", out)
        self.assertTrue(self.index.is_file())
        self.assertEqual(SECRET, os.environ["LITELLM_API_KEY"])
        self.assertNotIn(SECRET, out + err)

    def test_env_file_selected_via_environment_variable(self):
        os.environ["RAGLOGGER_ENV_FILE"] = str(self.env_path)
        code, _, err = self.run_cli("ingest", str(self.fixture))
        self.assertEqual(0, code, err)
        self.assertTrue(self.index.is_file())

    def test_shell_variable_overrides_env_file(self):
        shell_index = Path(self.tmp.name) / "from-shell.json.gz"
        os.environ["RAGLOGGER_INDEX"] = str(shell_index)
        code, _, err = self.run_cli("--env-file", str(self.env_path), "ingest", str(self.fixture))
        self.assertEqual(0, code, err)
        self.assertTrue(shell_index.is_file())
        self.assertFalse(self.index.exists())

    def test_missing_explicit_env_file_returns_error(self):
        code, _, err = self.run_cli("--env-file", str(Path(self.tmp.name) / "nope.env"), "stats")
        self.assertEqual(2, code)
        self.assertIn("Env file not found", err)


class EnvExampleTest(unittest.TestCase):
    """.env.example must parse cleanly, contain no secrets, and document every variable the app reads."""

    def setUp(self):
        self.text = (PROJECT_ROOT / ".env.example").read_text(encoding="utf-8")
        self.values, self.warnings = parse_env_text(self.text, {"HOME": "/home/u"})

    def test_parses_without_warnings_and_has_no_secret_values(self):
        self.assertEqual([], self.warnings)
        for key in ("LITELLM_API_KEY", "RAGLOGGER_API_KEY"):
            self.assertEqual("", self.values[key], f"{key} must stay blank in the committed template")

    def test_documents_every_variable_read_by_the_code(self):
        import re
        source = "".join(path.read_text(encoding="utf-8") for path in (PROJECT_ROOT / "raglogger").glob("*.py"))
        referenced = set(re.findall(r"\"((?:RAGLOGGER|LITELLM|OLLAMA|OPENAI)_[A-Z_]+)\"", source))
        referenced -= {"RAGLOGGER_ENV_FILE"}  # selects the file itself, cannot live inside it
        self.assertEqual(set(), referenced - set(self.values), "variables missing from .env.example")


if __name__ == "__main__":
    unittest.main()
