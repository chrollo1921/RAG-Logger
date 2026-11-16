"""Tests for the project-local data directory (raglogger.paths) and its use in Settings."""
import _isolation  # noqa: F401  (must be first: keeps tests out of <project>/.raglogger)
import os
import unittest
from pathlib import Path
from unittest import mock

from raglogger import paths
from raglogger.config import Settings

PROJECT = Path(__file__).resolve().parent.parent
_PATH_VARS = ("RAGLOGGER_HOME", "RAGLOGGER_INDEX", "RAGLOGGER_VECTOR_DIR", "RAGLOGGER_TRANSCRIPT_LOG",
              "RAGLOGGER_INTERACTION_LOG")


class DataDirTest(unittest.TestCase):
    def test_default_is_inside_the_project_not_the_user_home(self):
        self.assertEqual(PROJECT, paths.PROJECT_ROOT)
        self.assertEqual(PROJECT / ".raglogger", paths.data_dir({}))
        self.assertNotEqual(Path.home() / ".raglogger", paths.data_dir({}))
        self.assertEqual(PROJECT / ".raglogger" / "index.json.gz", paths.default_index_path({}))
        self.assertEqual(PROJECT / ".raglogger" / "logs" / "llm-transcript.log", paths.default_transcript_path({}))
        self.assertEqual(PROJECT / ".raglogger" / "logs" / "llm-interactions.jsonl",
                         paths.default_interaction_log_path({}))

    def test_raglogger_home_override_absolute_relative_and_tilde(self):
        self.assertEqual(Path("/data/rl"), paths.data_dir({"RAGLOGGER_HOME": "/data/rl"}))
        self.assertEqual(PROJECT / "var" / "rl", paths.data_dir({"RAGLOGGER_HOME": "var/rl"}))
        self.assertEqual(Path.home() / "rl", paths.data_dir({"RAGLOGGER_HOME": "~/rl"}))
        self.assertEqual(PROJECT / ".raglogger", paths.data_dir({"RAGLOGGER_HOME": "  "}))

    def test_relative_paths_do_not_depend_on_current_directory(self):
        with mock.patch("pathlib.Path.cwd", return_value=Path("/tmp")):
            self.assertEqual(PROJECT / "x" / "idx.json.gz", paths.resolve_path("x/idx.json.gz"))


class SettingsPathsTest(unittest.TestCase):
    def setUp(self):
        self.environ = mock.patch.dict(os.environ, {})
        self.environ.start()
        for key in _PATH_VARS:  # also drops the test-isolation RAGLOGGER_HOME set in tests/__init__.py
            os.environ.pop(key, None)

    def tearDown(self):
        self.environ.stop()

    def test_test_suite_is_isolated_from_real_data_dir(self):
        self.environ.stop()  # inspect the suite-wide environment set up by tests/_isolation.py
        try:
            self.assertNotEqual(PROJECT / ".raglogger", paths.data_dir())
            self.assertEqual(Path(os.environ["RAGLOGGER_TEST_HOME"]), paths.data_dir())
        finally:
            self.environ.start()
            for key in _PATH_VARS:
                os.environ.pop(key, None)

    def test_defaults_resolve_into_project_data_dir(self):
        settings = Settings.from_env()
        self.assertEqual(PROJECT / ".raglogger" / "index.json.gz", settings.index_path)
        self.assertEqual(PROJECT / ".raglogger" / "logs" / "llm-transcript.log", settings.transcript_log_path)
        self.assertEqual(PROJECT / ".raglogger" / "logs" / "llm-interactions.jsonl", settings.interaction_log_path)
        self.assertIsNone(settings.vector_dir_override)

    def test_raglogger_home_moves_every_default_and_is_read_after_env_load(self):
        os.environ["RAGLOGGER_HOME"] = "/srv/raglogger"  # as if just loaded from .env
        settings = Settings.from_env()
        self.assertEqual(Path("/srv/raglogger/index.json.gz"), settings.index_path)
        self.assertEqual(Path("/srv/raglogger/logs/llm-transcript.log"), settings.transcript_log_path)
        self.assertEqual(Path("/srv/raglogger/logs/llm-interactions.jsonl"), settings.interaction_log_path)

    def test_explicit_paths_win_relative_ones_are_project_based_and_off_disables(self):
        os.environ.update({"RAGLOGGER_HOME": "/srv/raglogger", "RAGLOGGER_INDEX": "data/idx.json.gz",
                           "RAGLOGGER_VECTOR_DIR": "data/vectors", "RAGLOGGER_TRANSCRIPT_LOG": "off",
                           "RAGLOGGER_INTERACTION_LOG": "/var/log/rl.jsonl"})
        settings = Settings.from_env()
        self.assertEqual(PROJECT / "data" / "idx.json.gz", settings.index_path)
        self.assertEqual(PROJECT / "data" / "vectors", settings.vector_dir_override)
        self.assertIsNone(settings.transcript_log_path)
        self.assertEqual(Path("/var/log/rl.jsonl"), settings.interaction_log_path)

    def test_blank_values_fall_back_to_defaults(self):
        os.environ.update({"RAGLOGGER_INDEX": "", "RAGLOGGER_TRANSCRIPT_LOG": "", "RAGLOGGER_VECTOR_DIR": ""})
        settings = Settings.from_env()
        self.assertEqual(PROJECT / ".raglogger" / "index.json.gz", settings.index_path)
        self.assertEqual(PROJECT / ".raglogger" / "logs" / "llm-transcript.log", settings.transcript_log_path)

    def test_vector_store_sits_next_to_index_inside_project(self):
        from raglogger.cli import vector_dir_for
        self.assertEqual(PROJECT / ".raglogger" / "index-chroma", vector_dir_for(Settings.from_env()))


class GitIgnoreTest(unittest.TestCase):
    def test_project_data_dir_is_git_ignored(self):
        self.assertIn(".raglogger/", (PROJECT / ".gitignore").read_text(encoding="utf-8").splitlines())


class PermissionsTest(unittest.TestCase):
    """Directories created inside the data dir must be owner-only: they hold production logs and full prompts."""

    def setUp(self):
        import tempfile
        self.tmp = tempfile.TemporaryDirectory()
        self.environ = mock.patch.dict(os.environ, {"RAGLOGGER_HOME": str(Path(self.tmp.name) / "data")})
        self.environ.start()

    def tearDown(self):
        self.environ.stop()
        self.tmp.cleanup()

    @unittest.skipUnless(os.name == "posix", "POSIX permissions")
    def test_index_and_logs_create_owner_only_directories(self):
        from raglogger.index import Bm25Index
        from raglogger.interaction_log import InteractionLogger
        from raglogger.redaction import Redactor
        root = Path(self.tmp.name) / "data"
        Bm25Index().save(paths.default_index_path())
        InteractionLogger(paths.default_interaction_log_path(), Redactor()).write_record({"type": "probe"})
        for directory in (root, root / "logs"):
            self.assertEqual(0o700, directory.stat().st_mode & 0o777, directory)
        self.assertTrue((root / "index.json.gz").is_file())

    @unittest.skipUnless(os.name == "posix", "POSIX permissions")
    def test_paths_outside_data_dir_keep_default_permissions(self):
        outside = Path(self.tmp.name) / "elsewhere" / "deep"
        paths.ensure_dir(outside)
        self.assertTrue(outside.is_dir())
        self.assertNotEqual(0o700, outside.stat().st_mode & 0o777)


if __name__ == "__main__":
    unittest.main()
