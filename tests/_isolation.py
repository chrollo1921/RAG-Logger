"""Isolates test runs from the real project data directory (<project>/.raglogger).

Imported at the very top of every test module (before any raglogger import), because ``unittest discover -s tests``
imports test modules as top-level modules and never runs ``tests/__init__.py``. Idempotent.
"""
import atexit
import os
import shutil
import tempfile

if not os.environ.get("RAGLOGGER_TEST_HOME"):
    _home = tempfile.mkdtemp(prefix="raglogger-test-home-")
    os.environ["RAGLOGGER_TEST_HOME"] = _home
    atexit.register(shutil.rmtree, _home, True)
os.environ["RAGLOGGER_HOME"] = os.environ["RAGLOGGER_TEST_HOME"]
for _name in ("RAGLOGGER_INDEX", "RAGLOGGER_VECTOR_DIR", "RAGLOGGER_TRANSCRIPT_LOG", "RAGLOGGER_INTERACTION_LOG"):
    os.environ.pop(_name, None)
