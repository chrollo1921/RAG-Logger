#!/usr/bin/env python3
"""Developer helper: normalise CRLF -> LF (editor on Windows writes CRLF) and run the test suite."""
import pathlib
import subprocess
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
PATTERNS = ("raglogger/*.py", "tests/*.py", "tests/fixtures/*", "scripts/*", "*.md", "*.txt", ".env.example",
            ".gitignore", ".gitattributes")

fixed = []
for pattern in PATTERNS:
    for path in ROOT.glob(pattern):
        if path.is_file():
            data = path.read_bytes()
            if b"\r\n" in data:
                path.write_bytes(data.replace(b"\r\n", b"\n"))
                fixed.append(str(path.relative_to(ROOT)))
print("normalised:", ", ".join(fixed) or "none")
if "--no-tests" not in sys.argv:
    python = ROOT / ".venv" / "bin" / "python"
    sys.exit(subprocess.call([str(python if python.exists() else sys.executable), "-m", "unittest", "discover",
                              "-s", "tests", *[arg for arg in sys.argv[1:] if arg != "--no-tests"]], cwd=ROOT))
