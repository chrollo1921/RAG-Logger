#!/usr/bin/env python3
"""Verifies docs/: every `module.symbol` reference exists in the code, links resolve, no secrets are present."""
import importlib
import pathlib
import re
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
DOCS = sorted((ROOT / "docs").glob("*.md"))
MODULES = {path.stem for path in (ROOT / "raglogger").glob("*.py")} - {"__main__"}
problems, checked = [], 0

for doc in DOCS:
    text = doc.read_text(encoding="utf-8")
    for target in re.findall(r"\]\(([^)#]+\.md)\)", text):          # relative links
        if not (doc.parent / target).is_file():
            problems.append(f"{doc.name}: broken link {target}")
    for module, symbol in set(re.findall(r"`?(\w+)\.(\w+(?:\.\w+)?)\(\)", text)):  # module.func() / Class.method()
        if module not in MODULES:
            continue
        checked += 1
        try:
            obj = importlib.import_module(f"raglogger.{module}")
            for part in symbol.split("."):
                obj = getattr(obj, part)
        except Exception as error:  # noqa: BLE001
            problems.append(f"{doc.name}: {module}.{symbol} not found ({type(error).__name__})")
    for path in set(re.findall(r"`(raglogger/\w+\.py|scripts/[\w.]+|tests/[\w/.-]+)`", text)):
        if not (ROOT / path).exists():
            problems.append(f"{doc.name}: path {path} does not exist")
    if re.search(r"sk-[A-Za-z0-9]{20,}", text):
        problems.append(f"{doc.name}: looks like it contains an API key")

# bare `Class.method()` / `function()` mentions: check against all module namespaces
names = set()
for module in MODULES:
    mod = importlib.import_module(f"raglogger.{module}")
    names.update(vars(mod))
    for value in vars(mod).values():
        if isinstance(value, type) and value.__module__ == mod.__name__:
            names.update(f"{value.__name__}.{attr}" for attr in vars(value))
            names.update(vars(value))  # bare method names, e.g. `_add()` written next to its class
names.add("reset_collection")  # langchain_chroma.Chroma method, documented as a third-party call
for doc in DOCS:
    for ref in set(re.findall(r"`((?:[A-Z]\w+\.)?_?[a-z]\w+)\(\)`", doc.read_text(encoding="utf-8"))):
        checked += 1
        if ref not in names and ref.split(".")[-1] not in names:
            problems.append(f"{doc.name}: `{ref}()` not found in raglogger")

print(f"{len(DOCS)} docs, {checked} code references checked")
print("\n".join(problems) if problems else "OK: all references resolve, links valid, no secrets")
sys.exit(1 if problems else 0)
