#!/usr/bin/env python3
"""Adds LITELLM_TEMPERATURE=omit to .env if it is not set yet. Prints only variable names, never values."""
import pathlib
import re
import sys

path = pathlib.Path(__file__).resolve().parent.parent / ".env"
if not path.is_file():
    sys.exit(".env not found")
text = path.read_text(encoding="utf-8")
names = re.findall(r"^\s*(?:export\s+)?([A-Z_][A-Z0-9_]*)\s*=", text, re.MULTILINE)
print("variables in .env:", ", ".join(names))
if re.search(r"^\s*(?:export\s+)?LITELLM_TEMPERATURE\s*=", text, re.MULTILINE):
    print("LITELLM_TEMPERATURE already present - left unchanged")
else:
    block = ("\n# Claude models behind LiteLLM may reject any temperature except their default; 'omit' sends none.\n"
             "LITELLM_TEMPERATURE=omit\n")
    path.write_text(text.rstrip("\n") + "\n" + block, encoding="utf-8")
    print("added LITELLM_TEMPERATURE=omit")
