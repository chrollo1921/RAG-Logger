#!/usr/bin/env bash
# Compares BM25-only (native engine) with hybrid vector+BM25 (LangChain engine) retrieval on the ingested log.
cd "$(dirname "$0")/.." || exit 1
PY=.venv/bin/python
for question in "why could the basket not be priced" "problems with customer account legal type" \
                "coupon promotion rules failing" "IllegalArgumentException removing item"; do
  echo "=== Q: $question"
  echo "--- native (BM25 only)"
  $PY -m raglogger --engine native search "$question" -k 3 2>/dev/null
  echo "--- langchain (hybrid: Chroma vectors + BM25)"
  $PY -m raglogger search "$question" -k 3 2>/dev/null
done
