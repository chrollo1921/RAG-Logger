#!/usr/bin/env python3
"""Prints current-format sample records for docs/ using the test fixture in a temp dir (no network, no secrets)."""
import gzip
import json
import pathlib
import sys
import tempfile
from unittest import mock

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from raglogger.chunker import LogChunker  # noqa: E402
from raglogger.generators import LiteLlmGenerator, build_context  # noqa: E402
from raglogger.index import Bm25Index  # noqa: E402
from raglogger.interaction_log import InteractionLogger  # noqa: E402
from raglogger.parser import ConsoleLogParser  # noqa: E402
from raglogger.redaction import Redactor  # noqa: E402
from raglogger.service import IngestService  # noqa: E402
from raglogger.vectorstore import chunk_metadata  # noqa: E402

with tempfile.TemporaryDirectory() as tmp:
    index_path = pathlib.Path(tmp) / "index.json.gz"
    IngestService(ConsoleLogParser(), Redactor(), LogChunker()).ingest_many(
        [ROOT / "tests" / "fixtures" / "sample-console.log"], index_path)
    payload = json.load(gzip.open(index_path, "rt", encoding="utf-8"))
    print("== metadata.sources:", json.dumps(payload["metadata"]["sources"], indent=2))
    incident = next(chunk for chunk in payload["chunks"] if chunk["kind"] == "incident" and "Remove" in chunk["title"])
    print("== incident chunk:", json.dumps(incident, indent=2))
    print("== chroma metadata for it:", json.dumps(chunk_metadata(Bm25Index.load(index_path).chunks[0]), indent=2))

    index = Bm25Index.load(index_path)
    hits = index.search("why did removing item fail", top_k=2)
    print("== context block:\n" + build_context(hits, 12000)[:900])

    captured = {}

    def fake_urlopen(request, timeout):
        captured.update(url=request.full_url, headers=dict(request.header_items()), body=json.loads(request.data))
        response = mock.MagicMock()
        response.read.return_value = json.dumps({"choices": [{"message": {"content": "ok"}}], "usage": {}}).encode()
        response.__enter__.return_value = response
        return response

    with mock.patch.dict("os.environ", {"LITELLM_API_KEY": "sk-doc-example"}), \
            mock.patch("urllib.request.urlopen", side_effect=fake_urlopen):
        LiteLlmGenerator("claude-opus-5-5", "https://gateway.example.com", InteractionLogger(None, Redactor()),
                         12000, None, 120, max_tokens=2048).generate("why did removing item fail", hits)
    captured["headers"]["Authorization"] = "Bearer ***"
    body = captured["body"]
    body["messages"] = [{"role": m["role"], "content": m["content"][:160] + "..."} for m in body["messages"]]
    print("== HTTP request:", json.dumps(captured, indent=2))
