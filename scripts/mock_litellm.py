#!/usr/bin/env python3
"""Minimal OpenAI-compatible mock of a LiteLLM gateway for local smoke tests (never use with real keys).

Usage:  .venv/bin/python scripts/mock_litellm.py 4011 &
Serves POST /chat/completions and GET /models; rejects requests without a Bearer token.
"""
import json
import os
import sys
from http.server import BaseHTTPRequestHandler, HTTPServer


class Handler(BaseHTTPRequestHandler):
    def _send(self, status, payload):
        body = json.dumps(payload).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _authorised(self):
        return self.headers.get("Authorization", "").startswith("Bearer ")

    def do_GET(self):  # noqa: N802
        if not self._authorised():
            return self._send(401, {"error": {"message": "missing key"}})
        self._send(200, {"data": [{"id": "claude-opus-mock"}, {"id": "text-embedding-mock"}]})

    def do_POST(self):  # noqa: N802
        if not self._authorised():
            return self._send(401, {"error": {"message": "missing key"}})
        request = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))))
        if os.environ.get("MOCK_REJECT_TEMPERATURE") and request.get("temperature") not in (None, 1, 1.0):
            # Mirrors the real LiteLLM error for Claude models that only accept their default temperature.
            message = (f"litellm.UnsupportedParamsError: global.anthropic.{request['model']} does not support "
                       f"temperature={request['temperature']}. Only temperature=1 is supported. To drop "
                       "unsupported params, set `litellm.drop_params = True`.")
            return self._send(400, {"error": {"message": message, "type": "invalid_request_error",
                                              "param": None, "code": "400"}})
        prompt = request["messages"][-1]["content"]
        answer = (f"[mock {request['model']}] Received {len(prompt)} chars of context. "
                  "Root cause per [1]: see the cited incident.")
        self._send(200, {"id": "mock-1", "object": "chat.completion", "model": request["model"],
                         "choices": [{"index": 0, "finish_reason": "stop",
                                      "message": {"role": "assistant", "content": answer}}],
                         "usage": {"prompt_tokens": len(prompt) // 4, "completion_tokens": 20,
                                   "total_tokens": len(prompt) // 4 + 20}})

    def log_message(self, *args):
        pass


if __name__ == "__main__":
    HTTPServer(("127.0.0.1", int(sys.argv[1]) if len(sys.argv) > 1 else 4011), Handler).serve_forever()
