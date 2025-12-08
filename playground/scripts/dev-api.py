#!/usr/bin/env python3
"""
Local development server for the Python API.

Usage:
    cd playground
    uv run python scripts/dev-api.py

Then in another terminal:
    pnpm dev

The Next.js dev server will proxy /api/* requests to this server.
"""

import json
import sys
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from typing import Any

# Add the kosong source to sys.path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

# Import after path setup
from kosong.conversion import (
    message_to_anthropic,
    message_to_google_genai,
    message_to_kimi,
    message_to_openai_legacy,
    message_to_openai_responses,
)
from kosong.message import Message


def convert_message(message_dict: dict[str, Any]) -> dict[str, Any]:
    """Convert a message dict to all provider formats."""
    message = Message.model_validate(message_dict)
    results: dict[str, Any] = {}

    converters = [
        ("anthropic", message_to_anthropic),
        ("openai_legacy", message_to_openai_legacy),
        ("openai_responses", message_to_openai_responses),
        ("google_genai", message_to_google_genai),
        ("kimi", message_to_kimi),
    ]

    for name, converter in converters:
        try:
            results[name] = {"data": converter(message)}
        except Exception as e:
            results[name] = {"error": str(e)}

    return {
        "success": True,
        "results": results,
        "input_validated": message.model_dump(),
    }


class APIHandler(BaseHTTPRequestHandler):
    def do_POST(self):
        if self.path == "/api/convert":
            try:
                content_length = int(self.headers.get("Content-Length", 0))
                body = self.rfile.read(content_length)
                data = json.loads(body.decode("utf-8"))
                message_dict = data.get("message")

                if not message_dict:
                    self._send_error(400, "Missing 'message' field")
                    return

                result = convert_message(message_dict)
                self._send_json(200, result)

            except json.JSONDecodeError as e:
                self._send_error(400, f"Invalid JSON: {e}")
            except Exception as e:
                self._send_error(500, f"Internal error: {e}")
        else:
            self._send_error(404, "Not found")

    def do_OPTIONS(self):
        self.send_response(200)
        self._send_cors_headers()
        self.end_headers()

    def _send_json(self, status: int, data: dict[str, Any]):
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self._send_cors_headers()
        self.end_headers()
        self.wfile.write(json.dumps(data, ensure_ascii=False).encode("utf-8"))

    def _send_error(self, status: int, message: str):
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self._send_cors_headers()
        self.end_headers()
        self.wfile.write(json.dumps({"success": False, "error": message}).encode("utf-8"))

    def _send_cors_headers(self):
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")

    def log_message(self, format: str, *args: Any):
        print(f"[API] {args[0]}")


if __name__ == "__main__":
    port = 8000
    server = HTTPServer(("", port), APIHandler)
    print(f"Python API server running at http://localhost:{port}")
    print("Endpoints: POST /api/convert")
    print("Press Ctrl+C to stop")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down...")
