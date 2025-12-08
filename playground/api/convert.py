"""Serverless conversion endpoint for Vercel Python runtime.

This version calls each provider's `generate` and captures the outgoing HTTP
request without sending traffic, so the playground can show a real request
preview for Anthropic/OpenAI/Kimi.
"""

from __future__ import annotations

import asyncio
import json
from collections.abc import Callable
from http.server import BaseHTTPRequestHandler
from typing import Any

import httpx
from pydantic import ValidationError

from kosong.chat_provider.kimi import Kimi
from kosong.contrib.chat_provider.anthropic import Anthropic
from kosong.contrib.chat_provider.openai_legacy import OpenAILegacy
from kosong.contrib.chat_provider.openai_responses import OpenAIResponses
from kosong.message import Message

_kimi_model = "kimi-k2"
_openai_responses_model = "gpt-4.1"
_openai_legacy_model = "gpt-4o-mini"
_anthropic_model = "claude-3-5-sonnet-20241022"
DUMMY_OPENAI_API_KEY = "sk-dummy"
DUMMY_ANTHROPIC_API_KEY = "anthropic-dummy"
DUMMY_KIMI_API_KEY = "sk-dummy"


class RequestCaptured(Exception):
    """Raised when the request has been intercepted."""

    def __init__(self, request: httpx.Request):
        super().__init__("request captured")
        self.request = request


class RequestCaptureTransport(httpx.AsyncBaseTransport):
    """Transport that records the request and aborts the real call."""

    def __init__(self) -> None:
        self.captured: httpx.Request | None = None

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        self.captured = request
        raise RequestCaptured(request)


def _find_transport_owner(provider: Any) -> Any:
    """Locate the object holding the httpx transport (mirrors chaos provider)."""
    candidates: list[Any] = []

    client = getattr(provider, "client", None)
    if client is not None:
        candidates.append(client)
        raw_client = getattr(client, "_client", None)
        if raw_client is not None:
            candidates.append(raw_client)

    inner_client = getattr(provider, "_client", None)
    if inner_client is not None:
        candidates.append(inner_client)

    for owner in candidates:
        if hasattr(owner, "_transport"):
            return owner
        nested = getattr(owner, "_client", None)
        if nested and hasattr(nested, "_transport"):
            return nested

    raise RuntimeError("Unable to locate httpx transport on provider")


def _request_to_preview(request: httpx.Request) -> dict[str, Any]:
    """Convert a captured request into a JSON-serializable preview."""
    headers = {k: v for k, v in request.headers.items()}
    payload: Any = None
    try:
        content = request.content
    except Exception:
        content = b""

    if content:
        try:
            payload = json.loads(content.decode("utf-8"))
        except Exception:
            payload = content.decode("utf-8", errors="replace")

    return {
        "method": request.method,
        "url": str(request.url),
        "headers": headers,
        "json": payload,
    }


def _anthropic_factory() -> Anthropic:
    return Anthropic(
        model=_anthropic_model,
        default_max_tokens=1024,
        stream=False,
        api_key=DUMMY_ANTHROPIC_API_KEY,
        max_retries=0,
    )


def _openai_legacy_factory() -> OpenAILegacy:
    return OpenAILegacy(
        model=_openai_legacy_model,
        stream=False,
        api_key=DUMMY_OPENAI_API_KEY,
        max_retries=0,
    )


def _openai_responses_factory() -> OpenAIResponses:
    return OpenAIResponses(
        model=_openai_responses_model,
        stream=False,
        api_key=DUMMY_OPENAI_API_KEY,
        max_retries=0,
    )


def _kimi_factory() -> Kimi:
    return Kimi(
        model=_kimi_model,
        stream=False,
        api_key=DUMMY_KIMI_API_KEY,
        base_url="https://api.moonshot.cn/v1",
        max_retries=0,
    )


PROVIDER_FACTORIES: dict[str, Callable[[], Any]] = {
    "anthropic": _anthropic_factory,
    "openai_legacy": _openai_legacy_factory,
    "openai_responses": _openai_responses_factory,
    "kimi": _kimi_factory,
}


async def _capture_provider_request(
    provider_factory: Callable[[], Any], message: Message
) -> httpx.Request:
    provider = provider_factory()
    transport_owner = _find_transport_owner(provider)
    transport = getattr(transport_owner, "_transport", None)
    if not isinstance(transport, httpx.AsyncBaseTransport):
        raise RuntimeError("Provider transport is not httpx.AsyncBaseTransport")

    capture_transport = RequestCaptureTransport()
    transport_owner._transport = capture_transport  # pyright: ignore[reportPrivateUsage]
    try:
        await provider.generate(system_prompt="", tools=[], history=[message])
    except RequestCaptured as exc:
        return exc.request
    except Exception:
        if capture_transport.captured is not None:
            return capture_transport.captured
        raise
    finally:
        transport_owner._transport = transport  # pyright: ignore[reportPrivateUsage]

    raise RuntimeError("Request was not captured")


async def _capture_all(message: Message) -> dict[str, dict[str, Any]]:
    async def run_one(name: str, factory: Callable[[], Any]) -> tuple[str, dict[str, Any]]:
        try:
            request = await _capture_provider_request(factory, message.model_copy(deep=True))
            return name, {"http_request": _request_to_preview(request)}
        except Exception as exc:  # pragma: no cover - playground safety
            return name, {"error": str(exc)}

    tasks = [run_one(name, factory) for name, factory in PROVIDER_FACTORIES.items()]
    results = await asyncio.gather(*tasks)
    output = {name: result for name, result in results}
    output["google_genai"] = {"error": "Not supported in playground request preview"}
    return output


def convert_message(message_dict: dict[str, Any]) -> dict[str, Any]:
    """Validate input and capture provider HTTP requests."""
    message = Message.model_validate(message_dict)
    results = asyncio.run(_capture_all(message))

    return {
        "success": True,
        "results": results,
        "input_validated": message.model_dump(),
    }


class handler(BaseHTTPRequestHandler):
    """Vercel entrypoint."""

    def do_POST(self) -> None:
        if self.path != "/api/convert":
            self._send_json(404, {"success": False, "error": "Not found"})
            return

        content_length = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(content_length)

        try:
            payload = json.loads(body.decode("utf-8") or "{}")
        except json.JSONDecodeError as exc:
            self._send_json(400, {"success": False, "error": f"Invalid JSON: {exc}"})
            return

        message_dict = payload.get("message")
        if message_dict is None:
            self._send_json(400, {"success": False, "error": "Missing 'message' field"})
            return

        try:
            result = convert_message(message_dict)
            self._send_json(200, result)
        except ValidationError as exc:
            self._send_json(400, {"success": False, "error": exc.errors()})
        except Exception as exc:  # pragma: no cover - defensive guard for playground
            self._send_json(500, {"success": False, "error": f"Internal error: {exc}"})

    def do_OPTIONS(self) -> None:
        self.send_response(200)
        self._send_cors_headers()
        self.end_headers()

    def _send_json(self, status: int, data: dict[str, Any]) -> None:
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self._send_cors_headers()
        self.end_headers()
        self.wfile.write(json.dumps(data, ensure_ascii=False).encode("utf-8"))

    def _send_cors_headers(self) -> None:
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")

    def log_message(self, format: str, *args: Any) -> None:  # pragma: no cover
        # Reduce noisy default logging in serverless environment.
        return

