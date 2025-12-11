"""Common test cases and utilities for snapshot tests."""

import json
from collections.abc import Sequence
from typing import Any, Protocol

import respx

from kosong.message import ImageURLPart, Message, TextPart, ThinkPart, ToolCall
from kosong.tooling import Tool


class ChatProvider(Protocol):
    async def generate(self, system: str, tools: Sequence[Tool], history: list[Message]): ...


B64_PNG = (
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAA"
    "DUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg=="
)

ADD_TOOL = Tool(
    name="add",
    description="Add two integers.",
    parameters={
        "type": "object",
        "properties": {
            "a": {"type": "integer", "description": "First number"},
            "b": {"type": "integer", "description": "Second number"},
        },
        "required": ["a", "b"],
    },
)

# Common test cases shared across providers
COMMON_CASES: dict[str, dict[str, Any]] = {
    "simple_user_message": {
        "system": "You are helpful.",
        "history": [Message(role="user", content="Hello!")],
    },
    "multi_turn_conversation": {
        "history": [
            Message(role="user", content="What is 2+2?"),
            Message(role="assistant", content="2+2 equals 4."),
            Message(role="user", content="And 3+3?"),
        ],
    },
    "tool_definition": {
        "history": [Message(role="user", content="Add 2 and 3")],
        "tools": [ADD_TOOL],
    },
    "assistant_with_tool_call": {
        "history": [
            Message(role="user", content="Add 2 and 3"),
            Message(
                role="assistant",
                content="I'll add those numbers for you.",
                tool_calls=[
                    ToolCall(id="call_abc123", function=ToolCall.FunctionBody(name="add", arguments='{"a": 2, "b": 3}'))
                ],
            ),
        ],
    },
    "tool_result": {
        "history": [
            Message(role="user", content="Add 2 and 3"),
            Message(
                role="assistant",
                content="",
                tool_calls=[
                    ToolCall(id="call_abc123", function=ToolCall.FunctionBody(name="add", arguments='{"a": 2, "b": 3}'))
                ],
            ),
            Message(role="tool", content="5", tool_call_id="call_abc123"),
        ],
    },
    "image_url": {
        "history": [
            Message(
                role="user",
                content=[
                    TextPart(text="What's in this image?"),
                    ImageURLPart(image_url=ImageURLPart.ImageURL(url="https://example.com/image.png")),
                ],
            )
        ],
    },
}


async def capture_request(
    mock: respx.MockRouter,
    provider: ChatProvider,
    system: str,
    tools: Sequence[Tool],
    history: list[Message],
) -> dict:
    """Generate and capture the request body."""
    stream = await provider.generate(system, tools, history)
    async for _ in stream:
        pass
    return json.loads(mock.calls.last.request.content.decode())


async def run_test_cases(
    mock: respx.MockRouter,
    provider: ChatProvider,
    cases: dict[str, dict[str, Any]],
    extract_keys: tuple[str, ...],
) -> dict[str, dict]:
    """Run all test cases and return results dict for snapshot comparison."""
    results = {}
    for name, case in cases.items():
        body = await capture_request(mock, provider, case.get("system", ""), case.get("tools", []), case["history"])
        results[name] = {k: v for k, v in body.items() if k in extract_keys}
    return results
