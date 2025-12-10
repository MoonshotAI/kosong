"""
Snapshot tests for Google GenAI (Gemini) chat provider.

These tests capture the HTTP requests sent to the Google GenAI API and verify
that message conversion is correct using inline snapshots.
"""

import json

import pytest
import respx
from httpx import Response
from inline_snapshot import snapshot

from kosong.contrib.chat_provider.google_genai import GoogleGenAI
from kosong.message import Message, TextPart, ThinkPart, ToolCall
from kosong.tooling import Tool


def make_genai_response(
    text: str = "Hello",
    function_calls: list[dict] | None = None,
) -> dict:
    """Helper to create a minimal valid Google GenAI API response."""
    parts = []
    if text:
        parts.append({"text": text})
    if function_calls:
        for fc in function_calls:
            parts.append({"functionCall": fc})

    return {
        "candidates": [
            {
                "content": {"parts": parts, "role": "model"},
                "finishReason": "STOP",
            }
        ],
        "usageMetadata": {
            "promptTokenCount": 10,
            "candidatesTokenCount": 5,
            "totalTokenCount": 15,
        },
        "modelVersion": "gemini-2.0-flash",
    }


@pytest.fixture
def genai_mock():
    """Mock Google GenAI API and capture requests."""
    with respx.mock(base_url="https://generativelanguage.googleapis.com") as mock:
        mock.route(method="POST", path__regex=r"/v1beta/models/.+:generateContent").mock(
            return_value=Response(200, json=make_genai_response())
        )
        yield mock


@pytest.fixture
def genai_provider():
    """Create a Google GenAI provider for testing."""
    return GoogleGenAI(
        model="gemini-2.0-flash",
        api_key="test-api-key",
        stream=False,  # Use non-streaming for simpler response mocking
    )


def get_request_body(mock: respx.MockRouter) -> dict:
    """Extract and parse the request body from the last call."""
    request = mock.calls.last.request
    return json.loads(request.content.decode())


# =============================================================================
# Basic Message Conversion Tests
# =============================================================================


@pytest.mark.asyncio
async def test_simple_user_message(genai_mock, genai_provider):
    """Test that a simple user message is correctly converted."""
    history = [Message(role="user", content="Hello, Gemini!")]

    stream = await genai_provider.generate("You are helpful.", [], history)
    async for _ in stream:
        pass

    body = get_request_body(genai_mock)
    assert body["contents"] == snapshot([{"parts": [{"text": "Hello, Gemini!"}], "role": "user"}])
    assert body["systemInstruction"] == snapshot(
        {"parts": [{"text": "You are helpful."}], "role": "user"}
    )


@pytest.mark.asyncio
async def test_multi_turn_conversation(genai_mock, genai_provider):
    """Test multi-turn conversation message conversion."""
    history = [
        Message(role="user", content="What is 2+2?"),
        Message(role="assistant", content="2+2 equals 4."),
        Message(role="user", content="And 3+3?"),
    ]

    stream = await genai_provider.generate("", [], history)
    async for _ in stream:
        pass

    body = get_request_body(genai_mock)
    # Note: Google GenAI uses "model" instead of "assistant"
    assert body["contents"] == snapshot(
        [
            {"parts": [{"text": "What is 2+2?"}], "role": "user"},
            {"parts": [{"text": "2+2 equals 4."}], "role": "model"},
            {"parts": [{"text": "And 3+3?"}], "role": "user"},
        ]
    )


# =============================================================================
# Tool Calling Tests
# =============================================================================


@pytest.mark.asyncio
async def test_tool_definition(genai_mock, genai_provider):
    """Test that tool definitions are correctly converted."""
    history = [Message(role="user", content="Add 2 and 3")]
    tools = [
        Tool(
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
    ]

    stream = await genai_provider.generate("", tools, history)
    async for _ in stream:
        pass

    body = get_request_body(genai_mock)
    assert body["tools"] == snapshot(
        [
            {
                "functionDeclarations": [
                    {
                        "name": "add",
                        "description": "Add two integers.",
                        "parameters": {
                            "type": "OBJECT",
                            "properties": {
                                "a": {"type": "INTEGER", "description": "First number"},
                                "b": {"type": "INTEGER", "description": "Second number"},
                            },
                            "required": ["a", "b"],
                        },
                    }
                ]
            }
        ]
    )


@pytest.mark.asyncio
async def test_assistant_message_with_tool_call(genai_mock, genai_provider):
    """Test that assistant messages with tool calls are correctly converted."""
    history = [
        Message(role="user", content="Add 2 and 3"),
        Message(
            role="assistant",
            content="I'll add those numbers for you.",
            tool_calls=[
                ToolCall(
                    id="add_call123",
                    function=ToolCall.FunctionBody(
                        name="add",
                        arguments='{"a": 2, "b": 3}',
                    ),
                )
            ],
        ),
    ]

    stream = await genai_provider.generate("", [], history)
    async for _ in stream:
        pass

    body = get_request_body(genai_mock)
    assert body["contents"] == snapshot(
        [
            {"parts": [{"text": "Add 2 and 3"}], "role": "user"},
            {
                "parts": [
                    {"text": "I'll add those numbers for you."},
                    {"functionCall": {"id": "add_call123", "name": "add", "args": {"a": 2, "b": 3}}},
                ],
                "role": "model",
            },
        ]
    )


@pytest.mark.asyncio
async def test_tool_result_message(genai_mock, genai_provider):
    """Test that tool result messages are correctly converted."""
    history = [
        Message(role="user", content="Add 2 and 3"),
        Message(
            role="assistant",
            content="",
            tool_calls=[
                ToolCall(
                    id="add_call123",
                    function=ToolCall.FunctionBody(name="add", arguments='{"a": 2, "b": 3}'),
                )
            ],
        ),
        Message(role="tool", content="5", tool_call_id="add_call123"),
    ]

    stream = await genai_provider.generate("", [], history)
    async for _ in stream:
        pass

    body = get_request_body(genai_mock)
    # Tool results in Google GenAI are sent as functionResponse
    assert body["contents"] == snapshot(
        [
            {"parts": [{"text": "Add 2 and 3"}], "role": "user"},
            {
                "parts": [
                    {"text": ""},
                    {"functionCall": {"id": "add_call123", "name": "add", "args": {"a": 2, "b": 3}}},
                ],
                "role": "model",
            },
            {
                "parts": [
                    {
                        "functionResponse": {
                            "parts": [],
                            "id": "add_call123",
                            "name": "add",
                            "response": {"output": "5"},
                        }
                    }
                ],
                "role": "user",
            },
        ]
    )


# =============================================================================
# Generation Kwargs Tests
# =============================================================================


@pytest.mark.asyncio
async def test_generation_kwargs(genai_mock, genai_provider):
    """Test that generation kwargs are correctly passed."""
    provider = genai_provider.with_generation_kwargs(
        temperature=0.7,
        top_p=0.9,
        max_output_tokens=2048,
    )
    history = [Message(role="user", content="Hi")]

    stream = await provider.generate("", [], history)
    async for _ in stream:
        pass

    body = get_request_body(genai_mock)
    config = body.get("generationConfig", {})
    assert config.get("temperature") == snapshot(0.7)
    assert config.get("topP") == snapshot(0.9)
    assert config.get("maxOutputTokens") == snapshot(2048)


@pytest.mark.asyncio
async def test_with_thinking_config(genai_mock, genai_provider):
    """Test that thinking configuration is correctly set."""
    provider = genai_provider.with_thinking("high")
    history = [Message(role="user", content="Think hard")]

    stream = await provider.generate("", [], history)
    async for _ in stream:
        pass

    body = get_request_body(genai_mock)
    config = body.get("generationConfig", {})
    # For non-gemini-3 models, thinking uses budget_tokens
    assert config.get("thinkingConfig") == snapshot(
        {"include_thoughts": True, "thinking_budget": 32000}
    )
