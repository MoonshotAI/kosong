"""
Snapshot tests for OpenAI Legacy (Chat Completions API) chat provider.

These tests capture the HTTP requests sent to the OpenAI API and verify
that message conversion is correct using inline snapshots.
"""

import json

import pytest
import respx
from httpx import Response
from inline_snapshot import snapshot

from kosong.contrib.chat_provider.openai_legacy import OpenAILegacy
from kosong.message import Message, TextPart, ThinkPart, ToolCall
from kosong.tooling import Tool


def make_openai_response(
    content: str = "Hello",
    tool_calls: list[dict] | None = None,
) -> dict:
    """Helper to create a minimal valid OpenAI Chat Completions API response."""
    message = {"role": "assistant", "content": content}
    if tool_calls:
        message["tool_calls"] = tool_calls
    return {
        "id": "chatcmpl-test123",
        "object": "chat.completion",
        "created": 1234567890,
        "model": "gpt-4o",
        "choices": [{"index": 0, "message": message, "finish_reason": "stop"}],
        "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
    }


@pytest.fixture
def openai_mock():
    """Mock OpenAI API and capture requests."""
    with respx.mock(base_url="https://api.openai.com") as mock:
        mock.post("/v1/chat/completions").mock(
            return_value=Response(200, json=make_openai_response())
        )
        yield mock


@pytest.fixture
def openai_provider():
    """Create an OpenAI Legacy provider for testing."""
    return OpenAILegacy(
        model="gpt-4o",
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
async def test_simple_user_message(openai_mock, openai_provider):
    """Test that a simple user message is correctly converted."""
    history = [Message(role="user", content="Hello, GPT!")]

    stream = await openai_provider.generate("You are helpful.", [], history)
    async for _ in stream:
        pass

    body = get_request_body(openai_mock)
    assert body["messages"] == snapshot(
        [
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "Hello, GPT!"},
        ]
    )


@pytest.mark.asyncio
async def test_multi_turn_conversation(openai_mock, openai_provider):
    """Test multi-turn conversation message conversion."""
    history = [
        Message(role="user", content="What is 2+2?"),
        Message(role="assistant", content="2+2 equals 4."),
        Message(role="user", content="And 3+3?"),
    ]

    stream = await openai_provider.generate("", [], history)
    async for _ in stream:
        pass

    body = get_request_body(openai_mock)
    assert body["messages"] == snapshot(
        [
            {"role": "user", "content": "What is 2+2?"},
            {"role": "assistant", "content": "2+2 equals 4."},
            {"role": "user", "content": "And 3+3?"},
        ]
    )


# =============================================================================
# Tool Calling Tests
# =============================================================================


@pytest.mark.asyncio
async def test_tool_definition(openai_mock, openai_provider):
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

    stream = await openai_provider.generate("", tools, history)
    async for _ in stream:
        pass

    body = get_request_body(openai_mock)
    assert body["tools"] == snapshot(
        [
            {
                "type": "function",
                "function": {
                    "name": "add",
                    "description": "Add two integers.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "a": {"type": "integer", "description": "First number"},
                            "b": {"type": "integer", "description": "Second number"},
                        },
                        "required": ["a", "b"],
                    },
                },
            }
        ]
    )


@pytest.mark.asyncio
async def test_assistant_message_with_tool_call(openai_mock, openai_provider):
    """Test that assistant messages with tool calls are correctly converted."""
    history = [
        Message(role="user", content="Add 2 and 3"),
        Message(
            role="assistant",
            content="I'll add those numbers for you.",
            tool_calls=[
                ToolCall(
                    id="call_abc123",
                    function=ToolCall.FunctionBody(
                        name="add",
                        arguments='{"a": 2, "b": 3}',
                    ),
                )
            ],
        ),
    ]

    stream = await openai_provider.generate("", [], history)
    async for _ in stream:
        pass

    body = get_request_body(openai_mock)
    assert body["messages"] == snapshot(
        [
            {"role": "user", "content": "Add 2 and 3"},
            {
                "role": "assistant",
                "content": "I'll add those numbers for you.",
                "tool_calls": [
                    {
                        "id": "call_abc123",
                        "type": "function",
                        "function": {"name": "add", "arguments": '{"a": 2, "b": 3}'},
                    }
                ],
            },
        ]
    )


@pytest.mark.asyncio
async def test_tool_result_message(openai_mock, openai_provider):
    """Test that tool result messages are correctly converted."""
    history = [
        Message(role="user", content="Add 2 and 3"),
        Message(
            role="assistant",
            content="",
            tool_calls=[
                ToolCall(
                    id="call_abc123",
                    function=ToolCall.FunctionBody(name="add", arguments='{"a": 2, "b": 3}'),
                )
            ],
        ),
        Message(role="tool", content="5", tool_call_id="call_abc123"),
    ]

    stream = await openai_provider.generate("", [], history)
    async for _ in stream:
        pass

    body = get_request_body(openai_mock)
    assert body["messages"] == snapshot(
        [
            {"role": "user", "content": "Add 2 and 3"},
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "id": "call_abc123",
                        "type": "function",
                        "function": {"name": "add", "arguments": '{"a": 2, "b": 3}'},
                    }
                ],
            },
            {"role": "tool", "content": "5", "tool_call_id": "call_abc123"},
        ]
    )


# =============================================================================
# Reasoning/Thinking Tests (with custom reasoning_key)
# =============================================================================


@pytest.mark.asyncio
async def test_reasoning_content_with_custom_key(openai_mock):
    """Test that reasoning content is correctly handled with custom key."""
    # Some OpenAI-compatible APIs use custom keys for reasoning content
    provider = OpenAILegacy(
        model="deepseek-reasoner",
        api_key="test-api-key",
        stream=False,
        reasoning_key="reasoning_content",
    )

    history = [
        Message(role="user", content="What is 2+2?"),
        Message(
            role="assistant",
            content=[
                ThinkPart(think="Let me think step by step..."),
                TextPart(text="The answer is 4."),
            ],
        ),
        Message(role="user", content="Thanks!"),
    ]

    stream = await provider.generate("", [], history)
    async for _ in stream:
        pass

    body = get_request_body(openai_mock)
    assert body["messages"] == snapshot(
        [
            {"role": "user", "content": "What is 2+2?"},
            {
                "role": "assistant",
                "content": "The answer is 4.",
                "reasoning_content": "Let me think step by step...",
            },
            {"role": "user", "content": "Thanks!"},
        ]
    )


# =============================================================================
# Image Content Tests
# =============================================================================


@pytest.mark.asyncio
async def test_image_url_message(openai_mock, openai_provider):
    """Test that image URL content is correctly converted."""
    from kosong.message import ImageURLPart

    history = [
        Message(
            role="user",
            content=[
                TextPart(text="What's in this image?"),
                ImageURLPart(image_url=ImageURLPart.ImageURL(url="https://example.com/image.png")),
            ],
        )
    ]

    stream = await openai_provider.generate("", [], history)
    async for _ in stream:
        pass

    body = get_request_body(openai_mock)
    assert body["messages"] == snapshot(
        [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "What's in this image?"},
                    {"type": "image_url", "image_url": {"url": "https://example.com/image.png", "id": None}},
                ],
            }
        ]
    )


# =============================================================================
# Generation Kwargs Tests
# =============================================================================


@pytest.mark.asyncio
async def test_generation_kwargs(openai_mock, openai_provider):
    """Test that generation kwargs are correctly passed."""
    provider = openai_provider.with_generation_kwargs(
        temperature=0.7,
        top_p=0.9,
        max_tokens=2048,
    )
    history = [Message(role="user", content="Hi")]

    stream = await provider.generate("", [], history)
    async for _ in stream:
        pass

    body = get_request_body(openai_mock)
    assert body["temperature"] == snapshot(0.7)
    assert body["top_p"] == snapshot(0.9)
    assert body["max_tokens"] == snapshot(2048)


@pytest.mark.asyncio
async def test_with_thinking_sets_reasoning_effort(openai_mock, openai_provider):
    """Test that with_thinking sets reasoning_effort correctly."""
    provider = openai_provider.with_thinking("high")
    history = [Message(role="user", content="Think hard")]

    stream = await provider.generate("", [], history)
    async for _ in stream:
        pass

    body = get_request_body(openai_mock)
    assert body["reasoning_effort"] == snapshot("high")
