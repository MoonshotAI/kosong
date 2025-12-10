"""
Snapshot tests for Kimi chat provider.

These tests capture the HTTP requests sent to the Kimi API and verify
that message conversion is correct using inline snapshots.
"""

import json

import pytest
import respx
from httpx import Response
from inline_snapshot import snapshot

from kosong.chat_provider.kimi import Kimi
from kosong.message import Message, TextPart, ThinkPart, ToolCall
from kosong.tooling import Tool


def make_kimi_response(
    content: str = "Hello",
    tool_calls: list[dict] | None = None,
    reasoning_content: str | None = None,
) -> dict:
    """Helper to create a minimal valid Kimi API response."""
    message: dict = {"role": "assistant", "content": content}
    if tool_calls:
        message["tool_calls"] = tool_calls
    if reasoning_content:
        message["reasoning_content"] = reasoning_content
    return {
        "id": "chatcmpl-test123",
        "object": "chat.completion",
        "created": 1234567890,
        "model": "kimi-k2-turbo-preview",
        "choices": [{"index": 0, "message": message, "finish_reason": "stop"}],
        "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
    }


@pytest.fixture
def kimi_mock():
    """Mock Kimi API and capture requests."""
    with respx.mock(base_url="https://api.moonshot.ai") as mock:
        mock.post("/v1/chat/completions").mock(
            return_value=Response(200, json=make_kimi_response())
        )
        yield mock


@pytest.fixture
def kimi_provider():
    """Create a Kimi provider for testing."""
    return Kimi(
        model="kimi-k2-turbo-preview",
        api_key="test-api-key",
        stream=False,
    )


def get_request_body(mock: respx.MockRouter) -> dict:
    """Extract and parse the request body from the last call."""
    request = mock.calls.last.request
    return json.loads(request.content.decode())


# =============================================================================
# Basic Message Conversion Tests
# =============================================================================


@pytest.mark.asyncio
async def test_simple_user_message(kimi_mock, kimi_provider):
    """Test that a simple user message is correctly converted."""
    history = [Message(role="user", content="Hello, Kimi!")]

    stream = await kimi_provider.generate("You are helpful.", [], history)
    async for _ in stream:
        pass

    body = get_request_body(kimi_mock)
    assert body["messages"] == snapshot(
        [
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "Hello, Kimi!"},
        ]
    )


@pytest.mark.asyncio
async def test_multi_turn_conversation(kimi_mock, kimi_provider):
    """Test multi-turn conversation message conversion."""
    history = [
        Message(role="user", content="What is 2+2?"),
        Message(role="assistant", content="2+2 equals 4."),
        Message(role="user", content="And 3+3?"),
    ]

    stream = await kimi_provider.generate("", [], history)
    async for _ in stream:
        pass

    body = get_request_body(kimi_mock)
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
async def test_tool_definition(kimi_mock, kimi_provider):
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

    stream = await kimi_provider.generate("", tools, history)
    async for _ in stream:
        pass

    body = get_request_body(kimi_mock)
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
async def test_builtin_tool_definition(kimi_mock, kimi_provider):
    """Test that Kimi builtin tools (starting with $) are correctly converted."""
    history = [Message(role="user", content="Search for something")]
    tools = [
        Tool(
            name="$web_search",
            description="Search the web",
            parameters={"type": "object", "properties": {}},
        )
    ]

    stream = await kimi_provider.generate("", tools, history)
    async for _ in stream:
        pass

    body = get_request_body(kimi_mock)
    assert body["tools"] == snapshot(
        [{"type": "builtin_function", "function": {"name": "$web_search"}}]
    )


@pytest.mark.asyncio
async def test_assistant_message_with_tool_call(kimi_mock, kimi_provider):
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

    stream = await kimi_provider.generate("", [], history)
    async for _ in stream:
        pass

    body = get_request_body(kimi_mock)
    assert body["messages"] == snapshot(
        [
            {"role": "user", "content": "Add 2 and 3"},
            {
                "role": "assistant",
                "content": "I'll add those numbers for you.",
                "tool_calls": [
                    {"type": "function", "id": "call_abc123",
                        "function": {"name": "add", "arguments": '{"a": 2, "b": 3}'},
                    }
                ],
            },
        ]
    )


@pytest.mark.asyncio
async def test_tool_result_message(kimi_mock, kimi_provider):
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

    stream = await kimi_provider.generate("", [], history)
    async for _ in stream:
        pass

    body = get_request_body(kimi_mock)
    assert body["messages"] == snapshot(
        [
            {"role": "user", "content": "Add 2 and 3"},
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {"type": "function", "id": "call_abc123",
                        "function": {"name": "add", "arguments": '{"a": 2, "b": 3}'},
                    }
                ],
            },
            {"role": "tool", "content": "5", "tool_call_id": "call_abc123"},
        ]
    )



# =============================================================================
# Reasoning/Thinking Tests
# =============================================================================


@pytest.mark.asyncio
async def test_assistant_message_with_reasoning(kimi_mock, kimi_provider):
    """Test that assistant messages with reasoning content are correctly converted."""
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

    stream = await kimi_provider.generate("", [], history)
    async for _ in stream:
        pass

    body = get_request_body(kimi_mock)
    # Kimi uses reasoning_content field for thinking
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


@pytest.mark.asyncio
async def test_multiple_think_parts_concatenated(kimi_mock, kimi_provider):
    """Test that multiple ThinkParts are concatenated into reasoning_content."""
    history = [
        Message(role="user", content="Complex question"),
        Message(
            role="assistant",
            content=[
                ThinkPart(think="First thought. "),
                ThinkPart(think="Second thought."),
                TextPart(text="Final answer."),
            ],
        ),
    ]

    stream = await kimi_provider.generate("", [], history)
    async for _ in stream:
        pass

    body = get_request_body(kimi_mock)
    assert body["messages"] == snapshot(
        [
            {"role": "user", "content": "Complex question"},
            {
                "role": "assistant",
                "content": "Final answer.",
                "reasoning_content": "First thought. Second thought.",
            },
        ]
    )


# =============================================================================
# Image Content Tests
# =============================================================================


@pytest.mark.asyncio
async def test_image_url_message(kimi_mock, kimi_provider):
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

    stream = await kimi_provider.generate("", [], history)
    async for _ in stream:
        pass

    body = get_request_body(kimi_mock)
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
async def test_generation_kwargs(kimi_mock, kimi_provider):
    """Test that generation kwargs are correctly passed."""
    provider = kimi_provider.with_generation_kwargs(
        temperature=0.7,
        top_p=0.9,
        max_tokens=2048,
    )
    history = [Message(role="user", content="Hi")]

    stream = await provider.generate("", [], history)
    async for _ in stream:
        pass

    body = get_request_body(kimi_mock)
    assert body["temperature"] == snapshot(0.7)
    assert body["top_p"] == snapshot(0.9)
    assert body["max_tokens"] == snapshot(2048)


@pytest.mark.asyncio
async def test_default_temperature_for_k2_model(kimi_mock, kimi_provider):
    """Test that default temperature is set correctly for kimi-k2 models."""
    history = [Message(role="user", content="Hi")]

    stream = await kimi_provider.generate("", [], history)
    async for _ in stream:
        pass

    body = get_request_body(kimi_mock)
    # kimi-k2-turbo-preview should default to 0.6
    assert body["temperature"] == snapshot(0.6)


@pytest.mark.asyncio
async def test_default_temperature_for_thinking_model(kimi_mock):
    """Test that default temperature is 1.0 for thinking models."""
    provider = Kimi(
        model="kimi-k2-thinking-preview",
        api_key="test-api-key",
        stream=False,
    )
    history = [Message(role="user", content="Hi")]

    stream = await provider.generate("", [], history)
    async for _ in stream:
        pass

    body = get_request_body(kimi_mock)
    # kimi-k2-thinking models should default to 1.0
    assert body["temperature"] == snapshot(1.0)


@pytest.mark.asyncio
async def test_with_thinking_sets_reasoning_effort(kimi_mock, kimi_provider):
    """Test that with_thinking sets reasoning_effort correctly."""
    provider = kimi_provider.with_thinking("high")
    history = [Message(role="user", content="Think hard")]

    stream = await provider.generate("", [], history)
    async for _ in stream:
        pass

    body = get_request_body(kimi_mock)
    assert body["reasoning_effort"] == snapshot("high")
    # When reasoning_effort is set, temperature should default to 1.0
    assert body["temperature"] == snapshot(1.0)


@pytest.mark.asyncio
async def test_with_thinking_off(kimi_mock, kimi_provider):
    """Test that with_thinking('off') sets reasoning_effort to None."""
    provider = kimi_provider.with_thinking("off")
    history = [Message(role="user", content="Hi")]

    stream = await provider.generate("", [], history)
    async for _ in stream:
        pass

    body = get_request_body(kimi_mock)
    # When thinking is off, reasoning_effort is set to None (which gets serialized)
    assert body["reasoning_effort"] == snapshot(None)


@pytest.mark.asyncio
async def test_default_max_tokens(kimi_mock, kimi_provider):
    """Test that default max_tokens is set to 32000."""
    history = [Message(role="user", content="Hi")]

    stream = await kimi_provider.generate("", [], history)
    async for _ in stream:
        pass

    body = get_request_body(kimi_mock)
    assert body["max_tokens"] == snapshot(32000)
