"""
Snapshot tests for Anthropic chat provider.

These tests capture the HTTP requests sent to the Anthropic API and verify
that message conversion is correct using inline snapshots.
"""

import json

import pytest
import respx
from httpx import Response
from inline_snapshot import snapshot

from kosong.contrib.chat_provider.anthropic import Anthropic
from kosong.message import Message, TextPart, ThinkPart, ToolCall
from kosong.tooling import Tool


def make_anthropic_response(
    content: list[dict] | None = None,
    usage: dict | None = None,
) -> dict:
    """Helper to create a minimal valid Anthropic API response."""
    return {
        "id": "msg_test_123",
        "type": "message",
        "role": "assistant",
        "model": "claude-sonnet-4-20250514",
        "content": content or [{"type": "text", "text": "Hello"}],
        "stop_reason": "end_turn",
        "usage": usage or {"input_tokens": 10, "output_tokens": 5},
    }


@pytest.fixture
def anthropic_mock():
    """Mock Anthropic API and capture requests."""
    with respx.mock(base_url="https://api.anthropic.com") as mock:
        mock.post("/v1/messages").mock(
            return_value=Response(200, json=make_anthropic_response())
        )
        yield mock


@pytest.fixture
def anthropic_provider():
    """Create an Anthropic provider for testing."""
    return Anthropic(
        model="claude-sonnet-4-20250514",
        api_key="test-api-key",
        default_max_tokens=1024,
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
async def test_simple_user_message(anthropic_mock, anthropic_provider):
    """Test that a simple user message is correctly converted."""
    history = [Message(role="user", content="Hello, Claude!")]

    stream = await anthropic_provider.generate("You are helpful.", [], history)
    async for _ in stream:
        pass

    body = get_request_body(anthropic_mock)
    assert body["messages"] == snapshot(
        [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Hello, Claude!", "cache_control": {"type": "ephemeral"}}
                ],
            }
        ]
    )


@pytest.mark.asyncio
async def test_system_prompt(anthropic_mock, anthropic_provider):
    """Test that system prompt is correctly set."""
    history = [Message(role="user", content="Hi")]

    stream = await anthropic_provider.generate("You are a helpful assistant.", [], history)
    async for _ in stream:
        pass

    body = get_request_body(anthropic_mock)
    assert body["system"] == snapshot(
        [{"type": "text", "text": "You are a helpful assistant.", "cache_control": {"type": "ephemeral"}}]
    )


@pytest.mark.asyncio
async def test_multi_turn_conversation(anthropic_mock, anthropic_provider):
    """Test multi-turn conversation message conversion."""
    history = [
        Message(role="user", content="What is 2+2?"),
        Message(role="assistant", content="2+2 equals 4."),
        Message(role="user", content="And 3+3?"),
    ]

    stream = await anthropic_provider.generate("", [], history)
    async for _ in stream:
        pass

    body = get_request_body(anthropic_mock)
    assert body["messages"] == snapshot(
        [
            {"role": "user", "content": [{"type": "text", "text": "What is 2+2?"}]},
            {"role": "assistant", "content": [{"type": "text", "text": "2+2 equals 4."}]},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "And 3+3?", "cache_control": {"type": "ephemeral"}}
                ],
            },
        ]
    )


# =============================================================================
# Tool Calling Tests
# =============================================================================


@pytest.mark.asyncio
async def test_tool_definition(anthropic_mock, anthropic_provider):
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

    stream = await anthropic_provider.generate("", tools, history)
    async for _ in stream:
        pass

    body = get_request_body(anthropic_mock)
    assert body["tools"] == snapshot(
        [
            {
                "name": "add",
                "description": "Add two integers.",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "a": {"type": "integer", "description": "First number"},
                        "b": {"type": "integer", "description": "Second number"},
                    },
                    "required": ["a", "b"],
                },
                "cache_control": {"type": "ephemeral"},
            }
        ]
    )


@pytest.mark.asyncio
async def test_assistant_message_with_tool_call(anthropic_mock, anthropic_provider):
    """Test that assistant messages with tool calls are correctly converted."""
    history = [
        Message(role="user", content="Add 2 and 3"),
        Message(
            role="assistant",
            content="I'll add those numbers for you.",
            tool_calls=[
                ToolCall(
                    id="toolu_abc123",
                    function=ToolCall.FunctionBody(
                        name="add",
                        arguments='{"a": 2, "b": 3}',
                    ),
                )
            ],
        ),
    ]

    stream = await anthropic_provider.generate("", [], history)
    async for _ in stream:
        pass

    body = get_request_body(anthropic_mock)
    assert body["messages"] == snapshot(
        [
            {"role": "user", "content": [{"type": "text", "text": "Add 2 and 3"}]},
            {
                "role": "assistant",
                "content": [
                    {"type": "text", "text": "I'll add those numbers for you."},
                    {
                        "type": "tool_use",
                        "id": "toolu_abc123",
                        "name": "add",
                        "input": {"a": 2, "b": 3},
                        "cache_control": {"type": "ephemeral"},
                    },
                ],
            },
        ]
    )


@pytest.mark.asyncio
async def test_tool_result_message(anthropic_mock, anthropic_provider):
    """Test that tool result messages are correctly converted."""
    history = [
        Message(role="user", content="Add 2 and 3"),
        Message(
            role="assistant",
            content="",
            tool_calls=[
                ToolCall(
                    id="toolu_abc123",
                    function=ToolCall.FunctionBody(name="add", arguments='{"a": 2, "b": 3}'),
                )
            ],
        ),
        Message(role="tool", content="5", tool_call_id="toolu_abc123"),
    ]

    stream = await anthropic_provider.generate("", [], history)
    async for _ in stream:
        pass

    body = get_request_body(anthropic_mock)
    assert body["messages"] == snapshot(
        [
            {"role": "user", "content": [{"type": "text", "text": "Add 2 and 3"}]},
            {
                "role": "assistant",
                "content": [
                    {"type": "text", "text": ""},
                    {"type": "tool_use", "id": "toolu_abc123", "name": "add", "input": {"a": 2, "b": 3}},
                ],
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": "toolu_abc123",
                        "content": [{"type": "text", "text": "5"}],
                        "cache_control": {"type": "ephemeral"},
                    }
                ],
            },
        ]
    )


# =============================================================================
# Thinking/Reasoning Tests
# =============================================================================


@pytest.mark.asyncio
async def test_assistant_message_with_thinking(anthropic_mock, anthropic_provider):
    """Test that assistant messages with thinking blocks are correctly converted."""
    history = [
        Message(role="user", content="What is 2+2?"),
        Message(
            role="assistant",
            content=[
                ThinkPart(think="Let me think about this...", encrypted="sig_abc123"),
                TextPart(text="The answer is 4."),
            ],
        ),
        Message(role="user", content="Thanks!"),
    ]

    stream = await anthropic_provider.generate("", [], history)
    async for _ in stream:
        pass

    body = get_request_body(anthropic_mock)
    assert body["messages"] == snapshot(
        [
            {"role": "user", "content": [{"type": "text", "text": "What is 2+2?"}]},
            {
                "role": "assistant",
                "content": [
                    {"type": "thinking", "thinking": "Let me think about this...", "signature": "sig_abc123"},
                    {"type": "text", "text": "The answer is 4."},
                ],
            },
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Thanks!", "cache_control": {"type": "ephemeral"}}
                ],
            },
        ]
    )


@pytest.mark.asyncio
async def test_thinking_without_signature_is_stripped(anthropic_mock, anthropic_provider):
    """Test that thinking blocks without signatures are stripped."""
    history = [
        Message(role="user", content="Hi"),
        Message(
            role="assistant",
            content=[
                ThinkPart(think="Some thinking without signature"),  # no encrypted field
                TextPart(text="Hello!"),
            ],
        ),
        Message(role="user", content="Bye"),
    ]

    stream = await anthropic_provider.generate("", [], history)
    async for _ in stream:
        pass

    body = get_request_body(anthropic_mock)
    # The thinking block without signature should be stripped
    assert body["messages"] == snapshot(
        [
            {"role": "user", "content": [{"type": "text", "text": "Hi"}]},
            {"role": "assistant", "content": [{"type": "text", "text": "Hello!"}]},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Bye", "cache_control": {"type": "ephemeral"}}
                ],
            },
        ]
    )


# =============================================================================
# Image Content Tests
# =============================================================================


@pytest.mark.asyncio
async def test_image_url_message(anthropic_mock, anthropic_provider):
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

    stream = await anthropic_provider.generate("", [], history)
    async for _ in stream:
        pass

    body = get_request_body(anthropic_mock)
    assert body["messages"] == snapshot(
        [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "What's in this image?"},
                    {
                        "type": "image",
                        "source": {"type": "url", "url": "https://example.com/image.png"},
                        "cache_control": {"type": "ephemeral"},
                    },
                ],
            }
        ]
    )


@pytest.mark.asyncio
async def test_base64_image_message(anthropic_mock, anthropic_provider):
    """Test that base64 image content is correctly converted."""
    from kosong.message import ImageURLPart

    # A minimal valid base64 PNG (1x1 transparent pixel)
    b64_data = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg=="
    data_url = f"data:image/png;base64,{b64_data}"

    history = [
        Message(
            role="user",
            content=[
                TextPart(text="Describe this:"),
                ImageURLPart(image_url=ImageURLPart.ImageURL(url=data_url)),
            ],
        )
    ]

    stream = await anthropic_provider.generate("", [], history)
    async for _ in stream:
        pass

    body = get_request_body(anthropic_mock)
    assert body["messages"] == snapshot(
        [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Describe this:"},
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "data": "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg==",
                            "media_type": "image/png",
                        },
                        "cache_control": {"type": "ephemeral"},
                    },
                ],
            }
        ]
    )


# =============================================================================
# Special Cases
# =============================================================================


@pytest.mark.asyncio
async def test_system_role_message_converted_to_user(anthropic_mock, anthropic_provider):
    """Test that system role messages in history are converted to user messages."""
    history = [
        Message(role="system", content="Additional context here."),
        Message(role="user", content="Hello"),
    ]

    stream = await anthropic_provider.generate("Main system prompt", [], history)
    async for _ in stream:
        pass

    body = get_request_body(anthropic_mock)
    # System messages in history should be wrapped in <system> tags and sent as user
    assert body["messages"] == snapshot(
        [
            {
                "role": "user",
                "content": [{"type": "text", "text": "<system>Additional context here.</system>"}],
            },
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Hello", "cache_control": {"type": "ephemeral"}}
                ],
            },
        ]
    )


@pytest.mark.asyncio
async def test_generation_kwargs(anthropic_mock, anthropic_provider):
    """Test that generation kwargs are correctly passed."""
    provider = anthropic_provider.with_generation_kwargs(
        temperature=0.7,
        top_p=0.9,
        max_tokens=2048,
    )
    history = [Message(role="user", content="Hi")]

    stream = await provider.generate("", [], history)
    async for _ in stream:
        pass

    body = get_request_body(anthropic_mock)
    assert body["temperature"] == snapshot(0.7)
    assert body["top_p"] == snapshot(0.9)
    assert body["max_tokens"] == snapshot(2048)


@pytest.mark.asyncio
async def test_with_thinking_config(anthropic_mock, anthropic_provider):
    """Test that thinking configuration is correctly set."""
    provider = anthropic_provider.with_thinking("high")
    history = [Message(role="user", content="Think hard")]

    stream = await provider.generate("", [], history)
    async for _ in stream:
        pass

    body = get_request_body(anthropic_mock)
    assert body["thinking"] == snapshot({"type": "enabled", "budget_tokens": 32000})


