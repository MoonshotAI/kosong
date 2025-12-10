"""
Snapshot tests for OpenAI Responses API chat provider.

These tests capture the HTTP requests sent to the OpenAI Responses API and verify
that message conversion is correct using inline snapshots.
"""

import json

import pytest
import respx
from httpx import Response
from inline_snapshot import snapshot

from kosong.contrib.chat_provider.openai_responses import OpenAIResponses
from kosong.message import Message, TextPart, ThinkPart, ToolCall
from kosong.tooling import Tool


def make_responses_api_response(
    text: str = "Hello",
    function_calls: list[dict] | None = None,
) -> dict:
    """Helper to create a minimal valid OpenAI Responses API response."""
    output = []
    if text:
        output.append(
            {
                "type": "message",
                "id": "msg_test123",
                "role": "assistant",
                "content": [{"type": "output_text", "text": text, "annotations": []}],
            }
        )
    if function_calls:
        for fc in function_calls:
            output.append(
                {
                    "type": "function_call",
                    "id": fc.get("id", "call_test"),
                    "call_id": fc.get("call_id", "call_test"),
                    "name": fc["name"],
                    "arguments": fc.get("arguments", "{}"),
                }
            )

    return {
        "id": "resp_test123",
        "object": "response",
        "created_at": 1234567890,
        "status": "completed",
        "model": "gpt-4o",
        "output": output,
        "usage": {"input_tokens": 10, "output_tokens": 5, "total_tokens": 15},
    }


@pytest.fixture
def responses_mock():
    """Mock OpenAI Responses API and capture requests."""
    with respx.mock(base_url="https://api.openai.com") as mock:
        mock.post("/v1/responses").mock(
            return_value=Response(200, json=make_responses_api_response())
        )
        yield mock


@pytest.fixture
def responses_provider():
    """Create an OpenAI Responses provider for testing."""
    return OpenAIResponses(
        model="gpt-4o",
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
async def test_simple_user_message(responses_mock, responses_provider):
    """Test that a simple user message is correctly converted."""
    history = [Message(role="user", content="Hello, GPT!")]

    stream = await responses_provider.generate("You are helpful.", [], history)
    async for _ in stream:
        pass

    body = get_request_body(responses_mock)
    # OpenAI Responses API uses "developer" role for system prompts
    assert body["input"] == snapshot(
        [
            {"role": "developer", "content": "You are helpful."},
            {"content": [{"type": "input_text", "text": "Hello, GPT!"}], "role": "user", "type": "message"},
        ]
    )


@pytest.mark.asyncio
async def test_multi_turn_conversation(responses_mock, responses_provider):
    """Test multi-turn conversation message conversion."""
    history = [
        Message(role="user", content="What is 2+2?"),
        Message(role="assistant", content="2+2 equals 4."),
        Message(role="user", content="And 3+3?"),
    ]

    stream = await responses_provider.generate("", [], history)
    async for _ in stream:
        pass

    body = get_request_body(responses_mock)
    assert body["input"] == snapshot(
        [
            {"content": [{"type": "input_text", "text": "What is 2+2?"}], "role": "user", "type": "message"},
            {
                "content": [{"type": "output_text", "text": "2+2 equals 4.", "annotations": []}],
                "role": "assistant",
                "type": "message",
            },
            {"content": [{"type": "input_text", "text": "And 3+3?"}], "role": "user", "type": "message"},
        ]
    )


# =============================================================================
# Tool Calling Tests
# =============================================================================


@pytest.mark.asyncio
async def test_tool_definition(responses_mock, responses_provider):
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

    stream = await responses_provider.generate("", tools, history)
    async for _ in stream:
        pass

    body = get_request_body(responses_mock)
    assert body["tools"] == snapshot(
        [
            {
                "type": "function",
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
                "strict": False,
            }
        ]
    )


@pytest.mark.asyncio
async def test_assistant_message_with_tool_call(responses_mock, responses_provider):
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

    stream = await responses_provider.generate("", [], history)
    async for _ in stream:
        pass

    body = get_request_body(responses_mock)
    assert body["input"] == snapshot(
        [
            {"content": [{"type": "input_text", "text": "Add 2 and 3"}], "role": "user", "type": "message"},
            {
                "content": [{"type": "output_text", "text": "I'll add those numbers for you.", "annotations": []}],
                "role": "assistant",
                "type": "message",
            },
            {"arguments": '{"a": 2, "b": 3}', "call_id": "call_abc123", "name": "add", "type": "function_call"},
        ]
    )


@pytest.mark.asyncio
async def test_tool_result_message(responses_mock, responses_provider):
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

    stream = await responses_provider.generate("", [], history)
    async for _ in stream:
        pass

    body = get_request_body(responses_mock)
    # Tool results in Responses API use function_call_output
    assert body["input"] == snapshot(
        [
            {"content": [{"type": "input_text", "text": "Add 2 and 3"}], "role": "user", "type": "message"},
            {"content": [], "role": "assistant", "type": "message"},
            {"arguments": '{"a": 2, "b": 3}', "call_id": "call_abc123", "name": "add", "type": "function_call"},
            {"call_id": "call_abc123", "output": [{"type": "input_text", "text": "5"}], "type": "function_call_output"},
        ]
    )


# =============================================================================
# Reasoning/Thinking Tests
# =============================================================================


@pytest.mark.asyncio
async def test_assistant_message_with_reasoning(responses_mock, responses_provider):
    """Test that assistant messages with reasoning blocks are correctly converted."""
    history = [
        Message(role="user", content="What is 2+2?"),
        Message(
            role="assistant",
            content=[
                ThinkPart(think="Let me think step by step...", encrypted="enc_abc123"),
                TextPart(text="The answer is 4."),
            ],
        ),
        Message(role="user", content="Thanks!"),
    ]

    stream = await responses_provider.generate("", [], history)
    async for _ in stream:
        pass

    body = get_request_body(responses_mock)
    assert body["input"] == snapshot(
        [
            {"content": [{"type": "input_text", "text": "What is 2+2?"}], "role": "user", "type": "message"},
            {
                "summary": [{"type": "summary_text", "text": "Let me think step by step..."}],
                "type": "reasoning",
                "encrypted_content": "enc_abc123",
            },
            {
                "content": [{"type": "output_text", "text": "The answer is 4.", "annotations": []}],
                "role": "assistant",
                "type": "message",
            },
            {"content": [{"type": "input_text", "text": "Thanks!"}], "role": "user", "type": "message"},
        ]
    )


# =============================================================================
# Image Content Tests
# =============================================================================


@pytest.mark.asyncio
async def test_image_url_message(responses_mock, responses_provider):
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

    stream = await responses_provider.generate("", [], history)
    async for _ in stream:
        pass

    body = get_request_body(responses_mock)
    assert body["input"] == snapshot(
        [
            {
                "content": [
                    {"type": "input_text", "text": "What's in this image?"},
                    {"type": "input_image", "detail": "auto", "image_url": "https://example.com/image.png"},
                ],
                "role": "user",
                "type": "message",
            }
        ]
    )


# =============================================================================
# Generation Kwargs Tests
# =============================================================================


@pytest.mark.asyncio
async def test_generation_kwargs(responses_mock, responses_provider):
    """Test that generation kwargs are correctly passed."""
    provider = responses_provider.with_generation_kwargs(
        temperature=0.7,
        top_p=0.9,
        max_output_tokens=2048,
    )
    history = [Message(role="user", content="Hi")]

    stream = await provider.generate("", [], history)
    async for _ in stream:
        pass

    body = get_request_body(responses_mock)
    assert body["temperature"] == snapshot(0.7)
    assert body["top_p"] == snapshot(0.9)
    assert body["max_output_tokens"] == snapshot(2048)


@pytest.mark.asyncio
async def test_with_thinking_sets_reasoning(responses_mock, responses_provider):
    """Test that with_thinking sets reasoning configuration correctly."""
    provider = responses_provider.with_thinking("high")
    history = [Message(role="user", content="Think hard")]

    stream = await provider.generate("", [], history)
    async for _ in stream:
        pass

    body = get_request_body(responses_mock)
    assert body["reasoning"] == snapshot({"effort": "high", "summary": "auto"})
