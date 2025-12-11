"""Snapshot tests for Anthropic chat provider."""

import json
from typing import Any

import pytest
import respx
from httpx import Response
from inline_snapshot import snapshot

from kosong.contrib.chat_provider.anthropic import Anthropic
from kosong.message import ImageURLPart, Message, TextPart, ThinkPart

from snapshot_common import ADD_TOOL, B64_PNG, COMMON_CASES, run_test_cases


def make_response() -> dict:
    return {
        "id": "msg_test_123",
        "type": "message",
        "role": "assistant",
        "model": "claude-sonnet-4-20250514",
        "content": [{"type": "text", "text": "Hello"}],
        "stop_reason": "end_turn",
        "usage": {"input_tokens": 10, "output_tokens": 5},
    }


TEST_CASES: dict[str, dict[str, Any]] = {
    **COMMON_CASES,
    "assistant_with_thinking": {
        "history": [
            Message(role="user", content="What is 2+2?"),
            Message(
                role="assistant",
                content=[ThinkPart(think="Let me think...", encrypted="sig_abc123"), TextPart(text="The answer is 4.")],
            ),
            Message(role="user", content="Thanks!"),
        ],
    },
    "thinking_without_signature_stripped": {
        "history": [
            Message(role="user", content="Hi"),
            Message(role="assistant", content=[ThinkPart(think="Thinking..."), TextPart(text="Hello!")]),
            Message(role="user", content="Bye"),
        ],
    },
    "base64_image": {
        "history": [
            Message(
                role="user",
                content=[TextPart(text="Describe:"), ImageURLPart(image_url=ImageURLPart.ImageURL(url=f"data:image/png;base64,{B64_PNG}"))],
            )
        ],
    },
}


@pytest.mark.asyncio
async def test_anthropic_message_conversion():
    with respx.mock(base_url="https://api.anthropic.com") as mock:
        mock.post("/v1/messages").mock(return_value=Response(200, json=make_response()))
        provider = Anthropic(model="claude-sonnet-4-20250514", api_key="test-key", default_max_tokens=1024, stream=False)
        results = await run_test_cases(mock, provider, TEST_CASES, ("messages", "system", "tools"))

        assert results == snapshot(
            {
                "simple_user_message": {
                    "messages": [{"role": "user", "content": [{"type": "text", "text": "Hello!", "cache_control": {"type": "ephemeral"}}]}],
                    "system": [{"type": "text", "text": "You are helpful.", "cache_control": {"type": "ephemeral"}}], "tools": []},
                "multi_turn_conversation": {
                    "messages": [
                        {"role": "user", "content": [{"type": "text", "text": "What is 2+2?"}]},
                        {"role": "assistant", "content": [{"type": "text", "text": "2+2 equals 4."}]},
                        {"role": "user", "content": [{"type": "text", "text": "And 3+3?", "cache_control": {"type": "ephemeral"}}]},
                    ], "tools": []},
                "tool_definition": {
                    "messages": [{"role": "user", "content": [{"type": "text", "text": "Add 2 and 3", "cache_control": {"type": "ephemeral"}}]}],
                    "tools": [
                        {
                            "name": "add",
                            "description": "Add two integers.",
                            "input_schema": {"type": "object", "properties": {"a": {"type": "integer", "description": "First number"}, "b": {"type": "integer", "description": "Second number"}}, "required": ["a", "b"]},
                            "cache_control": {"type": "ephemeral"},
                        }
                    ],
                },
                "assistant_with_tool_call": {
                    "messages": [
                        {"role": "user", "content": [{"type": "text", "text": "Add 2 and 3"}]},
                        {"role": "assistant", "content": [{"type": "text", "text": "I'll add those numbers for you."}, {"type": "tool_use", "id": "call_abc123", "name": "add", "input": {"a": 2, "b": 3}, "cache_control": {"type": "ephemeral"}}]},
                    ], "tools": []},
                "tool_result": {
                    "messages": [
                        {"role": "user", "content": [{"type": "text", "text": "Add 2 and 3"}]},
                        {"role": "assistant", "content": [{"type": "text", "text": ""}, {"type": "tool_use", "id": "call_abc123", "name": "add", "input": {"a": 2, "b": 3}}]},
                        {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "call_abc123", "content": [{"type": "text", "text": "5"}], "cache_control": {"type": "ephemeral"}}]},
                    ], "tools": []},
                "image_url": {
                    "messages": [{"role": "user", "content": [{"type": "text", "text": "What's in this image?"}, {"type": "image", "source": {"type": "url", "url": "https://example.com/image.png"}, "cache_control": {"type": "ephemeral"}}]}], "tools": []},
                "assistant_with_thinking": {
                    "messages": [
                        {"role": "user", "content": [{"type": "text", "text": "What is 2+2?"}]},
                        {"role": "assistant", "content": [{"type": "thinking", "thinking": "Let me think...", "signature": "sig_abc123"}, {"type": "text", "text": "The answer is 4."}]},
                        {"role": "user", "content": [{"type": "text", "text": "Thanks!", "cache_control": {"type": "ephemeral"}}]},
                    ], "tools": []},
                "thinking_without_signature_stripped": {
                    "messages": [
                        {"role": "user", "content": [{"type": "text", "text": "Hi"}]},
                        {"role": "assistant", "content": [{"type": "text", "text": "Hello!"}]},
                        {"role": "user", "content": [{"type": "text", "text": "Bye", "cache_control": {"type": "ephemeral"}}]},
                    ], "tools": []},
                "base64_image": {
                    "messages": [{"role": "user", "content": [{"type": "text", "text": "Describe:"}, {"type": "image", "source": {"type": "base64", "data": "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg==", "media_type": "image/png"}, "cache_control": {"type": "ephemeral"}}]}], "tools": []},
            }
        )


@pytest.mark.asyncio
async def test_anthropic_generation_kwargs():
    with respx.mock(base_url="https://api.anthropic.com") as mock:
        mock.post("/v1/messages").mock(return_value=Response(200, json=make_response()))
        provider = Anthropic(model="claude-sonnet-4-20250514", api_key="test-key", default_max_tokens=1024, stream=False).with_generation_kwargs(temperature=0.7, top_p=0.9, max_tokens=2048)
        stream = await provider.generate("", [], [Message(role="user", content="Hi")])
        async for _ in stream:
            pass
        body = json.loads(mock.calls.last.request.content.decode())
        assert (body["temperature"], body["top_p"], body["max_tokens"]) == snapshot((0.7, 0.9, 2048))


@pytest.mark.asyncio
async def test_anthropic_with_thinking():
    with respx.mock(base_url="https://api.anthropic.com") as mock:
        mock.post("/v1/messages").mock(return_value=Response(200, json=make_response()))
        provider = Anthropic(model="claude-sonnet-4-20250514", api_key="test-key", default_max_tokens=1024, stream=False).with_thinking("high")
        stream = await provider.generate("", [], [Message(role="user", content="Think")])
        async for _ in stream:
            pass
        body = json.loads(mock.calls.last.request.content.decode())
        assert body["thinking"] == snapshot({"type": "enabled", "budget_tokens": 32000})
