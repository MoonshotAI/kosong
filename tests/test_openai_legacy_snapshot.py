"""Snapshot tests for OpenAI Legacy (Chat Completions API) chat provider."""

import json
from typing import Any

import pytest
import respx
from httpx import Response
from inline_snapshot import snapshot

from kosong.contrib.chat_provider.openai_legacy import OpenAILegacy
from kosong.message import Message, TextPart, ThinkPart

from snapshot_common import COMMON_CASES, run_test_cases


def make_response() -> dict:
    return {
        "id": "chatcmpl-test123",
        "object": "chat.completion",
        "created": 1234567890,
        "model": "gpt-4.1",
        "choices": [{"index": 0, "message": {"role": "assistant", "content": "Hello"}, "finish_reason": "stop"}],
        "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
    }


TEST_CASES: dict[str, dict[str, Any]] = {**COMMON_CASES}


@pytest.mark.asyncio
async def test_openai_legacy_message_conversion():
    with respx.mock(base_url="https://api.openai.com") as mock:
        mock.post("/v1/chat/completions").mock(return_value=Response(200, json=make_response()))
        provider = OpenAILegacy(model="gpt-4.1", api_key="test-key", stream=False)
        results = await run_test_cases(mock, provider, TEST_CASES, ("messages", "tools"))

        assert results == snapshot(
            {
                "simple_user_message": {
                    "messages": [{"role": "system", "content": "You are helpful."}, {"role": "user", "content": "Hello!"}],
                    "tools": [],
                },
                "multi_turn_conversation": {
                    "messages": [
                        {"role": "user", "content": "What is 2+2?"},
                        {"role": "assistant", "content": "2+2 equals 4."},
                        {"role": "user", "content": "And 3+3?"},
                    ],
                    "tools": [],
                },
                "tool_definition": {
                    "messages": [{"role": "user", "content": "Add 2 and 3"}],
                    "tools": [{"type": "function", "function": {"name": "add", "description": "Add two integers.", "parameters": {"type": "object", "properties": {"a": {"type": "integer", "description": "First number"}, "b": {"type": "integer", "description": "Second number"}}, "required": ["a", "b"]}}}],
                },
                "assistant_with_tool_call": {
                    "messages": [
                        {"role": "user", "content": "Add 2 and 3"},
                        {"role": "assistant", "content": "I'll add those numbers for you.", "tool_calls": [{"id": "call_abc123", "type": "function", "function": {"name": "add", "arguments": '{"a": 2, "b": 3}'}}]},
                    ],
                    "tools": [],
                },
                "tool_result": {
                    "messages": [
                        {"role": "user", "content": "Add 2 and 3"},
                        {"role": "assistant", "content": "", "tool_calls": [{"id": "call_abc123", "type": "function", "function": {"name": "add", "arguments": '{"a": 2, "b": 3}'}}]},
                        {"role": "tool", "content": "5", "tool_call_id": "call_abc123"},
                    ],
                    "tools": [],
                },
                "image_url": {
                    "messages": [{"role": "user", "content": [{"type": "text", "text": "What's in this image?"}, {"type": "image_url", "image_url": {"url": "https://example.com/image.png", "id": None}}]}],
                    "tools": [],
                },
            }
        )


@pytest.mark.asyncio
async def test_openai_legacy_reasoning_content():
    with respx.mock(base_url="https://api.openai.com") as mock:
        mock.post("/v1/chat/completions").mock(return_value=Response(200, json=make_response()))
        provider = OpenAILegacy(model="deepseek-reasoner", api_key="test-key", stream=False, reasoning_key="reasoning_content")
        history = [
            Message(role="user", content="What is 2+2?"),
            Message(role="assistant", content=[ThinkPart(think="Thinking..."), TextPart(text="4.")]),
            Message(role="user", content="Thanks!"),
        ]
        stream = await provider.generate("", [], history)
        async for _ in stream:
            pass
        body = json.loads(mock.calls.last.request.content.decode())
        assert body["messages"] == snapshot([
            {"role": "user", "content": "What is 2+2?"},
            {"role": "assistant", "content": "4.", "reasoning_content": "Thinking..."},
            {"role": "user", "content": "Thanks!"},
        ])


@pytest.mark.asyncio
async def test_openai_legacy_generation_kwargs():
    with respx.mock(base_url="https://api.openai.com") as mock:
        mock.post("/v1/chat/completions").mock(return_value=Response(200, json=make_response()))
        provider = OpenAILegacy(model="gpt-4.1", api_key="test-key", stream=False).with_generation_kwargs(temperature=0.7, max_tokens=2048)
        stream = await provider.generate("", [], [Message(role="user", content="Hi")])
        async for _ in stream:
            pass
        body = json.loads(mock.calls.last.request.content.decode())
        assert (body["temperature"], body["max_tokens"]) == snapshot((0.7, 2048))


@pytest.mark.asyncio
async def test_openai_legacy_with_thinking():
    with respx.mock(base_url="https://api.openai.com") as mock:
        mock.post("/v1/chat/completions").mock(return_value=Response(200, json=make_response()))
        provider = OpenAILegacy(model="gpt-4.1", api_key="test-key", stream=False).with_thinking("high")
        stream = await provider.generate("", [], [Message(role="user", content="Think")])
        async for _ in stream:
            pass
        body = json.loads(mock.calls.last.request.content.decode())
        assert body["reasoning_effort"] == snapshot("high")
