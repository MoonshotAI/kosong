import asyncio

import pytest

from kosong.base.chat_provider import StreamedMessagePart
from kosong.base.message import Message, TextPart
from kosong.chat_provider import APIStatusError
from kosong.chat_provider.chaos import ChaosChatProvider, ChaosConfig
from kosong.chat_provider.mock import MockChatProvider


def test_mock_chat_provider():
    input_parts: list[StreamedMessagePart] = [
        TextPart(text="Hello, world!"),
    ]

    async def generate() -> list[StreamedMessagePart]:
        chat_provider = MockChatProvider(message_parts=input_parts)
        parts: list[StreamedMessagePart] = []
        async for part in await chat_provider.generate(system_prompt="", tools=[], history=[]):
            parts.append(part)
        return parts

    output_parts = asyncio.run(generate())
    assert output_parts == input_parts


@pytest.mark.asyncio
async def test_chaos_chat_provider():
    chat_provider = ChaosChatProvider(
        model="dummy", api_key="sk-1234567890", chaos_config=ChaosConfig(error_probability=1.0)
    )
    for _ in range(3):
        try:
            parts: list[StreamedMessagePart] = []
            async for part in await chat_provider.generate(
                system_prompt="",
                tools=[],
                history=[Message(role="user", content=[TextPart(text="Hello, world!")])],
            ):
                parts.append(part)
            raise AssertionError("Expected APIStatusError")
        except APIStatusError:
            pass
