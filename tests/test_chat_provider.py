import asyncio

from kosong.base.chat_provider import StreamedMessagePart
from kosong.base.message import TextPart
from kosong.chat_provider.mock import MockChatProvider


def test_mock_chat_provider():
    input_parts: list[StreamedMessagePart] = [
        TextPart(text="Hello, world!"),
    ]

    async def generate() -> list[StreamedMessagePart]:
        chat_provider = MockChatProvider(message_parts=input_parts)
        parts = []
        async for part in await chat_provider.generate(system_prompt="", tools=[], history=[]):
            parts.append(part)
        return parts

    output_parts = asyncio.run(generate())
    assert output_parts == input_parts
