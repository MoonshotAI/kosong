from collections.abc import AsyncIterator, Sequence

from kosong.base.chat_provider import StreamedMessagePart, TokenUsage
from kosong.base.message import Message
from kosong.base.tool import Tool


class MockChatProvider:
    """
    A mock chat provider.
    """

    name = "mock"

    def __init__(
        self,
        message_parts: list[StreamedMessagePart],
    ):
        self._message_parts = message_parts

    @property
    def model_name(self) -> str:
        return "mock"

    async def generate(
        self,
        system_prompt: str,
        tools: Sequence[Tool],
        history: Sequence[Message],
    ) -> "MockStreamedMessage":
        return MockStreamedMessage(self._message_parts)


class MockStreamedMessage:
    def __init__(self, message_parts: list[StreamedMessagePart]):
        self._iter = self._to_stream(message_parts)

    def __aiter__(self) -> AsyncIterator[StreamedMessagePart]:
        return self

    async def __anext__(self) -> StreamedMessagePart:
        return await self._iter.__anext__()

    async def _to_stream(
        self, message_parts: list[StreamedMessagePart]
    ) -> AsyncIterator[StreamedMessagePart]:
        for part in message_parts:
            yield part

    @property
    def id(self) -> str:
        return "mock"

    @property
    def usage(self) -> TokenUsage | None:
        return None
