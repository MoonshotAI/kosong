from collections.abc import AsyncIterator

from kosong.base.chat_provider import ChatProvider, StreamedMessage, StreamedMessagePart, TokenUsage
from kosong.base.context import Context


def _assert_types(
    chat_provider: "MockChatProvider",
    streamed_message: "MockStreamedMessage",
):
    """Use type checking to ensure the types are correct implemented."""
    _1: ChatProvider = chat_provider
    _2: StreamedMessage = streamed_message


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

    async def generate(self, context: Context) -> "MockStreamedMessage":
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
    def usage(self) -> TokenUsage | None:
        return None
