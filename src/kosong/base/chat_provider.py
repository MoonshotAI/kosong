from collections.abc import AsyncIterator, Sequence
from typing import NamedTuple, Protocol, runtime_checkable

from kosong.base.message import ContentPart, Message, ToolCall, ToolCallPart
from kosong.base.tool import Tool


@runtime_checkable
class ChatProvider(Protocol):
    name: str
    """
    The name of the chat provider.
    """

    @property
    def model_name(self) -> str:
        """
        The model name to use for the chat provider.
        """
        ...

    async def generate(
        self,
        system_prompt: str,
        tools: Sequence[Tool],
        history: Sequence[Message],
    ) -> "StreamedMessage":
        """
        Generate a new message based on the given system prompt, tools, and history.
        """
        ...


type StreamedMessagePart = ContentPart | ToolCall | ToolCallPart


@runtime_checkable
class StreamedMessage(Protocol):
    def __aiter__(self) -> AsyncIterator[StreamedMessagePart]:
        """Create an async iterator from the stream."""
        ...

    @property
    def id(self) -> str | None:
        """The ID of the streamed message."""
        ...

    @property
    def usage(self) -> "TokenUsage | None":
        """The usage of the streamed message."""
        ...


class TokenUsage(NamedTuple):
    input: int
    output: int
    # TODO: support `cached`

    @property
    def total(self) -> int:
        return self.input + self.output
