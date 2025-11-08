from collections.abc import AsyncIterator, Sequence
from typing import Literal, NamedTuple, Protocol, Self, runtime_checkable

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

    def with_thinking(self, effort: "ThinkingEffort") -> Self:
        """
        Return a copy of self configured with the given thinking effort.
        If the chat provider does not support thinking, simply return a copy of self.
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
    input_other: int
    output: int
    input_cache_read: int = 0
    input_cache_creation: int = 0
    """For now, only Anthropic API supports this."""

    @property
    def total(self) -> int:
        return self.input + self.output

    @property
    def input(self) -> int:
        """Total input tokens, including cached and uncached tokens"""
        return self.input_other + self.input_cache_read + self.input_cache_creation


type ThinkingEffort = Literal["off", "low", "medium", "high"]
