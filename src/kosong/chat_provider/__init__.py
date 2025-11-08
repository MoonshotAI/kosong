from collections.abc import AsyncIterator, Sequence
from typing import Literal, NamedTuple, Protocol, Self, runtime_checkable

from kosong.message import ContentPart, Message, ToolCall, ToolCallPart
from kosong.tooling import Tool


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

        Raises:
            APIConnectionError: If the API connection fails.
            APITimeoutError: If the API request times out.
            APIStatusError: If the API returns a status code of 4xx or 5xx.
            ChatProviderError: If any other recognized error occurs.
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


class ChatProviderError(Exception):
    """The error raised by a chat provider."""

    def __init__(self, message: str):
        super().__init__(message)


class APIConnectionError(ChatProviderError):
    """The error raised when the API connection fails."""


class APITimeoutError(ChatProviderError):
    """The error raised when the API request times out."""


class APIStatusError(ChatProviderError):
    """The error raised when the API returns a status code of 4xx or 5xx."""

    status_code: int

    def __init__(self, status_code: int, message: str):
        super().__init__(message)
        self.status_code = status_code
