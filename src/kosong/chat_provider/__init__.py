from typing import TYPE_CHECKING

from kosong.base.chat_provider import ChatProvider

__all__ = [
    "Anthropic",
    "OpenAILegacy",
    "OpenAIResponses",
    "Kimi",
    # for testing
    "MockChatProvider",
    "ChaosChatProvider",
]


if TYPE_CHECKING:

    def type_check(
        openai: "OpenAILegacy",
        openai_responses: "OpenAIResponses",
        kimi: "Kimi",
        anthropic: "Anthropic",
        mock: "MockChatProvider",
        chaos: "ChaosChatProvider",
    ):
        """Use type checking to ensure the types are correct implemented."""
        _: ChatProvider = openai
        _: ChatProvider = openai_responses
        _: ChatProvider = mock
        _: ChatProvider = kimi
        _: ChatProvider = anthropic
        _: ChatProvider = chaos


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


from .anthropic import Anthropic  # noqa: E402
from .chaos import ChaosChatProvider  # noqa: E402
from .kimi import Kimi  # noqa: E402
from .mock import MockChatProvider  # noqa: E402
from .openai_legacy import OpenAILegacy  # noqa: E402
from .openai_responses import OpenAIResponses  # noqa: E402
