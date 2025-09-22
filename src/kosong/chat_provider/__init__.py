from kosong.base.chat_provider import ChatProvider

__all__ = [
    "OpenAILegacy",
    "Kimi",
    # for testing
    "MockChatProvider",
    "ChaosChatProvider",
]


def __static_check_types(
    openai: "OpenAILegacy",
    kimi: "Kimi",
    mock: "MockChatProvider",
    chaos: "ChaosChatProvider",
):
    """Use type checking to ensure the types are correct implemented."""
    _: ChatProvider = openai
    _: ChatProvider = mock
    _: ChatProvider = kimi
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


from .chaos import ChaosChatProvider  # noqa: E402
from .kimi import Kimi  # noqa: E402
from .mock import MockChatProvider  # noqa: E402
from .openai_legacy import OpenAILegacy  # noqa: E402
