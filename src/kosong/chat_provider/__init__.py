from kosong.base.chat_provider import ChatProvider

__all__ = [
    "MockChatProvider",
    "OpenAILegacy",
    "Kimi",
]


def __static_check_types(
    mock: "MockChatProvider",
    openai: "OpenAILegacy",
    kimi: "Kimi",
):
    """Use type checking to ensure the types are correct implemented."""
    _: ChatProvider = openai
    _: ChatProvider = mock
    _: ChatProvider = kimi


class ChatProviderError(Exception):
    """The error raised by a chat provider."""

    def __init__(self, message: str):
        super().__init__(message)


from .kimi import Kimi  # noqa: E402
from .mock import MockChatProvider  # noqa: E402
from .openai_legacy import OpenAILegacy  # noqa: E402
