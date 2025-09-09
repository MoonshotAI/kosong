from kosong.base.chat_provider import ChatProvider

from .mock import MockChatProvider
from .openai_legacy import OpenAILegacyChatProvider

__all__ = [
    "MockChatProvider",
    "OpenAILegacyChatProvider",
]


def __static_check_types(
    openai: "OpenAILegacyChatProvider",
    mock: "MockChatProvider",
):
    """Use type checking to ensure the types are correct implemented."""
    _: ChatProvider = openai
    _: ChatProvider = mock
