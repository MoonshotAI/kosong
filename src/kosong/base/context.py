from collections.abc import Sequence
from typing import NamedTuple

from kosong.base.message import History
from kosong.base.tool import Tool


class Context(NamedTuple):
    """
    An immutable context for a chat.
    It is used to generate a new assistant message.
    """

    system: str
    """The system prompt."""

    tools: Sequence[Tool]
    """The tools to use."""

    history: History
    """The chat history."""
