from collections.abc import Sequence
from dataclasses import dataclass

from .message import History
from .tool import Tool


@dataclass
class Context:
    system: str
    """The system prompt."""

    tools: Sequence[Tool]
    """The tools to use."""

    history: History
    """The conversation history."""
