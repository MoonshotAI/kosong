from abc import ABC
from collections.abc import Sequence
from typing import Literal

from pydantic import BaseModel


class ContentPart(BaseModel, ABC):
    """A part of a message content."""

    type: str


class ToolCall(BaseModel):
    """
    A tool call requested by the assistant.

    >>> ToolCall(
    ...     id="123",
    ...     function=ToolCall.FunctionBody(
    ...         name="function",
    ...         arguments="{}"
    ...     ),
    ... ).model_dump()
    {'type': 'function', 'id': '123', 'function': {'name': 'function', 'arguments': '{}'}}
    """

    class FunctionBody(BaseModel):
        name: str
        arguments: str | None

    type: Literal["function"] = "function"

    id: str
    """The ID of the tool call."""
    function: FunctionBody
    """The function body of the tool call."""


class ToolCallPart(BaseModel):
    """A part of the tool call."""

    arguments_part: str | None = None
    """A part of the arguments of the tool call."""


class Message(BaseModel):
    """A message in a conversation."""

    role: Literal[
        "system",
        "developer",
        "user",
        "assistant",
        "tool",
    ]
    name: str | None = None

    content: str | list[ContentPart]
    """The content of the message."""

    tool_calls: list[ToolCall] | None = None
    """In assistant messages, there can be tool calls."""

    tool_call_id: str | None = None
    """In tool messages, there can be a tool call ID."""

    partial: bool | None = None


type History = Sequence[Message]
"""A history of messages in a conversation."""
