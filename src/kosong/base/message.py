from abc import ABC
from collections.abc import Sequence
from typing import Any, Literal, override

from pydantic import BaseModel


class MergableMixin:
    def merge_in_place(self, other: Any) -> bool:
        """Merge the other part into the current part. Return True if the merge is successful."""
        return False


class ContentPart(BaseModel, ABC, MergableMixin):
    """A part of a message content."""

    type: str
    ...  # to be added by subclasses


class TextPart(ContentPart):
    """
    >>> TextPart(text="Hello, world!").model_dump()
    {'type': 'text', 'text': 'Hello, world!'}
    """

    type: Literal["text"] = "text"
    text: str

    @override
    def merge_in_place(self, other) -> bool:
        if not isinstance(other, TextPart):
            return False
        self.text += other.text
        return True


class ToolCall(BaseModel, MergableMixin):
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

    @override
    def merge_in_place(self, other) -> bool:
        if not isinstance(other, ToolCallPart):
            return False
        if self.function.arguments is None:
            self.function.arguments = other.arguments_part
        else:
            self.function.arguments += other.arguments_part or ""
        return True


class ToolCallPart(BaseModel, MergableMixin):
    """A part of the tool call."""

    arguments_part: str | None = None
    """A part of the arguments of the tool call."""

    @override
    def merge_in_place(self, other) -> bool:
        if not isinstance(other, ToolCallPart):
            return False
        if self.arguments_part is None:
            self.arguments_part = other.arguments_part
        else:
            self.arguments_part += other.arguments_part or ""
        return True


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
"""A history of messages."""
