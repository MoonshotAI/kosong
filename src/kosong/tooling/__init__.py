from abc import ABC, abstractmethod
from asyncio import Future
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol, final, runtime_checkable

from kosong.base.message import ContentPart, ToolCall
from kosong.base.tool import Tool
from kosong.utils.typing import JsonType

__all__ = [
    "ToolOk",
    "ToolError",
    "ToolReturnType",
    "CallableTool",
    "ToolResult",
    "ToolResultFuture",
    "HandleResult",
    "Toolset",
    "EmptyToolset",
    "SimpleToolset",
]


@dataclass(frozen=True, kw_only=True)
class ToolOk:
    output: str | ContentPart | Sequence[ContentPart]
    """The output content returned by the tool."""
    message: str = ""
    """An explanatory message to be given to the model."""
    brief: str = ""
    """A brief message to be shown to the user."""


@dataclass(frozen=True, kw_only=True)
class ToolError:
    """The error returned by a tool. This is not an exception."""

    output: str = ""
    """The output content returned by the tool."""
    message: str
    """An error message to be given to the model."""
    brief: str
    """A brief message to be shown to the user."""


type ToolReturnType = ToolOk | ToolError


class CallableTool(Tool, ABC):
    """
    A tool that can be called as a callable object.

    The tool will be called with the arguments provided in the `ToolCall`.
    If the arguments are given as a JSON array, it will be unpacked into positional arguments.
    If the arguments are given as a JSON object, it will be unpacked into keyword arguments.
    Otherwise, the arguments will be passed as a single argument.
    """

    @final
    async def call(self, arguments: JsonType) -> ToolReturnType:
        if isinstance(arguments, list):
            ret = await self.__call__(*arguments)
        elif isinstance(arguments, dict):
            ret = await self.__call__(**arguments)
        else:
            ret = await self.__call__(arguments)
        if not isinstance(ret, ToolOk | ToolError):
            # let's do not trust the return type of the tool
            ret = ToolError(
                message=f"Invalid return type: {type(ret)}",
                brief="Invalid return type",
            )
        return ret

    @abstractmethod
    async def __call__(self, *args, **kwargs) -> ToolReturnType: ...


@dataclass(frozen=True)
class ToolResult:
    tool_call_id: str
    result: ToolReturnType


ToolResultFuture = Future[ToolResult]
type HandleResult = ToolResultFuture | ToolResult


@runtime_checkable
class Toolset(Protocol):
    """
    An abstraction of a toolset that can register tools and handle tool calls.
    """

    @property
    def tools(self) -> list[Tool]: ...

    def handle(self, tool_call: ToolCall) -> HandleResult:
        """
        Handle a tool call.
        The result of the tool call, or the async future of the result, should be returned.
        The result should be a `ToolReturnType`, which means `ToolOk` or `ToolError`.

        This method MUST NOT do any blocking operations because it will be called during
        consuming the chat response stream.
        This method MUST NOT raise any exception except for asyncio.CancelledError. Any other
        error should be returned as a `ToolError`.
        """
        ...


from .empty import EmptyToolset  # noqa: E402
from .simple import SimpleToolset  # noqa: E402


def __static_check_types(
    empty: "EmptyToolset",
    simple: "SimpleToolset",
):
    _: Toolset = empty
    _: Toolset = simple
