from abc import ABC, abstractmethod
from asyncio import Future
from collections.abc import Sequence
from typing import NamedTuple, Protocol, final, runtime_checkable

from kosong.base.message import ContentPart, ToolCall
from kosong.base.tool import Tool
from kosong.tooling.error import ToolError
from kosong.utils.typing import JsonType

__all__ = [
    "ToolReturnType",
    "CallableTool",
    "ToolResult",
    "ToolResultFuture",
    "Toolset",
    "EmptyToolset",
    "SimpleToolset",
]

type ToolReturnType = str | ContentPart | Sequence[ContentPart]


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
            return await self.__call__(*arguments)
        elif isinstance(arguments, dict):
            return await self.__call__(**arguments)
        else:
            return await self.__call__(arguments)

    @abstractmethod
    async def __call__(self, *args, **kwargs) -> ToolReturnType: ...


class ToolResult(NamedTuple):
    tool_call_id: str
    result: ToolReturnType | ToolError


ToolResultFuture = Future[ToolResult]


@runtime_checkable
class Toolset(Protocol):
    """
    An abstraction of a toolset that can register tools and handle tool calls.
    """

    @property
    def tools(self) -> list[Tool]: ...

    def handle(self, tool_call: ToolCall, future: ToolResultFuture):
        """
        Handle a tool call.
        The result of the tool call should be set to the future asynchronously.
        The result can be either a `ToolReturnType` or a `ToolError`.

        This method MUST NOT do any blocking operations because it will be called during
        consuming the chat response stream.
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
