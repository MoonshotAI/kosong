import asyncio
import inspect
import json
from collections.abc import Iterable
from typing import TYPE_CHECKING, Any, Self

from kosong.message import ToolCall
from kosong.tooling import (
    CallableTool,
    CallableTool2,
    HandleResult,
    Tool,
    ToolResult,
    ToolReturnType,
    Toolset,
)
from kosong.tooling.error import (
    ToolNotFoundError,
    ToolParseError,
    ToolRuntimeError,
)
from kosong.utils.typing import JsonType

if TYPE_CHECKING:

    def type_check(
        simple: "SimpleToolset",
    ):
        _: Toolset = simple


type ToolType = CallableTool | CallableTool2[Any]


class SimpleToolset:
    """A simple toolset that can handle tool calls concurrently."""

    _tool_dict: dict[str, ToolType]

    def __init__(self, tools: Iterable[ToolType] | None = None):
        self._tool_dict = {}
        if tools:
            for tool in tools:
                self += tool

    def __iadd__(self, tool: ToolType) -> Self:
        return_annotation = inspect.signature(tool.__call__).return_annotation
        if return_annotation is not ToolReturnType:
            raise TypeError(
                f"Expected tool `{tool.name}` to return `ToolReturnType`, "
                f"but got `{return_annotation}`"
            )
        self._tool_dict[tool.name] = tool
        return self

    def __add__(self, tool: ToolType) -> "SimpleToolset":
        new_toolset = SimpleToolset()
        new_toolset._tool_dict = self._tool_dict.copy()
        new_toolset += tool
        return new_toolset

    @property
    def tools(self) -> list[Tool]:
        return [tool.base for tool in self._tool_dict.values()]

    def handle(self, tool_call: ToolCall) -> HandleResult:
        if tool_call.function.name not in self._tool_dict:
            return ToolResult(
                tool_call.id,
                ToolNotFoundError(tool_call.function.name),
            )

        tool = self._tool_dict[tool_call.function.name]

        try:
            arguments: JsonType = json.loads(tool_call.function.arguments or "{}")
        except json.JSONDecodeError as e:
            return ToolResult(tool_call.id, ToolParseError(str(e)))

        async def _call():
            try:
                ret = await tool.call(arguments)
                return ToolResult(tool_call.id, ret)
            except Exception as e:
                return ToolResult(tool_call.id, ToolRuntimeError(str(e)))

        return asyncio.create_task(_call())
