from typing import TYPE_CHECKING

from kosong.base.message import ToolCall
from kosong.base.tool import Tool
from kosong.tooling import HandleResult, ToolResult, Toolset
from kosong.tooling.error import ToolNotFoundError

if TYPE_CHECKING:

    def type_check(empty: "EmptyToolset"):
        _: Toolset = empty


class EmptyToolset:
    @property
    def tools(self) -> list[Tool]:
        return []

    def handle(self, tool_call: ToolCall) -> HandleResult:
        return ToolResult(tool_call.id, ToolNotFoundError(tool_call.function.name))
