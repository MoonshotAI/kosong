from kosong.base.message import ToolCall
from kosong.base.tool import Tool
from kosong.tooling import HandleResult, ToolResult
from kosong.tooling.error import ToolNotFoundError


class EmptyToolset:
    @property
    def tools(self) -> list[Tool]:
        return []

    def handle(self, tool_call: ToolCall) -> HandleResult:
        return ToolResult(tool_call.id, ToolNotFoundError(tool_call.function.name))
