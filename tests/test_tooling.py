import asyncio
import json

from kosong.base.message import ToolCall
from kosong.base.tool import ParametersType
from kosong.tooling import CallableTool, ToolResult, ToolResultFuture
from kosong.tooling.error import (
    ToolNotFoundError,
    ToolParseError,
    ToolRuntimeError,
    ToolValidateError,
)
from kosong.tooling.simple import SimpleToolset


def test_callable_tool_int_argument():
    class TestTool(CallableTool):
        name: str = "test"
        description: str = "This is a test tool"
        parameters: ParametersType = {
            "type": "integer",
        }

        async def __call__(self, test: int) -> str:
            return f"Test tool called with {test}"

    tool = TestTool()
    assert asyncio.run(tool.call(1)) == "Test tool called with 1"


def test_callable_tool_list_argument():
    class TestTool(CallableTool):
        name: str = "test"
        description: str = "This is a test tool"
        parameters: ParametersType = {
            "type": "array",
            "items": {
                "type": "string",
            },
        }

        async def __call__(self, a: str, b: str) -> str:
            return f"Test tool called with {a} and {b}"

    tool = TestTool()
    assert asyncio.run(tool.call(["a", "b"])) == "Test tool called with a and b"


def test_callable_tool_dict_argument():
    class TestTool(CallableTool):
        name: str = "test"
        description: str = "This is a test tool"
        parameters: ParametersType = {
            "type": "object",
            "properties": {
                "a": {"type": "string"},
                "b": {"type": "integer"},
            },
        }

        async def __call__(self, a: str, b: int) -> str:
            return f"Test tool called with {a} and {b}"

    tool = TestTool()
    assert asyncio.run(tool.call({"a": "a", "b": 1})) == "Test tool called with a and 1"


def test_simple_toolset():
    class PlusTool(CallableTool):
        name: str = "plus"
        description: str = "This is a plus tool"
        parameters: ParametersType = {
            "type": "object",
            "properties": {
                "a": {"type": "integer"},
                "b": {"type": "integer"},
            },
            "required": ["a", "b"],
        }

        async def __call__(self, a: int, b: int) -> str:
            return str(a + b)

    class CompareTool(CallableTool):
        name: str = "compare"
        description: str = "This is a compare tool"
        parameters: ParametersType = {
            "type": "object",
            "properties": {
                "a": {"type": "integer"},
                "b": {"type": "integer"},
            },
            "required": ["a", "b"],
        }

        async def __call__(self, a: int, b: int) -> str:
            return "greater" if a > b else "less" if a < b else "equal"

    class RaiseTool(CallableTool):
        name: str = "raise"
        description: str = "This is a raise tool"
        parameters: ParametersType = {
            "type": "object",
            "properties": {},
        }

        async def __call__(self) -> str:
            raise Exception("test exception")

    toolset = SimpleToolset([PlusTool()])
    toolset += CompareTool()
    toolset += RaiseTool()
    assert toolset.tools[0].name == "plus"
    assert toolset.tools[1].name == "compare"
    assert toolset.tools[2].name == "raise"

    tool_calls = [
        ToolCall(
            id="1",
            function=ToolCall.FunctionBody(
                name="plus",
                arguments=json.dumps({"a": 1, "b": 2}),
            ),
        ),
        ToolCall(
            id="2",
            function=ToolCall.FunctionBody(
                name="compare",
                arguments='{"a": 1, b: 2}',
            ),
        ),
        ToolCall(
            id="3",
            function=ToolCall.FunctionBody(
                name="plus",
                arguments='{"a": 1}',
            ),
        ),
        ToolCall(
            id="4",
            function=ToolCall.FunctionBody(
                name="raise",
                arguments=None,
            ),
        ),
        ToolCall(
            id="5",
            function=ToolCall.FunctionBody(
                name="not_found",
                arguments=None,
            ),
        ),
    ]

    async def run() -> list[ToolResult]:
        futures: list[ToolResultFuture] = []
        for tool_call in tool_calls:
            result = toolset.handle(tool_call)
            if isinstance(result, ToolResult):
                future = ToolResultFuture()
                future.set_result(result)
                futures.append(future)
            else:
                futures.append(result)
        return await asyncio.gather(*futures)

    results = asyncio.run(run())
    assert results[0].tool_call_id == "1"
    assert results[0].result == "3"
    assert isinstance(results[1].result, ToolParseError)
    assert isinstance(results[2].result, ToolValidateError)
    assert isinstance(results[3].result, ToolRuntimeError)
    assert isinstance(results[4].result, ToolNotFoundError)
