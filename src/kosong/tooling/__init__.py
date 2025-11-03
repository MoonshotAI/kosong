from abc import ABC, abstractmethod
from asyncio import Future
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Protocol, override, runtime_checkable

import jsonschema
import pydantic
from pydantic import BaseModel
from pydantic.json_schema import GenerateJsonSchema

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

    @property
    def base(self) -> Tool:
        return self

    async def call(self, arguments: JsonType) -> ToolReturnType:
        from kosong.tooling.error import ToolValidateError

        try:
            jsonschema.validate(arguments, self.parameters)
        except jsonschema.ValidationError as e:
            return ToolValidateError(str(e))

        if isinstance(arguments, list):
            ret = await self.__call__(*arguments)
        elif isinstance(arguments, dict):
            ret = await self.__call__(**arguments)
        else:
            ret = await self.__call__(arguments)
        if not isinstance(ret, ToolOk | ToolError):  # pyright: ignore[reportUnnecessaryIsInstance]
            # let's do not trust the return type of the tool
            ret = ToolError(
                message=f"Invalid return type: {type(ret)}",
                brief="Invalid return type",
            )
        return ret

    @abstractmethod
    async def __call__(self, *args: Any, **kwargs: Any) -> ToolReturnType: ...


class _GenerateJsonSchemaNoTitles(GenerateJsonSchema):
    """Custom JSON schema generator that omits titles."""

    @override
    def field_title_should_be_set(self, schema) -> bool:  # pyright: ignore[reportMissingParameterType]
        return False

    @override
    def _update_class_schema(self, json_schema, cls, config) -> None:  # pyright: ignore[reportMissingParameterType]
        super()._update_class_schema(json_schema, cls, config)
        json_schema.pop("title", None)


class CallableTool2[Params: BaseModel](BaseModel, ABC):
    """
    A tool that can be called as a callable object, with type-safe parameters.

    The tool will be called with the arguments provided in the `ToolCall`.
    The arguments must be a JSON object, and will be validated by Pydantic to the `Params` type.
    """

    name: str
    description: str
    params: type[Params]

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._base = Tool(
            name=self.name,
            description=self.description,
            parameters=self.params.model_json_schema(schema_generator=_GenerateJsonSchemaNoTitles),
        )

    @property
    def base(self) -> Tool:
        return self._base

    async def call(self, arguments: JsonType) -> ToolReturnType:
        from kosong.tooling.error import ToolValidateError

        try:
            params = self.params.model_validate(arguments)
        except pydantic.ValidationError as e:
            return ToolValidateError(str(e))

        ret = await self.__call__(params)
        if not isinstance(ret, ToolOk | ToolError):  # pyright: ignore[reportUnnecessaryIsInstance]
            # let's do not trust the return type of the tool
            ret = ToolError(
                message=f"Invalid return type: {type(ret)}",
                brief="Invalid return type",
            )
        return ret

    @abstractmethod
    async def __call__(self, params: Params) -> ToolReturnType: ...


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
