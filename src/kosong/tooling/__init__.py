from abc import ABC, abstractmethod
from asyncio import Future
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Protocol, Self, override, runtime_checkable

import jsonschema
import pydantic
from pydantic import BaseModel, model_validator
from pydantic.json_schema import GenerateJsonSchema

from kosong.message import ContentPart, ToolCall
from kosong.utils.typing import JsonType

type ParametersType = dict[str, Any]


class Tool(BaseModel):
    """The definition of a tool that can be recognized by the model."""

    name: str
    """The name of the tool."""

    description: str
    """The description of the tool."""

    parameters: ParametersType
    """The parameters of the tool, in JSON Schema format."""

    @model_validator(mode="after")
    def _validate_parameters(self) -> Self:
        jsonschema.validate(self.parameters, jsonschema.Draft202012Validator.META_SCHEMA)
        return self


@dataclass(frozen=True, kw_only=True, slots=True)
class ToolOk:
    """The successful output returned by a tool."""

    output: str | ContentPart | Sequence[ContentPart]
    """The output content returned by the tool."""
    message: str = ""
    """An explanatory message to be given to the model."""
    brief: str = ""
    """A brief message to be shown to the user."""


@dataclass(frozen=True, kw_only=True, slots=True)
class ToolError:
    """The error returned by a tool. This is not an exception."""

    output: str = ""
    """The output content returned by the tool."""
    message: str
    """An error message to be given to the model."""
    brief: str
    """A brief message to be shown to the user."""


type ToolReturnType = ToolOk | ToolError
"""The return type of a callable tool."""


class CallableTool(Tool, ABC):
    """
    The abstract base class of tools that can be called as callables.

    The tool will be called with the arguments provided in the `ToolCall`.
    If the arguments are given as a JSON array, it will be unpacked into positional arguments.
    If the arguments are given as a JSON object, it will be unpacked into keyword arguments.
    Otherwise, the arguments will be passed as a single argument.
    """

    @property
    def base(self) -> Tool:
        """The base tool definition."""
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
    async def __call__(self, *args: Any, **kwargs: Any) -> ToolReturnType:
        """
        @public

        The implementation of the callable tool.
        """
        ...


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
    The abstract base class of tools that can be called as callables, with typed parameters.

    The tool will be called with the arguments provided in the `ToolCall`.
    The arguments must be a JSON object, and will be validated by Pydantic to the `Params` type.
    """

    name: str
    """The name of the tool."""
    description: str
    """The description of the tool."""
    params: type[Params]
    """The Pydantic model type of the tool parameters."""

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._base = Tool(
            name=self.name,
            description=self.description,
            parameters=self.params.model_json_schema(schema_generator=_GenerateJsonSchemaNoTitles),
        )

    @property
    def base(self) -> Tool:
        """The base tool definition."""
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
    async def __call__(self, params: Params) -> ToolReturnType:
        """
        @public

        The implementation of the callable tool.
        """
        ...


@dataclass(frozen=True)
class ToolResult:
    """The result of a tool call."""

    tool_call_id: str
    """The ID of the tool call."""
    result: ToolReturnType
    """The actual return value of the tool call."""


ToolResultFuture = Future[ToolResult]
type HandleResult = ToolResultFuture | ToolResult


@runtime_checkable
class Toolset(Protocol):
    """
    The interface of toolsets that can register tools and handle tool calls.
    """

    @property
    def tools(self) -> list[Tool]:
        """The list of tool definitions registered in this toolset."""
        ...

    def handle(self, tool_call: ToolCall) -> HandleResult:
        """
        Handle a tool call.
        The result of the tool call, or the async future of the result, should be returned.
        The result should be a `ToolReturnType`, which means `ToolOk` or `ToolError`.

        This method MUST NOT do any blocking operations because it will be called during
        consuming the chat response stream.
        This method MUST NOT raise any exception except for `asyncio.CancelledError`. Any other
        error should be returned as a `ToolError`.
        """
        ...
