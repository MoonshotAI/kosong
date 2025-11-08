import copy
import uuid
from collections.abc import AsyncIterator, Sequence
from typing import TYPE_CHECKING, Any, Self, cast

import openai
from openai import AsyncOpenAI, AsyncStream, Omit, OpenAIError, omit
from openai.types import CompletionUsage, ReasoningEffort
from openai.types.chat import (
    ChatCompletion,
    ChatCompletionChunk,
    ChatCompletionMessageFunctionToolCall,
    ChatCompletionMessageParam,
    ChatCompletionToolParam,
)

from kosong.base.chat_provider import ChatProvider, StreamedMessagePart, ThinkingEffort, TokenUsage
from kosong.base.message import Message, TextPart, ToolCall, ToolCallPart
from kosong.base.tool import Tool
from kosong.chat_provider import (
    APIConnectionError,
    APIStatusError,
    APITimeoutError,
    ChatProviderError,
)

if TYPE_CHECKING:

    def type_check(openai_legacy: "OpenAILegacy"):
        _: ChatProvider = openai_legacy


class OpenAILegacy(ChatProvider):
    """
    A chat provider that uses the OpenAI Chat Completions API.

    >>> chat_provider = OpenAILegacy(model="gpt-5", api_key="sk-1234567890")
    >>> chat_provider.name
    'openai'
    >>> chat_provider.model_name
    'gpt-5'
    """

    name = "openai"

    def __init__(
        self,
        *,
        model: str,
        api_key: str | None = None,
        base_url: str | None = None,
        stream: bool = True,
        **client_kwargs: Any,
    ):
        self.model = model
        self.stream = stream
        self.client = AsyncOpenAI(
            api_key=api_key,
            base_url=base_url,
            **client_kwargs,
        )
        self._reasoning_effort: ReasoningEffort | Omit = omit

    @property
    def model_name(self) -> str:
        return self.model

    async def generate(
        self,
        system_prompt: str,
        tools: Sequence[Tool],
        history: Sequence[Message],
    ) -> "OpenAILegacyStreamedMessage":
        messages: list[ChatCompletionMessageParam] = []
        if system_prompt:
            # `system` vs `developer`: see `message_to_openai` comments
            messages.append({"role": "system", "content": system_prompt})
        messages.extend(message_to_openai(message) for message in history)

        try:
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                tools=(tool_to_openai(tool) for tool in tools),
                stream=self.stream,
                stream_options={"include_usage": True},
                reasoning_effort=self._reasoning_effort,
            )
            return OpenAILegacyStreamedMessage(response)
        except OpenAIError as e:
            raise convert_error(e) from e

    def with_thinking(self, effort: ThinkingEffort) -> Self:
        new_self = copy.copy(self)
        new_self._reasoning_effort = thinking_effort_to_reasoning_effort(effort)
        return new_self


def message_to_openai(message: Message) -> ChatCompletionMessageParam:
    """Convert a single message to OpenAI message format."""
    # simply `model_dump` because the `Message` type is OpenAI-compatible
    # Note: for openai, `developer` role is more standard, but `system` is still accepted.
    # And many openai-compatible models do not accept `developer` role.
    # So we use `system` role here. OpenAIResponses will use `developer` role.
    # See https://cdn.openai.com/spec/model-spec-2024-05-08.html#definitions
    return cast(ChatCompletionMessageParam, message.model_dump(exclude_none=True))


def tool_to_openai(tool: Tool) -> ChatCompletionToolParam:
    """Convert a single tool to OpenAI tool format."""
    # simply `model_dump` because the `Tool` type is OpenAI-compatible
    return {
        "type": "function",
        "function": {
            "name": tool.name,
            "description": tool.description,
            "parameters": tool.parameters,
        },
    }


class OpenAILegacyStreamedMessage:
    def __init__(self, response: ChatCompletion | AsyncStream[ChatCompletionChunk]):
        if isinstance(response, ChatCompletion):
            self._iter = self._convert_non_stream_response(response)
        else:
            self._iter = self._convert_stream_response(response)
        self._id: str | None = None
        self._usage: CompletionUsage | None = None

    def __aiter__(self) -> AsyncIterator[StreamedMessagePart]:
        return self

    async def __anext__(self) -> StreamedMessagePart:
        return await self._iter.__anext__()

    @property
    def id(self) -> str | None:
        return self._id

    @property
    def usage(self) -> TokenUsage | None:
        if self._usage:
            cached = 0
            other_input = self._usage.prompt_tokens
            if (
                self._usage.prompt_tokens_details
                and self._usage.prompt_tokens_details.cached_tokens
            ):
                cached = self._usage.prompt_tokens_details.cached_tokens
                other_input -= cached
            return TokenUsage(
                input_other=other_input,
                output=self._usage.completion_tokens,
                input_cache_read=cached,
            )
        return None

    async def _convert_non_stream_response(
        self,
        response: ChatCompletion,
    ) -> AsyncIterator[StreamedMessagePart]:
        self._id = response.id
        self._usage = response.usage
        if response.choices[0].message.content:
            yield TextPart(text=response.choices[0].message.content)
        if response.choices[0].message.tool_calls:
            for tool_call in response.choices[0].message.tool_calls:
                if isinstance(tool_call, ChatCompletionMessageFunctionToolCall):
                    yield ToolCall(
                        id=tool_call.id or str(uuid.uuid4()),
                        function=ToolCall.FunctionBody(
                            name=tool_call.function.name,
                            arguments=tool_call.function.arguments,
                        ),
                    )

    async def _convert_stream_response(
        self,
        response: AsyncIterator[ChatCompletionChunk],
    ) -> AsyncIterator[StreamedMessagePart]:
        try:
            async for chunk in response:
                if chunk.id:
                    self._id = chunk.id
                if chunk.usage:
                    self._usage = chunk.usage

                if not chunk.choices:
                    continue

                delta = chunk.choices[0].delta

                # convert text content
                if delta.content:
                    yield TextPart(text=delta.content)

                # convert tool calls
                for tool_call in delta.tool_calls or []:
                    if not tool_call.function:
                        continue

                    if tool_call.function.name:
                        yield ToolCall(
                            id=tool_call.id or str(uuid.uuid4()),
                            function=ToolCall.FunctionBody(
                                name=tool_call.function.name,
                                arguments=tool_call.function.arguments,
                            ),
                        )
                    elif tool_call.function.arguments:
                        yield ToolCallPart(
                            arguments_part=tool_call.function.arguments,
                        )
                    else:
                        # skip empty tool calls
                        pass
        except OpenAIError as e:
            raise convert_error(e) from e


def convert_error(error: OpenAIError) -> ChatProviderError:
    if isinstance(error, openai.APIStatusError):
        return APIStatusError(error.status_code, error.message)
    elif isinstance(error, openai.APIConnectionError):
        return APIConnectionError(error.message)
    elif isinstance(error, openai.APITimeoutError):
        return APITimeoutError(error.message)
    else:
        return ChatProviderError(f"Error: {error}")


def thinking_effort_to_reasoning_effort(effort: ThinkingEffort) -> ReasoningEffort:
    match effort:
        case "off":
            return None
        case "low":
            return "low"
        case "medium":
            return "medium"
        case "high":
            return "high"


if __name__ == "__main__":

    async def _dev_main():
        chat = OpenAILegacy(model="gpt-4o", stream=False)
        system_prompt = "You are a helpful assistant."
        history = [Message(role="user", content="Hello, how are you?")]
        async for part in await chat.generate(system_prompt, [], history):
            print(part.model_dump(exclude_none=True))

        tools = [
            Tool(
                name="get_weather",
                description="Get the weather",
                parameters={
                    "type": "object",
                    "properties": {
                        "city": {
                            "type": "string",
                            "description": "The city to get the weather for.",
                        },
                    },
                },
            )
        ]
        history = [Message(role="user", content="What's the weather in Beijing?")]
        stream = await chat.generate(system_prompt, tools, history)
        async for part in stream:
            print(part.model_dump(exclude_none=True))
        print("usage:", stream.usage)

    import asyncio

    from dotenv import load_dotenv

    load_dotenv()
    asyncio.run(_dev_main())
