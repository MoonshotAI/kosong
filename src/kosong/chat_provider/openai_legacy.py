import uuid
from collections.abc import AsyncIterator, Sequence
from typing import cast

from openai import AsyncOpenAI, OpenAIError
from openai.types.chat import (
    ChatCompletionChunk,
    ChatCompletionMessageParam,
    ChatCompletionToolParam,
)
from openai.types.chat.chat_completion_tool_param import FunctionDefinition
from openai.types.completion_usage import CompletionUsage

from kosong.base.chat_provider import StreamedMessagePart, TokenUsage
from kosong.base.message import Message, TextPart, ToolCall, ToolCallPart
from kosong.base.tool import Tool
from kosong.chat_provider import ChatProviderError


class OpenAILegacy:
    """
    A chat provider that uses the OpenAI Chat Completion API.

    >>> chat_provider = OpenAILegacy(model="gpt-5", api_key="sk-1234567890")
    >>> chat_provider.name
    'openai'
    >>> chat_provider.model_name
    'gpt-5'
    """

    name = "openai"

    def __init__(
        self,
        model: str = "gpt-4o",
        api_key: str | None = None,
        base_url: str | None = None,
        **client_kwargs,
    ):
        self._model = model
        self._client = AsyncOpenAI(
            api_key=api_key,
            base_url=base_url,
            **client_kwargs,
        )

    @property
    def model_name(self) -> str:
        return self._model

    async def generate(
        self,
        system_prompt: str,
        tools: Sequence[Tool],
        history: Sequence[Message],
    ) -> "OpenAILegacyStreamedMessage":
        messages: list[ChatCompletionMessageParam] = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.extend(message_to_openai(message) for message in history)

        try:
            response = await self._client.chat.completions.create(
                model=self._model,
                messages=messages,
                tools=(tool_to_openai(tool) for tool in tools),
                stream=True,
                stream_options={"include_usage": True},
                max_completion_tokens=32000,
            )
            return OpenAILegacyStreamedMessage(response)
        except OpenAIError as e:
            raise ChatProviderError(f"Error generating message: {e}") from e


def message_to_openai(message: Message) -> ChatCompletionMessageParam:
    """Convert a single message to OpenAI message format."""
    # simply `model_dump` because the `Message` type is OpenAI-compatible
    return cast(ChatCompletionMessageParam, message.model_dump(exclude_none=True))


def tool_to_openai(tool: Tool) -> ChatCompletionToolParam:
    """Convert a single tool to OpenAI tool format."""
    # simply `model_dump` because the `Tool` type is OpenAI-compatible
    return {
        "type": "function",
        "function": cast(FunctionDefinition, tool.model_dump(exclude_none=True)),
    }


class OpenAILegacyStreamedMessage:
    def __init__(self, response: AsyncIterator[ChatCompletionChunk]):
        self._iter = self._convert_response(response)
        self._usage: CompletionUsage | None = None

    def __aiter__(self) -> AsyncIterator[StreamedMessagePart]:
        return self

    async def __anext__(self) -> StreamedMessagePart:
        return await self._iter.__anext__()

    @property
    def usage(self) -> TokenUsage | None:
        if self._usage:
            return TokenUsage(
                input=self._usage.prompt_tokens,
                output=self._usage.completion_tokens,
            )
        return None

    async def _convert_response(
        self,
        response: AsyncIterator[ChatCompletionChunk],
    ) -> AsyncIterator[StreamedMessagePart]:
        try:
            async for chunk in response:
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

                if chunk.usage:
                    self._usage = chunk.usage
        except OpenAIError as e:
            raise ChatProviderError(f"Error streaming response: {e}") from e


if __name__ == "__main__":

    async def _dev_main():
        chat = OpenAILegacy()
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
        async for part in await chat.generate(system_prompt, tools, history):
            print(part.model_dump(exclude_none=True))

    import asyncio

    from dotenv import load_dotenv

    load_dotenv()
    asyncio.run(_dev_main())
