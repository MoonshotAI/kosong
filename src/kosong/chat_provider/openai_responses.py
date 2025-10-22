import copy
import uuid
from collections.abc import AsyncIterator, Sequence
from typing import TypedDict, Unpack, cast

import openai
from openai import AsyncOpenAI, AsyncStream, OpenAIError
from openai.types.responses import (
    Response,
    ResponseInputItemParam,
    ResponseInputParam,
    ResponseOutputMessageParam,
    ResponseOutputTextParam,
    ResponseReasoningItemParam,
    ResponseStreamEvent,
    ResponseUsage,
    ToolParam,
)
from openai.types.responses.response_function_call_output_item_list_param import (
    ResponseFunctionCallOutputItemListParam,
)
from openai.types.responses.response_input_audio_param import ResponseInputAudioParam
from openai.types.responses.response_input_file_content_param import (
    ResponseInputFileContentParam,
)
from openai.types.responses.response_input_file_param import ResponseInputFileParam
from openai.types.responses.response_input_message_content_list_param import (
    ResponseInputMessageContentListParam,
)
from openai.types.shared.reasoning import Reasoning
from openai.types.shared.reasoning_effort import ReasoningEffort

from kosong.base.chat_provider import StreamedMessagePart, TokenUsage
from kosong.base.message import (
    AudioURLPart,
    ContentPart,
    ImageURLPart,
    Message,
    TextPart,
    ThinkPart,
    ToolCall,
    ToolCallPart,
)
from kosong.base.tool import Tool
from kosong.chat_provider import (
    APIConnectionError,
    APIStatusError,
    APITimeoutError,
    ChatProviderError,
)


class OpenAIResponses:
    """
    A chat provider that uses the OpenAI Responses API.

    Similar to `OpenAILegacy`, but uses `client.responses` under the hood.

    >>> chat_provider = OpenAIResponses(model="gpt-5", api_key="sk-1234567890")
    >>> chat_provider.name
    'openai-responses'
    >>> chat_provider.model_name
    'gpt-5'
    """

    name = "openai-responses"

    def __init__(
        self,
        *,
        model: str,
        api_key: str | None = None,
        base_url: str | None = None,
        stream: bool = True,
        **client_kwargs,
    ):
        self._model = model
        self._stream = stream
        self._client = AsyncOpenAI(
            api_key=api_key,
            base_url=base_url,
            **client_kwargs,
        )

        self._generation_kwargs = {}

    @property
    def model_name(self) -> str:
        return self._model

    async def generate(
        self,
        system_prompt: str,
        tools: Sequence[Tool],
        history: Sequence[Message],
    ) -> "OpenAIResponsesStreamedMessage":
        inputs: ResponseInputParam = []
        if system_prompt:
            inputs.append({"role": "system", "content": system_prompt})
        # The `Message` type is OpenAI-compatible for Responses API `input` messages.

        for m in history:
            inputs.extend(message_to_openai(m))

        generation_kwargs = {}
        generation_kwargs.update(self._generation_kwargs)
        if reasoning_effort := generation_kwargs.pop("reasoning_effort", None):
            generation_kwargs["reasoning"] = Reasoning(
                effort=reasoning_effort,
                summary="auto",
            )
            generation_kwargs["include"] = ["reasoning.encrypted_content"]

        try:
            response = await self._client.responses.create(
                stream=self._stream,
                model=self._model,
                input=inputs,
                tools=[tool_to_openai(tool) for tool in tools],
                store=False,
                **generation_kwargs,
            )
            return OpenAIResponsesStreamedMessage(response)
        except OpenAIError as e:
            raise convert_error(e) from e

    class GenerationKwargs(TypedDict, total=False):
        max_output_tokens: int | None
        max_tool_calls: int | None
        reasoning_effort: ReasoningEffort | None
        temperature: float | None
        top_logprobs: float | None
        top_p: float | None
        user: str | None

    def with_generation_kwargs(self, **kwargs: Unpack[GenerationKwargs]) -> "OpenAIResponses":
        new_self = copy.copy(self)
        new_self._generation_kwargs = kwargs
        return new_self


def tool_to_openai(tool: Tool) -> ToolParam:
    """Convert a single tool to the OpenAI Responses tool format."""
    return {
        "type": "function",
        "name": tool.name,
        "description": tool.description,
        "parameters": tool.parameters,
        "strict": False,
    }


def message_to_openai(message: Message) -> list[ResponseInputItemParam]:
    """Convert a single message to OpenAI Responses input format.

    Rules:
    - role in {user, system, developer, assistant}: map to EasyInputMessageParam
      content: str kept; list[ContentPart] mapped to ResponseInputMessageContentListParam
    - role == tool: map to FunctionCallOutput with call_id and output
    """

    role = message.role
    content = message.content

    # tool role → function_call_output (return value from a prior tool call)
    if role == "tool":
        call_id = message.tool_call_id or ""
        if isinstance(content, str):
            output: str | ResponseFunctionCallOutputItemListParam = content
        else:
            output = _content_parts_to_function_output_items(content)

        return [
            {
                "call_id": call_id,
                "output": output,
                "type": "function_call_output",
            }
        ]

    result: list[ResponseInputItemParam] = []

    # user/system/developer/assistant → message input item
    if isinstance(content, str):
        result.append(
            {
                "role": role,
                "type": "message",
                "content": content,
            }
        )
    elif len(content) > 0:
        # Split into two kinds of blocks: contiguous non-ThinkPart message blocks, and
        # contiguous ThinkPart groups (grouped by the same `encrypted` value)
        pending_parts: list[ContentPart] = []

        def flush_pending_parts() -> None:
            if not pending_parts:
                return
            if role == "assistant":
                # the "id" key is missing by purpose
                result.append(
                    cast(
                        ResponseOutputMessageParam,
                        {
                            "content": _content_parts_to_output_items(pending_parts),
                            "role": role,
                            "type": "message",
                        },
                    )
                )
            else:
                result.append(
                    {
                        "content": _content_parts_to_input_items(pending_parts),
                        "role": role,
                        "type": "message",
                    }
                )
            pending_parts.clear()

        i = 0
        n = len(content)
        while i < n:
            part = content[i]
            if isinstance(part, ThinkPart):
                # Flush accumulated non-reasoning parts first
                flush_pending_parts()
                # Aggregate consecutive ThinkPart items with the same `encrypted` value
                encrypted_value = part.encrypted
                summaries = [{"type": "summary_text", "text": part.think or ""}]
                i += 1
                while i < n:
                    next_part = content[i]
                    if not isinstance(next_part, ThinkPart):
                        break
                    if next_part.encrypted != encrypted_value:
                        break
                    summaries.append({"type": "summary_text", "text": next_part.think or ""})
                    i += 1
                result.append(
                    cast(
                        ResponseReasoningItemParam,
                        {
                            "summary": summaries,
                            "type": "reasoning",
                            "encrypted_content": encrypted_value,
                        },
                    )
                )
            else:
                pending_parts.append(part)
                i += 1

        # Handle remaining trailing non-reasoning parts
        flush_pending_parts()

    for tool_call in message.tool_calls or []:
        result.append(
            {
                "arguments": tool_call.function.arguments or "{}",
                "call_id": tool_call.id,
                "name": tool_call.function.name,
                "type": "function_call",
            }
        )

    return result


def _content_parts_to_input_items(parts: list[ContentPart]) -> ResponseInputMessageContentListParam:
    """Map internal ContentPart list → ResponseInputMessageContentListParam items."""
    items: ResponseInputMessageContentListParam = []
    for part in parts:
        if isinstance(part, TextPart):
            if part.text:
                items.append({"type": "input_text", "text": part.text})
        elif isinstance(part, ImageURLPart):
            # default detail
            url = part.image_url.url
            items.append(
                {
                    "type": "input_image",
                    "detail": "auto",
                    "image_url": url,
                }
            )
        elif isinstance(part, AudioURLPart):
            mapped = _map_audio_url_to_input_item(part.audio_url.url)
            if mapped is not None:
                items.append(mapped)
        else:
            # Unknown content – ignore
            continue
    return items


def _content_parts_to_output_items(parts: list[ContentPart]) -> list[ResponseOutputTextParam]:
    """Map internal ContentPart list → ResponseOutputTextParam list items."""
    items: list[ResponseOutputTextParam] = []
    for part in parts:
        if isinstance(part, TextPart):
            if part.text:
                items.append({"type": "output_text", "text": part.text, "annotations": []})
        else:
            # Unknown content – ignore
            continue
    return items


def _content_parts_to_function_output_items(
    parts: list[ContentPart],
) -> ResponseFunctionCallOutputItemListParam:
    """Map ContentPart list → ResponseFunctionCallOutputItemListParam items."""
    items: ResponseFunctionCallOutputItemListParam = []
    for part in parts:
        if isinstance(part, TextPart):
            if part.text:
                items.append({"type": "input_text", "text": part.text})
        elif isinstance(part, ImageURLPart):
            url = part.image_url.url
            items.append({"type": "input_image", "image_url": url})
        elif isinstance(part, AudioURLPart):
            mapped = _map_audio_url_to_file_content(part.audio_url.url)
            if mapped is not None:
                items.append(mapped)
        else:
            continue
    return items


def _map_audio_url_to_input_item(
    url: str,
) -> ResponseInputAudioParam | ResponseInputFileParam | None:
    """Map audio URL/data URI to an input content item.

    - data URI (audio/mp3|audio/mpeg|audio/wav) → input_audio
    - http(s) URL → input_file with file_url
    """
    if url.startswith("data:audio/"):
        try:
            header, b64 = url.split(",", 1)
            subtype = header.split("/")[1].split(";")[0].lower()
            fmt = "mp3" if subtype in {"mp3", "mpeg"} else ("wav" if subtype == "wav" else None)
            if fmt is None:
                return None
            return {
                "type": "input_audio",
                "input_audio": {
                    "data": b64,
                    "format": fmt,
                },
            }
        except Exception:
            return None
    if url.startswith("http://") or url.startswith("https://"):
        return {"type": "input_file", "file_url": url}
    return None


def _map_audio_url_to_file_content(url: str) -> ResponseInputFileContentParam | None:
    """Map audio URL/data URI to a file content item for function_call_output."""
    if url.startswith("http://") or url.startswith("https://"):
        return {"type": "input_file", "file_url": url}
    if url.startswith("data:audio/"):
        try:
            _, b64 = url.split(",", 1)
            # We can attach filename optionally; Responses accepts file_data only
            return {"type": "input_file", "file_data": b64}
        except Exception:
            return None
    return None


class OpenAIResponsesStreamedMessage:
    def __init__(self, response: Response | AsyncStream[ResponseStreamEvent]):
        if isinstance(response, Response):
            self._iter = self._convert_non_stream_response(response)
        else:
            self._iter = self._convert_stream_response(response)
        self._usage: ResponseUsage | None = None

    def __aiter__(self) -> AsyncIterator[StreamedMessagePart]:
        return self

    async def __anext__(self) -> StreamedMessagePart:
        return await self._iter.__anext__()

    @property
    def usage(self) -> TokenUsage | None:
        if self._usage:
            return TokenUsage(
                input=self._usage.input_tokens,
                output=self._usage.output_tokens,
            )
        return None

    async def _convert_non_stream_response(
        self, response: Response
    ) -> AsyncIterator[StreamedMessagePart]:
        """Convert a non-streaming Responses API result into message parts."""
        for item in response.output:
            if item.type == "message":
                for content in item.content:
                    if content.type == "output_text":
                        yield TextPart(text=content.text)
            elif item.type == "function_call":
                yield ToolCall(
                    id=item.call_id or str(uuid.uuid4()),
                    function=ToolCall.FunctionBody(
                        name=item.name,
                        arguments=item.arguments,
                    ),
                )
            elif item.type == "reasoning":
                for summary in item.summary:
                    yield ThinkPart(
                        think=summary.text,
                        encrypted=item.encrypted_content,
                    )
        self._usage = response.usage

    async def _convert_stream_response(
        self, response: AsyncStream[ResponseStreamEvent]
    ) -> AsyncIterator[StreamedMessagePart]:
        """Convert streaming Responses events into message parts."""
        try:
            async for chunk in response:
                if chunk.type == "response.output_text.delta":
                    yield TextPart(text=chunk.delta)
                elif chunk.type == "response.output_item.added":
                    item = chunk.item
                    if item.type == "function_call":
                        yield ToolCall(
                            id=item.call_id or str(uuid.uuid4()),
                            function=ToolCall.FunctionBody(
                                name=item.name,
                                arguments=item.arguments,
                            ),
                        )
                elif chunk.type == "response.output_item.done":
                    item = chunk.item
                    if item.type == "reasoning":
                        yield ThinkPart(think="", encrypted=item.encrypted_content)
                elif chunk.type == "response.function_call_arguments.delta":
                    yield ToolCallPart(arguments_part=chunk.delta)
                elif chunk.type == "response.reasoning_summary_part.added":
                    yield ThinkPart(think="")
                elif chunk.type == "response.reasoning_summary_text.delta":
                    yield ThinkPart(think=chunk.delta)
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


if __name__ == "__main__":

    async def _dev_main():
        # Non-streaming example
        chat = OpenAIResponses(model="gpt-5", stream=False).with_generation_kwargs(
            reasoning_effort="medium",
        )
        system_prompt = "You are a helpful assistant."
        history = [Message(role="user", content="Hello, how are you?")]

        from kosong.base import generate

        message, usage = await generate(chat, system_prompt, [], history)
        print(message)
        print(usage)
        history.append(message)

        # Streaming example with tools
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
        history.append(Message(role="user", content="What's the weather in Beijing?"))
        message, usage = await generate(chat, system_prompt, tools, history)
        print(message)
        print(usage)
        history.append(message)
        for tool_call in message.tool_calls or []:
            assert tool_call.function.name == "get_weather"
            history.append(Message(role="tool", tool_call_id=tool_call.id, content="Sunny"))
        message, usage = await generate(chat, system_prompt, tools, history)
        print(message)
        print(usage)

    import asyncio

    from dotenv import load_dotenv

    load_dotenv(override=True)
    asyncio.run(_dev_main())
