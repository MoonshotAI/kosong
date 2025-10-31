import copy
import json
from collections.abc import AsyncIterator, Mapping, Sequence
from typing import Any, Literal, TypedDict, Unpack, cast

from anthropic import (
    AnthropicError,
    AsyncAnthropic,
    AsyncStream,
    omit,
)
from anthropic import (
    APIConnectionError as AnthropicAPIConnectionError,
)
from anthropic import (
    APIStatusError as AnthropicAPIStatusError,
)
from anthropic import (
    APITimeoutError as AnthropicAPITimeoutError,
)
from anthropic import (
    AuthenticationError as AnthropicAuthenticationError,
)
from anthropic import (
    PermissionDeniedError as AnthropicPermissionDeniedError,
)
from anthropic import (
    RateLimitError as AnthropicRateLimitError,
)
from anthropic.lib.streaming import MessageStopEvent
from anthropic.types import (
    CacheControlEphemeralParam,
    ContentBlockParam,
    ImageBlockParam,
    MessageDeltaEvent,
    MessageParam,
    MessageStartEvent,
    RawContentBlockDeltaEvent,
    RawContentBlockStartEvent,
    RawMessageStreamEvent,
    TextBlockParam,
    ThinkingBlockParam,
    ThinkingConfigParam,
    ToolChoiceParam,
    ToolParam,
    ToolResultBlockParam,
    ToolUseBlockParam,
    URLImageSourceParam,
    Usage,
)
from anthropic.types import (
    Message as AnthropicMessage,
)
from anthropic.types.tool_result_block_param import Content as ToolResultContent

from kosong.base.chat_provider import StreamedMessagePart, TokenUsage
from kosong.base.message import (
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

type MessagePayload = tuple[str | None, list[MessageParam]]

type BetaFeatures = Literal["interleaved-thinking-2025-05-14"]


class Anthropic:
    """
    Chat provider backed by Anthropic's Messages API.
    """

    name = "anthropic"

    def __init__(
        self,
        *,
        model: str,
        api_key: str | None = None,
        base_url: str | None = None,
        stream: bool = True,
        # Must provide a max_tokens. Can be overridden by .with_generation_kwargs()
        default_max_tokens: int,
        **client_kwargs: Any,
    ):
        self._model = model
        self._stream = stream
        self._client = AsyncAnthropic(api_key=api_key, base_url=base_url, **client_kwargs)
        self._default_max_tokens = default_max_tokens
        self._generation_kwargs: Mapping[str, Any] = {}

    @property
    def model_name(self) -> str:
        return self._model

    async def generate(
        self,
        system_prompt: str,
        tools: Sequence[Tool],
        history: Sequence[Message],
    ) -> "AnthropicStreamedMessage":
        # https://docs.claude.com/en/api/messages#body-messages
        # Anthropic API does not support system roles, but just a system prompt.
        system = (
            [
                TextBlockParam(
                    text=system_prompt,
                    type="text",
                    cache_control=CacheControlEphemeralParam(type="ephemeral"),
                )
            ]
            if system_prompt
            else omit
        )
        messages: list[MessageParam] = []
        for m in history:
            messages.append(message_to_anthropic(m))
        if messages:
            last_message = messages[-1]
            last_content = last_message["content"]

            # inject cache control in the last content.
            # A maximum of 4 blocks with cache_control may be provided.
            if isinstance(last_content, list) and last_content:
                content_blocks = cast(list[ContentBlockParam], last_content)
                last_block = content_blocks[-1]
                if "cache_control" in last_block:  # noqa: SIM102
                    last_block["cache_control"] = CacheControlEphemeralParam(type="ephemeral")
        generation_kwargs: dict[str, Any] = {
            "max_tokens": self._default_max_tokens,
        }
        generation_kwargs.update(self._generation_kwargs)
        betas = generation_kwargs.pop("beta_features", [])
        extra_headers = {
            **{"anthropic-beta": ",".join(str(e) for e in betas)},
            **(generation_kwargs.pop("extra_headers", {})),
        }

        try:
            response = await self._client.messages.create(
                model=self._model,
                messages=messages,
                system=system,
                tools=[tool_to_anthropic(tool) for tool in tools],
                stream=self._stream,
                extra_headers=extra_headers,
                **generation_kwargs,
            )
            return AnthropicStreamedMessage(response)
        except AnthropicError as e:
            raise convert_error(e) from e

    class GenerationKwargs(TypedDict, total=False):
        max_tokens: int | None
        temperature: float | None
        top_k: int | None
        top_p: float | None
        # e.g., {"type": "enabled", "budget_tokens": 1024}
        thinking: ThinkingConfigParam | None
        # e.g., {"type": "auto", "disable_parallel_tool_use": True}
        tool_choice: ToolChoiceParam | None

        beta_features: list[BetaFeatures] | None
        extra_headers: Mapping[str, str] | None

    def with_generation_kwargs(self, **kwargs: Unpack[GenerationKwargs]) -> "Anthropic":
        new_self = copy.copy(self)
        new_self._generation_kwargs = kwargs
        return new_self


class AnthropicStreamedMessage:
    def __init__(self, response: AnthropicMessage | AsyncStream[RawMessageStreamEvent]):
        if isinstance(response, AnthropicMessage):
            self._iter = self._convert_non_stream_response(response)
        else:
            self._iter = self._convert_stream_response(response)
        self._id: str | None = None
        self._usage: Usage | None = None

    def __aiter__(self) -> AsyncIterator[StreamedMessagePart]:
        return self

    async def __anext__(self) -> StreamedMessagePart:
        return await self._iter.__anext__()

    @property
    def id(self) -> str | None:
        return self._id

    @property
    def usage(self) -> TokenUsage | None:
        if self._usage is None:
            return None
        return TokenUsage(input=self._usage.input_tokens, output=self._usage.output_tokens)

    async def _convert_non_stream_response(
        self,
        response: AnthropicMessage,
    ) -> AsyncIterator[StreamedMessagePart]:
        self._id = response.id
        self._usage = response.usage
        for block in response.content:
            match block.type:
                case "text":
                    yield TextPart(text=block.text)
                case "thinking":
                    yield ThinkPart(think=block.thinking)
                case "redacted_thinking":
                    yield ThinkPart(think="", encrypted=block.data)
                case "tool_use":
                    yield ToolCall(
                        id=block.id,
                        function=ToolCall.FunctionBody(
                            name=block.name, arguments=json.dumps(block.input)
                        ),
                    )
                case _:
                    continue

    async def _convert_stream_response(
        self,
        manager: AsyncStream[RawMessageStreamEvent],
    ) -> AsyncIterator[StreamedMessagePart]:
        try:
            async with manager as stream:
                async for event in stream:
                    if isinstance(event, MessageStartEvent):
                        self._id = event.message.id
                    elif isinstance(event, RawContentBlockStartEvent):
                        block = event.content_block
                        match block.type:
                            case "text":
                                yield TextPart(text=block.text)
                            case "thinking":
                                yield ThinkPart(think=block.thinking)
                            case "redacted_thinking":
                                yield ThinkPart(think="", encrypted=block.data)
                            case "tool_use":
                                yield ToolCall(
                                    id=block.id,
                                    function=ToolCall.FunctionBody(name=block.name, arguments=""),
                                )
                            case "server_tool_use" | "web_search_tool_result":
                                # ignore
                                continue
                    elif isinstance(event, RawContentBlockDeltaEvent):
                        delta = event.delta
                        match delta.type:
                            case "text_delta":
                                yield TextPart(text=delta.text)
                            case "thinking_delta":
                                yield ThinkPart(think=delta.thinking)
                            case "input_json_delta":
                                yield ToolCallPart(arguments_part=delta.partial_json)
                            case "signature_delta":
                                yield ThinkPart(think="", encrypted=delta.signature)
                            case "citations_delta":
                                # ignore
                                continue
                    elif isinstance(event, MessageDeltaEvent):
                        self._usage = cast(Usage, event.usage)
                    elif isinstance(event, MessageStopEvent):
                        continue
        except AnthropicError as exc:
            raise convert_error(exc) from exc


def tool_to_anthropic(tool: Tool) -> ToolParam:
    return {
        "name": tool.name,
        "description": tool.description,
        "input_schema": tool.parameters,
    }


def message_to_anthropic(message: Message) -> MessageParam:
    """Convert a single internal message into Anthropic wire format."""
    role = message.role
    content = message.content

    if role == "system":
        # Anthropic does not support system messages in the conversation.
        # We map it to a special user message.
        return MessageParam(
            role="user",
            content=[TextBlockParam(type="text", text=f"<system>{content}</system>")],
        )
    elif role == "tool":
        block = _tool_result_message_to_block(message)
        return MessageParam(role="user", content=[block])

    assert role in ("user", "assistant")
    if isinstance(content, str):
        return MessageParam(role=role, content=content)
    else:
        blocks: list[ContentBlockParam] = []
        for part in content:
            if isinstance(part, TextPart):
                blocks.append(TextBlockParam(type="text", text=part.text))
            elif isinstance(part, ImageURLPart):
                blocks.append(
                    ImageBlockParam(
                        type="image",
                        source=URLImageSourceParam(type="url", url=part.image_url.url),
                    )
                )
            elif isinstance(part, ThinkPart):
                if part.encrypted is None:
                    # missing signature, strip this thinking block.
                    continue
                else:
                    blocks.append(
                        ThinkingBlockParam(
                            type="thinking", thinking=part.think, signature=part.encrypted
                        )
                    )
            else:
                continue
        for tool_call in message.tool_calls or []:
            if tool_call.function.arguments:
                try:
                    parsed_arguments = json.loads(tool_call.function.arguments)
                except json.JSONDecodeError as exc:  # pragma: no cover - defensive guard
                    raise ChatProviderError("Tool call arguments must be valid JSON.") from exc
                if not isinstance(parsed_arguments, dict):
                    raise ChatProviderError("Tool call arguments must be a JSON object.")
                tool_input = cast(dict[str, object], parsed_arguments)
            else:
                tool_input = {}
            blocks.append(
                ToolUseBlockParam(
                    type="tool_use",
                    id=tool_call.id,
                    name=tool_call.function.name,
                    input=tool_input,
                )
            )
        return MessageParam(role=role, content=blocks)


def _tool_result_message_to_block(message: Message) -> ToolResultBlockParam:
    if message.tool_call_id is None:
        raise ChatProviderError("Tool response is missing `tool_call_id`")

    content: str | Sequence[ToolResultContent]
    if isinstance(message.content, str):
        content = message.content
    else:
        content_blocks: list[ToolResultContent] = []
        for part in message.content:
            if isinstance(part, TextPart):
                if part.text:
                    content_blocks.append(TextBlockParam(type="text", text=part.text))
            elif isinstance(part, ImageURLPart):
                content_blocks.append(
                    ImageBlockParam(
                        type="image",
                        source=URLImageSourceParam(type="url", url=part.image_url.url),
                    )
                )
            else:
                # https://docs.claude.com/en/docs/build-with-claude/files#file-types-and-content-blocks
                # Anthropic API supports very limited file types
                raise ChatProviderError(
                    f"Anthropic API does not support {type(part)} in tool result"
                )
        content = content_blocks

    return ToolResultBlockParam(
        type="tool_result",
        tool_use_id=message.tool_call_id,
        content=content,
    )


def convert_error(error: AnthropicError) -> ChatProviderError:
    if isinstance(error, AnthropicAPIStatusError):
        return APIStatusError(error.status_code, str(error))
    if isinstance(error, AnthropicAuthenticationError):
        return APIStatusError(getattr(error, "status_code", 401), str(error))
    if isinstance(error, AnthropicPermissionDeniedError):
        return APIStatusError(getattr(error, "status_code", 403), str(error))
    if isinstance(error, AnthropicRateLimitError):
        return APIStatusError(getattr(error, "status_code", 429), str(error))
    if isinstance(error, AnthropicAPIConnectionError):
        return APIConnectionError(str(error))
    if isinstance(error, AnthropicAPITimeoutError):
        return APITimeoutError(str(error))
    return ChatProviderError(f"Anthropic error: {error}")
