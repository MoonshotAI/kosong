import base64
import copy
import json
import mimetypes
from collections.abc import AsyncIterator, Sequence
from typing import TYPE_CHECKING, Any, Self, TypedDict, Unpack, cast

from google import genai
from google.genai import client as genai_client
from google.genai import errors as genai_errors
from google.genai.types import (
    Content,
    FunctionDeclaration,
    GenerateContentConfig,
    GenerateContentResponse,
    GenerateContentResponseUsageMetadata,
    HttpOptions,
    Part,
    ThinkingConfig,
    Tool,
    ToolConfig,
)

from kosong.chat_provider import (
    APIStatusError,
    APITimeoutError,
    ChatProvider,
    ChatProviderError,
    StreamedMessagePart,
    ThinkingEffort,
    TokenUsage,
)
from kosong.message import (
    ImageURLPart,
    Message,
    TextPart,
    ThinkPart,
    ToolCall,
)
from kosong.tooling import Tool as KosongTool

if TYPE_CHECKING:

    def type_check(gemini: "Gemini"):
        _: ChatProvider = gemini


class Gemini(ChatProvider):
    """
    Chat provider backed by Google's Gemini API.
    """

    name = "gemini"

    class GenerationKwargs(TypedDict, total=False):
        max_output_tokens: int | None
        temperature: float | None
        top_k: int | None
        top_p: float | None
        # Thinking configuration for supported models
        thinking_config: ThinkingConfig | None
        # Tool configuration
        tool_config: ToolConfig | None
        # Extra headers
        http_options: HttpOptions | None

    def __init__(
        self,
        *,
        model: str,
        api_key: str | None = None,
        base_url: str | None = None,
        stream: bool = True,
        # Must provide a max_output_tokens. Can be overridden by .with_generation_kwargs()
        default_max_tokens: int,
        **client_kwargs: Any,
    ):
        self._model = model
        self._stream = stream
        self._base_url = base_url
        self._client: genai_client.Client = genai.Client(
            http_options=HttpOptions(base_url=base_url),
            api_key=api_key,
            **client_kwargs,
        )
        self._generation_kwargs: Gemini.GenerationKwargs = {
            "max_output_tokens": default_max_tokens,
        }

    @property
    def model_name(self) -> str:
        return self._model

    async def generate(
        self,
        system_prompt: str,
        tools: Sequence[KosongTool],
        history: Sequence[Message],
    ) -> "GeminiStreamedMessage":
        # Convert messages to Gemini format
        contents: list[Content] = []

        # Convert history messages (excluding system prompt which is handled separately)
        for message in history:
            contents.append(message_to_gemini(message))

        generation_kwargs: dict[str, Any] = {}

        # Prepare tools
        gemini_tools = [tool_to_gemini(tool) for tool in tools]
        if gemini_tools:
            generation_kwargs["tools"] = gemini_tools

        # Prepare generation config with system instruction
        generation_kwargs.update(self._generation_kwargs)

        # Add system instruction if provided
        config_args: dict[str, Any] = {}
        if system_prompt:
            config_args["system_instruction"] = system_prompt

        generation_config = GenerateContentConfig(**config_args, **generation_kwargs)

        try:
            if self._stream:
                stream_response = await self._client.aio.models.generate_content_stream(  # type: ignore[reportUnknownMemberType]
                    model=self._model,
                    contents=contents,
                    config=generation_config,
                )
                return GeminiStreamedMessage(stream_response)
            else:
                response = await self._client.aio.models.generate_content(  # type: ignore[reportUnknownMemberType]
                    model=self._model,
                    contents=contents,
                    config=generation_config,
                )
                return GeminiStreamedMessage(response)
        except Exception as e:  # genai_errors.APIError and others
            raise _convert_error(e) from e

    def with_thinking(self, effort: "ThinkingEffort") -> Self:
        # Map thinking effort to budget tokens
        thinking_budget: int
        include_thoughts: bool
        match effort:
            case "off":
                thinking_budget = 0
                include_thoughts = False
            case "low":
                thinking_budget = 1024
                include_thoughts = True
            case "medium":
                thinking_budget = 4096
                include_thoughts = True
            case "high":
                thinking_budget = 32_000
                include_thoughts = True

        thinking_config = ThinkingConfig(
            thinking_budget=thinking_budget, include_thoughts=include_thoughts
        )
        return self.with_generation_kwargs(thinking_config=thinking_config)

    def with_generation_kwargs(self, **kwargs: Unpack[GenerationKwargs]) -> Self:
        """
        Copy the chat provider, updating the generation kwargs with the given values.

        Returns:
            Self: A new instance of the chat provider with updated generation kwargs.
        """
        new_self = copy.copy(self)
        new_self._generation_kwargs = copy.deepcopy(self._generation_kwargs)
        new_self._generation_kwargs.update(kwargs)
        return new_self

    @property
    def model_parameters(self) -> dict[str, Any]:
        """
        The parameters of the model to use.

        For tracing/logging purposes.
        """
        return {
            "model": self._model,
            "base_url": self._base_url,
            **self._generation_kwargs,
        }


class GeminiStreamedMessage:
    def __init__(self, response: GenerateContentResponse | AsyncIterator[GenerateContentResponse]):
        if isinstance(response, GenerateContentResponse):
            self._iter = self._convert_non_stream_response(response)
        else:
            self._iter = self._convert_stream_response(response)
        self._id: str | None = None
        self._usage: GenerateContentResponseUsageMetadata | None = None

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
        return TokenUsage(
            input_other=self._usage.prompt_token_count or 0,
            output=self._usage.candidates_token_count or 0,
            input_cache_read=self._usage.cached_content_token_count or 0,
            input_cache_creation=0,
        )

    async def _convert_non_stream_response(
        self,
        response: GenerateContentResponse,
    ) -> AsyncIterator[StreamedMessagePart]:
        # Extract usage information
        if response.usage_metadata:
            self._usage = response.usage_metadata
        # Extract ID if available
        if response.response_id is not None:
            self._id = response.response_id

        # Process candidates
        if response.candidates:
            for candidate in response.candidates:
                if candidate.content and candidate.content.parts:
                    for part in candidate.content.parts:
                        async for message_part in self._process_part_async(part):
                            yield message_part

    async def _convert_stream_response(
        self,
        response_stream: AsyncIterator[GenerateContentResponse],
    ) -> AsyncIterator[StreamedMessagePart]:
        try:
            async for response in response_stream:
                # Extract ID from first response
                if not self._id and response.response_id is not None:
                    self._id = response.response_id

                # Extract usage information
                if response.usage_metadata:
                    self._usage = response.usage_metadata

                # Process candidates
                if response.candidates:
                    for candidate in response.candidates:
                        if candidate.content and candidate.content.parts:
                            for part in candidate.content.parts:
                                async for message_part in self._process_part_async(part):
                                    yield message_part
        except genai_errors.APIError as exc:
            raise _convert_error(exc) from exc

    def _process_part(self, part: Part):
        """Process a single part and yield message components (synchronous generator).

        Handles different part types from Gemini API:
        - thinking parts (part.thought is True)
        - text parts
        - function calls
        """
        # Check if this is a thinking part first
        if getattr(part, 'thought', False):
            # This is a thinking/thought summary part (Gemini 2.5 models)
            if part.text:
                yield ThinkPart(think=part.text)
        elif part.text:
            # Regular text part (non-thinking)
            yield TextPart(text=part.text)
        elif part.function_call:
            func_call = part.function_call
            if func_call.name is None:
                # Skip function calls without a name
                return
            yield ToolCall(
                id=func_call.id if func_call.id is not None else f"call_{id(func_call)}",
                function=ToolCall.FunctionBody(
                    name=func_call.name,
                    arguments=json.dumps(func_call.args) if func_call.args else "{}",
                ),
            )

    async def _process_part_async(self, part: Part) -> AsyncIterator[StreamedMessagePart]:
        """Async wrapper for _process_part."""
        for message_part in self._process_part(part):
            yield message_part


def tool_to_gemini(tool: KosongTool) -> Tool:
    """Convert a Kosong tool to Gemini tool format."""
    # Kosong already validates parameters as JSON Schema format via jsonschema
    # The google-genai SDK accepts dict format and internally converts to Schema
    parameters_dict: dict[str, Any] = tool.parameters or {"type": "object", "properties": {}}

    return Tool(
        function_declarations=[
            FunctionDeclaration(
                name=tool.name,
                description=tool.description,
                parameters=parameters_dict,  # type: ignore[arg-type] # Gemini accepts dict
            )
        ]
    )


def _image_url_part_to_gemini(part: ImageURLPart) -> Part:
    """Convert an image URL part to Gemini format."""
    url = part.image_url.url

    # Handle data URLs
    if url.startswith("data:"):
        # data:[<media-type>][;base64],<data>
        res = url[5:].split(";base64,", 1)
        if len(res) != 2:
            raise ChatProviderError(f"Invalid data URL for image: {url}")

        media_type, data_b64 = res
        if media_type not in ("image/png", "image/jpeg", "image/gif", "image/webp"):
            raise ChatProviderError(
                f"Unsupported media type for base64 image: {media_type}, url: {url}"
            )

        # Decode base64 string to bytes
        data_bytes = base64.b64decode(data_b64)
        return Part.from_bytes(data=data_bytes, mime_type=media_type)
    else:
        # For regular URLs, try to detect MIME type from URL extension
        # If detection fails, use a safe default that Gemini accepts
        mime_type, _ = mimetypes.guess_type(url)
        if not mime_type or not mime_type.startswith("image/"):
            # Default to image/png if we can't detect or it's not an image type
            mime_type = "image/png"

        return Part.from_uri(file_uri=url, mime_type=mime_type)


def message_to_gemini(message: Message) -> Content:
    """Convert a single internal message into Gemini wire format."""
    role = message.role

    # Map role to Gemini roles
    # Gemini uses: "user" and "model" (not "assistant")
    if role == "assistant":
        gemini_role = "model"
    elif role == "tool":
        gemini_role = "user"  # Tool responses are sent as user messages
    else:
        gemini_role = role

    parts: list[Part] = []

    # Handle content (only for non-tool messages)
    if role != "tool":
        if isinstance(message.content, str):
            if message.content:
                parts.append(Part.from_text(text=message.content))
        else:
            for part in message.content:
                if isinstance(part, TextPart):
                    parts.append(Part.from_text(text=part.text))
                elif isinstance(part, ImageURLPart):
                    parts.append(_image_url_part_to_gemini(part))
                elif isinstance(part, ThinkPart):
                    # For models without native thinking support, include as special text block
                    if part.think:
                        parts.append(Part.from_text(text=f"<thinking>{part.think}</thinking>"))
                else:
                    # Skip unsupported parts
                    continue

    # Handle tool calls for assistant messages
    if message.tool_calls:
        for tool_call in message.tool_calls:
            if tool_call.function.arguments:
                try:
                    parsed_arguments = json.loads(tool_call.function.arguments)
                except json.JSONDecodeError as exc:  # pragma: no cover - defensive guard
                    raise ChatProviderError("Tool call arguments must be valid JSON.") from exc
                if not isinstance(parsed_arguments, dict):
                    raise ChatProviderError("Tool call arguments must be a JSON object.")
                args = cast(dict[str, object], parsed_arguments)
            else:
                args = {}

            parts.append(
                Part.from_function_call(
                    name=tool_call.function.name,
                    args=args,
                )
            )

    # Handle tool responses
    if role == "tool" and message.tool_call_id:
        if isinstance(message.content, str):
            response_content = message.content
        else:
            # Combine text parts for tool response
            response_parts: list[str] = []
            for part in message.content:
                if isinstance(part, TextPart):
                    response_parts.append(part.text)
            response_content = " ".join(response_parts) if response_parts else ""

        # Use tool_call_id as function name for the response
        # This matches the pattern used in the tool calls
        parts.append(
            Part.from_function_response(
                name=message.tool_call_id,
                response={"result": response_content},
            )
        )

    return Content(role=gemini_role, parts=parts)


def _convert_error(error: Exception) -> ChatProviderError:
    """Convert a Gemini error to a Kosong chat provider error."""
    # Handle specific Gemini error types with detailed status code mapping
    if isinstance(error, genai_errors.ClientError):
        # 4xx client errors
        status_code = getattr(error, "code", 400)
        if status_code == 401:
            return APIStatusError(401, f"Authentication failed: {error}")
        elif status_code == 403:
            return APIStatusError(403, f"Permission denied: {error}")
        elif status_code == 429:
            return APIStatusError(429, f"Rate limit exceeded: {error}")
        return APIStatusError(status_code, str(error))
    elif isinstance(error, genai_errors.ServerError):
        # 5xx server errors
        status_code = getattr(error, "code", 500)
        return APIStatusError(status_code, f"Server error: {error}")
    elif isinstance(error, genai_errors.APIError):
        # Generic API errors
        status_code = getattr(error, "code", 500)
        return APIStatusError(status_code, str(error))
    elif isinstance(error, TimeoutError):
        return APITimeoutError(f"Request timed out: {error}")
    else:
        # Fallback for unexpected errors
        return ChatProviderError(f"Unexpected Gemini error: {error}")
