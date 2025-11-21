import base64
import copy
import json
import mimetypes
from collections.abc import AsyncIterator, Sequence
from typing import TYPE_CHECKING, Any, Self, TypedDict, Unpack, cast

import httpx
from google import genai
from google.genai import client as genai_client
from google.genai import errors as genai_errors
from google.genai.types import (
    Content,
    FunctionCall,
    FunctionDeclaration,
    FunctionResponse,
    FunctionResponsePart,
    GenerateContentConfig,
    GenerateContentResponse,
    GenerateContentResponseUsageMetadata,
    HttpOptions,
    Part,
    ThinkingConfig,
    ThinkingLevel,
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
    AudioURLPart,
    ContentPart,
    ImageURLPart,
    Message,
    TextPart,
    ThinkPart,
    ToolCall,
)
from kosong.tooling import Tool as KosongTool

if TYPE_CHECKING:

    def type_check(google_genai: "GoogleGenAI"):
        _: ChatProvider = google_genai


class GoogleGenAI:
    """
    Chat provider backed by Google's Gemini API.
    """

    name = "google_genai"

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
        self._generation_kwargs: GoogleGenAI.GenerationKwargs = {}

    @property
    def model_name(self) -> str:
        return self._model

    async def generate(
        self,
        system_prompt: str,
        tools: Sequence[KosongTool],
        history: Sequence[Message],
    ) -> "GoogleGenAIStreamedMessage":
        contents = [message_to_google_genai(message) for message in history]

        config = GenerateContentConfig(**self._generation_kwargs)
        config.system_instruction = system_prompt
        config.tools = [tool_to_google_genai(tool) for tool in tools]

        try:
            if self._stream:
                stream_response = await self._client.aio.models.generate_content_stream(  # type: ignore[reportUnknownMemberType]
                    model=self._model,
                    contents=contents,
                    config=config,
                )
                return GoogleGenAIStreamedMessage(stream_response)
            else:
                response = await self._client.aio.models.generate_content(  # type: ignore[reportUnknownMemberType]
                    model=self._model,
                    contents=contents,
                    config=config,
                )
                return GoogleGenAIStreamedMessage(response)
        except Exception as e:  # genai_errors.APIError and others
            raise _convert_error(e) from e

    def with_thinking(self, effort: "ThinkingEffort") -> Self:
        thinking_config = ThinkingConfig(include_thoughts=True)

        # Map thinking effort to budget tokens
        if "gemini-3" in self._model:
            match effort:
                case "off":
                    # use default thinking config
                    pass
                case "low":
                    thinking_config.thinking_level = ThinkingLevel.LOW
                case "medium":
                    # FIXME: medium not supported yet, use high
                    thinking_config.thinking_level = ThinkingLevel.HIGH
                case "high":
                    thinking_config.thinking_level = ThinkingLevel.HIGH
        else:
            match effort:
                case "off":
                    thinking_config.thinking_budget = 0
                    thinking_config.include_thoughts = False
                case "low":
                    thinking_config.thinking_budget = 1024
                    thinking_config.include_thoughts = True
                case "medium":
                    thinking_config.thinking_budget = 4096
                    thinking_config.include_thoughts = True
                case "high":
                    thinking_config.thinking_budget = 32_000
                    thinking_config.include_thoughts = True

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


class GoogleGenAIStreamedMessage:
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
        for candidate in response.candidates or []:
            parts = candidate.content.parts if candidate.content else None
            if not parts:
                continue
            for part in parts:
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
                for candidate in response.candidates or []:
                    parts = candidate.content.parts if candidate.content else None
                    if not parts:
                        continue
                    for part in parts:
                        async for message_part in self._process_part_async(part):
                            yield message_part
        except genai_errors.APIError as exc:
            raise _convert_error(exc) from exc

    def _process_part(self, part: Part):
        """Process a single part and yield message components (synchronous generator).

        Handles different part types from Gemini API:
        - synthetic thinking parts (part.thought is True)
        - encrypted thinking parts (part.thought_signature is not None)
        - text parts
        - function calls
        """
        if part.thought:
            # Synthetic thinking part
            if part.text:
                yield ThinkPart(think=part.text)
        elif part.text:
            # Regular text part
            yield TextPart(text=part.text)
        elif part.function_call:
            func_call = part.function_call
            if func_call.name is None:
                # Skip function calls without a name
                return
            id_ = func_call.id if func_call.id is not None else f"{id(func_call)}"
            tool_call_id = f"{func_call.name}_{id_}"
            # Gemini uses thought_signature to store the encrypted thinking signature.
            # part.thought is synthetic
            # See: https://colab.research.google.com/github/GoogleCloudPlatform/generative-ai/blob/main/gemini/thinking/intro_thought_signatures.ipynb
            thought_signature_b64 = (
                base64.b64encode(part.thought_signature).decode("ascii")
                if part.thought_signature
                else None
            )
            yield ToolCall(
                id=tool_call_id,
                function=ToolCall.FunctionBody(
                    name=func_call.name,
                    arguments=json.dumps(func_call.args) if func_call.args else "{}",
                ),
                extras={
                    "thought_signature_b64": thought_signature_b64,
                }
                if thought_signature_b64
                else None,
            )

    async def _process_part_async(self, part: Part) -> AsyncIterator[StreamedMessagePart]:
        """Async wrapper for _process_part."""
        for message_part in self._process_part(part):
            yield message_part


def tool_to_google_genai(tool: KosongTool) -> Tool:
    """Convert a Kosong tool to GoogleGenAI tool format."""
    # Kosong already validates parameters as JSON Schema format via jsonschema
    # The google-genai SDK accepts dict format and internally converts to Schema
    return Tool(
        function_declarations=[
            FunctionDeclaration(
                name=tool.name,
                description=tool.description,
                parameters=tool.parameters,  # type: ignore[arg-type] # GoogleGenAI accepts dict
            )
        ]
    )


def _image_url_part_to_google_genai(part: ImageURLPart) -> Part:
    """Convert an image URL part to GoogleGenAI format."""
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
        # For regular URLs, try to download the image and convert to bytes
        mime_type, _ = mimetypes.guess_type(url)
        if not mime_type or not mime_type.startswith("image/"):
            # Default to image/png if we can't detect or it's not an image type
            mime_type = "image/png"
        response = httpx.get(url).raise_for_status()
        data_bytes = response.content
        return Part.from_bytes(data=data_bytes, mime_type=mime_type)


def _audio_url_part_to_google_genai(part: AudioURLPart) -> Part:
    """Convert an audio URL part to GoogleGenAI format."""
    url = part.audio_url.url

    # Handle data URLs
    if url.startswith("data:"):
        # data:[<media-type>][;base64],<data>
        res = url[5:].split(";base64,", 1)
        if len(res) != 2:
            raise ChatProviderError(f"Invalid data URL for audio: {url}")

        media_type, data_b64 = res
        # Supported audio formats for GoogleGenAI
        supported_audio_types = (
            "audio/wav",
            "audio/mp3",
            "audio/aiff",
            "audio/aac",
            "audio/ogg",
            "audio/flac",
        )
        if media_type not in supported_audio_types:
            error_msg = (
                f"Unsupported media type for base64 audio: {media_type}, url: {url}. "
                f"Supported types: {supported_audio_types}"
            )
            raise ChatProviderError(error_msg)

        # Decode base64 string to bytes
        data_bytes = base64.b64decode(data_b64)
        return Part.from_bytes(data=data_bytes, mime_type=media_type)
    else:
        # Fetch the audio and convert to bytes
        mime_type, _ = mimetypes.guess_type(url)
        if not mime_type or not mime_type.startswith("audio/"):
            # Default to audio/mp3 if we can't detect or it's not an audio type
            mime_type = "audio/mp3"
        response = httpx.get(url).raise_for_status()
        data_bytes = response.content
        return Part.from_bytes(data=data_bytes, mime_type=mime_type)


def _tool_result_to_response_and_parts(
    parts: list[ContentPart],
) -> tuple[dict[str, str], list[FunctionResponsePart]]:
    """Convert tool response content to Gemini function response format."""
    genai_parts: list[FunctionResponsePart] = []
    response: str = ""

    for part in parts:
        if isinstance(part, TextPart):
            if part.text:
                response += part.text
        elif isinstance(part, ImageURLPart):
            genai_parts.append(FunctionResponsePart.from_uri(file_uri=part.image_url.url))
        elif isinstance(part, AudioURLPart):
            genai_parts.append(FunctionResponsePart.from_uri(file_uri=part.audio_url.url))
        else:
            # Skip unsupported parts (like ThinkPart, etc.)
            continue

    return {"output": response}, genai_parts


def message_to_google_genai(message: Message) -> Content:
    """Convert a single internal message into GoogleGenAI wire format."""
    role = message.role

    # Tool responses are sent as user messages
    if role == "tool":
        google_genai_role = "user"  # Tool responses are sent as user messages
        if message.tool_call_id is None:
            raise ChatProviderError("Tool response is missing `tool_call_id`")

        # Convert content to Gemini function response format
        if isinstance(message.content, str):
            response_data = {"output": message.content}
            tool_result_parts = None
        else:
            response_data, tool_result_parts = _tool_result_to_response_and_parts(message.content)
        return Content(
            role="user",
            parts=[
                Part(
                    function_response=FunctionResponse(
                        id=message.tool_call_id,
                        name=message.tool_call_id.split("_", 1)[0],
                        response=response_data,
                        parts=tool_result_parts,
                    )
                ),
            ],
        )

    # GoogleGenAI uses: "user" and "model" (not "assistant")
    google_genai_role = "model" if role == "assistant" else role
    parts: list[Part] = []

    # Handle content parts
    if isinstance(message.content, str):
        if message.content:
            parts.append(Part.from_text(text=message.content))
    else:
        for part in message.content:
            if isinstance(part, TextPart):
                parts.append(Part.from_text(text=part.text))
            elif isinstance(part, ImageURLPart):
                parts.append(_image_url_part_to_google_genai(part))
            elif isinstance(part, AudioURLPart):
                parts.append(_audio_url_part_to_google_genai(part))
            elif isinstance(part, ThinkPart):
                # Note: skip part.thought because it is synthetic
                continue
            else:
                # Skip unsupported parts
                continue

    # Handle tool calls for assistant messages
    for tool_call in message.tool_calls or []:
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

        function_call = FunctionCall(
            id=tool_call.id,
            name=tool_call.function.name,
            args=args,
        )
        function_call_part = Part(function_call=function_call)
        # Add thought_signature back to function_call
        if tool_call.extras and "thought_signature_b64" in tool_call.extras:
            function_call_part.thought_signature = base64.b64decode(
                cast(str, tool_call.extras["thought_signature_b64"])
            )
        parts.append(function_call_part)

    return Content(role=google_genai_role, parts=parts)


def _convert_error(error: Exception) -> ChatProviderError:
    """Convert a GoogleGenAI error to a Kosong chat provider error."""
    # Handle specific GoogleGenAI error types with detailed status code mapping
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
        return ChatProviderError(f"Unexpected GoogleGenAI error: {error}")


if __name__ == "__main__":

    async def main():
        import kosong

        chat = GoogleGenAI(model="gemini-3-pro-preview").with_thinking("high")
        system_prompt = "You are a helpful assistant."
        history = [Message(role="user", content="Hello, how are you?")]
        async for part in await chat.generate(system_prompt, [], history):
            print(part.model_dump(exclude_none=True))

        tools = [
            kosong.tooling.Tool(
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
    asyncio.run(main())
