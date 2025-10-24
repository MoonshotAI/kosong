import copy
import os
from collections.abc import Sequence
from typing import TypedDict, Unpack, cast, override

from openai import OpenAIError
from openai.types.chat import ChatCompletionMessageParam, ChatCompletionToolParam

from kosong.base.message import Message
from kosong.base.tool import Tool
from kosong.chat_provider import ChatProviderError
from kosong.chat_provider.openai_legacy import (
    OpenAILegacy,
    OpenAILegacyStreamedMessage,
    convert_error,
    message_to_openai,
    tool_to_openai,
)


class Kimi(OpenAILegacy):
    """
    A chat provider that uses the Kimi API.

    >>> chat_provider = Kimi(model="kimi-k2-turbo-preview", api_key="sk-1234567890")
    >>> chat_provider.name
    'kimi'
    >>> chat_provider.model_name
    'kimi-k2-turbo-preview'
    >>> chat_provider.with_generation_kwargs(temperature=0)._generation_kwargs
    {'temperature': 0}
    >>> chat_provider._generation_kwargs
    {}
    """

    name = "kimi"

    def __init__(
        self,
        *,
        model: str,
        api_key: str | None = None,
        base_url: str | None = None,
        stream: bool = True,
        **client_kwargs,
    ):
        if api_key is None:
            api_key = os.getenv("KIMI_API_KEY")
        if api_key is None:
            raise ChatProviderError(
                "The api_key client option or the KIMI_API_KEY environment variable is not set"
            )
        if base_url is None:
            base_url = os.getenv("KIMI_BASE_URL", "https://api.moonshot.cn/v1")
        super().__init__(
            model=model,
            api_key=api_key,
            base_url=base_url,
            stream=stream,
            **client_kwargs,
        )

        self._generation_kwargs = {}

    @override
    async def generate(
        self,
        system_prompt: str,
        tools: Sequence[Tool],
        history: Sequence[Message],
    ) -> "KimiStreamedMessage":
        messages: list[ChatCompletionMessageParam] = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.extend(message_to_openai(message) for message in history)

        generation_kwargs = {
            # default kimi generation kwargs
            "max_tokens": 32000,
            "temperature": 0.6,
        }
        generation_kwargs.update(self._generation_kwargs)

        try:
            response = await self._client.chat.completions.create(
                model=self._model,
                messages=messages,
                tools=(tool_to_kimi(tool) for tool in tools),
                stream=self._stream,
                stream_options={"include_usage": True},
                **generation_kwargs,
            )
            return KimiStreamedMessage(response)
        except OpenAIError as e:
            raise convert_error(e) from e

    class GenerationKwargs(TypedDict, total=False):
        max_tokens: int | None
        temperature: float | None
        top_p: float | None
        n: int | None
        presence_penalty: float | None
        frequency_penalty: float | None
        stop: str | list[str] | None
        prompt_cache_key: str | None

    def with_generation_kwargs(self, **kwargs: Unpack[GenerationKwargs]) -> "Kimi":
        new_self = copy.copy(self)
        new_self._generation_kwargs = kwargs
        return new_self


def tool_to_kimi(tool: Tool) -> ChatCompletionToolParam:
    if tool.name.startswith("$"):
        # Kimi builtin functions start with `$`
        return cast(
            ChatCompletionToolParam,
            {
                "type": "builtin_function",
                "function": {
                    "name": tool.name,
                    # no need to set description and parameters
                },
            },
        )
    else:
        return tool_to_openai(tool)


KimiStreamedMessage = OpenAILegacyStreamedMessage


if __name__ == "__main__":

    async def _dev_main():
        chat = Kimi(model="kimi-k2-turbo-preview", stream=False)
        system_prompt = ""
        history = [
            Message(role="user", content="Hello, who is Confucius?"),
        ]
        stream = await chat.with_generation_kwargs(
            temperature=0,
            max_tokens=1000,
        ).generate(system_prompt, [], history)
        async for part in stream:
            print(part.model_dump(exclude_none=True))
        print("id:", stream.id)
        print("usage:", stream.usage)

    import asyncio

    from dotenv import load_dotenv

    load_dotenv()
    asyncio.run(_dev_main())
