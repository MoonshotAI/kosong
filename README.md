# Kosong

Kosong is an LLM abstraction layer designed for modern AI agent applications. It unifies message structures, asynchronous tool orchestration, and pluggable chat providers so you can build agents with ease and avoid vendor lock-in.

> Kosong means "empty" in Malay and Indonesian.

## Installation

Kosong requires Python 3.13 or higher. We recommend using uv as the package manager.

Init your project with:

```bash
uv init --python 3.13
```

Then add Kosong as a dependency:

```bash
uv add kosong
```

## Examples

### Simple chat completion

```python
import asyncio

import kosong
from kosong.chat_provider.kimi import Kimi
from kosong.message import Message


async def main() -> None:
    kimi = Kimi(
        base_url="https://api.moonshot.ai/v1",
        api_key="your_kimi_api_key_here",
        model="kimi-k2-turbo-preview",
    )

    history = [
        Message(role="user", content="Who are you?"),
    ]

    message, usage = await kosong.generate(
        chat_provider=kimi,
        system_prompt="You are a helpful assistant.",
        tools=[],
        history=history,
    )
    print(message)
    print(usage)


asyncio.run(main())
```

### Streaming output

```python
import asyncio

import kosong
from kosong.chat_provider import StreamedMessagePart
from kosong.chat_provider.kimi import Kimi
from kosong.message import Message


async def main() -> None:
    kimi = Kimi(
        base_url="https://api.moonshot.ai/v1",
        api_key="your_kimi_api_key_here",
        model="kimi-k2-turbo-preview",
    )

    history = [
        Message(role="user", content="Who are you?"),
    ]

    def output(message_part: StreamedMessagePart):
        print(message_part)

    message, usage = await kosong.generate(
        chat_provider=kimi,
        system_prompt="You are a helpful assistant.",
        tools=[],
        history=history,
        on_message_part=output,
    )
    print(message)
    print(usage)


asyncio.run(main())
```

### Tool calling with `kosong.step`

```python
import asyncio

from pydantic import BaseModel

import kosong
from kosong import StepResult
from kosong.chat_provider.kimi import Kimi
from kosong.message import Message
from kosong.tooling import CallableTool2, ToolOk, ToolReturnType
from kosong.tooling.simple import SimpleToolset


class AddToolParams(BaseModel):
    a: int
    b: int


class AddTool(CallableTool2[AddToolParams]):
    name: str = "add"
    description: str = "Add two integers."
    params: type[AddToolParams] = AddToolParams

    async def __call__(self, params: AddToolParams) -> ToolReturnType:
        return ToolOk(output=str(params.a + params.b))


async def main() -> None:
    kimi = Kimi(
        base_url="https://api.moonshot.ai/v1",
        api_key="your_kimi_api_key_here",
        model="kimi-k2-turbo-preview",
    )

    toolset = SimpleToolset()
    toolset += AddTool()

    history = [
        Message(role="user", content="Please add 2 and 3 with the add tool."),
    ]

    result: StepResult = await kosong.step(
        chat_provider=kimi,
        system_prompt="You are a precise math tutor.",
        toolset=toolset,
        history=history,
    )
    print(result.message)
    print(await result.tool_results())


asyncio.run(main())
```

## Builtin Demo

Kosong comes with a builtin demo agent that you can run locally. To start the demo, run:

```sh
export KIMI_BASE_URL="https://api.moonshot.ai/v1"
export KIMI_API_KEY="your_kimi_api_key"

uv run python -m kosong kimi --with-bash
```
