from collections.abc import Callable
from dataclasses import dataclass

from kosong.base import generate
from kosong.base.chat_provider import ChatProvider, StreamedMessagePart, TokenUsage
from kosong.base.message import Message, ToolCall
from kosong.context import Context
from kosong.tooling import ToolResult, ToolResultFuture
from kosong.utils.aio import Callback


async def step(
    chat_provider: ChatProvider,
    context: Context,
    *,
    on_message_part: Callback[[StreamedMessagePart], None] | None = None,
    on_tool_result: Callable[[ToolResult], None] | None = None,
) -> "StepResult":
    """
    Run one "step". In one step, the function generates LLM response based on the given context for
    exactly one time. All new message parts will be streamed to `on_message_part` in real-time if
    provided. Tool calls will be handled by `context.toolset`. The combined message will be returned
    in a `StepResult`. Depending on the toolset implementation, the tool calls may be handled
    asynchronously and the results need to be fetched by `await step_result.tool_results()`.

    The context will NOT be modified in this function.

    The token usage will be returned in the `StepResult` if available.
    """

    tool_calls: list[ToolCall] = []
    tool_result_futures: dict[str, ToolResultFuture] = {}

    async def on_tool_call(tool_call: ToolCall):
        def future_done_callback(result: ToolResultFuture):
            if on_tool_result:
                on_tool_result(result.result())

        tool_calls.append(tool_call)
        future = ToolResultFuture()
        future.add_done_callback(future_done_callback)
        tool_result_futures[tool_call.id] = future
        context.toolset.handle(tool_call, future)

    message, usage = await generate(
        chat_provider,
        context.system_prompt,
        context.toolset.tools,
        context.history,
        on_message_part=on_message_part,
        on_tool_call=on_tool_call,
    )

    return StepResult(message, usage, tool_calls, tool_result_futures)


@dataclass(frozen=True)
class StepResult:
    message: Message
    """The combined message generated in this step."""

    usage: TokenUsage | None
    """The token usage of the generated message."""

    tool_calls: list[ToolCall]
    """All the tool calls generated in this step."""

    tool_result_futures: dict[str, ToolResultFuture]
    """The futures of the results of the spawned tool calls."""

    async def tool_results(self) -> list[ToolResult]:
        """All the tool results returned by corresponding tool calls."""
        if not self.tool_result_futures:
            return []

        results: list[ToolResult] = []
        for tool_call in self.tool_calls:
            result = await self.tool_result_futures[tool_call.id]
            results.append(result)
        return results
