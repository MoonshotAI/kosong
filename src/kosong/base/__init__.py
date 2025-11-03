from collections.abc import Sequence

from loguru import logger

from kosong.base.chat_provider import ChatProvider, StreamedMessagePart, TokenUsage
from kosong.base.message import ContentPart, Message, TextPart, ToolCall
from kosong.base.tool import Tool
from kosong.utils.aio import Callback, callback


async def generate(
    chat_provider: ChatProvider,
    system_prompt: str,
    tools: Sequence[Tool],
    history: Sequence[Message],
    *,
    on_message_part: Callback[[StreamedMessagePart], None] | None = None,
    on_tool_call: Callback[[ToolCall], None] | None = None,
) -> tuple[Message, TokenUsage | None]:
    """
    Generate one message based on the given context. The given context will remain untouched.

    Parts of the message will be streamed to the given handlers:
    - `on_message_part` will be called for each raw part which may be incomplete.
    - `on_tool_call` will be called for each complete tool call.

    The generated message and the token usage will be returned. All parts in the message are
    guaranteed to be complete and merged as much as possible.
    """
    message = Message(role="assistant", content=[])
    pending_part: StreamedMessagePart | None = None  # message part that is currently incomplete

    logger.trace("Generating with history: {history}", history=history)
    stream = await chat_provider.generate(system_prompt, tools, history)
    async for part in stream:
        logger.trace("Received part: {part}", part=part)
        if on_message_part:
            await callback(on_message_part, part.model_copy(deep=True))

        if pending_part is None:
            pending_part = part
        elif not pending_part.merge_in_place(part):  # try merge into the pending part
            # unmergeable part must push the pending part to the buffer
            _message_append(message, pending_part)
            if isinstance(pending_part, ToolCall) and on_tool_call:
                await callback(on_tool_call, pending_part)
            pending_part = part

    # end of message
    if pending_part is not None:
        _message_append(message, pending_part)
        if isinstance(pending_part, ToolCall) and on_tool_call:
            await callback(on_tool_call, pending_part)

    return message, stream.usage


def _message_append(message: Message, part: StreamedMessagePart) -> None:
    match part:
        case ContentPart():
            if isinstance(message.content, str):
                message.content = [TextPart(text=message.content)]
            message.content.append(part)
        case ToolCall():
            if message.tool_calls is None:
                message.tool_calls = []
            message.tool_calls.append(part)
        case _:
            # may be an orphaned `ToolCallPart`
            return
