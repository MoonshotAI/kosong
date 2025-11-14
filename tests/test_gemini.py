from inline_snapshot import snapshot

from kosong.contrib.chat_provider.gemini import message_to_gemini
from kosong.message import Message, ToolCall


def test_message_to_gemini_includes_tool_use_block_for_string_content() -> None:
    from google.genai.types import Content, FunctionCall, Part

    message = Message(
        role="assistant",
        content="6",
        tool_calls=[
            ToolCall(
                id="abc",
                function=ToolCall.FunctionBody(
                    name="foo",
                    arguments='{"x":1}',
                ),
            )
        ],
    )

    gemini_payload = message_to_gemini(message)

    assert gemini_payload == snapshot(
        Content(
            parts=[Part(text="6"), Part(function_call=FunctionCall(args={"x": 1}, name="foo"))],
            role="model",
        )
    )
