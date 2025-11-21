from inline_snapshot import snapshot

from kosong.contrib.chat_provider.google_genai import message_to_google_genai
from kosong.message import Message, ToolCall


def test_message_to_google_genai_includes_tool_use_block_for_string_content() -> None:
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

    google_genai_payload = message_to_google_genai(message)

    assert google_genai_payload == snapshot(
        Content(
            parts=[
                Part(text="6"),
                Part(function_call=FunctionCall(id="abc", args={"x": 1}, name="foo")),
            ],
            role="model",
        )
    )
