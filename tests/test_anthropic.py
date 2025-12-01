from inline_snapshot import snapshot

from kosong.contrib.chat_provider.anthropic import message_to_anthropic, messages_to_anthropic
from kosong.message import Message, ThinkPart, ToolCall


def test_message_to_anthropic_includes_tool_use_block_for_string_content() -> None:
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

    anthropic_payload = message_to_anthropic(message)

    assert anthropic_payload == snapshot(
        {
            "role": "assistant",
            "content": [
                {"type": "text", "text": "6"},
                {"type": "tool_use", "id": "abc", "name": "foo", "input": {"x": 1}},
            ],
        }
    )


def test_messages_to_anthropic_batches_tool_results_into_one_user_message() -> None:
    history = [
        Message(role="user", content="Tell me the weather"),
        Message(
            role="assistant",
            content="On it",
            tool_calls=[
                ToolCall(
                    id="call_weather",
                    function=ToolCall.FunctionBody(
                        name="get_weather",
                        arguments='{"location": "San Francisco"}',
                    ),
                ),
                ToolCall(
                    id="call_time",
                    function=ToolCall.FunctionBody(
                        name="get_time",
                        arguments='{"timezone": "America/Los_Angeles"}',
                    ),
                ),
            ],
        ),
        Message(role="tool", tool_call_id="call_weather", content="68F and clear"),
        Message(role="tool", tool_call_id="call_time", content="2:30 PM"),
    ]

    anthropic_payload = messages_to_anthropic(history)

    assert anthropic_payload == snapshot(
        [
            {
                "role": "user",
                "content": [{"type": "text", "text": "Tell me the weather"}],
            },
            {
                "role": "assistant",
                "content": [
                    {"type": "text", "text": "On it"},
                    {
                        "type": "tool_use",
                        "id": "call_weather",
                        "name": "get_weather",
                        "input": {"location": "San Francisco"},
                    },
                    {
                        "type": "tool_use",
                        "id": "call_time",
                        "name": "get_time",
                        "input": {"timezone": "America/Los_Angeles"},
                    },
                ],
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": "call_weather",
                        "content": [{"type": "text", "text": "68F and clear"}],
                    },
                    {
                        "type": "tool_result",
                        "tool_use_id": "call_time",
                        "content": [{"type": "text", "text": "2:30 PM"}],
                    },
                ],
            },
        ]
    )


def test_messages_to_anthropic_skips_empty_messages() -> None:
    history = [
        Message(role="user", content="Hello"),
        Message(role="assistant", content=[ThinkPart(think="")]),  # will be stripped
        Message(
            role="assistant",
            content="Here's a tool call",
            tool_calls=[
                ToolCall(
                    id="tool123",
                    function=ToolCall.FunctionBody(name="do", arguments="{}"),
                )
            ],
        ),
    ]

    anthropic_payload = messages_to_anthropic(history)

    assert anthropic_payload == snapshot(
        [
            {
                "role": "user",
                "content": [{"type": "text", "text": "Hello"}],
            },
            {
                "role": "assistant",
                "content": [
                    {"type": "text", "text": "Here's a tool call"},
                    {"type": "tool_use", "id": "tool123", "name": "do", "input": {}},
                ],
            },
        ]
    )
