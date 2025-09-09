from kosong.base.message import Message, TextPart, ToolCall


def test_plain_text_message():
    message = Message(role="user", content="Hello, world!")
    assert message.role == "user"
    assert message.content == "Hello, world!"
    assert message.model_dump(exclude_none=True) == {
        "role": "user",
        "content": "Hello, world!",
    }


def test_message_with_tool_calls():
    message = Message(
        role="assistant",
        content=[TextPart(text="Hello, world!")],
        tool_calls=[
            ToolCall(id="123", function=ToolCall.FunctionBody(name="function", arguments="{}"))
        ],
    )
    assert message.model_dump(exclude_none=True) == {
        "role": "assistant",
        "content": [
            {
                "type": "text",
                "text": "Hello, world!",
            }
        ],
        "tool_calls": [
            {
                "type": "function",
                "id": "123",
                "function": {
                    "name": "function",
                    "arguments": "{}",
                },
            }
        ],
    }
