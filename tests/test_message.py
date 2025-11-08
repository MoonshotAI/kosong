from inline_snapshot import snapshot

from kosong.message import AudioURLPart, ImageURLPart, Message, TextPart, ThinkPart, ToolCall


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


def test_message_with_no_content():
    message = Message(
        role="assistant",
        content="",
        tool_calls=[
            ToolCall(id="123", function=ToolCall.FunctionBody(name="function", arguments="{}"))
        ],
    )

    assert message.model_dump(exclude_none=True) == snapshot(
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "type": "function",
                    "id": "123",
                    "function": {"name": "function", "arguments": "{}"},
                }
            ],
        }
    )


def test_message_deserialization():
    message = Message(
        role="user",
        content=[
            TextPart(text="Hello, world!"),
            ThinkPart(think="I think I need to think about this."),
            ImageURLPart(image_url=ImageURLPart.ImageURL(url="https://example.com/image.png")),
            AudioURLPart(audio_url=AudioURLPart.AudioURL(url="https://example.com/audio.mp3")),
        ],
        tool_calls=[
            ToolCall(id="123", function=ToolCall.FunctionBody(name="function", arguments="{}")),
        ],
    )

    dumped_message = message.model_dump(exclude_none=True)
    assert dumped_message == {
        "role": "user",
        "content": [
            TextPart(text="Hello, world!").model_dump(),
            ThinkPart(think="I think I need to think about this.").model_dump(),
            ImageURLPart(
                image_url=ImageURLPart.ImageURL(url="https://example.com/image.png")
            ).model_dump(),
            AudioURLPart(
                audio_url=AudioURLPart.AudioURL(url="https://example.com/audio.mp3")
            ).model_dump(),
        ],
        "tool_calls": [
            ToolCall(
                id="123", function=ToolCall.FunctionBody(name="function", arguments="{}")
            ).model_dump(),
        ],
    }

    assert Message.model_validate(dumped_message) == message


def test_deserialize_from_json_with_content_and_tool_calls():
    data = {
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
                "id": "tc_123",
                "function": {"name": "do_something", "arguments": '{"x":1}'},
            }
        ],
    }
    message = Message.model_validate(data)
    assert message.model_dump(exclude_none=True) == data


def test_deserialize_from_json_none_content_with_tool_calls():
    data = {
        "role": "assistant",
        "content": None,
        "tool_calls": [
            {
                "type": "function",
                "id": "tc_456",
                "function": {"name": "do_other", "arguments": "{}"},
            }
        ],
    }
    message = Message.model_validate(data)
    # Round-trip back to dict (exclude_none to keep content=None as in input)
    assert message.model_dump(exclude_none=True) == data


def test_deserialize_from_json_with_content_but_no_tool_calls():
    data = {
        "role": "user",
        "content": [
            {
                "type": "text",
                "text": "Only content, no tools.",
            }
        ],
    }
    message = Message.model_validate(data)
    assert message.model_dump(exclude_none=True) == data
