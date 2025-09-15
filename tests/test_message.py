from kosong.base.message import AudioURLPart, ImageURLPart, Message, TextPart, ThinkPart, ToolCall


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
