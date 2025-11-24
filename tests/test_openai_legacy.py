from kosong.contrib.chat_provider.openai_legacy import message_to_openai
from kosong.message import AudioURLPart, ImageURLPart, Message, TextPart


def test_message_to_openai_serializes_tool_content_parts():
    message = Message(
        role="tool",
        content=[
            TextPart(text="result text"),
            ImageURLPart(image_url=ImageURLPart.ImageURL(url="https://example.com/image.png")),
            AudioURLPart(audio_url=AudioURLPart.AudioURL(url="https://example.com/audio.mp3")),
        ],
        tool_call_id="tool-call-id",
    )

    converted = message_to_openai(message, reasoning_key=None)
    content = converted["content"]

    assert isinstance(content, str)
    assert "result text" in content
    assert '"image_url"' in content
    assert "image.png" in content
    assert '"audio_url"' in content
    assert "audio.mp3" in content
