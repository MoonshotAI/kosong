from kosong.contrib.chat_provider.openai_legacy import OpenAILegacy, message_to_openai
from kosong.message import (
    AudioURLPart,
    ContentPart,
    ImageURLPart,
    Message,
    TextPart,
    ThinkPart,
)


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
    assert "content" in converted
    content = converted["content"]

    assert content == (
        "{'type': 'text', 'text': 'result text'}\n"
        "{'type': 'image_url', 'image_url': {'url': 'https://example.com/image.png', 'id': None}}\n"
        "{'type': 'audio_url', 'audio_url': {'url': 'https://example.com/audio.mp3', 'id': None}}"
    )


def test_message_to_openai_skips_unsupported_tool_content_parts():
    class UnknownPart(ContentPart):
        type: str = "unknown"
        payload: str

    message = Message(
        role="tool",
        content=[
            TextPart(text="kept text"),
            ThinkPart(think="secret reasoning"),
            UnknownPart(payload="ignore me"),
        ],
        tool_call_id="tool-call-id",
    )

    converted = message_to_openai(message, reasoning_key=None)
    assert "content" in converted
    content = converted["content"]

    assert content == "{'type': 'text', 'text': 'kept text'}"


def test_message_to_openai_includes_empty_reasoning_for_assistant_when_requested():
    message = Message(role="assistant", content=[])

    converted = message_to_openai(message, reasoning_key="reasoning_content")

    assert converted["reasoning_content"] == ""
    assert converted["role"] == "assistant"


def test_openai_legacy_defaults_reasoning_key_for_deepseek():
    provider = OpenAILegacy(
        model="deepseek-chat",
        api_key="sk-test",
        base_url="https://api.deepseek.com",
        stream=False,
    )

    assert provider.reasoning_key == "reasoning_content"
