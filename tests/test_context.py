import asyncio
from pathlib import Path

from kosong.base.message import Message
from kosong.context.linear import JsonlLinearStorage, LinearContext, MemoryLinearStorage


def test_linear_context():
    context = LinearContext(
        storage=MemoryLinearStorage(),
    )
    assert context.history == []

    async def run():
        await context.add_message(Message(role="user", content="abc"))
        await context.add_message(Message(role="assistant", content="def"))
        return context.history

    history = asyncio.run(run())
    assert history == [
        Message(role="user", content="abc"),
        Message(role="assistant", content="def"),
    ]


def test_linear_context_with_jsonl_storage():
    test_path = Path(__file__).parent / "test.jsonl"
    if test_path.exists():
        test_path.unlink()

    async def run():
        storage = JsonlLinearStorage(path=test_path)
        context = LinearContext(
            storage=storage,
        )
        await context.add_message(Message(role="user", content="abc"))
        await context.add_message(Message(role="assistant", content="def"))
        return context.history

    history = asyncio.run(run())
    assert history == [
        Message(role="user", content="abc"),
        Message(role="assistant", content="def"),
    ]

    with open(test_path) as f:
        expected = """\
{"role":"user","content":"abc"}
{"role":"assistant","content":"def"}
"""
        assert f.read() == expected

    test_path.unlink()


def test_linear_context_statistics():
    context = LinearContext(
        storage=MemoryLinearStorage(),
    )
    assert context.statistics == {
        "token_count": 0,
        "message_count": 0,
        "user_message_count": 0,
        "assistant_message_count": 0,
        "tool_message_count": 0,
        "system_message_count": 0,
    }

    async def run():
        await context.add_message(Message(role="system", content="System prompt"))
        await context.add_message(Message(role="user", content="Hello"))
        await context.add_message(Message(role="assistant", content="Hi"))
        await context.add_message(Message(role="user", content="How are you?"))
        await context.add_message(Message(role="assistant", content="I'm fine"))
        await context.add_message(Message(role="tool", content="Result", tool_call_id="123"))
        return context.statistics

    stats = asyncio.run(run())
    assert stats == {
        "token_count": 0,
        "message_count": 6,
        "user_message_count": 2,
        "assistant_message_count": 2,
        "tool_message_count": 1,
        "system_message_count": 1,
    }


def test_linear_context_extract_texts():
    from kosong.base.message import ImageURLPart, TextPart

    context = LinearContext(
        storage=MemoryLinearStorage(),
    )
    assert context.extract_texts() == []

    async def run():
        await context.add_message(Message(role="user", content="Hello"))
        await context.add_message(Message(role="assistant", content="Hi there"))
        await context.add_message(
            Message(
                role="user",
                content=[
                    TextPart(text="What is "),
                    ImageURLPart(image_url=ImageURLPart.ImageURL(url="https://example.com/img.png")),
                    TextPart(text="this?"),
                ],
            )
        )
        return context.extract_texts()

    texts = asyncio.run(run())
    assert texts == ["Hello", "Hi there", "What is this?"]


def test_linear_context_extract_texts_with_think():
    from kosong.base.message import TextPart, ThinkPart

    context = LinearContext(
        storage=MemoryLinearStorage(),
    )

    async def run():
        await context.add_message(Message(role="user", content="Hello"))
        await context.add_message(
            Message(
                role="assistant",
                content=[
                    TextPart(text="Let me think..."),
                    ThinkPart(think="I need to consider this carefully."),
                    TextPart(text="Here's my answer."),
                ],
            )
        )
        return context.extract_texts(include_think=False), context.extract_texts(include_think=True)

    texts_without_think, texts_with_think = asyncio.run(run())
    assert texts_without_think == ["Hello", "Let me think...Here's my answer."]
    assert texts_with_think == [
        "Hello",
        "Let me think...I need to consider this carefully.Here's my answer.",
    ]
