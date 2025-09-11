import asyncio
from pathlib import Path

from kosong.base.message import Message
from kosong.context import LinearContext
from kosong.context.linear import JsonlLinearStorage, MemoryLinearStorage
from kosong.tooling import EmptyToolset


def test_linear_context():
    context = LinearContext(
        system_prompt="abc",
        toolset=EmptyToolset(),
        storage=MemoryLinearStorage(),
    )
    assert context.system_prompt == "abc"
    assert context.history == []
    assert isinstance(context.toolset, EmptyToolset)

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
            system_prompt="abc",
            toolset=EmptyToolset(),
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
