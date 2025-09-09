import asyncio

from kosong.base.message import Message
from kosong.context import LinearContext
from kosong.tooling import EmptyToolset


def test_linear_context():
    context = LinearContext(system_prompt="abc", toolset=EmptyToolset())
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
