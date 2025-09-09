from kosong.base.message import Message
from kosong.tooling import Toolset


class LinearContext:
    """
    A context that contains a linear history of messages.
    """

    def __init__(self, system_prompt: str, toolset: Toolset):
        self._system_prompt = system_prompt
        self._toolset = toolset
        self._history: list[Message] = []

    @property
    def system_prompt(self) -> str:
        return self._system_prompt

    @property
    def toolset(self) -> Toolset:
        return self._toolset

    @property
    def history(self) -> list[Message]:
        return self._history

    async def add_message(self, message: Message):
        self._history.append(message)
