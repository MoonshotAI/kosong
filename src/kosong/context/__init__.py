from collections.abc import Sequence
from typing import Protocol, runtime_checkable

from kosong.base.message import Message
from kosong.tooling import Toolset

__all__ = [
    "Context",
    "LinearContext",
]


@runtime_checkable
class Context(Protocol):
    @property
    def system_prompt(self) -> str:
        """The system prompt to use for the context."""
        ...

    @property
    def toolset(self) -> Toolset:
        """The toolset to use for the context."""
        ...

    @property
    def history(self) -> Sequence[Message]:
        """The history of the context."""
        ...

    async def add_message(self, message: Message) -> None:
        """Add message to the context."""
        ...


from .linear import LinearContext  # noqa: E402


def __static_check_types(
    linear: "LinearContext",
):
    _: Context = linear
