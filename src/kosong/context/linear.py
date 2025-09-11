import asyncio
import json
from pathlib import Path
from typing import IO, Protocol, runtime_checkable

from kosong.base.message import Message
from kosong.tooling import Toolset


class LinearContext:
    """
    A context that contains a linear history of messages.
    """

    def __init__(
        self,
        system_prompt: str,
        toolset: Toolset,
        storage: "LinearStorage",
    ):
        self._system_prompt = system_prompt
        self._toolset = toolset
        # TODO: support forget/compact active messages
        self._active_messages: list[Message] = []
        self._storage = storage

    @property
    def system_prompt(self) -> str:
        return self._system_prompt

    @property
    def toolset(self) -> Toolset:
        return self._toolset

    @property
    def history(self) -> list[Message]:
        return self._active_messages

    async def add_message(self, message: Message):
        self._active_messages.append(message)
        await self._storage.append_message(message)


@runtime_checkable
class LinearStorage(Protocol):
    async def append_message(self, message: Message) -> None: ...


class MemoryLinearStorage:
    """
    A linear storage that stores messages in memory, only for testing.
    """

    def __init__(self):
        self._messages: list[Message] = []

    async def append_message(self, message: Message):
        self._messages.append(message)


class JsonlLinearStorage:
    """
    A linear storage that stores messages in a JSONL file.
    """

    def __init__(self, path: Path | str):
        self._path = path
        self._file: IO[str] | None = None
        self._dump_fn = lambda m: m.model_dump(exclude_none=True)

    def _get_file(self) -> IO[str]:
        if self._file is None:
            self._file = open(self._path, "w", encoding="utf-8")  # noqa: SIM115
        return self._file

    def __del__(self):
        if self._file:
            self._file.close()

    async def append_message(self, message: Message):
        file = self._get_file()

        def _write():
            json.dump(self._dump_fn(message), file, ensure_ascii=False, separators=(",", ":"))
            file.write("\n")

        await asyncio.to_thread(_write)
