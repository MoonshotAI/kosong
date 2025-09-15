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
        self._storage = storage

    @property
    def system_prompt(self) -> str:
        return self._system_prompt

    @property
    def toolset(self) -> Toolset:
        return self._toolset

    @property
    def history(self) -> list[Message]:
        return self._storage.list_messages()

    async def add_message(self, message: Message):
        await self._storage.append_message(message)


@runtime_checkable
class LinearStorage(Protocol):
    def list_messages(self) -> list[Message]:
        """
        List all messages in the storage.
        All messages should have a copy in memory so this method should be non-async.
        """
        ...

    async def append_message(self, message: Message) -> None: ...


class MemoryLinearStorage:
    """
    A linear storage that stores messages in memory, only for testing.
    """

    def __init__(self):
        self._messages: list[Message] = []

    def list_messages(self) -> list[Message]:
        return self._messages

    async def append_message(self, message: Message):
        self._messages.append(message)


class JsonlLinearStorage(MemoryLinearStorage):
    """
    A linear storage that stores messages in a JSONL file.
    """

    def __init__(self, path: Path | str):
        super().__init__()
        self._path = path if isinstance(path, Path) else Path(path)
        self._file: IO[str] | None = None

    async def restore(self):
        """Restore all messages from the JSONL file."""
        if self._messages:
            raise RuntimeError("The storage is already modified")
        if not self._path.exists():
            return

        def _restore():
            with open(self._path, encoding="utf-8") as f:
                for line in f:
                    message = Message.model_validate(json.loads(line))
                    self._messages.append(message)

        await asyncio.to_thread(_restore)

    def _get_file(self) -> IO[str]:
        if self._file is None:
            self._file = open(self._path, "a", encoding="utf-8")  # noqa: SIM115
        return self._file

    def __del__(self):
        if self._file:
            self._file.close()

    def list_messages(self) -> list[Message]:
        return super().list_messages()

    async def append_message(self, message: Message):
        await super().append_message(message)

        def _write():
            file = self._get_file()
            json.dump(
                message.model_dump(exclude_none=True),
                file,
                ensure_ascii=False,
                separators=(",", ":"),
            )
            file.write("\n")

        await asyncio.to_thread(_write)
