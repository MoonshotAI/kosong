from typing import Any, Protocol, runtime_checkable

JsonType = None | int | float | str | bool | list[Any] | dict[str, Any]


@runtime_checkable
class Stringifyable(Protocol):
    def __str__(self) -> str: ...
