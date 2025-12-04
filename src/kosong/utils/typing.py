from __future__ import annotations

from typing import Literal

type JsonType = None | int | float | str | bool | list[JsonType] | dict[str, JsonType]

type ToolResultProcess = Literal["raw", "extract_text"]
