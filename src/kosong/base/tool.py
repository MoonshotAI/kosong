from typing import Any, Self

import jsonschema
from pydantic import BaseModel, model_validator


class Tool(BaseModel):
    name: str
    """The name of the tool."""

    description: str
    """The description of the tool."""

    parameters: dict[str, Any]
    """The parameters of the tool, in JSON Schema format."""

    @model_validator(mode="after")
    def validate_parameters(self) -> Self:
        jsonschema.validate(self.parameters, jsonschema.Draft202012Validator.META_SCHEMA)
        return self
