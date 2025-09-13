from kosong.tooling import ToolError


class ToolNotFoundError(ToolError):
    """The tool was not found."""

    def __init__(self, tool_name: str):
        super().__init__(f"Tool `{tool_name}` not found", "Tool not found")


class ToolParseError(ToolError):
    """The arguments of the tool are not valid JSON."""

    def __init__(self, message: str):
        super().__init__(f"Error parsing JSON arguments: {message}", "Invalid arguments")


class ToolValidateError(ToolError):
    """The arguments of the tool are not valid."""

    def __init__(self, message: str):
        super().__init__(f"Error validating JSON arguments: {message}", "Invalid arguments")


class ToolRuntimeError(ToolError):
    """The tool failed to run."""

    def __init__(self, message: str):
        super().__init__(f"Error running tool: {message}", "Tool runtime error")
