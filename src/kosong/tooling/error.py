class ToolError:
    """The error raised by a tool. This is not an exception."""

    def __init__(self, message: str):
        self.message = message


class ToolNotFoundError(ToolError):
    """The tool was not found."""

    def __init__(self, tool_name: str):
        super().__init__(f"Tool `{tool_name}` not found")


class ToolParseError(ToolError):
    """The arguments of the tool are not valid JSON."""

    def __init__(self, message: str):
        super().__init__(f"Error parsing JSON arguments: {message}")


class ToolValidateError(ToolError):
    """The arguments of the tool are not valid."""

    def __init__(self, message: str):
        super().__init__(f"Error validating JSON arguments: {message}")


class ToolRuntimeError(ToolError):
    """The tool failed to run."""

    def __init__(self, message: str):
        super().__init__(f"Error running tool: {message}")
