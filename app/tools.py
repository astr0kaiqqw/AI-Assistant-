import json
from collections.abc import Callable
from datetime import datetime
from math import isfinite
from typing import Annotated, Literal
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from pydantic import BaseModel, ConfigDict, Field, FiniteFloat, StringConstraints


DEFAULT_TIMEZONE = "Asia/Almaty"
DOCUMENT_SEARCH_TOOL = "search_documents"


class DivisionByZero(ValueError):
    pass


def get_current_datetime(timezone: str = DEFAULT_TIMEZONE) -> str:
    """Get the current date and time in an IANA timezone. Defaults to Asia/Almaty."""
    return datetime.now(ZoneInfo(timezone)).isoformat(timespec="seconds")


def calculator(a: float, b: float, operation: str) -> float:
    """Calculate two numbers using add, subtract, multiply or divide."""

    if operation == "add":
        return a + b
    if operation == "subtract":
        return a - b
    if operation == "multiply":
        return a * b
    if operation == "divide":
        if b == 0:
            raise DivisionByZero("Cannot divide by zero")
        return a / b

    raise ValueError(f"Unsupported operation: {operation}")


class ToolArguments(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)


class DateTimeArguments(ToolArguments):
    timezone: str = DEFAULT_TIMEZONE


class CalculatorArguments(ToolArguments):
    a: FiniteFloat
    b: FiniteFloat
    operation: Literal["add", "subtract", "multiply", "divide"]


class DocumentSearchArguments(ToolArguments):
    """Search the local workspace for source excerpts. Use for questions about uploaded documents.
    Ground document claims in returned excerpts and cite their references as [S1], [S2], etc.
    If the excerpts do not support an answer, reply exactly INSUFFICIENT_DOCUMENT_EVIDENCE.
    """

    query: Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=300)]
    top_k: Annotated[int, Field(ge=1, le=5)] = 4


TOOL_REGISTRY = {
    "get_current_datetime": (get_current_datetime, DateTimeArguments),
    "calculator": (calculator, CalculatorArguments),
    DOCUMENT_SEARCH_TOOL: (None, DocumentSearchArguments),
}

TOOL_DEFINITIONS = [
    {
        "type": "function",
        "function": {
            "name": name,
            "description": handler.__doc__ if handler else arguments.__doc__,
            "parameters": arguments.model_json_schema(),
        },
    }
    for name, (handler, arguments) in TOOL_REGISTRY.items()
]


def encode_tool_result(result: object) -> str:
    return json.dumps({"result": result}, ensure_ascii=False, separators=(",", ":"))


def execute_tool(
    name: str, arguments: str, *, document_search: Callable | None = None,
) -> str:
    tool = TOOL_REGISTRY.get(name)
    if tool is None:
        return json.dumps({"error": "Unknown tool"})

    handler, schema = tool
    if handler is None:
        if document_search is None:
            return json.dumps({"error": "Document search is unavailable"})
        handler = document_search
    try:
        validated = schema.model_validate_json(arguments)
    except ValueError:
        return json.dumps({"error": "Invalid tool arguments"})

    try:
        result = handler(**validated.model_dump())
    except DivisionByZero:
        return json.dumps({"error": "Cannot divide by zero"})
    except ZoneInfoNotFoundError:
        return json.dumps({"error": "Unknown timezone. Use an IANA timezone name."})
    except ValueError:
        return json.dumps({"error": "Invalid tool operation or timezone"})

    if isinstance(result, float) and not isfinite(result):
        return json.dumps({"error": "Result is outside the supported numeric range"})
    return encode_tool_result(result)
