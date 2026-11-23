import json
from datetime import datetime
from math import isfinite
from typing import Literal
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from pydantic import BaseModel, ConfigDict, FiniteFloat


DEFAULT_TIMEZONE = "Asia/Almaty"


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


TOOL_REGISTRY = {
    "get_current_datetime": (get_current_datetime, DateTimeArguments),
    "calculator": (calculator, CalculatorArguments),
}

TOOL_DEFINITIONS = [
    {
        "type": "function",
        "function": {
            "name": name,
            "description": handler.__doc__,
            "parameters": arguments.model_json_schema(),
        },
    }
    for name, (handler, arguments) in TOOL_REGISTRY.items()
]


def execute_tool(name: str, arguments: str) -> str:
    tool = TOOL_REGISTRY.get(name)
    if tool is None:
        return json.dumps({"error": "Unknown tool"})

    handler, schema = tool
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
    return json.dumps({"result": result})
