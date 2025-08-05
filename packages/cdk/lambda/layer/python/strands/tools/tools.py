"""Core tool implementations.

This module provides the base classes for all tool implementations in the SDK, including function-based tools and
Python module-based tools, as well as utilities for validating tool uses and normalizing tool schemas.
"""

import asyncio
import inspect
import logging
import re
from typing import Any

from typing_extensions import override

from ..types.tools import AgentTool, ToolFunc, ToolGenerator, ToolSpec, ToolUse

logger = logging.getLogger(__name__)


class InvalidToolUseNameException(Exception):
    """Exception raised when a tool use has an invalid name."""

    pass


def validate_tool_use(tool: ToolUse) -> None:
    """Validate a tool use request.

    Args:
        tool: The tool use to validate.
    """
    validate_tool_use_name(tool)


def validate_tool_use_name(tool: ToolUse) -> None:
    """Validate the name of a tool use.

    Args:
        tool: The tool use to validate.

    Raises:
        InvalidToolUseNameException: If the tool name is invalid.
    """
    # We need to fix some typing here, because we don't actually expect a ToolUse, but dict[str, Any]
    if "name" not in tool:
        message = "tool name missing"  # type: ignore[unreachable]
        logger.warning(message)
        raise InvalidToolUseNameException(message)

    tool_name = tool["name"]
    tool_name_pattern = r"^[a-zA-Z0-9_\-]{1,}$"
    tool_name_max_length = 64
    valid_name_pattern = bool(re.match(tool_name_pattern, tool_name))
    tool_name_len = len(tool_name)

    if not valid_name_pattern:
        message = f"tool_name=<{tool_name}> | invalid tool name pattern"
        logger.warning(message)
        raise InvalidToolUseNameException(message)

    if tool_name_len > tool_name_max_length:
        message = f"tool_name=<{tool_name}>, tool_name_max_length=<{tool_name_max_length}> | invalid tool name length"
        logger.warning(message)
        raise InvalidToolUseNameException(message)


def _normalize_property(prop_name: str, prop_def: Any) -> dict[str, Any]:
    """Normalize a single property definition.

    Args:
        prop_name: The name of the property.
        prop_def: The property definition to normalize.

    Returns:
        The normalized property definition.
    """
    if not isinstance(prop_def, dict):
        return {"type": "string", "description": f"Property {prop_name}"}

    if prop_def.get("type") == "object" and "properties" in prop_def:
        return normalize_schema(prop_def)  # Recursive call

    # Copy existing property, ensuring defaults
    normalized_prop = prop_def.copy()

    # It is expected that type and description are already included in referenced $def.
    if "$ref" in normalized_prop:
        return normalized_prop

    normalized_prop.setdefault("type", "string")
    normalized_prop.setdefault("description", f"Property {prop_name}")
    return normalized_prop


def normalize_schema(schema: dict[str, Any]) -> dict[str, Any]:
    """Normalize a JSON schema to match expectations.

    This function recursively processes nested objects to preserve the complete schema structure.
    Uses a copy-then-normalize approach to preserve all original schema properties.

    Args:
        schema: The schema to normalize.

    Returns:
        The normalized schema.
    """
    # Start with a complete copy to preserve all existing properties
    normalized = schema.copy()

    # Ensure essential structure exists
    normalized.setdefault("type", "object")
    normalized.setdefault("properties", {})
    normalized.setdefault("required", [])

    # Process properties recursively
    if "properties" in normalized:
        properties = normalized["properties"]
        for prop_name, prop_def in properties.items():
            normalized["properties"][prop_name] = _normalize_property(prop_name, prop_def)

    return normalized


def normalize_tool_spec(tool_spec: ToolSpec) -> ToolSpec:
    """Normalize a complete tool specification by transforming its inputSchema.

    Args:
        tool_spec: The tool specification to normalize.

    Returns:
        The normalized tool specification.
    """
    normalized = tool_spec.copy()

    # Handle inputSchema
    if "inputSchema" in normalized:
        if isinstance(normalized["inputSchema"], dict):
            if "json" in normalized["inputSchema"]:
                # Schema is already in correct format, just normalize inner schema
                normalized["inputSchema"]["json"] = normalize_schema(normalized["inputSchema"]["json"])
            else:
                # Convert direct schema to proper format
                normalized["inputSchema"] = {"json": normalize_schema(normalized["inputSchema"])}

    return normalized


class PythonAgentTool(AgentTool):
    """Tool implementation for Python-based tools.

    This class handles tools implemented as Python functions, providing a simple interface for executing Python code
    as SDK tools.
    """

    _tool_name: str
    _tool_spec: ToolSpec
    _tool_func: ToolFunc

    def __init__(self, tool_name: str, tool_spec: ToolSpec, tool_func: ToolFunc) -> None:
        """Initialize a Python-based tool.

        Args:
            tool_name: Unique identifier for the tool.
            tool_spec: Tool specification defining parameters and behavior.
            tool_func: Python function to execute when the tool is invoked.
        """
        super().__init__()

        self._tool_name = tool_name
        self._tool_spec = tool_spec
        self._tool_func = tool_func

    @property
    def tool_name(self) -> str:
        """Get the name of the tool.

        Returns:
            The name of the tool.
        """
        return self._tool_name

    @property
    def tool_spec(self) -> ToolSpec:
        """Get the tool specification for this Python-based tool.

        Returns:
            The tool specification.
        """
        return self._tool_spec

    @property
    def tool_type(self) -> str:
        """Identifies this as a Python-based tool implementation.

        Returns:
            "python".
        """
        return "python"

    @override
    async def stream(self, tool_use: ToolUse, invocation_state: dict[str, Any], **kwargs: Any) -> ToolGenerator:
        """Stream the Python function with the given tool use request.

        Args:
            tool_use: The tool use request.
            invocation_state: Context for the tool invocation, including agent state.
            **kwargs: Additional keyword arguments for future extensibility.

        Yields:
            Tool events with the last being the tool result.
        """
        if inspect.iscoroutinefunction(self._tool_func):
            result = await self._tool_func(tool_use, **invocation_state)
        else:
            result = await asyncio.to_thread(self._tool_func, tool_use, **invocation_state)

        yield result
