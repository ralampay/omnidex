"""Generic planning data structures."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(slots=True)
class ToolPlanStep:
    """One normalized tool-plan step."""

    step: int
    tool_name: str
    inputs: dict[str, object]
    output_key: str
    reason: str
