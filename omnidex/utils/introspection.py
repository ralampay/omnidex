"""Helpers for inspecting tools and call signatures."""

from __future__ import annotations

import inspect


def describe_tools(tools: list[object]) -> list[dict[str, object]]:
    """Return structured tool metadata for LLM planning."""
    described_tools: list[dict[str, object]] = []
    for tool in tools:
        tool_name = getattr(tool, "name", tool.__class__.__name__)
        description = (
            getattr(tool, "__doc__", None)
            or getattr(tool.run, "__doc__", None)
            or "Tool available for agent workflows."
        )
        normalized_description = " ".join(description.strip().split())

        expected_inputs: list[str] = []
        try:
            signature = inspect.signature(tool.run)
        except (TypeError, ValueError):
            signature = None

        if signature is not None:
            for parameter in signature.parameters.values():
                if parameter.name == "self":
                    continue
                if parameter.kind in {
                    inspect.Parameter.VAR_POSITIONAL,
                    inspect.Parameter.VAR_KEYWORD,
                }:
                    continue
                expected_inputs.append(parameter.name)

        described_tools.append(
            {
                "name": tool_name,
                "description": normalized_description,
                "inputs": expected_inputs or ["generic_input"],
                "outputs": list(getattr(tool, "output_fields", ("content",))),
            }
        )

    return described_tools


def missing_required_inputs(
    tool: object,
    inputs: dict[str, object],
    *,
    derived_inputs: set[str] | None = None,
) -> list[str]:
    """Return required tool inputs that are missing or unresolved."""
    try:
        signature = inspect.signature(tool.run)
    except (TypeError, ValueError):
        return []

    available_inputs = set(inputs)
    if derived_inputs:
        available_inputs.update(derived_inputs)

    missing: list[str] = []
    for parameter in signature.parameters.values():
        if parameter.name == "self":
            continue
        if parameter.kind in {
            inspect.Parameter.VAR_POSITIONAL,
            inspect.Parameter.VAR_KEYWORD,
        }:
            continue
        if parameter.default is not inspect._empty:
            continue
        if parameter.name not in available_inputs:
            missing.append(parameter.name)
            continue
        if inputs.get(parameter.name) is None:
            missing.append(parameter.name)
    return missing
