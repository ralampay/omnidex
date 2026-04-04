"""Shared command for structured tool-plan generation."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Callable

from omnidex.engine.planner_prompts import (
    build_plan_messages,
    build_plan_repair_messages,
)
from omnidex.utils.introspection import describe_tools
from omnidex.utils.json_tools import load_json_object
from omnidex.utils.planning import ToolPlanStep
from omnidex.utils.text import preview_text


class GeneratePlanCommand:
    """Generate and normalize a tool execution plan."""

    def __init__(
        self,
        *,
        model: object,
        tools: list[object],
        emit: Callable[[str], None],
        log: Callable[[str], None],
    ) -> None:
        self.model = model
        self.tools = tools
        self.emit = emit
        self.log = log

    def execute(
        self,
        query: str,
        *,
        initial_state: dict[str, object] | None = None,
    ) -> list[ToolPlanStep]:
        """Use the local model to generate a structured tool plan."""
        tool_descriptions = describe_tools(self.tools)
        available_tool_names = {
            str(tool_description.get("name", "")).strip()
            for tool_description in tool_descriptions
            if str(tool_description.get("name", "")).strip()
        }
        previous_output = ""
        failure_reason = ""

        for attempt in range(1, 4):
            messages = (
                build_plan_messages(
                    query,
                    tool_descriptions,
                    initial_values=initial_state,
                )
                if attempt == 1
                else build_plan_repair_messages(
                    query,
                    tool_descriptions,
                    initial_values=initial_state,
                    previous_output=previous_output,
                    failure_reason=failure_reason or "Planner returned no usable steps.",
                )
            )

            try:
                raw_content = self.model.generate_text(messages, stream=False)
            except Exception as exc:
                self.emit(f"Plan generation failed: {exc}", style="yellow")
                return []

            normalized_plan, failure_reason = self._normalize_plan(
                raw_content,
                available_tool_names=available_tool_names,
            )
            if normalized_plan:
                return normalized_plan

            previous_output = raw_content
            self.emit(
                f"Planner attempt {attempt} produced no usable steps: {failure_reason}",
                style="yellow",
            )
            self.log(f"Raw plan output on attempt {attempt}: {raw_content}")

        return []

    def _normalize_plan(
        self,
        raw_content: str,
        *,
        available_tool_names: set[str],
    ) -> tuple[list[ToolPlanStep], str]:
        """Normalize raw model output into valid tool plan steps."""
        payload = load_json_object(raw_content)
        if payload is None:
            return [], "invalid JSON output"

        raw_plan = payload.get("plan", [])
        if not isinstance(raw_plan, list):
            return [], "plan must be a list"

        normalized_plan: list[ToolPlanStep] = []
        self.log(f"Normalizing raw tool plan with {len(raw_plan)} step candidate(s).")
        for index, step in enumerate(raw_plan, start=1):
            if not isinstance(step, dict):
                self.emit(f"Skipping invalid plan step at position {index}.", style="yellow")
                continue

            tool_name = str(step.get("tool_name", "")).strip()
            inputs = step.get("inputs", {})
            output_key = str(step.get("output_key", "")).strip()
            reason = str(step.get("reason", "")).strip()
            self.log(
                f"Inspecting raw plan step {index}: tool={tool_name or '<missing>'}, "
                f"inputs={list(inputs) if isinstance(inputs, dict) else type(inputs).__name__}."
            )

            if not tool_name:
                self.emit(f"Skipping plan step {index}: missing tool_name.", style="yellow")
                continue
            if tool_name not in available_tool_names:
                self.emit(
                    f"Skipping plan step {index}: unknown tool '{tool_name}'.",
                    style="yellow",
                )
                continue
            if not isinstance(inputs, dict):
                self.emit(f"Skipping plan step {index}: inputs must be an object.", style="yellow")
                continue
            if not output_key:
                self.emit(f"Skipping plan step {index}: missing output_key.", style="yellow")
                continue

            normalized_plan.append(
                ToolPlanStep(
                    step=index if not isinstance(step.get("step"), int) else step["step"],
                    tool_name=tool_name,
                    inputs=inputs,
                    output_key=output_key,
                    reason=reason or "No reason provided.",
                )
            )
            self.log(
                f"Accepted plan step {index}: tool={tool_name}, "
                f"reason={preview_text(reason or 'No reason provided.')}"
            )

        if normalized_plan:
            validation_error = self._validate_plan_references(normalized_plan)
            if validation_error:
                return [], validation_error
            return normalized_plan, ""
        return [], "planner returned an empty or invalid plan"

    def _validate_plan_references(self, plan: list[ToolPlanStep]) -> str:
        """Validate that state references only point to earlier plan outputs."""
        available_outputs: set[str] = set()
        for step in plan:
            missing_reference = self._first_missing_state_reference(
                step.inputs,
                available_outputs=available_outputs,
            )
            if missing_reference:
                return (
                    f"step '{step.tool_name}' references missing state key "
                    f"'{missing_reference}' before it is produced"
                )
            available_outputs.add(step.output_key)
        return ""

    def _first_missing_state_reference(
        self,
        value: object,
        *,
        available_outputs: set[str],
    ) -> str:
        """Return the first unresolved $state.<output_key> reference, if any."""
        for reference in self._iter_state_references(value):
            output_key = reference.split(".", 1)[0].strip()
            if output_key and output_key not in available_outputs:
                return output_key
        return ""

    def _iter_state_references(self, value: object) -> Iterable[str]:
        """Yield dotted $state references found anywhere inside a plan input payload."""
        if isinstance(value, str):
            if value.startswith("$state."):
                yield value[len("$state.") :]
            return
        if isinstance(value, dict):
            for item in value.values():
                yield from self._iter_state_references(item)
            return
        if isinstance(value, list):
            for item in value:
                yield from self._iter_state_references(item)
