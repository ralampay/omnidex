"""Generic planned step execution helpers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

from omnidex.utils.planning import ToolPlanStep
from omnidex.utils.introspection import missing_required_inputs


def _normalize_tool_result(result: Any) -> dict[str, Any]:
    """Normalize tool output into a structured state payload."""
    if isinstance(result, dict):
        normalized = dict(result)
        normalized.setdefault("status", "ok")
        if "content" not in normalized:
            for key in ("answer", "summary", "text", "value"):
                if key in normalized:
                    normalized["content"] = normalized[key]
                    break
        return normalized
    return {"status": "ok", "value": result, "content": str(result)}


def _lookup_path(state: dict[str, Any], path: str) -> Any:
    """Resolve a dotted path from nested execution state."""
    current: Any = state
    for part in path.split("."):
        if isinstance(current, dict):
            current = current.get(part)
        else:
            current = getattr(current, part, None)
        if current is None:
            return None
    return current


def resolve_input_references(value: Any, *, state: dict[str, Any]) -> Any:
    """Resolve references like $query, $context, and $state.foo.bar."""
    if isinstance(value, str):
        if value.startswith("$") and "." not in value[1:]:
            return state.get(value[1:])
        if value.startswith("$state."):
            return _lookup_path(state, value[len("$state."):])
        return value
    if isinstance(value, dict):
        return {
            key: resolve_input_references(item, state=state)
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [resolve_input_references(item, state=state) for item in value]
    return value


@dataclass(slots=True)
class PlanExecutionResult:
    """State snapshot and final output after plan execution."""

    state: dict[str, Any]
    final_output: str


def execute_tool_plan(
    plan: list[ToolPlanStep],
    *,
    get_tool: Callable[[str], Any | None],
    record_tool_use: Callable[[str, str], None],
    emit: Callable[[str, str], None],
    log: Callable[[str], None],
    query: str,
    context: str = "",
    initial_state: dict[str, Any] | None = None,
) -> PlanExecutionResult:
    """Execute a plan through a generic tool loop."""
    state: dict[str, Any] = {
        "query": query,
        "context": context,
    }
    if initial_state:
        state.update(initial_state)
    last_result: dict[str, Any] | None = None

    for index, step in enumerate(plan, start=1):
        tool = get_tool(step.tool_name)
        if tool is None:
            emit(f"Tool unavailable: {step.tool_name}", "red")
            continue

        resolved_inputs = resolve_input_references(step.inputs, state=state)
        if not isinstance(resolved_inputs, dict):
            emit(f"Skipping {step.tool_name}: inputs must resolve to an object.", "yellow")
            continue
        missing_inputs = missing_required_inputs(tool, resolved_inputs)
        if missing_inputs:
            emit(
                f"Skipping {step.tool_name}: missing required inputs {', '.join(missing_inputs)}.",
                "yellow",
            )
            continue

        record_tool_use(step.tool_name, step.reason)
        log(
            f"Executing step {index}/{len(plan)}: tool={step.tool_name} "
            f"output_key={step.output_key}."
        )
        try:
            raw_result = tool.run(**resolved_inputs)
        except Exception as exc:
            return PlanExecutionResult(
                state=state,
                final_output=f"Tool {step.tool_name} failed: {exc}",
            )
        result = _normalize_tool_result(raw_result)
        state[step.output_key] = result
        last_result = result

        if str(result.get("status", "ok")) == "error":
            message = str(result.get("content") or result.get("message") or "Tool failed.")
            return PlanExecutionResult(state=state, final_output=message)

    final_candidate = state.get("final")
    if isinstance(final_candidate, dict):
        final_output = str(final_candidate.get("content", "")).strip()
        if final_output:
            return PlanExecutionResult(state=state, final_output=final_output)

    if last_result is not None:
        final_output = str(last_result.get("content", "")).strip()
        if final_output:
            return PlanExecutionResult(state=state, final_output=final_output)

    return PlanExecutionResult(state=state, final_output="")
