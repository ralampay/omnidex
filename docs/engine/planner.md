# Planner

This document describes the shared planner infrastructure under
`omnidex/engine`.

## Purpose

The planner is the control-plane component that turns:

- a user query
- a live tool catalog
- optional initial state

into a normalized list of executable tool steps.

It is not itself a tool. It is the component that decides which tools to use.

## Modules

The shared planner currently lives in:

- [`omnidex/engine/planner.py`](../../omnidex/engine/planner.py)
- [`omnidex/engine/planner_prompts.py`](../../omnidex/engine/planner_prompts.py)

`planner.py` owns:

- tool catalog generation through `describe_tools(...)`
- planner LLM calls
- repair retries when the model returns invalid plan JSON
- normalization into `ToolPlanStep`
- plan validation for internal state references before execution

`planner_prompts.py` owns:

- the planner prompt
- the planner-repair prompt
- shared plan-format examples and constraints

## Planner Input Contract

`GeneratePlanCommand` is initialized with:

```python
GeneratePlanCommand(
    model=model,
    tools=tools,
    emit=emit,
    log=log,
)
```

Its runtime input is:

```python
plan = planner.execute(
    query,
    initial_state={
        "last_response": "...",
        "last_artifact_content": "...",
    },
)
```

So the planner sees:

- the current query
- the introspected tool catalog
- optional initial values injected by the agent

## Tool Catalog Structure

The planner does not consume handwritten JSON Schema files.

Instead, `describe_tools(...)` builds a lightweight tool description from live
tool objects:

```python
{
    "name": "tool_name",
    "description": "Normalized docstring text.",
    "inputs": ["param_a", "param_b"],
    "outputs": ["content", "field_b"],
}
```

That structure is inferred from:

- `tool.name`
- `tool.output_fields`
- `inspect.signature(tool.run)`
- the tool docstring

## Planner Output Contract

The planner is expected to return JSON in this shape:

```json
{
  "plan": [
    {
      "step": 1,
      "tool_name": "tool_name_here",
      "inputs": {
        "param": "value"
      },
      "output_key": "state_key_here",
      "reason": "why this step is needed"
    }
  ]
}
```

Accepted steps are normalized into:

```python
@dataclass(slots=True)
class ToolPlanStep:
    step: int
    tool_name: str
    inputs: dict[str, object]
    output_key: str
    reason: str
```

After normalization, the planner also validates cross-step references.

That means a step such as:

```json
{
  "tool_name": "create_output",
  "inputs": {
    "content": "$last_artifact_content",
    "filename": "/home/user/out.md",
    "write_output": "$state.write_request.write_output"
  },
  "output_key": "final"
}
```

is rejected if no earlier step produced `write_request`.

This prevents incomplete save/export plans from being executed with unresolved
`$state.*` references.

## Execution Relationship

The planner does not run tools directly.

The normal flow is:

1. build planner
2. generate plan
3. validate that any `$state.<output_key>...` references point to earlier plan outputs
4. if validation fails, ask the model for a repaired plan
5. pass the validated plan to `execute_tool_plan(...)`
6. let the executor resolve references and run tools

That separation matters:

- planner decides
- planner validates plan structure and cross-step references
- executor performs
- tools do domain work

## Why Validation Was Added

The planner can sometimes emit a save/export plan that looks plausible but is
internally incomplete.

Example failure mode:

1. emit a `create_output` step
2. reference `$state.write_request.write_output`
3. never produce a `write_request` step earlier in the plan

Without validation, that plan can render the artifact successfully while
silently skipping the file write because `write_output` resolves to `None`.

The current planner rejects that plan and uses the repair pass instead.

## Atomic Example

This example shows a very small agent with two tools:

- one tool reads a directory tree
- one tool summarizes the directory structure

The point is not the sophistication of the tools. The point is to show how a new
agent can reuse the shared planner without building a planner from scratch.

### Example Tools

```python
from __future__ import annotations

from pathlib import Path

from omnidex.tools.base import BaseTool


class ReadDirectoryTreeTool(BaseTool):
    """Read a bounded recursive directory listing from disk."""

    name = "read_directory_tree"
    output_fields = ("path", "tree", "content")

    def run(self, path: str) -> dict[str, object]:
        root = Path(path).expanduser()
        if not root.exists() or not root.is_dir():
            return {
                "status": "error",
                "path": str(root),
                "content": f"Error: directory not found: {root}",
            }

        lines: list[str] = []
        for entry in sorted(root.rglob("*"))[:200]:
            relative = entry.relative_to(root)
            prefix = "[D]" if entry.is_dir() else "[F]"
            lines.append(f"{prefix} {relative}")

        tree = "\n".join(lines)
        return {
            "status": "ok",
            "path": str(root),
            "tree": tree,
            "content": tree,
        }


class SummarizeDirectoryTreeTool(BaseTool):
    """Summarize a directory tree into a short structural overview."""

    name = "summarize_directory_tree"
    output_fields = ("summary", "content")

    def run(self, tree: str) -> dict[str, object]:
        lines = [line for line in tree.splitlines() if line.strip()]
        dir_count = sum(1 for line in lines if line.startswith("[D]"))
        file_count = sum(1 for line in lines if line.startswith("[F]"))
        sample = "\n".join(lines[:12])
        summary = (
            f"Directory summary:\n"
            f"- {dir_count} directories\n"
            f"- {file_count} files\n\n"
            f"Top entries:\n{sample}"
        )
        return {
            "status": "ok",
            "summary": summary,
            "content": summary,
        }
```

### Example Agent

```python
from __future__ import annotations

from omnidex.agents.base import BaseAgent
from omnidex.engine import GeneratePlanCommand
from omnidex.runtime import LocalChatModel, LocalLLMSettings
from omnidex.tools.create_output import CreateOutputTool
from omnidex.utils.plan_execution import execute_tool_plan


class DirectoryAssistant(BaseAgent):
    name = "directory_assistant"
    description = "Understands directory-inspection requests."

    def __init__(self, **kwargs) -> None:
        super().__init__(tools=[], **kwargs)
        self.settings = LocalLLMSettings.from_env(
            system_prompt="You are a directory analysis assistant."
        )
        self.model = LocalChatModel(self.settings)
        self.tools = [
            ReadDirectoryTreeTool(),
            SummarizeDirectoryTreeTool(),
            CreateOutputTool(),
        ]
        self.generate_plan_command = GeneratePlanCommand(
            model=self.model,
            tools=self.tools,
            emit=self.emit,
            log=self.log,
        )

    def run(self, query: str, context: str = "") -> str:
        initial_state = {
            "query": query,
            "context": context,
        }
        plan = self.generate_plan_command.execute(
            query,
            initial_state=initial_state,
        )
        execution = execute_tool_plan(
            plan,
            get_tool=self.get_tool,
            record_tool_use=lambda tool_name, reason: self.record_tool_use(
                tool_name,
                reason=reason,
            ),
            emit=lambda message, style: self.emit(message, style=style),
            log=self.log,
            query=query,
            context=context,
            initial_state=initial_state,
        )
        return execution.final_output or "No output generated."
```

### Example User Query

```text
Summarize the structure of /home/user/my_project
```

### Likely Plan

```json
{
  "plan": [
    {
      "step": 1,
      "tool_name": "read_directory_tree",
      "inputs": {
        "path": "/home/user/my_project"
      },
      "output_key": "tree",
      "reason": "Read the directory contents before summarizing them."
    },
    {
      "step": 2,
      "tool_name": "summarize_directory_tree",
      "inputs": {
        "tree": "$state.tree.tree"
      },
      "output_key": "summary",
      "reason": "Summarize the directory structure."
    },
    {
      "step": 3,
      "tool_name": "create_output",
      "inputs": {
        "content": "$state.summary.content"
      },
      "output_key": "final",
      "reason": "Format the final response."
    }
  ]
}
```

## Why This Example Matters

This example demonstrates the intended reuse model for future agents:

- the planner is shared
- the tools are agent-specific
- the execution loop is shared
- the agent only contributes its own tool set, system prompt, and any direct
  fast paths it wants

That is the intended architecture for adding more tool-using agents without
copying planner logic into each one.

## Design Guidance

Use the shared planner when an agent:

- has multiple tools
- needs dynamic multi-step decisions
- benefits from stateful tool chaining
- can benefit from automatic repair when the model emits an incomplete plan

Skip planner use when an agent:

- only has deterministic workflows
- can be handled with a few direct branches
- would pay more latency than value for planning

## Relevant Files

- [`omnidex/engine/planner.py`](../../omnidex/engine/planner.py)
- [`omnidex/engine/planner_prompts.py`](../../omnidex/engine/planner_prompts.py)
- [`omnidex/utils/introspection.py`](../../omnidex/utils/introspection.py)
- [`omnidex/utils/planning.py`](../../omnidex/utils/planning.py)
- [`omnidex/utils/plan_execution.py`](../../omnidex/utils/plan_execution.py)
