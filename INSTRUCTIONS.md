You are an expert python programmer that is helping build this `omnidex` software which is a python based cli program used to run various agents using gguf models and the llama-cpp-python library.

This serves as a repository for agents, tools where each agent is developed as a standalone class of sorts.

The agent's capabilities can be found in:

- docs/agents/chat_agent.md
- docs/agents/research_summarizer.md
- docs/tutorials/demonstrating_shared_memory.md

## Coding Convention

Code heavy business logic using command patterns.

```python
class MyFunction:
    def __init__(self, param1:, param2:):
        self.param1 = param1
        self.param2 = param2

    def run(self):
        # Logic here
        pass

cmd = MyFunction(param1: "foo", param2: "bar")
cmd.run()
```

The CLI uses `rich` for spinners, styled logs, and chat thinking indicators.

## Agent Conventions

All execution agents should inherit from `BaseAgent`.

Shared agent/session behavior belongs on `BaseAgent`, not in ad hoc per-agent
dict shapes. The shared session contract includes:

- `last_response`
- `last_artifact_content`
- `last_artifact_responder`
- `last_responder`
- `last_tools_used`
- `artifact_history`

Prefer the base lifecycle helpers instead of custom session-state code:

- `empty_session_state()`
- `copy_session_state()`
- `apply_session_state(...)`
- `update_session_state(...)`
- `propose_handoff(...)`

Do not invent a chat-specific memory format when the state should be shared
across all agents.

## Chat Agent Convention

`chat_agent` should remain tool-free.

Its role is:

- general conversation on any topic
- follow-up questions about the current artifact
- preserving artifact continuity during chat turns

It should not:

- open files
- save/export artifacts
- compete with `research_assistant` for specialized research workflows

When `chat_agent` updates session state, it should usually call:

```python
self.update_session_state(response=response)
```

and should not replace `artifact_content` or `artifact_responder` unless it is
actually creating a new artifact.

This preserves research artifact ownership across generic chat turns.

## Research Agent Convention

`research_assistant` owns artifact-producing and artifact-persisting workflows.

That includes:

- PDF ingestion
- summarization
- structured insight extraction
- save/export flows for the active artifact

If a research flow creates a new artifact, it must update session state with the
new artifact content so later chat and save turns can refer to it.

Generic planned answers should not overwrite the active artifact unless a tool
explicitly produced new `artifact_content`.

## Orchestrator Convention

`orchestrator` is a delegator/router, not the final chat responder.

It should:

- route to the best initial agent
- copy shared session state into the delegated agent before execution
- copy resulting session state back after execution
- allow model-proposed handoffs
- validate routes and handoffs through the shared policy layer

Keep orchestration agentic, but keep workflow ownership guarded by explicit
validation rules.

## Prompting Convention

Prompts for `chat_agent` should:

- allow general-topic chat
- use shared context when relevant
- use session artifact state when relevant
- treat artifact context as supporting evidence, not a hard boundary

Prompts for `research_assistant` should focus on research workflows and tool use,
not general chat.

## Shared Memory Convention

If two agents need to cooperate over a multi-turn workflow, prefer shared
session-state continuity over hidden local memory.

The intended pattern is:

1. one agent creates an artifact
2. another agent can discuss that artifact
3. artifact ownership is preserved
4. a later save/export request can still resolve to the correct owner

This is the default pattern for `chat_agent` and `research_assistant`.

## Planner Convention

Planner-based workflows must be internally valid before execution.

If a plan references `$state.some_output_key...`, that output key must be
produced by an earlier step in the same plan. Incomplete plans should be
rejected and repaired rather than executed partially.

For save/export flows:

- do not rely on `create_output` alone if write intent is referenced from a
  missing step
- do not allow incomplete save plans to silently skip file writes

Prefer direct deterministic save flows for obvious follow-up persistence
requests when the artifact already exists in session state.
