# Demonstrating Shared Memory

This tutorial explains how `chat_agent` and `research_assistant` interact
through shared session state, and how `chat_agent` is implemented so it can:

- answer general questions on any topic
- answer follow-up questions about the current artifact
- preserve artifact ownership for later save/export requests

The point is not to make `chat_agent` a research tool. The point is to let it
talk naturally about whatever the user just worked on without breaking the
artifact workflow owned by `research_assistant`.

## What Problem This Solves

Consider this sequence:

1. `Give me insights on /home/user/paper.pdf`
2. `What is CIFAR-10 and how was it used here?`
3. `Save it to /home/user/output.md`

Expected behavior:

- step 1 should go to `research_assistant`
- step 2 can go to `chat_agent`, because it is ordinary discussion about the
  current artifact
- step 3 should still save the artifact produced by `research_assistant`

That only works if both agents share the same session contract.

## The Architecture

There are three important pieces:

1. `orchestrator` routes and delegates
2. `research_assistant` creates artifacts
3. `chat_agent` reads shared context and preserves artifact state

The orchestrator is not the final responder. It copies shared session state into
the selected agent before a run, then copies the resulting state back out after
the run finishes.

That state currently includes:

- `last_response`
- `last_artifact_content`
- `last_artifact_responder`
- `last_responder`
- `last_tools_used`
- `artifact_history`

## Step 1: Put The Shared Contract In `BaseAgent`

Do not invent a local chat-specific memory shape.

Put the shared session-state contract on `BaseAgent` so every execution agent
can participate in the same artifact continuity rules.

Minimal shape:

```python
class SessionState(TypedDict, total=False):
    last_response: str
    last_artifact_content: str
    last_artifact_responder: str
    last_responder: str
    last_tools_used: list[str]
    artifact_history: list[dict[str, object]]
```

The base class should also own the lifecycle helpers:

- `empty_session_state()`
- `copy_session_state()`
- `apply_session_state(...)`
- `update_session_state(...)`
- `propose_handoff(...)`

Why this matters:

- `research_assistant` and `chat_agent` now speak the same session language
- the orchestrator can move state across delegation boundaries without special
  cases
- future agents can opt into the same contract automatically

## Step 2: Keep `chat_agent` Tool-Free

`chat_agent` should not compete with `research_assistant` on tool execution.

Its purpose is narrower:

- answer general conversation normally
- answer questions about the current artifact when the shared state is enough
- ask for a handoff when the turn is really a specialized research workflow

That means the class can stay small:

```python
class ChatAgent(BaseAgent):
    name = "chat_agent"
    description = (
        "Handles general conversation and questions about the current session "
        "context or active artifact without using tools."
    )

    def __init__(self, **kwargs) -> None:
        super().__init__(tools=[], **kwargs)
        self.settings = LocalLLMSettings.from_env(
            system_prompt=DEFAULT_SYSTEM_PROMPT
        )
        self.model = LocalChatModel(self.settings)
```

The important part is `tools=[]`. `chat_agent` is not opening files or saving
output. It is only reasoning over the prompt, memory context, and shared session
artifact state.

## Step 3: Build A Session-Aware Prompt

`chat_agent` needs two inputs:

- normal conversation or memory context
- active artifact context from shared session state

The prompt should treat artifact state as supporting evidence, not as a hard
limit on what the agent may discuss.

That is why the prompt should explicitly say:

- it can chat about any general topic
- it can discuss the current artifact when relevant
- it should admit when the available session context is insufficient

The prompt assembly pattern is:

```python
system_prompt = build_system_prompt(
    self.settings.system_prompt,
    context=self._bounded_context(context, limit=1800),
    session_artifact_context=self._session_artifact_context(),
)
messages = build_answer_messages(system_prompt, query)
```

This is the key design point:

- general chat still works when there is no artifact
- artifact-grounded chat works when there is one

## Step 4: Render Shared Artifact Context Compactly

`chat_agent` should not receive the entire previous session verbatim.

Instead, render a compact state block from the shared session fields:

```python
def _session_artifact_context(self) -> str:
    lines = []
    if last_responder:
        lines.append(f"Last responder: {last_responder}")
    if last_artifact_responder:
        lines.append(f"Active artifact owner: {last_artifact_responder}")
    if last_response:
        lines.append(f"Last response excerpt:\\n{last_response}")
    if last_artifact_content:
        lines.append(f"Active artifact excerpt:\\n{last_artifact_content}")
    return "\\n\\n".join(lines)
```

That gives `chat_agent` enough context to answer:

- `What does CIFAR-10 mean here?`
- `What was the paper's novel approach?`
- `Can you explain the limitations more simply?`

without letting the prompt explode in size.

## Step 5: Preserve Artifact Ownership On Chat Turns

This is the most important behavior.

When `chat_agent` responds, it should update the shared session state like this:

```python
def run(self, query: str, context: str = "") -> str:
    response = self._generate_response(query, context=context)
    self.update_session_state(response=response)
    return response
```

Notice what is missing:

- no `artifact_content=...`
- no `artifact_responder=...`

That is deliberate.

Because `BaseAgent.update_session_state(...)` preserves the current artifact
when `artifact_content` is omitted, `chat_agent` can answer the user's question
without stealing ownership of the artifact from `research_assistant`.

So after:

1. `research_assistant` creates insights
2. `chat_agent` explains a term from those insights

the active artifact is still the research artifact.

That is what makes a later `save it` request route back to
`research_assistant`.

## Step 6: Let `research_assistant` Create And Save Artifacts

`research_assistant` is still the artifact-producing agent.

It updates session state with explicit artifact data after:

- direct PDF insight flows
- direct PDF summary flows
- direct save/export follow-ups
- planned execution that actually produced `artifact_content`

This separation is what keeps responsibilities clear:

- `research_assistant` creates or persists artifacts
- `chat_agent` discusses them

## Step 7: Add Handoff Instead Of Reintroducing Deterministic Routing

You still need a handoff path, because some turns that initially land in
`chat_agent` are really research workflows.

Use a structured handoff contract:

```python
@dataclass(slots=True)
class HandoffDecision:
    target_agent: str
    reason: str = ""
    confidence: float = 0.0
```

Then let `chat_agent` propose either:

- answer directly
- hand off to `research_assistant`

The orchestrator validates that handoff through the shared policy layer.

This keeps the system agentic without allowing obvious workflow ownership bugs.

## End-To-End Example

A typical flow now looks like this:

1. User asks for PDF insights.
2. `orchestrator` routes to `research_assistant`.
3. `research_assistant` creates a rendered artifact and stores:
   - `last_response`
   - `last_artifact_content`
   - `last_artifact_responder="research_assistant"`
4. User asks a generic follow-up about a term in the artifact.
5. `orchestrator` can route that to `chat_agent`.
6. `chat_agent` answers using memory context plus `last_artifact_content`.
7. `chat_agent` updates `last_response` and `last_responder`, but preserves:
   - `last_artifact_content`
   - `last_artifact_responder`
   - `artifact_history`
8. User says `save it to /home/user/output.md`.
9. `orchestrator` or the policy layer keeps that request with
   `research_assistant` because the active artifact owner is still
   `research_assistant`.
10. `research_assistant` writes the file.

That is the shared-memory interaction in practice.

## Minimal Build Checklist

If you want to create a `chat_agent` like the one in OmniDex, the minimum steps
are:

1. inherit from `BaseAgent`
2. keep `tools=[]`
3. create a prompt that accepts both general context and session artifact state
4. render a compact artifact context block from the shared session fields
5. call `update_session_state(response=response)` without replacing artifact
   fields
6. implement `propose_handoff(...)` so specialized research turns can move to
   `research_assistant`
7. let the orchestrator copy session state in and out of the agent

## Common Failure Modes

- `chat_agent` overwrites `last_artifact_content`
  This breaks later `save it` requests because the artifact owner is lost.

- `chat_agent` is given tools
  This blurs the boundary with `research_assistant` and makes routing harder.

- session state lives only inside the orchestrator
  Then delegated agents cannot reason about the active artifact directly.

- prompts describe `chat_agent` as artifact-only
  Then general-topic chat becomes artificially constrained.

- prompts describe `chat_agent` as unrestricted file agent
  Then it starts competing with `research_assistant` for research workflows.

## Summary

The correct pattern is:

- shared state contract in `BaseAgent`
- `research_assistant` owns artifact creation and persistence
- `chat_agent` owns general discussion and artifact-aware follow-up chat
- orchestrator copies shared state across delegation boundaries
- handoffs remain agentic, but artifact continuity remains stable
