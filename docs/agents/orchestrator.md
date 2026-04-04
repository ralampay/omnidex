# orchestrator

`orchestrator` is the top-level interactive agent for OmniDex.

It is responsible for:

- receiving the user turn
- gathering memory and session context
- choosing the best initial delegate
- allowing bounded agent-to-agent handoffs
- copying shared session state across delegation boundaries
- rendering the delegated agent's response in the CLI

It is not the final chat responder. Its job is delegation, validation, and
state coordination.

## Purpose

The orchestrator exists so specialist agents can stay narrow:

- `chat_agent` handles general conversation and artifact-aware follow-up chat
- `research_assistant` handles research workflows, PDF analysis, and
  save/export flows

Without an orchestrator, each agent would need to understand routing,
follow-up ownership, and cross-agent continuity on its own.

## Registered Agents

The orchestrator currently registers:

- `chat_agent`
- `research_assistant`

The registered set is used for:

- initial route selection
- handoff proposals
- routing prompt construction

## Routing Model

The orchestrator uses a policy-validated agentic routing flow:

1. gather bounded memory context
2. gather bounded session artifact context
3. ask the local model for the best initial route
4. validate the proposed route through the shared policy layer
5. delegate to the selected agent

The model chooses the initial route, but the policy validator can override it
for hard constraints.

Examples of hard constraints:

- explicit PDF workflows must stay with `research_assistant`
- explicit save/export follow-ups that research can handle must stay with
  `research_assistant`

## Handoff Model

Initial routing is not always final routing.

After the first route is chosen, the orchestrator gives the active agent a
chance to propose a handoff.

Current handoff flow:

1. apply shared session state to the active agent
2. ask that agent for `propose_handoff(...)`
3. validate the proposed handoff through the shared policy layer
4. accept the handoff only if:
   - it is valid
   - it targets a different registered agent
   - the confidence is at least `0.5`
   - the target was not already visited in this turn

The orchestrator also keeps a bounded handoff chain so a turn cannot bounce
forever.

## Shared Session State

The orchestrator is the session boundary between turns.

It keeps a shared session-state payload and moves it between delegated agents.
That payload includes:

- `last_response`
- `last_artifact_content`
- `last_artifact_responder`
- `last_responder`
- `last_tools_used`
- `artifact_history`

Before running an agent, the orchestrator applies the current session state to
that agent.

After the delegated run finishes, the orchestrator copies the resulting session
state back and preserves the active artifact when a chat turn did not replace
it.

This is what allows a flow like:

1. `research_assistant` creates insights
2. `chat_agent` explains a term from those insights
3. `research_assistant` still handles `save it to ...`

## Memory Context

The orchestrator combines:

- short-term conversational memory
- long-term memory retrieval
- session artifact state

The result is used as routing context so short follow-ups such as `save it` or
`what does that term mean here?` can still be interpreted correctly.

This is different from artifact continuity itself:

- memory helps with retrieval and broader context
- session state preserves the exact active artifact and its owner

## CLI Responsibilities

The orchestrator also owns the interactive terminal behavior:

- prompt loop
- slash commands such as `/help`, `/clear`, and `/exit`
- route trace output
- delegated-agent status messages
- final response rendering

It uses `rich` for:

- banners
- colored event lines
- markdown rendering
- spinners during routing and delegated execution

## Failure Behavior

If route selection fails, the orchestrator falls back to `chat_agent`.

If a handoff proposal is invalid or blocked by policy, the orchestrator keeps
the current agent.

If the user clears the session with `/clear` or `/reset`, the orchestrator
clears short-term memory and resets the shared session state.

## Design Rules

- Keep the orchestrator focused on delegation, not domain work.
- Keep initial route choice model-proposed.
- Keep route and handoff validation narrow and explicit.
- Preserve shared session state across delegation boundaries.
- Treat artifact ownership separately from generic chat continuity.
- Do not let the orchestrator become a third specialist agent.

## Related Docs

- [Chat Agent](./chat_agent.md)
- [Agents Overview](./README.md)
- [Orchestration Routing](../engine/orchestration_routing.md)
- [State Artifacts](../engine/state_artifacts.md)
