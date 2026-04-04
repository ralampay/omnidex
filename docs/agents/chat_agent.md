# chat_agent

`chat_agent` is the dedicated general-conversation agent used by the
`orchestrator`.

## Purpose

It handles:

- generic questions about any topic
- generic questions about the current conversation
- follow-up questions about the latest generated content
- questions that should rely on shared memory or the active artifact rather than
  tool execution

It does not own any tools. Its job is to answer from:

- retrieved memory context
- short-term conversation context
- shared session artifact state copied in by the orchestrator

## Why It Exists

This keeps the architecture consistent:

- the orchestrator routes
- `research_assistant` handles research and file-oriented workflows
- `chat_agent` handles generic conversational turns

That means even plain chat follows the same delegation path as the specialized
agents.

## Handoffs

`chat_agent` can still request a handoff when the user is really asking for a
specialized research workflow.

That handoff is not hard-coded in the agent. The current flow is:

1. `chat_agent` uses the local model to propose either `answer` or `handoff`
2. the orchestrator validates the proposal through the shared policy layer
3. the turn stays in `chat_agent` unless the proposal is valid and confident

This keeps ordinary chat broad and flexible while preserving hard workflow
ownership for file-backed research tasks.

## Shared Context

Before delegation, the orchestrator copies session artifact state into
`chat_agent`.

The most important fields are:

- `last_response`
- `last_artifact_content`
- `last_artifact_responder`
- `last_responder`
- `last_tools_used`
- `artifact_history`

`chat_agent` preserves the active artifact fields when it answers a generic
question. That allows a later follow-up such as `save it to ./notes.md` to route
back to `research_assistant` even if the previous turn was answered by
`chat_agent`.
