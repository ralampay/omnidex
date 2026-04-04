"""Prompt builders for the chat agent."""

from __future__ import annotations

DEFAULT_SYSTEM_PROMPT = (
    "You are ChatAgent, the general conversation agent for OmniDex. "
    "You can chat about any general topic, answer broad questions, explain concepts, "
    "brainstorm ideas, and discuss the current conversation or active session artifact "
    "when those are relevant. "
    "You do not have tools in this turn. Use the provided context and artifact state "
    "when they matter, but do not act as if you are limited to them. "
    "When the available context does not support a claim about the current session "
    "content, say so clearly."
)


def build_system_prompt(
    system_prompt: str,
    *,
    context: str = "",
    session_artifact_context: str = "",
) -> str:
    """Build a system prompt with optional shared context and artifact state."""
    prompt = system_prompt.strip()
    if context.strip():
        prompt = (
            f"{prompt}\n\n"
            "Use the conversation and memory context below when it is relevant. "
            "It is supporting context, not a hard boundary on what you may discuss. "
            "Do not repeat it verbatim unless the user asks.\n\n"
            f"{context.strip()}"
        )
    if session_artifact_context.strip():
        prompt = (
            f"{prompt}\n\n"
            "Use the active session artifact state below when the user asks about "
            "the current content, a previous answer, or the latest generated artifact. "
            "Prefer this artifact state over vague recollection for session-specific "
            "questions, but continue to answer ordinary general-topic questions normally.\n\n"
            f"{session_artifact_context.strip()}"
        )
    return prompt


def build_answer_messages(
    system_prompt: str,
    query: str,
) -> list[dict[str, str]]:
    """Build messages for a chat response generation pass."""
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": query},
    ]


def build_handoff_messages(
    *,
    system_prompt: str,
    query: str,
    context: str = "",
    session_artifact_context: str = "",
    available_agents: tuple[str, ...] = (),
) -> list[dict[str, str]]:
    """Build messages for a structured chat-agent handoff decision."""
    options = "\n".join(f"- {name}" for name in available_agents)
    contextual_notes = []
    if context.strip():
        contextual_notes.append(
            "Conversation and memory context:\n"
            f"{context.strip()}"
        )
    if session_artifact_context.strip():
        contextual_notes.append(
            "Session artifact state:\n"
            f"{session_artifact_context.strip()}"
        )
    context_block = "\n\n".join(contextual_notes)
    return [
        {
            "role": "system",
            "content": (
                f"{system_prompt.strip()}\n\n"
                "You are deciding whether chat_agent should answer the user directly "
                "or hand the turn to another agent.\n"
                "Choose handoff only when the user is asking for a new specialized "
                "research workflow such as analyzing a PDF, summarizing a paper, "
                "extracting structured insights, or persisting/transformation steps "
                "that should stay with the research artifact owner.\n"
                "If the user is asking a generic question about any topic, asking a "
                "question about the current content, asking what a term means, or "
                "continuing discussion about the latest artifact, chat_agent should "
                "answer directly.\n"
                "Return ONLY valid JSON in one of these forms:\n"
                '{"action":"answer","confidence":0.0}\n'
                '{"action":"handoff","target_agent":"research_assistant","reason":"short reason","confidence":0.0}\n\n'
                f"Available agents:\n{options}"
            ),
        },
        {
            "role": "user",
            "content": (
                f"User query:\n{query.strip()}\n\n"
                f"{context_block}" if context_block else f"User query:\n{query.strip()}"
            ),
        },
    ]
