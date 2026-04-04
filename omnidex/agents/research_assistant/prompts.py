"""Prompt builders for the research assistant."""

from __future__ import annotations

DEFAULT_SYSTEM_PROMPT = (
    "You are a research assistant. Provide clear, structured, and "
    "well-reasoned answers. Explain reasoning when it is helpful. "
    "Do not hallucinate unknown facts, and if you are uncertain, "
    "say so explicitly."
)


def build_summary_chunk_messages(
    system_prompt: str,
    chunk: str,
    *,
    focus_query: str = "",
) -> list[dict[str, str]]:
    """Build messages for chunk-level summarization."""
    focus_instructions = ""
    if focus_query.strip():
        focus_instructions = (
            f"\n\nPrioritize details relevant to this query:\n{focus_query.strip()}\n"
            "Keep concrete facts that help answer it, especially named entities, "
            "metrics, conclusions, and caveats."
        )
    return [
        {
            "role": "system",
            "content": (
                f"{system_prompt.strip()}\n\n"
                "Summarize the provided document excerpt. Focus on the main "
                "arguments, findings, and notable details. Keep the summary "
                "concise and factual, but preserve concrete information that "
                "will matter later. Prefer up to 6 short bullet points over "
                "generic wording."
                f"{focus_instructions}"
            ),
        },
        {"role": "user", "content": chunk},
    ]


def build_combine_summaries_messages(
    system_prompt: str,
    summaries: list[str],
    *,
    focus_query: str = "",
) -> list[dict[str, str]]:
    """Build messages for summary-combination passes."""
    focus_instructions = ""
    if focus_query.strip():
        focus_instructions = (
            f" Focus the merged summary on answering this query: {focus_query.strip()}. "
            "Keep the most relevant facts, numbers, conclusions, and caveats."
        )
    return [
        {
            "role": "system",
            "content": (
                f"{system_prompt.strip()}\n\n"
                "Combine the partial summaries into a single clear research "
                "summary. Remove repetition and preserve the key findings. "
                "Keep the final response short and focused."
                f"{focus_instructions}"
            ),
        },
        {"role": "user", "content": "\n\n".join(summaries)},
    ]


def build_extract_report_insights_messages(
    system_prompt: str,
    text: str,
    *,
    focus_query: str = "",
) -> list[dict[str, str]]:
    """Build messages for structured insight extraction from document text."""
    focus_instructions = ""
    if focus_query.strip():
        focus_instructions = (
            f"\n\nPrioritize content relevant to this query:\n{focus_query.strip()}\n"
            "Prefer the facts, findings, methods, metrics, and caveats that best answer it."
        )
    return [
        {
            "role": "system",
            "content": (
                f"{system_prompt.strip()}\n\n"
                "You are extracting structured insights from document text that has "
                "already been provided to you.\n"
                "Do not mention inability to access files, tools, paths, or the local filesystem.\n"
                "Use only the provided document text.\n"
                "Return ONLY valid JSON with these keys:\n"
                "- title: string\n"
                "- keywords: list of short strings\n"
                "- strengths: list of concise factual bullets\n"
                "- novel_approach: string\n"
                "- gaps_and_limitations: list of concise factual bullets\n"
                "Keep outputs concrete and specific. Avoid generic filler."
                f"{focus_instructions}"
            ),
        },
        {"role": "user", "content": text},
    ]


def build_system_prompt(
    system_prompt: str,
    *,
    context: str = "",
    pdf_text: str = "",
) -> str:
    """Build a system prompt with optional context and PDF excerpts."""
    prompt = system_prompt.strip()
    if context.strip():
        prompt = (
            f"{prompt}\n\n"
            "Use the context below if relevant. Do not repeat it verbatim "
            "unless the user asks.\n\n"
            f"{context.strip()}"
        )
    if pdf_text.strip():
        prompt = (
            f"{prompt}\n\n"
            "Use the PDF excerpts below when answering questions about the "
            "document. Base your answer on this content and say when the "
            "document does not support a claim.\n\n"
            f"{pdf_text.strip()}"
        )
    return prompt


def build_answer_messages(
    system_prompt: str,
    query: str,
) -> list[dict[str, str]]:
    """Build messages for a final answer generation pass."""
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
    """Build messages for a structured research-agent handoff decision."""
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
                "You are deciding whether research_assistant should answer the user "
                "directly or hand the turn to another agent.\n"
                "Research_assistant should keep turns that request explicit research "
                "workflows such as reading PDFs, summarizing papers, extracting "
                "insights, structured research analysis, or save/export actions tied "
                "to a research artifact.\n"
                "Research_assistant should hand off to chat_agent when the user is "
                "asking a generic question about the latest content, asking what a "
                "specific term means, or continuing normal discussion about an "
                "already-generated artifact rather than asking for a new research task.\n"
                "Return ONLY valid JSON in one of these forms:\n"
                '{"action":"answer","confidence":0.0}\n'
                '{"action":"handoff","target_agent":"chat_agent","reason":"short reason","confidence":0.0}\n\n'
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
