"""Memory manager used by the orchestrator agent."""

from __future__ import annotations

from pathlib import Path
import re

from .long_term import LongTermMemory
from .short_term import ShortTermMemory


class MemoryManager:
    """Coordinates short-term and long-term memory for one agent session."""

    DURABLE_FACT_PATTERNS = (
        "i am",
        "i'm",
        "i prefer",
        "i want",
        "my goal is",
        "my name is",
        "i work on",
        "i am working on",
    )

    def __init__(
        self,
        *,
        short_term_limit: int = 5,
        long_term_path: Path | None = None,
        long_term_limit: int = 250,
    ) -> None:
        memory_path = long_term_path or Path(".omnidex/orchestrator_memory.json")
        self.short_term = ShortTermMemory(max_interactions=short_term_limit)
        self.long_term = LongTermMemory(storage_path=memory_path, max_items=long_term_limit)

    def add_interaction(self, role: str, content: str) -> None:
        """Append one message to short-term conversational memory."""
        self.short_term.add(role=role, content=content)

    def get_context(self, query: str) -> str:
        """Build the structured memory context for the next LLM call."""
        recent_messages = self.short_term.get_messages()
        relevant_memories = self.long_term.search(query)

        recent_lines = ["Recent Conversation:"]
        if recent_messages:
            recent_lines.extend(
                f"{message['role'].capitalize()}: {message['content']}"
                for message in recent_messages
            )
        else:
            recent_lines.append("(no recent conversation)")

        memory_lines = ["Relevant Memory:"]
        if relevant_memories:
            memory_lines.extend(f"* {memory}" for memory in relevant_memories)
        else:
            memory_lines.append("* (no relevant memory)")

        return "\n".join([*recent_lines, "", *memory_lines])

    def extract_and_store(self, user_input: str, assistant_response: str) -> None:
        """Persist durable facts expressed by the user in the latest interaction."""
        del assistant_response
        for memory_item in self._extract_durable_facts(user_input):
            self.long_term.add(memory_item)

    def clear_short_term(self) -> None:
        """Reset only the rolling conversation window."""
        self.short_term.clear()

    def _extract_durable_facts(self, content: str) -> list[str]:
        """Return durable user facts that should be kept in long-term memory."""
        extracted: list[str] = []
        candidates = re.split(r"[\n\r]+|(?<=[.!?])\s+", content.strip())
        for candidate in candidates:
            normalized = " ".join(candidate.split())
            if not normalized:
                continue

            lowered = normalized.casefold()
            if not any(keyword in lowered for keyword in self.DURABLE_FACT_PATTERNS):
                continue

            extracted.append(normalized.rstrip(".!?"))
        return extracted
