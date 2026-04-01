"""Short-term conversational memory."""

from __future__ import annotations

from collections import deque
from typing import Deque


class ShortTermMemory:
    """In-memory rolling window of recent conversation messages."""

    def __init__(self, max_interactions: int = 5) -> None:
        self.max_interactions = max_interactions
        self._messages: Deque[dict[str, str]] = deque(maxlen=max_interactions)

    def add(self, role: str, content: str) -> None:
        """Append one message to the recent conversation window."""
        message = {"role": role, "content": content.strip()}
        if message["content"]:
            self._messages.append(message)

    def get_messages(self) -> list[dict[str, str]]:
        """Return the current short-term memory window."""
        return list(self._messages)

    def clear(self) -> None:
        """Reset the short-term memory window."""
        self._messages.clear()
