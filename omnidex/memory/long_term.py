"""Long-term memory storage for durable conversation facts."""

from __future__ import annotations

import json
from pathlib import Path
import re


class LongTermMemory:
    """Persistent store for durable memory items.

    This implementation uses a small JSON file with keyword-based retrieval so it
    can later be replaced by a vector store without changing the orchestrator API.
    """

    def __init__(self, storage_path: Path, max_items: int = 250) -> None:
        self.storage_path = storage_path
        self.max_items = max_items
        self._items = self._load()

    def add(self, memory_item: str) -> None:
        """Persist a memory item if it is not already stored."""
        normalized_item = memory_item.strip()
        if not normalized_item:
            return

        existing = {item.casefold() for item in self._items}
        if normalized_item.casefold() in existing:
            return

        self._items.append(normalized_item)
        if len(self._items) > self.max_items:
            self._items = self._items[-self.max_items :]
        self._save()

    def search(self, query: str, limit: int = 3) -> list[str]:
        """Return the most relevant stored memories for a query."""
        query_tokens = self._tokenize(query)
        if not query_tokens:
            return self._items[-limit:]

        scored_items: list[tuple[int, int, str]] = []
        for index, item in enumerate(self._items):
            item_tokens = self._tokenize(item)
            overlap = len(query_tokens & item_tokens)
            if overlap == 0 and query.casefold() not in item.casefold():
                continue
            scored_items.append((overlap, index, item))

        scored_items.sort(key=lambda row: (row[0], row[1]), reverse=True)
        return [item for _, _, item in scored_items[:limit]]

    def clear(self) -> None:
        """Remove all stored long-term memories."""
        self._items = []
        self._save()

    def _load(self) -> list[str]:
        """Load memory items from disk if they exist."""
        if not self.storage_path.exists():
            return []

        raw_content = json.loads(self.storage_path.read_text(encoding="utf-8"))
        if not isinstance(raw_content, list):
            return []
        return [str(item).strip() for item in raw_content if str(item).strip()]

    def _save(self) -> None:
        """Write memory items to disk."""
        self.storage_path.parent.mkdir(parents=True, exist_ok=True)
        self.storage_path.write_text(
            json.dumps(self._items, indent=2),
            encoding="utf-8",
        )

    def _tokenize(self, content: str) -> set[str]:
        """Split text into lowercase keyword tokens."""
        return {token for token in re.findall(r"[a-zA-Z0-9_]+", content.casefold())}
