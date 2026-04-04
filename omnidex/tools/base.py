"""Base abstractions for OmniDex tools."""

from __future__ import annotations

from abc import ABC, abstractmethod
import json
from typing import Any


class BaseTool(ABC):
    """Base command-style class for agent tools."""

    name = "base_tool"
    output_fields: tuple[str, ...] = ("content",)

    def _coerce_text(self, value: Any, *, fallback: str = "") -> str:
        """Normalize scalar or structured tool inputs into text."""
        if value is None:
            return fallback
        if isinstance(value, str):
            return value.strip()
        if isinstance(value, dict):
            for key in ("content", "answer", "summary", "text", "value", "file_path"):
                candidate = value.get(key)
                if isinstance(candidate, str) and candidate.strip():
                    return candidate.strip()
            return json.dumps(value, ensure_ascii=False, sort_keys=True)
        if isinstance(value, list):
            return "\n".join(str(item).strip() for item in value if str(item).strip())
        return str(value).strip()

    @abstractmethod
    def run(self, *args: Any, **kwargs: Any) -> Any:
        """Execute the tool."""
