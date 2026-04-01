"""Output formatting tool for agent responses."""

from __future__ import annotations

from omnidex.tools.base import BaseTool


class CreateOutputTool(BaseTool):
    """Finalize agent output into a clean display string."""

    name = "create_output"

    def run(self, content: str, *, title: str | None = None) -> str:
        """Normalize final output text for display."""
        normalized_content = content.strip()
        if not normalized_content:
            normalized_content = "No output generated."

        normalized_title = (title or "").strip()
        if not normalized_title:
            return normalized_content

        return f"# {normalized_title}\n\n{normalized_content}"
