"""Tool for determining whether the user explicitly requested output writing."""

from __future__ import annotations

from omnidex.tools.base import BaseTool
from omnidex.utils.paths import has_explicit_output_request


class OutputWriteIntentTool(BaseTool):
    """Detect whether the user explicitly asked to save, write, or export output."""

    name = "determine_output_write"
    output_fields = ("write_output", "content")

    def run(self, query: str) -> dict[str, object]:
        """Return whether the current request explicitly asks for persistence."""
        write_output = has_explicit_output_request(query)
        return {
            "status": "ok",
            "write_output": write_output,
            "content": "write_output" if write_output else "",
        }
