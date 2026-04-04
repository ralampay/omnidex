"""Tool for extracting output-writing intent from user input."""

from __future__ import annotations

from omnidex.tools.base import BaseTool
from omnidex.utils.paths import extract_output_filename, has_explicit_output_request


class OutputRequestTool(BaseTool):
    """Detect whether the user asked to save the result and where."""

    name = "extract_output_request"
    output_fields = ("filename", "write_output", "content")

    def run(self, query: str) -> dict[str, object]:
        """Extract the requested output filename, if any."""
        filename = extract_output_filename(query)
        write_output = has_explicit_output_request(query)
        return {
            "status": "ok",
            "filename": filename,
            "write_output": write_output,
            "content": filename or "",
        }
