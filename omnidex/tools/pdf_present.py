"""Tool for detecting PDF references in user input."""

from __future__ import annotations

from omnidex.tools.base import BaseTool
from omnidex.utils.paths import find_path_with_suffix


class PDFPresentTool(BaseTool):
    """Detect whether the user request references a PDF path."""

    name = "pdf_present"
    output_fields = ("found", "file_path", "content")

    def run(self, query: str) -> dict[str, object]:
        """Return whether a PDF path is present and the detected path."""
        file_path = find_path_with_suffix(query, ".pdf")
        return {
            "status": "ok",
            "found": bool(file_path),
            "file_path": file_path,
            "content": file_path or "",
        }

