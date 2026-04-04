"""Output formatting tool for agent responses."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from omnidex.tools.base import BaseTool


class CreateOutputTool(BaseTool):
    """Finalize agent output into a clean display string."""

    name = "create_output"
    output_fields = ("content", "artifact_content", "saved_path")

    def run(
        self,
        content: Any,
        *,
        title: Any | None = None,
        filename: Any | None = None,
        write_output: Any | None = False,
    ) -> dict[str, object]:
        """Normalize final output text for display and optionally write it to disk."""
        normalized_content = self._coerce_text(content)
        if not normalized_content:
            normalized_content = "No output generated."

        normalized_title = self._coerce_text(title)
        rendered_output = normalized_content
        if normalized_title:
            rendered_output = f"# {normalized_title}\n\n{normalized_content}"

        normalized_filename = self._coerce_text(filename)
        should_write_output = bool(write_output)
        if not normalized_filename or not should_write_output:
            return {
                "status": "ok",
                "content": rendered_output,
                "artifact_content": rendered_output,
                "saved_path": None,
            }

        output_path = Path(normalized_filename).expanduser()
        try:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_text(rendered_output + "\n", encoding="utf-8")
        except OSError as exc:
            return {
                "status": "error",
                "content": (
                    f"{rendered_output}\n\n"
                    f"[Output file error: could not write to {output_path}: {exc}]"
                ),
                "artifact_content": rendered_output,
                "saved_path": None,
            }

        return {
            "status": "ok",
            "content": (
                f"{rendered_output}\n\n"
                f"[Saved output to {output_path.resolve()}]"
            ),
            "artifact_content": rendered_output,
            "saved_path": str(output_path.resolve()),
        }
