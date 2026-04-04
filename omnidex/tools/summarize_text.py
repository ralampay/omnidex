"""Tool for summarizing long-form text with a local model."""

from __future__ import annotations

from omnidex.agents.research_assistant.commands.summarize_pdf import SummarizePdfCommand
from omnidex.tools.base import BaseTool


class SummarizeTextTool(BaseTool):
    """Summarize a long text input into a concise factual summary."""

    name = "summarize_text"
    output_fields = ("summary", "content")

    def __init__(
        self,
        *,
        summarize_command: SummarizePdfCommand,
    ) -> None:
        self.summarize_command = summarize_command

    def run(self, text: str, focus_query: str = "") -> dict[str, object]:
        """Summarize the text, using the focus query to trim source text when helpful."""
        normalized_text = self._coerce_text(text)
        normalized_focus_query = self._coerce_text(focus_query)
        if not normalized_text:
            return {
                "status": "error",
                "summary": "",
                "content": "Error: No source text was provided for summarization.",
            }
        summary = self.summarize_command.execute_with_focus(
            normalized_text,
            focus_query=normalized_focus_query,
        )
        return {
            "status": "ok",
            "summary": summary,
            "content": summary,
        }
