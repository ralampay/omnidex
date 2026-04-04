"""Tool for deriving report insight fields from document text."""

from __future__ import annotations

import re

from omnidex.agents.research_assistant.commands.summarize_pdf import SummarizePdfCommand
from omnidex.agents.research_assistant.prompts import build_extract_report_insights_messages
from omnidex.tools.base import BaseTool
from omnidex.utils.json_tools import load_json_object
from omnidex.utils.text import extract_keywords, extract_title


class ExtractReportInsightsTool(BaseTool):
    """Extract report-insight fields from source text."""

    name = "extract_report_insights"
    output_fields = (
        "title",
        "keywords",
        "strengths",
        "novel_approach",
        "gaps_and_limitations",
        "content",
    )

    def __init__(self, *, summarize_command: SummarizePdfCommand) -> None:
        self.summarize_command = summarize_command

    def run(self, text: str, focus_query: str = "") -> dict[str, object]:
        """Return structured insight fields derived from the source text."""
        normalized_text = self._coerce_text(text)
        normalized_focus_query = self._coerce_text(focus_query)
        if not normalized_text:
            return {
                "status": "error",
                "content": "Error: No source text was provided for report insight extraction.",
                "title": "Untitled Report",
                "keywords": [],
                "strengths": [],
                "novel_approach": "",
                "gaps_and_limitations": [],
            }
        source_text = self.summarize_command.select_summary_source(
            normalized_text,
            normalized_focus_query,
        )
        raw_response = self.summarize_command.model.generate_text(
            build_extract_report_insights_messages(
                self.summarize_command.system_prompt,
                source_text,
                focus_query=normalized_focus_query,
            ),
            stream=False,
        )
        payload = load_json_object(raw_response)

        if payload is None:
            return self._fallback_from_text(normalized_text, source_text)

        strengths = self._normalize_items(payload.get("strengths"))[:4]
        gaps = self._normalize_items(payload.get("gaps_and_limitations"))[:4]
        keywords = self._normalize_items(payload.get("keywords"))[:6]
        title = self._coerce_text(payload.get("title")) or extract_title(normalized_text)
        novel_approach = (
            self._coerce_text(payload.get("novel_approach"))
            or "No clearly novel approach was identified from the extracted text."
        )

        return {
            "status": "ok",
            "title": title,
            "keywords": keywords or extract_keywords(normalized_text),
            "strengths": strengths,
            "novel_approach": novel_approach,
            "gaps_and_limitations": gaps
            or ["Potential limitations were not explicitly identified in the extracted text."],
            "content": raw_response,
        }

    def _fallback_from_text(self, full_text: str, source_text: str) -> dict[str, object]:
        """Provide deterministic fallback insights when JSON extraction fails."""
        strengths = [
            sentence.strip(" -.")
            for sentence in re.split(r"[.\n]+", source_text)
            if sentence.strip()
        ][:3]
        return {
            "status": "ok",
            "title": extract_title(full_text),
            "keywords": extract_keywords(full_text),
            "strengths": strengths,
            "novel_approach": (
                strengths[0]
                if strengths
                else "No clearly novel approach was identified from the extracted text."
            ),
            "gaps_and_limitations": [
                "Potential limitations were not explicitly identified in the extracted text.",
            ],
            "content": "\n".join(strengths),
        }

    def _normalize_items(self, value: object) -> list[str]:
        """Normalize model-produced list-like values into cleaned strings."""
        if value is None:
            return []
        if isinstance(value, str):
            items = [
                part.strip(" -*")
                for part in re.split(r"[\n,]+", value)
                if part.strip(" -*")
            ]
            return items
        if isinstance(value, list):
            return [self._coerce_text(item) for item in value if self._coerce_text(item)]
        return []
