"""Markdown formatter for report insight summaries."""

from __future__ import annotations

from typing import Iterable

from omnidex.tools.base import BaseTool


class ReportInsightsTool(BaseTool):
    """Format report insights into a consistent markdown structure."""

    name = "report_insights"

    def run(
        self,
        *,
        title: str,
        keywords: Iterable[str] | str | None = None,
        strengths: Iterable[str] | str | None = None,
        novel_approach: str = "",
        gaps_and_limitations: Iterable[str] | str | None = None,
    ) -> str:
        """Return a markdown report insight summary."""
        normalized_title = title.strip() or "Untitled Report"
        normalized_keywords = self._normalize_items(keywords)
        normalized_strengths = self._normalize_items(strengths)
        normalized_gaps = self._normalize_items(gaps_and_limitations)
        normalized_novel_approach = novel_approach.strip() or "None identified."

        keyword_line = ", ".join(normalized_keywords) if normalized_keywords else "None"
        strength_lines = self._bullet_list(normalized_strengths)
        gap_lines = self._bullet_list(normalized_gaps)

        return (
            f"# {normalized_title}\n\n"
            f"## Keywords\n"
            f"{keyword_line}\n\n"
            f"## Strengths\n"
            f"{strength_lines}\n\n"
            f"## Novel Approach (if any)\n\n"
            f"{normalized_novel_approach}\n\n"
            f"## Gaps and Limitations (in bullet points)\n\n"
            f"{gap_lines}"
        )

    def _normalize_items(self, items: Iterable[str] | str | None) -> list[str]:
        """Normalize a string or iterable of strings into a cleaned list."""
        if items is None:
            return []
        if isinstance(items, str):
            raw_items = [part.strip() for part in items.split(",")]
        else:
            raw_items = [str(item).strip() for item in items]
        return [item for item in raw_items if item]

    def _bullet_list(self, items: list[str]) -> str:
        """Render a markdown bullet list with a safe empty fallback."""
        if not items:
            return "* None identified."
        return "\n".join(f"* {item}" for item in items)
