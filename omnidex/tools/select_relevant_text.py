"""Tool for extracting the most relevant text slices for a query."""

from __future__ import annotations

from typing import Callable

from omnidex.tools.base import BaseTool
from omnidex.utils.text import chunk_text, tokenize_for_matching


class SelectRelevantTextTool(BaseTool):
    """Select a bounded set of excerpts relevant to the current query."""

    name = "select_relevant_text"
    output_fields = ("text", "content")

    def __init__(
        self,
        *,
        log: Callable[[str], None],
        input_char_budget: int,
        chunk_size: int = 1200,
    ) -> None:
        self.log = log
        self.input_char_budget = input_char_budget
        self.chunk_size = chunk_size

    def run(self, text: str, query: str) -> dict[str, object]:
        """Return selected excerpts ranked against the query."""
        budget = self.input_char_budget
        normalized_text = self._coerce_text(text)
        normalized_query = self._coerce_text(query)
        chunks = chunk_text(normalized_text, chunk_size=self.chunk_size)
        if not chunks:
            return {"status": "warning", "text": "", "content": ""}

        query_terms = tokenize_for_matching(normalized_query)
        self.log(
            f"Scoring {len(chunks)} text chunk(s) against "
            f"{len(query_terms)} query term(s)."
        )
        scored_chunks: list[tuple[int, int, str]] = []
        for index, chunk in enumerate(chunks):
            chunk_terms = tokenize_for_matching(chunk)
            score = len(query_terms & chunk_terms)
            scored_chunks.append((score, -index, chunk))

        scored_chunks.sort(reverse=True)
        selected: list[str] = []
        total_length = 0
        for score, _, chunk in scored_chunks:
            if score == 0 and selected:
                break
            snippet = chunk.strip()
            if not snippet:
                continue
            projected_length = total_length + len(snippet) + 12
            if selected and projected_length > budget:
                continue
            selected.append(snippet)
            total_length = projected_length
            if total_length >= budget:
                break

        if not selected:
            selected_text = chunks[0][:budget].strip()
        else:
            selected_text = "\n\n---\n\n".join(selected)
        return {
            "status": "ok",
            "text": selected_text,
            "content": selected_text,
        }
