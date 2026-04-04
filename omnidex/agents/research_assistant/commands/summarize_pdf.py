"""Command for PDF summarization workflows."""

from __future__ import annotations

from typing import Callable

from omnidex.agents.research_assistant.prompts import (
    build_combine_summaries_messages,
    build_summary_chunk_messages,
)
from omnidex.utils.text import chunk_text, tokenize_for_matching

WARNING_NO_TEXT = "Warning: PDF contains no extractable text."

SUMMARY_QUERY_STOP_WORDS = {
    "summarize",
    "summary",
    "summarise",
    "pdf",
    "document",
    "report",
    "overview",
    "insights",
}


class SummarizePdfCommand:
    """Summarize PDF text through chunking and refinement passes."""

    def __init__(
        self,
        *,
        model: object,
        system_prompt: str,
        log: Callable[[str], None],
        summary_chunk_size: int,
        max_summary_chunks: int,
        input_char_budget: int,
    ) -> None:
        self.model = model
        self.system_prompt = system_prompt
        self.log = log
        self.summary_chunk_size = summary_chunk_size
        self.max_summary_chunks = max_summary_chunks
        self.input_char_budget = input_char_budget

    def execute(self, text: str) -> str:
        """Summarize long PDF text by chunking, then refining the result."""
        return self.execute_with_focus(text, focus_query="")

    def execute_with_focus(self, text: str, *, focus_query: str = "") -> str:
        """Summarize text with optional query-guided source selection."""
        selected_chunks = self.select_summary_chunks(text, focus_query)
        if not selected_chunks:
            return WARNING_NO_TEXT

        self.log(
            f"Summarizing PDF across {len(selected_chunks)} selected chunk(s)."
        )
        partial_summaries: list[str] = []
        for index, chunk in enumerate(selected_chunks, start=1):
            self.log(
                f"Summarizing chunk {index}/{len(selected_chunks)} "
                f"({len(chunk)} characters)."
            )
            summary = self._summarize_chunk(chunk, focus_query=focus_query)
            self.log(
                f"Finished chunk {index}/{len(selected_chunks)} summary "
                f"({len(summary)} characters)."
            )
            partial_summaries.append(summary)

        self.log(
            f"Generated {len(partial_summaries)} partial summary chunk(s); "
            "starting refinement."
        )
        return self._refine_summaries(partial_summaries, focus_query=focus_query)

    def select_summary_source(self, text: str, query: str) -> str:
        """Choose representative PDF excerpts for faster summarization."""
        return "\n\n---\n\n".join(self.select_summary_chunks(text, query))

    def select_summary_chunks(self, text: str, query: str) -> list[str]:
        """Choose representative PDF chunks for bounded summarization."""
        selection_chunk_size = min(
            self.input_char_budget,
            max(self.summary_chunk_size * 3, 4000),
        )
        selection_chunks = chunk_text(text, chunk_size=selection_chunk_size)
        if not selection_chunks:
            return []
        if len(selection_chunks) <= self.max_summary_chunks:
            self.log(
                f"PDF fits within {len(selection_chunks)} selection chunk(s); "
                "using all chunks for summarization."
            )
            return [
                chunk.strip() for chunk in selection_chunks if chunk.strip()
            ]

        query_terms = tokenize_for_matching(query) - SUMMARY_QUERY_STOP_WORDS
        total_chunks = len(selection_chunks)
        selected_indices: list[int] = [0]
        scored_candidates: list[tuple[int, int]] = []

        self.log(
            f"Selecting up to {self.max_summary_chunks} representative summary "
            f"chunk(s) from {total_chunks} total selection chunk(s) "
            f"using a selection chunk size of {selection_chunk_size}."
        )

        for index, chunk in enumerate(selection_chunks):
            if index == 0:
                continue
            score = 0
            if query_terms:
                chunk_terms = tokenize_for_matching(chunk)
                score = len(query_terms & chunk_terms)
            scored_candidates.append((score, index))
            self.log(
                f"Summary candidate chunk {index + 1}/{total_chunks} "
                f"scored {score}."
            )

        scored_candidates.sort(key=lambda item: (-item[0], item[1]))
        for score, index in scored_candidates:
            if len(selected_indices) >= self.max_summary_chunks - 1:
                break
            if score <= 0:
                continue
            selected_indices.append(index)
            self.log(
                f"Selected chunk {index + 1}/{total_chunks} from score-based "
                f"ranking (score={score})."
            )

        if total_chunks > 1 and len(selected_indices) < self.max_summary_chunks:
            tail_index = total_chunks - 1
            if tail_index not in selected_indices:
                selected_indices.append(tail_index)
                self.log(
                    f"Selected trailing chunk {tail_index + 1}/{total_chunks} "
                    "to preserve the document conclusion."
                )

        if len(selected_indices) < self.max_summary_chunks:
            stride = max(1, total_chunks // self.max_summary_chunks)
            for index in range(stride, total_chunks - 1, stride):
                if len(selected_indices) >= self.max_summary_chunks:
                    break
                if index in selected_indices:
                    continue
                selected_indices.append(index)
                self.log(
                    f"Selected chunk {index + 1}/{total_chunks} as a coverage "
                    "sample from the middle of the document."
                )

        selected_indices = sorted(set(selected_indices))[:self.max_summary_chunks]
        self.log(
            f"Representative summary selection complete with chunk indices "
            f"{[index + 1 for index in selected_indices]}."
        )
        return [
            selection_chunks[index].strip()
            for index in selected_indices
            if selection_chunks[index].strip()
        ]

    def _summarize_chunk(self, chunk: str, *, focus_query: str = "") -> str:
        """Summarize one chunk of PDF text using the local model."""
        messages = build_summary_chunk_messages(
            self.system_prompt,
            chunk,
            focus_query=focus_query,
        )
        return self.model.generate_text(messages, stream=False)

    def _refine_summaries(
        self,
        summaries: list[str],
        *,
        focus_query: str = "",
    ) -> str:
        """Recursively combine partial summaries within the prompt budget."""
        cleaned = [summary.strip() for summary in summaries if summary.strip()]
        if not cleaned:
            return WARNING_NO_TEXT
        if len(cleaned) == 1:
            return cleaned[0]

        budget = self.input_char_budget
        total_length = sum(len(summary) for summary in cleaned) + max(0, len(cleaned) - 1) * 2
        if total_length <= budget:
            self.log(
                f"Combining all {len(cleaned)} summary chunk(s) in a single pass "
                f"within the {budget} character budget."
            )
            return self._combine_summary_group(cleaned, focus_query=focus_query)

        self.log(
            f"Refining {len(cleaned)} summary chunk(s) with an input budget "
            f"of {budget} characters."
        )
        grouped_summaries: list[str] = []
        current_group: list[str] = []
        current_length = 0

        for index, summary in enumerate(cleaned, start=1):
            projected_length = current_length + len(summary) + 2
            if current_group and projected_length > budget:
                self.log(
                    f"Closing summary group {len(grouped_summaries) + 1} with "
                    f"{len(current_group)} item(s) at {current_length} characters."
                )
                grouped_summaries.append(
                    self._combine_summary_group(current_group, focus_query=focus_query)
                )
                current_group = [summary]
                current_length = len(summary)
                self.log(
                    f"Starting summary group {len(grouped_summaries) + 1} "
                    f"with summary {index}/{len(cleaned)}."
                )
            else:
                current_group.append(summary)
                current_length = projected_length
                self.log(
                    f"Added summary {index}/{len(cleaned)} to current group "
                    f"({len(current_group)} item(s), {current_length} characters)."
                )

        if current_group:
            self.log(
                f"Closing summary group {len(grouped_summaries) + 1} with "
                f"{len(current_group)} item(s) at {current_length} characters."
            )
            grouped_summaries.append(
                self._combine_summary_group(current_group, focus_query=focus_query)
            )

        self.log(
            f"Refinement pass produced {len(grouped_summaries)} combined "
            "summary group(s)."
        )
        return self._refine_summaries(grouped_summaries, focus_query=focus_query)

    def _combine_summary_group(
        self,
        summaries: list[str],
        *,
        focus_query: str = "",
    ) -> str:
        """Combine a bounded set of partial summaries into one summary."""
        self.log(
            f"Combining {len(summaries)} summary fragment(s) into a single "
            "summary group."
        )
        messages = build_combine_summaries_messages(
            self.system_prompt,
            summaries,
            focus_query=focus_query,
        )
        combined = self.model.generate_text(messages, stream=False)
        self.log(f"Combined summary group complete ({len(combined)} characters).")
        return combined
