"""LLM-backed research assistant agent."""

from __future__ import annotations

import re

from omnidex.agents.base import BaseAgent
from omnidex.runtime import LocalChatModel, LocalLLMSettings
from omnidex.tools.pdf_reader import PDFReaderTool


DEFAULT_SYSTEM_PROMPT = (
    "You are a research assistant. Provide clear, structured, and "
    "well-reasoned answers. Explain reasoning when it is helpful. "
    "Do not hallucinate unknown facts, and if you are uncertain, "
    "say so explicitly."
)


class ResearchAssistant(BaseAgent):
    """Specialized agent for research-oriented questions."""

    name = "research_assistant"
    description = "Handles research queries and provides detailed answers"

    def __init__(self, *, tools: list[object] | None = None, **kwargs) -> None:
        """Initialize the agent and its local GGUF-backed chat model."""
        resolved_tools = list(tools or [PDFReaderTool()])
        super().__init__(tools=resolved_tools, **kwargs)
        self.settings = LocalLLMSettings.from_env(
            system_prompt=DEFAULT_SYSTEM_PROMPT
        )
        self.model = LocalChatModel(self.settings)

    def _chunk_text(self, text: str, chunk_size: int = 2000) -> list[str]:
        """Split long text into manageable chunks for summarization."""
        normalized = text.strip()
        if not normalized:
            return []
        return [
            normalized[index:index + chunk_size]
            for index in range(0, len(normalized), chunk_size)
        ]

    def _input_char_budget(self) -> int:
        """Estimate a safe character budget for prompt inputs."""
        available_tokens = max(1024, self.settings.ctx_size - self.settings.max_tokens - 1024)
        return max(2000, min(6000, available_tokens * 2))

    def _summarize_chunk(self, chunk: str) -> str:
        """Summarize one chunk of PDF text using the local model."""
        messages = [
            {
                "role": "system",
                "content": (
                    f"{self.settings.system_prompt.strip()}\n\n"
                    "Summarize the provided document excerpt. Focus on the main "
                    "arguments, findings, and notable details. Keep the summary "
                    "concise and factual."
                ),
            },
            {"role": "user", "content": chunk},
        ]
        completion = self.model.complete(messages, stream=False)
        response = completion["choices"][0]["message"]["content"]
        return response.strip()

    def _summarize_pdf_text(self, text: str) -> str:
        """Summarize long PDF text by chunking, then refining the result."""
        chunks = self._chunk_text(text)
        if not chunks:
            return "Warning: PDF contains no extractable text."

        self.log(f"Summarizing PDF across {len(chunks)} chunk(s).")
        partial_summaries = [self._summarize_chunk(chunk) for chunk in chunks]
        return self._refine_summaries(partial_summaries)

    def _refine_summaries(self, summaries: list[str]) -> str:
        """Recursively combine partial summaries within the model's context budget."""
        cleaned = [summary.strip() for summary in summaries if summary.strip()]
        if not cleaned:
            return "Warning: PDF contains no extractable text."
        if len(cleaned) == 1:
            return cleaned[0]

        budget = self._input_char_budget()
        grouped_summaries: list[str] = []
        current_group: list[str] = []
        current_length = 0

        for summary in cleaned:
            projected_length = current_length + len(summary) + 2
            if current_group and projected_length > budget:
                grouped_summaries.append(self._combine_summary_group(current_group))
                current_group = [summary]
                current_length = len(summary)
            else:
                current_group.append(summary)
                current_length = projected_length

        if current_group:
            grouped_summaries.append(self._combine_summary_group(current_group))

        return self._refine_summaries(grouped_summaries)

    def _combine_summary_group(self, summaries: list[str]) -> str:
        """Combine a bounded set of partial summaries into one summary."""
        messages = [
            {
                "role": "system",
                "content": (
                    f"{self.settings.system_prompt.strip()}\n\n"
                    "Combine the partial summaries into a single clear research "
                    "summary. Remove repetition and preserve the key findings."
                ),
            },
            {"role": "user", "content": "\n\n".join(summaries)},
        ]
        completion = self.model.complete(messages, stream=False)
        response = completion["choices"][0]["message"]["content"]
        return response.strip()

    def _extract_pdf_path(self, query: str) -> str:
        """Resolve a PDF path from supported query forms."""
        normalized = query.strip()
        lowered = normalized.lower()
        if lowered.startswith("summarize pdf:"):
            normalized = normalized[len("summarize pdf:"):].strip()
        return normalized.strip("\"'")

    def _find_pdf_path(self, query: str) -> str | None:
        """Extract the first PDF-like path reference from a free-form query."""
        match = re.search(
            r'(?P<path>"[^"]+\.pdf"|\'[^\']+\.pdf\'|[^\s"\']+\.pdf)',
            query,
            re.IGNORECASE,
        )
        if not match:
            return None
        return match.group("path").strip("\"'")

    def _route_tool_request(self, query: str) -> dict[str, str] | None:
        """Choose a tool and action for the query using simple heuristics."""
        normalized_query = query.strip()
        lowered_query = normalized_query.lower()
        pdf_path = self._find_pdf_path(normalized_query)
        pdf_reader = self.get_tool("pdf_reader")

        strategies = [
            {
                "tool_name": "pdf_reader",
                "mode": "summarize",
                "file_path": self._extract_pdf_path(normalized_query),
                "reason": "explicit summarize pdf command",
                "matched": bool(pdf_reader) and lowered_query.startswith("summarize pdf:"),
            },
            {
                "tool_name": "pdf_reader",
                "mode": "summarize",
                "file_path": pdf_path or "",
                "reason": "bare PDF path detected",
                "matched": bool(pdf_reader)
                and bool(pdf_path)
                and normalized_query.strip("\"'").lower() == (pdf_path or "").lower(),
            },
            {
                "tool_name": "pdf_reader",
                "mode": "question_answer",
                "file_path": pdf_path or "",
                "reason": "query references a PDF document",
                "matched": bool(pdf_reader) and bool(pdf_path),
            },
        ]

        for strategy in strategies:
            if strategy["matched"]:
                return {
                    "tool_name": str(strategy["tool_name"]),
                    "mode": str(strategy["mode"]),
                    "file_path": str(strategy["file_path"]),
                    "reason": str(strategy["reason"]),
                }
        return None

    def _build_system_prompt(self, context: str = "", pdf_text: str = "") -> str:
        """Build the system prompt with optional memory and PDF context."""
        system_prompt = self.settings.system_prompt.strip()
        if context.strip():
            system_prompt = (
                f"{system_prompt}\n\n"
                "Use the context below if relevant. Do not repeat it verbatim "
                "unless the user asks.\n\n"
                f"{context.strip()}"
            )
        if pdf_text.strip():
            system_prompt = (
                f"{system_prompt}\n\n"
                "Use the PDF excerpts below when answering questions about the "
                "document. Base your answer on this content and say when the "
                "document does not support a claim.\n\n"
                f"{pdf_text.strip()}"
            )
        return system_prompt

    def _tokenize_for_matching(self, text: str) -> set[str]:
        """Extract simple lowercase keyword tokens for heuristic matching."""
        return {
            token
            for token in re.findall(r"[a-zA-Z0-9_]{3,}", text.lower())
            if token not in {"the", "and", "for", "with", "that", "this", "from", "about"}
        }

    def _select_relevant_pdf_context(self, text: str, query: str) -> str:
        """Select a bounded set of relevant PDF excerpts for question answering."""
        budget = self._input_char_budget()
        chunks = self._chunk_text(text, chunk_size=1200)
        if not chunks:
            return ""

        query_terms = self._tokenize_for_matching(query)
        scored_chunks: list[tuple[int, int, str]] = []
        for index, chunk in enumerate(chunks):
            chunk_terms = self._tokenize_for_matching(chunk)
            score = len(query_terms & chunk_terms)
            scored_chunks.append((score, -index, chunk))

        scored_chunks.sort(reverse=True)
        selected: list[str] = []
        total_length = 0
        for score, _, chunk in scored_chunks:
            if score == 0 and selected:
                continue
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
            return chunks[0][:budget].strip()
        return "\n\n---\n\n".join(selected)

    def run(self, query: str, context: str = "") -> str:
        """Answer a research query using the local chat model."""
        self.log(f"Running research query: {query}")

        pdf_text = ""
        tool_request = self._route_tool_request(query)
        if tool_request is not None:
            tool = self.get_tool(tool_request["tool_name"])
            if tool is not None:
                self.emit(
                    f"Using tool: {tool.name} ({tool_request['reason']})",
                    style="yellow",
                )
                self.log(
                    f"Tool route selected: {tool_request['tool_name']} "
                    f"mode={tool_request['mode']} path={tool_request['file_path']}"
                )
                pdf_text = tool.run(tool_request["file_path"])
            else:
                self.emit(
                    f"Tool unavailable: {tool_request['tool_name']}",
                    style="red",
                )
                pdf_text = ""

            if pdf_text.startswith("Error:") or pdf_text.startswith("Warning:"):
                return pdf_text
            if tool_request["mode"] == "summarize":
                return self._summarize_pdf_text(pdf_text)
            pdf_text = self._select_relevant_pdf_context(pdf_text, query)
            self.log(
                f"Selected {len(pdf_text)} characters of PDF context "
                f"within a {self._input_char_budget()} character budget."
            )

        system_prompt = self._build_system_prompt(context=context, pdf_text=pdf_text)

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": query},
        ]
        completion = self.model.complete(messages, stream=False)
        response = completion["choices"][0]["message"]["content"]
        cleaned = response.strip()
        self.log("Research query completed.")
        return cleaned
