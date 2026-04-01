"""LLM-backed research assistant agent."""

from __future__ import annotations

import re

from omnidex.agents.base import BaseAgent
from omnidex.runtime import LocalChatModel, LocalLLMSettings
from omnidex.tools.create_output import CreateOutputTool
from omnidex.tools.pdf_reader import PDFReaderTool
from omnidex.tools.report_insights import ReportInsightsTool


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
        resolved_tools = list(
            tools or [PDFReaderTool(), ReportInsightsTool(), CreateOutputTool()]
        )
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

    def _route_tool_plan(self, query: str) -> list[dict[str, str]]:
        """Choose a multi-step tool plan for the query using simple heuristics."""
        normalized_query = query.strip()
        lowered_query = normalized_query.lower()
        pdf_path = self._find_pdf_path(normalized_query)
        pdf_reader = self.get_tool("pdf_reader")
        report_insights = self.get_tool("report_insights")
        create_output = self.get_tool("create_output")

        wants_summary = (
            lowered_query.startswith("summarize pdf:")
            or ("summarize" in lowered_query and bool(pdf_path))
            or normalized_query.strip("\"'").lower() == (pdf_path or "").lower()
        )
        wants_insights = any(
            keyword in lowered_query
            for keyword in {"insight", "keywords", "strength", "novel", "gap", "limitation"}
        )

        if bool(pdf_reader) and bool(pdf_path) and wants_insights:
            plan = [
                {
                    "tool_name": "pdf_reader",
                    "mode": "read",
                    "file_path": pdf_path or self._extract_pdf_path(normalized_query),
                    "reason": "load PDF content for downstream processing",
                }
            ]
            if bool(report_insights):
                plan.append(
                    {
                        "tool_name": "report_insights",
                        "mode": "insights",
                        "reason": "generate structured report insights",
                    }
                )
            if bool(create_output):
                plan.append(
                    {
                        "tool_name": "create_output",
                        "mode": "finalize",
                        "title": "Report Insights",
                        "reason": "final response formatting",
                    }
                )
            return plan

        if bool(pdf_reader) and bool(pdf_path) and wants_summary:
            plan = [
                {
                    "tool_name": "pdf_reader",
                    "mode": "read",
                    "file_path": pdf_path or self._extract_pdf_path(normalized_query),
                    "reason": "load PDF content for summarization",
                }
            ]
            if bool(create_output):
                plan.append(
                    {
                        "tool_name": "create_output",
                        "mode": "finalize",
                        "title": "PDF Summary",
                        "reason": "final response formatting",
                    }
                )
            return plan

        if bool(pdf_reader) and bool(pdf_path):
            plan = [
                {
                    "tool_name": "pdf_reader",
                    "mode": "question_answer",
                    "file_path": pdf_path,
                    "reason": "query references a PDF document",
                }
            ]
            if bool(create_output):
                plan.append(
                    {
                        "tool_name": "create_output",
                        "mode": "finalize",
                        "reason": "final response formatting",
                    }
                )
            return plan

        return []

    def _extract_keywords(self, text: str, *, limit: int = 5) -> list[str]:
        """Extract simple frequent keywords from document text."""
        stop_words = {
            "the", "and", "for", "with", "that", "this", "from", "about", "into",
            "their", "there", "have", "using", "used", "such", "than", "then",
            "they", "them", "were", "been", "also", "can", "could", "would",
        }
        counts: dict[str, int] = {}
        for token in re.findall(r"[a-zA-Z][a-zA-Z0-9_-]{3,}", text.lower()):
            if token in stop_words:
                continue
            counts[token] = counts.get(token, 0) + 1
        ranked = sorted(counts.items(), key=lambda item: (-item[1], item[0]))
        return [keyword for keyword, _ in ranked[:limit]]

    def _extract_title(self, text: str) -> str:
        """Extract a best-effort document title from the first non-empty line."""
        for line in text.splitlines():
            normalized = line.strip()
            if normalized:
                return normalized[:120]
        return "Untitled Report"

    def _generate_report_insights(self, text: str) -> str:
        """Generate structured report insights and format them via the tool."""
        tool = self.get_tool("report_insights")
        if tool is None:
            return text.strip()

        summary = self._summarize_pdf_text(text)
        strengths = [
            sentence.strip(" -.")
            for sentence in re.split(r"[.\n]+", summary)
            if sentence.strip()
        ][:3]
        gaps = [
            "Potential limitations were not explicitly identified in the available summary.",
        ]
        novel_approach = (
            strengths[0]
            if strengths
            else "No clearly novel approach was identified from the extracted text."
        )
        return tool.run(
            title=self._extract_title(text),
            keywords=self._extract_keywords(text),
            strengths=strengths,
            novel_approach=novel_approach,
            gaps_and_limitations=gaps,
        )

    def _finalize_output(self, content: str, *, title: str | None = None) -> str:
        """Send the response through the output tool when available."""
        tool = self.get_tool("create_output")
        if tool is None:
            return content.strip()
        self.record_tool_use(tool.name, reason="final response formatting")
        return tool.run(content, title=title)

    def _build_system_prompt(self, context: str = "", pdf_text: str = "") -> str:
        """Build the system prompt with optional memory and PDF context."""
        system_prompt = self.settings.system_prompt.strip()
        bounded_context = self._truncate_text(context, limit=1800)
        if bounded_context:
            system_prompt = (
                f"{system_prompt}\n\n"
                "Use the context below if relevant. Do not repeat it verbatim "
                "unless the user asks.\n\n"
                f"{bounded_context}"
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

    def _truncate_text(self, text: str, *, limit: int) -> str:
        """Trim arbitrary context to a bounded size for local prompts."""
        normalized = text.strip()
        if len(normalized) <= limit:
            return normalized
        return f"{normalized[:limit].rstrip()}\n\n[Context truncated]"

    def run(self, query: str, context: str = "") -> str:
        """Answer a research query using the local chat model."""
        self.log(f"Running research query: {query}")

        pdf_text = ""
        tool_plan = self._route_tool_plan(query)
        if tool_plan:
            plan_names = " -> ".join(step["tool_name"] for step in tool_plan)
            self.emit(f"Tool plan: {plan_names}", style="yellow")

            final_output = ""
            for step in tool_plan:
                tool = self.get_tool(step["tool_name"])
                if tool is None:
                    self.emit(f"Tool unavailable: {step['tool_name']}", style="red")
                    continue

                self.record_tool_use(tool.name, reason=step["reason"])
                self.log(
                    f"Tool step selected: {step['tool_name']} "
                    f"mode={step['mode']}"
                )

                if step["tool_name"] == "pdf_reader":
                    pdf_text = tool.run(step["file_path"])
                    if pdf_text.startswith("Error:") or pdf_text.startswith("Warning:"):
                        return pdf_text
                elif step["tool_name"] == "report_insights":
                    final_output = self._generate_report_insights(pdf_text)
                elif step["tool_name"] == "create_output":
                    content = final_output or self._summarize_pdf_text(pdf_text)
                    final_output = tool.run(content, title=step.get("title"))

            if final_output:
                return final_output
            if pdf_text:
                return self._finalize_output(self._summarize_pdf_text(pdf_text), title="PDF Summary")

        pdf_path = self._find_pdf_path(query)
        if pdf_path:
            pdf_tool = self.get_tool("pdf_reader")
            if pdf_tool is not None:
                self.record_tool_use(pdf_tool.name, reason="query references a PDF document")
                pdf_text = pdf_tool.run(pdf_path)
                if pdf_text.startswith("Error:") or pdf_text.startswith("Warning:"):
                    return pdf_text
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
        return self._finalize_output(cleaned)
