"""LLM-backed research assistant agent."""

from __future__ import annotations

import inspect
import json
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
        return self.model.generate_text(messages, stream=False)

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
        return self.model.generate_text(messages, stream=False)

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

    def _describe_tools(self) -> list[dict[str, object]]:
        """Return structured tool metadata for LLM planning."""
        described_tools: list[dict[str, object]] = []
        for tool in self.tools:
            tool_name = getattr(tool, "name", tool.__class__.__name__)
            description = (
                getattr(tool, "__doc__", None)
                or getattr(tool.run, "__doc__", None)
                or "Tool available for agent workflows."
            )
            normalized_description = " ".join(description.strip().split())

            expected_inputs: list[str] = []
            try:
                signature = inspect.signature(tool.run)
            except (TypeError, ValueError):
                signature = None

            if signature is not None:
                for parameter in signature.parameters.values():
                    if parameter.name == "self":
                        continue
                    if parameter.kind in {
                        inspect.Parameter.VAR_POSITIONAL,
                        inspect.Parameter.VAR_KEYWORD,
                    }:
                        continue
                    expected_inputs.append(parameter.name)

            described_tools.append(
                {
                    "name": tool_name,
                    "description": normalized_description,
                    "inputs": expected_inputs or ["generic_input"],
                }
            )

        return described_tools

    def _generate_plan(self, query: str) -> list[dict]:
        """Use the local model to generate a structured tool plan."""
        tool_descriptions = self._describe_tools()
        messages = [
            {
                "role": "system",
                "content": (
                    "You are an expert AI planning agent.\n\n"
                    "Your job is to:\n"
                    "- Analyze the user query\n"
                    "- Break it into steps\n"
                    "- Decide which tools to use\n"
                    "- Produce a structured execution plan\n\n"
                    "You must:\n"
                    "- Only use available tools\n"
                    "- Be concise and logical\n"
                    "- Avoid hallucinating tools\n"
                    "- Return ONLY valid JSON\n"
                    "- Use an empty plan when no tool is needed"
                ),
            },
            {
                "role": "user",
                "content": (
                    f"User Query:\n{query.strip()}\n\n"
                    f"Available Tools:\n{json.dumps(tool_descriptions, indent=2)}\n\n"
                    "---\n\n"
                    "Return ONLY valid JSON in this format:\n\n"
                    "{\n"
                    '  "plan": [\n'
                    "    {\n"
                    '      "step": 1,\n'
                    '      "tool_name": "tool_name_here",\n'
                    '      "mode": "optional_mode",\n'
                    '      "inputs": { "param": "value" },\n'
                    '      "reason": "why this step is needed"\n'
                    "    }\n"
                    "  ]\n"
                    "}"
                ),
            },
        ]

        try:
            raw_content = self.model.generate_text(messages, stream=False)
        except Exception as exc:
            self.emit(f"Plan generation failed: {exc}", style="yellow")
            return []

        cleaned_content = re.sub(
            r"^```(?:json)?\s*|\s*```$",
            "",
            raw_content,
            flags=re.IGNORECASE | re.DOTALL,
        ).strip()

        try:
            payload = json.loads(cleaned_content)
        except json.JSONDecodeError as exc:
            json_start = cleaned_content.find("{")
            json_end = cleaned_content.rfind("}")
            if json_start == -1 or json_end == -1 or json_end <= json_start:
                self.emit(f"Plan parsing failed: {exc}", style="yellow")
                self.log(f"Raw plan output: {raw_content}")
                return []
            try:
                payload = json.loads(cleaned_content[json_start:json_end + 1])
            except json.JSONDecodeError as nested_exc:
                self.emit(f"Plan parsing failed: {nested_exc}", style="yellow")
                self.log(f"Raw plan output: {raw_content}")
                return []

        raw_plan = payload.get("plan", [])
        if not isinstance(raw_plan, list):
            self.emit("Plan parsing failed: plan must be a list.", style="yellow")
            return []

        normalized_plan: list[dict] = []
        for index, step in enumerate(raw_plan, start=1):
            if not isinstance(step, dict):
                self.emit(f"Skipping invalid plan step at position {index}.", style="yellow")
                continue

            tool_name = str(step.get("tool_name", "")).strip()
            mode = step.get("mode")
            inputs = step.get("inputs", {})
            reason = str(step.get("reason", "")).strip()

            if not tool_name:
                self.emit(f"Skipping plan step {index}: missing tool_name.", style="yellow")
                continue
            if not isinstance(inputs, dict):
                self.emit(f"Skipping plan step {index}: inputs must be an object.", style="yellow")
                continue

            normalized_plan.append(
                {
                    "step": step.get("step", index),
                    "tool_name": tool_name,
                    "mode": str(mode).strip() if mode is not None else "",
                    "inputs": inputs,
                    "reason": reason or "No reason provided.",
                }
            )

        self.emit(f"Generated Plan: {normalized_plan}", style="cyan")
        return normalized_plan

    def _missing_required_inputs(
        self,
        tool: object,
        inputs: dict[str, object],
        *,
        derived_inputs: set[str] | None = None,
    ) -> list[str]:
        """Return required tool inputs that are missing from the plan step."""
        try:
            signature = inspect.signature(tool.run)
        except (TypeError, ValueError):
            return []

        available_inputs = set(inputs)
        if derived_inputs:
            available_inputs.update(derived_inputs)

        missing: list[str] = []
        for parameter in signature.parameters.values():
            if parameter.name == "self":
                continue
            if parameter.kind in {
                inspect.Parameter.VAR_POSITIONAL,
                inspect.Parameter.VAR_KEYWORD,
            }:
                continue
            if parameter.default is not inspect._empty:
                continue
            if parameter.name not in available_inputs:
                missing.append(parameter.name)
        return missing

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
        tool_plan = self._generate_plan(query)
        if tool_plan:
            plan_names = " -> ".join(step["tool_name"] for step in tool_plan)
            self.emit(f"Tool plan: {plan_names}", style="yellow")

            final_output = ""
            for step in tool_plan:
                tool_name = str(step.get("tool_name", "")).strip()
                mode = str(step.get("mode", "")).strip()
                inputs = step.get("inputs", {})

                if not tool_name:
                    self.emit("Skipping invalid step with missing tool_name.", style="yellow")
                    continue
                if not isinstance(inputs, dict):
                    self.emit(f"Skipping invalid step for {tool_name}: inputs must be a dict.", style="yellow")
                    continue

                tool = self.get_tool(tool_name)
                if tool is None:
                    self.emit(f"Tool unavailable: {tool_name}", style="red")
                    continue

                derived_inputs: set[str] = set()
                if tool_name == "report_insights":
                    derived_inputs = {
                        "title",
                        "keywords",
                        "strengths",
                        "novel_approach",
                        "gaps_and_limitations",
                    }
                elif tool_name == "create_output":
                    derived_inputs = {"content"}

                missing_inputs = self._missing_required_inputs(
                    tool,
                    inputs,
                    derived_inputs=derived_inputs,
                )
                if missing_inputs:
                    self.emit(
                        f"Skipping {tool_name} step: missing required inputs "
                        f"{', '.join(missing_inputs)}.",
                        style="yellow",
                    )
                    continue

                self.record_tool_use(tool.name, reason=step["reason"])
                self.log(
                    f"Tool step selected: {tool_name} "
                    f"mode={mode}"
                )

                if tool_name == "pdf_reader":
                    file_path = str(inputs.get("file_path", "")).strip()
                    if not file_path:
                        self.emit("Skipping pdf_reader step: missing file_path input.", style="yellow")
                        continue
                    pdf_text = tool.run(file_path)
                    if pdf_text.startswith("Error:") or pdf_text.startswith("Warning:"):
                        return pdf_text
                elif tool_name == "report_insights":
                    if not pdf_text.strip():
                        self.emit("Skipping report_insights step: no PDF content available.", style="yellow")
                        continue
                    final_output = self._generate_report_insights(pdf_text)
                elif tool_name == "create_output":
                    if not final_output and not pdf_text.strip():
                        self.emit("Skipping create_output step: no content available.", style="yellow")
                        continue
                    title = str(inputs.get("title", "")).strip() or step.get("title")
                    content = final_output or self._summarize_pdf_text(pdf_text)
                    final_output = tool.run(content, title=title)

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
        cleaned = self.model.generate_text(messages, stream=False)
        self.log("Research query completed.")
        return self._finalize_output(cleaned)
