"""LLM-backed research assistant agent."""

from __future__ import annotations

import json

from rich.table import Table

from omnidex.agents.base import BaseAgent
from omnidex.agents.research_assistant.commands import SummarizePdfCommand
from omnidex.engine import GeneratePlanCommand
from omnidex.agents.research_assistant.prompts import DEFAULT_SYSTEM_PROMPT
from omnidex.runtime import LocalChatModel, LocalLLMSettings
from omnidex.tools.answer_question import AnswerQuestionTool
from omnidex.tools.create_output import CreateOutputTool
from omnidex.tools.extract_report_insights import ExtractReportInsightsTool
from omnidex.tools.output_request import OutputRequestTool
from omnidex.tools.output_write_intent import OutputWriteIntentTool
from omnidex.tools.pdf_present import PDFPresentTool
from omnidex.tools.pdf_reader import PDFReaderTool
from omnidex.tools.report_insights import ReportInsightsTool
from omnidex.tools.select_relevant_text import SelectRelevantTextTool
from omnidex.tools.summarize_text import SummarizeTextTool
from omnidex.utils.plan_execution import execute_tool_plan
from omnidex.utils.planning import ToolPlanStep
from omnidex.utils.text import compute_input_char_budget, truncate_text


class ResearchAssistant(BaseAgent):
    """Specialized agent for research-oriented questions."""

    name = "research_assistant"
    description = "Handles research queries and provides detailed answers"
    summary_chunk_size = 1600
    max_summary_chunks = 4

    def __init__(self, *, tools: list[object] | None = None, **kwargs) -> None:
        """Initialize the agent and its local GGUF-backed chat model."""
        verbose = bool(kwargs.pop("verbose", True))
        super().__init__(tools=[], verbose=verbose, **kwargs)

        self.settings = LocalLLMSettings.from_env(
            system_prompt=DEFAULT_SYSTEM_PROMPT
        )
        self.model = LocalChatModel(self.settings)

        input_char_budget = compute_input_char_budget(
            self.settings.ctx_size,
            self.settings.max_tokens,
        )
        summarize_command = SummarizePdfCommand(
            model=self.model,
            system_prompt=self.settings.system_prompt.strip(),
            log=self.log,
            summary_chunk_size=self.summary_chunk_size,
            max_summary_chunks=self.max_summary_chunks,
            input_char_budget=input_char_budget,
        )

        self.tools = list(
            tools
            or [
                OutputWriteIntentTool(),
                OutputRequestTool(),
                PDFPresentTool(),
                PDFReaderTool(),
                SelectRelevantTextTool(
                    log=self.log,
                    input_char_budget=input_char_budget,
                ),
                SummarizeTextTool(summarize_command=summarize_command),
                ExtractReportInsightsTool(summarize_command=summarize_command),
                ReportInsightsTool(),
                AnswerQuestionTool(
                    model=self.model,
                    system_prompt=self.settings.system_prompt.strip(),
                ),
                CreateOutputTool(),
            ]
        )
        self.generate_plan_command = GeneratePlanCommand(
            model=self.model,
            tools=self.tools,
            emit=self.emit,
            log=self.log,
        )
        self.session_state: dict[str, object] = {}

    def _render_plan(self, plan: list[ToolPlanStep]) -> None:
        """Render a generated plan as a readable table."""
        if not plan:
            self.emit("Generated Plan: none", style="cyan")
            return

        table = Table(
            title=f"{self.name} plan",
            show_header=True,
            header_style="bold cyan",
        )
        table.add_column("Step", style="cyan", justify="right", no_wrap=True)
        table.add_column("Tool", style="green", no_wrap=True)
        table.add_column("Output", style="blue", no_wrap=True)
        table.add_column("Inputs", style="yellow")
        table.add_column("Reason", style="white")

        for step in plan:
            rendered_inputs = json.dumps(step.inputs, ensure_ascii=True, sort_keys=True)
            table.add_row(
                str(step.step),
                step.tool_name,
                step.output_key,
                rendered_inputs,
                step.reason,
            )

        self.console.print(table)

    def _classify_direct_pdf_intent(self, query: str) -> str | None:
        """Detect simple PDF requests that can skip planner overhead."""
        normalized_query = query.casefold()
        if ".pdf" not in normalized_query:
            return None
        if any(
            keyword in normalized_query
            for keyword in (
                "insight",
                "insights",
                "strength",
                "strengths",
                "novel",
                "gap",
                "gaps",
                "limitation",
                "limitations",
            )
        ):
            return "insights"
        if any(
            keyword in normalized_query
            for keyword in ("summarize", "summarise", "summary", "overview", "tl;dr")
        ):
            return "summary"
        return None

    def _extract_artifact_content(self, payload: object) -> str:
        """Extract clean artifact content from a tool result when available."""
        if isinstance(payload, dict):
            artifact_content = str(payload.get("artifact_content") or "").strip()
            if artifact_content:
                return artifact_content
            content = str(payload.get("content") or "").strip()
            if content:
                return content
        return ""

    def _update_session_artifact(
        self,
        *,
        response: str,
        artifact_content: str = "",
    ) -> None:
        """Persist the latest response and artifact for follow-up actions."""
        preserved_artifact = artifact_content.strip() or str(
            self.session_state.get("last_artifact_content", "") or ""
        ).strip()
        self.session_state = {
            **self.session_state,
            "last_response": response,
            "last_responder": self.name,
            "last_tools_used": list(self.last_used_tools),
            "last_artifact_content": preserved_artifact,
        }

    def _run_direct_pdf_flow(self, query: str) -> str | None:
        """Handle obvious PDF summary/insight requests without planner latency."""
        intent = self._classify_direct_pdf_intent(query)
        if intent is None:
            return None

        pdf_present_tool = self.get_tool("pdf_present")
        pdf_reader_tool = self.get_tool("pdf_reader")
        output_request_tool = self.get_tool("extract_output_request")
        create_output_tool = self.get_tool("create_output")
        if (
            pdf_present_tool is None
            or pdf_reader_tool is None
            or output_request_tool is None
            or create_output_tool is None
        ):
            return None

        pdf_ref = pdf_present_tool.run(query=query)
        if not pdf_ref.get("found"):
            return None

        self.emit(f"Direct PDF {intent} path selected.", style="yellow")
        self.record_tool_use("pdf_present", reason="detected explicit PDF fast path")
        self.record_tool_use("pdf_reader", reason="loaded PDF text without planner step")
        pdf_doc = pdf_reader_tool.run(file_path=str(pdf_ref.get("file_path") or ""))
        if str(pdf_doc.get("status", "ok")) == "error":
            return str(pdf_doc.get("content", "")).strip() or "Error: Unable to read PDF."

        text = str(pdf_doc.get("text") or "")
        output_request = output_request_tool.run(query=query)

        if intent == "insights":
            extract_insights_tool = self.get_tool("extract_report_insights")
            report_insights_tool = self.get_tool("report_insights")
            if extract_insights_tool is None or report_insights_tool is None:
                return None
            self.record_tool_use(
                "extract_report_insights",
                reason="generated structured PDF insights directly",
            )
            insights = extract_insights_tool.run(text=text, focus_query=query)
            if str(insights.get("status", "ok")) == "error":
                return str(insights.get("content", "")).strip()
            self.record_tool_use(
                "report_insights",
                reason="formatted structured insights for display",
            )
            rendered = report_insights_tool.run(
                title=str(insights.get("title") or ""),
                keywords=insights.get("keywords"),
                strengths=insights.get("strengths"),
                novel_approach=str(insights.get("novel_approach") or ""),
                gaps_and_limitations=insights.get("gaps_and_limitations"),
            )
            content = str(rendered.get("content") or "")
        else:
            summarize_tool = self.get_tool("summarize_text")
            if summarize_tool is None:
                return None
            self.record_tool_use(
                "summarize_text",
                reason="summarized PDF directly without planner step",
            )
            summary = summarize_tool.run(text=text, focus_query=query)
            if str(summary.get("status", "ok")) == "error":
                return str(summary.get("content", "")).strip()
            content = str(summary.get("content") or "")

        self.record_tool_use("create_output", reason="finalized direct PDF result")
        final = create_output_tool.run(
            content=content,
            filename=output_request.get("filename"),
            write_output=output_request.get("write_output"),
        )
        response = str(final.get("content") or "").strip()
        self._update_session_artifact(
            response=response,
            artifact_content=self._extract_artifact_content(final),
        )
        return response

    def _run_direct_save_followup(self, query: str) -> str | None:
        """Persist the previous artifact when the user makes a direct save request."""
        normalized_query = query.casefold()
        if ".pdf" in normalized_query:
            return None

        last_artifact_content = str(
            self.session_state.get("last_artifact_content", "") or ""
        ).strip()
        last_response = str(self.session_state.get("last_response", "") or "").strip()
        content_to_save = last_artifact_content or last_response
        if not content_to_save:
            return None

        output_request_tool = self.get_tool("extract_output_request")
        create_output_tool = self.get_tool("create_output")
        if output_request_tool is None or create_output_tool is None:
            return None

        output_request = output_request_tool.run(query=query)
        write_output = bool(output_request.get("write_output"))
        filename = output_request.get("filename")
        if not write_output or not filename:
            return None

        self.emit("Direct save follow-up path selected.", style="yellow")
        self.record_tool_use(
            "extract_output_request",
            reason="detected explicit save request for the previous artifact",
        )
        self.record_tool_use(
            "create_output",
            reason="wrote the previous artifact without planner overhead",
        )
        final = create_output_tool.run(
            content=content_to_save,
            filename=filename,
            write_output=write_output,
        )
        response = str(final.get("content") or "").strip()
        self._update_session_artifact(
            response=response,
            artifact_content=self._extract_artifact_content(final) or content_to_save,
        )
        return response

    def run(self, query: str, context: str = "") -> str:
        """Answer a research query using a generic planned tool loop."""
        self.log(f"Running research query: {query}")
        direct_save_response = self._run_direct_save_followup(query)
        if direct_save_response is not None:
            self.log("Research query completed through direct save follow-up flow.")
            return direct_save_response

        direct_pdf_response = self._run_direct_pdf_flow(query)
        if direct_pdf_response is not None:
            self.log("Research query completed through direct PDF flow.")
            return direct_pdf_response

        initial_state = {
            "query": query,
            "last_response": self.session_state.get("last_response", ""),
            "last_artifact_content": self.session_state.get("last_artifact_content", ""),
            "last_responder": self.session_state.get("last_responder", ""),
            "last_tools_used": self.session_state.get("last_tools_used", []),
        }
        planning_state = {
            "last_response": truncate_text(str(initial_state["last_response"]), limit=1500),
            "last_artifact_content": truncate_text(
                str(initial_state["last_artifact_content"]),
                limit=1500,
            ),
            "last_responder": initial_state["last_responder"],
            "last_tools_used": initial_state["last_tools_used"],
        }
        tool_plan = self.generate_plan_command.execute(
            query,
            initial_state=planning_state,
        )
        if not tool_plan:
            self.emit(
                "Planner returned no usable steps; not applying any deterministic fallback.",
                style="yellow",
            )
            return (
                "Unable to determine a tool plan for this request. "
                "Please rephrase the prompt or provide more explicit instructions."
            )

        self._render_plan(tool_plan)
        self.emit(
            f"Tool plan: {' -> '.join(step.tool_name for step in tool_plan)}",
            style="yellow",
        )

        execution = execute_tool_plan(
            tool_plan,
            get_tool=self.get_tool,
            record_tool_use=lambda tool_name, reason: self.record_tool_use(tool_name, reason=reason),
            emit=lambda message, style: self.emit(message, style=style),
            log=self.log,
            query=query,
            context=context,
            initial_state=initial_state,
        )
        self.log("Research query completed.")
        final_output = execution.final_output or "No output generated."
        artifact_content = ""
        final_state = execution.state.get("final")
        artifact_content = self._extract_artifact_content(final_state)
        if not artifact_content:
            artifact_content = final_output.strip()
        self._update_session_artifact(
            response=final_output,
            artifact_content=artifact_content,
        )
        return final_output
