"""Prompt builders for shared tool-plan generation."""

from __future__ import annotations

import json


def build_plan_messages(
    query: str,
    tool_descriptions: list[dict[str, object]],
    *,
    initial_values: dict[str, object] | None = None,
) -> list[dict[str, str]]:
    """Build messages for tool-plan generation."""
    rendered_initial_values = {
        key: value
        for key, value in (initial_values or {}).items()
        if value not in (None, "", [], {})
    }
    return [
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
                "- The available tools are exactly the tools listed below\n"
                "- Use references like $query, $context, or $state.output_key.field\n"
                "- Every step must include output_key\n"
                "- Use output_key 'final' for the final user-facing result whenever possible\n"
                "- Do not rename the original file passed to you.\n"
                "- Never use an input/source file path as the create_output filename.\n"
                "- Never overwrite a file that was read earlier in the plan.\n"
                "- Only call create_output when you have content to present.\n"
                "- create_output must only write to disk when write_output is true.\n"
                "- Set write_output to true only when the user explicitly asked to save, write, or export the output.\n"
                "- Use determine_output_write when you need to decide whether output should be written.\n"
                "- Only include a filename when it is explicitly available from the user input or state and is not the same as any source file path.\n"
                "- A save/export request is not complete until create_output is included with content and filename.\n"
                "- Never return a plan that only checks write intent for a save/export request.\n"
                "- If no tool is needed, return an empty plan.\n"
            ),
        },
        {
            "role": "user",
            "content": (
                f"User Query:\n{query.strip()}\n\n"
                f"Available Tools:\n{json.dumps(tool_descriptions, indent=2)}\n\n"
                "Initial Values:\n"
                "- $query: the raw user query\n"
                "- $context: additional context or memory\n\n"
                f"Additional Initial Values:\n{json.dumps(rendered_initial_values, indent=2)}\n\n"
                "Example pattern:\n"
                '- detect PDF: {"tool_name": "pdf_present", "inputs": {"query": "$query"}, "output_key": "pdf_ref"}\n'
                '- read it: {"tool_name": "pdf_reader", "inputs": {"file_path": "$state.pdf_ref.file_path"}, "output_key": "pdf_doc"}\n'
                '- summarize it: {"tool_name": "summarize_text", "inputs": {"text": "$state.pdf_doc.text", "focus_query": "$query"}, "output_key": "summary"}\n'
                '- finalize it: {"tool_name": "create_output", "inputs": {"content": "$state.summary.content"}, "output_key": "final"}\n\n'
                "Insights pattern:\n"
                '- read PDF text: {"tool_name": "pdf_reader", "inputs": {"file_path": "$state.pdf_ref.file_path"}, "output_key": "pdf_doc"}\n'
                '- extract insights: {"tool_name": "extract_report_insights", "inputs": {"text": "$state.pdf_doc.text", "focus_query": "$query"}, "output_key": "insights"}\n'
                '- format insights: {"tool_name": "report_insights", "inputs": {"title": "$state.insights.title", "keywords": "$state.insights.keywords", "strengths": "$state.insights.strengths", "novel_approach": "$state.insights.novel_approach", "gaps_and_limitations": "$state.insights.gaps_and_limitations"}, "output_key": "final"}\n\n'
                "Do not do this:\n"
                '- {"tool_name": "create_output", "inputs": {"content": "$state.summary.content", "filename": "$state.pdf_ref.file_path"}, "output_key": "final"}\n\n'
                "Follow-up save example:\n"
                '- determine write intent: {"tool_name": "determine_output_write", "inputs": {"query": "$query"}, "output_key": "write_request"}\n'
                '- extract filename: {"tool_name": "extract_output_request", "inputs": {"query": "$query"}, "output_key": "output_request"}\n'
                '- save previous artifact: {"tool_name": "create_output", "inputs": {"content": "$last_artifact_content", "filename": "$state.output_request.filename", "write_output": "$state.write_request.write_output"}, "output_key": "final"}\n\n'
                "Do not stop after determine_output_write for save/export requests.\n\n"
                "---\n\n"
                "Return ONLY valid JSON in this format:\n\n"
                "{\n"
                '  "plan": [\n'
                "    {\n"
                '      "step": 1,\n'
                '      "tool_name": "tool_name_here",\n'
                '      "inputs": { "param": "value" },\n'
                '      "output_key": "state_key_here",\n'
                '      "reason": "why this step is needed"\n'
                "    }\n"
                "  ]\n"
                "}"
            ),
        },
    ]


def build_plan_repair_messages(
    query: str,
    tool_descriptions: list[dict[str, object]],
    *,
    initial_values: dict[str, object] | None = None,
    previous_output: str,
    failure_reason: str,
) -> list[dict[str, str]]:
    """Build messages for a plan-repair retry after invalid planner output."""
    rendered_initial_values = {
        key: value
        for key, value in (initial_values or {}).items()
        if value not in (None, "", [], {})
    }
    return [
        {
            "role": "system",
            "content": (
                "You are an expert AI planning agent repairing a failed tool plan.\n\n"
                "Return ONLY valid JSON.\n"
                "Do not explain mistakes.\n"
                "Do not wrap JSON in markdown.\n"
                "Only use tools from the provided catalog.\n"
                "Every step must include tool_name, inputs, output_key, and reason.\n"
                "Use references like $query, $context, or $state.output_key.field.\n"
                "Use output_key 'final' for the final user-facing result whenever possible.\n"
                "Never use an input/source file path as the create_output filename.\n"
                "Never overwrite a file that was read earlier in the plan.\n"
                "create_output must only write to disk when write_output is true.\n"
                "Set write_output to true only when the user explicitly asked to save, write, or export the output.\n"
                "Use determine_output_write when you need to decide whether output should be written.\n"
                "A save/export request is not complete until create_output is included with content and filename.\n"
                "Never return a plan that only checks write intent for a save/export request.\n"
                "If no tool is needed, return {\"plan\": []}.\n"
            ),
        },
        {
            "role": "user",
            "content": (
                f"User Query:\n{query.strip()}\n\n"
                f"Available Tools:\n{json.dumps(tool_descriptions, indent=2)}\n\n"
                "Initial Values:\n"
                "- $query: the raw user query\n"
                "- $context: additional context or memory\n\n"
                f"Additional Initial Values:\n{json.dumps(rendered_initial_values, indent=2)}\n\n"
                f"Previous planner failure:\n{failure_reason.strip()}\n\n"
                f"Previous planner output:\n{previous_output.strip()}\n\n"
                "Repair the plan and return ONLY valid JSON in this format:\n\n"
                "{\n"
                '  "plan": [\n'
                "    {\n"
                '      "step": 1,\n'
                '      "tool_name": "tool_name_here",\n'
                '      "inputs": { "param": "value" },\n'
                '      "output_key": "state_key_here",\n'
                '      "reason": "why this step is needed"\n'
                "    }\n"
                "  ]\n"
                "}"
            ),
        },
    ]
