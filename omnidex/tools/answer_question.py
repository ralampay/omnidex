"""Tool for answering a question from provided context and evidence."""

from __future__ import annotations

from omnidex.agents.research_assistant.prompts import (
    build_answer_messages,
    build_system_prompt,
)
from omnidex.tools.base import BaseTool
from omnidex.utils.text import truncate_text


class AnswerQuestionTool(BaseTool):
    """Answer a question using optional memory context and evidence excerpts."""

    name = "answer_question"
    output_fields = ("answer", "content")

    def __init__(self, *, model: object, system_prompt: str) -> None:
        self.model = model
        self.system_prompt = system_prompt

    def run(
        self,
        query: str,
        context: str = "",
        evidence: str = "",
    ) -> dict[str, object]:
        """Generate an answer grounded in the provided inputs."""
        normalized_query = self._coerce_text(query)
        normalized_context = self._coerce_text(context)
        normalized_evidence = self._coerce_text(evidence)
        if not normalized_query:
            return {
                "status": "error",
                "answer": "",
                "content": "Error: No query was provided for answer generation.",
            }
        system_prompt = build_system_prompt(
            self.system_prompt,
            context=truncate_text(normalized_context, limit=1800),
            pdf_text=normalized_evidence,
        )
        answer = self.model.generate_text(
            build_answer_messages(system_prompt, normalized_query),
            stream=False,
        )
        return {
            "status": "ok",
            "answer": answer,
            "content": answer,
        }
