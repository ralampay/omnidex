"""LLM-backed general chat agent."""

from __future__ import annotations

from typing import Iterable

from rich.console import Group
from rich.live import Live
from rich.markdown import Markdown
from rich.spinner import Spinner
from rich.text import Text

from omnidex.agents.base import BaseAgent
from omnidex.agents.handoffs import HandoffDecision
from omnidex.agents.chat_agent.prompts import (
    DEFAULT_SYSTEM_PROMPT,
    build_answer_messages,
    build_handoff_messages,
    build_system_prompt,
)
from omnidex.agents.chat_agent.types import SessionState
from omnidex.runtime import LocalChatModel, LocalLLMSettings
from omnidex.utils.json_tools import load_json_object


class ChatAgent(BaseAgent):
    """General conversation agent that relies on shared context, not tools."""

    name = "chat_agent"
    description = (
        "Handles general conversation and questions about the current session "
        "context or active artifact without using tools."
    )

    def __init__(self, **kwargs) -> None:
        """Initialize the chat agent and its local GGUF-backed chat model."""
        verbose = bool(kwargs.pop("verbose", True))
        super().__init__(tools=[], verbose=verbose, **kwargs)
        self.settings = LocalLLMSettings.from_env(
            system_prompt=DEFAULT_SYSTEM_PROMPT
        )
        self.model = LocalChatModel(self.settings)
        self._stream_override: bool | None = None

    def set_stream_override(self, stream: bool | None) -> None:
        """Override the default stream behavior for the next response only."""
        self._stream_override = stream

    def _bounded_context(self, text: str, *, limit: int = 1200) -> str:
        """Trim shared context to a safe prompt size."""
        normalized = text.strip()
        if len(normalized) <= limit:
            return normalized
        return f"{normalized[:limit].rstrip()}\n\n[Context truncated]"

    def _session_artifact_context(self) -> str:
        """Render compact session artifact state for the prompt."""
        last_response = self._bounded_context(
            str(self.session_state.get("last_response", "") or ""),
            limit=500,
        )
        last_artifact_content = self._bounded_context(
            str(self.session_state.get("last_artifact_content", "") or ""),
            limit=900,
        )
        last_artifact_responder = str(
            self.session_state.get("last_artifact_responder", "") or ""
        ).strip()
        last_responder = str(self.session_state.get("last_responder", "") or "").strip()

        lines = []
        if last_responder:
            lines.append(f"Last responder: {last_responder}")
        if last_artifact_responder:
            lines.append(f"Active artifact owner: {last_artifact_responder}")
        if last_response:
            lines.append(f"Last response excerpt:\n{last_response}")
        if last_artifact_content:
            lines.append(f"Active artifact excerpt:\n{last_artifact_content}")
        return "\n\n".join(lines)

    def propose_handoff(
        self,
        query: str,
        *,
        context: str = "",
        available_agents: tuple[str, ...] = (),
    ) -> HandoffDecision | None:
        """Use the local model to decide whether another agent should take over."""
        messages = build_handoff_messages(
            system_prompt=self.settings.system_prompt,
            query=query,
            context=self._bounded_context(context, limit=1200),
            session_artifact_context=self._session_artifact_context(),
            available_agents=available_agents,
        )
        output = self.model.generate_text(messages, stream=False)
        payload = load_json_object(output)
        if payload is None:
            return None

        action = str(payload.get("action", "")).strip().casefold()
        if action != "handoff":
            return None

        target_agent = str(payload.get("target_agent", "")).strip()
        if not target_agent or target_agent == self.name:
            return None

        try:
            confidence = float(payload.get("confidence", 0.0))
        except (TypeError, ValueError):
            confidence = 0.0

        return HandoffDecision(
            target_agent=target_agent,
            reason=str(payload.get("reason", "")).strip(),
            confidence=max(0.0, min(1.0, confidence)),
        )

    def _build_messages(self, query: str, context: str) -> list[dict[str, str]]:
        """Construct the message list sent to the local model."""
        system_prompt = build_system_prompt(
            self.settings.system_prompt,
            context=self._bounded_context(context, limit=1800),
            session_artifact_context=self._session_artifact_context(),
        )
        return build_answer_messages(system_prompt, query)

    def _assistant_panel(
        self,
        response: str,
        *,
        status: str | None = None,
    ):
        """Build a Rich renderable for streaming chat output."""
        if status and not response:
            return Group(
                Spinner("dots", text=status, style="magenta"),
                Text(" ", style="dim"),
            )
        return (
            Markdown(response or " ")
            if self.settings.render_markdown
            else Text(response or " ")
        )

    def _collect_stream(self, events: Iterable[dict]) -> str:
        """Render streaming tokens as they arrive and return the full response."""
        chunks: list[str] = []
        with Live(
            self._assistant_panel("", status="Thinking..."),
            console=self.console,
            refresh_per_second=12,
        ) as live:
            for event in events:
                delta = event["choices"][0].get("delta", {})
                piece = delta.get("content")
                if not piece:
                    continue
                chunks.append(piece)
                live.update(self._assistant_panel("".join(chunks)))
        self.console.print()
        return "".join(chunks).strip()

    def _generate_response(self, query: str, *, context: str) -> str:
        """Generate a response from the local model with optional streaming."""
        should_stream = (
            self.settings.stream
            if self._stream_override is None
            else self._stream_override
        )
        try:
            if should_stream:
                completion = self.model.complete(
                    self._build_messages(query, context),
                    stream=True,
                )
                return self._collect_stream(completion)

            message = self.model.generate_text(
                self._build_messages(query, context),
                stream=False,
            )
            return message.strip()
        finally:
            self._stream_override = None

    def run(self, query: str, context: str = "") -> str:
        """Answer a general chat request using shared context and artifact state."""
        self.log(f"Running chat query: {query}")
        response = self._generate_response(query, context=context)
        self.update_session_state(response=response)
        self.log("Chat query completed.")
        return response
