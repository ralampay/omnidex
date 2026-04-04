"""Base abstractions for OmniDex execution agents."""

from __future__ import annotations

from typing import Any, TypedDict

from omnidex.agents.handoffs import HandoffDecision
from rich.console import Console
from rich.text import Text


class SessionState(TypedDict, total=False):
    """Shared cross-agent session and artifact continuity contract."""

    last_response: str
    last_artifact_content: str
    last_artifact_responder: str
    last_responder: str
    last_tools_used: list[str]
    artifact_history: list[dict[str, object]]


class BaseAgent:
    """Foundation class for execution-focused agents."""

    name = "base_agent"
    description = "Base execution agent"

    def __init__(
        self,
        *,
        memory: Any | None = None,
        verbose: bool = False,
        console: Console | None = None,
        tools: list[Any] | None = None,
    ) -> None:
        """Initialize the agent with optional shared memory and logging."""
        self.memory = memory
        self.verbose = verbose
        self.console = console or Console()
        self.tools = list(tools or [])
        self.last_used_tools: list[str] = []
        self.session_state: SessionState = self.empty_session_state()

    @classmethod
    def empty_session_state(cls) -> SessionState:
        """Return the canonical empty shared session-state payload."""
        return {
            "last_response": "",
            "last_artifact_content": "",
            "last_artifact_responder": "",
            "last_responder": "",
            "last_tools_used": [],
            "artifact_history": [],
        }

    def copy_session_state(self) -> SessionState:
        """Return a shallow copy of the current shared session state."""
        return {
            "last_response": str(self.session_state.get("last_response", "") or ""),
            "last_artifact_content": str(
                self.session_state.get("last_artifact_content", "") or ""
            ),
            "last_artifact_responder": str(
                self.session_state.get("last_artifact_responder", "") or ""
            ),
            "last_responder": str(self.session_state.get("last_responder", "") or ""),
            "last_tools_used": list(self.session_state.get("last_tools_used", [])),
            "artifact_history": [
                dict(item)
                for item in self.session_state.get("artifact_history", [])
                if isinstance(item, dict)
            ],
        }

    def apply_session_state(self, state: SessionState | dict[str, object] | None) -> None:
        """Load shared session state while preserving the canonical contract shape."""
        incoming = state or {}
        self.session_state = {
            "last_response": str(incoming.get("last_response", "") or ""),
            "last_artifact_content": str(
                incoming.get("last_artifact_content", "") or ""
            ),
            "last_artifact_responder": str(
                incoming.get("last_artifact_responder", "") or ""
            ),
            "last_responder": str(incoming.get("last_responder", "") or ""),
            "last_tools_used": list(incoming.get("last_tools_used", [])),
            "artifact_history": [
                dict(item)
                for item in incoming.get("artifact_history", [])
                if isinstance(item, dict)
            ],
        }

    def update_session_state(
        self,
        *,
        response: str,
        artifact_content: str | None = None,
        artifact_responder: str | None = None,
    ) -> None:
        """Update the shared session state after an agent response."""
        preserved_artifact = (
            str(self.session_state.get("last_artifact_content", "") or "").strip()
        )
        next_artifact = (
            preserved_artifact
            if artifact_content is None
            else str(artifact_content or "").strip()
        )
        preserved_artifact_responder = str(
            self.session_state.get("last_artifact_responder", "") or ""
        ).strip()
        next_artifact_responder = (
            preserved_artifact_responder
            if artifact_responder is None
            else str(artifact_responder or "").strip()
        )
        artifact_history = [
            dict(item)
            for item in self.session_state.get("artifact_history", [])
            if isinstance(item, dict)
        ]
        if (
            artifact_content is not None
            and next_artifact
            and next_artifact != preserved_artifact
        ):
            artifact_history.append(
                {
                    "content": next_artifact,
                    "responder": next_artifact_responder or self.name,
                    "response": response,
                    "tools_used": list(self.last_used_tools),
                }
            )
            artifact_history = artifact_history[-8:]
        if not next_artifact:
            next_artifact_responder = ""
        self.session_state = {
            "last_response": response,
            "last_artifact_content": next_artifact,
            "last_artifact_responder": next_artifact_responder,
            "last_responder": self.name,
            "last_tools_used": list(self.last_used_tools),
            "artifact_history": artifact_history,
        }

    def run(self, query: str, context: str = "") -> str:
        """Execute the agent's core task for the given query and context."""
        raise NotImplementedError("Subclasses must implement run().")

    def propose_handoff(
        self,
        query: str,
        *,
        context: str = "",
        available_agents: tuple[str, ...] = (),
    ) -> HandoffDecision | None:
        """Optionally ask the orchestrator to hand off to another agent."""
        return None

    def __call__(self, query: str, context: str = "") -> str:
        """Invoke the agent like a callable."""
        return self.run(query, context)

    def before_run(self, query: str, context: str) -> None:
        """Hook called before agent execution."""
        self.last_used_tools = []
        self.emit("Starting task.", style="cyan")

    def after_run(self, query: str, response: str) -> None:
        """Hook called after agent execution."""
        self.emit("Response ready.", style="green")

    def safe_run(self, query: str, context: str = "") -> str:
        """Execute the agent with lifecycle hooks and failure handling."""
        try:
            self.before_run(query, context)
            response = self.run(query, context)
            self.after_run(query, response)
            return response
        except Exception as exc:
            self.emit(f"Execution failed: {exc}", style="red")
            return "Agent failed to process the request."

    def emit(self, message: str, *, style: str = "cyan") -> None:
        """Print a standard agent event line."""
        speaker = Text(f"{self.name}: ", style=f"bold {style}")
        speaker.append(message)
        self.console.print(speaker)

    def get_tool(self, name: str) -> Any | None:
        """Return a tool by name from the agent's registered tool list."""
        for tool in self.tools:
            if getattr(tool, "name", None) == name:
                return tool
        return None

    def record_tool_use(self, tool_name: str, *, reason: str = "") -> None:
        """Record and announce a tool usage event."""
        self.last_used_tools.append(tool_name)
        if reason:
            self.emit(f"Using tool: {tool_name} ({reason})", style="yellow")
        else:
            self.emit(f"Using tool: {tool_name}", style="yellow")

    def log(self, message: str) -> None:
        """Print a verbose debug message when enabled."""
        if self.verbose:
            prefix = Text(f"[{self.name}] ", style="dim")
            prefix.append(message, style="dim")
            self.console.print(prefix)


class EchoAgent(BaseAgent):
    """Simple example agent that echoes the incoming query."""

    name = "echo_agent"
    description = "Echoes the provided query."

    def run(self, query: str, context: str = "") -> str:
        """Return the query as a plain echoed response."""
        return f"Echo: {query}"


__all__ = ["BaseAgent", "EchoAgent", "SessionState"]
