"""Base abstractions for OmniDex execution agents."""

from __future__ import annotations

from typing import Any

from rich.console import Console
from rich.panel import Panel
from rich.text import Text


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

    def run(self, query: str, context: str = "") -> str:
        """Execute the agent's core task for the given query and context."""
        raise NotImplementedError("Subclasses must implement run().")

    def __call__(self, query: str, context: str = "") -> str:
        """Invoke the agent like a callable."""
        return self.run(query, context)

    def before_run(self, query: str, context: str) -> None:
        """Hook called before agent execution."""
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
        title = Text(self.name, style=f"bold {style}")
        body = Text(message)
        self.console.print(
            Panel.fit(body, title=title, border_style=style, padding=(0, 1))
        )

    def get_tool(self, name: str) -> Any | None:
        """Return a tool by name from the agent's registered tool list."""
        for tool in self.tools:
            if getattr(tool, "name", None) == name:
                return tool
        return None

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
