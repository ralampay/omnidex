"""Rich-based interactive orchestrator agent."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Sequence

from rich.console import Console, Group
from rich.live import Live
from rich.markdown import Markdown
from rich.panel import Panel
from rich.prompt import Prompt
from rich.spinner import Spinner
from rich.text import Text

from omnidex.memory import MemoryManager
from omnidex.agents.research_assistant import ResearchAssistant
from omnidex.runtime import LocalChatModel, LocalLLMSettings


DEFAULT_SYSTEM_PROMPT = """You are OmniDex, a local orchestrator agent.

You run entirely against a local model and help the user interact with local agents.
For now, behave as a capable general chat assistant. Be concise, accurate, and explicit
about uncertainty. Do not claim to access remote services or tools unless the user asks
for code or instructions to add them later.
"""

ROUTER_SYSTEM_PROMPT = """You are a routing agent.

Your task is to choose the best handler for the user input.

Available options:
- chat: general conversation
- research_assistant: research queries, summarization, PDFs, keyword extraction

Rules:
- Choose "research_assistant" if the query involves:
  - summarization
  - PDFs
  - documents
  - extracting keywords
  - analysis of text
- Otherwise choose "chat"

Respond with ONLY one word:
chat
or
research_assistant

Do not explain your answer.
"""


class OrchestratorAgent:
    """Local first-chat agent with a Rich terminal UI."""

    def __init__(
        self,
        *,
        console: Console | None = None,
        system_prompt: str = DEFAULT_SYSTEM_PROMPT,
    ) -> None:
        self.console = console or Console()
        self.system_prompt = system_prompt
        self.settings = LocalLLMSettings.from_env(system_prompt=system_prompt)
        self.model = LocalChatModel(self.settings)
        self.memory = MemoryManager(
            short_term_limit=self._short_term_limit(),
            long_term_path=self._memory_path(),
        )
        self.agents = {
            "research_assistant": ResearchAssistant(console=self.console),
        }

    def run(self) -> int:
        """Run the interactive chat session."""
        self._render_banner()
        self._render_help()

        while True:
            try:
                prompt = Prompt.ask("[bold cyan]You[/bold cyan]")
            except (EOFError, KeyboardInterrupt):
                self.console.print()
                self.console.print("[yellow]Session closed.[/yellow]")
                return 0

            command_result = self._handle_command(prompt.strip())
            if command_result is not None:
                return command_result

    def ask(self, prompt: str, *, stream: bool | None = None) -> str:
        """Send one prompt and render the assistant response."""
        user_prompt = prompt.strip()
        if not user_prompt:
            return ""

        context = self.memory.get_context(user_prompt)
        with self.console.status(
            "[bold blue]OmniDex is routing...[/bold blue]",
            spinner="dots",
        ):
            try:
                route = self._llm_route(user_prompt, context)
            except Exception:
                route = "chat"
        self._render_event(f"Route selected: {route} (llm)", style="blue")
        self._debug_route(route)

        if route == "chat":
            response, already_rendered = self._generate_response(
                user_prompt,
                context=context,
                stream=stream,
            )
            responder = "OmniDex"
        else:
            agent = self.agents[route]
            self._render_event(f"Delegating to agent: {agent.name}", style="cyan")
            with self.console.status(
                f"[bold cyan]{agent.name} is working...[/bold cyan]",
                spinner="dots",
            ):
                response = agent.safe_run(user_prompt, context=context)
            already_rendered = False
            responder = agent.name

        self.memory.add_interaction("user", user_prompt)
        self.memory.add_interaction("assistant", response)
        self.memory.extract_and_store(user_prompt, response)
        self._render_event(f"Response source: {responder}", style="green")
        if not already_rendered:
            self._render_response(response, title=responder)
        return response

    def _llm_route(self, user_input: str, context: str) -> str:
        """Ask the local model to choose a route and normalize the result."""
        routing_input = user_input.strip()
        if context.strip():
            routing_input = (
                f"Context:\n{context.strip()}\n\n"
                f"User input:\n{routing_input}"
            )

        messages = [
            {"role": "system", "content": ROUTER_SYSTEM_PROMPT},
            {"role": "user", "content": routing_input},
        ]
        completion = self.model.complete(messages, stream=False)
        choice = completion["choices"][0]
        output = choice["message"]["content"].strip().lower()
        if "research" in output:
            return "research_assistant"
        if "chat" in output:
            return "chat"
        return "chat"

    def _debug_route(self, route: str) -> None:
        """Print a dim route trace when verbose logging is enabled."""
        if self.settings.verbose:
            self.console.print(f"[dim]Route selected:[/dim] {route}")

    def _generate_response(
        self,
        user_prompt: str,
        *,
        context: str,
        stream: bool | None = None,
    ) -> tuple[str, bool]:
        """Generate a response from the local model."""
        should_stream = self.settings.stream if stream is None else stream
        if should_stream:
            completion = self.model.complete(
                self._build_messages(user_prompt, context),
                stream=True,
            )
            return self._collect_stream(completion), True

        with self.console.status(
            "[bold magenta]OmniDex is thinking...[/bold magenta]",
            spinner="dots",
        ):
            completion = self.model.complete(
                self._build_messages(user_prompt, context),
                stream=False,
            )

        choice = completion["choices"][0]
        message = choice["message"]["content"]
        return message.strip(), False

    def _collect_stream(self, events: Sequence[dict] | object) -> str:
        """Render streaming tokens as they arrive."""
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

    def _handle_command(self, raw_input: str) -> int | None:
        """Handle slash commands; return an exit code when the session should stop."""
        if not raw_input:
            return None

        if not raw_input.startswith("/"):
            self.ask(raw_input)
            return None

        command = raw_input.strip().lower()
        if command in {"/exit", "/quit"}:
            self.console.print("[yellow]Session closed.[/yellow]")
            return 0
        if command in {"/clear", "/reset"}:
            self.memory.clear_short_term()
            self.console.print("[green]Conversation reset.[/green]")
            return None
        if command == "/help":
            self._render_help()
            return None

        self.console.print(f"[red]Unknown command:[/red] {raw_input}")
        self._render_help()
        return None

    def _render_banner(self) -> None:
        """Show the session banner."""
        lines = Group(
            Text("OmniDex Orchestrator", style="bold white"),
            Text("Local GGUF runtime via llama-cpp-python", style="cyan"),
            Text(str(self.settings.model_path), style="dim"),
        )
        self.console.print(Panel(lines, border_style="blue"))

    def _render_help(self) -> None:
        """Show interactive help."""
        self.console.print(
            Panel(
                "/help  show commands\n"
                "/clear reset the conversation\n"
                "/exit  leave the session",
                title="Commands",
                border_style="green",
            )
        )

    def _render_event(self, message: str, *, style: str = "cyan") -> None:
        """Render a standard orchestrator event line."""
        title = Text("orchestrator", style=f"bold {style}")
        body = Text(message)
        self.console.print(
            Panel.fit(body, title=title, border_style=style, padding=(0, 1))
        )

    def _render_response(self, response: str, *, title: str = "OmniDex") -> None:
        """Render the assistant response."""
        self.console.print(self._assistant_panel(response, title=title))

    def _assistant_panel(
        self,
        response: str,
        *,
        status: str | None = None,
        title: str = "OmniDex",
    ) -> Panel:
        """Build the assistant panel for plain text or Markdown output."""
        if status and not response:
            body = Group(
                Spinner("dots", text=status, style="magenta"),
                Text(" ", style="dim"),
            )
        else:
            body = (
                Markdown(response or " ")
                if self.settings.render_markdown
                else Text(response or " ")
            )
        return Panel(body, title=title, border_style="magenta")

    def _build_messages(self, user_prompt: str, context: str) -> list[dict[str, str]]:
        """Construct the message list sent to the LLM."""
        system_content = self.system_prompt.strip()
        memory_context = context.strip()
        if memory_context:
            system_content = (
                f"{system_content}\n\n"
                "Use the memory context below when it is relevant. "
                "Do not repeat it verbatim unless the user asks.\n\n"
                f"{memory_context}"
            )

        return [
            {"role": "system", "content": system_content},
            {"role": "user", "content": user_prompt},
        ]

    def _memory_path(self) -> Path:
        """Resolve the persistent long-term memory location."""
        configured = os.getenv("OMNIDEX_MEMORY_PATH")
        if configured:
            return Path(configured).expanduser()
        return Path(".omnidex/orchestrator_memory.json")

    def _short_term_limit(self) -> int:
        """Resolve the short-term memory window size."""
        configured = os.getenv("OMNIDEX_SHORT_TERM_LIMIT")
        if not configured:
            return 5
        return max(1, int(configured))
