"""Rich-based interactive orchestrator agent."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Sequence

from rich.console import Console, Group
from rich.live import Live
from rich.markdown import Markdown
from rich.prompt import Prompt
from rich.spinner import Spinner
from rich.text import Text

from omnidex.memory import MemoryManager
from omnidex.agents.research_assistant import ResearchAssistant
from omnidex.runtime import LocalChatModel, LocalLLMSettings
from omnidex.utils.json_tools import load_json_object


DEFAULT_SYSTEM_PROMPT = """You are OmniDex, a local orchestrator agent.

You run entirely against a local model and help the user interact with local agents.
For now, behave as a capable general chat assistant. Be concise, accurate, and explicit
about uncertainty. Do not claim to access remote services or tools unless the user asks
for code or instructions to add them later. Depending on the inquiry, you should delegate 
tasks to one or more of the following agents:
- research_assistant
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
        self.session_state: dict[str, object] = {
            "last_response": "",
            "last_artifact_content": "",
            "last_responder": "",
            "last_tools_used": [],
        }
        self.agents = {
            "research_assistant": ResearchAssistant(
                console=self.console,
                verbose=True,
            ),
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

        context = self._bounded_context(self.memory.get_context(user_prompt))
        routing_context = self._routing_context(context)
        with self.console.status(
            "[bold blue]OmniDex is routing...[/bold blue]",
            spinner="dots",
        ):
            try:
                route, confidence, route_source = self._route(user_prompt, routing_context)
            except Exception:
                route = "chat"
                confidence = 0.0
                route_source = "default_fallback"
        self._render_event(
            f"Route selected: {route} ({route_source}, confidence={confidence:.2f})",
            style="blue",
        )
        self._debug_route(route, confidence, route_source)

        if route == "chat":
            self._render_event("Active agent: chat", style="cyan")
            response, already_rendered = self._generate_response(
                user_prompt,
                context=context,
                stream=stream,
            )
            responder = "OmniDex"
            tools_used: list[str] = []
        else:
            agent = self.agents[route]
            if hasattr(agent, "session_state"):
                agent.session_state = dict(self.session_state)
            self._render_event(f"Active agent: {agent.name}", style="cyan")
            self._render_event(f"Delegating to agent: {agent.name}", style="cyan")
            with self.console.status(
                f"[bold cyan]{agent.name} is working...[/bold cyan]",
                spinner="dots",
            ):
                response = agent.safe_run(user_prompt, context=context)
            already_rendered = False
            responder = agent.name
            tools_used = list(agent.last_used_tools)
            delegated_state = dict(getattr(agent, "session_state", {}))

        self.memory.add_interaction("user", user_prompt)
        self.memory.add_interaction("assistant", response)
        self.memory.extract_and_store(user_prompt, response)
        preserved_artifact = str(self.session_state.get("last_artifact_content", "") or "").strip()
        if route == "chat":
            self.session_state = {
                "last_response": response,
                "last_artifact_content": preserved_artifact,
                "last_responder": responder,
                "last_tools_used": tools_used,
            }
        else:
            delegated_artifact = str(
                delegated_state.get("last_artifact_content", "") or ""
            ).strip()
            self.session_state = {
                "last_response": str(delegated_state.get("last_response") or response),
                "last_artifact_content": delegated_artifact or preserved_artifact,
                "last_responder": str(delegated_state.get("last_responder") or responder),
                "last_tools_used": delegated_state.get("last_tools_used", tools_used),
            }
        self._render_event(
            f"Delegated agent tools used: {', '.join(tools_used) if tools_used else 'none'}",
            style="yellow",
        )
        self._render_event(f"Response source: {responder}", style="green")
        if not already_rendered:
            self._render_response(response, title=responder)
        return response

    def _route(self, user_input: str, context: str) -> tuple[str, float, str]:
        """Route through the model over registered handlers."""
        route, confidence = self._llm_route(user_input, context)
        if route in self._route_options():
            return route, confidence, "llm"
        return "chat", 0.0, "default_fallback"

    def _llm_route(self, user_input: str, context: str) -> tuple[str, float]:
        """Ask the local model to choose a route from registered options."""
        routing_input = user_input.strip()
        if context.strip():
            routing_input = (
                f"Context:\n{self._bounded_context(context, limit=1200)}\n\n"
                f"User input:\n{routing_input}"
            )

        messages = [
            {"role": "system", "content": self._build_router_system_prompt()},
            {"role": "user", "content": routing_input},
        ]
        output = self.model.generate_text(messages, stream=False)
        payload = load_json_object(output)
        if payload is None:
            normalized_output = " ".join(output.lower().split())
            if normalized_output in self._route_options():
                return normalized_output, 1.0
            return "", 0.0

        route = str(payload.get("route", "")).strip()
        raw_confidence = payload.get("confidence", 0.0)
        try:
            confidence = float(raw_confidence)
        except (TypeError, ValueError):
            confidence = 0.0
        return route, max(0.0, min(1.0, confidence))

    def _route_options(self) -> set[str]:
        """Return the available route names."""
        return {"chat", *self.agents.keys()}

    def _build_router_system_prompt(self) -> str:
        """Build a routing prompt from the registered agents."""
        options = [
            {
                "name": "chat",
                "description": "General conversation handled directly by the orchestrator.",
            }
        ]
        for name, agent in self.agents.items():
            description = getattr(agent, "description", "") or f"Delegates to {name}."
            options.append({"name": name, "description": description})

        option_lines = "\n".join(
            f'- {option["name"]}: {option["description"]}'
            for option in options
        )
        return (
            "You are a routing agent.\n\n"
            "Choose the best handler for the user input from the available options.\n"
            "You will receive both conversation memory and session artifact state.\n"
            "When the user refers to a previous artifact using phrases like 'save it', "
            "'save this', 'write that', or 'export it', interpret that as a continuation "
            "of the previous artifact if the session artifact state supports it.\n"
            "Prefer the same handler that produced the prior artifact when the current "
            "request is about persisting, exporting, or transforming that artifact.\n"
            "Do not ignore session artifact state when the current user input is short or ambiguous.\n"
            "Return ONLY valid JSON in this format:\n"
            '{\n  "route": "chat",\n  "confidence": 0.0\n}\n\n'
            f"Available options:\n{option_lines}\n"
        )

    def _routing_context(self, memory_context: str) -> str:
        """Combine memory context with short-lived session artifact state."""
        last_response = str(self.session_state.get("last_response", "") or "").strip()
        last_artifact_content = str(
            self.session_state.get("last_artifact_content", "") or ""
        ).strip()
        last_responder = str(self.session_state.get("last_responder", "") or "").strip()
        last_tools_used = self.session_state.get("last_tools_used", [])
        session_lines = ["Session Artifact State:"]
        if last_response:
            excerpt = self._bounded_context(last_response, limit=500)
            session_lines.extend(
                [
                    f"Last responder: {last_responder or 'unknown'}",
                    f"Last tools used: {', '.join(last_tools_used) if last_tools_used else 'none'}",
                    f"Last response excerpt:\n{excerpt}",
                ]
            )
            if last_artifact_content:
                artifact_excerpt = self._bounded_context(last_artifact_content, limit=500)
                session_lines.append(f"Last artifact excerpt:\n{artifact_excerpt}")
        else:
            session_lines.append("(no prior artifact)")
        return "\n\n".join([memory_context, "\n".join(session_lines)])

    def _debug_route(self, route: str, confidence: float, source: str) -> None:
        """Print a dim route trace when verbose logging is enabled."""
        if self.settings.verbose:
            trace = Text("Route selected: ", style="dim")
            trace.append(route, style="bold dim")
            trace.append(" ", style="dim")
            trace.append(f"(source={source}, confidence={confidence:.2f})", style="dim")
            self.console.print(trace)

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
            message = self.model.generate_text(
                self._build_messages(user_prompt, context),
                stream=False,
            )

        return message, False

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
            self.session_state = {
                "last_response": "",
                "last_artifact_content": "",
                "last_responder": "",
                "last_tools_used": [],
            }
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
        self.console.print(Text("OmniDex Orchestrator", style="bold white"))
        self.console.print(
            Text("Local GGUF runtime via llama-cpp-python", style="cyan")
        )
        self.console.print(Text(str(self.settings.model_path), style="dim"))
        self.console.print()

    def _render_help(self) -> None:
        """Show interactive help."""
        self.console.print(Text("Commands:", style="bold green"))
        self.console.print(Text("/help  show commands", style="green"))
        self.console.print(Text("/clear reset the conversation", style="green"))
        self.console.print(Text("/exit  leave the session", style="green"))
        self.console.print()

    def _render_event(self, message: str, *, style: str = "cyan") -> None:
        """Render a standard orchestrator event line."""
        speaker = Text("orchestrator: ", style=f"bold {style}")
        speaker.append(message)
        self.console.print(speaker)

    def _render_response(self, response: str, *, title: str = "OmniDex") -> None:
        """Render the assistant response."""
        self.console.print(Text(f"{title}:", style="bold magenta"))
        body = (
            Markdown(response or " ")
            if self.settings.render_markdown
            else Text(response or " ")
        )
        self.console.print(body)

    def _assistant_panel(
        self,
        response: str,
        *,
        status: str | None = None,
        title: str = "OmniDex",
    ):
        """Build the assistant panel for plain text or Markdown output."""
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

    def _build_messages(self, user_prompt: str, context: str) -> list[dict[str, str]]:
        """Construct the message list sent to the LLM."""
        system_content = self.system_prompt.strip()
        memory_context = self._bounded_context(context)
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

    def _bounded_context(self, context: str, *, limit: int = 2500) -> str:
        """Trim memory context to a safe length for local model prompts."""
        normalized = context.strip()
        if len(normalized) <= limit:
            return normalized
        return f"{normalized[:limit].rstrip()}\n\n[Context truncated]"

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
