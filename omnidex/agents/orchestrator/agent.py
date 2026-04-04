"""Rich-based interactive orchestrator agent."""

from __future__ import annotations

import os
from pathlib import Path

from rich.console import Console
from rich.markdown import Markdown
from rich.prompt import Prompt
from rich.text import Text

from omnidex.agents.base import BaseAgent, SessionState
from omnidex.agents.chat_agent import ChatAgent
from omnidex.agents.handoffs import HandoffDecision
from omnidex.agents.policy import AgentPolicyValidator
from omnidex.memory import MemoryManager
from omnidex.agents.research_assistant import ResearchAssistant
from omnidex.runtime import LocalChatModel, LocalLLMSettings
from omnidex.utils.json_tools import load_json_object


DEFAULT_SYSTEM_PROMPT = """You are OmniDex, a local orchestrator agent.

You run entirely against a local model and help the user interact with local agents.
Be concise, accurate, and explicit about uncertainty. Route each turn to the best
specialized agent instead of answering as the final responder yourself.
Available agents:
- chat_agent
- research_assistant
"""

class OrchestratorAgent:
    """Delegating orchestrator with a Rich terminal UI."""

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
        self.session_state: SessionState = BaseAgent.empty_session_state()
        self.agents = {
            "chat_agent": ChatAgent(
                console=self.console,
                verbose=True,
            ),
            "research_assistant": ResearchAssistant(
                console=self.console,
                verbose=True,
            ),
        }
        self.policy = AgentPolicyValidator(agents=self.agents)

    def run(self) -> int:
        """Run the interactive orchestration session."""
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
                route = "chat_agent"
                confidence = 0.0
                route_source = "default_fallback"
        self._render_event(
            f"Route selected: {route} ({route_source}, confidence={confidence:.2f})",
            style="blue",
        )
        self._debug_route(route, confidence, route_source)
        response, responder, tools_used, delegated_state = self._delegate_with_handoffs(
            route=route,
            user_prompt=user_prompt,
            context=context,
            stream=stream,
        )
        already_rendered = False

        self.memory.add_interaction("user", user_prompt)
        self.memory.add_interaction("assistant", response)
        self.memory.extract_and_store(user_prompt, response)
        preserved_artifact = str(self.session_state.get("last_artifact_content", "") or "").strip()
        preserved_artifact_responder = str(
            self.session_state.get("last_artifact_responder", "") or ""
        ).strip()
        delegated_artifact = str(
            delegated_state.get("last_artifact_content", "") or ""
        ).strip()
        delegated_artifact_responder = str(
            delegated_state.get("last_artifact_responder", "") or ""
        ).strip()
        self.session_state = {
            "last_response": str(delegated_state.get("last_response") or response),
            "last_artifact_content": delegated_artifact or preserved_artifact,
            "last_artifact_responder": (
                delegated_artifact_responder
                or (preserved_artifact_responder if delegated_artifact or preserved_artifact else "")
            ),
            "last_responder": str(delegated_state.get("last_responder") or responder),
            "last_tools_used": delegated_state.get("last_tools_used", tools_used),
            "artifact_history": delegated_state.get(
                "artifact_history",
                self.session_state.get("artifact_history", []),
            ),
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
        proposed_route, confidence = self._llm_route(user_input, context)
        validation = self.policy.validate_initial_route(
            proposed_route=proposed_route,
            query=user_input,
            session_state=self.session_state,
        )
        return validation.target_agent, confidence, validation.source

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
        return set(self.agents.keys())

    def _build_router_system_prompt(self) -> str:
        """Build a routing prompt from the registered agents."""
        options = []
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
            "Pick the best initial agent, not necessarily the final one. "
            "The selected agent may request a handoff after seeing the turn.\n"
            "Prefer chat_agent for generic conversation, questions about the current "
            "content, requests to explain a term from the latest artifact, and normal "
            "follow-up discussion about previously generated output.\n"
            "Prefer research_assistant for explicit research workflows such as PDF "
            "ingestion, summarization, insight extraction, structured research analysis, "
            "or artifact persistence/transformation requests.\n"
            "When the user refers to a previous artifact using phrases like 'save it', "
            "'save this', 'write that', or 'export it', interpret that as a continuation "
            "of the previous artifact if the session artifact state supports it.\n"
            "Prefer the same handler that owns the active artifact when the current "
            "request is about persisting, exporting, or transforming that artifact.\n"
            "Do not ignore session artifact state when the current user input is short or ambiguous.\n"
            "Return ONLY valid JSON in this format:\n"
            '{\n  "route": "chat_agent",\n  "confidence": 0.0\n}\n\n'
            f"Available options:\n{option_lines}\n"
        )

    def _request_agent_handoff(
        self,
        *,
        agent_name: str,
        user_prompt: str,
        context: str,
    ) -> HandoffDecision | None:
        """Ask the current agent whether the turn should be handed off."""
        agent = self.agents[agent_name]
        proposer = getattr(agent, "propose_handoff", None)
        if proposer is None:
            return None
        try:
            decision = proposer(
                user_prompt,
                context=context,
                available_agents=tuple(self.agents.keys()),
            )
        except Exception as exc:
            self._render_event(
                f"Handoff evaluation failed for {agent_name}: {exc}",
                style="red",
            )
            return None
        if decision is None:
            return None
        if decision.target_agent not in self.agents:
            return None
        if decision.target_agent == agent_name:
            return None
        return decision

    def _delegate_with_handoffs(
        self,
        *,
        route: str,
        user_prompt: str,
        context: str,
        stream: bool | None,
    ) -> tuple[str, str, list[str], dict[str, object]]:
        """Execute the selected agent, allowing bounded model-driven handoffs."""
        current_route = route
        visited = {current_route}
        handoff_limit = 2

        for _ in range(handoff_limit + 1):
            agent = self.agents[current_route]
            if isinstance(agent, BaseAgent):
                agent.apply_session_state(self.session_state)
            if hasattr(agent, "set_stream_override"):
                agent.set_stream_override(stream)

            self._render_event(f"Active agent: {agent.name}", style="cyan")
            decision = self._request_agent_handoff(
                agent_name=current_route,
                user_prompt=user_prompt,
                context=context,
            )
            validation = self.policy.validate_handoff(
                current_agent=current_route,
                decision=decision,
                query=user_prompt,
                session_state=self.session_state,
            )
            if (
                validation.accepted
                and validation.source == "handoff_validated"
                and decision is not None
                and decision.confidence >= 0.5
                and validation.target_agent not in visited
            ):
                self._render_event(
                    "Agent handoff: "
                    f"{current_route} -> {validation.target_agent} "
                    f"(confidence={decision.confidence:.2f})"
                    + (f" [{decision.reason}]" if decision.reason else ""),
                    style="magenta",
                )
                current_route = validation.target_agent
                visited.add(current_route)
                continue
            if (
                decision is not None
                and validation.source == "handoff_rejected"
                and validation.reason
            ):
                self._render_event(
                    f"Handoff blocked: {validation.reason}",
                    style="yellow",
                )

            self._render_event(f"Delegating to agent: {agent.name}", style="cyan")
            with self.console.status(
                f"[bold cyan]{agent.name} is working...[/bold cyan]",
                spinner="dots",
            ):
                response = agent.safe_run(user_prompt, context=context)
            return (
                response,
                agent.name,
                list(agent.last_used_tools),
                agent.copy_session_state() if isinstance(agent, BaseAgent) else {},
            )

        agent = self.agents[current_route]
        self._render_event(f"Delegating to agent: {agent.name}", style="cyan")
        with self.console.status(
            f"[bold cyan]{agent.name} is working...[/bold cyan]",
            spinner="dots",
        ):
            response = agent.safe_run(user_prompt, context=context)
        return (
            response,
            agent.name,
            list(agent.last_used_tools),
            agent.copy_session_state() if isinstance(agent, BaseAgent) else {},
        )

    def _routing_context(self, memory_context: str) -> str:
        """Combine memory context with short-lived session artifact state."""
        last_response = str(self.session_state.get("last_response", "") or "").strip()
        last_artifact_content = str(
            self.session_state.get("last_artifact_content", "") or ""
        ).strip()
        last_artifact_responder = str(
            self.session_state.get("last_artifact_responder", "") or ""
        ).strip()
        last_responder = str(self.session_state.get("last_responder", "") or "").strip()
        last_tools_used = self.session_state.get("last_tools_used", [])
        session_lines = ["Session Artifact State:"]
        if last_response:
            excerpt = self._bounded_context(last_response, limit=500)
            session_lines.extend(
                [
                    f"Last responder: {last_responder or 'unknown'}",
                    f"Active artifact owner: {last_artifact_responder or 'unknown'}",
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
            self.session_state = BaseAgent.empty_session_state()
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
