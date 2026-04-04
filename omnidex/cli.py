"""Command-line interface for OmniDex."""

from __future__ import annotations

import argparse

from dotenv import load_dotenv
from rich.console import Console

from omnidex.agents.orchestrator import OrchestratorAgent
from omnidex.agents.orchestrator.agent import DEFAULT_SYSTEM_PROMPT


def build_parser() -> argparse.ArgumentParser:
    """Create the OmniDex CLI parser."""
    parser = argparse.ArgumentParser(
        prog="python -m omnidex",
        description="Run the local OmniDex orchestrator delegator.",
    )
    parser.add_argument(
        "--prompt",
        help="Run one prompt non-interactively and print the response.",
    )
    parser.add_argument(
        "--system-prompt",
        help="Override the default system prompt for this session.",
    )
    parser.add_argument(
        "--no-stream",
        action="store_true",
        help="Disable token streaming for this invocation.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run the OmniDex CLI."""
    load_dotenv()
    args = build_parser().parse_args(argv)

    console = Console()
    try:
        agent = OrchestratorAgent(
            console=console,
            system_prompt=args.system_prompt or DEFAULT_SYSTEM_PROMPT,
        )
    except (FileNotFoundError, ValueError) as exc:
        console.print(f"[red]Startup failed:[/red] {exc}")
        return 1

    if args.prompt:
        agent.ask(args.prompt, stream=False if args.no_stream else None)
        return 0

    return agent.run()
