"""Shared handoff contract for agent-to-agent delegation."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(slots=True)
class HandoffDecision:
    """Structured request for the orchestrator to hand work to another agent."""

    target_agent: str
    reason: str = ""
    confidence: float = 0.0


__all__ = ["HandoffDecision"]
