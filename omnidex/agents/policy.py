"""Shared policy proposal validation for agent routing and handoffs."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from omnidex.agents.base import BaseAgent, SessionState
from omnidex.agents.handoffs import HandoffDecision

if TYPE_CHECKING:
    from omnidex.agents.research_assistant.agent import ResearchAssistant


@dataclass(slots=True)
class PolicyValidationResult:
    """Normalized result for a validated route or handoff policy decision."""

    accepted: bool
    target_agent: str
    source: str
    reason: str = ""


class AgentPolicyValidator:
    """Validate model-proposed routing and handoffs against hard constraints."""

    def __init__(self, *, agents: dict[str, BaseAgent]) -> None:
        self.agents = agents

    def validate_initial_route(
        self,
        *,
        proposed_route: str,
        query: str,
        session_state: SessionState,
    ) -> PolicyValidationResult:
        """Validate the initial model-proposed route."""
        constrained_route, reason = self._constrained_route(
            query=query,
            session_state=session_state,
        )
        if constrained_route:
            return PolicyValidationResult(
                accepted=True,
                target_agent=constrained_route,
                source=(
                    "policy_override" if constrained_route != proposed_route else "policy_validated"
                ),
                reason=reason,
            )
        if proposed_route in self.agents:
            return PolicyValidationResult(
                accepted=True,
                target_agent=proposed_route,
                source="policy_validated",
            )
        return PolicyValidationResult(
            accepted=True,
            target_agent="chat_agent",
            source="default_fallback",
            reason="invalid proposed route",
        )

    def validate_handoff(
        self,
        *,
        current_agent: str,
        decision: HandoffDecision | None,
        query: str,
        session_state: SessionState,
    ) -> PolicyValidationResult:
        """Validate a model-proposed handoff decision."""
        if decision is None:
            return PolicyValidationResult(
                accepted=False,
                target_agent=current_agent,
                source="no_handoff",
            )
        if decision.target_agent not in self.agents or decision.target_agent == current_agent:
            return PolicyValidationResult(
                accepted=False,
                target_agent=current_agent,
                source="handoff_rejected",
                reason="invalid target",
            )

        constrained_route, reason = self._constrained_route(
            query=query,
            session_state=session_state,
        )
        if constrained_route and constrained_route != decision.target_agent:
            return PolicyValidationResult(
                accepted=False,
                target_agent=current_agent,
                source="handoff_rejected",
                reason=reason,
            )

        return PolicyValidationResult(
            accepted=True,
            target_agent=decision.target_agent,
            source="handoff_validated",
            reason=decision.reason,
        )

    def _constrained_route(
        self,
        *,
        query: str,
        session_state: SessionState,
    ) -> tuple[str | None, str]:
        """Return a hard-constrained route when the turn must stay with research."""
        research_agent = self.agents.get("research_assistant")
        if research_agent is None:
            return None, ""
        research_agent.apply_session_state(session_state)
        return self._research_constraints(research_agent, query)

    def _research_constraints(
        self,
        research_agent: BaseAgent,
        query: str,
    ) -> tuple[str | None, str]:
        """Enforce direct workflow ownership for the research assistant."""
        classify_direct_pdf_intent = getattr(research_agent, "_classify_direct_pdf_intent", None)
        if callable(classify_direct_pdf_intent) and classify_direct_pdf_intent(query) is not None:
            return "research_assistant", "explicit PDF workflow must stay with research_assistant"

        should_keep_direct_save_followup = getattr(
            research_agent,
            "_should_keep_direct_save_followup",
            None,
        )
        if callable(should_keep_direct_save_followup) and should_keep_direct_save_followup(query):
            return "research_assistant", "explicit save/export follow-up must stay with research_assistant"

        return None, ""


__all__ = ["AgentPolicyValidator", "PolicyValidationResult"]
