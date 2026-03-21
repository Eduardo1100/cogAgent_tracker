from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol
from uuid import uuid4

from src.agent.integrations.openclaw.protocol import (
    OpenClawDecisionResponse,
    OpenClawDeliberationJob,
    OpenClawDeliberationRequest,
)
from src.agent.runtime import RuntimeDecision, RuntimeSession


def _protocol_hint_metadata(session: RuntimeSession) -> dict[str, object]:
    snapshot = session.snapshot()
    protocol_summary = dict(snapshot.protocol_summary)
    latest_decision = session.latest_decision()
    return {
        "protocol_hint": {
            "recommended_mode": protocol_summary.get("recommended_mode", "protocol"),
            "recommended_mode_reason": protocol_summary.get(
                "recommended_mode_reason", "stable_protocol_state"
            ),
            "protocol_suggested_action": latest_decision.suggested_action,
            "protocol_repair_operator": latest_decision.repair_operator,
            "approaching_escalation": bool(
                protocol_summary.get("approaching_escalation", False)
            ),
            "repair_pending": bool(protocol_summary.get("repair_pending", False)),
            "contradiction_debt": float(
                protocol_summary.get("contradiction_debt", 0.0)
            ),
        }
    }


def _disagreement_summary(
    protocol_hint_metadata: dict[str, object],
    decision: RuntimeDecision | None,
) -> dict[str, object]:
    protocol_hint = dict(protocol_hint_metadata.get("protocol_hint", {}))
    if decision is None:
        return {
            "available": False,
            "reason": "decision_unavailable",
        }
    protocol_mode = protocol_hint.get("recommended_mode")
    protocol_action = protocol_hint.get("protocol_suggested_action")
    protocol_repair_operator = protocol_hint.get("protocol_repair_operator")
    mode_diverged = protocol_mode != decision.mode_used
    action_diverged = protocol_action != decision.suggested_action
    repair_operator_diverged = protocol_repair_operator != decision.repair_operator
    if action_diverged or repair_operator_diverged:
        backend_alignment = "strategy_diverged"
        if action_diverged and repair_operator_diverged:
            backend_alignment_reason = "action_and_repair_changed_for_recovery"
        elif action_diverged:
            backend_alignment_reason = "action_changed_for_recovery"
        else:
            backend_alignment_reason = "repair_operator_changed_for_recovery"
    elif mode_diverged:
        backend_alignment = "mode_only_diverged"
        backend_alignment_reason = "mode_changed_for_deliberation"
    else:
        backend_alignment = "aligned"
        backend_alignment_reason = "protocol_and_backend_aligned"
    return {
        "available": True,
        "backend_alignment": backend_alignment,
        "backend_alignment_reason": backend_alignment_reason,
        "mode_diverged": mode_diverged,
        "action_diverged": action_diverged,
        "repair_operator_diverged": repair_operator_diverged,
        "any_diverged": any([mode_diverged, action_diverged, repair_operator_diverged]),
        "protocol_mode": protocol_mode,
        "backend_mode": decision.mode_used,
        "protocol_suggested_action": protocol_action,
        "backend_suggested_action": decision.suggested_action,
        "protocol_repair_operator": protocol_repair_operator,
        "backend_repair_operator": decision.repair_operator,
    }


class DeliberativeBackend(Protocol):
    backend_name: str

    def submit(
        self,
        *,
        session: RuntimeSession,
        request: OpenClawDeliberationRequest,
    ) -> OpenClawDeliberationJob: ...

    def get_job(self, job_id: str) -> OpenClawDeliberationJob | None: ...

    def clear(self) -> None: ...


@dataclass
class InMemoryDeliberativeBackend:
    jobs: dict[str, OpenClawDeliberationJob] = field(default_factory=dict)
    backend_name: str = "in_memory_stub"

    def submit(
        self,
        *,
        session: RuntimeSession,
        request: OpenClawDeliberationRequest,
    ) -> OpenClawDeliberationJob:
        job_id = uuid4().hex
        protocol_hint = _protocol_hint_metadata(session)
        decision = session.escalate(
            reason=request.reason or "deliberative_job_requested"
        )
        disagreement_summary = _disagreement_summary(protocol_hint, decision)
        session.record_comparison_event(
            source="async_deliberation_job",
            protocol_hint=protocol_hint,
            disagreement_summary=disagreement_summary,
            decision=decision,
            backend=self.backend_name,
            metadata={"job_id": job_id, "status": "completed"},
        )
        job = OpenClawDeliberationJob.completed(
            job_id=job_id,
            session_id=request.session_id,
            decision=OpenClawDecisionResponse.from_runtime_decision(
                request.session_id, decision
            ),
            backend=self.backend_name,
            metadata={
                **dict(request.metadata),
                **protocol_hint,
                "disagreement_summary": disagreement_summary,
                "stub": True,
            },
        )
        self.jobs[job_id] = job
        return job

    def get_job(self, job_id: str) -> OpenClawDeliberationJob | None:
        return self.jobs.get(job_id)

    def clear(self) -> None:
        self.jobs.clear()
