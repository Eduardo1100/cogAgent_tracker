from __future__ import annotations

from dataclasses import dataclass, field

from fastapi import HTTPException

from src.agent.integrations.openclaw.adapter import OpenClawAdapter
from src.agent.integrations.openclaw.deliberation import (
    InMemoryDeliberativeBackend,
    _disagreement_summary,
    _protocol_hint_metadata,
)
from src.agent.integrations.openclaw.gwt_backend import GWTDeliberativeBackend
from src.agent.integrations.openclaw.protocol import (
    OpenClawActionApplyRequest,
    OpenClawDecisionResponse,
    OpenClawDeliberationJob,
    OpenClawDeliberationRequest,
    OpenClawObservationRequest,
    OpenClawSessionCreateRequest,
    OpenClawSessionSnapshot,
)
from src.agent.runtime import RuntimeDriver, RuntimeSession


@dataclass
class OpenClawRuntimeService:
    driver: RuntimeDriver = field(default_factory=RuntimeDriver)
    deliberative_backends: dict[str, object] = field(
        default_factory=lambda: {
            "in_memory_stub": InMemoryDeliberativeBackend(),
            "gwt": GWTDeliberativeBackend(),
        }
    )
    deliberation_job_index: dict[str, str] = field(default_factory=dict)

    def _get_session(self, session_id: str) -> RuntimeSession:
        try:
            return self.driver.get_session(session_id)
        except KeyError as exc:
            raise HTTPException(
                status_code=404, detail="Runtime session not found"
            ) from exc

    def _get_adapter(self, session: RuntimeSession) -> OpenClawAdapter:
        adapter = session.adapter
        if not isinstance(adapter, OpenClawAdapter):
            raise HTTPException(
                status_code=409,
                detail="Runtime session is not bound to an OpenClaw adapter",
            )
        return adapter

    def _build_decision_response(
        self,
        *,
        session: RuntimeSession,
        session_id: str,
        decision,
        protocol_hint: dict[str, object] | None = None,
        telemetry_source: str = "sync_decision",
    ) -> OpenClawDecisionResponse:
        if decision.escalated and decision.mode_used == "deliberative":
            effective_protocol_hint = protocol_hint or {}
            disagreement_summary = _disagreement_summary(
                effective_protocol_hint, decision
            )
            session.record_comparison_event(
                source=telemetry_source,
                protocol_hint=effective_protocol_hint,
                disagreement_summary=disagreement_summary,
                decision=decision,
                metadata={"session_id": session_id},
            )
            return OpenClawDecisionResponse(
                session_id=session_id,
                suggested_action=decision.suggested_action,
                option_family=decision.option_family,
                repair_operator=decision.repair_operator,
                planner_stop_reason=decision.planner_stop_reason,
                revision_required=decision.revision_required,
                mode_used=decision.mode_used,
                escalated=decision.escalated,
                escalation_reason=decision.escalation_reason,
                metadata={
                    **dict(decision.metadata),
                    **effective_protocol_hint,
                    "disagreement_summary": disagreement_summary,
                },
            )
        return OpenClawDecisionResponse.from_runtime_decision(session_id, decision)

    def create_session(
        self, request: OpenClawSessionCreateRequest
    ) -> OpenClawSessionSnapshot:
        session = self.driver.create_session()
        adapter = OpenClawAdapter(
            task_text=request.task_text,
            capabilities=request.capabilities_record(),
            initial_event=request.initial_adapter_event(),
            session_metadata=request.session_metadata,
        )
        session.bind_adapter(
            adapter,
            capabilities=adapter.get_v2_capabilities(),
            info=request.session_metadata,
            mode=request.mode,
            escalation_policy=request.escalation_policy,
        )
        initial_event = request.initial_adapter_event()
        if initial_event is not None:
            session.observe_adapter_event(
                initial_event,
                capabilities=adapter.get_v2_capabilities(),
            )
        return OpenClawSessionSnapshot.from_runtime_snapshot(
            session.snapshot(),
            latest_decision=session.latest_decision(),
        )

    def observe(self, request: OpenClawObservationRequest) -> OpenClawSessionSnapshot:
        session = self._get_session(request.session_id)
        adapter = self._get_adapter(session)
        event = adapter.ingest_observation(request)
        session.observe_adapter_event(
            event,
            capabilities=adapter.get_v2_capabilities(),
        )
        return OpenClawSessionSnapshot.from_runtime_snapshot(
            session.snapshot(),
            latest_decision=session.latest_decision(),
        )

    def decide(
        self, session_id: str, *, mode_override: str | None = None
    ) -> OpenClawDecisionResponse:
        session = self._get_session(session_id)
        protocol_hint = _protocol_hint_metadata(session)
        decision = session.decide(mode_override=mode_override)
        return self._build_decision_response(
            session=session,
            session_id=session_id,
            decision=decision,
            protocol_hint=protocol_hint,
            telemetry_source="sync_decide",
        )

    def escalate(
        self, session_id: str, *, reason: str | None = None
    ) -> OpenClawDecisionResponse:
        session = self._get_session(session_id)
        protocol_hint = _protocol_hint_metadata(session)
        decision = session.escalate(reason=reason)
        return self._build_decision_response(
            session=session,
            session_id=session_id,
            decision=decision,
            protocol_hint=protocol_hint,
            telemetry_source="sync_escalate",
        )

    def deliberate(
        self, request: OpenClawDeliberationRequest
    ) -> OpenClawDeliberationJob:
        session = self._get_session(request.session_id)
        backend_name = request.backend or "in_memory_stub"
        backend = self.deliberative_backends.get(backend_name)
        if backend is None:
            raise HTTPException(
                status_code=404, detail="Deliberative backend not found"
            )
        job = backend.submit(session=session, request=request)
        self.deliberation_job_index[job.job_id] = backend_name
        return job

    def get_deliberation(self, job_id: str) -> OpenClawDeliberationJob:
        backend_name = self.deliberation_job_index.get(job_id)
        if backend_name is None:
            raise HTTPException(status_code=404, detail="Deliberation job not found")
        backend = self.deliberative_backends.get(backend_name)
        job = None if backend is None else backend.get_job(job_id)
        if job is None:
            raise HTTPException(status_code=404, detail="Deliberation job not found")
        return job

    def apply(self, request: OpenClawActionApplyRequest) -> OpenClawSessionSnapshot:
        session = self._get_session(request.session_id)
        adapter = self._get_adapter(session)
        adapter.apply_action_result(
            action_text=request.action_text,
            action_executed=request.action_executed,
            reward_delta=request.reward_delta,
            status_delta=request.status_delta,
            metadata=request.metadata,
        )
        return OpenClawSessionSnapshot.from_runtime_snapshot(
            session.snapshot(),
            latest_decision=session.latest_decision(),
        )

    def get_snapshot(self, session_id: str) -> OpenClawSessionSnapshot:
        session = self._get_session(session_id)
        return OpenClawSessionSnapshot.from_runtime_snapshot(
            session.snapshot(),
            latest_decision=session.latest_decision(),
        )

    def export_comparison_telemetry(
        self, *, session_ids: list[str] | None = None
    ) -> dict[str, object]:
        requested_ids = [session_id for session_id in (session_ids or []) if session_id]
        sessions = (
            [self._get_session(session_id) for session_id in requested_ids]
            if requested_ids
            else self.driver.list_sessions()
        )
        alignment_counts: dict[str, int] = {}
        source_counts: dict[str, int] = {}
        backend_counts: dict[str, int] = {}
        alignment_reason_counts: dict[str, int] = {}
        session_entries: list[dict[str, object]] = []
        event_count = 0
        available_count = 0
        for session in sessions:
            snapshot = session.snapshot()
            comparison_telemetry = dict(snapshot.comparison_telemetry)
            event_count += int(comparison_telemetry.get("event_count", 0))
            available_count += int(comparison_telemetry.get("available_count", 0))

            session_alignment_counts: dict[str, int] = {}
            for alignment, count in dict(
                comparison_telemetry.get("alignment_counts", {})
            ).items():
                normalized_alignment = str(alignment)
                normalized_count = int(count)
                alignment_counts[normalized_alignment] = (
                    alignment_counts.get(normalized_alignment, 0) + normalized_count
                )
                session_alignment_counts[normalized_alignment] = normalized_count

            session_source_counts: dict[str, int] = {}
            for source, count in dict(
                comparison_telemetry.get("source_counts", {})
            ).items():
                normalized_source = str(source)
                normalized_count = int(count)
                source_counts[normalized_source] = (
                    source_counts.get(normalized_source, 0) + normalized_count
                )
                session_source_counts[normalized_source] = normalized_count

            recent_events = list(comparison_telemetry.get("recent_events", []))
            for event in session.comparison_events:
                backend = event.get("backend")
                if backend:
                    normalized_backend = str(backend)
                    backend_counts[normalized_backend] = (
                        backend_counts.get(normalized_backend, 0) + 1
                    )
                disagreement_summary = event.get("disagreement_summary", {})
                if isinstance(disagreement_summary, dict) and disagreement_summary.get(
                    "available"
                ):
                    reason = disagreement_summary.get(
                        "backend_alignment_reason", "reason_unknown"
                    )
                    normalized_reason = str(reason)
                    alignment_reason_counts[normalized_reason] = (
                        alignment_reason_counts.get(normalized_reason, 0) + 1
                    )

            session_entries.append(
                {
                    "session_id": session.session_id,
                    "task": snapshot.task,
                    "event_count": int(comparison_telemetry.get("event_count", 0)),
                    "available_count": int(
                        comparison_telemetry.get("available_count", 0)
                    ),
                    "alignment_counts": session_alignment_counts,
                    "source_counts": session_source_counts,
                    "latest_alignment": (
                        dict(recent_events[-1].get("disagreement_summary", {})).get(
                            "backend_alignment"
                        )
                        if recent_events
                        else None
                    ),
                    "latest_alignment_reason": (
                        dict(recent_events[-1].get("disagreement_summary", {})).get(
                            "backend_alignment_reason"
                        )
                        if recent_events
                        else None
                    ),
                }
            )

        session_entries.sort(key=lambda entry: str(entry["session_id"]))
        aligned_count = alignment_counts.get("aligned", 0)
        mode_only_diverged_count = alignment_counts.get("mode_only_diverged", 0)
        strategy_diverged_count = alignment_counts.get("strategy_diverged", 0)
        available_denominator = max(available_count, 1)
        top_alignment_reason = None
        top_strategy_reason = None
        if alignment_reason_counts:
            top_alignment_reason = max(
                alignment_reason_counts.items(),
                key=lambda item: (item[1], item[0]),
            )[0]
            strategy_reasons = {
                reason: count
                for reason, count in alignment_reason_counts.items()
                if reason.endswith("_for_recovery")
            }
            if strategy_reasons:
                top_strategy_reason = max(
                    strategy_reasons.items(),
                    key=lambda item: (item[1], item[0]),
                )[0]
        return {
            "session_count": len(sessions),
            "sessions_with_events": sum(
                1 for entry in session_entries if int(entry["event_count"]) > 0
            ),
            "sessions_with_available_comparisons": sum(
                1 for entry in session_entries if int(entry["available_count"]) > 0
            ),
            "event_count": event_count,
            "available_count": available_count,
            "alignment_counts": alignment_counts,
            "alignment_reason_counts": alignment_reason_counts,
            "source_counts": source_counts,
            "backend_counts": backend_counts,
            "summary_metrics": {
                "aligned_count": aligned_count,
                "mode_only_diverged_count": mode_only_diverged_count,
                "strategy_diverged_count": strategy_diverged_count,
                "aligned_rate": (
                    aligned_count / available_denominator if available_count else 0.0
                ),
                "mode_only_diverged_rate": (
                    mode_only_diverged_count / available_denominator
                    if available_count
                    else 0.0
                ),
                "strategy_diverged_rate": (
                    strategy_diverged_count / available_denominator
                    if available_count
                    else 0.0
                ),
                "top_alignment_reason": top_alignment_reason,
                "top_strategy_reason": top_strategy_reason,
            },
            "sessions": session_entries,
        }

    def delete_session(self, session_id: str) -> dict[str, str]:
        session = self.driver.close_session(session_id)
        if session is None:
            raise HTTPException(status_code=404, detail="Runtime session not found")
        return {"status": "deleted", "session_id": session_id}


runtime_service = OpenClawRuntimeService()
