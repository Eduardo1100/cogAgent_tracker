from __future__ import annotations

import os
import tempfile
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from threading import Thread
from types import SimpleNamespace
from typing import Any, Protocol
from uuid import uuid4

from src.agent.integrations.openclaw.deliberation import (
    DeliberativeBackend,
    _disagreement_summary,
    _protocol_hint_metadata,
)
from src.agent.integrations.openclaw.protocol import (
    OpenClawDecisionResponse,
    OpenClawDeliberationJob,
    OpenClawDeliberationRequest,
)
from src.agent.runtime import RuntimeDecision, RuntimeSession, RuntimeSessionSnapshot
from src.agent.v2.types import OperatorCandidate, WorldModelSnapshot


class GWTDeliberationHarness(Protocol):
    def build_deliberation_recommendation(
        self,
        *,
        snapshot: RuntimeSessionSnapshot,
        world_snapshot: WorldModelSnapshot | None,
        latest_decision: RuntimeDecision | None,
        reason: str | None,
        request_metadata: dict[str, Any],
    ) -> RuntimeDecision: ...


def _job_metadata(
    request_metadata: dict[str, Any],
    protocol_hint: dict[str, object],
    decision: RuntimeDecision | None = None,
) -> dict[str, Any]:
    return {
        **dict(request_metadata),
        **protocol_hint,
        "disagreement_summary": _disagreement_summary(protocol_hint, decision),
    }


def _string_score(haystack: str, *needles: str) -> int:
    lowered = haystack.lower()
    return sum(1 for needle in needles if needle in lowered)


def _infer_family(action_label: str) -> str:
    lowered = action_label.lower()
    if lowered.startswith("type "):
        return "ui_type"
    if lowered.startswith("tap ") or lowered.startswith("click "):
        return "ui_click"
    if any(token in lowered for token in {"navigate back", "back", "dismiss", "close"}):
        return "ui_navigation"
    if any(token in lowered for token in {"save", "submit", "done", "confirm"}):
        return "ui_submit"
    if lowered in {"search", "look", "inspect"}:
        return "inspect"
    return "interaction"


def _input_state(snapshot: WorldModelSnapshot | None) -> dict[str, Any]:
    if snapshot is None:
        return {}
    input_state = snapshot.metadata.get("input_state", {})
    if not input_state:
        ui_context = snapshot.metadata.get("ui_context", {})
        if isinstance(ui_context, dict):
            input_state = ui_context.get("input_state", {})
    return dict(input_state) if isinstance(input_state, dict) else {}


def _ui_context(snapshot: WorldModelSnapshot | None) -> dict[str, Any]:
    if snapshot is None:
        return {}
    ui_context = snapshot.metadata.get("ui_context", {})
    return dict(ui_context) if isinstance(ui_context, dict) else {}


def _candidate_pool(
    snapshot: RuntimeSessionSnapshot,
    world_snapshot: WorldModelSnapshot | None,
) -> list[OperatorCandidate]:
    pooled: dict[str, OperatorCandidate] = {}
    if world_snapshot is not None:
        for candidate in world_snapshot.operator_candidates:
            if candidate.action_label:
                pooled[candidate.action_label] = candidate
    for index, action_label in enumerate(snapshot.admissible_actions):
        if action_label in pooled:
            continue
        pooled[action_label] = OperatorCandidate(
            operator_id=f"synthetic:{index}",
            family=_infer_family(action_label),
            action_label=action_label,
        )
    return list(pooled.values())


def _visible_element_map(
    ui_context: dict[str, Any],
) -> dict[str, dict[str, Any]]:
    visible_elements = ui_context.get("visible_elements", [])
    element_map: dict[str, dict[str, Any]] = {}
    if not isinstance(visible_elements, list):
        return element_map
    for raw_element in visible_elements:
        if not isinstance(raw_element, dict):
            continue
        element_id = raw_element.get("element_id")
        if element_id:
            element_map[str(element_id)] = dict(raw_element)
    return element_map


def _looks_like_escape(action_label: str, escape_actions: set[str]) -> bool:
    lowered = action_label.lower()
    return action_label in escape_actions or any(
        token in lowered for token in {"navigate back", "back", "dismiss", "close"}
    )


def _looks_like_submit(action_label: str) -> bool:
    return _string_score(action_label, "save", "submit", "done", "confirm") > 0


def _score_candidate(
    candidate: OperatorCandidate,
    *,
    latest_decision: RuntimeDecision | None,
    world_snapshot: WorldModelSnapshot | None,
) -> tuple[int, list[str]]:
    notes: list[str] = []
    score = 0
    input_state = _input_state(world_snapshot)
    ui_context = _ui_context(world_snapshot)
    element_map = _visible_element_map(ui_context)
    focused_element_id = ui_context.get("focused_element_id")
    active_input_target = input_state.get("active_input_target")
    text_entry_admissible = bool(input_state.get("text_entry_admissible"))
    submit_action_available = bool(input_state.get("submit_action_available"))
    input_detour = bool(input_state.get("input_detour"))
    escape_actions = {
        str(action) for action in input_state.get("escape_actions", []) if action
    }
    repair_operator = (
        world_snapshot.repair_directive.operator
        if world_snapshot is not None and world_snapshot.repair_directive is not None
        else latest_decision.repair_operator
        if latest_decision is not None
        else None
    )
    blocked_actions = (
        {
            str(action)
            for action in (
                world_snapshot.metadata.get("blocked_action_labels", []) or []
            )
            if action
        }
        if world_snapshot is not None
        else set()
    )
    invalidated_actions = (
        {
            str(action)
            for action in (
                world_snapshot.metadata.get("invalidated_action_labels", []) or []
            )
            if action
        }
        if world_snapshot is not None
        else set()
    )
    contradiction_actions = (
        {
            str(action)
            for action in (
                world_snapshot.metadata.get("recent_contradiction_actions", []) or []
            )
            if action
        }
        if world_snapshot is not None
        else set()
    )

    if latest_decision is not None and latest_decision.suggested_action:
        if latest_decision.suggested_action == candidate.action_label:
            score += 18
            notes.append("Preserve action continuity while the operator remains valid.")

    if candidate.action_label in blocked_actions:
        score -= 160
        notes.append("Avoid recently blocked actions.")
    if candidate.action_label in invalidated_actions:
        score -= 120
        notes.append("Avoid stale invalidated actions.")
    if candidate.action_label in contradiction_actions:
        score -= 90
        notes.append("Avoid actions tied to recent contradictions.")

    if candidate.action_label == "wait":
        score -= 35
        notes.append("Avoid no-op actions during repair.")

    if world_snapshot is not None and world_snapshot.active_option is not None:
        option = world_snapshot.active_option
        if option.family == "manipulate_target" and candidate.family.startswith("ui_"):
            score += 12
        if option.target_signature:
            if candidate.action_label == option.target_signature:
                score += 12
            if option.target_signature in candidate.target_ids:
                score += 18

    if input_detour:
        if _looks_like_escape(candidate.action_label, escape_actions):
            score += 70
            notes.append("Recover the task surface before further manipulation.")
        elif candidate.family.startswith("ui_"):
            score -= 25

    if repair_operator == "recover_task_surface":
        if _looks_like_escape(candidate.action_label, escape_actions):
            score += 45
        elif candidate.family == "inspect":
            score += 10

    if candidate.family == "ui_type":
        if not text_entry_admissible:
            score -= 170
            notes.append(
                "Typing is not admissible until focus/input state is repaired."
            )
        else:
            score += 55
            if active_input_target and active_input_target in candidate.target_ids:
                score += 20
            if (
                focused_element_id
                and focused_element_id in candidate.target_ids
                and focused_element_id == active_input_target
            ):
                score += 8
            notes.append(
                "Input state is writable; commit text before changing UI mode."
            )

    if candidate.family == "ui_submit":
        if submit_action_available or _looks_like_submit(candidate.action_label):
            score += 42
            notes.append("A commit/submit action is locally admissible.")
        else:
            score -= 12

    if candidate.family == "ui_click":
        editable_target = False
        if candidate.target_ids:
            for target_id in candidate.target_ids:
                element = element_map.get(target_id, {})
                attrs = (
                    element.get("attributes", {}) if isinstance(element, dict) else {}
                )
                role = str(element.get("role", "")).lower()
                if bool(attrs.get("editable")) or role in {"textbox", "input"}:
                    editable_target = True
                    break
        if repair_operator == "repair_input_focus" and editable_target:
            score += 55
            notes.append("Repair focus on the editable target before typing.")
        elif repair_operator == "repair_input_surface" and editable_target:
            score += 34
            notes.append("Re-open or stabilize the input surface via the active field.")
        elif editable_target and not text_entry_admissible:
            score += 30
        elif editable_target and text_entry_admissible:
            score -= 12
        if active_input_target and active_input_target in candidate.target_ids:
            score += 18
        if focused_element_id and focused_element_id in candidate.target_ids:
            score += 6

    if candidate.family == "ui_navigation" and _looks_like_escape(
        candidate.action_label, escape_actions
    ):
        score += 18

    if candidate.family == "inspect" and world_snapshot is not None:
        if world_snapshot.revision_required:
            score += 18
            notes.append("Inspection is a safe way to re-ground the local state.")

    return score, notes


def _recommendation_hint(
    candidate: OperatorCandidate | None,
    *,
    latest_decision: RuntimeDecision | None,
    world_snapshot: WorldModelSnapshot | None,
) -> str:
    repair_operator = (
        world_snapshot.repair_directive.operator
        if world_snapshot is not None and world_snapshot.repair_directive is not None
        else latest_decision.repair_operator
        if latest_decision is not None
        else None
    )
    input_state = _input_state(world_snapshot)
    input_detour = bool(input_state.get("input_detour"))
    if candidate is None:
        return "Re-ground the task surface and verify local operator preconditions."
    if input_detour or repair_operator == "recover_task_surface":
        return "The current UI is off the task surface; use a safe escape action before resuming task progress."
    if candidate.family == "ui_type":
        return "The active field is writable. Commit text now instead of changing focus again."
    if candidate.family == "ui_click":
        return (
            "Text entry is not stable yet. Re-focus the editable target before typing."
        )
    if candidate.family == "ui_submit":
        return "Local state appears ready enough to attempt a bounded submit/commit action."
    if repair_operator:
        return f"Repair operator `{repair_operator}` is active; prefer the lowest-risk admissible action that satisfies it."
    return (
        "Preserve local task continuity with the safest currently admissible operator."
    )


def _compact_world_summary(world_snapshot: WorldModelSnapshot | None) -> dict[str, Any]:
    if world_snapshot is None:
        return {}
    input_state = _input_state(world_snapshot)
    ui_context = _ui_context(world_snapshot)
    frontier_nodes = world_snapshot.frontier_summary.get("frontier_nodes", [])
    return {
        "active_option_family": (
            world_snapshot.active_option.family
            if world_snapshot.active_option is not None
            else None
        ),
        "active_option_target": (
            world_snapshot.active_option.target_signature
            if world_snapshot.active_option is not None
            else None
        ),
        "revision_required": world_snapshot.revision_required,
        "repair_operator": (
            world_snapshot.repair_directive.operator
            if world_snapshot.repair_directive is not None
            else None
        ),
        "entity_count": len(world_snapshot.entities),
        "operator_count": len(world_snapshot.operator_candidates),
        "uncertainty_count": len(world_snapshot.uncertainty),
        "contradiction_count": len(world_snapshot.contradictions),
        "frontier_count": len(frontier_nodes)
        if isinstance(frontier_nodes, list)
        else 0,
        "input_target": input_state.get("active_input_target"),
        "text_entry_admissible": bool(input_state.get("text_entry_admissible")),
        "input_detour": bool(input_state.get("input_detour")),
        "focused_element_id": ui_context.get("focused_element_id"),
        "page_title": ui_context.get("page_title"),
    }


def _recent_contradiction_history(
    world_snapshot: WorldModelSnapshot | None,
) -> list[dict[str, Any]]:
    if world_snapshot is None:
        return []
    history: list[dict[str, Any]] = []
    for contradiction in world_snapshot.contradictions[-3:]:
        history.append(
            {
                "category": contradiction.category,
                "subject": contradiction.subject,
                "action_text": contradiction.action_text,
                "evidence": contradiction.evidence,
            }
        )
    return history


def _world_summary_notes(world_summary: dict[str, Any]) -> list[str]:
    if not world_summary:
        return [
            "No world-model summary was available; fall back to admissibility cues."
        ]
    notes: list[str] = []
    option_family = world_summary.get("active_option_family")
    option_target = world_summary.get("active_option_target")
    if option_family:
        if option_target:
            notes.append(
                f"World summary: active option is `{option_family}` targeting `{option_target}`."
            )
        else:
            notes.append(f"World summary: active option is `{option_family}`.")
    if world_summary.get("repair_operator"):
        notes.append(
            f"World summary: repair operator `{world_summary['repair_operator']}` is active."
        )
    elif world_summary.get("revision_required"):
        notes.append(
            "World summary: revision is still required before open-ended action."
        )
    if world_summary.get("input_detour"):
        notes.append("World summary: the UI is currently in an input/system detour.")
    elif world_summary.get("text_entry_admissible"):
        notes.append(
            "World summary: the active input target is writable and suitable for bounded text entry."
        )
    return notes


def _contradiction_notes(contradictions: list[dict[str, Any]]) -> list[str]:
    if not contradictions:
        return []
    latest = contradictions[-1]
    category = latest.get("category") or "unknown"
    subject = latest.get("subject")
    action_text = latest.get("action_text")
    note = f"Recent contradiction: `{category}`"
    if subject:
        note += f" on `{subject}`"
    if action_text:
        note += f" after `{action_text}`"
    note += "."
    notes = [note]
    if len(contradictions) > 1:
        notes.append(
            f"Contradiction history: {len(contradictions)} recent records inform the current recovery choice."
        )
    return notes


@dataclass
class DefaultGWTDeliberationHarness:
    agent: Any

    def build_deliberation_recommendation(
        self,
        *,
        snapshot: RuntimeSessionSnapshot,
        world_snapshot: WorldModelSnapshot | None,
        latest_decision: RuntimeDecision | None,
        reason: str | None,
        request_metadata: dict[str, Any],
    ) -> RuntimeDecision:
        option_family = (
            world_snapshot.active_option.family
            if world_snapshot is not None and world_snapshot.active_option is not None
            else latest_decision.option_family
            if latest_decision is not None
            else None
        )
        repair_operator = (
            world_snapshot.repair_directive.operator
            if world_snapshot is not None
            and world_snapshot.repair_directive is not None
            else latest_decision.repair_operator
            if latest_decision is not None
            else None
        )
        candidate_pool = _candidate_pool(snapshot, world_snapshot)
        scored_candidates = sorted(
            (
                (
                    _score_candidate(
                        candidate,
                        latest_decision=latest_decision,
                        world_snapshot=world_snapshot,
                    ),
                    candidate,
                )
                for candidate in candidate_pool
            ),
            key=lambda item: (
                item[0][0],
                1 if item[1].family.startswith("ui_") else 0,
                len(item[1].target_ids),
                item[1].action_label,
            ),
            reverse=True,
        )
        selected_candidate = scored_candidates[0][1] if scored_candidates else None
        world_summary = _compact_world_summary(world_snapshot)
        contradiction_history = _recent_contradiction_history(world_snapshot)
        planner_notes: list[str] = []
        if reason:
            planner_notes.append(f"Escalation reason: {reason}.")
        planner_notes.extend(_world_summary_notes(world_summary))
        planner_notes.extend(_contradiction_notes(contradiction_history))
        if scored_candidates:
            top_score, top_notes = scored_candidates[0][0]
            if top_notes:
                planner_notes.extend(top_notes[:2])
            planner_notes.append(
                f"Selected `{selected_candidate.action_label}` as the lowest-risk admissible operator (score={top_score})."
            )
        recovery_hint = _recommendation_hint(
            selected_candidate,
            latest_decision=latest_decision,
            world_snapshot=world_snapshot,
        )
        if recovery_hint not in planner_notes:
            planner_notes.append(recovery_hint)
        return RuntimeDecision(
            suggested_action=(
                selected_candidate.action_label
                if selected_candidate is not None
                else None
            ),
            option_family=option_family,
            repair_operator=repair_operator,
            planner_stop_reason="gwt_deliberation_recommendation",
            revision_required=False,
            mode_used="deliberative",
            escalated=True,
            escalation_reason=reason,
            metadata={
                "backend": "gwt",
                "agent_class": type(self.agent).__name__,
                "semantic_recovery_recommendation": recovery_hint,
                "planner_notes": planner_notes[:6],
                "selected_action_family": (
                    selected_candidate.family
                    if selected_candidate is not None
                    else None
                ),
                "selected_target_ids": (
                    list(selected_candidate.target_ids)
                    if selected_candidate is not None
                    else []
                ),
                "world_model_available": world_snapshot is not None,
                "world_summary": world_summary,
                "recent_contradictions": contradiction_history,
                **dict(request_metadata),
            },
        )


def _default_gwt_harness_factory(job_id: str) -> GWTDeliberationHarness:
    from src.agent.gwt_agent import GWTAutogenAgent

    llm_profile = {
        "config_list": [
            {
                "model": os.getenv("OPENCLAW_GWT_MODEL", "gpt-4o-mini"),
                "api_key": os.getenv("OPENAI_API_KEY", "stub-key"),
            }
        ],
        "temperature": 0.0,
        "cache_seed": 42,
    }
    log_path = Path(tempfile.gettempdir()) / "openclaw-deliberation" / job_id
    agent = GWTAutogenAgent(
        llm_profile=llm_profile,
        log_path=str(log_path),
        max_chat_round=1,
        max_actions=1,
        rounds_per_game=1,
        args=SimpleNamespace(env_type="openclaw"),
    )
    return DefaultGWTDeliberationHarness(agent=agent)


@dataclass
class GWTDeliberativeBackend(DeliberativeBackend):
    jobs: dict[str, OpenClawDeliberationJob] = field(default_factory=dict)
    backend_name: str = "gwt"
    harness_factory: Callable[[str], GWTDeliberationHarness] = (
        _default_gwt_harness_factory
    )

    def submit(
        self,
        *,
        session: RuntimeSession,
        request: OpenClawDeliberationRequest,
    ) -> OpenClawDeliberationJob:
        job_id = uuid4().hex
        protocol_hint = _protocol_hint_metadata(session)
        job = OpenClawDeliberationJob(
            job_id=job_id,
            session_id=request.session_id,
            status="pending",
            mode="deliberative",
            backend=self.backend_name,
            metadata=_job_metadata(request.metadata, protocol_hint),
        )
        self.jobs[job_id] = job
        snapshot = session.snapshot()
        world_snapshot = (
            session.world_model.to_snapshot()
            if session.world_model is not None
            else None
        )
        latest_decision = session.latest_decision()
        worker = Thread(
            target=self._run_job,
            args=(
                job_id,
                snapshot,
                world_snapshot,
                latest_decision,
                request,
                protocol_hint,
                session,
            ),
            daemon=True,
        )
        worker.start()
        return job

    def _run_job(
        self,
        job_id: str,
        snapshot: RuntimeSessionSnapshot,
        world_snapshot: WorldModelSnapshot | None,
        latest_decision: RuntimeDecision | None,
        request: OpenClawDeliberationRequest,
        protocol_hint: dict[str, object],
        session: RuntimeSession,
    ) -> None:
        self.jobs[job_id] = OpenClawDeliberationJob(
            job_id=job_id,
            session_id=request.session_id,
            status="running",
            mode="deliberative",
            backend=self.backend_name,
            metadata=_job_metadata(request.metadata, protocol_hint),
        )
        try:
            harness = self.harness_factory(job_id)
            decision = harness.build_deliberation_recommendation(
                snapshot=snapshot,
                world_snapshot=world_snapshot,
                latest_decision=latest_decision,
                reason=request.reason,
                request_metadata=request.metadata,
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
            self.jobs[job_id] = OpenClawDeliberationJob.completed(
                job_id=job_id,
                session_id=request.session_id,
                decision=OpenClawDecisionResponse.from_runtime_decision(
                    request.session_id, decision
                ),
                backend=self.backend_name,
                metadata=_job_metadata(request.metadata, protocol_hint, decision),
            )
        except Exception as exc:
            disagreement_summary = _disagreement_summary(protocol_hint, None)
            session.record_comparison_event(
                source="async_deliberation_job",
                protocol_hint=protocol_hint,
                disagreement_summary=disagreement_summary,
                decision=None,
                backend=self.backend_name,
                metadata={"job_id": job_id, "status": "failed", "error": str(exc)},
            )
            self.jobs[job_id] = OpenClawDeliberationJob(
                job_id=job_id,
                session_id=request.session_id,
                status="failed",
                mode="deliberative",
                backend=self.backend_name,
                error=str(exc),
                metadata=_job_metadata(request.metadata, protocol_hint),
            )

    def get_job(self, job_id: str) -> OpenClawDeliberationJob | None:
        return self.jobs.get(job_id)

    def clear(self) -> None:
        self.jobs.clear()
