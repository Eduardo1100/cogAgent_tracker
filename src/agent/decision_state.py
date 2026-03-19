from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


def _drop_empty(value: Any) -> Any:
    if isinstance(value, dict):
        cleaned = {
            key: _drop_empty(item)
            for key, item in value.items()
            if _drop_empty(item) not in ({}, [], "", None, False)
        }
        return cleaned
    if isinstance(value, list):
        return [
            _drop_empty(item)
            for item in value
            if _drop_empty(item) not in ({}, [], "", None, False)
        ]
    return value


@dataclass(frozen=True)
class ActionSurfaceState:
    total_actions: int | None = None
    current_phase: str | None = None
    salient_entities: list[str] = field(default_factory=list)
    shortlist: list[str] = field(default_factory=list)
    deprioritized_families: list[str] = field(default_factory=list)
    required_families: list[str] = field(default_factory=list)
    interaction_opportunity_count: int = 0


@dataclass(frozen=True)
class GoalState:
    task_status: str | None = None
    current_phase: str | None = None
    task_contract: dict[str, Any] = field(default_factory=dict)
    ordered_target_progress: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class GroundingState:
    candidate_tracking: dict[str, Any] = field(default_factory=dict)
    relation_frontier: dict[str, Any] = field(default_factory=dict)
    substance_search: dict[str, Any] = field(default_factory=dict)
    artifact_creation: dict[str, Any] = field(default_factory=dict)
    measurement_tracking: dict[str, Any] = field(default_factory=dict)
    comparison_tracking: dict[str, Any] = field(default_factory=dict)
    conditional_branch_tracking: dict[str, Any] = field(default_factory=dict)
    remote_room_signal: dict[str, Any] = field(default_factory=dict)
    referent_resolution: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class OptionState:
    current_option: dict[str, Any] = field(default_factory=dict)
    pending_sequence: list[str] = field(default_factory=list)
    persistent_admissible_actions: list[str] = field(default_factory=list)
    recently_executed_actions: list[str] = field(default_factory=list)
    recently_failed_actions: list[str] = field(default_factory=list)
    last_interrupt: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ProgressState:
    actions_taken: int = 0
    actions_since_last_thinking: int = 0
    last_burst_size: int = 1
    last_burst_stop_reason: str = ""
    affordance_delta_count: int = 0
    task_relevant_affordance_delta_count: int = 0
    interaction_opportunity_delta: int = 0
    option_step: int = 0
    option_step_budget: int = 0
    option_stagnation_steps: int = 0
    option_progress_events: int = 0
    option_progress_debt: int = 0
    option_progress_debt_limit: int = 0
    option_revisitation_count: int = 0
    option_frontier_expansion_events: int = 0
    option_novelty_rate: float | None = None
    option_family_value: int = 0
    option_outcome_events: int = 0


@dataclass(frozen=True)
class UncertaintyState:
    admissible_actions_unchanged: bool | None = None
    referent_resolution: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class DecisionState:
    action_surface: ActionSurfaceState
    goal_state: GoalState
    grounding_state: GroundingState
    option_state: OptionState
    progress_state: ProgressState
    uncertainty_state: UncertaintyState

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def to_compact_dict(self) -> dict[str, Any]:
        return _drop_empty(self.to_dict())


def build_action_runtime_snapshot(state: DecisionState) -> dict[str, Any]:
    grounding = state.grounding_state
    option_state = state.option_state
    progress = state.progress_state
    uncertainty = state.uncertainty_state
    snapshots = {
        "ordered_target_progress": state.goal_state.ordered_target_progress,
        "candidate_tracking": grounding.candidate_tracking,
        "relation_frontier": grounding.relation_frontier,
        "substance_search": grounding.substance_search,
        "artifact_creation": grounding.artifact_creation,
        "measurement_tracking": grounding.measurement_tracking,
        "comparison_tracking": grounding.comparison_tracking,
        "conditional_branch_tracking": grounding.conditional_branch_tracking,
        "remote_room_signal": grounding.remote_room_signal,
        "referent_resolution": grounding.referent_resolution,
        "current_option": option_state.current_option,
        "recently_failed_actions": option_state.recently_failed_actions,
        "recently_executed_actions": option_state.recently_executed_actions,
        "pending_sequence": option_state.pending_sequence,
        "persistent_admissible_actions": option_state.persistent_admissible_actions,
        "last_option_interrupt_decision": option_state.last_interrupt,
        "task_relevant_affordance_delta_count": progress.task_relevant_affordance_delta_count,
        "option_step": progress.option_step,
        "option_step_budget": progress.option_step_budget,
        "option_stagnation_steps": progress.option_stagnation_steps,
        "option_progress_events": progress.option_progress_events,
        "option_progress_debt": progress.option_progress_debt,
        "option_progress_debt_limit": progress.option_progress_debt_limit,
        "option_revisitation_count": progress.option_revisitation_count,
        "option_frontier_expansion_events": progress.option_frontier_expansion_events,
        "option_novelty_rate": progress.option_novelty_rate,
        "option_family_value": progress.option_family_value,
        "option_outcome_events": progress.option_outcome_events,
        "admissible_actions_unchanged": (
            True if uncertainty.admissible_actions_unchanged else None
        ),
    }
    return _drop_empty(snapshots)


def build_analyst_runtime_snapshot(state: DecisionState) -> dict[str, Any]:
    grounding = state.grounding_state
    progress = state.progress_state
    snapshots = {
        "ordered_progress": {
            "focused": state.goal_state.ordered_target_progress.get(
                "focused_stage_labels", []
            )[:3],
            "pending": state.goal_state.ordered_target_progress.get(
                "pending_stage_candidates", []
            )[:3],
        },
        "candidate": {
            "active": grounding.candidate_tracking.get("active_candidate"),
            "last_seen_room": grounding.candidate_tracking.get("last_seen_room"),
            "rejected": grounding.candidate_tracking.get("rejected_candidates", [])[:2],
        },
        "measurement": {
            "target": grounding.measurement_tracking.get("measurement_target"),
            "property": grounding.measurement_tracking.get("measurement_property"),
            "resolved": grounding.measurement_tracking.get("property_resolved"),
            "branch_ready": grounding.measurement_tracking.get("branch_ready"),
        },
        "comparison": {
            "targets": grounding.comparison_tracking.get("comparison_targets", [])[:2],
            "resolved_target": grounding.comparison_tracking.get("selected_target"),
        },
        "conditional_branch": {
            "evidence_target": grounding.conditional_branch_tracking.get(
                "evidence_target"
            ),
            "resolved_target": grounding.conditional_branch_tracking.get(
                "selected_branch"
            ),
        },
        "relation_frontier": {
            "referents": grounding.relation_frontier.get("frontier_referents", [])[:4],
            "control_candidates": grounding.relation_frontier.get(
                "control_candidates", []
            )[:2],
        },
        "remote_room": {
            "room": grounding.remote_room_signal.get("room"),
            "reason": grounding.remote_room_signal.get("reason"),
        },
        "substance_search": {
            "phase": grounding.substance_search.get("phase"),
            "grounded_substances": grounding.substance_search.get(
                "grounded_substances", []
            )[:3],
            "source_candidates": grounding.substance_search.get(
                "source_candidates", []
            )[:3],
        },
        "artifact_creation": {
            "artifact_type": grounding.artifact_creation.get("artifact_type"),
            "grounded_artifacts": grounding.artifact_creation.get(
                "grounded_artifacts", []
            )[:3],
        },
        "current_option": {
            "objective": state.option_state.current_option.get("objective"),
            "primary_family": state.option_state.current_option.get("primary_family"),
            "option_mode": state.option_state.current_option.get("option_mode"),
            "option_family": state.option_state.current_option.get("option_family"),
            "target_signature": state.option_state.current_option.get(
                "target_signature"
            ),
            "expected_progress_signals": state.option_state.current_option.get(
                "expected_progress_signals", []
            )[:4],
            "expected_outcomes": state.option_state.current_option.get(
                "expected_outcomes", []
            )[:4],
            "realized_outcomes": state.option_state.current_option.get(
                "realized_outcomes", {}
            ),
            "step_budget": progress.option_step_budget,
            "step": progress.option_step,
            "stagnation_steps": progress.option_stagnation_steps,
            "progress_events": progress.option_progress_events,
            "progress_debt": progress.option_progress_debt,
            "progress_debt_limit": progress.option_progress_debt_limit,
            "family_value": progress.option_family_value,
            "outcome_events": progress.option_outcome_events,
        },
        "progress": {
            "affordance_delta_count": progress.affordance_delta_count,
            "task_relevant_affordance_delta_count": (
                progress.task_relevant_affordance_delta_count
            ),
            "interaction_opportunity_delta": progress.interaction_opportunity_delta,
            "option_revisitation_count": progress.option_revisitation_count,
            "option_frontier_expansion_events": (
                progress.option_frontier_expansion_events
            ),
            "option_novelty_rate": progress.option_novelty_rate,
            "option_family_value": progress.option_family_value,
            "option_outcome_events": progress.option_outcome_events,
        },
        "last_option_interrupt_decision": state.option_state.last_interrupt,
        "recently_executed_actions": state.option_state.recently_executed_actions,
    }
    return _drop_empty(snapshots)
