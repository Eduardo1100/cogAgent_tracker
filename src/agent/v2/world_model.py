from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

from src.agent.decision_state import DecisionState
from src.agent.v2.types import (
    ActionOutcomeRecord,
    ActionResult,
    AdapterCapabilities,
    AdapterEvent,
    ContradictionRecord,
    GoalRecord,
    MemoryCue,
    OperatorCandidate,
    OptionContract,
    RelationRecord,
    RepairDirective,
    UIContext,
    UncertaintyRecord,
    WorldModelSnapshot,
)

_OBS_BLOCKED_RE = re.compile(
    r"(?:it's a wall|you can't|you cannot|blocked|nothing happens|no effect)",
    re.IGNORECASE,
)
_UI_BINDING_FAILURE_RE = re.compile(
    r"(?:not found|not visible|not interactable|stale|detached|unknown element|missing element|no such element|disabled)",
    re.IGNORECASE,
)
_NON_PROGRESS_STATUS_KEYS = {"T", "time", "turn", "turns"}


def _extract_direction_from_action(action_text: str | None) -> str | None:
    if not action_text:
        return None
    normalized = action_text.strip().lower()
    if normalized.startswith("move "):
        return normalized.removeprefix("move ").strip() or None
    if normalized == "go up":
        return "up"
    if normalized == "go down":
        return "down"
    return None


def _extract_ui_target_from_action(action_text: str | None) -> str | None:
    if not action_text:
        return None
    normalized = action_text.strip().lower()
    for prefix in (
        "click ",
        "tap ",
        "press ",
        "select ",
        "choose ",
        "type ",
        "fill ",
        "enter ",
        "focus ",
    ):
        if normalized.startswith(prefix):
            target = normalized.removeprefix(prefix).strip()
            return target or None
    return None


def _ui_target_visible(target: str | None, ui_context: UIContext | None) -> bool:
    if not target or ui_context is None:
        return False
    normalized_target = target.strip().lower()
    if not normalized_target:
        return False
    for element in ui_context.visible_elements:
        for candidate in (
            element.element_id,
            element.name,
            element.text,
            element.selector,
        ):
            if candidate and normalized_target in str(candidate).lower():
                return True
    return False


def _partition_status_delta(
    status_delta: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    beneficial: dict[str, Any] = {}
    harmful: dict[str, Any] = {}
    neutral: dict[str, Any] = {}
    for key, value in status_delta.items():
        if key in _NON_PROGRESS_STATUS_KEYS:
            continue
        if value in (None, "", 0, 0.0):
            continue
        if isinstance(value, bool):
            if value:
                beneficial[key] = value
            continue
        if isinstance(value, (int, float)):
            if value > 0:
                beneficial[key] = value
            elif value < 0:
                harmful[key] = value
            continue
        neutral[key] = value
    return beneficial, harmful, neutral


@dataclass(frozen=True)
class WorldEntity:
    entity_id: str
    label: str | None = None
    entity_type: str | None = None
    aliases: list[str] = field(default_factory=list)
    properties: dict[str, Any] = field(default_factory=dict)
    last_seen_step: int = 0
    confidence: float | None = None
    source_channels: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        payload = {
            "entity_id": self.entity_id,
            "label": self.label,
            "entity_type": self.entity_type,
            "aliases": self.aliases,
            "properties": self.properties,
            "last_seen_step": self.last_seen_step,
            "confidence": self.confidence,
            "source_channels": self.source_channels,
        }
        return {
            key: value
            for key, value in payload.items()
            if value not in (None, [], {}, "")
        }


@dataclass(frozen=True)
class FrontierState:
    current_region: str | None = None
    current_node_id: str | None = None
    visible_nodes: list[str] = field(default_factory=list)
    frontier_nodes: list[str] = field(default_factory=list)
    blocked_nodes: list[str] = field(default_factory=list)
    opened_nodes: list[str] = field(default_factory=list)
    closed_nodes: list[str] = field(default_factory=list)
    revisited_nodes: list[str] = field(default_factory=list)
    visit_counts: dict[str, int] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        payload = {
            "current_region": self.current_region,
            "current_node_id": self.current_node_id,
            "visible_nodes": self.visible_nodes,
            "frontier_nodes": self.frontier_nodes,
            "blocked_nodes": self.blocked_nodes,
            "opened_nodes": self.opened_nodes,
            "closed_nodes": self.closed_nodes,
            "revisited_nodes": self.revisited_nodes,
            "visit_counts": self.visit_counts,
        }
        return {
            key: value
            for key, value in payload.items()
            if value not in (None, [], {}, "")
        }


def _merge_unique(base: list[str], additions: list[str]) -> list[str]:
    seen = dict.fromkeys(base)
    for item in additions:
        if item:
            seen.setdefault(item, None)
    return list(seen.keys())


def _derive_unresolved_referents(
    referent_resolution: dict[str, Any],
    entities: dict[str, WorldEntity],
) -> list[str]:
    unresolved: list[str] = []
    if referent_resolution:
        requested = referent_resolution.get("requested_target")
        resolved = referent_resolution.get("resolved_target")
        if requested and requested != resolved:
            unresolved.append(str(requested))
    for entity in entities.values():
        label = (entity.label or "").lower()
        if "candidate" in label or "unknown" in label:
            unresolved.append(entity.entity_id)
    return sorted(dict.fromkeys(unresolved))


def _bridge_entities_from_decision_state(state: DecisionState) -> list[dict[str, Any]]:
    entities: list[dict[str, Any]] = []
    candidate = state.grounding_state.candidate_tracking.get("active_candidate")
    if candidate:
        entities.append(
            {
                "entity_id": f"candidate:{candidate}",
                "label": candidate,
                "entity_type": "candidate",
                "properties": {
                    "last_seen_room": state.grounding_state.candidate_tracking.get(
                        "last_seen_room"
                    )
                },
            }
        )
    for substance in state.grounding_state.substance_search.get(
        "grounded_substances", []
    ):
        entities.append(
            {
                "entity_id": f"substance:{substance}",
                "label": substance,
                "entity_type": "substance",
            }
        )
    for artifact in state.grounding_state.artifact_creation.get(
        "grounded_artifacts", []
    ):
        entities.append(
            {
                "entity_id": f"artifact:{artifact}",
                "label": artifact,
                "entity_type": "artifact",
            }
        )
    for target in state.grounding_state.comparison_tracking.get(
        "comparison_targets", []
    ):
        entities.append(
            {
                "entity_id": f"comparison:{target}",
                "label": target,
                "entity_type": "comparison_target",
            }
        )
    measurement_target = state.grounding_state.measurement_tracking.get(
        "measurement_target"
    )
    if measurement_target:
        entities.append(
            {
                "entity_id": f"measurement:{measurement_target}",
                "label": str(measurement_target),
                "entity_type": "measurement_target",
            }
        )
    for referent in state.grounding_state.relation_frontier.get(
        "frontier_referents", []
    ):
        entities.append(
            {
                "entity_id": f"relation:{referent}",
                "label": referent,
                "entity_type": "relation_frontier",
            }
        )
    deduped: dict[str, dict[str, Any]] = {}
    for entity in entities:
        deduped[entity["entity_id"]] = entity
    return list(deduped.values())


def build_world_model_snapshot_from_decision_state(
    state: DecisionState,
    *,
    task_text: str,
    observation_mode: str = "text",
    memory_cues: list[MemoryCue] | None = None,
    active_option: OptionContract | None = None,
    metadata: dict[str, Any] | None = None,
) -> WorldModelSnapshot:
    operator_candidates = [
        OperatorCandidate(
            operator_id=f"legacy-shortlist:{idx}",
            family="legacy_shortlist",
            action_label=action,
        )
        for idx, action in enumerate(state.action_surface.shortlist)
    ]
    frontier_summary: dict[str, Any] = {}
    remote_room = state.grounding_state.remote_room_signal
    if remote_room:
        frontier_summary = {
            "current_region": remote_room.get("room"),
            "reason": remote_room.get("reason"),
        }
    uncertainty: list[UncertaintyRecord] = []
    if state.uncertainty_state.referent_resolution:
        uncertainty.append(
            UncertaintyRecord(
                category="referent_resolution",
                magnitude=0.7,
                referents=[
                    str(
                        state.uncertainty_state.referent_resolution.get(
                            "requested_target", ""
                        )
                    )
                ],
                rationale="Legacy referent-resolution signal indicates target remapping.",
            )
        )
    snapshot_metadata = {
        "legacy_progress": {
            "task_relevant_affordance_delta_count": (
                state.progress_state.task_relevant_affordance_delta_count
            ),
            "option_progress_debt": state.progress_state.option_progress_debt,
            "option_family_value": state.progress_state.option_family_value,
        }
    }
    if metadata:
        snapshot_metadata.update(metadata)
    return WorldModelSnapshot(
        observation_mode=observation_mode,  # type: ignore[arg-type]
        goals=[GoalRecord(goal_id="primary", description=task_text)],
        entities=_bridge_entities_from_decision_state(state),
        operator_candidates=operator_candidates,
        unresolved_referents=_derive_unresolved_referents(
            state.uncertainty_state.referent_resolution,
            {
                entity["entity_id"]: WorldEntity(
                    entity_id=entity["entity_id"],
                    label=entity.get("label"),
                    entity_type=entity.get("entity_type"),
                )
                for entity in _bridge_entities_from_decision_state(state)
            },
        ),
        frontier_summary=frontier_summary,
        uncertainty=uncertainty,
        active_option=active_option,
        memory_cues=memory_cues or [],
        metadata=snapshot_metadata,
    )


class WorldModel:
    def __init__(
        self,
        *,
        capabilities: AdapterCapabilities,
        task_text: str,
        primary_goal_id: str = "primary",
    ):
        self.capabilities = capabilities
        self.task_text = task_text
        self.primary_goal_id = primary_goal_id
        self.goals: list[GoalRecord] = [
            GoalRecord(goal_id=primary_goal_id, description=task_text)
        ]
        self.entities: dict[str, WorldEntity] = {}
        self.relations: dict[tuple[str, str, str], RelationRecord] = {}
        self.operator_candidates: list[OperatorCandidate] = []
        self.frontier_state = FrontierState()
        self.memory_cues: list[MemoryCue] = []
        self.active_option: OptionContract | None = None
        self.ui_context: UIContext | None = None
        self.step_index = 0
        self.total_reward = 0.0
        self.status_snapshot: dict[str, Any] = {}
        self.last_status_delta: dict[str, Any] = {}
        self.last_action_result = ActionResult()
        self.last_event_metadata: dict[str, Any] = {}
        self.last_raw_observation = ""
        self.last_normalized_observation = ""
        self.observation_mode = capabilities.observation_mode
        self.novelty_events = 0
        self.frontier_change_events = 0
        self.action_outcomes: list[ActionOutcomeRecord] = []
        self.contradictions: list[ContradictionRecord] = []
        self.contradiction_debt = 0
        self.no_progress_streak = 0
        self.revision_required = False
        self.revision_reason: str | None = None
        self.repair_pending = False
        self.escalation_required = False
        self.repair_directive: RepairDirective | None = None
        self.blocked_action_counts: dict[str, int] = {}
        self.recent_contradiction_actions: list[str] = []
        self.repair_cycle_active = False
        self.repair_attempt_count = 0
        self.max_repair_attempts = 2

    @classmethod
    def from_event(
        cls, event: AdapterEvent, *, capabilities: AdapterCapabilities
    ) -> WorldModel:
        model = cls(capabilities=capabilities, task_text=event.task_text)
        model.apply_event(event)
        return model

    def set_memory_cues(self, memory_cues: list[MemoryCue]) -> None:
        self.memory_cues = list(memory_cues)

    def set_active_option(self, option: OptionContract | None) -> None:
        self.active_option = option

    def _derive_feedback_state(
        self,
        event: AdapterEvent,
        *,
        previous_ui_context: UIContext | None = None,
    ) -> tuple[list[ActionOutcomeRecord], list[ContradictionRecord]]:
        outcomes: list[ActionOutcomeRecord] = []
        contradictions: list[ContradictionRecord] = []

        action_text = event.action_result.action_text
        observation = event.normalized_observation or event.raw_observation
        observation_lower = observation.lower()
        direction = _extract_direction_from_action(action_text)
        ui_target = _extract_ui_target_from_action(action_text)
        beneficial_status, harmful_status, _neutral_status = _partition_status_delta(
            event.status_delta
        )
        new_entity_ids = [
            str(update.get("entity_id"))
            for update in event.entity_updates
            if update.get("entity_id")
            and str(update.get("entity_id")) not in self.entities
        ]
        new_relation_keys = [
            (relation.relation_type, relation.source_id, relation.target_id)
            for relation in event.relation_updates
            if (
                relation.relation_type,
                relation.source_id,
                relation.target_id,
            )
            not in self.relations
        ]

        progress_signals: list[str] = []
        reward_delta = float(event.reward_delta or 0.0)
        if reward_delta > 0.0:
            outcomes.append(
                ActionOutcomeRecord(
                    outcome_type="resource_change",
                    signals=["reward_delta"],
                    rationale="Reward delta changed after the action.",
                )
            )
            progress_signals.append("reward_delta")
        elif reward_delta < 0.0:
            outcomes.append(
                ActionOutcomeRecord(
                    outcome_type="contradiction",
                    signals=["negative_reward_delta"],
                    rationale="Reward decreased after the action.",
                )
            )
            contradictions.append(
                ContradictionRecord(
                    category="negative_reward_change",
                    step_index=event.step_index,
                    severity=0.75,
                    action_text=action_text,
                    evidence=str(reward_delta),
                )
            )
        if beneficial_status:
            outcomes.append(
                ActionOutcomeRecord(
                    outcome_type="status_change",
                    signals=sorted(beneficial_status.keys()),
                    rationale="Meaningful status fields improved after the action.",
                )
            )
            progress_signals.extend(sorted(beneficial_status.keys()))
        if harmful_status:
            outcomes.append(
                ActionOutcomeRecord(
                    outcome_type="contradiction",
                    signals=sorted(harmful_status.keys()),
                    rationale=("Meaningful status fields worsened after the action."),
                )
            )
            contradictions.append(
                ContradictionRecord(
                    category="negative_status_change",
                    step_index=event.step_index,
                    severity=0.75,
                    action_text=action_text,
                    evidence=", ".join(
                        f"{key}={value}"
                        for key, value in sorted(harmful_status.items())
                    )
                    or None,
                )
            )
        novelty_signals = list(event.novelty_signals or [])
        if novelty_signals:
            outcomes.append(
                ActionOutcomeRecord(
                    outcome_type="novelty_gain",
                    signals=novelty_signals[:4],
                    rationale="New novelty signals were emitted by the adapter.",
                )
            )
        if event.frontier_delta.opened or new_entity_ids or new_relation_keys:
            signals = (
                [f"frontier:{node}" for node in event.frontier_delta.opened[:3]]
                or [f"entity:{entity_id}" for entity_id in new_entity_ids[:3]]
                or [
                    f"relation:{relation_type}"
                    for relation_type, _, _ in new_relation_keys[:3]
                ]
            )
            outcomes.append(
                ActionOutcomeRecord(
                    outcome_type="progress",
                    signals=signals,
                    rationale="The action expanded the frontier or grounded new world structure.",
                )
            )
            progress_signals.extend(signals)

        ui_binding_failure = False
        if event.ui_context is not None and action_text:
            explicit_failure = str(event.action_result.failure_reason or "")
            binding_evidence = " ".join(
                part for part in (explicit_failure, observation) if part
            ).strip()
            action_family = event.action_result.operator_family or ""
            if action_family.startswith("ui_") or action_text.lower().startswith(
                ("click ", "tap ", "press ", "select ", "choose ", "type ", "fill ")
            ):
                missing_target = ui_target is not None and not _ui_target_visible(
                    ui_target, event.ui_context
                )
                if _UI_BINDING_FAILURE_RE.search(binding_evidence) or missing_target:
                    ui_binding_failure = True

        if event.ui_context is not None and not ui_binding_failure:
            ui_progress_signals: list[str] = []
            previous_visible = (
                {element.element_id for element in previous_ui_context.visible_elements}
                if previous_ui_context is not None
                else set()
            )
            current_visible = {
                element.element_id for element in event.ui_context.visible_elements
            }
            if (
                previous_ui_context is not None
                and event.ui_context.page_url
                and event.ui_context.page_url != previous_ui_context.page_url
            ):
                ui_progress_signals.append("page_url_changed")
            if (
                previous_ui_context is not None
                and event.ui_context.page_title
                and event.ui_context.page_title != previous_ui_context.page_title
            ):
                ui_progress_signals.append("page_title_changed")
            if (
                previous_ui_context is not None
                and event.ui_context.focused_element_id
                and event.ui_context.focused_element_id
                != previous_ui_context.focused_element_id
            ):
                ui_progress_signals.append("focus_changed")
            newly_visible = sorted(current_visible - previous_visible)
            if newly_visible:
                ui_progress_signals.extend(
                    f"visible:{element_id}" for element_id in newly_visible[:3]
                )
            if ui_progress_signals:
                outcomes.append(
                    ActionOutcomeRecord(
                        outcome_type="progress",
                        signals=ui_progress_signals[:4],
                        rationale="The UI state changed in a structured way after the action.",
                    )
                )
                progress_signals.extend(ui_progress_signals)

        blocked = False
        if direction and event.spatial_context is not None:
            blocked = direction in set(event.spatial_context.blocked_directions)
        blocked = blocked or bool(_OBS_BLOCKED_RE.search(observation_lower))
        if blocked and action_text:
            outcomes.append(
                ActionOutcomeRecord(
                    outcome_type="blocked",
                    signals=[direction] if direction else [],
                    rationale="The attempted action produced explicit blockage feedback.",
                )
            )
            contradictions.append(
                ContradictionRecord(
                    category="blocked_path",
                    step_index=event.step_index,
                    severity=1.0,
                    subject=direction,
                    action_text=action_text,
                    evidence=observation.splitlines()[0][:160] if observation else None,
                )
            )

        if event.ui_context is not None and action_text:
            explicit_failure = str(event.action_result.failure_reason or "")
            binding_evidence = " ".join(
                part for part in (explicit_failure, observation) if part
            ).strip()
            if ui_binding_failure:
                outcomes.append(
                    ActionOutcomeRecord(
                        outcome_type="contradiction",
                        signals=["ui_target_binding"],
                        rationale=(
                            "The intended UI target appears stale, unavailable, or no longer actionable."
                        ),
                    )
                )
                contradictions.append(
                    ContradictionRecord(
                        category="ui_target_binding",
                        step_index=event.step_index,
                        severity=1.0,
                        subject=ui_target,
                        action_text=action_text,
                        evidence=binding_evidence[:160] if binding_evidence else None,
                    )
                )

        if event.action_result.failure_reason:
            outcomes.append(
                ActionOutcomeRecord(
                    outcome_type="contradiction",
                    signals=["failure_reason"],
                    rationale="The adapter reported an explicit failure reason.",
                )
            )
            contradictions.append(
                ContradictionRecord(
                    category="execution_failure",
                    step_index=event.step_index,
                    severity=1.0,
                    action_text=action_text,
                    evidence=str(event.action_result.failure_reason),
                )
            )

        if (
            action_text
            and not progress_signals
            and not blocked
            and not ui_binding_failure
        ):
            outcomes.append(
                ActionOutcomeRecord(
                    outcome_type="no_effect",
                    rationale="The action consumed a step without reward, novelty, or state progress.",
                )
            )

        if not outcomes:
            outcomes.append(
                ActionOutcomeRecord(
                    outcome_type="unknown",
                    rationale="No typed outcome signal was derived from the latest event.",
                )
            )

        return outcomes, contradictions

    def _update_feedback_memory(
        self,
        *,
        event: AdapterEvent,
        outcomes: list[ActionOutcomeRecord],
        contradictions: list[ContradictionRecord],
    ) -> None:
        was_repair_cycle_active = self.repair_cycle_active
        repair_step_executed = (
            self.active_option is not None
            and self.active_option.family == "recover_from_failure"
            and bool(event.action_result.action_text)
        )
        progress_hit = any(
            outcome.outcome_type in {"progress", "resource_change", "status_change"}
            for outcome in outcomes
        )
        contradiction_hit = any(
            outcome.outcome_type in {"blocked", "contradiction"} for outcome in outcomes
        )
        no_effect_hit = any(outcome.outcome_type == "no_effect" for outcome in outcomes)
        structural_progress_hit = (
            progress_hit and not contradiction_hit and not no_effect_hit
        )

        self.action_outcomes = (self.action_outcomes + list(outcomes))[-8:]
        self.contradictions = (self.contradictions + list(contradictions))[-8:]

        action_text = event.action_result.action_text
        if action_text and contradiction_hit:
            normalized_action = action_text.strip().lower()
            self.blocked_action_counts[normalized_action] = (
                self.blocked_action_counts.get(normalized_action, 0) + 1
            )
            self.recent_contradiction_actions = (
                self.recent_contradiction_actions + [normalized_action]
            )[-6:]

        if structural_progress_hit:
            self.no_progress_streak = 0
            self.contradiction_debt = max(0, self.contradiction_debt - 1)
            self.repair_cycle_active = False
            self.repair_attempt_count = 0
        else:
            if event.action_result.action_text:
                self.no_progress_streak += 1
            if contradiction_hit:
                self.contradiction_debt += 2
            elif no_effect_hit:
                self.contradiction_debt += 1
            if repair_step_executed:
                self.repair_attempt_count += 1

        repeated_contradiction = False
        ui_binding_hit = any(
            contradiction.category == "ui_target_binding"
            for contradiction in contradictions
        )
        if action_text:
            repeated_contradiction = (
                self.blocked_action_counts.get(action_text.strip().lower(), 0) >= 2
            )

        trigger_reason: str | None = None
        if ui_binding_hit:
            trigger_reason = "ui_target_binding"
        elif repeated_contradiction:
            trigger_reason = "repeated_contradiction"
        elif self.contradiction_debt >= 3:
            trigger_reason = "contradiction_debt_exceeded"
        elif self.no_progress_streak >= 4:
            trigger_reason = "nonprogress_streak"

        if structural_progress_hit:
            self.repair_pending = False
            self.escalation_required = False
        elif was_repair_cycle_active:
            self.repair_cycle_active = True
            if (
                repair_step_executed
                and self.repair_attempt_count >= self.max_repair_attempts
            ):
                self.repair_pending = False
                self.escalation_required = True
                trigger_reason = "repair_budget_exhausted"
            else:
                self.repair_pending = True
                self.escalation_required = False
                if repair_step_executed:
                    trigger_reason = "repair_incomplete"
                elif trigger_reason is None:
                    trigger_reason = "repair_incomplete"
        elif trigger_reason is not None:
            self.repair_cycle_active = True
            self.repair_attempt_count = 0
            self.repair_pending = True
            self.escalation_required = False
        else:
            self.repair_cycle_active = False
            self.repair_pending = False
            self.escalation_required = False

        self.revision_required = False
        self.revision_reason = None
        if self.repair_pending or self.escalation_required:
            self.revision_required = True
            self.revision_reason = trigger_reason
        self.repair_directive = self._build_repair_directive()

    def _build_repair_directive(self) -> RepairDirective | None:
        if not self.revision_required:
            return None

        invalidated_actions = sorted(
            action for action, count in self.blocked_action_counts.items() if count > 0
        )[:6]
        recent_categories = [
            contradiction.category for contradiction in self.contradictions[-3:]
        ]

        if self.ui_context is not None:
            if "ui_target_binding" in recent_categories:
                return RepairDirective(
                    operator="repair_target_binding",
                    rationale=(
                        "The current UI target appears stale or unavailable; refresh target binding before continuing."
                    ),
                    preferred_families=[
                        "inspect",
                        "ui_navigation",
                        "ui_select",
                        "ui_click",
                        "ui_type",
                    ],
                    invalidated_actions=invalidated_actions,
                    metadata={"recent_categories": recent_categories},
                )
            if "execution_failure" in recent_categories:
                return RepairDirective(
                    operator="repair_operator_set",
                    rationale=(
                        "The current UI action surface may be stale; refresh available operators before continuing."
                    ),
                    preferred_families=[
                        "inspect",
                        "ui_navigation",
                        "ui_click",
                        "ui_type",
                    ],
                    invalidated_actions=invalidated_actions,
                    metadata={"recent_categories": recent_categories},
                )
            return RepairDirective(
                operator="repair_target_binding",
                rationale=(
                    "The current UI target or action scope may be stale; refresh the "
                    "binding before continuing."
                ),
                preferred_families=["inspect", "ui_select", "ui_click", "ui_type"],
                invalidated_actions=invalidated_actions,
                metadata={"recent_categories": recent_categories},
            )

        if any(action in {"go up", "go down"} for action in invalidated_actions) or (
            "execution_failure" in recent_categories
        ):
            return RepairDirective(
                operator="repair_transition_model",
                rationale=(
                    "The attempted transition or operator preconditions were contradicted; "
                    "refresh local transition assumptions before continuing."
                ),
                preferred_families=["inspect", "interaction", "relocation"],
                invalidated_actions=invalidated_actions,
                metadata={"recent_categories": recent_categories},
            )

        if not self.operator_candidates:
            return RepairDirective(
                operator="repair_operator_set",
                rationale=(
                    "The local operator set is unstable or empty; refresh admissible "
                    "affordances before continuing."
                ),
                preferred_families=["inspect", "interaction"],
                invalidated_actions=invalidated_actions,
                metadata={"recent_categories": recent_categories},
            )

        return RepairDirective(
            operator="repair_topology",
            rationale=(
                "Recent blocked or contradictory outcomes indicate the local topology "
                "model is stale and should be repaired before continuing."
            ),
            preferred_families=["inspect", "relocation", "interaction"],
            invalidated_actions=invalidated_actions,
            metadata={"recent_categories": recent_categories},
        )

    def apply_event(self, event: AdapterEvent) -> None:
        self.step_index = event.step_index
        self.last_raw_observation = event.raw_observation
        self.last_normalized_observation = event.normalized_observation
        self.last_action_result = event.action_result
        self.last_event_metadata = dict(event.metadata)
        self.operator_candidates = list(event.operator_candidates)
        self.total_reward += float(event.reward_delta or 0.0)
        previous_ui_context = self.ui_context
        if event.status_delta:
            self.status_snapshot.update(event.status_delta)
            self.last_status_delta = dict(event.status_delta)
        else:
            self.last_status_delta = {}
        if event.ui_context is not None:
            self.ui_context = event.ui_context
            self.observation_mode = "ui"
        elif event.spatial_context is not None:
            self.observation_mode = "grid"
            visit_counts = dict(self.frontier_state.visit_counts)
            current_node_id = event.spatial_context.current_node_id
            revisited_nodes = list(event.frontier_delta.revisited)
            if current_node_id:
                visit_counts[current_node_id] = visit_counts.get(current_node_id, 0) + 1
                if visit_counts[current_node_id] > 1:
                    revisited_nodes = _merge_unique(revisited_nodes, [current_node_id])
            self.frontier_state = FrontierState(
                current_region=event.spatial_context.current_region,
                current_node_id=current_node_id,
                visible_nodes=list(event.spatial_context.visible_nodes),
                frontier_nodes=list(event.spatial_context.frontier_nodes),
                blocked_nodes=list(event.spatial_context.blocked_directions),
                opened_nodes=_merge_unique(
                    self.frontier_state.opened_nodes, list(event.frontier_delta.opened)
                ),
                closed_nodes=_merge_unique(
                    self.frontier_state.closed_nodes, list(event.frontier_delta.closed)
                ),
                revisited_nodes=_merge_unique(
                    self.frontier_state.revisited_nodes, revisited_nodes
                ),
                visit_counts=visit_counts,
            )
            if (
                event.frontier_delta.opened
                or event.frontier_delta.closed
                or event.frontier_delta.blocked
            ):
                self.frontier_change_events += 1
        else:
            self.observation_mode = self.capabilities.observation_mode
        if event.novelty_signals:
            self.novelty_events += len(event.novelty_signals)
        outcomes, contradictions = self._derive_feedback_state(
            event,
            previous_ui_context=previous_ui_context,
        )
        self._update_feedback_memory(
            event=event,
            outcomes=outcomes,
            contradictions=contradictions,
        )
        for update in event.entity_updates:
            entity_id = str(update.get("entity_id") or f"entity:{len(self.entities)}")
            existing = self.entities.get(entity_id)
            aliases = list(update.get("aliases", []))
            source_channel = (
                update.get("source_channel") or self.capabilities.adapter_name
            )
            if existing is None:
                self.entities[entity_id] = WorldEntity(
                    entity_id=entity_id,
                    label=update.get("label"),
                    entity_type=update.get("entity_type"),
                    aliases=aliases,
                    properties=dict(update.get("properties", {})),
                    last_seen_step=event.step_index,
                    confidence=update.get("confidence"),
                    source_channels=[source_channel] if source_channel else [],
                )
                continue
            self.entities[entity_id] = WorldEntity(
                entity_id=entity_id,
                label=update.get("label", existing.label),
                entity_type=update.get("entity_type", existing.entity_type),
                aliases=_merge_unique(existing.aliases, aliases),
                properties={
                    **existing.properties,
                    **dict(update.get("properties", {})),
                },
                last_seen_step=event.step_index,
                confidence=update.get("confidence", existing.confidence),
                source_channels=_merge_unique(
                    existing.source_channels,
                    [source_channel] if source_channel else [],
                ),
            )
        for relation in event.relation_updates:
            key = (relation.relation_type, relation.source_id, relation.target_id)
            self.relations[key] = relation

    def _build_uncertainty(self) -> list[UncertaintyRecord]:
        uncertainty: list[UncertaintyRecord] = []
        unresolved = _derive_unresolved_referents(
            self.last_event_metadata.get("referent_resolution", {}),
            self.entities,
        )
        if unresolved:
            uncertainty.append(
                UncertaintyRecord(
                    category="referent",
                    magnitude=0.7,
                    referents=unresolved,
                    rationale="Entity labels or referent-resolution metadata remain unresolved.",
                )
            )
        if not self.operator_candidates:
            uncertainty.append(
                UncertaintyRecord(
                    category="action_surface",
                    magnitude=0.8,
                    rationale="No operator candidates are currently available.",
                )
            )
        return uncertainty

    def to_snapshot(self) -> WorldModelSnapshot:
        metadata: dict[str, Any] = {
            "step_index": self.step_index,
            "total_reward": self.total_reward,
            "status_snapshot": dict(self.status_snapshot),
            "last_status_delta": dict(self.last_status_delta),
            "last_action_result": self.last_action_result.to_dict(),
            "novelty_events": self.novelty_events,
            "frontier_change_events": self.frontier_change_events,
            "contradiction_debt": self.contradiction_debt,
            "no_progress_streak": self.no_progress_streak,
            "revision_reason": self.revision_reason,
            "repair_cycle_active": self.repair_cycle_active,
            "repair_pending": self.repair_pending,
            "escalation_required": self.escalation_required,
            "repair_attempt_count": self.repair_attempt_count,
            "max_repair_attempts": self.max_repair_attempts,
            "blocked_action_labels": sorted(
                action
                for action, count in self.blocked_action_counts.items()
                if count > 0
            )[:6],
            "recent_contradiction_actions": list(
                self.recent_contradiction_actions[-4:]
            ),
            "capabilities": self.capabilities.to_dict(),
        }
        if self.ui_context is not None:
            metadata["ui_context"] = self.ui_context.to_dict()
        elif self.frontier_state.current_region or self.frontier_state.frontier_nodes:
            metadata["frontier_state"] = self.frontier_state.to_dict()
        return WorldModelSnapshot(
            observation_mode=self.observation_mode,
            goals=list(self.goals),
            entities=[entity.to_dict() for entity in self.entities.values()],
            relations=list(self.relations.values()),
            operator_candidates=list(self.operator_candidates),
            unresolved_referents=_derive_unresolved_referents({}, self.entities),
            frontier_summary=self.frontier_state.to_dict(),
            uncertainty=self._build_uncertainty(),
            active_option=self.active_option,
            memory_cues=list(self.memory_cues),
            action_outcomes=list(self.action_outcomes),
            contradictions=list(self.contradictions),
            revision_required=self.revision_required,
            repair_directive=self.repair_directive,
            metadata=metadata,
        )

    def to_analyst_view(self) -> dict[str, Any]:
        snapshot = self.to_snapshot()
        analyst_view = {
            "task": self.task_text,
            "observation_mode": snapshot.observation_mode,
            "goals": [goal.to_dict() for goal in self.goals],
            "frontier": snapshot.frontier_summary,
            "status": dict(self.status_snapshot),
            "last_action_result": self.last_action_result.to_dict(),
            "operator_families": sorted(
                {candidate.family for candidate in self.operator_candidates}
            ),
            "entity_count": len(self.entities),
            "uncertainty": [record.to_dict() for record in snapshot.uncertainty],
            "revision_required": snapshot.revision_required,
            "revision_reason": snapshot.metadata.get("revision_reason"),
            "contradiction_debt": snapshot.metadata.get("contradiction_debt", 0),
            "repair_operator": (
                snapshot.repair_directive.operator
                if snapshot.repair_directive is not None
                else None
            ),
            "repair_cycle_active": snapshot.metadata.get("repair_cycle_active", False),
            "repair_pending": snapshot.metadata.get("repair_pending", False),
            "escalation_required": snapshot.metadata.get("escalation_required", False),
            "repair_attempt_count": snapshot.metadata.get("repair_attempt_count", 0),
            "recent_outcomes": [
                outcome.outcome_type for outcome in snapshot.action_outcomes[-3:]
            ],
            "memory_cue_count": len(self.memory_cues),
            "active_option": (
                self.active_option.to_dict() if self.active_option is not None else {}
            ),
        }
        if self.ui_context is not None:
            analyst_view["ui"] = {
                "page_title": self.ui_context.page_title,
                "page_url": self.ui_context.page_url,
                "focused_element_id": self.ui_context.focused_element_id,
                "visible_element_count": len(self.ui_context.visible_elements),
            }
        return analyst_view
