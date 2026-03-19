from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from src.agent.decision_state import DecisionState
from src.agent.v2.types import (
    ActionResult,
    AdapterCapabilities,
    AdapterEvent,
    GoalRecord,
    MemoryCue,
    OperatorCandidate,
    OptionContract,
    RelationRecord,
    UIContext,
    UncertaintyRecord,
    WorldModelSnapshot,
)


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

    def apply_event(self, event: AdapterEvent) -> None:
        self.step_index = event.step_index
        self.last_raw_observation = event.raw_observation
        self.last_normalized_observation = event.normalized_observation
        self.last_action_result = event.action_result
        self.last_event_metadata = dict(event.metadata)
        self.operator_candidates = list(event.operator_candidates)
        self.total_reward += float(event.reward_delta or 0.0)
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
