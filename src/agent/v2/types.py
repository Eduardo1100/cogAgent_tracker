from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Literal

ObservationMode = Literal["text", "grid", "ui", "hybrid"]
TopologyType = Literal["grid", "graph", "ui", "mixed"]
ReasoningTier = Literal["none", "light", "full"]
RepairOperator = Literal[
    "repair_topology",
    "repair_transition_model",
    "repair_target_binding",
    "repair_operator_set",
]
OutcomeType = Literal[
    "progress",
    "blocked",
    "no_effect",
    "unexpected_transition",
    "contradiction",
    "resource_change",
    "status_change",
    "novelty_gain",
    "unknown",
]


def _drop_empty(value: Any) -> Any:
    if isinstance(value, dict):
        cleaned = {
            key: _drop_empty(item)
            for key, item in value.items()
            if _drop_empty(item) not in ({}, [], "", None)
        }
        return cleaned
    if isinstance(value, list):
        return [
            _drop_empty(item)
            for item in value
            if _drop_empty(item) not in ({}, [], "", None)
        ]
    return value


@dataclass(frozen=True)
class _CompactSerializable:
    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def to_compact_dict(self) -> dict[str, Any]:
        return _drop_empty(self.to_dict())


@dataclass(frozen=True)
class AdapterCapabilities(_CompactSerializable):
    adapter_name: str
    observation_mode: ObservationMode = "text"
    supports_spatial_context: bool = False
    supports_ui_context: bool = False
    supports_reward_delta: bool = False
    supports_status_delta: bool = False
    supports_operator_candidates: bool = True
    supports_episode_memory: bool = True


@dataclass(frozen=True)
class OperatorCandidate(_CompactSerializable):
    operator_id: str
    family: str
    action_label: str
    target_ids: list[str] = field(default_factory=list)
    preconditions: list[str] = field(default_factory=list)
    expected_effects: list[str] = field(default_factory=list)
    reversible: bool = True
    grounded: bool = True
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class RelationRecord(_CompactSerializable):
    relation_type: str
    source_id: str
    target_id: str
    confidence: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class SpatialContext(_CompactSerializable):
    topology: TopologyType = "grid"
    current_region: str | None = None
    current_node_id: str | None = None
    visible_nodes: list[str] = field(default_factory=list)
    frontier_nodes: list[str] = field(default_factory=list)
    passable_directions: list[str] = field(default_factory=list)
    blocked_directions: list[str] = field(default_factory=list)
    local_map_summary: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class UIElementRecord(_CompactSerializable):
    element_id: str
    role: str
    name: str | None = None
    text: str | None = None
    selector: str | None = None
    interactable: bool = True
    disabled: bool = False
    selected: bool = False
    bounds: dict[str, float] = field(default_factory=dict)
    attributes: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class UIContext(_CompactSerializable):
    page_url: str | None = None
    page_title: str | None = None
    focused_element_id: str | None = None
    active_dialog: str | None = None
    visible_elements: list[UIElementRecord] = field(default_factory=list)
    action_scope: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class FrontierDelta(_CompactSerializable):
    opened: list[str] = field(default_factory=list)
    closed: list[str] = field(default_factory=list)
    revisited: list[str] = field(default_factory=list)
    blocked: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class ActionResult(_CompactSerializable):
    action_text: str | None = None
    action_executed: bool | None = None
    operator_family: str | None = None
    failure_reason: str | None = None
    changed_observation: bool | None = None


@dataclass(frozen=True)
class ActionOutcomeRecord(_CompactSerializable):
    outcome_type: OutcomeType
    confidence: float = 1.0
    signals: list[str] = field(default_factory=list)
    rationale: str | None = None


@dataclass(frozen=True)
class ContradictionRecord(_CompactSerializable):
    category: str
    step_index: int
    severity: float = 1.0
    subject: str | None = None
    action_text: str | None = None
    evidence: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class RepairDirective(_CompactSerializable):
    operator: RepairOperator
    rationale: str
    preferred_families: list[str] = field(default_factory=list)
    invalidated_actions: list[str] = field(default_factory=list)
    target_signature: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class AdapterEvent(_CompactSerializable):
    step_index: int
    task_text: str
    raw_observation: str
    normalized_observation: str
    action_result: ActionResult = field(default_factory=ActionResult)
    operator_candidates: list[OperatorCandidate] = field(default_factory=list)
    entity_updates: list[dict[str, Any]] = field(default_factory=list)
    relation_updates: list[RelationRecord] = field(default_factory=list)
    reward_delta: float | None = None
    status_delta: dict[str, float | int | str | bool] = field(default_factory=dict)
    frontier_delta: FrontierDelta = field(default_factory=FrontierDelta)
    novelty_signals: list[str] = field(default_factory=list)
    spatial_context: SpatialContext | None = None
    ui_context: UIContext | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class GoalRecord(_CompactSerializable):
    goal_id: str
    description: str
    priority: float = 1.0
    status: str = "active"
    parent_goal_id: str | None = None


@dataclass(frozen=True)
class MemoryCue(_CompactSerializable):
    cue_id: str
    summary: str
    source_scope: str
    relevance: float | None = None
    applicability: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class UncertaintyRecord(_CompactSerializable):
    category: str
    magnitude: float
    referents: list[str] = field(default_factory=list)
    rationale: str | None = None


@dataclass(frozen=True)
class OptionContract(_CompactSerializable):
    family: str
    objective: str
    target_signature: str | None = None
    expected_outcomes: list[str] = field(default_factory=list)
    progress_budget: int = 0
    failure_budget: int = 0
    termination_conditions: list[str] = field(default_factory=list)
    interrupt_conditions: list[str] = field(default_factory=list)
    reasoning_budget: ReasoningTier = "light"
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class WorldModelSnapshot(_CompactSerializable):
    observation_mode: ObservationMode = "text"
    goals: list[GoalRecord] = field(default_factory=list)
    entities: list[dict[str, Any]] = field(default_factory=list)
    relations: list[RelationRecord] = field(default_factory=list)
    operator_candidates: list[OperatorCandidate] = field(default_factory=list)
    unresolved_referents: list[str] = field(default_factory=list)
    frontier_summary: dict[str, Any] = field(default_factory=dict)
    uncertainty: list[UncertaintyRecord] = field(default_factory=list)
    active_option: OptionContract | None = None
    memory_cues: list[MemoryCue] = field(default_factory=list)
    action_outcomes: list[ActionOutcomeRecord] = field(default_factory=list)
    contradictions: list[ContradictionRecord] = field(default_factory=list)
    revision_required: bool = False
    repair_directive: RepairDirective | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class PlannerDirective(_CompactSerializable):
    option: OptionContract | None = None
    reasoning_tier: ReasoningTier = "light"
    continue_current_option: bool = False
    planner_notes: list[str] = field(default_factory=list)
    memory_queries: list[str] = field(default_factory=list)
    stop_reason: str | None = None


@dataclass(frozen=True)
class V2RuntimeConfig(_CompactSerializable):
    adapter_namespace: str = "generic"
    max_execution_steps_per_option: int = 8
    max_memory_cues: int = 6
    max_planner_tokens: int = 1200
    allow_lightweight_replan: bool = True
    enable_online_memory_updates: bool = True
    planner_reasoning_tier: ReasoningTier = "light"
