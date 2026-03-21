from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

from src.agent.runtime import RuntimeDecision, RuntimeSessionSnapshot
from src.agent.v2.types import (
    ActionResult,
    AdapterCapabilities,
    AdapterEvent,
    FrontierDelta,
    InputState,
    OperatorCandidate,
    RelationRecord,
    SpatialContext,
    UIContext,
    UIElementRecord,
)


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
class _BridgeSerializable:
    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def to_compact_dict(self) -> dict[str, Any]:
        return _drop_empty(self.to_dict())


def _parse_ui_element_records(
    items: list[dict[str, Any]] | None,
) -> list[UIElementRecord]:
    return [UIElementRecord(**dict(item)) for item in items or []]


def _parse_input_state(data: dict[str, Any] | None) -> InputState | None:
    if not data:
        return None
    return InputState(**dict(data))


def _parse_ui_context(data: dict[str, Any] | None) -> UIContext | None:
    if not data:
        return None
    payload = dict(data)
    payload["input_state"] = _parse_input_state(payload.get("input_state"))
    payload["visible_elements"] = _parse_ui_element_records(
        payload.get("visible_elements")
    )
    return UIContext(**payload)


def _parse_spatial_context(data: dict[str, Any] | None) -> SpatialContext | None:
    if not data:
        return None
    return SpatialContext(**dict(data))


def _parse_action_result(data: dict[str, Any] | None) -> ActionResult:
    if not data:
        return ActionResult()
    return ActionResult(**dict(data))


def _parse_frontier_delta(data: dict[str, Any] | None) -> FrontierDelta:
    if not data:
        return FrontierDelta()
    return FrontierDelta(**dict(data))


def _parse_operator_candidates(
    items: list[dict[str, Any]] | None,
) -> list[OperatorCandidate]:
    return [OperatorCandidate(**dict(item)) for item in items or []]


def _parse_relation_updates(
    items: list[dict[str, Any]] | None,
) -> list[RelationRecord]:
    return [RelationRecord(**dict(item)) for item in items or []]


def _parse_capabilities(data: dict[str, Any] | None) -> AdapterCapabilities | None:
    if not data:
        return None
    return AdapterCapabilities(**dict(data))


def _serialize_bridge_dataclass(value: Any) -> Any:
    if hasattr(value, "to_dict") and callable(value.to_dict):
        return value.to_dict()
    return value


@dataclass(frozen=True)
class OpenClawSessionCreateRequest(_BridgeSerializable):
    task_text: str
    mode: str = "protocol"
    escalation_policy: str = "auto"
    session_metadata: dict[str, Any] = field(default_factory=dict)
    capabilities: dict[str, Any] | None = None
    initial_event: dict[str, Any] | None = None

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> OpenClawSessionCreateRequest:
        return cls(
            task_text=str(data["task_text"]),
            mode=str(data.get("mode", "protocol")),
            escalation_policy=str(data.get("escalation_policy", "auto")),
            session_metadata=dict(data.get("session_metadata") or {}),
            capabilities=dict(data["capabilities"])
            if data.get("capabilities")
            else None,
            initial_event=dict(data["initial_event"])
            if data.get("initial_event")
            else None,
        )

    def capabilities_record(self) -> AdapterCapabilities | None:
        return _parse_capabilities(self.capabilities)

    def initial_adapter_event(self) -> AdapterEvent | None:
        if not self.initial_event:
            return None
        return OpenClawObservationRequest.from_dict(
            {
                "session_id": "bootstrap",
                "event": self.initial_event,
                "metadata": self.session_metadata,
            }
        ).to_adapter_event()


@dataclass(frozen=True)
class OpenClawObservationRequest(_BridgeSerializable):
    session_id: str
    event: dict[str, Any]
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> OpenClawObservationRequest:
        return cls(
            session_id=str(data["session_id"]),
            event=dict(data["event"]),
            metadata=dict(data.get("metadata") or {}),
        )

    def to_adapter_event(self) -> AdapterEvent:
        payload = dict(self.event)
        return AdapterEvent(
            step_index=int(payload["step_index"]),
            task_text=str(payload["task_text"]),
            raw_observation=str(payload["raw_observation"]),
            normalized_observation=str(
                payload.get("normalized_observation", payload["raw_observation"])
            ),
            action_result=_parse_action_result(payload.get("action_result")),
            operator_candidates=_parse_operator_candidates(
                payload.get("operator_candidates")
            ),
            entity_updates=list(payload.get("entity_updates") or []),
            relation_updates=_parse_relation_updates(payload.get("relation_updates")),
            reward_delta=payload.get("reward_delta"),
            status_delta=dict(payload.get("status_delta") or {}),
            frontier_delta=_parse_frontier_delta(payload.get("frontier_delta")),
            novelty_signals=list(payload.get("novelty_signals") or []),
            spatial_context=_parse_spatial_context(payload.get("spatial_context")),
            ui_context=_parse_ui_context(payload.get("ui_context")),
            metadata=dict(payload.get("metadata") or {}),
        )


@dataclass(frozen=True)
class OpenClawActionApplyRequest(_BridgeSerializable):
    session_id: str
    action_text: str | None = None
    action_executed: bool | None = None
    reward_delta: float | None = None
    status_delta: dict[str, float | int | str | bool] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> OpenClawActionApplyRequest:
        return cls(
            session_id=str(data["session_id"]),
            action_text=data.get("action_text"),
            action_executed=data.get("action_executed"),
            reward_delta=data.get("reward_delta"),
            status_delta=dict(data.get("status_delta") or {}),
            metadata=dict(data.get("metadata") or {}),
        )


@dataclass(frozen=True)
class OpenClawDeliberationRequest(_BridgeSerializable):
    session_id: str
    backend: str | None = None
    reason: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> OpenClawDeliberationRequest:
        return cls(
            session_id=str(data["session_id"]),
            backend=(str(data["backend"]) if data.get("backend") is not None else None),
            reason=data.get("reason"),
            metadata=dict(data.get("metadata") or {}),
        )


@dataclass(frozen=True)
class OpenClawDecisionResponse(_BridgeSerializable):
    session_id: str
    suggested_action: str | None = None
    option_family: str | None = None
    repair_operator: str | None = None
    planner_stop_reason: str | None = None
    revision_required: bool = False
    mode_used: str = "protocol"
    escalated: bool = False
    escalation_reason: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> OpenClawDecisionResponse:
        return cls(
            session_id=str(data["session_id"]),
            suggested_action=data.get("suggested_action"),
            option_family=data.get("option_family"),
            repair_operator=data.get("repair_operator"),
            planner_stop_reason=data.get("planner_stop_reason"),
            revision_required=bool(data.get("revision_required", False)),
            mode_used=str(data.get("mode_used", "protocol")),
            escalated=bool(data.get("escalated", False)),
            escalation_reason=data.get("escalation_reason"),
            metadata=dict(data.get("metadata") or {}),
        )

    @classmethod
    def from_runtime_decision(
        cls, session_id: str, decision: RuntimeDecision
    ) -> OpenClawDecisionResponse:
        return cls(
            session_id=session_id,
            suggested_action=decision.suggested_action,
            option_family=decision.option_family,
            repair_operator=decision.repair_operator,
            planner_stop_reason=decision.planner_stop_reason,
            revision_required=decision.revision_required,
            mode_used=decision.mode_used,
            escalated=decision.escalated,
            escalation_reason=decision.escalation_reason,
            metadata=dict(decision.metadata),
        )


@dataclass(frozen=True)
class OpenClawDeliberationJob(_BridgeSerializable):
    job_id: str
    session_id: str
    status: str
    mode: str = "deliberative"
    backend: str = "in_memory_stub"
    decision: dict[str, Any] | None = None
    error: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> OpenClawDeliberationJob:
        return cls(
            job_id=str(data["job_id"]),
            session_id=str(data["session_id"]),
            status=str(data["status"]),
            mode=str(data.get("mode", "deliberative")),
            backend=str(data.get("backend", "in_memory_stub")),
            decision=dict(data["decision"]) if data.get("decision") else None,
            error=data.get("error"),
            metadata=dict(data.get("metadata") or {}),
        )

    @classmethod
    def completed(
        cls,
        *,
        job_id: str,
        session_id: str,
        decision: OpenClawDecisionResponse,
        backend: str,
        metadata: dict[str, Any] | None = None,
    ) -> OpenClawDeliberationJob:
        return cls(
            job_id=job_id,
            session_id=session_id,
            status="completed",
            mode="deliberative",
            backend=backend,
            decision=decision.to_dict(),
            metadata=dict(metadata or {}),
        )


@dataclass(frozen=True)
class OpenClawSessionSnapshot(_BridgeSerializable):
    session_id: str
    game_no: int | None = None
    task: str | None = None
    observation: str = ""
    admissible_actions: list[str] = field(default_factory=list)
    runtime_advisory: dict[str, Any] = field(default_factory=dict)
    protocol_summary: dict[str, Any] = field(default_factory=dict)
    comparison_telemetry: dict[str, Any] = field(default_factory=dict)
    log_paths: dict[str, str] = field(default_factory=dict)
    latest_decision: dict[str, Any] | None = None
    mode: str = "protocol"
    escalation_policy: str = "auto"
    escalation_state: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> OpenClawSessionSnapshot:
        return cls(
            session_id=str(data["session_id"]),
            game_no=data.get("game_no"),
            task=data.get("task"),
            observation=str(data.get("observation", "")),
            admissible_actions=list(data.get("admissible_actions") or []),
            runtime_advisory=dict(data.get("runtime_advisory") or {}),
            protocol_summary=dict(data.get("protocol_summary") or {}),
            comparison_telemetry=dict(data.get("comparison_telemetry") or {}),
            log_paths=dict(data.get("log_paths") or {}),
            latest_decision=dict(data["latest_decision"])
            if data.get("latest_decision")
            else None,
            mode=str(data.get("mode", "protocol")),
            escalation_policy=str(data.get("escalation_policy", "auto")),
            escalation_state=dict(data.get("escalation_state") or {}),
        )

    @classmethod
    def from_runtime_snapshot(
        cls,
        snapshot: RuntimeSessionSnapshot,
        *,
        latest_decision: RuntimeDecision | None = None,
    ) -> OpenClawSessionSnapshot:
        return cls(
            session_id=snapshot.session_id,
            game_no=snapshot.game_no,
            task=snapshot.task,
            observation=snapshot.observation,
            admissible_actions=list(snapshot.admissible_actions),
            runtime_advisory=dict(snapshot.runtime_advisory),
            protocol_summary=dict(snapshot.protocol_summary),
            comparison_telemetry=dict(snapshot.comparison_telemetry),
            log_paths=dict(snapshot.log_paths),
            latest_decision=(
                OpenClawDecisionResponse.from_runtime_decision(
                    snapshot.session_id, latest_decision
                ).to_dict()
                if latest_decision is not None
                else None
            ),
            mode=snapshot.mode,
            escalation_policy=snapshot.escalation_policy,
            escalation_state=dict(snapshot.escalation_state),
        )

    def to_runtime_decision(self) -> RuntimeDecision | None:
        if not self.latest_decision:
            return None
        payload = dict(self.latest_decision)
        return RuntimeDecision(
            suggested_action=payload.get("suggested_action"),
            option_family=payload.get("option_family"),
            repair_operator=payload.get("repair_operator"),
            planner_stop_reason=payload.get("planner_stop_reason"),
            revision_required=bool(payload.get("revision_required", False)),
            mode_used=str(payload.get("mode_used", "protocol")),
            escalated=bool(payload.get("escalated", False)),
            escalation_reason=payload.get("escalation_reason"),
            metadata=dict(payload.get("metadata") or {}),
        )
