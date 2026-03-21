from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any
from uuid import uuid4

from src.agent.v2.controller import V2Controller
from src.agent.v2.types import AdapterCapabilities, AdapterEvent
from src.agent.v2.world_model import WorldModel


@dataclass(frozen=True)
class RuntimeDecision:
    suggested_action: str | None = None
    option_family: str | None = None
    repair_operator: str | None = None
    planner_stop_reason: str | None = None
    revision_required: bool = False
    mode_used: str = "protocol"
    escalated: bool = False
    escalation_reason: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class RuntimeSessionSnapshot:
    session_id: str
    game_no: int | None = None
    task: str | None = None
    observation: str = ""
    admissible_actions: list[str] = field(default_factory=list)
    runtime_advisory: dict[str, Any] = field(default_factory=dict)
    protocol_summary: dict[str, Any] = field(default_factory=dict)
    comparison_telemetry: dict[str, Any] = field(default_factory=dict)
    log_paths: dict[str, str] = field(default_factory=dict)
    mode: str = "protocol"
    escalation_policy: str = "auto"
    escalation_state: dict[str, Any] = field(default_factory=dict)


@dataclass
class RuntimeSession:
    agent: Any | None = None
    session_id: str = field(default_factory=lambda: uuid4().hex)
    env: Any | None = None
    game_no: int | None = None
    info: dict[str, Any] = field(default_factory=dict)
    adapter: Any | None = None
    initialized: bool = False
    controller: V2Controller = field(default_factory=V2Controller)
    world_model: WorldModel | None = None
    capabilities: AdapterCapabilities | None = None
    _latest_decision: RuntimeDecision | None = None
    _latest_event: AdapterEvent | None = None
    mode: str = "protocol"
    escalation_policy: str = "auto"
    protocol_revision_streak: int = 0
    last_escalation_reason: str | None = None
    comparison_events: list[dict[str, Any]] = field(default_factory=list)

    def initialize(
        self,
        *,
        env: Any,
        obs: Any,
        info: dict[str, Any] | None,
        game_no: int,
        adapter: Any | None = None,
    ) -> None:
        if self.agent is None:
            raise RuntimeError(
                "RuntimeSession.initialize() requires an agent-backed session."
            )
        self.env = env
        self.game_no = game_no
        self.info = dict(info or {})
        self.adapter = adapter
        self.agent.set_environment(
            env,
            obs,
            self.info,
            game_no,
            adapter=adapter,
        )
        self.initialized = True

    @property
    def initial_message(self) -> str:
        return str(getattr(self.agent, "initial_message", ""))

    @property
    def log_paths(self) -> dict[str, str]:
        return dict(getattr(self.agent, "log_paths", {}) or {})

    @property
    def group_chat(self) -> Any:
        return (
            getattr(self.agent, "group_chat", None) if self.agent is not None else None
        )

    @property
    def group_chat_manager(self) -> Any:
        return (
            getattr(self.agent, "group_chat_manager", None)
            if self.agent is not None
            else None
        )

    def run_chat(self, message: str):
        if self.agent is None:
            raise RuntimeError(
                "RuntimeSession.run_chat() requires an agent-backed session."
            )
        return self.agent.run_chat(message)

    def bind_adapter(
        self,
        adapter: Any,
        *,
        capabilities: AdapterCapabilities | None = None,
        game_no: int | None = None,
        info: dict[str, Any] | None = None,
        mode: str | None = None,
        escalation_policy: str | None = None,
    ) -> None:
        self.adapter = adapter
        self.capabilities = (
            capabilities or getattr(adapter, "get_v2_capabilities", lambda: None)()
        )
        if game_no is not None:
            self.game_no = game_no
        if info is not None:
            self.info = dict(info)
        if mode is not None:
            self.mode = mode
        if escalation_policy is not None:
            self.escalation_policy = escalation_policy
        self.initialized = True

    def observe_adapter_event(
        self,
        event: AdapterEvent,
        *,
        capabilities: AdapterCapabilities | None = None,
    ) -> None:
        effective_capabilities = (
            capabilities
            or self.capabilities
            or (
                getattr(self.adapter, "get_v2_capabilities", lambda: None)()
                if self.adapter is not None
                else None
            )
        )
        if effective_capabilities is None:
            raise RuntimeError(
                "RuntimeSession.observe_adapter_event() requires adapter capabilities."
            )
        self.capabilities = effective_capabilities
        self._latest_event = event
        if self.world_model is None:
            self.world_model = WorldModel.from_event(
                event,
                capabilities=effective_capabilities,
            )
            self.controller.memory.record_event(
                self.world_model.to_snapshot(),
                event,
                current_option=None,
            )
            return
        self.controller.observe(self.world_model, event)

    def should_escalate(
        self, *, mode_override: str | None = None
    ) -> tuple[bool, str | None]:
        effective_mode = mode_override or self.mode
        if self.escalation_policy == "force_protocol":
            return False, None
        if self.escalation_policy == "force_deliberative":
            return True, "forced_deliberative_policy"
        if effective_mode == "deliberative":
            return True, "deliberative_mode_requested"
        if self.protocol_revision_streak >= 2:
            return True, "revision_streak_exceeded"
        if self.world_model is None:
            return False, None
        snapshot = self.world_model.to_snapshot()
        if bool(snapshot.metadata.get("escalation_required", False)):
            return True, "protocol_escalation_required"
        if bool(snapshot.metadata.get("repair_pending", False)) and int(
            snapshot.metadata.get("repair_attempt_count", 0)
        ) >= int(snapshot.metadata.get("max_repair_attempts", 2)):
            return True, "repair_attempts_exhausted"
        return False, None

    def _build_deliberative_stub_decision(self, reason: str | None) -> RuntimeDecision:
        self.last_escalation_reason = reason
        self._latest_decision = RuntimeDecision(
            suggested_action=None,
            option_family=None,
            repair_operator=None,
            planner_stop_reason="deliberative_stub",
            revision_required=True,
            mode_used="deliberative",
            escalated=True,
            escalation_reason=reason,
            metadata={
                "planner_notes": ["Protocol engine escalated to deliberative mode."],
                "stub": True,
            },
        )
        return self._latest_decision

    def decide(self, *, mode_override: str | None = None) -> RuntimeDecision:
        should_escalate, escalation_reason = self.should_escalate(
            mode_override=mode_override
        )
        if should_escalate:
            return self._build_deliberative_stub_decision(escalation_reason)
        if self.world_model is None:
            raise RuntimeError(
                "RuntimeSession.decide() requires at least one observed adapter event."
            )
        controller_step = self.controller.step(self.world_model)
        snapshot = self.world_model.to_snapshot()
        option = controller_step.planner_directive.option
        if snapshot.revision_required:
            self.protocol_revision_streak += 1
        else:
            self.protocol_revision_streak = 0
        self.last_escalation_reason = None
        self._latest_decision = RuntimeDecision(
            suggested_action=controller_step.execution_step.action_label,
            option_family=option.family if option is not None else None,
            repair_operator=(
                snapshot.repair_directive.operator
                if snapshot.repair_directive is not None
                else None
            ),
            planner_stop_reason=controller_step.planner_directive.stop_reason
            or controller_step.execution_step.stop_reason,
            revision_required=snapshot.revision_required,
            mode_used="protocol",
            escalated=False,
            escalation_reason=None,
            metadata={
                "planner_notes": list(
                    controller_step.planner_directive.planner_notes[:3]
                ),
                "continue_current_option": (
                    controller_step.planner_directive.continue_current_option
                ),
                "execution_stop_reason": controller_step.execution_step.stop_reason,
                "repair_pending": snapshot.metadata.get("repair_pending", False),
                "escalation_required": snapshot.metadata.get(
                    "escalation_required", False
                ),
            },
        )
        return self._latest_decision

    def latest_decision(self) -> RuntimeDecision:
        if self._latest_decision is not None:
            return self._latest_decision
        advisory = dict(getattr(self.agent, "_v2_runtime_advisory", {}) or {})
        return RuntimeDecision(
            suggested_action=advisory.get("suggested_action"),
            option_family=advisory.get("option_family"),
            repair_operator=advisory.get("repair_operator"),
            planner_stop_reason=advisory.get("planner_stop_reason"),
            revision_required=bool(advisory.get("revision_required", False)),
            mode_used=str(advisory.get("mode_used", "protocol")),
            escalated=bool(advisory.get("escalated", False)),
            escalation_reason=advisory.get("escalation_reason"),
            metadata=advisory,
        )

    def escalate(self, *, reason: str | None = None) -> RuntimeDecision:
        return self._build_deliberative_stub_decision(reason or "manual_escalation")

    def _build_protocol_summary(self) -> dict[str, Any]:
        if self.world_model is None:
            return {}
        snapshot = self.world_model.to_snapshot()
        contradiction_categories = [
            record.category for record in snapshot.contradictions[-3:]
        ]
        contradiction_debt = float(snapshot.metadata.get("contradiction_debt", 0.0))
        repair_pending = bool(snapshot.metadata.get("repair_pending", False))
        repair_attempt_count = int(snapshot.metadata.get("repair_attempt_count", 0))
        max_repair_attempts = int(snapshot.metadata.get("max_repair_attempts", 0))
        escalation_required = bool(snapshot.metadata.get("escalation_required", False))
        approaching_escalation = escalation_required or bool(
            snapshot.revision_required
            or self.protocol_revision_streak >= 1
            or contradiction_debt >= 2.0
            or (
                repair_pending
                and repair_attempt_count >= max(max_repair_attempts - 1, 1)
            )
        )
        recommended_mode = "deliberative" if approaching_escalation else "protocol"
        recommended_mode_reason = "stable_protocol_state"
        if escalation_required:
            recommended_mode_reason = "escalation_required"
        elif snapshot.revision_required:
            recommended_mode_reason = "revision_required"
        elif repair_pending and repair_attempt_count >= max(max_repair_attempts - 1, 1):
            recommended_mode_reason = "repair_budget_nearly_exhausted"
        elif contradiction_debt >= 2.0:
            recommended_mode_reason = "contradiction_debt_elevated"
        elif self.protocol_revision_streak >= 1:
            recommended_mode_reason = "protocol_revision_streak_active"
        return {
            "contradiction_debt": contradiction_debt,
            "recent_contradiction_count": len(contradiction_categories),
            "recent_contradiction_categories": contradiction_categories,
            "revision_required": snapshot.revision_required,
            "repair_cycle_active": bool(
                snapshot.metadata.get("repair_cycle_active", False)
            ),
            "repair_pending": repair_pending,
            "repair_attempt_count": repair_attempt_count,
            "max_repair_attempts": max_repair_attempts,
            "repair_operator": (
                snapshot.repair_directive.operator
                if snapshot.repair_directive is not None
                else None
            ),
            "escalation_required": escalation_required,
            "protocol_revision_streak": self.protocol_revision_streak,
            "no_progress_streak": int(snapshot.metadata.get("no_progress_streak", 0)),
            "approaching_escalation": approaching_escalation,
            "recommended_mode": recommended_mode,
            "recommended_mode_reason": recommended_mode_reason,
        }

    def record_comparison_event(
        self,
        *,
        source: str,
        protocol_hint: dict[str, object],
        disagreement_summary: dict[str, object],
        decision: RuntimeDecision | None,
        backend: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        protocol_hint_payload = dict(protocol_hint.get("protocol_hint", {}))
        event = {
            "event_index": len(self.comparison_events) + 1,
            "source": source,
            "backend": backend,
            "mode_used": decision.mode_used if decision is not None else None,
            "escalated": decision.escalated if decision is not None else False,
            "escalation_reason": (
                decision.escalation_reason if decision is not None else None
            ),
            "suggested_action": (
                decision.suggested_action if decision is not None else None
            ),
            "repair_operator": (
                decision.repair_operator if decision is not None else None
            ),
            "protocol_hint": protocol_hint_payload,
            "disagreement_summary": dict(disagreement_summary),
            "metadata": dict(metadata or {}),
        }
        self.comparison_events.append(event)

    def _build_comparison_telemetry(self) -> dict[str, Any]:
        if not self.comparison_events:
            return {
                "event_count": 0,
                "available_count": 0,
                "alignment_counts": {},
                "source_counts": {},
                "recent_events": [],
            }
        alignment_counts: dict[str, int] = {}
        source_counts: dict[str, int] = {}
        available_count = 0
        for event in self.comparison_events:
            source = str(event.get("source", "unknown"))
            source_counts[source] = source_counts.get(source, 0) + 1
            disagreement = event.get("disagreement_summary", {})
            if isinstance(disagreement, dict) and disagreement.get("available"):
                available_count += 1
                alignment = str(
                    disagreement.get("backend_alignment", "alignment_unknown")
                )
                alignment_counts[alignment] = alignment_counts.get(alignment, 0) + 1
        return {
            "event_count": len(self.comparison_events),
            "available_count": available_count,
            "alignment_counts": alignment_counts,
            "source_counts": source_counts,
            "recent_events": list(self.comparison_events[-5:]),
        }

    def snapshot(self) -> RuntimeSessionSnapshot:
        adapter = self.adapter or getattr(self.agent, "adapter", None)
        task = getattr(adapter, "task", None)
        observation = getattr(adapter, "observation", "") if adapter is not None else ""
        admissible_actions = (
            list(getattr(self.agent, "admissible_actions", []) or [])
            if self.agent is not None
            else list(getattr(adapter, "admissible_actions", []) or [])
        )
        return RuntimeSessionSnapshot(
            session_id=self.session_id,
            game_no=self.game_no,
            task=task,
            observation=observation,
            admissible_actions=admissible_actions,
            runtime_advisory=(
                dict(getattr(self.agent, "_v2_runtime_advisory", {}) or {})
                if self.agent is not None
                else (self._latest_decision.metadata if self._latest_decision else {})
            ),
            protocol_summary=self._build_protocol_summary(),
            comparison_telemetry=self._build_comparison_telemetry(),
            log_paths=self.log_paths,
            mode=self.mode,
            escalation_policy=self.escalation_policy,
            escalation_state={
                "protocol_revision_streak": self.protocol_revision_streak,
                "last_escalation_reason": self.last_escalation_reason,
                "latest_mode_used": (
                    self._latest_decision.mode_used if self._latest_decision else None
                ),
                "latest_escalated": (
                    self._latest_decision.escalated
                    if self._latest_decision is not None
                    else False
                ),
            },
        )
