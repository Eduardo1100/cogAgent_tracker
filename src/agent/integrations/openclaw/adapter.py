from __future__ import annotations

from typing import Any

from src.agent.env_adapter import _infer_operator_family
from src.agent.integrations.openclaw.protocol import OpenClawObservationRequest
from src.agent.v2.types import (
    ActionResult,
    AdapterCapabilities,
    AdapterEvent,
    FrontierDelta,
)


def _event_success(event: AdapterEvent) -> bool:
    status = event.status_delta or {}
    metadata = event.metadata or {}
    return bool(
        status.get("success")
        or status.get("won")
        or status.get("has_won")
        or metadata.get("success")
        or metadata.get("won")
        or metadata.get("has_won")
    )


class OpenClawAdapter:
    def __init__(
        self,
        *,
        task_text: str,
        capabilities: AdapterCapabilities | None = None,
        initial_event: AdapterEvent | None = None,
        session_metadata: dict[str, Any] | None = None,
    ):
        self._task_text = task_text
        self._capabilities = capabilities or AdapterCapabilities(
            adapter_name="openclaw",
            observation_mode="ui",
            supports_ui_context=True,
            supports_status_delta=True,
            supports_reward_delta=True,
            supports_operator_candidates=True,
        )
        self._session_metadata = dict(session_metadata or {})
        self._last_action_text: str | None = None
        self._last_action_executed: bool | None = None
        self._reward_delta = 0.0
        self._status_delta: dict[str, float | int | str | bool] = {}
        self._has_won = False
        self._current_event = initial_event or AdapterEvent(
            step_index=0,
            task_text=task_text,
            raw_observation="",
            normalized_observation="",
            action_result=ActionResult(),
            frontier_delta=FrontierDelta(),
            metadata={"adapter_name": "openclaw", **self._session_metadata},
        )
        self._initial_observation = (
            self._current_event.normalized_observation
            or self._current_event.raw_observation
            or ""
        )
        self._sync_from_event(self._current_event)

    def _sync_from_event(self, event: AdapterEvent) -> None:
        self._current_event = event
        self._task_text = event.task_text or self._task_text
        self._reward_delta = float(event.reward_delta or 0.0)
        self._status_delta = dict(event.status_delta or {})
        self._has_won = _event_success(event)

    @property
    def observation(self) -> str:
        return (
            self._current_event.normalized_observation
            or self._current_event.raw_observation
        )

    @property
    def admissible_actions(self) -> list[str]:
        return [
            candidate.action_label
            for candidate in self._current_event.operator_candidates
            if candidate.action_label
        ]

    @property
    def has_won(self) -> bool:
        return self._has_won

    @property
    def task(self) -> str:
        return self._task_text

    @property
    def initial_observation(self) -> str:
        return self._initial_observation

    def ingest_observation(
        self, observation: OpenClawObservationRequest | AdapterEvent | dict[str, Any]
    ) -> AdapterEvent:
        if isinstance(observation, AdapterEvent):
            event = observation
        elif isinstance(observation, OpenClawObservationRequest):
            event = observation.to_adapter_event()
        else:
            event = OpenClawObservationRequest.from_dict(observation).to_adapter_event()
        self._sync_from_event(event)
        return event

    def apply_action_result(
        self,
        *,
        action_text: str | None = None,
        action_executed: bool | None = None,
        reward_delta: float | None = None,
        status_delta: dict[str, float | int | str | bool] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        self._last_action_text = action_text
        self._last_action_executed = action_executed
        if reward_delta is not None:
            self._reward_delta = float(reward_delta)
        if status_delta is not None:
            self._status_delta = dict(status_delta)
        if metadata:
            self._session_metadata.update(metadata)

    def step(self, action: str) -> str:
        raise RuntimeError(
            "OpenClawAdapter does not execute actions directly. "
            "Use an external controller to apply action results."
        )

    def set_observation(self, msg: str) -> None:
        self._current_event = AdapterEvent(
            step_index=self._current_event.step_index,
            task_text=self._task_text,
            raw_observation=msg,
            normalized_observation=msg,
            action_result=self._current_event.action_result,
            operator_candidates=self._current_event.operator_candidates,
            entity_updates=self._current_event.entity_updates,
            relation_updates=self._current_event.relation_updates,
            reward_delta=self._current_event.reward_delta,
            status_delta=self._current_event.status_delta,
            frontier_delta=self._current_event.frontier_delta,
            novelty_signals=self._current_event.novelty_signals,
            spatial_context=self._current_event.spatial_context,
            ui_context=self._current_event.ui_context,
            metadata=self._current_event.metadata,
        )

    def infer_task_type(self) -> int | None:
        return None

    def count_inadmissible_actions(self, log_path: str) -> int:
        try:
            with open(log_path) as f:
                return sum(
                    1
                    for line in f
                    if "unsupported openclaw action" in line.lower()
                    or "inadmissible" in line.lower()
                )
        except Exception:
            return 0

    def get_v2_capabilities(self) -> AdapterCapabilities:
        return self._capabilities

    def build_v2_event(
        self,
        *,
        step_index: int,
        action_text: str | None = None,
        action_executed: bool | None = None,
        reward_delta: float | None = None,
        status_delta: dict[str, float | int | str | bool] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> AdapterEvent:
        base = self._current_event
        effective_action_text = (
            action_text or self._last_action_text or base.action_result.action_text
        )
        effective_action_executed = (
            action_executed
            if action_executed is not None
            else (
                self._last_action_executed
                if self._last_action_executed is not None
                else base.action_result.action_executed
            )
        )
        event_metadata = dict(base.metadata or {})
        event_metadata.setdefault("adapter_name", "openclaw")
        if self._session_metadata:
            event_metadata.update(self._session_metadata)
        if metadata:
            event_metadata.update(metadata)
        return AdapterEvent(
            step_index=step_index,
            task_text=self._task_text,
            raw_observation=base.raw_observation,
            normalized_observation=base.normalized_observation,
            action_result=ActionResult(
                action_text=effective_action_text,
                action_executed=effective_action_executed,
                operator_family=(
                    _infer_operator_family(effective_action_text)
                    if effective_action_text
                    else None
                ),
                failure_reason=base.action_result.failure_reason,
                changed_observation=base.action_result.changed_observation,
            ),
            operator_candidates=list(base.operator_candidates),
            entity_updates=list(base.entity_updates),
            relation_updates=list(base.relation_updates),
            reward_delta=reward_delta
            if reward_delta is not None
            else self._reward_delta,
            status_delta=status_delta or dict(self._status_delta),
            frontier_delta=base.frontier_delta,
            novelty_signals=list(base.novelty_signals),
            spatial_context=base.spatial_context,
            ui_context=base.ui_context,
            metadata=event_metadata,
        )
