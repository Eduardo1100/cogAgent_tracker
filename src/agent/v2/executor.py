from __future__ import annotations

from dataclasses import dataclass

from src.agent.v2.types import (
    OperatorCandidate,
    OptionContract,
    RepairDirective,
    WorldModelSnapshot,
)


def _ui_context_maps(
    snapshot: WorldModelSnapshot,
) -> tuple[str | None, dict[str, dict[str, object]]]:
    ui_context = snapshot.metadata.get("ui_context", {})
    focused_element_id = ui_context.get("focused_element_id")
    visible_elements = ui_context.get("visible_elements", [])
    element_map: dict[str, dict[str, object]] = {}
    for element in visible_elements:
        element_id = element.get("element_id")
        if element_id:
            element_map[str(element_id)] = dict(element)
    return (
        str(focused_element_id) if focused_element_id else None,
        element_map,
    )


def _input_state_map(snapshot: WorldModelSnapshot) -> dict[str, object]:
    input_state = snapshot.metadata.get("input_state", {})
    if not input_state:
        ui_context = snapshot.metadata.get("ui_context", {})
        if isinstance(ui_context, dict):
            input_state = ui_context.get("input_state", {})
    return dict(input_state) if isinstance(input_state, dict) else {}


def _score_ui_preconditions(
    candidate: OperatorCandidate,
    snapshot: WorldModelSnapshot,
) -> int:
    focused_element_id, element_map = _ui_context_maps(snapshot)
    input_state = _input_state_map(snapshot)
    active_input_target = input_state.get("active_input_target")
    text_entry_admissible = bool(input_state.get("text_entry_admissible"))
    submit_action_available = bool(input_state.get("submit_action_available"))
    input_detour = bool(input_state.get("input_detour"))
    escape_actions = {
        str(action) for action in input_state.get("escape_actions", []) if action
    }
    if candidate.family == "ui_type":
        if not candidate.target_ids:
            return -90
        target_id = candidate.target_ids[0]
        if not text_entry_admissible:
            return -140
        if active_input_target == target_id or focused_element_id == target_id:
            return 35
        return -120
    if candidate.family == "ui_click" and candidate.target_ids:
        target_id = candidate.target_ids[0]
        element = element_map.get(target_id, {})
        attrs = element.get("attributes", {}) if isinstance(element, dict) else {}
        if input_detour and candidate.action_label in escape_actions:
            return 35
        if bool(attrs.get("editable")) and focused_element_id != target_id:
            return 20
        if (
            bool(attrs.get("editable"))
            and text_entry_admissible
            and (active_input_target == target_id or focused_element_id == target_id)
        ):
            return -10
    if candidate.family == "ui_navigation":
        if input_detour and candidate.action_label in escape_actions:
            return 40
    if candidate.family == "ui_submit":
        if submit_action_available:
            return 18
        return -20
    return 0


def _score_operator(
    candidate: OperatorCandidate,
    option: OptionContract,
    *,
    snapshot: WorldModelSnapshot,
    failed_actions: set[str],
    executed_actions: list[str],
) -> tuple[int, int, int, str]:
    if candidate.action_label in failed_actions:
        return (-1000, 0, 0, candidate.action_label)
    blocked_actions = {
        str(action)
        for action in snapshot.metadata.get("blocked_action_labels", [])
        if action
    }
    contradiction_actions = {
        str(action)
        for action in snapshot.metadata.get("recent_contradiction_actions", [])
        if action
    }
    score = 0
    if option.family == "explore_frontier" and candidate.family == "relocation":
        score += 30
    if option.family == "inspect_novelty" and candidate.family == "inspect":
        score += 25
    if option.family == "manipulate_target" and candidate.family.startswith("ui_"):
        score += 30
    if option.family == "verify_outcome" and candidate.family in {
        "interaction",
        "tool_application",
        "inspect",
    }:
        score += 20
    if option.family == "pursue_reward" and candidate.family in {
        "interaction",
        "inventory",
        "relocation",
    }:
        score += 18
    if option.family == "commit_transition" and candidate.action_label in {
        "go up",
        "go down",
    }:
        score += 30
    if option.family == "recover_from_failure":
        if candidate.family == "inspect":
            score += 28
        elif candidate.family == "interaction":
            score += 16
        elif candidate.family == "relocation":
            score += 12
        if candidate.action_label == "wait":
            score -= 18
    if option.target_signature:
        if candidate.action_label == option.target_signature:
            score += 25
        if option.target_signature in candidate.target_ids:
            score += 25
        if option.target_signature in candidate.action_label:
            score += 12
    if snapshot.metadata.get("ui_context"):
        score += _score_ui_preconditions(candidate, snapshot)
    if candidate.action_label in blocked_actions:
        score -= 120
    if candidate.action_label in contradiction_actions:
        score -= 80
    recency_penalty = 5 if candidate.action_label in executed_actions[-3:] else 0
    score -= recency_penalty
    return (score, -recency_penalty, len(candidate.target_ids), candidate.action_label)


@dataclass(frozen=True)
class ExecutionStep:
    option: OptionContract
    operator: OperatorCandidate | None
    action_label: str | None
    stop_reason: str


@dataclass(frozen=True)
class DeterministicExecutor:
    def select_repair_step(
        self,
        snapshot: WorldModelSnapshot,
        option: OptionContract,
        repair_directive: RepairDirective,
        *,
        failed_actions: set[str] | None = None,
        executed_actions: list[str] | None = None,
    ) -> ExecutionStep:
        failed = failed_actions or set()
        executed = executed_actions or []
        invalidated = set(repair_directive.invalidated_actions)

        def _score_repair_candidate(
            candidate: OperatorCandidate,
        ) -> tuple[int, int, str]:
            if candidate.action_label in failed:
                return (-1000, 0, candidate.action_label)
            if candidate.action_label in invalidated:
                return (-900, 0, candidate.action_label)

            score = 0
            operator = repair_directive.operator
            preferred = set(repair_directive.preferred_families)

            if candidate.family in preferred:
                score += 18

            if snapshot.metadata.get("ui_context"):
                score += _score_ui_preconditions(candidate, snapshot)

            if operator == "repair_topology":
                if candidate.action_label in {"search", "look"}:
                    score += 30
                elif candidate.family == "inspect":
                    score += 22
                elif candidate.family == "relocation":
                    score += 10
            elif operator == "repair_transition_model":
                if candidate.action_label in {"search", "look"}:
                    score += 24
                elif candidate.family == "interaction":
                    score += 18
                elif candidate.family == "inspect":
                    score += 16
            elif operator == "repair_target_binding":
                if candidate.family == "ui_navigation":
                    score += 30
                elif candidate.family == "inspect":
                    score += 24
                elif candidate.family == "ui_select":
                    score += 22
                elif candidate.family.startswith("ui_"):
                    score += 20
            elif operator == "repair_operator_set":
                if candidate.family == "inspect":
                    score += 20
                elif candidate.family == "ui_navigation":
                    score += 18
                elif candidate.family == "interaction":
                    score += 16
            elif operator == "repair_input_focus":
                if candidate.family == "ui_click":
                    score += 28
                elif candidate.family == "ui_select":
                    score += 20
            elif operator == "repair_input_surface":
                if candidate.family == "ui_navigation":
                    score += 28
                elif candidate.family == "ui_click":
                    score += 22
                elif candidate.family == "ui_submit":
                    score += 14
            elif operator == "recover_task_surface":
                if candidate.family == "ui_navigation":
                    score += 32
                elif candidate.family == "inspect":
                    score += 14

            blocked_actions = {
                str(action)
                for action in snapshot.metadata.get("blocked_action_labels", [])
                if action
            }
            contradiction_actions = {
                str(action)
                for action in snapshot.metadata.get("recent_contradiction_actions", [])
                if action
            }
            if candidate.action_label in blocked_actions:
                score -= 120
            if candidate.action_label in contradiction_actions:
                score -= 80
            if candidate.action_label == "wait":
                score -= 25

            recency_penalty = 5 if candidate.action_label in executed[-3:] else 0
            score -= recency_penalty
            return (score, -recency_penalty, candidate.action_label)

        ranked = sorted(
            snapshot.operator_candidates,
            key=_score_repair_candidate,
            reverse=True,
        )
        chosen = (
            ranked[0]
            if ranked and _score_repair_candidate(ranked[0])[0] > -1000
            else None
        )
        if chosen is None:
            return ExecutionStep(
                option=option,
                operator=None,
                action_label=None,
                stop_reason="no_repair_operator_available",
            )
        return ExecutionStep(
            option=option,
            operator=chosen,
            action_label=chosen.action_label,
            stop_reason="repair_operator_selected",
        )

    def select_next_step(
        self,
        snapshot: WorldModelSnapshot,
        option: OptionContract,
        *,
        failed_actions: set[str] | None = None,
        executed_actions: list[str] | None = None,
    ) -> ExecutionStep:
        failed = failed_actions or set()
        executed = executed_actions or []
        ranked = sorted(
            snapshot.operator_candidates,
            key=lambda candidate: _score_operator(
                candidate,
                option,
                snapshot=snapshot,
                failed_actions=failed,
                executed_actions=executed,
            ),
            reverse=True,
        )
        chosen = (
            ranked[0]
            if ranked
            and _score_operator(
                ranked[0],
                option,
                snapshot=snapshot,
                failed_actions=failed,
                executed_actions=executed,
            )[0]
            > -1000
            else None
        )
        if chosen is None:
            return ExecutionStep(
                option=option,
                operator=None,
                action_label=None,
                stop_reason="no_operator_available",
            )
        return ExecutionStep(
            option=option,
            operator=chosen,
            action_label=chosen.action_label,
            stop_reason="operator_selected",
        )
