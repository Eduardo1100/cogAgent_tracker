from __future__ import annotations

from dataclasses import dataclass

from src.agent.v2.types import OperatorCandidate, OptionContract, WorldModelSnapshot


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
    if option.target_signature:
        if candidate.action_label == option.target_signature:
            score += 25
        if option.target_signature in candidate.target_ids:
            score += 25
        if option.target_signature in candidate.action_label:
            score += 12
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
