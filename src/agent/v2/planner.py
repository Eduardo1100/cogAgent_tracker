from __future__ import annotations

from dataclasses import dataclass

from src.agent.v2.types import (
    GoalRecord,
    OperatorCandidate,
    OptionContract,
    PlannerDirective,
    WorldModelSnapshot,
)


def _goal_text(goals: list[GoalRecord]) -> str:
    if not goals:
        return "Make progress toward the task."
    return goals[0].description


def _has_family(operator_candidates: list[OperatorCandidate], *families: str) -> bool:
    allowed = set(families)
    return any(candidate.family in allowed for candidate in operator_candidates)


def _pick_ui_target(snapshot: WorldModelSnapshot) -> str | None:
    ui_context = snapshot.metadata.get("ui_context", {})
    focused = ui_context.get("focused_element_id")
    if focused:
        return str(focused)
    visible_elements = ui_context.get("visible_elements", [])
    if visible_elements:
        first = visible_elements[0]
        element_id = first.get("element_id")
        if element_id:
            return str(element_id)
    for candidate in snapshot.operator_candidates:
        if candidate.target_ids:
            return candidate.target_ids[0]
    return None


def _pick_frontier_target(snapshot: WorldModelSnapshot) -> str | None:
    frontier = snapshot.frontier_summary.get("frontier_nodes", [])
    if frontier:
        return str(frontier[0])
    opened = snapshot.frontier_summary.get("opened_nodes", [])
    if opened:
        return str(opened[0])
    return None


def _pick_entity_target(snapshot: WorldModelSnapshot) -> str | None:
    if snapshot.entities:
        entity_id = snapshot.entities[0].get("entity_id")
        if entity_id:
            return str(entity_id)
    return None


def _preferred_family_from_memory(snapshot: WorldModelSnapshot) -> str | None:
    applicability: list[str] = []
    for cue in snapshot.memory_cues:
        applicability.extend(cue.applicability)
    for family in [
        "manipulate_target",
        "explore_frontier",
        "inspect_novelty",
        "pursue_reward",
        "verify_outcome",
        "recover_from_failure",
    ]:
        if family in applicability:
            return family
    return None


def build_option_contract(
    *,
    family: str,
    objective: str,
    target_signature: str | None = None,
) -> OptionContract:
    expected_outcomes_by_family = {
        "explore_frontier": ["frontier_expansion", "novel_observation"],
        "inspect_novelty": ["grounding_progress", "novel_observation"],
        "manipulate_target": ["ui_state_change", "goal_progress"],
        "commit_transition": ["environment_transition", "goal_progress"],
        "pursue_reward": ["reward_gain", "goal_progress"],
        "verify_outcome": ["status_change", "goal_progress"],
        "recover_from_failure": ["operator_recovery", "novel_observation"],
    }
    interrupt_conditions_by_family = {
        "explore_frontier": ["frontier_exhausted", "contradiction", "blocked"],
        "inspect_novelty": ["novelty_resolved", "contradiction"],
        "manipulate_target": [
            "ui_scope_changed",
            "target_unavailable",
            "contradiction",
        ],
        "commit_transition": ["target_unavailable", "contradiction"],
        "pursue_reward": ["reward_absent", "contradiction"],
        "verify_outcome": ["evidence_resolved", "contradiction"],
        "recover_from_failure": ["recovery_complete", "contradiction"],
    }
    budgets = {
        "explore_frontier": (5, 2, "light"),
        "inspect_novelty": (3, 2, "light"),
        "manipulate_target": (3, 1, "light"),
        "commit_transition": (2, 1, "light"),
        "pursue_reward": (3, 1, "light"),
        "verify_outcome": (2, 1, "light"),
        "recover_from_failure": (2, 2, "full"),
    }
    progress_budget, failure_budget, reasoning_budget = budgets.get(
        family, (3, 1, "light")
    )
    return OptionContract(
        family=family,
        objective=objective,
        target_signature=target_signature,
        expected_outcomes=expected_outcomes_by_family.get(family, ["goal_progress"]),
        progress_budget=progress_budget,
        failure_budget=failure_budget,
        termination_conditions=["goal_progress", "target_resolved"],
        interrupt_conditions=interrupt_conditions_by_family.get(
            family, ["contradiction"]
        ),
        reasoning_budget=reasoning_budget,
    )


def can_continue_option(
    snapshot: WorldModelSnapshot, option: OptionContract | None
) -> tuple[bool, str]:
    if option is None:
        return False, "no_active_option"
    if option.family == "explore_frontier":
        if snapshot.frontier_summary.get("frontier_nodes"):
            return True, "frontier_available"
        return False, "frontier_exhausted"
    if option.family == "manipulate_target":
        target = option.target_signature
        if not target:
            return False, "target_missing"
        ui_context = snapshot.metadata.get("ui_context", {})
        focused = ui_context.get("focused_element_id")
        visible_elements = ui_context.get("visible_elements", [])
        visible_ids = {
            str(element.get("element_id"))
            for element in visible_elements
            if element.get("element_id")
        }
        if target == focused or target in visible_ids:
            return True, "target_still_visible"
        return False, "target_unavailable"
    if option.family in {"inspect_novelty", "verify_outcome"}:
        if snapshot.entities or snapshot.operator_candidates:
            return True, "evidence_still_available"
        return False, "evidence_missing"
    if option.family in {"commit_transition", "pursue_reward"}:
        if snapshot.operator_candidates:
            return True, "operators_available"
        return False, "operators_missing"
    return False, "unknown_family"


@dataclass(frozen=True)
class SparsePlanner:
    default_reasoning_tier: str = "light"

    def plan(
        self,
        snapshot: WorldModelSnapshot,
        *,
        current_option: OptionContract | None = None,
    ) -> PlannerDirective:
        can_continue, reason = can_continue_option(snapshot, current_option)
        if can_continue and current_option is not None:
            return PlannerDirective(
                option=current_option,
                reasoning_tier=current_option.reasoning_budget,
                continue_current_option=True,
                planner_notes=[f"Continuing active option because {reason}."],
                stop_reason=reason,
            )

        option_family = "inspect_novelty"
        target_signature: str | None = None
        notes: list[str] = []
        goal_text = _goal_text(snapshot.goals)
        memory_preference = _preferred_family_from_memory(snapshot)

        if memory_preference == "manipulate_target" and snapshot.metadata.get(
            "ui_context"
        ):
            option_family = "manipulate_target"
            target_signature = _pick_ui_target(snapshot)
            notes.append("Memory cues favor targeted UI manipulation.")
        elif memory_preference == "explore_frontier" and snapshot.frontier_summary.get(
            "frontier_nodes"
        ):
            option_family = "explore_frontier"
            target_signature = _pick_frontier_target(snapshot)
            notes.append("Memory cues favor frontier expansion.")
        elif memory_preference == "inspect_novelty" and snapshot.entities:
            option_family = "inspect_novelty"
            target_signature = _pick_entity_target(snapshot)
            notes.append("Memory cues favor inspection of grounded entities.")
        elif snapshot.metadata.get("ui_context") and _has_family(
            snapshot.operator_candidates, "ui_click", "ui_type", "ui_select"
        ):
            option_family = "manipulate_target"
            target_signature = _pick_ui_target(snapshot)
            notes.append("UI context is present; prefer targeted manipulation.")
        elif snapshot.frontier_summary.get("frontier_nodes") and _has_family(
            snapshot.operator_candidates, "relocation"
        ):
            option_family = "explore_frontier"
            target_signature = _pick_frontier_target(snapshot)
            notes.append("Frontier nodes are available; prefer exploration.")
        elif snapshot.metadata.get("last_status_delta", {}).get("S", 0):
            option_family = "pursue_reward"
            target_signature = _pick_entity_target(snapshot)
            notes.append("Recent score/reward change suggests a reward-bearing route.")
        elif snapshot.entities:
            option_family = "inspect_novelty"
            target_signature = _pick_entity_target(snapshot)
            notes.append("Grounded entities are available; inspect before expanding.")
        elif _has_family(
            snapshot.operator_candidates, "interaction", "tool_application"
        ):
            option_family = "verify_outcome"
            notes.append("Interaction operators are available without clear grounding.")
        elif snapshot.uncertainty:
            option_family = "recover_from_failure"
            notes.append("Uncertainty is elevated; allocate recovery option.")
        else:
            notes.append("Falling back to generic inspection.")

        option = build_option_contract(
            family=option_family,
            objective=goal_text,
            target_signature=target_signature,
        )
        memory_queries = [f"{option_family}:{goal_text}"]
        return PlannerDirective(
            option=option,
            reasoning_tier=option.reasoning_budget,
            continue_current_option=False,
            planner_notes=notes,
            memory_queries=memory_queries,
            stop_reason=reason if current_option is not None else "new_option",
        )
