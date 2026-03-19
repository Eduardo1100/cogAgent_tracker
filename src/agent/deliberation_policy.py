from __future__ import annotations

import re
from dataclasses import dataclass

from src.agent.decision_state import DecisionState


@dataclass(frozen=True)
class DeliberationDecision:
    should_deliberate: bool
    reason: str
    signal_flags: dict[str, bool]


@dataclass(frozen=True)
class SequenceInterruptDecision:
    should_interrupt: bool
    reason: str
    signal_flags: dict[str, bool]


def evaluate_deliberation_policy(
    *,
    state: DecisionState,
    belief_content: str,
    uncertainty_pattern: re.Pattern[str],
    forced_interval: int,
) -> DeliberationDecision:
    normalized_belief = belief_content.lower()
    grounding = state.grounding_state
    option_state = state.option_state
    progress = state.progress_state
    uncertainty = state.uncertainty_state

    signal_flags = {
        "textual_uncertainty": bool(
            uncertainty_pattern.search(normalized_belief)
            or "no observation" in normalized_belief
        ),
        "forced_interval": progress.actions_since_last_thinking >= forced_interval,
        "referent_ambiguity": bool(grounding.referent_resolution),
        "recent_failures": bool(option_state.recently_failed_actions),
        "interrupted_option": progress.last_burst_stop_reason
        not in {"", "sequence_complete", "single_action"},
        "state_uncertainty": uncertainty.admissible_actions_unchanged is False
        and progress.last_burst_stop_reason == "observation_changed",
        "remote_room_signal": bool(grounding.remote_room_signal),
    }

    if signal_flags["textual_uncertainty"]:
        return DeliberationDecision(True, "textual_uncertainty", signal_flags)
    if signal_flags["referent_ambiguity"]:
        return DeliberationDecision(True, "referent_ambiguity", signal_flags)
    if signal_flags["recent_failures"]:
        return DeliberationDecision(True, "recent_failures", signal_flags)
    if signal_flags["interrupted_option"]:
        return DeliberationDecision(True, "interrupted_option", signal_flags)
    if signal_flags["forced_interval"]:
        return DeliberationDecision(True, "forced_interval", signal_flags)
    if signal_flags["state_uncertainty"]:
        return DeliberationDecision(True, "state_uncertainty", signal_flags)
    if signal_flags["remote_room_signal"]:
        return DeliberationDecision(True, "remote_room_signal", signal_flags)
    return DeliberationDecision(False, "stable_progress", signal_flags)


def evaluate_sequence_interrupt_policy(
    *,
    previous_state: DecisionState,
    current_state: DecisionState,
    action_family: str,
    observation_changed: bool,
    projected_stagnation_steps: int = 0,
    observation_novelty_gain: bool = False,
    local_revisitation: bool = False,
    projected_progress_debt: int = 0,
    family_outcome_hit: bool = False,
    projected_family_value: int = 0,
) -> SequenceInterruptDecision:
    current_option = current_state.option_state.current_option
    option_mode = str(current_option.get("option_mode") or "")
    interaction_opportunity_changed = (
        previous_state.action_surface.interaction_opportunity_count
        != current_state.action_surface.interaction_opportunity_count
    )
    task_relevant_affordance_changed = (
        current_state.progress_state.task_relevant_affordance_delta_count > 0
    )
    signal_flags = {
        "goal_progress_changed": (
            previous_state.goal_state.ordered_target_progress
            != current_state.goal_state.ordered_target_progress
        ),
        "grounding_changed": (
            previous_state.grounding_state != current_state.grounding_state
        ),
        "action_surface_changed": (
            current_state.uncertainty_state.admissible_actions_unchanged is False
        ),
        "task_relevant_affordance_changed": task_relevant_affordance_changed,
        "interaction_opportunity_changed": interaction_opportunity_changed,
        "observation_changed": observation_changed,
        "observation_novelty_gain": observation_novelty_gain,
        "local_revisitation": local_revisitation,
        "relocation_action": action_family == "relocation",
    }
    expected_progress_signals = list(
        current_option.get("expected_progress_signals", []) if current_option else []
    )
    expected_progress_hit = any(
        signal_flags.get(signal, False) for signal in expected_progress_signals
    )
    if option_mode in {"explore_frontier", "inspect_novelty"}:
        expected_progress_hit = expected_progress_hit or observation_novelty_gain
    signal_flags["expected_progress_hit"] = expected_progress_hit
    progress_debt_limit = int(current_option.get("progress_debt_limit", 0) or 0)
    signal_flags["progress_debt_exceeded"] = bool(
        current_option
        and progress_debt_limit > 0
        and projected_progress_debt >= progress_debt_limit
    )
    signal_flags["family_outcome_hit"] = family_outcome_hit
    signal_flags["family_value_negative"] = bool(
        current_option
        and projected_family_value <= -2
        and int(current_option.get("steps_taken", 0) or 0) >= 1
    )
    signal_flags["expected_progress_missing"] = bool(
        current_option
        and expected_progress_signals
        and not expected_progress_hit
        and projected_stagnation_steps >= 2
        and signal_flags["progress_debt_exceeded"]
        and int(current_option.get("steps_taken", 0) or 0)
        < int(current_option.get("step_budget", 0) or 0)
    )

    if signal_flags["goal_progress_changed"]:
        return SequenceInterruptDecision(True, "goal_progress_changed", signal_flags)
    if signal_flags["grounding_changed"]:
        return SequenceInterruptDecision(True, "grounding_changed", signal_flags)
    if signal_flags["observation_changed"] and not signal_flags["relocation_action"]:
        return SequenceInterruptDecision(True, "observation_changed", signal_flags)
    if signal_flags["expected_progress_missing"]:
        return SequenceInterruptDecision(
            True, "expected_progress_missing", signal_flags
        )
    if signal_flags["family_value_negative"]:
        return SequenceInterruptDecision(True, "family_value_negative", signal_flags)
    if signal_flags["interaction_opportunity_changed"] and (
        not current_option or not signal_flags["relocation_action"]
    ):
        return SequenceInterruptDecision(
            True, "interaction_opportunity_changed", signal_flags
        )
    if signal_flags["action_surface_changed"] and (
        not current_option or not signal_flags["relocation_action"]
    ):
        return SequenceInterruptDecision(True, "action_surface_changed", signal_flags)
    if signal_flags["task_relevant_affordance_changed"] and (
        not current_option or not signal_flags["relocation_action"]
    ):
        return SequenceInterruptDecision(
            True, "task_relevant_affordance_changed", signal_flags
        )
    return SequenceInterruptDecision(False, "stable_progress", signal_flags)
