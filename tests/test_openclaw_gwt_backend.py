from src.agent.integrations.openclaw.gwt_backend import DefaultGWTDeliberationHarness
from src.agent.runtime import RuntimeDecision, RuntimeSessionSnapshot
from src.agent.v2.types import (
    ContradictionRecord,
    OperatorCandidate,
    OptionContract,
    RepairDirective,
    WorldModelSnapshot,
)


def test_default_gwt_harness_prefers_escape_action_for_input_detour():
    harness = DefaultGWTDeliberationHarness(agent=object())
    session_snapshot = RuntimeSessionSnapshot(
        session_id="sess-1",
        admissible_actions=["tap [7]", "navigate back"],
    )
    world_snapshot = WorldModelSnapshot(
        operator_candidates=[
            OperatorCandidate(
                operator_id="tap-7",
                family="ui_click",
                action_label="tap [7]",
                target_ids=["ui-7"],
            ),
            OperatorCandidate(
                operator_id="back",
                family="ui_navigation",
                action_label="navigate back",
            ),
        ],
        repair_directive=RepairDirective(
            operator="recover_task_surface",
            rationale="Exit the system overlay.",
        ),
        metadata={
            "input_state": {
                "input_detour": True,
                "escape_actions": ["navigate back"],
            }
        },
    )

    decision = harness.build_deliberation_recommendation(
        snapshot=session_snapshot,
        world_snapshot=world_snapshot,
        latest_decision=None,
        reason="input_detour",
        request_metadata={},
    )

    assert decision.suggested_action == "navigate back"
    assert decision.repair_operator == "recover_task_surface"
    assert decision.metadata["selected_action_family"] == "ui_navigation"


def test_default_gwt_harness_prefers_focus_click_before_typing():
    harness = DefaultGWTDeliberationHarness(agent=object())
    session_snapshot = RuntimeSessionSnapshot(
        session_id="sess-2",
        admissible_actions=['type "Hugo" into [7]', "tap [7]"],
    )
    world_snapshot = WorldModelSnapshot(
        operator_candidates=[
            OperatorCandidate(
                operator_id="type-7",
                family="ui_type",
                action_label='type "Hugo" into [7]',
                target_ids=["ui-7"],
                preconditions=["text_entry_admissible"],
            ),
            OperatorCandidate(
                operator_id="tap-7",
                family="ui_click",
                action_label="tap [7]",
                target_ids=["ui-7"],
            ),
        ],
        repair_directive=RepairDirective(
            operator="repair_input_focus",
            rationale="Field is visible but not focused.",
        ),
        metadata={
            "ui_context": {
                "visible_elements": [
                    {
                        "element_id": "ui-7",
                        "role": "textbox",
                        "attributes": {"editable": True},
                    }
                ]
            },
            "input_state": {
                "active_input_target": "ui-7",
                "text_entry_admissible": False,
            },
        },
    )

    decision = harness.build_deliberation_recommendation(
        snapshot=session_snapshot,
        world_snapshot=world_snapshot,
        latest_decision=RuntimeDecision(
            suggested_action='type "Hugo" into [7]',
            option_family="manipulate_target",
            repair_operator="repair_input_focus",
        ),
        reason="repair_focus",
        request_metadata={},
    )

    assert decision.suggested_action == "tap [7]"
    assert decision.repair_operator == "repair_input_focus"
    assert (
        "Re-focus the editable target"
        in decision.metadata["semantic_recovery_recommendation"]
    )


def test_default_gwt_harness_prefers_typing_when_input_state_is_stable():
    harness = DefaultGWTDeliberationHarness(agent=object())
    session_snapshot = RuntimeSessionSnapshot(
        session_id="sess-3",
        admissible_actions=["tap [7]", 'type "Hugo" into [7]'],
    )
    world_snapshot = WorldModelSnapshot(
        operator_candidates=[
            OperatorCandidate(
                operator_id="tap-7",
                family="ui_click",
                action_label="tap [7]",
                target_ids=["ui-7"],
            ),
            OperatorCandidate(
                operator_id="type-7",
                family="ui_type",
                action_label='type "Hugo" into [7]',
                target_ids=["ui-7"],
                preconditions=["text_entry_admissible"],
            ),
        ],
        metadata={
            "ui_context": {
                "focused_element_id": "ui-7",
                "visible_elements": [
                    {
                        "element_id": "ui-7",
                        "role": "textbox",
                        "attributes": {"editable": True},
                    }
                ],
            },
            "input_state": {
                "active_input_target": "ui-7",
                "text_entry_admissible": True,
                "input_surface": "virtual_keyboard",
            },
        },
    )

    decision = harness.build_deliberation_recommendation(
        snapshot=session_snapshot,
        world_snapshot=world_snapshot,
        latest_decision=RuntimeDecision(
            suggested_action="tap [7]",
            option_family="manipulate_target",
        ),
        reason="stable_input_state",
        request_metadata={},
    )

    assert decision.suggested_action == 'type "Hugo" into [7]'
    assert decision.metadata["selected_action_family"] == "ui_type"
    assert "Commit text now" in decision.metadata["semantic_recovery_recommendation"]


def test_default_gwt_harness_emits_world_summary_and_contradiction_notes():
    harness = DefaultGWTDeliberationHarness(agent=object())
    session_snapshot = RuntimeSessionSnapshot(
        session_id="sess-4",
        admissible_actions=["navigate back", "tap [7]"],
    )
    world_snapshot = WorldModelSnapshot(
        operator_candidates=[
            OperatorCandidate(
                operator_id="back",
                family="ui_navigation",
                action_label="navigate back",
            ),
            OperatorCandidate(
                operator_id="tap-7",
                family="ui_click",
                action_label="tap [7]",
                target_ids=["ui-7"],
            ),
        ],
        active_option=OptionContract(
            family="manipulate_target",
            objective="Add contact details",
            target_signature="ui-7",
        ),
        repair_directive=RepairDirective(
            operator="recover_task_surface",
            rationale="Dismiss the detour overlay.",
        ),
        contradictions=[
            ContradictionRecord(
                category="input_focus_missing",
                step_index=3,
                subject="ui-7",
                action_text='type "Hugo" into [7]',
                evidence="typing had no effect",
            )
        ],
        metadata={
            "ui_context": {
                "page_title": "Choose input method",
                "focused_element_id": "ui-7",
            },
            "input_state": {
                "active_input_target": "ui-7",
                "text_entry_admissible": False,
                "input_detour": True,
                "escape_actions": ["navigate back"],
            },
        },
    )

    decision = harness.build_deliberation_recommendation(
        snapshot=session_snapshot,
        world_snapshot=world_snapshot,
        latest_decision=None,
        reason="semantic_recovery_needed",
        request_metadata={},
    )

    assert decision.suggested_action == "navigate back"
    assert (
        decision.metadata["world_summary"]["active_option_family"]
        == "manipulate_target"
    )
    assert (
        decision.metadata["world_summary"]["repair_operator"] == "recover_task_surface"
    )
    assert (
        decision.metadata["recent_contradictions"][0]["category"]
        == "input_focus_missing"
    )
    assert any(
        "active option is `manipulate_target` targeting `ui-7`" in note
        for note in decision.metadata["planner_notes"]
    )
    assert any(
        "Recent contradiction: `input_focus_missing` on `ui-7`" in note
        for note in decision.metadata["planner_notes"]
    )
