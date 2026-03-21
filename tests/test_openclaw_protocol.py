from src.agent.integrations.openclaw import (
    OpenClawActionApplyRequest,
    OpenClawDecisionResponse,
    OpenClawDeliberationJob,
    OpenClawDeliberationRequest,
    OpenClawObservationRequest,
    OpenClawSessionCreateRequest,
    OpenClawSessionSnapshot,
)
from src.agent.runtime import RuntimeDecision, RuntimeSessionSnapshot


def test_openclaw_observation_request_builds_adapter_event():
    request = OpenClawObservationRequest.from_dict(
        {
            "session_id": "sess-1",
            "event": {
                "step_index": 4,
                "task_text": "Add Hugo Pereira to contacts",
                "raw_observation": "Contacts editor is visible.",
                "normalized_observation": "Contact form visible",
                "action_result": {
                    "action_text": "tap [7]",
                    "action_executed": True,
                    "operator_family": "ui_click",
                },
                "operator_candidates": [
                    {
                        "operator_id": "androidworld:action:7",
                        "family": "ui_click",
                        "action_label": "tap [7]",
                        "target_ids": ["ui-7"],
                    },
                    {
                        "operator_id": "androidworld:action:10",
                        "family": "ui_type",
                        "action_label": 'type "Hugo" into [7]',
                        "target_ids": ["ui-7"],
                        "preconditions": ["text_entry_admissible"],
                    },
                ],
                "ui_context": {
                    "page_title": "Create contact",
                    "focused_element_id": "ui-7",
                    "input_state": {
                        "active_input_target": "ui-7",
                        "input_modality": "text",
                        "input_surface": "virtual_keyboard",
                        "text_entry_admissible": True,
                    },
                    "visible_elements": [
                        {
                            "element_id": "ui-7",
                            "role": "textbox",
                            "name": "First name",
                        }
                    ],
                    "action_scope": ["click", "type"],
                },
                "metadata": {"source": "openclaw"},
            },
        }
    )

    event = request.to_adapter_event()

    assert event.step_index == 4
    assert event.task_text == "Add Hugo Pereira to contacts"
    assert event.action_result.action_text == "tap [7]"
    assert event.ui_context is not None
    assert event.ui_context.focused_element_id == "ui-7"
    assert event.ui_context.input_state is not None
    assert event.ui_context.input_state.text_entry_admissible is True
    assert event.operator_candidates[1].preconditions == ["text_entry_admissible"]


def test_openclaw_session_create_request_parses_initial_event():
    request = OpenClawSessionCreateRequest.from_dict(
        {
            "task_text": "Navigate to contacts",
            "mode": "protocol",
            "escalation_policy": "auto",
            "session_metadata": {"origin": "openclaw"},
            "capabilities": {
                "adapter_name": "openclaw",
                "observation_mode": "ui",
                "supports_ui_context": True,
            },
            "initial_event": {
                "step_index": 0,
                "task_text": "Navigate to contacts",
                "raw_observation": "Launcher visible",
                "normalized_observation": "Launcher visible",
            },
        }
    )

    capabilities = request.capabilities_record()
    event = request.initial_adapter_event()

    assert capabilities is not None
    assert capabilities.adapter_name == "openclaw"
    assert capabilities.supports_ui_context is True
    assert request.mode == "protocol"
    assert request.escalation_policy == "auto"
    assert event is not None
    assert event.step_index == 0
    assert event.task_text == "Navigate to contacts"


def test_openclaw_decision_response_round_trips_runtime_decision():
    decision = RuntimeDecision(
        suggested_action="tap [7]",
        option_family="manipulate_target",
        repair_operator="repair_input_focus",
        planner_stop_reason="local_repair_selected",
        revision_required=True,
        mode_used="deliberative",
        escalated=True,
        escalation_reason="repair_attempts_exhausted",
        metadata={"planner_notes": ["Focus the field first."]},
    )

    response = OpenClawDecisionResponse.from_runtime_decision("sess-2", decision)
    restored = OpenClawDecisionResponse.from_dict(response.to_dict())

    assert restored.session_id == "sess-2"
    assert restored.suggested_action == "tap [7]"
    assert restored.repair_operator == "repair_input_focus"
    assert restored.revision_required is True
    assert restored.mode_used == "deliberative"
    assert restored.escalated is True
    assert restored.escalation_reason == "repair_attempts_exhausted"


def test_openclaw_session_snapshot_round_trips_runtime_snapshot():
    snapshot = RuntimeSessionSnapshot(
        session_id="sess-3",
        game_no=2,
        task="ContactsAddContact",
        observation="Contact form visible",
        admissible_actions=["tap [7]", "navigate back"],
        runtime_advisory={"suggested_action": "tap [7]"},
        protocol_summary={
            "contradiction_debt": 1.0,
            "repair_pending": True,
            "approaching_escalation": True,
            "recommended_mode": "deliberative",
            "recommended_mode_reason": "repair_budget_nearly_exhausted",
        },
        comparison_telemetry={
            "event_count": 1,
            "alignment_counts": {"strategy_diverged": 1},
        },
        log_paths={"chat_history_path": "/tmp/chat.txt"},
    )
    decision = RuntimeDecision(
        suggested_action="tap [7]",
        option_family="manipulate_target",
        planner_stop_reason="continue_option",
        mode_used="protocol",
    )

    bridge_snapshot = OpenClawSessionSnapshot.from_runtime_snapshot(
        snapshot,
        latest_decision=decision,
    )
    restored = OpenClawSessionSnapshot.from_dict(bridge_snapshot.to_dict())
    restored_decision = restored.to_runtime_decision()

    assert restored.session_id == "sess-3"
    assert restored.game_no == 2
    assert restored.task == "ContactsAddContact"
    assert restored.admissible_actions == ["tap [7]", "navigate back"]
    assert restored.protocol_summary["contradiction_debt"] == 1.0
    assert restored.protocol_summary["approaching_escalation"] is True
    assert restored.protocol_summary["recommended_mode"] == "deliberative"
    assert restored.comparison_telemetry["event_count"] == 1
    assert restored.mode == "protocol"
    assert restored.escalation_policy == "auto"
    assert restored_decision is not None
    assert restored_decision.suggested_action == "tap [7]"
    assert restored_decision.option_family == "manipulate_target"


def test_openclaw_action_apply_request_parses_optional_fields():
    request = OpenClawActionApplyRequest.from_dict(
        {
            "session_id": "sess-4",
            "action_text": "tap [7]",
            "action_executed": True,
            "reward_delta": 1.0,
            "status_delta": {"score": 1},
            "metadata": {"source": "openclaw"},
        }
    )

    assert request.session_id == "sess-4"
    assert request.action_text == "tap [7]"
    assert request.action_executed is True
    assert request.reward_delta == 1.0
    assert request.status_delta == {"score": 1}


def test_openclaw_deliberation_types_round_trip():
    request = OpenClawDeliberationRequest.from_dict(
        {
            "session_id": "sess-5",
            "reason": "need deeper semantic recovery",
            "metadata": {"requested_by": "api"},
        }
    )
    job = OpenClawDeliberationJob.from_dict(
        {
            "job_id": "job-1",
            "session_id": "sess-5",
            "status": "completed",
            "backend": "in_memory_stub",
            "decision": {
                "session_id": "sess-5",
                "mode_used": "deliberative",
                "escalated": True,
            },
        }
    )

    assert request.session_id == "sess-5"
    assert request.reason == "need deeper semantic recovery"
    assert request.metadata == {"requested_by": "api"}
    assert job.job_id == "job-1"
    assert job.status == "completed"
    assert job.backend == "in_memory_stub"
    assert job.decision == {
        "session_id": "sess-5",
        "mode_used": "deliberative",
        "escalated": True,
    }
