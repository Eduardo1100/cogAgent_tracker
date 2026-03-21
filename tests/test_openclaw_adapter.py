import pytest

from src.agent.integrations.openclaw import (
    OpenClawAdapter,
    OpenClawObservationRequest,
    OpenClawSessionCreateRequest,
)


def _make_observation_request() -> OpenClawObservationRequest:
    return OpenClawObservationRequest.from_dict(
        {
            "session_id": "sess-1",
            "event": {
                "step_index": 2,
                "task_text": "Add Hugo Pereira to contacts",
                "raw_observation": "Contact editor visible",
                "normalized_observation": "Contact editor visible",
                "operator_candidates": [
                    {
                        "operator_id": "openclaw:tap:7",
                        "family": "ui_click",
                        "action_label": "tap [7]",
                        "target_ids": ["ui-7"],
                    },
                    {
                        "operator_id": "openclaw:type:7",
                        "family": "ui_type",
                        "action_label": 'type "Hugo" into [7]',
                        "target_ids": ["ui-7"],
                        "preconditions": ["text_entry_admissible"],
                    },
                ],
                "status_delta": {"success": False},
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


def test_openclaw_adapter_bootstraps_from_create_request():
    create_request = OpenClawSessionCreateRequest.from_dict(
        {
            "task_text": "Add Hugo Pereira to contacts",
            "session_metadata": {"origin": "openclaw"},
            "capabilities": {
                "adapter_name": "openclaw",
                "observation_mode": "ui",
                "supports_ui_context": True,
            },
            "initial_event": {
                "step_index": 0,
                "task_text": "Add Hugo Pereira to contacts",
                "raw_observation": "Launcher visible",
                "normalized_observation": "Launcher visible",
            },
        }
    )

    adapter = OpenClawAdapter(
        task_text=create_request.task_text,
        capabilities=create_request.capabilities_record(),
        initial_event=create_request.initial_adapter_event(),
        session_metadata=create_request.session_metadata,
    )

    assert adapter.task == "Add Hugo Pereira to contacts"
    assert adapter.initial_observation == "Launcher visible"
    assert adapter.get_v2_capabilities().adapter_name == "openclaw"
    assert adapter.observation == "Launcher visible"


def test_openclaw_adapter_ingests_observation_and_exposes_action_surface():
    adapter = OpenClawAdapter(task_text="Add Hugo Pereira to contacts")
    event = adapter.ingest_observation(_make_observation_request())

    assert event.task_text == "Add Hugo Pereira to contacts"
    assert adapter.observation == "Contact editor visible"
    assert adapter.admissible_actions == ["tap [7]", 'type "Hugo" into [7]']
    assert adapter.has_won is False


def test_openclaw_adapter_build_v2_event_uses_latest_external_state():
    adapter = OpenClawAdapter(task_text="Add Hugo Pereira to contacts")
    adapter.ingest_observation(_make_observation_request())

    event = adapter.build_v2_event(
        step_index=3,
        action_text="tap [7]",
        action_executed=True,
        metadata={"source": "runtime"},
    )

    assert event.step_index == 3
    assert event.task_text == "Add Hugo Pereira to contacts"
    assert event.action_result.action_text == "tap [7]"
    assert event.action_result.operator_family == "ui_click"
    assert event.ui_context is not None
    assert event.ui_context.input_state is not None
    assert event.ui_context.input_state.active_input_target == "ui-7"
    assert event.metadata["adapter_name"] == "openclaw"
    assert event.metadata["source"] == "runtime"


def test_openclaw_adapter_step_is_not_supported():
    adapter = OpenClawAdapter(task_text="Add Hugo Pereira to contacts")

    with pytest.raises(RuntimeError, match="does not execute actions directly"):
        adapter.step("tap [7]")
