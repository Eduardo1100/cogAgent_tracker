import time

from fastapi import FastAPI
from fastapi.testclient import TestClient

from src.agent.integrations.openclaw import GWTDeliberativeBackend
from src.agent.integrations.openclaw.service import runtime_service
from src.agent.runtime import RuntimeDecision
from src.api.v1.openclaw_runtime import router


def _make_client() -> TestClient:
    runtime_service.driver.sessions.clear()
    runtime_service.deliberation_job_index.clear()
    for backend in runtime_service.deliberative_backends.values():
        backend.clear()
    app = FastAPI()
    app.include_router(router, prefix="/api/v1")
    return TestClient(app)


def test_openclaw_runtime_api_create_observe_decide_apply_and_delete():
    client = _make_client()

    create_response = client.post(
        "/api/v1/runtime/sessions",
        json={
            "task_text": "Add Hugo Pereira to contacts",
            "mode": "protocol",
            "escalation_policy": "auto",
            "session_metadata": {"origin": "openclaw"},
            "capabilities": {
                "adapter_name": "openclaw",
                "observation_mode": "ui",
                "supports_ui_context": True,
                "supports_status_delta": True,
                "supports_reward_delta": True,
                "supports_operator_candidates": True,
            },
        },
    )
    assert create_response.status_code == 200
    session_payload = create_response.json()
    session_id = session_payload["session_id"]

    observe_response = client.post(
        f"/api/v1/runtime/sessions/{session_id}/observe",
        json={
            "event": {
                "step_index": 1,
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
            }
        },
    )
    assert observe_response.status_code == 200
    observed_payload = observe_response.json()
    assert observed_payload["task"] == "Add Hugo Pereira to contacts"
    assert observed_payload["observation"] == "Contact editor visible"
    assert observed_payload["mode"] == "protocol"
    assert observed_payload["escalation_policy"] == "auto"
    assert observed_payload["protocol_summary"]["repair_pending"] is False
    assert observed_payload["protocol_summary"]["approaching_escalation"] is False
    assert observed_payload["protocol_summary"]["recommended_mode"] == "protocol"

    decide_response = client.post(f"/api/v1/runtime/sessions/{session_id}/decide")
    assert decide_response.status_code == 200
    decision_payload = decide_response.json()
    assert decision_payload["session_id"] == session_id
    assert decision_payload["mode_used"] in {"protocol", "deliberative"}

    apply_response = client.post(
        f"/api/v1/runtime/sessions/{session_id}/apply",
        json={
            "action_text": "tap [7]",
            "action_executed": True,
            "reward_delta": 1.0,
            "status_delta": {"score": 1},
        },
    )
    assert apply_response.status_code == 200
    applied_payload = apply_response.json()
    assert applied_payload["session_id"] == session_id

    get_response = client.get(f"/api/v1/runtime/sessions/{session_id}")
    assert get_response.status_code == 200
    snapshot_payload = get_response.json()
    assert snapshot_payload["session_id"] == session_id
    assert snapshot_payload["latest_decision"] is not None
    assert "contradiction_debt" in snapshot_payload["protocol_summary"]
    assert "recommended_mode" in snapshot_payload["protocol_summary"]

    escalate_response = client.post(
        f"/api/v1/runtime/sessions/{session_id}/escalate",
        json={"reason": "manual_escalation_test"},
    )
    assert escalate_response.status_code == 200
    escalated_payload = escalate_response.json()
    assert escalated_payload["mode_used"] == "deliberative"
    assert escalated_payload["escalated"] is True
    assert escalated_payload["escalation_reason"] == "manual_escalation_test"
    assert (
        escalated_payload["metadata"]["protocol_hint"]["recommended_mode"] == "protocol"
    )
    assert (
        escalated_payload["metadata"]["disagreement_summary"]["backend_alignment"]
        == "strategy_diverged"
    )
    assert (
        escalated_payload["metadata"]["disagreement_summary"][
            "backend_alignment_reason"
        ]
        == "action_changed_for_recovery"
    )
    post_escalate_snapshot = client.get(f"/api/v1/runtime/sessions/{session_id}")
    assert post_escalate_snapshot.status_code == 200
    assert post_escalate_snapshot.json()["comparison_telemetry"]["event_count"] == 1
    assert (
        post_escalate_snapshot.json()["comparison_telemetry"]["source_counts"][
            "sync_escalate"
        ]
        == 1
    )

    delete_response = client.delete(f"/api/v1/runtime/sessions/{session_id}")
    assert delete_response.status_code == 200
    assert delete_response.json() == {"status": "deleted", "session_id": session_id}


def test_openclaw_runtime_api_returns_404_for_unknown_session():
    client = _make_client()

    response = client.post("/api/v1/runtime/sessions/missing/decide")

    assert response.status_code == 404
    assert response.json()["detail"] == "Runtime session not found"


def test_openclaw_runtime_api_force_deliberative_policy():
    client = _make_client()

    create_response = client.post(
        "/api/v1/runtime/sessions",
        json={
            "task_text": "Add Hugo Pereira to contacts",
            "mode": "protocol",
            "escalation_policy": "force_deliberative",
        },
    )
    session_id = create_response.json()["session_id"]

    response = client.post(
        f"/api/v1/runtime/sessions/{session_id}/decide",
        json={"mode": "protocol"},
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["mode_used"] == "deliberative"
    assert payload["escalated"] is True
    assert payload["escalation_reason"] == "forced_deliberative_policy"
    assert payload["metadata"]["protocol_hint"]["recommended_mode"] == "protocol"
    assert payload["metadata"]["disagreement_summary"]["available"] is True


def test_openclaw_runtime_api_decide_auto_escalation_uses_comparison_surface():
    client = _make_client()

    create_response = client.post(
        "/api/v1/runtime/sessions",
        json={"task_text": "Add Hugo Pereira to contacts"},
    )
    session_id = create_response.json()["session_id"]
    session = runtime_service.driver.get_session(session_id)
    session.protocol_revision_streak = 2

    response = client.post(f"/api/v1/runtime/sessions/{session_id}/decide")

    assert response.status_code == 200
    payload = response.json()
    assert payload["mode_used"] == "deliberative"
    assert payload["escalated"] is True
    assert payload["escalation_reason"] == "revision_streak_exceeded"
    assert payload["metadata"]["protocol_hint"]["recommended_mode"] == "protocol"
    assert payload["metadata"]["disagreement_summary"]["available"] is True
    assert (
        payload["metadata"]["disagreement_summary"]["backend_alignment"]
        == "mode_only_diverged"
    )
    assert (
        payload["metadata"]["disagreement_summary"]["backend_alignment_reason"]
        == "mode_changed_for_deliberation"
    )
    session_snapshot = client.get(f"/api/v1/runtime/sessions/{session_id}")
    assert session_snapshot.status_code == 200
    assert session_snapshot.json()["comparison_telemetry"]["event_count"] == 1
    assert (
        session_snapshot.json()["comparison_telemetry"]["source_counts"]["sync_decide"]
        == 1
    )


def test_openclaw_runtime_api_deliberation_job_flow():
    client = _make_client()

    create_response = client.post(
        "/api/v1/runtime/sessions",
        json={"task_text": "Add Hugo Pereira to contacts"},
    )
    session_id = create_response.json()["session_id"]

    deliberate_response = client.post(
        f"/api/v1/runtime/sessions/{session_id}/deliberate",
        json={
            "reason": "semantic_recovery_needed",
            "metadata": {"requested_by": "test"},
        },
    )

    assert deliberate_response.status_code == 200
    job_payload = deliberate_response.json()
    assert job_payload["session_id"] == session_id
    assert job_payload["status"] == "completed"
    assert job_payload["backend"] == "in_memory_stub"
    assert job_payload["decision"] is not None
    assert job_payload["decision"]["mode_used"] == "deliberative"
    assert job_payload["decision"]["escalated"] is True
    assert job_payload["metadata"]["protocol_hint"]["recommended_mode"] == "protocol"
    assert (
        job_payload["metadata"]["protocol_hint"]["recommended_mode_reason"]
        == "stable_protocol_state"
    )
    assert job_payload["metadata"]["protocol_hint"]["protocol_suggested_action"] is None
    assert job_payload["metadata"]["protocol_hint"]["protocol_repair_operator"] is None
    assert job_payload["metadata"]["disagreement_summary"]["available"] is True
    assert (
        job_payload["metadata"]["disagreement_summary"]["backend_alignment"]
        == "mode_only_diverged"
    )
    assert (
        job_payload["metadata"]["disagreement_summary"]["backend_alignment_reason"]
        == "mode_changed_for_deliberation"
    )
    assert job_payload["metadata"]["disagreement_summary"]["mode_diverged"] is True
    assert job_payload["metadata"]["disagreement_summary"]["action_diverged"] is False
    assert (
        job_payload["metadata"]["disagreement_summary"]["repair_operator_diverged"]
        is False
    )

    get_response = client.get(f"/api/v1/runtime/deliberations/{job_payload['job_id']}")

    assert get_response.status_code == 200
    assert get_response.json() == job_payload
    session_snapshot = client.get(f"/api/v1/runtime/sessions/{session_id}")
    assert session_snapshot.status_code == 200
    assert session_snapshot.json()["comparison_telemetry"]["event_count"] == 1
    assert (
        session_snapshot.json()["comparison_telemetry"]["source_counts"][
            "async_deliberation_job"
        ]
        == 1
    )


def test_openclaw_runtime_api_returns_404_for_unknown_deliberation_job():
    client = _make_client()

    response = client.get("/api/v1/runtime/deliberations/missing")

    assert response.status_code == 404
    assert response.json()["detail"] == "Deliberation job not found"


def test_openclaw_runtime_api_gwt_backend_job_flow():
    class _FakeHarness:
        def build_deliberation_recommendation(
            self,
            *,
            snapshot,
            world_snapshot,
            latest_decision,
            reason,
            request_metadata,
        ):
            from src.agent.runtime import RuntimeDecision

            return RuntimeDecision(
                suggested_action="tap [7]",
                option_family="recover_from_failure",
                planner_stop_reason="gwt_deliberation_recommendation",
                mode_used="deliberative",
                escalated=True,
                escalation_reason=reason,
                metadata={"request_metadata": dict(request_metadata)},
            )

    original_backend = runtime_service.deliberative_backends["gwt"]
    runtime_service.deliberative_backends["gwt"] = GWTDeliberativeBackend(
        harness_factory=lambda _job_id: _FakeHarness()
    )
    try:
        client = _make_client()

        create_response = client.post(
            "/api/v1/runtime/sessions",
            json={"task_text": "Add Hugo Pereira to contacts"},
        )
        session_id = create_response.json()["session_id"]

        deliberate_response = client.post(
            f"/api/v1/runtime/sessions/{session_id}/deliberate",
            json={
                "backend": "gwt",
                "reason": "gwt_semantic_recovery",
                "metadata": {"requested_by": "test"},
            },
        )

        assert deliberate_response.status_code == 200
        job_payload = deliberate_response.json()
        assert job_payload["backend"] == "gwt"

        job_id = job_payload["job_id"]
        final_payload = job_payload
        for _ in range(20):
            get_response = client.get(f"/api/v1/runtime/deliberations/{job_id}")
            assert get_response.status_code == 200
            final_payload = get_response.json()
            if final_payload["status"] == "completed":
                break
            time.sleep(0.01)

        assert final_payload["status"] == "completed"
        assert final_payload["decision"]["suggested_action"] == "tap [7]"
        assert final_payload["decision"]["mode_used"] == "deliberative"
        assert final_payload["decision"]["escalation_reason"] == "gwt_semantic_recovery"
        assert (
            final_payload["metadata"]["protocol_hint"]["recommended_mode"] == "protocol"
        )
        assert (
            final_payload["metadata"]["protocol_hint"]["protocol_suggested_action"]
            is None
        )
        assert (
            final_payload["metadata"]["protocol_hint"]["protocol_repair_operator"]
            is None
        )
        assert final_payload["metadata"]["disagreement_summary"]["available"] is True
        assert (
            final_payload["metadata"]["disagreement_summary"]["backend_alignment"]
            == "strategy_diverged"
        )
        assert (
            final_payload["metadata"]["disagreement_summary"][
                "backend_alignment_reason"
            ]
            == "action_changed_for_recovery"
        )
        assert (
            final_payload["metadata"]["disagreement_summary"]["mode_diverged"] is True
        )
        assert (
            final_payload["metadata"]["disagreement_summary"]["action_diverged"] is True
        )
        assert "protocol_hint" in final_payload["metadata"]
    finally:
        runtime_service.deliberative_backends["gwt"] = original_backend


def test_openclaw_runtime_api_exports_comparison_telemetry_summary():
    client = _make_client()

    first_session = client.post(
        "/api/v1/runtime/sessions",
        json={"task_text": "First task"},
    ).json()["session_id"]
    second_session = client.post(
        "/api/v1/runtime/sessions",
        json={"task_text": "Second task"},
    ).json()["session_id"]

    session = runtime_service.driver.get_session(first_session)
    session.protocol_revision_streak = 2
    decide_response = client.post(f"/api/v1/runtime/sessions/{first_session}/decide")
    assert decide_response.status_code == 200

    runtime_service.driver.get_session(
        second_session
    )._latest_decision = RuntimeDecision(
        suggested_action="tap [7]",
        mode_used="protocol",
    )
    escalate_response = client.post(
        f"/api/v1/runtime/sessions/{second_session}/escalate",
        json={"reason": "manual_escalation_test"},
    )
    assert escalate_response.status_code == 200

    summary_response = client.get("/api/v1/runtime/telemetry/comparison")
    assert summary_response.status_code == 200
    summary_payload = summary_response.json()
    assert summary_payload["session_count"] == 2
    assert summary_payload["sessions_with_events"] == 2
    assert summary_payload["event_count"] == 2
    assert summary_payload["available_count"] == 2
    assert summary_payload["alignment_counts"]["mode_only_diverged"] == 1
    assert summary_payload["alignment_counts"]["strategy_diverged"] == 1
    assert summary_payload["summary_metrics"]["mode_only_diverged_count"] == 1
    assert summary_payload["summary_metrics"]["strategy_diverged_count"] == 1
    assert summary_payload["summary_metrics"]["mode_only_diverged_rate"] == 0.5
    assert summary_payload["summary_metrics"]["strategy_diverged_rate"] == 0.5
    assert (
        summary_payload["alignment_reason_counts"]["mode_changed_for_deliberation"] == 1
    )
    assert (
        summary_payload["alignment_reason_counts"]["action_changed_for_recovery"] == 1
    )
    assert (
        summary_payload["summary_metrics"]["top_alignment_reason"]
        == "mode_changed_for_deliberation"
    )
    assert (
        summary_payload["summary_metrics"]["top_strategy_reason"]
        == "action_changed_for_recovery"
    )
    assert summary_payload["source_counts"]["sync_decide"] == 1
    assert summary_payload["source_counts"]["sync_escalate"] == 1
    assert len(summary_payload["sessions"]) == 2
    assert {
        entry["session_id"]: entry["latest_alignment"]
        for entry in summary_payload["sessions"]
    } == {
        first_session: "mode_only_diverged",
        second_session: "strategy_diverged",
    }

    filtered_response = client.get(
        "/api/v1/runtime/telemetry/comparison",
        params=[("session_id", first_session)],
    )
    assert filtered_response.status_code == 200
    filtered_payload = filtered_response.json()
    assert filtered_payload["session_count"] == 1
    assert filtered_payload["event_count"] == 1
    assert filtered_payload["alignment_counts"] == {"mode_only_diverged": 1}
