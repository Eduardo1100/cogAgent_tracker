import json
from pathlib import Path

from scripts.run_protocol_mode_slice import (
    build_compact_output,
    build_comparison_result,
    load_trace,
)


def test_load_trace_requires_task_text_and_steps(tmp_path):
    trace_path = tmp_path / "trace.json"
    trace_path.write_text(
        json.dumps(
            {
                "task_text": "Add Hugo Pereira to contacts",
                "steps": [
                    {
                        "event": {
                            "step_index": 1,
                            "task_text": "Add Hugo Pereira to contacts",
                            "raw_observation": "Contact form visible",
                        }
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    loaded = load_trace(str(trace_path))

    assert loaded["task_text"] == "Add Hugo Pereira to contacts"
    assert len(loaded["steps"]) == 1


def test_build_compact_output_returns_side_by_side_summary():
    result = build_comparison_result(
        trace_path="fixtures/trace.json",
        protocol_only={
            "session_count": 1,
            "sessions_with_events": 1,
            "available_count": 1,
            "alignment_counts": {"strategy_diverged": 1},
            "summary_metrics": {"strategy_diverged_rate": 1.0, "aligned_rate": 0.0},
        },
        auto_escalate={
            "session_count": 1,
            "sessions_with_events": 1,
            "available_count": 1,
            "alignment_counts": {"mode_only_diverged": 1},
            "summary_metrics": {
                "mode_only_diverged_rate": 1.0,
                "strategy_diverged_rate": 0.0,
                "aligned_rate": 0.0,
            },
        },
        protocol_session_id="sess-protocol",
        auto_session_id="sess-auto",
    )

    compact = build_compact_output(result)

    assert compact["session_ids"]["protocol_only"] == "sess-protocol"
    assert compact["protocol_only"]["alignment_counts"] == {"strategy_diverged": 1}
    assert compact["auto_escalate"]["alignment_counts"] == {"mode_only_diverged": 1}
    assert compact["comparison"]["strategy_diverged_rate_delta"] == -1.0


def test_checked_in_contacts_fixture_loads():
    fixture_path = (
        Path(__file__).resolve().parents[1]
        / "fixtures"
        / "openclaw"
        / "contacts_focus_repair_trace.json"
    )

    loaded = load_trace(str(fixture_path))

    assert loaded["task_text"] == "Add Hugo Pereira to contacts"
    assert len(loaded["steps"]) == 2
    assert (
        loaded["steps"][0]["event"]["ui_context"]["input_state"][
            "text_entry_admissible"
        ]
        is False
    )
    assert (
        loaded["steps"][1]["event"]["ui_context"]["input_state"][
            "text_entry_admissible"
        ]
        is True
    )


def test_checked_in_input_method_detour_fixture_loads():
    fixture_path = (
        Path(__file__).resolve().parents[1]
        / "fixtures"
        / "openclaw"
        / "input_method_detour_trace.json"
    )

    loaded = load_trace(str(fixture_path))

    assert loaded["task_text"] == "Add Hugo Pereira to contacts"
    assert len(loaded["steps"]) == 2
    assert loaded["steps"][0]["event"]["ui_context"]["overlay_kind"] == (
        "system_input_picker"
    )
    assert loaded["steps"][0]["event"]["ui_context"]["input_state"][
        "escape_actions"
    ] == ["navigate back"]
    assert loaded["steps"][1]["event"]["normalized_observation"] == (
        "Contact editor visible after detour recovery"
    )
