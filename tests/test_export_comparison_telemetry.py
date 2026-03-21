from scripts.export_comparison_telemetry import build_request_url, build_summary


def test_build_request_url_supports_repeated_session_ids():
    assert (
        build_request_url("http://localhost:8000/", ["abc", "def"])
        == "http://localhost:8000/api/v1/runtime/telemetry/comparison?"
        "session_id=abc&session_id=def"
    )


def test_build_summary_returns_compact_analysis_view():
    payload = {
        "session_count": 3,
        "sessions_with_events": 2,
        "available_count": 2,
        "alignment_counts": {
            "mode_only_diverged": 1,
            "strategy_diverged": 1,
        },
        "summary_metrics": {
            "mode_only_diverged_rate": 0.5,
            "strategy_diverged_rate": 0.5,
        },
        "sessions": [{"session_id": "abc"}],
    }

    assert build_summary(payload) == {
        "session_count": 3,
        "sessions_with_events": 2,
        "available_count": 2,
        "alignment_counts": {
            "mode_only_diverged": 1,
            "strategy_diverged": 1,
        },
        "summary_metrics": {
            "mode_only_diverged_rate": 0.5,
            "strategy_diverged_rate": 0.5,
        },
    }
