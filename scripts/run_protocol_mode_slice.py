from __future__ import annotations
# ruff: noqa: E402, I001

import argparse
import json
import sys
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.export_comparison_telemetry import build_summary as build_telemetry_summary

DEFAULT_BASE_URL = "http://localhost:8000"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Replay a JSON observation trace into protocol-only and auto-escalate "
            "OpenClaw runtime sessions, then print a side-by-side telemetry summary."
        )
    )
    parser.add_argument("trace", help="Path to the JSON trace file to replay.")
    parser.add_argument(
        "--base-url",
        default=DEFAULT_BASE_URL,
        help="Base URL for the CogAgentLab API.",
    )
    parser.add_argument(
        "--field",
        choices={"summary", "json"},
        default="summary",
        help="Emit a compact side-by-side summary or the full JSON result.",
    )
    return parser


def _post_json(base_url: str, path: str, payload: dict[str, Any]) -> dict[str, Any]:
    request = Request(
        f"{base_url.rstrip('/')}{path}",
        data=json.dumps(payload).encode("utf-8"),
        headers={"content-type": "application/json"},
        method="POST",
    )
    with urlopen(request) as response:  # noqa: S310
        return json.loads(response.read().decode("utf-8"))


def _get_json(base_url: str, path: str) -> dict[str, Any]:
    with urlopen(f"{base_url.rstrip('/')}{path}") as response:  # noqa: S310
        return json.loads(response.read().decode("utf-8"))


def load_trace(path: str) -> dict[str, Any]:
    with open(path, encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError("Trace payload must be a JSON object.")
    if "task_text" not in payload:
        raise ValueError("Trace payload must include task_text.")
    if "steps" not in payload or not isinstance(payload["steps"], list):
        raise ValueError("Trace payload must include a steps list.")
    return payload


def create_session(
    base_url: str,
    *,
    task_text: str,
    capabilities: dict[str, Any] | None,
    session_metadata: dict[str, Any],
    escalation_policy: str,
) -> str:
    response = _post_json(
        base_url,
        "/api/v1/runtime/sessions",
        {
            "task_text": task_text,
            "mode": "protocol",
            "escalation_policy": escalation_policy,
            "capabilities": capabilities or {},
            "session_metadata": session_metadata,
        },
    )
    return str(response["session_id"])


def replay_trace(
    base_url: str,
    *,
    session_id: str,
    steps: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    decisions: list[dict[str, Any]] = []
    for step in steps:
        event = dict(step.get("event") or {})
        if event:
            _post_json(
                base_url,
                f"/api/v1/runtime/sessions/{session_id}/observe",
                {
                    "event": event,
                    "metadata": dict(step.get("metadata") or {}),
                },
            )
        if bool(step.get("decide", True)):
            decisions.append(
                _post_json(
                    base_url,
                    f"/api/v1/runtime/sessions/{session_id}/decide",
                    {},
                )
            )
        apply_payload = step.get("apply")
        if isinstance(apply_payload, dict):
            _post_json(
                base_url,
                f"/api/v1/runtime/sessions/{session_id}/apply",
                dict(apply_payload),
            )
    return decisions


def fetch_telemetry(base_url: str, *, session_ids: list[str]) -> dict[str, Any]:
    query = "&".join(f"session_id={session_id}" for session_id in session_ids)
    suffix = f"?{query}" if query else ""
    return _get_json(base_url, f"/api/v1/runtime/telemetry/comparison{suffix}")


def build_comparison_result(
    *,
    trace_path: str,
    protocol_only: dict[str, Any],
    auto_escalate: dict[str, Any],
    protocol_session_id: str,
    auto_session_id: str,
) -> dict[str, Any]:
    protocol_metrics = dict(protocol_only.get("summary_metrics", {}))
    auto_metrics = dict(auto_escalate.get("summary_metrics", {}))
    return {
        "trace_path": trace_path,
        "session_ids": {
            "protocol_only": protocol_session_id,
            "auto_escalate": auto_session_id,
        },
        "protocol_only": protocol_only,
        "auto_escalate": auto_escalate,
        "comparison": {
            "strategy_diverged_rate_delta": float(
                auto_metrics.get("strategy_diverged_rate", 0.0)
            )
            - float(protocol_metrics.get("strategy_diverged_rate", 0.0)),
            "mode_only_diverged_rate_delta": float(
                auto_metrics.get("mode_only_diverged_rate", 0.0)
            )
            - float(protocol_metrics.get("mode_only_diverged_rate", 0.0)),
            "aligned_rate_delta": float(auto_metrics.get("aligned_rate", 0.0))
            - float(protocol_metrics.get("aligned_rate", 0.0)),
            "top_strategy_reason_protocol_only": protocol_metrics.get(
                "top_strategy_reason"
            ),
            "top_strategy_reason_auto_escalate": auto_metrics.get(
                "top_strategy_reason"
            ),
        },
    }


def build_compact_output(result: dict[str, Any]) -> dict[str, Any]:
    return {
        "trace_path": result["trace_path"],
        "session_ids": result["session_ids"],
        "protocol_only": build_telemetry_summary(result["protocol_only"]),
        "auto_escalate": build_telemetry_summary(result["auto_escalate"]),
        "comparison": dict(result["comparison"]),
    }


def run_slice(
    base_url: str, trace_payload: dict[str, Any], trace_path: str
) -> dict[str, Any]:
    task_text = str(trace_payload["task_text"])
    capabilities = (
        dict(trace_payload["capabilities"])
        if isinstance(trace_payload.get("capabilities"), dict)
        else None
    )
    steps = [dict(step) for step in trace_payload["steps"]]
    shared_metadata = (
        dict(trace_payload.get("session_metadata"))
        if isinstance(trace_payload.get("session_metadata"), dict)
        else {}
    )
    protocol_session_id = create_session(
        base_url,
        task_text=task_text,
        capabilities=capabilities,
        session_metadata={**shared_metadata, "cohort": "protocol_only"},
        escalation_policy="force_protocol",
    )
    auto_session_id = create_session(
        base_url,
        task_text=task_text,
        capabilities=capabilities,
        session_metadata={**shared_metadata, "cohort": "auto_escalate"},
        escalation_policy="auto",
    )
    replay_trace(base_url, session_id=protocol_session_id, steps=steps)
    replay_trace(base_url, session_id=auto_session_id, steps=steps)
    protocol_summary = fetch_telemetry(base_url, session_ids=[protocol_session_id])
    auto_summary = fetch_telemetry(base_url, session_ids=[auto_session_id])
    return build_comparison_result(
        trace_path=trace_path,
        protocol_only=protocol_summary,
        auto_escalate=auto_summary,
        protocol_session_id=protocol_session_id,
        auto_session_id=auto_session_id,
    )


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    try:
        trace_payload = load_trace(args.trace)
        result = run_slice(args.base_url, trace_payload, args.trace)
    except ValueError as exc:
        parser.exit(2, f"{exc}\n")
    except HTTPError as exc:
        parser.exit(2, f"Runtime API request failed: {exc}\n")
    except URLError as exc:
        parser.exit(
            2,
            (
                "Could not reach the CogAgentLab API at "
                f"{args.base_url.rstrip('/')}. "
                f"Reason: {exc.reason}\n"
            ),
        )
    output = result if args.field == "json" else build_compact_output(result)
    print(json.dumps(output, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
