from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import urlopen

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BASE_URL = "http://localhost:8000"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Export aggregated OpenClaw runtime comparison telemetry from the local API."
        )
    )
    parser.add_argument(
        "--base-url",
        default=DEFAULT_BASE_URL,
        help="Base URL for the CogAgentLab API.",
    )
    parser.add_argument(
        "--session-id",
        action="append",
        default=[],
        help="Limit the export to one or more runtime session ids.",
    )
    parser.add_argument(
        "--field",
        choices={"json", "summary"},
        default="summary",
        help="Emit the full JSON payload or a compact summary subset.",
    )
    return parser


def build_request_url(base_url: str, session_ids: list[str]) -> str:
    normalized_base = base_url.rstrip("/")
    endpoint = f"{normalized_base}/api/v1/runtime/telemetry/comparison"
    if not session_ids:
        return endpoint
    return f"{endpoint}?{urlencode([('session_id', item) for item in session_ids])}"


def fetch_json(url: str) -> dict[str, object]:
    with urlopen(url) as response:  # noqa: S310
        return json.loads(response.read().decode("utf-8"))


def build_summary(payload: dict[str, object]) -> dict[str, object]:
    summary_metrics = dict(payload.get("summary_metrics", {}))
    return {
        "session_count": int(payload.get("session_count", 0)),
        "sessions_with_events": int(payload.get("sessions_with_events", 0)),
        "available_count": int(payload.get("available_count", 0)),
        "alignment_counts": dict(payload.get("alignment_counts", {})),
        "summary_metrics": summary_metrics,
    }


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    request_url = build_request_url(args.base_url, args.session_id)
    try:
        payload = fetch_json(request_url)
    except HTTPError as exc:
        parser.exit(exc.code, f"Comparison telemetry request failed: {exc}\n")
    except URLError as exc:
        parser.exit(
            2,
            (
                "Could not reach the CogAgentLab API at "
                f"{args.base_url.rstrip('/')}. "
                f"Reason: {exc.reason}\n"
            ),
        )
    output = payload if args.field == "json" else build_summary(payload)
    print(json.dumps(output, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
