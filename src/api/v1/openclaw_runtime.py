from fastapi import APIRouter, Query

from src.agent.integrations.openclaw.protocol import (
    OpenClawActionApplyRequest,
    OpenClawDeliberationRequest,
    OpenClawObservationRequest,
    OpenClawSessionCreateRequest,
)
from src.agent.integrations.openclaw.service import runtime_service

router = APIRouter()


@router.post("/runtime/sessions")
def create_runtime_session(payload: dict):
    request = OpenClawSessionCreateRequest.from_dict(payload)
    return runtime_service.create_session(request).to_dict()


@router.post("/runtime/sessions/{session_id}/observe")
def observe_runtime_session(session_id: str, payload: dict):
    request = OpenClawObservationRequest.from_dict(
        {
            "session_id": session_id,
            "event": payload.get("event", {}),
            "metadata": payload.get("metadata", {}),
        }
    )
    return runtime_service.observe(request).to_dict()


@router.post("/runtime/sessions/{session_id}/decide")
def decide_runtime_session(session_id: str, payload: dict | None = None):
    payload = payload or {}
    return runtime_service.decide(
        session_id,
        mode_override=payload.get("mode"),
    ).to_dict()


@router.post("/runtime/sessions/{session_id}/escalate")
def escalate_runtime_session(session_id: str, payload: dict | None = None):
    payload = payload or {}
    return runtime_service.escalate(
        session_id,
        reason=payload.get("reason"),
    ).to_dict()


@router.post("/runtime/sessions/{session_id}/deliberate")
def deliberate_runtime_session(session_id: str, payload: dict | None = None):
    payload = payload or {}
    request = OpenClawDeliberationRequest.from_dict(
        {
            "session_id": session_id,
            "backend": payload.get("backend"),
            "reason": payload.get("reason"),
            "metadata": payload.get("metadata", {}),
        }
    )
    return runtime_service.deliberate(request).to_dict()


@router.post("/runtime/sessions/{session_id}/apply")
def apply_runtime_session_action(session_id: str, payload: dict):
    request = OpenClawActionApplyRequest.from_dict(
        {
            "session_id": session_id,
            **payload,
        }
    )
    return runtime_service.apply(request).to_dict()


@router.get("/runtime/sessions/{session_id}")
def get_runtime_session(session_id: str):
    return runtime_service.get_snapshot(session_id).to_dict()


@router.get("/runtime/telemetry/comparison")
def get_runtime_comparison_telemetry(
    session_id: list[str] | None = Query(default=None),
):
    return runtime_service.export_comparison_telemetry(session_ids=session_id)


@router.get("/runtime/deliberations/{job_id}")
def get_runtime_deliberation(job_id: str):
    return runtime_service.get_deliberation(job_id).to_dict()


@router.delete("/runtime/sessions/{session_id}")
def delete_runtime_session(session_id: str):
    return runtime_service.delete_session(session_id)
