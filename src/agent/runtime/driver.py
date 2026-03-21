from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from src.agent.runtime.session import RuntimeSession


@dataclass
class RuntimeDriver:
    sessions: dict[str, RuntimeSession] = field(default_factory=dict)

    def create_session(self, agent: Any | None = None) -> RuntimeSession:
        session = RuntimeSession(agent=agent)
        self.sessions[session.session_id] = session
        return session

    def get_session(self, session_id: str) -> RuntimeSession:
        return self.sessions[session_id]

    def close_session(self, session_id: str) -> RuntimeSession | None:
        return self.sessions.pop(session_id, None)

    def list_sessions(self) -> list[RuntimeSession]:
        return list(self.sessions.values())
