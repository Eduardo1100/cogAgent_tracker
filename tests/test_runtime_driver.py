from src.agent.runtime import RuntimeDriver, RuntimeSession
from src.agent.v2.types import RepairDirective, WorldModelSnapshot


class DummyAgent:
    def __init__(self):
        self.initial_message = "start"
        self.log_paths = {"chat_history_path": "/tmp/chat.txt"}
        self.group_chat = None
        self.group_chat_manager = None
        self.admissible_actions = ["inspect room"]
        self._v2_runtime_advisory = {
            "suggested_action": "inspect room",
            "option_family": "inspect",
            "planner_stop_reason": "continue_option",
        }
        self.adapter = None
        self.set_environment_calls = []
        self.run_chat_calls = []

    def set_environment(self, env, obs, info, game_no, adapter=None):
        self.set_environment_calls.append(
            {
                "env": env,
                "obs": obs,
                "info": info,
                "game_no": game_no,
                "adapter": adapter,
            }
        )
        self.adapter = adapter

    def run_chat(self, message):
        self.run_chat_calls.append(message)
        return {"ok": True}, None


class DummyAdapter:
    task = "dummy task"
    observation = "dummy observation"


class SnapshotOnlyWorldModel:
    def __init__(self, snapshot):
        self._snapshot = snapshot

    def to_snapshot(self):
        return self._snapshot


def test_runtime_session_initializes_agent_environment():
    agent = DummyAgent()
    adapter = DummyAdapter()
    session = RuntimeSession(agent=agent)

    session.initialize(
        env="env",
        obs="observation",
        info={"task": "dummy task"},
        game_no=3,
        adapter=adapter,
    )

    assert session.initialized is True
    assert session.game_no == 3
    assert agent.set_environment_calls == [
        {
            "env": "env",
            "obs": "observation",
            "info": {"task": "dummy task"},
            "game_no": 3,
            "adapter": adapter,
        }
    ]


def test_runtime_session_proxies_chat_and_snapshot():
    agent = DummyAgent()
    adapter = DummyAdapter()
    session = RuntimeSession(agent=agent)
    session.initialize(
        env="env",
        obs="observation",
        info={"task": "dummy task"},
        game_no=1,
        adapter=adapter,
    )

    result, error = session.run_chat(session.initial_message)
    snapshot = session.snapshot()
    decision = session.latest_decision()

    assert result == {"ok": True}
    assert error is None
    assert agent.run_chat_calls == ["start"]
    assert snapshot.task == "dummy task"
    assert snapshot.observation == "dummy observation"
    assert snapshot.admissible_actions == ["inspect room"]
    assert decision.suggested_action == "inspect room"
    assert decision.option_family == "inspect"


def test_runtime_session_force_deliberative_policy_escalates():
    session = RuntimeSession(agent=None)
    session.mode = "protocol"
    session.escalation_policy = "force_deliberative"

    decision = session.escalate(reason="manual_test")
    snapshot = session.snapshot()

    assert decision.mode_used == "deliberative"
    assert decision.escalated is True
    assert decision.escalation_reason == "manual_test"
    assert snapshot.escalation_state["last_escalation_reason"] == "manual_test"


def test_runtime_session_should_escalate_on_revision_streak():
    session = RuntimeSession(agent=None)
    session.protocol_revision_streak = 2

    should_escalate, reason = session.should_escalate()

    assert should_escalate is True
    assert reason == "revision_streak_exceeded"


def test_runtime_driver_tracks_sessions():
    driver = RuntimeDriver()
    session = driver.create_session(DummyAgent())

    assert driver.get_session(session.session_id) is session
    assert driver.list_sessions() == [session]
    assert driver.close_session(session.session_id) is session
    assert driver.list_sessions() == []


def test_runtime_session_snapshot_exposes_protocol_summary():
    session = RuntimeSession(agent=None)
    session.protocol_revision_streak = 1
    session.world_model = SnapshotOnlyWorldModel(
        WorldModelSnapshot(
            revision_required=False,
            repair_directive=RepairDirective(
                operator="repair_input_focus",
                rationale="Re-focus the target field.",
            ),
            contradictions=[],
            metadata={
                "contradiction_debt": 2.0,
                "repair_cycle_active": True,
                "repair_pending": True,
                "repair_attempt_count": 1,
                "max_repair_attempts": 2,
                "escalation_required": False,
                "no_progress_streak": 3,
            },
        )
    )

    snapshot = session.snapshot()

    assert snapshot.protocol_summary["contradiction_debt"] == 2.0
    assert snapshot.protocol_summary["repair_operator"] == "repair_input_focus"
    assert snapshot.protocol_summary["repair_pending"] is True
    assert snapshot.protocol_summary["approaching_escalation"] is True
    assert snapshot.protocol_summary["recommended_mode"] == "deliberative"
    assert (
        snapshot.protocol_summary["recommended_mode_reason"]
        == "repair_budget_nearly_exhausted"
    )


def test_runtime_session_snapshot_exposes_comparison_telemetry():
    session = RuntimeSession(agent=None)
    session.record_comparison_event(
        source="sync_escalate",
        protocol_hint={
            "protocol_hint": {
                "recommended_mode": "protocol",
                "protocol_suggested_action": "tap [7]",
            }
        },
        disagreement_summary={
            "available": True,
            "backend_alignment": "strategy_diverged",
            "backend_alignment_reason": "action_changed_for_recovery",
        },
        decision=None,
        backend="in_memory_stub",
        metadata={"status": "completed"},
    )

    snapshot = session.snapshot()

    assert snapshot.comparison_telemetry["event_count"] == 1
    assert snapshot.comparison_telemetry["available_count"] == 1
    assert snapshot.comparison_telemetry["alignment_counts"]["strategy_diverged"] == 1
    assert snapshot.comparison_telemetry["source_counts"]["sync_escalate"] == 1
    assert (
        snapshot.comparison_telemetry["recent_events"][0]["backend"] == "in_memory_stub"
    )
