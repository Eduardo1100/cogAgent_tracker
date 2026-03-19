import importlib
import signal
import sys
import types
from pathlib import Path

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

_STUBBED_MODULE_NAMES = [
    "autogen",
    "autogen.oai",
    "autogen.oai.client",
    "wandb",
    "src.agent.baseline_agent",
    "src.agent.gwt_agent",
    "src.agent.env_adapter",
    "src.config.schema_health",
    "scripts.run_agent",
]


@pytest.fixture(autouse=True)
def _restore_stubbed_modules():
    originals = {name: sys.modules.get(name) for name in _STUBBED_MODULE_NAMES}
    yield
    for name, module in originals.items():
        if module is None:
            sys.modules.pop(name, None)
        else:
            sys.modules[name] = module


def _install_run_agent_stubs() -> None:
    autogen_module = types.ModuleType("autogen")
    autogen_module.gather_usage_summary = lambda agents: {}
    sys.modules["autogen"] = autogen_module

    autogen_oai_module = types.ModuleType("autogen.oai")
    sys.modules["autogen.oai"] = autogen_oai_module

    autogen_oai_client_module = types.ModuleType("autogen.oai.client")

    class OpenAIClient:  # pragma: no cover
        @staticmethod
        def cost(*args, **kwargs):
            return 0.0

    autogen_oai_client_module.OpenAIClient = OpenAIClient
    sys.modules["autogen.oai.client"] = autogen_oai_client_module

    wandb_module = types.ModuleType("wandb")
    wandb_module.login = lambda *args, **kwargs: None
    wandb_module.init = lambda *args, **kwargs: None
    wandb_module.log = lambda *args, **kwargs: None
    wandb_module.finish = lambda *args, **kwargs: None
    sys.modules["wandb"] = wandb_module

    baseline_module = types.ModuleType("src.agent.baseline_agent")
    baseline_module.BaselineAutogenAgent = type("BaselineAutogenAgent", (), {})
    sys.modules["src.agent.baseline_agent"] = baseline_module

    gwt_module = types.ModuleType("src.agent.gwt_agent")
    gwt_module.GWTAutogenAgent = type("GWTAutogenAgent", (), {})
    sys.modules["src.agent.gwt_agent"] = gwt_module

    env_adapter_module = types.ModuleType("src.agent.env_adapter")
    env_adapter_module.ScienceWorldAdapter = type("ScienceWorldAdapter", (), {})
    env_adapter_module.TalesAdapter = type("TalesAdapter", (), {})
    env_adapter_module.NetHackAdapter = type("NetHackAdapter", (), {})
    env_adapter_module.WebArenaAdapter = type("WebArenaAdapter", (), {})
    env_adapter_module.infer_task_type = lambda task: None
    sys.modules["src.agent.env_adapter"] = env_adapter_module

    schema_health_module = types.ModuleType("src.config.schema_health")
    schema_health_module.require_current_schema = lambda *args, **kwargs: None
    sys.modules["src.config.schema_health"] = schema_health_module


def _load_run_agent_module():
    sys.modules.pop("scripts.run_agent", None)
    return importlib.import_module("scripts.run_agent")


@pytest.fixture()
def tmp_db(tmp_path, monkeypatch):
    from src.storage.models import Base

    _install_run_agent_stubs()
    run_agent = _load_run_agent_module()

    db_path = tmp_path / "test.sqlite"
    engine = create_engine(f"sqlite:///{db_path}")
    TestingSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    Base.metadata.create_all(bind=engine)
    monkeypatch.setattr(run_agent, "SessionLocal", TestingSessionLocal)
    return TestingSessionLocal


def test_finalize_experiment_persists_cancelled_after_session_commit_failure(
    tmp_db, monkeypatch
):
    from src.storage.models import ExperimentRun

    TestingSessionLocal = tmp_db
    run_agent = sys.modules["scripts.run_agent"]

    with TestingSessionLocal() as seed_db:
        experiment = ExperimentRun(
            agent_name="GWTAutogenAgent",
            llm_model="test-model",
            eval_env_type="alfworld",
            max_actions_per_game=10,
            max_chat_rounds=20,
            status="RUNNING",
            current_game_number=7,
            current_game_label="Game #7",
        )
        seed_db.add(experiment)
        seed_db.commit()
        experiment_id = experiment.id

    db = TestingSessionLocal()
    experiment = db.get(ExperimentRun, experiment_id)

    def fail_commit():
        raise RuntimeError("simulated interrupted session")

    monkeypatch.setattr(db, "commit", fail_commit)

    run_agent.finalize_experiment(
        db,
        experiment,
        cumulative_runtime=12.5,
        success_rate=0.5,
        error_adjusted_success_rate=0.5,
        error_count=1,
        avg_actions_per_successful_game=4.0,
        avg_chat_rounds_per_successful_game=8.0,
        avg_runtime_per_successful_game=2.0,
        avg_actions_per_failing_game=9.0,
        avg_chat_rounds_per_failing_game=12.0,
        avg_runtime_per_failing_game=3.0,
        status="CANCELLED",
    )

    db.close()

    with TestingSessionLocal() as verify_db:
        persisted = verify_db.get(ExperimentRun, experiment_id)
        assert persisted is not None
        assert persisted.status == "CANCELLED"
        assert persisted.current_game_number is None
        assert persisted.current_game_label is None
        assert persisted.end_time is not None
        assert persisted.total_runtime_minutes == 12.5


def test_sigterm_handler_routes_to_keyboard_interrupt():
    _install_run_agent_stubs()
    run_agent = _load_run_agent_module()

    with pytest.raises(KeyboardInterrupt, match="Received signal"):
        run_agent._raise_keyboard_interrupt(signal.SIGTERM, None)


def test_signal_handler_marks_active_experiment_cancelled(tmp_db):
    from src.storage.models import ExperimentRun

    TestingSessionLocal = tmp_db
    run_agent = sys.modules["scripts.run_agent"]

    with TestingSessionLocal() as db:
        experiment = ExperimentRun(
            agent_name="GWTAutogenAgent",
            llm_model="test-model",
            eval_env_type="scienceworld",
            max_actions_per_game=10,
            max_chat_rounds=20,
            status="RUNNING",
            current_game_number=3,
            current_game_label="Game #3",
        )
        db.add(experiment)
        db.commit()
        experiment_id = experiment.id

    run_agent.set_active_experiment(experiment_id)

    with pytest.raises(KeyboardInterrupt, match="Received signal"):
        run_agent._raise_keyboard_interrupt(signal.SIGTERM, None)

    with TestingSessionLocal() as verify_db:
        persisted = verify_db.get(ExperimentRun, experiment_id)
        assert persisted is not None
        assert persisted.status == "CANCELLED"
        assert persisted.current_game_number is None
        assert persisted.current_game_label is None
        assert persisted.end_time is not None


def test_parse_arguments_accepts_webarena_env(monkeypatch):
    _install_run_agent_stubs()
    run_agent = _load_run_agent_module()
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_agent.py",
            "src/agent/configs/webarena.yaml",
            "--gwt",
            "--env-type",
            "webarena",
            "--webarena-task-ids",
            "17",
            "42",
        ],
    )

    args = run_agent.parse_arguments()

    assert args.env_type == "webarena"
    assert args.webarena_task_ids == [17, 42]


def test_persist_chat_artifacts_recovers_in_memory_group_chat(tmp_path):
    _install_run_agent_stubs()
    run_agent = _load_run_agent_module()

    game_dir = tmp_path / "game_1"
    agent = types.SimpleNamespace(
        log_paths={
            "chat_history_path": str(game_dir / "chat_history.txt"),
            "analyst_trace_path": str(game_dir / "analyst_trace.txt"),
            "analyst_trace_ansi_path": str(game_dir / "analyst_trace.ansi"),
        },
        group_chat=types.SimpleNamespace(
            messages=[
                {
                    "name": "Belief_State_Agent",
                    "role": "assistant",
                    "content": "BELIEF STATE: [I see a mouse.]",
                },
                {
                    "name": "Action_Agent",
                    "role": "assistant",
                    "content": "execute_action('focus on mouse')",
                },
            ]
        ),
        group_chat_manager=None,
    )

    artifacts = run_agent.persist_chat_artifacts(agent)

    chat_history_path = Path(agent.log_paths["chat_history_path"])
    analyst_trace_path = Path(agent.log_paths["analyst_trace_path"])
    analyst_trace_ansi_path = Path(agent.log_paths["analyst_trace_ansi_path"])
    transition_path = game_dir / "transition_log.json"

    assert artifacts["chat_rounds"] == 2
    assert (
        "Recovered transcript from in-memory group chat state."
        in artifacts["chat_text"]
    )
    assert "Belief_State_Agent" in chat_history_path.read_text()
    assert "focus on mouse" in chat_history_path.read_text()
    assert "Analyst Trace Fallback" in analyst_trace_path.read_text()
    assert "BELIEF STATE: [I see a mouse.]" in analyst_trace_path.read_text()
    assert analyst_trace_ansi_path.read_text()
    assert transition_path.exists()
    assert artifacts["transitions"][0]["from"] == "Belief_State_Agent"
    assert artifacts["transitions"][0]["to"] == "Action_Agent"


def test_run_game_persists_partial_chat_on_keyboard_interrupt(tmp_path):
    _install_run_agent_stubs()
    run_agent = _load_run_agent_module()

    game_dir = tmp_path / "game_1"
    agent = types.SimpleNamespace(
        initial_message="start",
        log_paths={
            "chat_history_path": str(game_dir / "chat_history.txt"),
            "error_message_path": str(game_dir / "error_message.txt"),
        },
        group_chat=types.SimpleNamespace(
            messages=[
                {
                    "name": "Belief_State_Agent",
                    "role": "assistant",
                    "content": "BELIEF STATE: [I opened the closet and may need to inspect it.]",
                }
            ]
        ),
        group_chat_manager=None,
    )

    def _raise_interrupt(_message):
        raise KeyboardInterrupt("Received signal 2")

    agent.run_chat = _raise_interrupt

    with pytest.raises(KeyboardInterrupt, match="Received signal 2"):
        run_agent.run_game(agent, game_no=1)

    chat_text = (game_dir / "chat_history.txt").read_text()
    error_text = (game_dir / "error_message.txt").read_text()

    assert "Interrupted during run_chat" in chat_text
    assert "Belief_State_Agent" in chat_text
    assert "Run interrupted: Received signal 2" in error_text


def test_persist_interrupted_episode_run_saves_episode_and_chat_key(
    tmp_path, monkeypatch
):
    from src.storage.models import Base, EpisodeRun, ExperimentRun

    _install_run_agent_stubs()
    run_agent = _load_run_agent_module()

    db_path = tmp_path / "interrupted_episode.sqlite"
    engine = create_engine(f"sqlite:///{db_path}")
    TestingSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    Base.metadata.create_all(bind=engine)

    with TestingSessionLocal() as db:
        experiment = ExperimentRun(
            agent_name="GWTAutogenAgent",
            llm_model="test-model",
            eval_env_type="scienceworld",
            max_actions_per_game=10,
            max_chat_rounds=20,
            status="RUNNING",
        )
        db.add(experiment)
        db.commit()
        db.refresh(experiment)

        game_dir = tmp_path / "game_2"
        chat_history_path = game_dir / "chat_history.txt"
        chat_history_path.parent.mkdir(parents=True, exist_ok=True)
        chat_history_path.write_text(
            "\n".join(
                [
                    "NOTE: Interrupted during run_chat; recovered the latest partial transcript from in-memory group chat state.",
                    "--------------------",
                    "name: Belief_State_Agent",
                    "role: assistant",
                    "content:",
                    "BELIEF STATE: [I see water in the sink.]",
                    "--------------------",
                    "name: Action_Agent",
                    "role: assistant",
                    "content:",
                    "execute_action('focus on water')",
                    "",
                ]
            )
        )
        (game_dir / "analyst_trace.txt").write_text(
            "\n".join(
                [
                    "T1 | locate_substance | INCOMPLETE",
                    "Action: focus on water",
                    "Observation: You focus on the water.",
                    "",
                ]
            )
        )
        (game_dir / "history.txt").write_text(
            "action: 'focus on water'. observation: 'You focus on the water.'\n"
        )

        class FakeS3:
            def __init__(self):
                self.calls = []

            def put_object(self, **kwargs):
                self.calls.append(kwargs)

        fake_s3 = FakeS3()
        agent = types.SimpleNamespace(
            log_paths={
                "chat_history_path": str(chat_history_path),
                "analyst_trace_path": str(game_dir / "analyst_trace.txt"),
                "history_path": str(game_dir / "history.txt"),
            },
            group_chat=types.SimpleNamespace(agents=[]),
            curr_episodic_memory=["memory"],
            task="Change the state of matter of water.",
            num_actions_taken=2,
            adapter=types.SimpleNamespace(
                infer_task_type=lambda: 4,
                count_inadmissible_actions=lambda _path: 1,
            ),
            get_architecture_metrics=lambda: {
                "version": 4,
                "thinking_count": 2,
                "belief_update_count": 1,
                "deliberation_count": 3,
                "repeated_action_density": 0.25,
                "repeated_state_density": 0.5,
                "observation_novelty_rate": 0.5,
                "grounded_entity_growth_rate": 1.5,
                "unique_grounded_entities": 4,
                "burst_count": 2,
                "mean_burst_length": 1.0,
                "burst_length_histogram": {"1": 2},
                "burst_stop_reasons": {"single_action": 2},
                "option_count": 2,
                "mean_option_length": 1.0,
                "mean_option_progress_debt": 1.5,
                "mean_option_revisitation_count": 0.5,
                "mean_option_family_value": 0.5,
                "option_stop_reasons": {"single_action": 2},
                "option_interrupt_count": 1,
                "option_interrupt_reasons": {"expected_progress_missing": 1},
            },
        )

        totals = run_agent.persist_interrupted_episode_run(
            db,
            experiment=experiment,
            agent=agent,
            game_number=2,
            s3=fake_s3,
            total_run_usage={
                "total_tokens": 0,
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "total_cost": 0.0,
            },
            elapsed_minutes=1.25,
            success_rate=0.0,
            error_adjusted_success_rate=0.0,
            error_message="Run interrupted: Received signal 2",
        )

        assert totals["total_tokens"] == 0
        persisted_episode = (
            db.query(EpisodeRun).filter(EpisodeRun.experiment_id == experiment.id).one()
        )
        assert persisted_episode.success is False
        assert persisted_episode.chat_rounds == 2
        assert persisted_episode.runtime_minutes == 1.25
        assert persisted_episode.error_message == "Run interrupted: Received signal 2"
        assert (
            persisted_episode.chat_history_s3_key
            == f"experiments/run_{experiment.id}/game_2_chat.txt"
        )
        assert persisted_episode.belief_state["memory"] == [
            "[I see water in the sink.]"
        ]
        assert "T1 | locate_substance | INCOMPLETE" in persisted_episode.analyst_trace
        assert persisted_episode.architecture_metrics["deliberation_count"] == 3
        assert persisted_episode.architecture_metrics["tokens_per_action"] == 0.0
        persisted_experiment = db.get(ExperimentRun, experiment.id)
        assert persisted_experiment is not None
        assert persisted_experiment.architecture_metrics["version"] == 4
        assert persisted_experiment.architecture_metrics["episode_count"] == 1
        assert persisted_experiment.architecture_metrics["deliberation_count"] == 3
        assert (
            persisted_experiment.architecture_metrics["burst_stop_reasons"][
                "single_action"
            ]
            == 2
        )
        assert persisted_experiment.architecture_metrics["option_count"] == 2
        assert (
            persisted_experiment.architecture_metrics["option_interrupt_reasons"][
                "expected_progress_missing"
            ]
            == 1
        )
        assert (
            persisted_experiment.architecture_metrics["mean_option_progress_debt"]
            == 1.5
        )
        assert (
            persisted_experiment.architecture_metrics["mean_option_family_value"] == 0.5
        )
        assert (
            fake_s3.calls[0]["Key"]
            == f"experiments/run_{experiment.id}/game_2_chat.txt"
        )


def test_persist_interrupted_episode_run_tolerates_dead_adapter_task_inference(
    tmp_path, monkeypatch
):
    from src.storage.models import Base, EpisodeRun, ExperimentRun

    _install_run_agent_stubs()
    run_agent = _load_run_agent_module()

    db_path = tmp_path / "interrupted_dead_adapter.sqlite"
    engine = create_engine(f"sqlite:///{db_path}")
    TestingSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    Base.metadata.create_all(bind=engine)

    with TestingSessionLocal() as db:
        experiment = ExperimentRun(
            agent_name="GWTAutogenAgent",
            llm_model="test-model",
            eval_env_type="scienceworld",
            max_actions_per_game=10,
            max_chat_rounds=20,
            status="RUNNING",
        )
        db.add(experiment)
        db.commit()
        db.refresh(experiment)

        game_dir = tmp_path / "game_3"
        chat_history_path = game_dir / "chat_history.txt"
        chat_history_path.parent.mkdir(parents=True, exist_ok=True)
        chat_history_path.write_text(
            "\n".join(
                [
                    "NOTE: Interrupted during run_chat; recovered the latest partial transcript from in-memory group chat state.",
                    "--------------------",
                    "name: Belief_State_Agent",
                    "role: assistant",
                    "content:",
                    "BELIEF STATE: [The red light bulb is wired but still off.]",
                    "",
                ]
            )
        )
        (game_dir / "analyst_trace.txt").write_text(
            "\n".join(
                [
                    "T2 | inspect_target_mechanism | INCOMPLETE",
                    "Action: connect red light bulb cathode to solar panel cathode",
                    "Observation: Nothing obvious happens.",
                    "",
                ]
            )
        )
        (game_dir / "history.txt").write_text(
            "action: 'connect red light bulb cathode to solar panel cathode'. observation: 'Nothing obvious happens.'\n"
        )

        class FakeS3:
            def __init__(self):
                self.calls = []

            def put_object(self, **kwargs):
                self.calls.append(kwargs)

        fake_s3 = FakeS3()

        def _raise_dead_adapter():
            raise RuntimeError("java server went away")

        agent = types.SimpleNamespace(
            log_paths={
                "chat_history_path": str(chat_history_path),
                "analyst_trace_path": str(game_dir / "analyst_trace.txt"),
                "history_path": str(game_dir / "history.txt"),
            },
            group_chat=types.SimpleNamespace(agents=[]),
            curr_episodic_memory=["memory"],
            task="Focus on the red light bulb and create an electrical circuit.",
            num_actions_taken=5,
            adapter=types.SimpleNamespace(
                infer_task_type=_raise_dead_adapter,
                count_inadmissible_actions=lambda _path: 0,
            ),
        )

        totals = run_agent.persist_interrupted_episode_run(
            db,
            experiment=experiment,
            agent=agent,
            game_number=3,
            s3=fake_s3,
            total_run_usage={
                "total_tokens": 0,
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "total_cost": 0.0,
            },
            elapsed_minutes=0.5,
            success_rate=0.0,
            error_adjusted_success_rate=0.0,
            error_message="Run interrupted: Received signal 2",
        )

        assert totals["total_tokens"] == 0
        persisted_episode = (
            db.query(EpisodeRun).filter(EpisodeRun.experiment_id == experiment.id).one()
        )
        assert persisted_episode.task_type is None
        assert persisted_episode.chat_rounds == 1
        assert persisted_episode.architecture_metrics["belief_update_count"] == 1
        assert persisted_episode.architecture_metrics["thinking_count"] == 0
        assert (
            "T2 | inspect_target_mechanism | INCOMPLETE"
            in persisted_episode.analyst_trace
        )
        assert persisted_episode.chat_history_s3_key == (
            f"experiments/run_{experiment.id}/game_3_chat.txt"
        )
        assert fake_s3.calls[0]["Key"] == (
            f"experiments/run_{experiment.id}/game_3_chat.txt"
        )


def test_configure_live_analyst_trace_persists_existing_trace(tmp_db, tmp_path):
    from src.storage.models import ExperimentRun

    TestingSessionLocal = tmp_db
    run_agent = sys.modules["scripts.run_agent"]

    with TestingSessionLocal() as db:
        experiment = ExperimentRun(
            agent_name="GWTAutogenAgent",
            llm_model="test-model",
            eval_env_type="scienceworld",
            max_actions_per_game=10,
            max_chat_rounds=20,
            status="RUNNING",
        )
        db.add(experiment)
        db.commit()
        db.refresh(experiment)
        experiment_id = experiment.id

    analyst_trace_path = tmp_path / "game_1" / "analyst_trace.txt"
    analyst_trace_path.parent.mkdir(parents=True, exist_ok=True)
    analyst_trace_path.write_text(
        "\n".join(
            [
                "T0 | locate_primary_target | INCOMPLETE",
                "Action: None",
                "Observation: You are in the hallway.",
                "",
            ]
        )
    )
    agent = types.SimpleNamespace(
        log_paths={"analyst_trace_path": str(analyst_trace_path)},
        get_analyst_trace_text=lambda: analyst_trace_path.read_text(),
    )

    run_agent.configure_live_analyst_trace(agent, experiment_id)

    with TestingSessionLocal() as db:
        persisted = db.get(ExperimentRun, experiment_id)
        assert persisted is not None
        assert (
            "T0 | locate_primary_target | INCOMPLETE" in persisted.current_analyst_trace
        )

    callback = getattr(agent, "analyst_trace_callback")
    callback("T1 | locate_primary_target | INCOMPLETE\nAction: open door to kitchen\n")

    with TestingSessionLocal() as db:
        persisted = db.get(ExperimentRun, experiment_id)
        assert persisted is not None
        assert persisted.current_analyst_trace.startswith(
            "T1 | locate_primary_target | INCOMPLETE"
        )


def test_get_agent_usage_totals_prefers_agent_runtime_hook():
    _install_run_agent_stubs()
    run_agent = _load_run_agent_module()

    agent = types.SimpleNamespace(
        get_usage_totals=lambda: {
            "prompt_tokens": 11,
            "completion_tokens": 7,
            "total_tokens": 18,
            "total_cost": 0.125,
        },
        group_chat=types.SimpleNamespace(agents=["ignored"]),
    )

    totals = run_agent.get_agent_usage_totals(agent)

    assert totals == {
        "prompt_tokens": 11,
        "completion_tokens": 7,
        "total_tokens": 18,
        "total_cost": 0.125,
    }


def test_parse_arguments_accepts_runtime_summary_flag(monkeypatch):
    _install_run_agent_stubs()
    run_agent = _load_run_agent_module()

    monkeypatch.setattr(
        sys,
        "argv",
        ["run_agent.py", "config.yaml", "--gwt", "--show_runtime_summary"],
    )

    args = run_agent.parse_arguments()

    assert args.gwt is True
    assert args.show_runtime_summary is True


def test_aggregate_architecture_metrics_counts_terminal_status_reasons():
    _install_run_agent_stubs()
    run_agent = _load_run_agent_module()

    aggregated = run_agent.aggregate_architecture_metrics(
        [
            {
                "version": 9,
                "terminal_status_reason": "budget_exhausted",
                "v2_revision_required_count": 1,
                "v2_revision_action_count": 2,
            },
            {"version": 9, "terminal_status_reason": "success"},
            {"version": 9, "terminal_status_reason": "budget_exhausted"},
        ]
    )

    assert aggregated["version"] == 9
    assert aggregated["v2_revision_required_count"] == 1
    assert aggregated["v2_revision_action_count"] == 2
    assert aggregated["terminal_status_reasons"] == {
        "budget_exhausted": 2,
        "success": 1,
    }
