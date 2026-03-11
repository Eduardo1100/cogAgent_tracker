import importlib
import signal
import sys
import types

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker


def _install_run_agent_stubs() -> None:
    autogen_module = types.ModuleType("autogen")
    autogen_module.gather_usage_summary = lambda agents: {}
    sys.modules.setdefault("autogen", autogen_module)

    autogen_oai_module = types.ModuleType("autogen.oai")
    sys.modules.setdefault("autogen.oai", autogen_oai_module)

    autogen_oai_client_module = types.ModuleType("autogen.oai.client")

    class OpenAIClient:  # pragma: no cover
        @staticmethod
        def cost(*args, **kwargs):
            return 0.0

    autogen_oai_client_module.OpenAIClient = OpenAIClient
    sys.modules.setdefault("autogen.oai.client", autogen_oai_client_module)

    wandb_module = types.ModuleType("wandb")
    wandb_module.login = lambda *args, **kwargs: None
    wandb_module.init = lambda *args, **kwargs: None
    wandb_module.log = lambda *args, **kwargs: None
    wandb_module.finish = lambda *args, **kwargs: None
    sys.modules.setdefault("wandb", wandb_module)

    baseline_module = types.ModuleType("src.agent.baseline_agent")
    baseline_module.BaselineAutogenAgent = type("BaselineAutogenAgent", (), {})
    sys.modules.setdefault("src.agent.baseline_agent", baseline_module)

    gwt_module = types.ModuleType("src.agent.gwt_agent")
    gwt_module.GWTAutogenAgent = type("GWTAutogenAgent", (), {})
    sys.modules.setdefault("src.agent.gwt_agent", gwt_module)

    env_adapter_module = types.ModuleType("src.agent.env_adapter")
    env_adapter_module.ScienceWorldAdapter = type("ScienceWorldAdapter", (), {})
    env_adapter_module.infer_task_type = lambda task: None
    sys.modules.setdefault("src.agent.env_adapter", env_adapter_module)

    schema_health_module = types.ModuleType("src.config.schema_health")
    schema_health_module.require_current_schema = lambda *args, **kwargs: None
    sys.modules.setdefault("src.config.schema_health", schema_health_module)


def _load_run_agent_module():
    sys.modules.pop("scripts.run_agent", None)
    return importlib.import_module("scripts.run_agent")


def test_finalize_experiment_persists_cancelled_after_session_commit_failure(
    tmp_path, monkeypatch
):
    from src.storage.models import Base, ExperimentRun

    _install_run_agent_stubs()
    run_agent = _load_run_agent_module()

    db_path = tmp_path / "status_retry.sqlite"
    engine = create_engine(f"sqlite:///{db_path}")
    TestingSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    Base.metadata.create_all(bind=engine)

    monkeypatch.setattr(run_agent, "SessionLocal", TestingSessionLocal)

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


def test_signal_handler_marks_active_experiment_cancelled(tmp_path, monkeypatch):
    from src.storage.models import Base, ExperimentRun

    _install_run_agent_stubs()
    run_agent = _load_run_agent_module()

    db_path = tmp_path / "signal_cancel.sqlite"
    engine = create_engine(f"sqlite:///{db_path}")
    TestingSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    Base.metadata.create_all(bind=engine)

    monkeypatch.setattr(run_agent, "SessionLocal", TestingSessionLocal)

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
