import os
from datetime import UTC, datetime
from pathlib import Path

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from scripts.get_latest_experiment import (
    build_db_unavailable_message,
    describe_database_target,
)
from scripts.iterate_scienceworld import (
    CODEX_MODEL,
    CODEX_REASONING_EFFORT,
    build_codex_command,
    build_parser,
    load_prompt_template,
)
from src.automation.iteration import (
    get_latest_experiment,
    next_agent_iteration_number,
    parse_agent_iteration_number,
    render_iteration_prompt,
    summarize_experiment,
)
from src.storage.models import Base, EpisodeRun, ExperimentRun


def _build_session(tmp_path):
    db_path = tmp_path / "iteration.sqlite"
    engine = create_engine(f"sqlite:///{db_path}")
    TestingSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    Base.metadata.create_all(bind=engine)
    return TestingSessionLocal


def test_parse_agent_iteration_number_handles_branch_names():
    assert parse_agent_iteration_number("agent-iter-09-relation-frontier") == 9
    assert parse_agent_iteration_number("origin/agent-iter-12-topic") == 12
    assert parse_agent_iteration_number("main") is None


def test_next_agent_iteration_number_uses_highest_seen_value():
    branch_names = [
        "main",
        "agent-iter-03-referent-grounding",
        "origin/agent-iter-08-artifact-creation-grounding",
    ]

    assert next_agent_iteration_number(branch_names) == 9


def test_summarize_experiment_includes_latest_episode_and_run_dir(tmp_path):
    TestingSessionLocal = _build_session(tmp_path)
    runs_root = tmp_path / "runs"
    older_run = runs_root / "2026-03-10 01:00:00"
    newer_run = runs_root / "2026-03-11 01:00:00"
    older_run.mkdir(parents=True)
    newer_run.mkdir(parents=True)
    older_time = datetime(2026, 3, 10, 1, 0, tzinfo=UTC).timestamp()
    newer_time = datetime(2026, 3, 11, 1, 0, tzinfo=UTC).timestamp()
    os.utime(older_run, (older_time, older_time))
    os.utime(newer_run, (newer_time, newer_time))

    with TestingSessionLocal() as db:
        experiment = ExperimentRun(
            agent_name="GWTAutogenAgent",
            llm_model="test-model",
            eval_env_type="scienceworld",
            max_actions_per_game=30,
            max_chat_rounds=150,
            git_branch="agent-iter-10-generic-state-change-parsing",
            git_commit="abc1234",
            status="CONCLUDED",
            current_game_number=4,
            current_game_label="Game #4",
        )
        db.add(experiment)
        db.commit()

        episode = EpisodeRun(
            experiment_id=experiment.id,
            game_number=4,
            success=False,
            actions_taken=18,
            chat_rounds=90,
            runtime_minutes=2.5,
            error_message="timed out",
            task_type=3,
            inadmissible_action_count=1,
        )
        db.add(episode)
        db.commit()

        summary = summarize_experiment(db, experiment, runs_root=runs_root)

    assert summary.experiment_id == experiment.id
    assert summary.episode_count == 1
    assert summary.latest_episode == {
        "game_number": 4,
        "success": False,
        "actions_taken": 18,
        "chat_rounds": 90,
        "runtime_minutes": 2.5,
        "error_message": "timed out",
        "task_type": 3,
        "inadmissible_action_count": 1,
    }
    assert Path(summary.latest_run_dir) == newer_run


def test_get_latest_experiment_respects_branch_and_after_id(tmp_path):
    TestingSessionLocal = _build_session(tmp_path)

    with TestingSessionLocal() as db:
        exp_one = ExperimentRun(
            agent_name="GWTAutogenAgent",
            llm_model="test-model",
            eval_env_type="scienceworld",
            max_actions_per_game=30,
            max_chat_rounds=150,
            git_branch="agent-iter-08-artifact-creation-grounding",
        )
        exp_two = ExperimentRun(
            agent_name="GWTAutogenAgent",
            llm_model="test-model",
            eval_env_type="scienceworld",
            max_actions_per_game=30,
            max_chat_rounds=150,
            git_branch="agent-iter-09-relation-frontier-grounding",
        )
        db.add_all([exp_one, exp_two])
        db.commit()

        latest = get_latest_experiment(
            db,
            env_type="scienceworld",
            git_branch="agent-iter-09-relation-frontier-grounding",
            after_id=exp_one.id,
        )

    assert latest is not None
    assert latest.id == exp_two.id


def test_render_iteration_prompt_replaces_template_fields(tmp_path):
    prompt = render_iteration_prompt(
        "Experiment {experiment_id} on {current_branch} -> {next_iteration}\n{experiment_summary_json}\n{repo_root}",
        experiment_id=82,
        experiment_summary_json='{"status": "CONCLUDED"}',
        current_branch="agent-iter-11-property-aware-measurement",
        next_iteration=12,
        repo_root=tmp_path,
    )

    assert "Experiment 82" in prompt
    assert "agent-iter-11-property-aware-measurement" in prompt
    assert "12" in prompt
    assert '{"status": "CONCLUDED"}' in prompt
    assert str(tmp_path) in prompt


def test_build_codex_command_pins_model_and_reasoning_effort():
    command = build_codex_command(dangerous=False)

    assert command[:5] == ["codex", "exec", "--full-auto", "--model", CODEX_MODEL]
    assert "--dangerously-bypass-approvals-and-sandbox" not in command
    assert "--config" in command
    assert f'model_reasoning_effort="{CODEX_REASONING_EFFORT}"' in command


def test_build_codex_command_includes_dangerous_flag_when_requested():
    command = build_codex_command(dangerous=True)

    assert "--dangerously-bypass-approvals-and-sandbox" in command


def test_load_prompt_template_supports_iteration_and_ablation_modes():
    iterate_template = load_prompt_template("iterate")
    ablate_template = load_prompt_template("ablate")

    assert "continuing the cognitive-agent iteration workflow" in iterate_template
    assert "update the analyst trace structure" in iterate_template
    assert "CI-equivalent checks pass" in iterate_template
    assert "continuing the cognitive-agent consolidation workflow" in ablate_template
    assert "update the analyst trace format" in ablate_template
    assert "CI-equivalent checks pass" in ablate_template
    assert "agent-iter-{next_iteration}-consolidate-<topic>" in ablate_template


def test_build_parser_accepts_ablation_mode():
    parser = build_parser()

    parsed = parser.parse_args(["--mode", "ablate", "--skip-debug", "--prompt-only"])

    assert parsed.mode == "ablate"
    assert parsed.skip_debug is True
    assert parsed.prompt_only is True


def test_describe_database_target_masks_to_host_port_and_db():
    assert (
        describe_database_target(
            "postgresql+psycopg2://devuser:devpass@localhost:5432/devdb"
        )
        == "localhost:5432/devdb"
    )


def test_build_db_unavailable_message_mentions_dotenv_loading():
    message = build_db_unavailable_message(
        "postgresql+psycopg2://devuser:devpass@localhost:5432/devdb"
    )

    assert "localhost:5432/devdb" in message
    assert "loads DATABASE_URL from `.env` inside Python" in message
