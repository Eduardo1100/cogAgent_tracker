from __future__ import annotations

import json
import re
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

from sqlalchemy import select
from sqlalchemy.orm import Session

from src.storage.models import EpisodeRun, ExperimentRun

_AGENT_ITERATION_RE = re.compile(r"(^|/|:)agent-iter-(\d+)-")
_COGFIX_RE = re.compile(r"(^|/|:)cogfix-(\d+)")


@dataclass(frozen=True)
class ExperimentSummary:
    experiment_id: int
    status: str | None
    git_branch: str | None
    git_commit: str | None
    start_time: str
    end_time: str | None
    episode_count: int
    current_game_number: int | None
    current_game_label: str | None
    latest_episode: dict | None
    latest_run_dir: str | None

    def to_json(self) -> str:
        return json.dumps(
            {
                "experiment_id": self.experiment_id,
                "status": self.status,
                "git_branch": self.git_branch,
                "git_commit": self.git_commit,
                "start_time": self.start_time,
                "end_time": self.end_time,
                "episode_count": self.episode_count,
                "current_game_number": self.current_game_number,
                "current_game_label": self.current_game_label,
                "latest_episode": self.latest_episode,
                "latest_run_dir": self.latest_run_dir,
            }
        )


def parse_agent_iteration_number(branch_name: str) -> int | None:
    match = _AGENT_ITERATION_RE.search(branch_name or "")
    if not match:
        return None
    return int(match.group(2))


def next_agent_iteration_number(branch_names: Iterable[str]) -> int:
    max_seen = 0
    for branch_name in branch_names:
        parsed = parse_agent_iteration_number(branch_name)
        if parsed is not None:
            max_seen = max(max_seen, parsed)
    return max_seen + 1


def parse_cogfix_number(branch_name: str) -> int | None:
    match = _COGFIX_RE.search(branch_name or "")
    if not match:
        return None
    return int(match.group(2))


def next_cogfix_number(branch_names: Iterable[str]) -> int:
    max_seen = 0
    for branch_name in branch_names:
        parsed = parse_cogfix_number(branch_name)
        if parsed is not None:
            max_seen = max(max_seen, parsed)
    return max_seen + 1


def latest_run_dir(runs_root: Path) -> Path | None:
    candidates = [path for path in runs_root.iterdir() if path.is_dir()]
    if not candidates:
        return None
    return max(candidates, key=lambda path: path.stat().st_mtime)


def get_latest_experiment(
    session: Session,
    *,
    env_type: str,
    git_branch: str | None = None,
    after_id: int | None = None,
) -> ExperimentRun | None:
    stmt = select(ExperimentRun).where(ExperimentRun.eval_env_type == env_type)
    if git_branch:
        stmt = stmt.where(ExperimentRun.git_branch == git_branch)
    if after_id is not None:
        stmt = stmt.where(ExperimentRun.id > after_id)
    stmt = stmt.order_by(ExperimentRun.id.desc())
    return session.scalars(stmt).first()


def summarize_experiment(
    session: Session,
    experiment: ExperimentRun,
    *,
    runs_root: Path | None = None,
) -> ExperimentSummary:
    latest_episode = session.scalars(
        select(EpisodeRun)
        .where(EpisodeRun.experiment_id == experiment.id)
        .order_by(EpisodeRun.game_number.desc())
    ).first()
    episode_count = (
        session.query(EpisodeRun)
        .filter(EpisodeRun.experiment_id == experiment.id)
        .count()
    )

    latest_episode_summary = None
    if latest_episode is not None:
        latest_episode_summary = {
            "game_number": latest_episode.game_number,
            "success": latest_episode.success,
            "actions_taken": latest_episode.actions_taken,
            "chat_rounds": latest_episode.chat_rounds,
            "runtime_minutes": latest_episode.runtime_minutes,
            "error_message": latest_episode.error_message,
            "task_type": latest_episode.task_type,
            "inadmissible_action_count": latest_episode.inadmissible_action_count,
        }

    run_dir = latest_run_dir(runs_root) if runs_root else None

    return ExperimentSummary(
        experiment_id=experiment.id,
        status=experiment.status,
        git_branch=experiment.git_branch,
        git_commit=experiment.git_commit,
        start_time=experiment.start_time.isoformat(),
        end_time=experiment.end_time.isoformat() if experiment.end_time else None,
        episode_count=episode_count,
        current_game_number=experiment.current_game_number,
        current_game_label=experiment.current_game_label,
        latest_episode=latest_episode_summary,
        latest_run_dir=str(run_dir) if run_dir else None,
    )


def render_iteration_prompt(
    template_text: str,
    *,
    experiment_id: int,
    experiment_summary_json: str,
    current_branch: str,
    next_iteration: int,
    repo_root: Path,
) -> str:
    return template_text.format(
        experiment_id=experiment_id,
        experiment_summary_json=experiment_summary_json,
        current_branch=current_branch,
        next_iteration=f"{next_iteration:02d}",
        repo_root=str(repo_root),
    )
