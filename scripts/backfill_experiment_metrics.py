import argparse
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.storage.database import SessionLocal
from src.storage.models import EpisodeRun, ExperimentRun

REPO_ROOT = Path(__file__).resolve().parents[1]
DEEPSEEK_V32_INPUT_CACHE_MISS_PER_1M = 0.28
DEEPSEEK_V32_OUTPUT_PER_1M = 0.42


def _git_output(*args: str) -> str | None:
    try:
        return subprocess.check_output(
            [
                "git",
                "-c",
                f"safe.directory={REPO_ROOT}",
                "-C",
                str(REPO_ROOT),
                *args,
            ],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except Exception:
        return None


def get_git_commit() -> str | None:
    return _git_output("rev-parse", "HEAD")


def get_git_branch() -> str | None:
    branch = _git_output("symbolic-ref", "--short", "HEAD")
    if branch:
        return branch

    branch = _git_output("rev-parse", "--abbrev-ref", "HEAD")
    if branch and branch != "HEAD":
        return branch

    commit = get_git_commit()
    if commit:
        return f"detached@{commit[:7]}"

    return None


def is_deepseek_profile(experiment: ExperimentRun) -> bool:
    llm_model = (experiment.llm_model or "").lower()
    return "deepseek" in llm_model


def estimate_deepseek_v32_cost(prompt_tokens: int, completion_tokens: int) -> float:
    # Historical rows do not store provider cache-hit token counts, so use the
    # cache-miss input rate as a consistent upper-bound estimate.
    return (
        prompt_tokens * DEEPSEEK_V32_INPUT_CACHE_MISS_PER_1M
        + completion_tokens * DEEPSEEK_V32_OUTPUT_PER_1M
    ) / 1_000_000


def compress_game_list(game_numbers: list[int]) -> str:
    if not game_numbers:
        return "none"

    ranges: list[str] = []
    start = prev = game_numbers[0]
    for number in game_numbers[1:]:
        if number == prev + 1:
            prev = number
            continue
        ranges.append(f"{start}-{prev}" if start != prev else str(start))
        start = prev = number
    ranges.append(f"{start}-{prev}" if start != prev else str(start))
    return ", ".join(ranges)


def looks_cumulative(values: list[float]) -> bool:
    if len(values) < 2:
        return False
    if values[-1] <= 0:
        return False
    if any(curr < prev for prev, curr in zip(values, values[1:])):
        return False
    return sum(values) > values[-1]


def to_per_episode(values: list[float]) -> list[float]:
    if not looks_cumulative(values):
        return values

    deltas: list[float] = []
    previous = 0.0
    for value in values:
        deltas.append(max(value - previous, 0.0))
        previous = value
    return deltas


def backfill_experiment(
    experiment: ExperimentRun,
    *,
    dry_run: bool,
    repair_git: bool,
    repair_deepseek_costs: bool,
) -> dict[str, float | int | bool]:
    db = SessionLocal()
    try:
        experiment = db.get(ExperimentRun, experiment.id)
        assert experiment is not None
        episodes = (
            db.query(EpisodeRun)
            .filter(EpisodeRun.experiment_id == experiment.id)
            .order_by(EpisodeRun.game_number.asc(), EpisodeRun.id.asc())
            .all()
        )

        prompt_values = [float(ep.prompt_tokens or 0) for ep in episodes]
        completion_values = [float(ep.completion_tokens or 0) for ep in episodes]
        cost_values = [float(ep.episode_cost or 0.0) for ep in episodes]

        per_episode_prompt = to_per_episode(prompt_values)
        per_episode_completion = to_per_episode(completion_values)
        per_episode_cost = to_per_episode(cost_values)

        for idx, episode in enumerate(episodes):
            episode.prompt_tokens = int(round(per_episode_prompt[idx]))
            episode.completion_tokens = int(round(per_episode_completion[idx]))
            episode.episode_cost = float(per_episode_cost[idx])

        repriced = False
        if repair_deepseek_costs and is_deepseek_profile(experiment):
            repriced = True
            for episode in episodes:
                episode.episode_cost = estimate_deepseek_v32_cost(
                    int(episode.prompt_tokens or 0),
                    int(episode.completion_tokens or 0),
                )

        completed = len(episodes)
        successes = sum(1 for ep in episodes if ep.success)
        error_free = sum(
            1 for ep in episodes if ep.success or ep.error_message is None
        )
        experiment.total_tokens = int(
            round(sum(per_episode_prompt) + sum(per_episode_completion))
        )
        experiment.prompt_tokens = int(round(sum(per_episode_prompt)))
        experiment.completion_tokens = int(round(sum(per_episode_completion)))
        experiment.total_cost = float(
            sum(float(ep.episode_cost or 0.0) for ep in episodes)
        )
        experiment.num_games = experiment.num_games or completed
        experiment.success_rate = successes / completed if completed else 0.0
        experiment.error_adjusted_success_rate = (
            successes / error_free if error_free else 0.0
        )
        experiment.num_errors = sum(
            1 for ep in episodes if ep.error_message and not ep.success
        )
        experiment.total_runtime_minutes = float(
            sum(ep.runtime_minutes or 0.0 for ep in episodes)
        )
        if experiment.end_time is None and completed == (experiment.num_games or 0):
            experiment.end_time = datetime.now(UTC)
        if not experiment.status:
            experiment.status = "CONCLUDED" if experiment.end_time else "RUNNING"

        if not experiment.selected_games:
            game_numbers = [ep.game_number for ep in episodes]
            experiment.selected_games = game_numbers
            experiment.selected_games_display = compress_game_list(game_numbers)
        elif not experiment.selected_games_display:
            selected_games = [
                int(game_number)
                for game_number in experiment.selected_games
                if isinstance(game_number, (int, float))
            ]
            experiment.selected_games_display = compress_game_list(selected_games)

        if experiment.status != "RUNNING":
            experiment.current_game_number = None
            experiment.current_game_label = None

        git_repaired = False
        if repair_git:
            if not experiment.git_commit:
                experiment.git_commit = get_git_commit()
                git_repaired = git_repaired or bool(experiment.git_commit)
            if not experiment.git_branch:
                experiment.git_branch = get_git_branch()
                git_repaired = git_repaired or bool(experiment.git_branch)

        result = {
            "episodes": completed,
            "total_tokens": experiment.total_tokens or 0,
            "total_cost": experiment.total_cost or 0.0,
            "git_repaired": git_repaired,
            "repriced": repriced,
        }

        if dry_run:
            db.rollback()
        else:
            db.commit()

        return result
    finally:
        db.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Repair historical experiment metrics and run metadata."
    )
    parser.add_argument(
        "--experiment-id",
        type=int,
        action="append",
        dest="experiment_ids",
        help="Specific experiment id to repair. May be repeated.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Compute repairs but do not write them to the database.",
    )
    parser.add_argument(
        "--repair-git",
        action="store_true",
        help="Fill missing git branch/commit metadata from the current repo checkout.",
    )
    parser.add_argument(
        "--repair-deepseek-costs",
        action="store_true",
        help=(
            "Reprice DeepSeek-profile experiments using current V3.2 rates and "
            "stored prompt/completion token totals. This is an upper-bound estimate "
            "because historical cache-hit token counts were not stored."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    db = SessionLocal()
    try:
        query = db.query(ExperimentRun).order_by(ExperimentRun.id.asc())
        if args.experiment_ids:
            query = query.filter(ExperimentRun.id.in_(args.experiment_ids))
        experiments = query.all()
    finally:
        db.close()

    for experiment in experiments:
        result = backfill_experiment(
            experiment,
            dry_run=args.dry_run,
            repair_git=args.repair_git,
            repair_deepseek_costs=args.repair_deepseek_costs,
        )
        print(
            f"experiment={experiment.id} episodes={result['episodes']} "
            f"tokens={result['total_tokens']} cost=${result['total_cost']:.4f} "
            f"git_repaired={result['git_repaired']} repriced={result['repriced']}"
        )


if __name__ == "__main__":
    main()
