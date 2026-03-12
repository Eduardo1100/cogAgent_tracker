from __future__ import annotations
# ruff: noqa: E402, I001

import argparse
import subprocess
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from sqlalchemy.exc import SQLAlchemyError

from src.automation.iteration import (
    get_latest_experiment,
    next_agent_iteration_number,
    render_iteration_prompt,
    summarize_experiment,
)
from src.storage.database import SessionLocal
from src.storage.models import ExperimentRun


RUNS_ROOT = REPO_ROOT / "runs"
PROMPT_TEMPLATE_PATH = REPO_ROOT / "prompts" / "iterate_scienceworld_prompt.txt"


class IterationWorkflowError(RuntimeError):
    pass


def db_unavailable_error() -> IterationWorkflowError:
    return IterationWorkflowError(
        "Could not reach Postgres for the local iteration workflow. "
        "Start the local stack with `make up` or verify DATABASE_URL."
    )


def get_current_branch() -> str:
    result = subprocess.run(
        ["git", "branch", "--show-current"],
        cwd=REPO_ROOT,
        check=True,
        text=True,
        capture_output=True,
    )
    branch = result.stdout.strip()
    if not branch:
        raise RuntimeError("Could not determine the current git branch.")
    return branch


def get_branch_names() -> list[str]:
    result = subprocess.run(
        [
            "git",
            "for-each-ref",
            "--format=%(refname:short)",
            "refs/heads",
            "refs/remotes/origin",
        ],
        cwd=REPO_ROOT,
        check=True,
        text=True,
        capture_output=True,
    )
    return [
        line
        for line in (entry.strip() for entry in result.stdout.splitlines())
        if line and line != "origin/HEAD"
    ]


def get_dirty_paths() -> list[str]:
    result = subprocess.run(
        ["git", "status", "--short"],
        cwd=REPO_ROOT,
        check=True,
        text=True,
        capture_output=True,
    )
    paths = []
    for line in result.stdout.splitlines():
        if not line:
            continue
        path_text = line[3:]
        if " -> " in path_text:
            path_text = path_text.split(" -> ", maxsplit=1)[1]
        paths.append(path_text)
    return paths


def load_prompt_template() -> str:
    if not PROMPT_TEMPLATE_PATH.exists():
        raise FileNotFoundError(f"Prompt template not found: {PROMPT_TEMPLATE_PATH}")
    return PROMPT_TEMPLATE_PATH.read_text()


def wait_for_experiment(
    *,
    env_type: str,
    git_branch: str,
    after_id: int | None,
    timeout_seconds: int,
) -> int:
    deadline = time.monotonic() + timeout_seconds
    while time.monotonic() < deadline:
        try:
            with SessionLocal() as db:
                experiment = get_latest_experiment(
                    db,
                    env_type=env_type,
                    git_branch=git_branch,
                    after_id=after_id,
                )
                if experiment is not None:
                    return experiment.id
        except SQLAlchemyError as exc:
            raise db_unavailable_error() from exc
        time.sleep(2)
    raise TimeoutError(
        f"No new {env_type} experiment was recorded on branch {git_branch!r} "
        f"within {timeout_seconds} seconds."
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run a random ScienceWorld debug episode and hand the resulting experiment to Codex."
    )
    parser.add_argument(
        "--env",
        default="scienceworld",
        help="Environment type to use for make debug and experiment lookup.",
    )
    parser.add_argument(
        "--skip-debug",
        action="store_true",
        help="Skip `make debug` and use the latest experiment on the current branch.",
    )
    parser.add_argument(
        "--experiment-id",
        type=int,
        default=None,
        help="Use an explicit experiment id instead of resolving the latest run.",
    )
    parser.add_argument(
        "--allow-dirty",
        action="store_true",
        help="Allow running with local uncommitted changes.",
    )
    parser.add_argument(
        "--prompt-only",
        action="store_true",
        help="Print the rendered Codex prompt and stop before invoking Codex.",
    )
    parser.add_argument(
        "--dangerous",
        action="store_true",
        help="Invoke Codex with --dangerously-bypass-approvals-and-sandbox.",
    )
    parser.add_argument(
        "--timeout-seconds",
        type=int,
        default=120,
        help="How long to wait for a new experiment row after `make debug`.",
    )
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    dirty_paths = get_dirty_paths()
    if dirty_paths and not args.allow_dirty:
        parser.error(
            "Worktree is dirty. Commit or stash changes first, or rerun with "
            f"--allow-dirty. Dirty paths: {', '.join(dirty_paths)}"
        )

    current_branch = get_current_branch()

    try:
        with SessionLocal() as db:
            previous = get_latest_experiment(
                db,
                env_type=args.env,
                git_branch=current_branch,
            )
            previous_id = previous.id if previous is not None else None
    except SQLAlchemyError as exc:
        raise db_unavailable_error() from exc

    experiment_id = args.experiment_id
    if experiment_id is None and not args.skip_debug:
        debug_result = subprocess.run(
            ["make", "debug", f"ENV={args.env}"],
            cwd=REPO_ROOT,
            check=False,
        )
        if debug_result.returncode not in {0, 130}:
            print(
                f"`make debug ENV={args.env}` exited with code "
                f"{debug_result.returncode}. Continuing because interrupted runs "
                "still create experiment rows.",
                file=sys.stderr,
            )
        experiment_id = wait_for_experiment(
            env_type=args.env,
            git_branch=current_branch,
            after_id=previous_id,
            timeout_seconds=args.timeout_seconds,
        )

    try:
        with SessionLocal() as db:
            if experiment_id is None:
                experiment = get_latest_experiment(
                    db,
                    env_type=args.env,
                    git_branch=current_branch,
                )
            else:
                experiment = db.get(ExperimentRun, experiment_id)
                if experiment is not None and experiment.eval_env_type != args.env:
                    raise IterationWorkflowError(
                        f"Experiment {experiment_id} is {experiment.eval_env_type}, not {args.env}."
                    )
                if experiment is not None and experiment.git_branch not in {
                    None,
                    current_branch,
                }:
                    raise IterationWorkflowError(
                        f"Experiment {experiment_id} belongs to branch "
                        f"{experiment.git_branch!r}, not {current_branch!r}."
                    )
            if experiment is None:
                raise IterationWorkflowError(
                    f"No {args.env} experiment found for branch {current_branch!r}."
                )
            summary = summarize_experiment(db, experiment, runs_root=RUNS_ROOT)
    except SQLAlchemyError as exc:
        raise db_unavailable_error() from exc

    next_iteration = next_agent_iteration_number(get_branch_names())
    template_text = load_prompt_template()
    prompt = render_iteration_prompt(
        template_text,
        experiment_id=summary.experiment_id,
        experiment_summary_json=summary.to_json(),
        current_branch=current_branch,
        next_iteration=next_iteration,
        repo_root=REPO_ROOT,
    )

    if args.prompt_only:
        print(prompt)
        return 0

    codex_command = ["codex", "exec", "--full-auto"]
    if args.dangerous:
        codex_command.append("--dangerously-bypass-approvals-and-sandbox")
    codex_command.extend(["-C", str(REPO_ROOT), "-"])
    return subprocess.run(
        codex_command,
        cwd=REPO_ROOT,
        text=True,
        input=prompt,
        check=False,
    ).returncode


if __name__ == "__main__":
    try:
        sys.exit(main())
    except IterationWorkflowError as exc:
        print(exc, file=sys.stderr)
        sys.exit(1)
