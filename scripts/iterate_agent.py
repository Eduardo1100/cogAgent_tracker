from __future__ import annotations
# ruff: noqa: E402, I001

import argparse
import os
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
    next_cogfix_number,
    render_iteration_prompt,
    summarize_experiment,
)
from src.storage.database import SessionLocal
from src.storage.models import ExperimentRun


RUNS_ROOT = REPO_ROOT / "runs"
PROMPT_TEMPLATE_PATHS = {
    ("iterate", "claudecode"): REPO_ROOT / "prompts" / "iterate_claude_prompt.txt",
    ("ablate", "claudecode"): REPO_ROOT / "prompts" / "ablate_claude_prompt.txt",
    ("iterate", "codex"): REPO_ROOT / "prompts" / "iterate_codex_prompt.txt",
    ("ablate", "codex"): REPO_ROOT / "prompts" / "ablate_codex_prompt.txt",
}
CODEX_MODEL = "gpt-5.4"
CODEX_REASONING_EFFORT = "xhigh"

AGENT_CHOICES = ("claudecode", "codex")


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


def load_prompt_template(mode: str, agent: str) -> str:
    template_path = PROMPT_TEMPLATE_PATHS[(mode, agent)]
    if not template_path.exists():
        raise FileNotFoundError(f"Prompt template not found: {template_path}")
    return template_path.read_text()


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
        description=(
            "Run or reuse a tales debug episode and hand the resulting "
            "experiment to an agent for either a new iteration or a consolidation pass."
        )
    )
    parser.add_argument(
        "--mode",
        choices=("iterate", "ablate"),
        default="iterate",
        help=(
            "Iteration mode. `iterate` asks the agent for a new improvement iteration; "
            "`ablate` asks the agent to simplify or consolidate recent changes."
        ),
    )
    parser.add_argument(
        "--env",
        default=None,
        help="Environment type to use for make debug and experiment lookup. "
        "When omitted, --skip-debug finds the latest experiment across all env types; "
        "otherwise a random env is chosen for the debug run.",
    )
    parser.add_argument(
        "--games",
        type=int,
        default=None,
        help="Number of games to run in the debug episode (passed as GAMES=N to make debug).",
    )
    parser.add_argument(
        "--agent",
        choices=AGENT_CHOICES,
        default="claudecode",
        help="Agent to invoke for the iteration pass (default: claudecode).",
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
        help="Print the rendered prompt and stop before invoking the agent.",
    )
    parser.add_argument(
        "--dangerous",
        action="store_true",
        help="Invoke the agent with its bypass-approvals/sandbox flag.",
    )
    parser.add_argument(
        "--max-actions",
        type=int,
        default=None,
        help="Override max environment actions per game (passed as MAX_ACTIONS to make debug).",
    )
    parser.add_argument(
        "--max-chatrounds",
        type=int,
        default=None,
        help="Override max chat rounds per game (passed as MAX_CHATROUNDS to make debug).",
    )
    parser.add_argument(
        "--timeout-seconds",
        type=int,
        default=120,
        help="How long to wait for a new experiment row after `make debug`.",
    )
    return parser


def build_codex_command(*, dangerous: bool) -> list[str]:
    command = [
        "codex",
        "exec",
        "--full-auto",
        "--model",
        CODEX_MODEL,
        "--config",
        f'model_reasoning_effort="{CODEX_REASONING_EFFORT}"',
    ]
    if dangerous:
        command.append("--dangerously-bypass-approvals-and-sandbox")
    command.extend(["-C", str(REPO_ROOT), "-"])
    return command


def build_claude_code_command(*, dangerous: bool) -> list[str]:
    command = ["claude", "--print", "--output-format", "stream-json", "--verbose"]
    if dangerous:
        command.append("--dangerously-skip-permissions")
    return command


def build_agent_command(agent: str, *, dangerous: bool) -> list[str]:
    if agent == "codex":
        return build_codex_command(dangerous=dangerous)
    return build_claude_code_command(dangerous=dangerous)


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

    # Resolve env: if the user didn't specify --env and we need to run debug,
    # pick a random env.  If skipping debug, leave env_type=None so we find
    # the latest experiment across all env types.
    env_explicit = args.env is not None
    if args.env is None and not args.skip_debug:
        import random as _rng

        args.env = _rng.choice(["tales", "nethack"])
        env_explicit = True  # we chose one, so filter by it

    # env_type for DB queries: None means "any env type"
    env_filter = args.env if env_explicit else None

    try:
        with SessionLocal() as db:
            previous = get_latest_experiment(
                db,
                env_type=env_filter,
                git_branch=current_branch,
            )
            previous_id = previous.id if previous is not None else None
    except SQLAlchemyError as exc:
        raise db_unavailable_error() from exc

    experiment_id = args.experiment_id
    if experiment_id is None and not args.skip_debug:
        debug_cmd = ["make", "debug", f"ENV={args.env}"]
        if args.games is not None:
            debug_cmd.append(f"GAMES={args.games}")
        if args.max_actions is not None:
            debug_cmd.append(f"MAX_ACTIONS={args.max_actions}")
        if args.max_chatrounds is not None:
            debug_cmd.append(f"MAX_CHATROUNDS={args.max_chatrounds}")
        debug_result = subprocess.run(
            debug_cmd,
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
                    env_type=env_filter,
                    git_branch=current_branch,
                )
            else:
                experiment = db.get(ExperimentRun, experiment_id)
                if (
                    experiment is not None
                    and env_filter is not None
                    and experiment.eval_env_type != env_filter
                ):
                    raise IterationWorkflowError(
                        f"Experiment {experiment_id} is {experiment.eval_env_type}, not {env_filter}."
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
                env_label = env_filter or "any"
                raise IterationWorkflowError(
                    f"No {env_label} experiment found for branch {current_branch!r}."
                )
            summary = summarize_experiment(db, experiment, runs_root=RUNS_ROOT)
    except SQLAlchemyError as exc:
        raise db_unavailable_error() from exc

    branch_names = get_branch_names()
    if args.agent == "claudecode":
        next_iteration = next_cogfix_number(branch_names)
    else:
        next_iteration = next_agent_iteration_number(branch_names)
    template_text = load_prompt_template(args.mode, args.agent)
    prompt = render_iteration_prompt(
        template_text,
        experiment_id=summary.experiment_id,
        experiment_summary_json=summary.to_json(),
        current_branch=current_branch,
        next_iteration=next_iteration,
        repo_root=REPO_ROOT,
        latest_run_dir=summary.latest_run_dir,
    )

    if args.prompt_only:
        print(prompt)
        return 0

    print(
        f"Invoking {args.agent} agent on experiment {summary.experiment_id} "
        f"(branch: {current_branch}, next: {next_iteration})...",
        file=sys.stderr,
    )
    env = os.environ.copy()
    if args.agent == "claudecode":
        env.pop("ANTHROPIC_API_KEY", None)
    return subprocess.run(
        build_agent_command(args.agent, dangerous=args.dangerous),
        cwd=REPO_ROOT,
        text=True,
        input=prompt,
        check=False,
        env=env,
    ).returncode


if __name__ == "__main__":
    try:
        sys.exit(main())
    except IterationWorkflowError as exc:
        print(exc, file=sys.stderr)
        sys.exit(1)
