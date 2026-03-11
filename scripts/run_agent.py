import json
import os
import subprocess

os.environ["TOKENIZERS_PARALLELISM"] = "false"

import argparse
import random
import re
import time
from datetime import UTC, datetime
from pathlib import Path

import autogen
import yaml
from botocore.exceptions import ClientError
from dotenv import load_dotenv

import wandb
from src.agent.baseline_agent import BaselineAutogenAgent
from src.agent.env_adapter import ScienceWorldAdapter, infer_task_type
from src.agent.gwt_agent import GWTAutogenAgent
from src.storage import cache
from src.storage.database import SessionLocal, engine
from src.storage.models import Base, EpisodeRun, ExperimentRun
from src.storage.s3 import get_s3_client

load_dotenv()

BUCKET_NAME = "alfworld-experiments"
WANDB_PROJECT = "cognitive_agents"
WANDB_ENTITY = "eduardocortes1100-university-of-california-berkeley"


def get_git_commit() -> str | None:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        return None


def get_git_branch() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"], text=True
        ).strip()
    except Exception:
        return None


# ── Helpers ────────────────────────────────────────────────────────────────────


def get_llm_profile(config_data):
    profile_name = os.getenv("ACTIVE_LLM_PROFILE", "gemini_with_deepseek_reasoner")
    print(f"🔄 Switching to LLM Profile: {profile_name}")

    profiles_list = config_data.get("config_list") or config_data.get(
        "llm_profiles", []
    )
    curr_profile = None
    for item in profiles_list:
        if isinstance(item, dict) and profile_name in item:
            curr_profile = item[profile_name]
            break

    if not curr_profile:
        print(
            f"⚠️ Profile '{profile_name}' not found. Falling back to the first available."
        )
        if profiles_list and isinstance(profiles_list[0], dict):
            fallback_key = list(profiles_list[0].keys())[0]
            curr_profile = profiles_list[0][fallback_key]
        else:
            raise ValueError("No valid LLM profiles found in config file!")

    for model_config in curr_profile:
        for key, value in list(model_config.items()):
            if (
                isinstance(value, str)
                and value.startswith("${")
                and value.endswith("}")
            ):
                env_var = value[2:-1]
                model_config[key] = os.getenv(env_var, "")

    _cache_seed_env = os.getenv("CACHE_SEED", "42")
    _cache_seed = None if _cache_seed_env.lower() == "null" else int(_cache_seed_env)
    return {"config_list": curr_profile, "cache_seed": _cache_seed, "temperature": 0.0}


def ensure_s3_bucket(s3, bucket_name: str) -> None:
    try:
        s3.head_bucket(Bucket=bucket_name)
    except ClientError:
        s3.create_bucket(Bucket=bucket_name)
        print(f"🪣 Created new S3 bucket: {bucket_name}")


def run_game(agent, game_no: int, max_retries: int = 3):
    """Run a single game with rate-limit retry logic.

    Returns (chat_result, error_message, elapsed_minutes).
    """
    chat_result, error_message = None, None  # always initialised (#1)
    game_completed = False
    retry_count = 0

    start_time = time.time()
    while retry_count < max_retries and not game_completed:
        try:
            print(f"🚀 [Attempt {retry_count + 1}] Starting Game #{game_no}...")
            chat_result, error_message = agent.run_chat(agent.initial_message)
            game_completed = True
        except Exception as e:
            error_msg = str(e).upper()
            if "429" in error_msg or "RESOURCE_EXHAUSTED" in error_msg:
                retry_count += 1
                wait_time = 65 * retry_count
                print(
                    f"\n🛑 RATE LIMIT: Sleeping {wait_time}s then retrying Game #{game_no}..."
                )
                time.sleep(wait_time)
            else:
                print(f"❌ Critical Error in Game {game_no}: {e}")
                break

    elapsed_minutes = (time.time() - start_time) / 60
    return chat_result, error_message, elapsed_minutes


def extract_chat_metadata(chat_text: str):
    """Extract agent transitions and belief-state summaries from a chat log.

    Returns (transitions, belief_matches) where transitions is a list of
    {from, to, step} dicts and belief_matches is a list of matched strings.
    """
    transition_names = re.findall(r"name: (.*)", chat_text)
    transitions = [
        {"from": transition_names[i], "to": transition_names[i + 1], "step": i}
        for i in range(len(transition_names) - 1)
    ]
    # Use a distinct variable so belief results never alias transition results (#4)
    belief_matches = re.findall(r"Belief State: (.*)", chat_text, re.IGNORECASE)
    return transitions, belief_matches


def resolve_train_eval_mode(
    split_name, resolved_eval_path, eval_id_path, eval_ood_path, dataset_cfg
):
    if eval_id_path is not None and resolved_eval_path == eval_id_path:
        dataset_cfg["eval_id_data_path"] = str(resolved_eval_path)
        return "eval_in_distribution"
    if eval_ood_path is not None and resolved_eval_path == eval_ood_path:
        dataset_cfg["eval_ood_data_path"] = str(resolved_eval_path)
        return "eval_out_of_distribution"
    # Fallback by split name
    if split_name in {"valid_seen", "test_seen", "valid_train"}:
        dataset_cfg["eval_id_data_path"] = str(resolved_eval_path)
        return "eval_in_distribution"
    if split_name in {"valid_unseen", "test_unseen"}:
        dataset_cfg["eval_ood_data_path"] = str(resolved_eval_path)
        return "eval_out_of_distribution"
    raise ValueError(
        f"Could not infer evaluation mode from split={split_name}. "
        "Expected one of valid_seen, valid_unseen, valid_train, test_seen, test_unseen."
    )


# ── CLI ────────────────────────────────────────────────────────────────────────


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Evaluate different Autogen Agents on the ALFWorld environment."
    )
    parser.add_argument("config_file", help="Path to the YAML config file")

    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--baseline", action="store_true", help="Use BaselineAutogenAgent"
    )
    group.add_argument("--gwt", action="store_true", help="Use GWTAutogenAgent")

    parser.add_argument(
        "--num_games",
        type=int,
        default=-1,
        help="Games to evaluate per split. -1 means all games (default).",
    )
    parser.add_argument(
        "--max_actions", type=int, default=35, help="Max environment actions per game"
    )
    parser.add_argument(
        "--max_chat_rounds", type=int, default=150, help="Max chat rounds per game"
    )
    parser.add_argument(
        "--rag_episode_k",
        type=int,
        default=5,
        help="Episodes retrieved by retrieve_memory() (mid-game RAG call)",
    )
    parser.add_argument(
        "--rag_concept_k",
        type=int,
        default=5,
        help="Knowledge concepts retrieved by retrieve_memory() (mid-game RAG call)",
    )
    parser.add_argument(
        "--rag_episode_k_initial",
        type=int,
        default=10,
        help="Episodes injected in the initial message (start-of-game RAG)",
    )
    parser.add_argument(
        "--rag_concept_k_initial",
        type=int,
        default=5,
        help="Knowledge concepts injected in the initial message (start-of-game RAG)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for game selection (default: None = truly random).",
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        default=None,
        help="Dataset splits to evaluate (overrides config). E.g. --splits valid_seen",
    )
    parser.add_argument(
        "--game_ids",
        type=str,
        default=None,
        help="Comma-separated game indices to run, e.g. '1,5,10' (debug mode).",
    )
    parser.add_argument(
        "--task_type",
        type=int,
        default=None,
        choices=[1, 2, 3, 4, 5, 6],
        help="Run one random game of this task type (debug mode). "
        "1=pick&place 2=examine 3=clean 4=heat 5=cool 6=pick-two.",
    )
    parser.add_argument(
        "--env-type",
        choices=["alfworld", "scienceworld"],
        default="alfworld",
        help="Which environment to evaluate (default: alfworld).",
    )
    parser.add_argument(
        "--sw-tasks",
        nargs="+",
        default=None,
        help="ScienceWorld task names to evaluate (default: all tasks).",
    )
    parser.add_argument(
        "--sw-variations",
        type=int,
        default=None,
        help="Number of variations per task (default: all variations).",
    )
    return parser.parse_args()


# ── Helpers ────────────────────────────────────────────────────────────────────


def _scan_task_types(env, total_num_games: int) -> dict[int | None, list[int]]:
    """Pre-scan all games to build a task-type -> [game_index] mapping.

    ALFWorld advances env state on every env.reset() call, so we must cycle
    through all games once to identify task types before the real run loop.
    """
    _task_re = re.compile(r"your task is to[:\s]+(.+)", re.IGNORECASE)
    index: dict[int | None, list[int]] = {}
    for i in range(1, total_num_games + 1):
        obs, _ = env.reset()
        raw_obs = obs[0] if isinstance(obs, (list, tuple)) else obs
        m = _task_re.search(raw_obs)
        task_desc = m.group(1).strip() if m else raw_obs
        tt = infer_task_type(task_desc)
        index.setdefault(tt, []).append(i)
    return index


# ── ScienceWorld eval loop ──────────────────────────────────────────────────────


def run_scienceworld_eval(agent, agent_name, args, llm_profile_name, s3, db):
    from scienceworld import ScienceWorldEnv

    sw_env = ScienceWorldEnv("")
    task_names = args.sw_tasks or sw_env.get_task_names()

    # Per-run metrics
    chat_round_list: list[int] = []
    error_list: list[int] = []
    success_list: list[int] = []
    failure_list: list[int] = []
    cumulative_successful_actions = cumulative_failing_actions = 0
    cumulative_successful_chat_rounds = cumulative_failing_chat_rounds = 0
    cumulative_successful_runtime = cumulative_failing_runtime = 0
    avg_actions_taken_per_successful_game = avg_actions_taken_per_failing_game = 0.0
    avg_chat_rounds_per_successful_game = avg_chat_rounds_per_failing_game = 0.0
    avg_runtime_per_successful_game = avg_runtime_per_failing_game = 0.0
    cumulative_runtime = 0.0
    num_games_evaluated = num_successes = 0
    error_adjusted_success_rate = 0.0
    total_run_usage: dict[str, float] = {
        "total_tokens": 0,
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "total_cost": 0.0,
    }

    experiment = ExperimentRun(
        agent_name=agent_name,
        llm_model=llm_profile_name,
        eval_env_type="scienceworld",
        max_actions_per_game=args.max_actions,
        max_chat_rounds=args.max_chat_rounds,
        start_time=datetime.now(UTC),
        split="scienceworld",
        num_games=0,  # updated after counting
    )
    db.add(experiment)
    db.commit()
    db.refresh(experiment)
    print(f"✅ Started DB Experiment Run ID: {experiment.id}")

    experiment.agents_config = agent.agents_info
    experiment.git_commit = get_git_commit()
    experiment.git_branch = get_git_branch()
    db.commit()

    game_no = 0
    total_games = sum(
        min(args.sw_variations, sw_env.get_max_variations(t))
        if args.sw_variations
        else sw_env.get_max_variations(t)
        for t in task_names
    )
    experiment.num_games = total_games
    db.commit()

    for task_name in task_names:
        total_vars = sw_env.get_max_variations(task_name)
        num_vars = (
            min(args.sw_variations, total_vars) if args.sw_variations else total_vars
        )
        for var_idx in range(num_vars):
            game_no += 1
            num_games_evaluated += 1

            sw_env.load(task_name, var_idx)
            obs, info = sw_env.reset()
            adapter = ScienceWorldAdapter(sw_env, obs, info, task_name=task_name)
            agent.set_environment(sw_env, obs, info, game_no, adapter=adapter)
            log_paths = agent.log_paths

            print(f"\n[Running Game #{game_no}] task={task_name} var={var_idx}")
            chat_result, error_message, elapsed_minutes = run_game(agent, game_no)
            cumulative_runtime += elapsed_minutes

            raw_usage = autogen.gather_usage_summary(agent.group_chat.agents)
            usage_data = (raw_usage or {}).get("usage_including_cached_inference", {})
            game_prompt_tokens = sum(
                v.get("prompt_tokens", 0)
                for v in usage_data.values()
                if isinstance(v, dict)
            )
            game_completion_tokens = sum(
                v.get("completion_tokens", 0)
                for v in usage_data.values()
                if isinstance(v, dict)
            )
            game_total_tokens = game_prompt_tokens + game_completion_tokens
            game_total_cost = usage_data.get("total_cost", 0.0)
            game_usage = usage_data
            if game_usage:
                total_run_usage["prompt_tokens"] += game_prompt_tokens
                total_run_usage["completion_tokens"] += game_completion_tokens
                total_run_usage["total_tokens"] += game_total_tokens
                total_run_usage["total_cost"] += game_total_cost
                wandb.log(
                    {
                        "game/total_tokens": game_total_tokens,
                        "game/cost": game_total_cost,
                    },
                    step=num_games_evaluated,
                )

            if error_message:
                error_list.append(game_no)
                with open(log_paths["error_message_path"], "a") as f:
                    f.write(f"Run Chat: {error_message}\n")

            if chat_result and getattr(chat_result, "chat_history", []):
                with open(log_paths["chat_history_path"], "w") as f:
                    for message in chat_result.chat_history:
                        f.write("-" * 20 + "\n")
                        for key in ["name", "role", "content"]:
                            if key in message:
                                f.write(
                                    f"{key}:\n{message[key]}\n"
                                    if key == "content"
                                    else f"{key}: {message[key]}\n"
                                )
                        for k, v in message.items():
                            if k not in ["name", "role", "content"]:
                                f.write(f"{k}: {v}\n")
                chat_round_list.append(len(chat_result.chat_history))
            else:
                chat_round_list.append(-1)
                with open(log_paths["chat_history_path"], "w") as f:
                    f.write("Error Message: no chat history in chat result\n")

            with open(log_paths["chat_history_path"]) as f:
                chat_text = f.read()

            transitions, belief_matches = extract_chat_metadata(chat_text)

            transition_path = os.path.join(
                os.path.dirname(log_paths["chat_history_path"]), "transition_log.json"
            )
            with open(transition_path, "w") as f:
                json.dump(transitions, f, indent=2)

            agent.prev_episodic_memories.append(
                {
                    "episode_number": num_games_evaluated,
                    "task_outcome": agent.task_status,
                    "memory": belief_matches
                    if belief_matches
                    else agent.curr_episodic_memory,
                }
            )

            s3_key = None
            try:
                _s3_key = f"experiments/run_{experiment.id}/game_{game_no}_chat.txt"
                s3.put_object(
                    Bucket=BUCKET_NAME,
                    Key=_s3_key,
                    Body=chat_text.encode("utf-8"),
                    ContentType="text/plain",
                )
                s3_key = _s3_key
                print(f"☁️ Uploaded chat history to S3: {s3_key}")
            except Exception as e:
                print(f"⚠️ S3 upload failed: {e}")

            success = agent.success
            if success:
                num_successes += 1
                success_list.append(game_no)
                cumulative_successful_actions += agent.num_actions_taken
                cumulative_successful_chat_rounds += chat_round_list[-1]
                cumulative_successful_runtime += elapsed_minutes
                avg_actions_taken_per_successful_game = (
                    cumulative_successful_actions / num_successes
                )
                avg_chat_rounds_per_successful_game = (
                    cumulative_successful_chat_rounds / num_successes
                )
                avg_runtime_per_successful_game = (
                    cumulative_successful_runtime / num_successes
                )
            else:
                num_failures = num_games_evaluated - num_successes
                failure_list.append(game_no)
                cumulative_failing_actions += agent.num_actions_taken
                cumulative_failing_chat_rounds += chat_round_list[-1]
                cumulative_failing_runtime += elapsed_minutes
                avg_actions_taken_per_failing_game = (
                    cumulative_failing_actions / num_failures
                )
                avg_chat_rounds_per_failing_game = (
                    cumulative_failing_chat_rounds / num_failures
                )
                avg_runtime_per_failing_game = cumulative_failing_runtime / num_failures

            success_rate = num_successes / num_games_evaluated
            num_games_no_error = num_games_evaluated - len(
                [g for g in error_list if g not in success_list]
            )
            error_adjusted_success_rate = (
                num_successes / num_games_no_error if num_games_no_error > 0 else 0.0
            )

            wandb.log(
                {
                    "task_name": task_name,
                    "var_idx": var_idx,
                    "game_no": game_no,
                    "success": int(success),
                    "actions_taken": agent.num_actions_taken,
                    "success_rate": success_rate,
                    "runtime": elapsed_minutes,
                    "cumulative_runtime": cumulative_runtime,
                    "chat_rounds": chat_round_list[-1],
                    "error_adjusted_success_rate": error_adjusted_success_rate,
                    "final/total_tokens": total_run_usage["total_tokens"],
                    "final/total_cost": total_run_usage["total_cost"],
                },
                step=num_games_evaluated,
            )

            concept_matches = re.findall(
                r"CONCEPT DISCOVERED: \[(.*?)\]", chat_text, re.DOTALL
            )
            concept_matches = [c.strip() for c in concept_matches]
            concept_matches = [
                c for c in concept_matches if not c.upper().startswith("NO CONCEPT")
            ]

            try:
                experiment.total_tokens = total_run_usage["total_tokens"]
                experiment.total_cost = total_run_usage["total_cost"]
                experiment.prompt_tokens = int(total_run_usage["prompt_tokens"])
                experiment.completion_tokens = int(total_run_usage["completion_tokens"])
                db.commit()
            except Exception as e:
                print(f"⚠️ Database logging failed: {e}")
                db.rollback()

            episode = EpisodeRun(
                experiment_id=experiment.id,
                game_number=game_no,
                success=bool(success),
                actions_taken=agent.num_actions_taken,
                chat_rounds=chat_round_list[-1],
                runtime_minutes=elapsed_minutes,
                error_message=str(error_message) if error_message else None,
                transitions={"transitions": transitions},
                belief_state={
                    "memory": belief_matches
                    if belief_matches
                    else agent.curr_episodic_memory
                },
                task=agent.task,
                task_type=agent.adapter.infer_task_type(),
                inadmissible_action_count=agent.adapter.count_inadmissible_actions(
                    log_paths["history_path"]
                ),
                concepts_learned=concept_matches if concept_matches else None,
                prompt_tokens=game_prompt_tokens if game_usage else None,
                completion_tokens=game_completion_tokens if game_usage else None,
                episode_cost=game_total_cost if game_usage else None,
                success_rate=success_rate,
                error_adjusted_success_rate=error_adjusted_success_rate,
                chat_history_s3_key=s3_key,
            )
            db.add(episode)
            db.commit()
            print(f"✅ Saved Game #{game_no} to PostgreSQL Database!")

            print(f"[Ran Game #{game_no}] task={task_name} var={var_idx}")
            print(
                f"Success: {success} | Actions: {agent.num_actions_taken} | Runtime: {elapsed_minutes:.2f}m"
            )
            print(
                f"Success Rate: {num_successes}/{num_games_evaluated} = {100 * success_rate:.2f}%"
            )

    experiment.end_time = datetime.now(UTC)
    experiment.total_runtime_minutes = cumulative_runtime
    experiment.success_rate = success_rate if num_games_evaluated else 0.0
    experiment.error_adjusted_success_rate = error_adjusted_success_rate
    experiment.num_errors = len(error_list)
    experiment.avg_actions_per_successful_game = avg_actions_taken_per_successful_game
    experiment.avg_chat_rounds_per_successful_game = avg_chat_rounds_per_successful_game
    experiment.avg_runtime_per_successful_game = avg_runtime_per_successful_game
    experiment.avg_actions_per_failing_game = avg_actions_taken_per_failing_game
    experiment.avg_chat_rounds_per_failing_game = avg_chat_rounds_per_failing_game
    experiment.avg_runtime_per_failing_game = avg_runtime_per_failing_game
    db.commit()
    print("✅ ScienceWorld experiment finalized in the database.")

    print(
        f"Final Success Rate: {num_successes}/{num_games_evaluated} = "
        f"{100 * (num_successes / num_games_evaluated if num_games_evaluated else 0):.2f}%"
    )


# ── Main ───────────────────────────────────────────────────────────────────────


def main():
    args = parse_arguments()
    random.seed(args.seed)  # reproducible game selection (#10)

    Base.metadata.create_all(bind=engine)

    agent_class, agent_name = (
        (BaselineAutogenAgent, "BaselineAutogenAgent")
        if args.baseline
        else (GWTAutogenAgent, "GWTAutogenAgent")
    )

    llm_profile_name = os.getenv("ACTIVE_LLM_PROFILE", "gemini_free")
    with open(args.config_file) as f:
        config = yaml.safe_load(f)

    # S3 bucket ensured once at startup (#8)
    s3 = get_s3_client()
    ensure_s3_bucket(s3, BUCKET_NAME)

    wandb.login()
    wandb.init(project=WANDB_PROJECT, entity=WANDB_ENTITY)

    llm_profile = get_llm_profile(config)
    base_path = os.path.join("runs", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    os.makedirs(base_path, exist_ok=True)

    agent = agent_class(
        llm_profile,
        log_path=base_path,
        max_chat_round=args.max_chat_rounds,
        max_actions=args.max_actions,
        rounds_per_game=1,
        rag_episode_k=args.rag_episode_k,
        rag_concept_k=args.rag_concept_k,
        rag_episode_k_initial=args.rag_episode_k_initial,
        rag_concept_k_initial=args.rag_concept_k_initial,
        args=args,
    )

    if args.env_type == "scienceworld":
        db = SessionLocal()
        try:
            run_scienceworld_eval(agent, agent_name, args, llm_profile_name, s3, db)
        finally:
            db.close()
        wandb.finish()
        return

    eval_splits = args.splits or config["general"]["evaluate"]["splits"]
    eval_envs = config["general"]["evaluate"]["envs"]
    controllers = config["general"]["evaluate"]["controllers"]

    dataset_cfg = config.setdefault("dataset", {})
    dataset_cfg.setdefault("num_train_games", -1)
    dataset_cfg.setdefault("num_eval_games", -1)

    dataset_root_cfg = dataset_cfg.get("root")
    if not dataset_root_cfg:
        raise ValueError("Missing dataset.root in config")

    dataset_root = Path(os.path.expandvars(dataset_root_cfg)).resolve()
    if not dataset_root.exists():
        raise FileNotFoundError(f"Dataset root does not exist: {dataset_root}")
    print(
        "Available dataset splits:",
        sorted(p.name for p in dataset_root.iterdir() if p.is_dir()),
    )

    eval_id_path = (
        Path(os.path.expandvars(dataset_cfg["eval_id_data_path"])).resolve()
        if dataset_cfg.get("eval_id_data_path")
        else None
    )
    eval_ood_path = (
        Path(os.path.expandvars(dataset_cfg["eval_ood_data_path"])).resolve()
        if dataset_cfg.get("eval_ood_data_path")
        else None
    )

    from alfworld.agents.environment.alfred_tw_env import AlfredTWEnv

    for eval_env_type in eval_envs:
        for controller_type in (
            controllers if eval_env_type == "AlfredThorEnv" else ["tw"]
        ):
            for split_name in eval_splits:
                resolved_eval_path = (dataset_root / split_name).resolve()
                if not resolved_eval_path.exists():
                    raise FileNotFoundError(
                        f"Eval split path does not exist: {resolved_eval_path}"
                    )

                config["general"]["evaluate"]["env"]["type"] = eval_env_type
                config["controller"]["type"] = controller_type

                train_eval_mode = resolve_train_eval_mode(
                    split_name,
                    resolved_eval_path,
                    eval_id_path,
                    eval_ood_path,
                    dataset_cfg,
                )
                split_start_time = datetime.now(UTC)
                print(f"Evaluating split: {split_name} ({train_eval_mode})")
                print(f"Split start time: {split_start_time.isoformat()}")
                wandb.config.update(
                    {
                        "split": split_name,
                        "split_start_time": split_start_time.isoformat(),
                    },
                    allow_val_change=True,
                )
                if hasattr(agent, "read_only_memory"):
                    agent.read_only_memory = split_name == "valid_unseen"

                alfred_env = AlfredTWEnv(config, train_eval=train_eval_mode)
                env = alfred_env.init_env(batch_size=1)
                total_num_games = alfred_env.num_games

                if total_num_games == 0:
                    raise RuntimeError(
                        f"No ALFWorld games found for split={split_name} "
                        f"at path={resolved_eval_path} with train_eval_mode={train_eval_mode}"
                    )

                if args.game_ids is not None:
                    # Debug mode: specific game list
                    selected_games = sorted(
                        int(g.strip()) for g in args.game_ids.split(",") if g.strip()
                    )
                    invalid = [
                        g for g in selected_games if not (1 <= g <= total_num_games)
                    ]
                    if invalid:
                        raise ValueError(
                            f"Game IDs out of range [1, {total_num_games}]: {invalid}"
                        )
                    num_games_to_evaluate = len(selected_games)
                elif args.task_type is not None:
                    # Debug mode: one random game of the requested task type.
                    # Requires a pre-scan pass; afterwards reinitialise env from game 1.
                    task_type_index = _scan_task_types(env, total_num_games)
                    alfred_env2 = AlfredTWEnv(config, train_eval=train_eval_mode)
                    env = alfred_env2.init_env(batch_size=1)
                    matching = task_type_index.get(args.task_type, [])
                    if not matching:
                        raise RuntimeError(
                            f"No games found for task_type={args.task_type} in split={split_name}"
                        )
                    selected_games = [random.choice(matching)]
                    num_games_to_evaluate = 1
                else:
                    # Normal mode: all games or a random sample
                    if args.num_games <= 0:
                        num_games_to_evaluate = total_num_games
                        selected_games = list(range(1, total_num_games + 1))
                    else:
                        num_games_to_evaluate = min(args.num_games, total_num_games)
                        selected_games = sorted(
                            random.sample(
                                range(1, total_num_games + 1), num_games_to_evaluate
                            )
                        )

                print(f"Selected {num_games_to_evaluate} Games: {selected_games}")

                # Per-split metrics (reset each split) (#9 / #11)
                chat_round_list: list[int] = []
                error_list: list[int] = []
                success_list: list[int] = []
                failure_list: list[int] = []
                cumulative_successful_actions = cumulative_failing_actions = 0
                cumulative_successful_chat_rounds = cumulative_failing_chat_rounds = 0
                cumulative_successful_runtime = cumulative_failing_runtime = 0
                avg_actions_taken_per_successful_game = (
                    avg_actions_taken_per_failing_game
                ) = 0.0
                avg_chat_rounds_per_successful_game = (
                    avg_chat_rounds_per_failing_game
                ) = 0.0
                avg_runtime_per_successful_game = avg_runtime_per_failing_game = 0.0
                cumulative_runtime = 0.0
                num_games_evaluated = num_successes = 0
                error_adjusted_success_rate = 0.0  # always defined before use (#3)
                total_run_usage: dict[str, float] = {
                    "total_tokens": 0,
                    "prompt_tokens": 0,
                    "completion_tokens": 0,
                    "total_cost": 0.0,
                }

                # DB session guarded by try/finally to prevent leaks (#6, #7)
                db = SessionLocal()
                try:
                    experiment = ExperimentRun(
                        agent_name=agent_name,
                        llm_model=llm_profile_name,
                        eval_env_type=eval_env_type,
                        max_actions_per_game=args.max_actions,
                        max_chat_rounds=args.max_chat_rounds,
                        start_time=datetime.now(UTC),
                        split=split_name,
                        num_games=num_games_to_evaluate,
                    )
                    db.add(experiment)
                    db.commit()
                    db.refresh(experiment)
                    print(f"✅ Started DB Experiment Run ID: {experiment.id}")

                    # Persist agent prompts, transition graph, and git commit
                    experiment.agents_config = agent.agents_info
                    experiment.git_commit = get_git_commit()
                    experiment.git_branch = get_git_branch()
                    db.commit()
                    cache.set_cache(
                        f"agents_config:{experiment.id}",
                        json.dumps(agent.agents_info),
                        expire=86400,
                    )
                    print(
                        f"✅ Logged agents_config for Experiment Run ID: {experiment.id}"
                    )

                    # ALFWorld cycles games sequentially; env.reset() must be called
                    # for every index to advance the environment state even for skips.
                    for i in range(1, total_num_games + 1):
                        obs, info = env.reset()

                        if i not in selected_games:
                            continue

                        num_games_evaluated += 1
                        agent.set_environment(env, obs, info, i)
                        log_paths = agent.log_paths
                        print(
                            f"\n[Running Game #{i}] ({num_games_evaluated}/{num_games_to_evaluate})"
                        )

                        chat_result, error_message, elapsed_minutes = run_game(agent, i)
                        cumulative_runtime += elapsed_minutes

                        # Token usage — gather_usage_summary returns a nested dict:
                        # {"usage_including_cached_inference": {"total_cost": X, "<model>": {"prompt_tokens": ..., ...}}}
                        raw_usage = autogen.gather_usage_summary(
                            agent.group_chat.agents
                        )
                        usage_data = (raw_usage or {}).get(
                            "usage_including_cached_inference", {}
                        )
                        game_prompt_tokens = sum(
                            v.get("prompt_tokens", 0)
                            for v in usage_data.values()
                            if isinstance(v, dict)
                        )
                        game_completion_tokens = sum(
                            v.get("completion_tokens", 0)
                            for v in usage_data.values()
                            if isinstance(v, dict)
                        )
                        game_total_tokens = game_prompt_tokens + game_completion_tokens
                        game_total_cost = usage_data.get("total_cost", 0.0)
                        game_usage = usage_data  # truthy check below
                        if game_usage:
                            total_run_usage["prompt_tokens"] += game_prompt_tokens
                            total_run_usage["completion_tokens"] += (
                                game_completion_tokens
                            )
                            total_run_usage["total_tokens"] += game_total_tokens
                            total_run_usage["total_cost"] += game_total_cost
                            wandb.log(
                                {
                                    "game/total_tokens": game_total_tokens,
                                    "game/cost": game_total_cost,
                                },
                                step=num_games_evaluated,
                            )

                        # Log errors
                        if error_message:
                            error_list.append(i)
                            with open(log_paths["error_message_path"], "a") as f:
                                f.write(f"Run Chat: {error_message}\n")

                        # Write chat history
                        if chat_result and getattr(chat_result, "chat_history", []):
                            with open(log_paths["chat_history_path"], "w") as f:
                                for message in chat_result.chat_history:
                                    f.write("-" * 20 + "\n")
                                    for key in ["name", "role", "content"]:
                                        if key in message:
                                            f.write(
                                                f"{key}:\n{message[key]}\n"
                                                if key == "content"
                                                else f"{key}: {message[key]}\n"
                                            )
                                    for k, v in message.items():
                                        if k not in ["name", "role", "content"]:
                                            f.write(f"{k}: {v}\n")
                            chat_round_list.append(len(chat_result.chat_history))
                        else:
                            chat_round_list.append(-1)
                            with open(log_paths["chat_history_path"], "w") as f:
                                f.write(
                                    "Error Message: no chat history in chat result\n"
                                )

                        # Extract metadata from chat log
                        with open(log_paths["chat_history_path"]) as f:
                            chat_text = f.read()

                        transitions, belief_matches = extract_chat_metadata(
                            chat_text
                        )  # (#4)

                        transition_path = os.path.join(
                            os.path.dirname(log_paths["chat_history_path"]),
                            "transition_log.json",
                        )
                        with open(transition_path, "w") as f:
                            json.dump(transitions, f, indent=2)

                        agent.prev_episodic_memories.append(
                            {
                                "episode_number": num_games_evaluated,
                                "task_outcome": agent.task_status,
                                "memory": belief_matches
                                if belief_matches
                                else agent.curr_episodic_memory,
                            }
                        )

                        # Upload to S3
                        s3_key = None
                        try:
                            _s3_key = (
                                f"experiments/run_{experiment.id}/game_{i}_chat.txt"
                            )
                            s3.put_object(
                                Bucket=BUCKET_NAME,
                                Key=_s3_key,
                                Body=chat_text.encode("utf-8"),
                                ContentType="text/plain",
                            )
                            s3_key = _s3_key
                            print(f"☁️ Uploaded chat history to S3: {s3_key}")
                        except Exception as e:
                            print(f"⚠️ S3 upload failed: {e}")

                        # Update running metrics
                        success = agent.success
                        if success:
                            num_successes += 1
                            success_list.append(i)
                            cumulative_successful_actions += agent.num_actions_taken
                            cumulative_successful_chat_rounds += chat_round_list[-1]
                            cumulative_successful_runtime += elapsed_minutes
                            avg_actions_taken_per_successful_game = (
                                cumulative_successful_actions / num_successes
                            )
                            avg_chat_rounds_per_successful_game = (
                                cumulative_successful_chat_rounds / num_successes
                            )
                            avg_runtime_per_successful_game = (
                                cumulative_successful_runtime / num_successes
                            )
                        else:
                            num_failures = num_games_evaluated - num_successes
                            failure_list.append(i)
                            cumulative_failing_actions += agent.num_actions_taken
                            cumulative_failing_chat_rounds += chat_round_list[-1]
                            cumulative_failing_runtime += elapsed_minutes
                            avg_actions_taken_per_failing_game = (
                                cumulative_failing_actions / num_failures
                            )
                            avg_chat_rounds_per_failing_game = (
                                cumulative_failing_chat_rounds / num_failures
                            )
                            avg_runtime_per_failing_game = (
                                cumulative_failing_runtime / num_failures
                            )

                        success_rate = num_successes / num_games_evaluated
                        num_games_no_error = num_games_evaluated - len(
                            [g for g in error_list if g not in success_list]
                        )
                        if num_games_no_error > 0:
                            error_adjusted_success_rate = (
                                num_successes / num_games_no_error
                            )
                        else:
                            error_adjusted_success_rate = 0.0
                            print(
                                "No valid games completed due to errors. Success rate is 0."
                            )

                        wandb.log(
                            {
                                "split": split_name,
                                "game_no": i,
                                "success": int(success),
                                "actions_taken": agent.num_actions_taken,
                                "success_rate": success_rate,
                                "avg_actions_taken_per_successful_game": avg_actions_taken_per_successful_game,
                                "avg_chat_rounds_per_successful_game": avg_chat_rounds_per_successful_game,
                                "avg_runtime_per_successful_game": avg_runtime_per_successful_game,
                                "runtime": elapsed_minutes,
                                "cumulative_runtime": cumulative_runtime,
                                "chat_rounds": chat_round_list[-1],
                                "error_adjusted_success_rate": error_adjusted_success_rate,
                                "final/total_tokens": total_run_usage["total_tokens"],
                                "final/total_cost": total_run_usage["total_cost"],
                                "final/prompt_tokens": total_run_usage["prompt_tokens"],
                                "final/completion_tokens": total_run_usage[
                                    "completion_tokens"
                                ],
                            },
                            step=num_games_evaluated,
                        )

                        # Persist experiment-level totals after each game
                        try:
                            experiment.total_tokens = total_run_usage["total_tokens"]
                            experiment.total_cost = total_run_usage["total_cost"]
                            experiment.prompt_tokens = int(
                                total_run_usage["prompt_tokens"]
                            )
                            experiment.completion_tokens = int(
                                total_run_usage["completion_tokens"]
                            )
                            db.commit()
                            print(
                                f"✅ Database updated for Run ID: {experiment.id}"
                            )  # (#2)
                        except Exception as e:
                            print(f"⚠️ Database logging failed: {e}")
                            db.rollback()

                        concept_matches = re.findall(
                            r"CONCEPT DISCOVERED: \[(.*?)\]", chat_text, re.DOTALL
                        )
                        concept_matches = [c.strip() for c in concept_matches]
                        concept_matches = [
                            c
                            for c in concept_matches
                            if not c.upper().startswith("NO CONCEPT")
                        ]
                        episode = EpisodeRun(
                            experiment_id=experiment.id,
                            game_number=i,
                            success=bool(success),
                            actions_taken=agent.num_actions_taken,
                            chat_rounds=chat_round_list[-1],
                            runtime_minutes=elapsed_minutes,
                            error_message=str(error_message) if error_message else None,
                            transitions={"transitions": transitions},
                            belief_state={
                                "memory": belief_matches
                                if belief_matches
                                else agent.curr_episodic_memory
                            },
                            task=agent.task,
                            task_type=agent.adapter.infer_task_type(),
                            inadmissible_action_count=agent.adapter.count_inadmissible_actions(
                                log_paths["history_path"]
                            ),
                            concepts_learned=concept_matches
                            if concept_matches
                            else None,
                            prompt_tokens=game_prompt_tokens if game_usage else None,
                            completion_tokens=game_completion_tokens
                            if game_usage
                            else None,
                            episode_cost=game_total_cost if game_usage else None,
                            success_rate=success_rate,
                            error_adjusted_success_rate=error_adjusted_success_rate,
                            chat_history_s3_key=s3_key,
                        )
                        db.add(episode)
                        db.commit()
                        print(f"✅ Saved Game #{i} to PostgreSQL Database!")

                        # Per-game summary
                        print(f"[Ran Game #{i}]")
                        print(
                            f"Evaluation {num_games_evaluated} of {num_games_to_evaluate}"
                        )
                        print(f"Success: {success}")
                        print(f"Runtime: {elapsed_minutes:.2f} minutes")
                        print(
                            f"Actions Taken: {agent.num_actions_taken} out of {args.max_actions}"
                        )
                        print(f"Chat Rounds Taken: {chat_round_list[-1]}")
                        print(
                            f"Success Rate: {num_successes}/{num_games_evaluated} = {100 * success_rate:.2f}%"
                        )
                        print(
                            f"Average Actions per Successful Game: {avg_actions_taken_per_successful_game:.2f} out of {args.max_actions}"
                        )
                        print(
                            f"Average Chat Rounds per Successful Game: {avg_chat_rounds_per_successful_game:.2f} out of {args.max_chat_rounds}"
                        )
                        print(
                            f"Average Runtime per Successful Game: {avg_runtime_per_successful_game:.2f} minutes"
                        )
                        print(
                            f"Average Actions per Failing Game: {avg_actions_taken_per_failing_game:.2f} out of {args.max_actions}"
                        )
                        print(
                            f"Average Chat Rounds per Failing Game: {avg_chat_rounds_per_failing_game:.2f} out of {args.max_chat_rounds}"
                        )
                        print(
                            f"Average Runtime per Failing Game: {avg_runtime_per_failing_game:.2f} minutes"
                        )
                        print(f"Successes: {success_list}")
                        print(f"Failures: {failure_list}")
                        print(f"Errors: {error_list}")
                        print(
                            f"Error-Adjusted Success Rate: {num_successes}/{num_games_no_error} = "
                            f"{100 * error_adjusted_success_rate:.2f}%"
                        )
                        print(
                            f"Remaining Games: {selected_games[num_games_evaluated:]}"
                        )
                        total_seconds = int(cumulative_runtime * 60)
                        print(
                            f"Cumulative Runtime: "
                            f"{total_seconds // 3600:02}:{(total_seconds % 3600) // 60:02}:{total_seconds % 60:02}\n"
                        )

                        if not selected_games[num_games_evaluated:]:
                            break

                    experiment.end_time = datetime.now(UTC)
                    experiment.total_runtime_minutes = cumulative_runtime
                    experiment.success_rate = success_rate
                    experiment.error_adjusted_success_rate = error_adjusted_success_rate
                    experiment.num_errors = len(error_list)
                    experiment.avg_actions_per_successful_game = (
                        avg_actions_taken_per_successful_game
                    )
                    experiment.avg_chat_rounds_per_successful_game = (
                        avg_chat_rounds_per_successful_game
                    )
                    experiment.avg_runtime_per_successful_game = (
                        avg_runtime_per_successful_game
                    )
                    experiment.avg_actions_per_failing_game = (
                        avg_actions_taken_per_failing_game
                    )
                    experiment.avg_chat_rounds_per_failing_game = (
                        avg_chat_rounds_per_failing_game
                    )
                    experiment.avg_runtime_per_failing_game = (
                        avg_runtime_per_failing_game
                    )
                    db.commit()
                    print("✅ Experiment securely finalized in the database.")

                finally:
                    db.close()  # always runs, even on exception (#6)

                print(
                    f"Final Success Rate: {num_successes}/{num_games_evaluated} = "
                    f"{100 * (num_successes / num_games_evaluated if num_games_evaluated else 0):.2f}%"
                )
                print(
                    f"Final Error-Adjusted Success Rate: {num_successes}/{num_games_no_error} = "
                    f"{100 * error_adjusted_success_rate:.2f}%"
                )

    wandb.finish()


if __name__ == "__main__":
    main()
