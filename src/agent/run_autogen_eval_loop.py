import json
import os
import re

os.environ["TOKENIZERS_PARALLELISM"] = "false"

import argparse
import random
import time
from datetime import UTC, datetime

UTC = UTC

import yaml
from botocore.exceptions import ClientError  # <-- ADD THIS
from dotenv import load_dotenv

# ALFWorld & Agents
# External packages
import wandb  # Make sure to run `wandb login` beforehand
from src.agent.baseline_agent import BaselineAutogenAgent
from src.agent.gwt_agent import GWTAutogenAgent
from src.storage.database import Base, SessionLocal, engine
from src.storage.models import EpisodeRun, ExperimentRun
from src.storage.s3 import get_s3_client  # <-- ADD THIS

load_dotenv()  # Loads your .env file


def get_llm_profile(config_data):
    # Get the active profile name, defaulting to your GWT setup
    profile_name = os.getenv("ACTIVE_LLM_PROFILE", "gemini_with_deepseek_reasoner")
    print(f"🔄 Switching to LLM Profile: {profile_name}")

    # Safely grab the list of profiles from the YAML
    # (Handles whether your top level key is config_list or llm_profiles)
    profiles_list = config_data.get("config_list") or config_data.get(
        "llm_profiles", []
    )

    curr_profile = None

    # Loop through the list to find the dictionary containing our target profile
    for item in profiles_list:
        if isinstance(item, dict) and profile_name in item:
            curr_profile = item[profile_name]
            break

    if not curr_profile:
        print(
            f"⚠️ Profile '{profile_name}' not found. Falling back to the first available."
        )
        # Fall back to the very first profile in the YAML
        if profiles_list and isinstance(profiles_list[0], dict):
            fallback_key = list(profiles_list[0].keys())[0]
            curr_profile = profiles_list[0][fallback_key]
        else:
            raise ValueError("No valid LLM profiles found in config file!")

    # Resolve Environment Variables (e.g. ${GEMINI_API_KEY}) for the chosen profile
    for model_config in curr_profile:
        for key, value in list(model_config.items()):
            if (
                isinstance(value, str)
                and value.startswith("${")
                and value.endswith("}")
            ):
                env_var = value[2:-1]
                model_config[key] = os.getenv(env_var, "")

    return {
        "config_list": curr_profile,
        "cache_seed": 42,  # Fixed seed for reproducibility
        "temperature": 0.0,  # Standard for eval
    }


global_num_games_to_evaluate = 139
global_max_actions_per_game = 60
global_max_chat_rounds_per_game = 500
global_split_rounds_per_game = 1
base_path = os.path.join("runs", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
os.makedirs(base_path, exist_ok=True)


def parse_arguments():
    """
    Parse command-line arguments for evaluating Autogen Agents on the ALFWorld environment.
    """
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
        "--long_term_guidance", action="store_true", help="Enable long-term guidance"
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_arguments()

    # --- 1. DATABASE & SCHEMA SETUP ---
    from src.storage.database import SessionLocal, engine
    from src.storage.models import (  # Ensure Base is imported from models
        Base,
        ExperimentRun,
    )

    # This is the "Magic Line" that creates the tables in the Docker DB
    Base.metadata.create_all(bind=engine)
    db = SessionLocal()

    # 1. Agent selection (Keep your existing if/elif here)
    if args.baseline:
        agent_class = BaselineAutogenAgent
        agent_name = "BaselineAutogenAgent"
    elif args.gwt:
        agent_class = GWTAutogenAgent
        agent_name = "GWTAutogenAgent"

    # 2. Load config and get the model name
    llm_profile_name = os.getenv("ACTIVE_LLM_PROFILE", "gemini_free")
    with open(args.config_file) as reader:
        config = yaml.safe_load(reader)

    # 👇 S3 SETUP BLOCK 👇
    s3 = get_s3_client()
    bucket_name = "alfworld-experiments"

    # Check if bucket exists, create if it doesn't
    try:
        s3.head_bucket(Bucket=bucket_name)
    except ClientError:
        s3.create_bucket(Bucket=bucket_name)
        print(f"🪣 Created new S3 bucket: {bucket_name}")
    # 👆 END S3 SETUP BLOCK 👆

    # Initialize Weights & Biases
    wandb.login()
    wandb.init(
        project="cognitive_agents",
        entity="eduardocortes1100-university-of-california-berkeley",
    )

    # Setup API key
    # llm_config = {"config_list": [{"model": "gpt-4o", "api_key":os.environ.get("OPENAI_API_KEY")}]}
    llm_profile = get_llm_profile(config)

    # Initialize Agent
    agent = agent_class(
        llm_profile,
        log_path=base_path,
        max_chat_round=global_max_chat_rounds_per_game,
        max_actions=global_max_actions_per_game,
        rounds_per_game=global_split_rounds_per_game,
        args=args,
    )

    # Extract evaluation parameters
    eval_paths = config["general"]["evaluate"]["eval_paths"]
    eval_envs = config["general"]["evaluate"]["envs"]
    controllers = config["general"]["evaluate"]["controllers"]
    repeats = config["general"]["evaluate"]["repeats"]

    chat_round_list = []

    for eval_env_type in eval_envs:
        for controller_type in (
            controllers if eval_env_type == "AlfredThorEnv" else ["tw"]
        ):
            for eval_path in eval_paths:
                print(f"Evaluating: {eval_path}")

                # Configure the evaluation environment
                config["general"]["evaluate"]["env"]["type"] = eval_env_type
                config["dataset"]["eval_ood_data_path"] = eval_path
                config["controller"]["type"] = controller_type

                # 1. First, make sure you have the right import at the top of the file
                from alfworld.agents.environment.alfred_tw_env import AlfredTWEnv

                # 2. Then, around line 163, replace the getattr logic:
                try:
                    # Instead of: env_class = getattr(environment, eval_env_type)
                    # Use the direct class for TextWorld:
                    env_class = AlfredTWEnv
                    print(
                        "✅ Successfully loaded AlfredTWEnv for text-based evaluation."
                    )
                except ImportError:
                    print(
                        "❌ Could not find AlfredTWEnv. Check your alfworld installation."
                    )
                    raise

                alfred_env = env_class(config, train_eval="eval_out_of_distribution")
                env = alfred_env.init_env(batch_size=1)
                total_num_games = alfred_env.num_games

                # Random selection of evaluation games
                if global_num_games_to_evaluate > total_num_games:
                    num_games_to_evaluate = total_num_games
                else:
                    num_games_to_evaluate = global_num_games_to_evaluate

                selected_games = sorted(
                    random.sample(range(1, total_num_games + 1), num_games_to_evaluate)
                )
                # selected_games = [124, 125] #[35, 94, 124, 125] #[35, 52, 57, 69, 77, 79, 86, 93, 94, 107, 109, 121, 123, 124, 125, 139]
                # num_games_to_evaluate = len(selected_games)
                print(f"Selected {num_games_to_evaluate} Games: {selected_games}")

                # 👇 DATABASE BLOCK 👇
                db = SessionLocal()
                # 1. Create the overarching Experiment Record
                experiment = ExperimentRun(
                    agent_name=agent_name,
                    llm_model=llm_profile_name,
                    eval_env_type=eval_env_type,
                    long_term_guidance=args.long_term_guidance,
                    max_actions_per_game=global_max_actions_per_game,
                    max_chat_rounds=global_max_chat_rounds_per_game,
                    start_time=datetime.now(UTC),
                )
                db.add(experiment)
                db.commit()  # Saves it to Postgres
                db.refresh(
                    experiment
                )  # Retrieves the new ID auto-generated by Postgres
                print(f"✅ Started DB Experiment Run ID: {experiment.id}")
                s3 = get_s3_client()
                bucket_name = "alfworld-experiments"

                # Check if bucket exists, create if it doesn't
                try:
                    s3.head_bucket(Bucket=bucket_name)
                except ClientError:
                    s3.create_bucket(Bucket=bucket_name)
                    print(f"🪣 Created new S3 bucket: {bucket_name}")
                # 👆 END DATABASE BLOCK 👆

                error_list = []
                success_list = []
                failure_list = []

                # Track metrics
                cumulative_successful_actions = 0
                avg_actions_taken_per_successful_game = 0
                cumulative_failing_actions = 0
                avg_actions_taken_per_failing_game = 0

                cumulative_successful_chat_rounds = 0
                avg_chat_rounds_per_successful_game = 0
                cumulative_failing_chat_rounds = 0
                avg_chat_rounds_per_failing_game = 0

                cumulative_successful_runtime = 0
                avg_runtime_per_successful_game = 0
                cumulative_failing_runtime = 0
                avg_runtime_per_failing_game = 0

                cumulative_runtime = 0

                num_games_evaluated = 0
                num_successes = 0
                success_rate = 0
                num_games_no_error = 0

                # Initialize global usage tracking
                total_run_usage = {
                    "total_tokens": 0,
                    "prompt_tokens": 0,
                    "completion_tokens": 0,
                    "total_cost": 0.0,
                }

                # inner game loop
                for i in range(1, total_num_games + 1):
                    # reset environment
                    obs, info = env.reset()

                    if i not in selected_games:
                        print(f"Skipped Game #{i}")
                        continue

                    num_games_evaluated += 1
                    agent.set_environment(env, obs, info, i)
                    log_paths = agent.log_paths
                    print(f"\n[Running Game #{i}]")
                    print(
                        f"Evaluation {num_games_evaluated} of {num_games_to_evaluate}"
                    )

                    start_time = time.time()
                    # RUN GAME
                    max_retries = 3
                    retry_count = 0
                    game_completed = False

                    while retry_count < max_retries and not game_completed:
                        try:
                            print(
                                f"🚀 [Attempt {retry_count + 1}] Starting Game #{i}..."
                            )
                            chat_result, error_message = agent.run_chat(
                                agent.initial_message
                            )
                            game_completed = True
                        except Exception as e:
                            error_msg = str(e).upper()
                            # If it's a rate limit error, wait and retry
                            if "429" in error_msg or "RESOURCE_EXHAUSTED" in error_msg:
                                retry_count += 1
                                wait_time = 65 * retry_count
                                print(
                                    f"\n🛑 RATE LIMIT: Sleeping {wait_time}s then retrying Game #{i}..."
                                )
                                time.sleep(wait_time)
                            else:
                                chat_result = None
                                print(f"❌ Critical Error in Game {i}: {e}")
                                break  # Exit the while loop for non-rate errors
                    end_time = time.time()
                    # GAME OVER

                    import autogen

                    game_usage = autogen.gather_usage_summary(agent.group_chat.agents)

                    if game_usage:
                        # Update global totals
                        total_run_usage["total_tokens"] += game_usage.get(
                            "total_tokens", 0
                        )
                        total_run_usage["prompt_tokens"] += game_usage.get(
                            "prompt_tokens", 0
                        )
                        total_run_usage["completion_tokens"] += game_usage.get(
                            "completion_tokens", 0
                        )
                        total_run_usage["total_cost"] += game_usage.get("total_cost", 0)

                        # (Optional) Log per-game usage to wandb
                        wandb.log(
                            {
                                "game/total_tokens": game_usage.get("total_tokens", 0),
                                "game/cost": game_usage.get("total_cost", 0),
                            }
                        )

                    # Log errors
                    if error_message:
                        error_list.append(i)
                        with open(log_paths["error_message_path"], "a") as f:
                            f.write(f"Run Chat: {error_message}\n")

                    # Log chat history
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

                    # Read the chat history to extract meta-data
                    with open(log_paths["chat_history_path"]) as f:
                        chat_text = f.read()

                    transitions = []
                    transition_pattern = r"name: (.*)"
                    matches = re.findall(transition_pattern, chat_text)
                    for idx in range(len(matches) - 1):
                        transitions.append(
                            {"from": matches[idx], "to": matches[idx + 1], "step": idx}
                        )

                    # Save transitions to file
                    transition_path = os.path.join(
                        os.path.dirname(log_paths["chat_history_path"]),
                        "transition_log.json",
                    )
                    with open(transition_path, "w") as f:
                        json.dump(transitions, f, indent=2)

                    belief_state_pattern = r"Belief State: (.*)"
                    matches = re.findall(belief_state_pattern, chat_text, re.IGNORECASE)
                    if matches:
                        agent.prev_episodic_memories.append(
                            {
                                "episode_number": num_games_evaluated,
                                "task_outcome": agent.task_status,
                                "memory": matches,
                            }
                        )
                    else:
                        agent.prev_episodic_memories.append(
                            {
                                "episode_number": num_games_evaluated,
                                "task_outcome": agent.task_status,
                                "memory": agent.curr_episodic_memory,
                            }
                        )

                    # 👇 S3 UPLOAD BLOCK 👇
                    # Create a clean, organized folder structure in your S3 bucket
                    s3_key = f"experiments/run_{experiment.id}/game_{i}_chat.txt"

                    # Upload the raw text directly to S3
                    s3.put_object(
                        Bucket=bucket_name,
                        Key=s3_key,
                        Body=chat_text.encode("utf-8"),
                        ContentType="text/plain",
                    )
                    print(f"☁️ Uploaded chat history to S3 Vault: {s3_key}")
                    # 👆 END S3 UPLOAD BLOCK 👆

                    # Evaluate and log success
                    elapsed_minutes = (end_time - start_time) / 60
                    cumulative_runtime += elapsed_minutes

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
                        [game for game in error_list if game not in success_list]
                    )

                    if num_games_no_error > 0:
                        error_adjusted_success_rate = num_successes / num_games_no_error
                    else:
                        error_adjusted_success_rate = 0.0
                        print(
                            "No valid games completed due to errors. Success rate is 0."
                        )

                    wandb.log(
                        {
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

                    current_experiment_id = None  # Initialize a local variable

                    # Safely update the experiment using the already open 'db' session
                    try:
                        experiment.total_tokens = total_run_usage["total_tokens"]
                        experiment.total_cost = total_run_usage["total_cost"]
                        db.commit()
                        print(f"✅ Database updated for Run ID: {experiment.id}")
                    except Exception as e:
                        print(f"⚠️ Database logging failed: {e}")
                        db.rollback()  # Important: undoes the failed transaction so the loop doesn't break

                    print(f"🚀 FINAL RUN REPORT for ID: {current_experiment_id}")

                    # 👇 DATABASE BLOCK 👇
                    # 2. Package the episodic memory for the database
                    belief_data = (
                        {"memory": matches}
                        if matches
                        else {"memory": agent.curr_episodic_memory}
                    )

                    # 3. Create the Episode Record for this specific game
                    episode = EpisodeRun(
                        experiment_id=experiment.id,
                        game_number=i,
                        success=bool(success),
                        actions_taken=agent.num_actions_taken,
                        chat_rounds=chat_round_list[-1],
                        runtime_minutes=elapsed_minutes,
                        error_message=str(error_message) if error_message else None,
                        transitions={
                            "transitions": transitions
                        },  # Saves your extracted regex dict natively!
                        belief_state=belief_data,
                    )

                    db.add(episode)
                    db.commit()
                    print(f"✅ Saved Game #{i} to PostgreSQL Database!")
                    # 👆 END DATABASE BLOCK 👆

                    print(f"[Ran Game #{i}]")
                    print(
                        f"Evaluation {num_games_evaluated} of {num_games_to_evaluate}"
                    )
                    print(f"Success: {success}")
                    print(f"Runtime: {elapsed_minutes:.2f} minutes")
                    print(
                        f"Rounds Taken: {global_split_rounds_per_game - agent.rounds_left} out of {global_split_rounds_per_game}"
                    )
                    print(
                        f"Actions Taken: {agent.num_actions_taken} out of {global_max_actions_per_game}"
                    )
                    print(f"Chat Rounds Taken: {chat_round_list[-1]}")
                    print(
                        f"Success Rate: {num_successes}/{num_games_evaluated} = {100 * success_rate:.2f}%"
                    )
                    print(
                        f"Average Actions per Successful Game: {avg_actions_taken_per_successful_game:.2f} out of {global_max_actions_per_game}"
                    )
                    print(
                        f"Average Chat Rounds per Successful Game: {avg_chat_rounds_per_successful_game:.2f} out of {global_max_chat_rounds_per_game}"
                    )
                    print(
                        f"Average Runtime per Successful Game: {avg_runtime_per_successful_game:.2f} minutes"
                    )
                    print(
                        f"Average Actions per Failing Game: {avg_actions_taken_per_failing_game:.2f} out of {global_max_actions_per_game}"
                    )
                    print(
                        f"Average Chat Rounds per Failing Game: {avg_chat_rounds_per_failing_game:.2f} out of {global_max_chat_rounds_per_game}"
                    )
                    print(
                        f"Average Runtime per Failing Game: {avg_runtime_per_failing_game:.2f} minutes"
                    )
                    print(f"Successes: {success_list}")
                    print(f"Failures: {failure_list}")
                    print(f"Errors: {error_list}")
                    print(
                        f"Error-Adjusted Success Rate: {num_successes}/{num_games_no_error} = {100 * error_adjusted_success_rate if num_games_no_error > 0 else 0:.2f}%"
                    )
                    print(f"Remaining Games: {selected_games[num_games_evaluated:]}")

                    total_seconds = int(cumulative_runtime * 60)
                    hours = total_seconds // 3600
                    minutes = (total_seconds % 3600) // 60
                    secs = total_seconds % 60
                    print(f"Cumulative Runtime: {hours:02}:{minutes:02}:{secs:02}\n")

                    if not selected_games[num_games_evaluated:]:
                        break

                # 👇 DATABASE BLOCK 👇
                # 4. Finalize the Experiment in the DB
                experiment.end_time = datetime.now(UTC)
                experiment.total_runtime_minutes = cumulative_runtime
                db.commit()
                db.close()  # Always close the connection when done!
                print("✅ Experiment securely finalized in the database.")
                # 👆 END DATABASE BLOCK 👆

                # Final Success Summary
                print(
                    f"Final Success Rate: {num_successes}/{num_games_evaluated} = {100 * success_rate:.2f}%"
                )
                print(
                    f"Final Error-Adjusted Success Rate: {num_successes}/{num_games_no_error} = {100 * num_successes / num_games_no_error if num_games_no_error > 0 else 0:.2f}%"
                )

    wandb.finish()
