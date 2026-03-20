import argparse
import importlib
import json
import os
import random
import re
import signal
import subprocess
import time
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
import autogen
import yaml
from autogen.oai.client import OpenAIClient
from botocore.exceptions import ClientError
from dotenv import load_dotenv
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

import wandb
from src.agent.baseline_agent import BaselineAutogenAgent
from src.agent.env_adapter import (
    AndroidWorldAdapter,
    NetHackAdapter,
    ScienceWorldAdapter,
    TalesAdapter,
    WebArenaAdapter,
    infer_task_type,
)
from src.agent.gwt_agent import GWTAutogenAgent
from src.config.androidworld_runtime import (
    androidworld_install_timeout_from_env,
    prepare_androidworld_runtime,
)
from src.config.androidworld_validation import (
    validate_androidworld_runtime,
    wait_for_androidworld_device_ready,
)
from src.config.env_validation import require_env_vars
from src.config.schema_health import require_current_schema
from src.config.webarena_validation import validate_webarena_instance_urls
from src.storage import cache
from src.storage.database import SessionLocal
from src.storage.models import EpisodeRun, ExperimentRun
from src.storage.s3 import get_s3_client

os.environ["TOKENIZERS_PARALLELISM"] = "false"
load_dotenv()

BUCKET_NAME = "alfworld-experiments"
WANDB_PROJECT = "cognitive_agents"
WANDB_ENTITY = "eduardocortes1100-university-of-california-berkeley"
REPO_ROOT = Path(__file__).resolve().parents[1]
# DeepSeek's current quick-start pricing page lists a shared V3.2 price table
# for both deepseek-chat and deepseek-reasoner.
DEEPSEEK_PRICING_PER_1M: dict[str, tuple[float, float, float]] = {
    "deepseek-chat": (0.028, 0.28, 0.42),
    "deepseek-reasoner": (0.028, 0.28, 0.42),
}
_ACTIVE_EXPERIMENT_ID: int | None = None

_ANDROIDWORLD_SMOKE_TASK_GROUPS: dict[str, list[str]] = {
    "browser": ["BrowserMaze", "BrowserDraw", "BrowserMultiply"],
    "core": ["BrowserMaze", "MarkorCreateNote", "ContactsAddContact"],
}


def _coerce_text(value) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, (list, tuple, set)):
        parts = [_coerce_text(item).strip() for item in value]
        return " ".join(part for part in parts if part)
    return str(value)


def _is_androidworld_device_offline_error(exc: Exception) -> bool:
    parts = [str(exc).lower()]
    for attr in ("output", "stdout", "stderr"):
        value = getattr(exc, attr, None)
        if isinstance(value, bytes):
            parts.append(value.decode("utf-8", errors="ignore").lower())
        elif isinstance(value, str):
            parts.append(value.lower())
    return "device offline" in " ".join(parts)


def _is_androidworld_retryable_reset_error(exc: Exception) -> bool:
    message = str(exc).lower()
    return _is_androidworld_device_offline_error(exc) or (
        "could not get a11y tree" in message
    )


def _reset_androidworld_env(
    env,
    *,
    adb_path: str,
    console_port: int,
) -> None:
    wait_for_androidworld_device_ready(
        adb_path=adb_path,
        console_port=console_port,
    )
    try:
        env.reset(go_home=True)
        return
    except Exception as exc:
        if not _is_androidworld_retryable_reset_error(exc):
            raise
    wait_for_androidworld_device_ready(
        adb_path=adb_path,
        console_port=console_port,
    )
    env.reset(go_home=True)


def set_active_experiment(experiment_id: int | None) -> None:
    global _ACTIVE_EXPERIMENT_ID
    _ACTIVE_EXPERIMENT_ID = experiment_id


def mark_active_experiment_cancelled() -> None:
    if _ACTIVE_EXPERIMENT_ID is None:
        return

    db = SessionLocal()
    try:
        experiment = db.get(ExperimentRun, _ACTIVE_EXPERIMENT_ID)
        if experiment is None or experiment.status == "CANCELLED":
            return

        experiment.status = "CANCELLED"
        experiment.end_time = experiment.end_time or datetime.now(UTC)
        experiment.current_game_number = None
        experiment.current_game_label = None
        db.commit()
        print(f"⚠️ Marked experiment {_ACTIVE_EXPERIMENT_ID} as CANCELLED.")
    except Exception as exc:
        db.rollback()
        print(
            "⚠️ Failed to mark active experiment as CANCELLED: "
            f"{_ACTIVE_EXPERIMENT_ID} ({exc})"
        )
    finally:
        db.close()


def _raise_keyboard_interrupt(signum, frame) -> None:
    mark_active_experiment_cancelled()
    raise KeyboardInterrupt(f"Received signal {signum}")


signal.signal(signal.SIGINT, _raise_keyboard_interrupt)
signal.signal(signal.SIGTERM, _raise_keyboard_interrupt)


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
    for env_var in (
        "GIT_BRANCH",
        "BRANCH_NAME",
        "GITHUB_REF_NAME",
        "CI_COMMIT_REF_NAME",
    ):
        value = os.getenv(env_var)
        if value:
            return value

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


def get_usage_totals(agents) -> dict[str, float]:
    usage_data = (autogen.gather_usage_summary(agents) or {}).get(
        "usage_excluding_cached_inference", {}
    )
    prompt_tokens = int(
        sum(
            v.get("prompt_tokens", 0)
            for v in usage_data.values()
            if isinstance(v, dict)
        )
    )
    completion_tokens = int(
        sum(
            v.get("completion_tokens", 0)
            for v in usage_data.values()
            if isinstance(v, dict)
        )
    )
    total_cost = float(usage_data.get("total_cost") or 0.0)

    return {
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": prompt_tokens + completion_tokens,
        "total_cost": total_cost,
    }


def _deepseek_response_cost(response) -> float | None:
    usage = getattr(response, "usage", None)
    model = getattr(response, "model", None)
    if usage is None or model not in DEEPSEEK_PRICING_PER_1M:
        return None

    prompt_tokens = int(getattr(usage, "prompt_tokens", 0) or 0)
    completion_tokens = int(getattr(usage, "completion_tokens", 0) or 0)
    prompt_details = getattr(usage, "prompt_tokens_details", None)
    cached_tokens = int(getattr(prompt_details, "cached_tokens", 0) or 0)
    missed_tokens = max(prompt_tokens - cached_tokens, 0)
    cache_hit_price, cache_miss_price, output_price = DEEPSEEK_PRICING_PER_1M[model]

    return (
        cached_tokens * cache_hit_price
        + missed_tokens * cache_miss_price
        + completion_tokens * output_price
    ) / 1_000_000


def patch_autogen_costing() -> None:
    if getattr(OpenAIClient, "_cogagent_patched_cost", False):
        return

    original_cost = OpenAIClient.cost

    def patched_cost(self, response):
        deepseek_cost = _deepseek_response_cost(response)
        if deepseek_cost is not None:
            return deepseek_cost
        return original_cost(self, response)

    OpenAIClient.cost = patched_cost
    OpenAIClient._cogagent_patched_cost = True


patch_autogen_costing()


def get_usage_delta(
    current_totals: dict[str, float], previous_totals: dict[str, float]
) -> dict[str, float]:
    return {
        key: max(current_totals.get(key, 0) - previous_totals.get(key, 0), 0)
        for key in ("prompt_tokens", "completion_tokens", "total_tokens", "total_cost")
    }


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


def update_experiment_runtime_state(
    db,
    experiment: ExperimentRun,
    *,
    status: str | None = None,
    current_game_number: int | None = None,
    current_game_label: str | None = None,
    current_analyst_trace: str | None = None,
) -> None:
    updates = {
        "current_game_number": current_game_number,
        "current_game_label": current_game_label,
        "current_analyst_trace": current_analyst_trace,
    }
    if status is not None:
        updates["status"] = status
    persist_experiment_updates(db, experiment, **updates)


def persist_experiment_updates(db, experiment: ExperimentRun, **updates) -> None:
    for field_name, field_value in updates.items():
        setattr(experiment, field_name, field_value)

    try:
        db.commit()
        db.refresh(experiment)
        return
    except Exception as exc:
        db.rollback()
        print(
            "⚠️ Experiment state commit failed for "
            f"run {experiment.id}: {exc}. Retrying with a fresh DB session."
        )

    retry_db = SessionLocal()
    try:
        persisted = retry_db.get(ExperimentRun, experiment.id)
        if persisted is None:
            raise RuntimeError(f"Experiment run {experiment.id} no longer exists.")

        for field_name, field_value in updates.items():
            setattr(persisted, field_name, field_value)

        retry_db.commit()
        retry_db.refresh(persisted)

        for field_name in updates:
            setattr(experiment, field_name, getattr(persisted, field_name))
    finally:
        retry_db.close()


def finalize_experiment(
    db,
    experiment: ExperimentRun,
    *,
    cumulative_runtime: float,
    success_rate: float,
    error_adjusted_success_rate: float,
    error_count: int,
    avg_actions_per_successful_game: float,
    avg_chat_rounds_per_successful_game: float,
    avg_runtime_per_successful_game: float,
    avg_actions_per_failing_game: float,
    avg_chat_rounds_per_failing_game: float,
    avg_runtime_per_failing_game: float,
    status: str,
) -> None:
    persist_experiment_architecture_snapshot(db, experiment)
    persist_experiment_updates(
        db,
        experiment,
        end_time=datetime.now(UTC),
        total_runtime_minutes=cumulative_runtime,
        success_rate=success_rate,
        error_adjusted_success_rate=error_adjusted_success_rate,
        num_errors=error_count,
        avg_actions_per_successful_game=avg_actions_per_successful_game,
        avg_chat_rounds_per_successful_game=avg_chat_rounds_per_successful_game,
        avg_runtime_per_successful_game=avg_runtime_per_successful_game,
        avg_actions_per_failing_game=avg_actions_per_failing_game,
        avg_chat_rounds_per_failing_game=avg_chat_rounds_per_failing_game,
        avg_runtime_per_failing_game=avg_runtime_per_failing_game,
        status=status,
        current_game_number=None,
        current_game_label=None,
    )
    if _ACTIVE_EXPERIMENT_ID == experiment.id:
        set_active_experiment(None)


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

    required_env_vars: set[str] = set()

    for model_config in curr_profile:
        for key, value in list(model_config.items()):
            if (
                isinstance(value, str)
                and value.startswith("${")
                and value.endswith("}")
            ):
                env_var = value[2:-1]
                required_env_vars.add(env_var)
                model_config[key] = os.getenv(env_var, "")
        if model_config.get("model") in DEEPSEEK_PRICING_PER_1M:
            model_config.pop("price", None)

    if required_env_vars:
        require_env_vars(
            sorted(required_env_vars),
            context=f"LLM profile '{profile_name}'",
        )

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
        except KeyboardInterrupt as exc:
            try:
                persist_chat_artifacts(
                    agent,
                    note=(
                        "Interrupted during run_chat; recovered the latest partial "
                        "transcript from in-memory group chat state."
                    ),
                )
                error_path = agent.log_paths.get("error_message_path")
                if error_path:
                    _append_text_file(error_path, f"Run interrupted: {exc}\n")
            except Exception as persist_exc:
                print(
                    "⚠️ Failed to persist partial chat transcript during interrupt: "
                    f"{persist_exc}"
                )
            raise
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


def _write_text_file(path: str, text: str, *, mode: str = "w") -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, mode) as f:
        f.write(text)
        f.flush()
        os.fsync(f.fileno())


def _append_text_file(path: str, text: str) -> None:
    _write_text_file(path, text, mode="a")


def _collect_chat_messages(agent, chat_result=None) -> tuple[list[dict], str]:
    if chat_result and getattr(chat_result, "chat_history", None):
        return list(chat_result.chat_history), "chat_result"

    group_chat = getattr(agent, "group_chat", None)
    if group_chat and getattr(group_chat, "messages", None):
        return list(group_chat.messages), "group_chat"

    group_chat_manager = getattr(agent, "group_chat_manager", None)
    nested_group_chat = getattr(group_chat_manager, "groupchat", None)
    if nested_group_chat and getattr(nested_group_chat, "messages", None):
        return list(nested_group_chat.messages), "group_chat_manager"

    return [], "none"


def _format_chat_history(messages: list[dict], *, note: str | None = None) -> str:
    lines: list[str] = []
    if note:
        lines.append(f"NOTE: {note}")

    if not messages:
        lines.append("Error Message: no chat history in chat result or in-memory state")
        return "\n".join(lines) + "\n"

    for message in messages:
        lines.append("-" * 20)
        if isinstance(message, dict):
            for key in ["name", "role", "content"]:
                if key in message:
                    lines.append(
                        f"{key}:\n{message[key]}"
                        if key == "content"
                        else f"{key}: {message[key]}"
                    )
            for key, value in message.items():
                if key not in {"name", "role", "content"}:
                    lines.append(f"{key}: {value}")
            continue

        lines.append("content:")
        lines.append(str(message))

    return "\n".join(lines) + "\n"


def _synthesize_analyst_trace_from_messages(
    messages: list[dict], *, styles: bool = False
) -> str:
    if not messages:
        return ""
    console = Console(
        record=True,
        width=140,
        soft_wrap=True,
        force_terminal=styles,
        color_system="truecolor" if styles else None,
    )
    console.print(
        Panel(
            Text(
                "Fallback analyst trace reconstructed from stored group-chat messages. "
                "Use this when the structured per-step runtime trace is unavailable."
            ),
            title="[bold bright_white]Analyst Trace Fallback[/bold bright_white]",
            border_style="bright_white",
        )
    )

    for idx, message in enumerate(messages, start=1):
        if isinstance(message, dict):
            name = str(message.get("name") or message.get("role") or "unknown")
            role = str(message.get("role") or "unknown")
            content_value = message.get("content")
            metadata = {
                key: value
                for key, value in message.items()
                if key not in {"name", "role", "content"}
            }
        else:
            name = "unknown"
            role = "unknown"
            content_value = message
            metadata = {}

        if isinstance(content_value, str):
            content = content_value.rstrip() or "[empty]"
        elif content_value in (None, ""):
            content = "[empty]"
        else:
            content = json.dumps(content_value, indent=2, sort_keys=True, default=str)

        table = Table(expand=True, show_header=False, box=None)
        table.add_column("key", style="bold yellow", width=18)
        table.add_column("value", style="white")
        table.add_row(Text("speaker"), Text(name))
        table.add_row(Text("role"), Text(role))
        table.add_row(Text("raw_content"), Text(content))
        if metadata:
            table.add_row(
                Text("metadata"),
                Text(json.dumps(metadata, indent=2, sort_keys=True, default=str)),
            )

        console.print(
            Panel(
                table,
                title=f"[bold cyan]T{idx} | transcript | {role}[/bold cyan]",
                border_style="cyan",
            )
        )

    return console.export_text(styles=styles)


def persist_chat_artifacts(agent, *, chat_result=None, note: str | None = None) -> dict:
    log_paths = getattr(agent, "log_paths", {}) or {}
    chat_history_path = log_paths.get("chat_history_path")
    if not chat_history_path:
        return {
            "chat_text": "",
            "chat_rounds": -1,
            "transitions": [],
            "belief_matches": [],
            "analyst_trace": "",
        }

    messages, source = _collect_chat_messages(agent, chat_result=chat_result)
    effective_note = note
    if source != "chat_result" and messages and effective_note is None:
        effective_note = "Recovered transcript from in-memory group chat state."

    chat_text = _format_chat_history(messages, note=effective_note)
    _write_text_file(chat_history_path, chat_text)

    analyst_trace_path = log_paths.get("analyst_trace_path")
    analyst_trace_ansi_path = log_paths.get("analyst_trace_ansi_path")
    analyst_trace = get_agent_analyst_trace_text(agent, prefer_existing=True)
    analyst_trace_ansi = ""
    if not analyst_trace:
        analyst_trace = _synthesize_analyst_trace_from_messages(messages, styles=False)
        analyst_trace_ansi = _synthesize_analyst_trace_from_messages(
            messages, styles=True
        )
    else:
        getter = getattr(agent, "get_analyst_trace_ansi_text", None)
        if callable(getter):
            try:
                analyst_trace_ansi = getter()
            except Exception as exc:
                print(
                    f"⚠️ Analyst-trace ANSI collection failed from agent runtime: {exc}"
                )
    if analyst_trace_path and analyst_trace:
        _write_text_file(analyst_trace_path, analyst_trace)
    if analyst_trace_ansi_path and analyst_trace_ansi:
        _write_text_file(analyst_trace_ansi_path, analyst_trace_ansi)

    transitions, belief_matches = extract_chat_metadata(chat_text)
    transition_path = os.path.join(
        os.path.dirname(chat_history_path),
        "transition_log.json",
    )
    _write_text_file(transition_path, json.dumps(transitions, indent=2))

    return {
        "chat_text": chat_text,
        "chat_rounds": len(messages) if messages else -1,
        "transitions": transitions,
        "belief_matches": belief_matches,
        "analyst_trace": analyst_trace,
    }


def load_chat_artifacts_from_disk(chat_history_path: str) -> dict:
    if not chat_history_path or not os.path.exists(chat_history_path):
        return {
            "chat_text": "",
            "chat_rounds": -1,
            "transitions": [],
            "belief_matches": [],
            "analyst_trace": "",
        }

    chat_text = Path(chat_history_path).read_text()
    transitions, belief_matches = extract_chat_metadata(chat_text)
    chat_rounds = len(re.findall(r"^name:", chat_text, re.MULTILINE))
    analyst_trace_path = str(Path(chat_history_path).with_name("analyst_trace.txt"))
    analyst_trace = load_analyst_trace_from_disk(analyst_trace_path)
    return {
        "chat_text": chat_text,
        "chat_rounds": chat_rounds if chat_rounds > 0 else -1,
        "transitions": transitions,
        "belief_matches": belief_matches,
        "analyst_trace": analyst_trace,
    }


def ensure_chat_artifacts(
    agent, *, chat_result=None, note: str | None = None, prefer_existing: bool = False
) -> dict:
    chat_history_path = (getattr(agent, "log_paths", {}) or {}).get("chat_history_path")
    if prefer_existing and chat_history_path:
        existing = load_chat_artifacts_from_disk(chat_history_path)
        if existing["chat_text"]:
            return existing
    return persist_chat_artifacts(agent, chat_result=chat_result, note=note)


def get_agent_usage_totals(agent) -> dict[str, float]:
    usage_getter = getattr(agent, "get_usage_totals", None)
    if callable(usage_getter):
        try:
            candidate = usage_getter()
            if isinstance(candidate, dict):
                return {
                    "prompt_tokens": int(candidate.get("prompt_tokens", 0) or 0),
                    "completion_tokens": int(
                        candidate.get("completion_tokens", 0) or 0
                    ),
                    "total_tokens": int(candidate.get("total_tokens", 0) or 0),
                    "total_cost": float(candidate.get("total_cost", 0.0) or 0.0),
                }
        except Exception as exc:
            print(f"⚠️ Usage collection failed from agent runtime: {exc}")
    group_chat = getattr(agent, "group_chat", None)
    agents = getattr(group_chat, "agents", None) or []
    return get_usage_totals(agents)


def persist_experiment_usage_snapshot(
    db, experiment: ExperimentRun, usage_totals: dict[str, float]
) -> None:
    try:
        experiment.total_tokens = usage_totals["total_tokens"]
        experiment.total_cost = usage_totals["total_cost"]
        experiment.prompt_tokens = int(usage_totals["prompt_tokens"])
        experiment.completion_tokens = int(usage_totals["completion_tokens"])
        db.commit()
    except Exception as exc:
        print(f"⚠️ Database logging failed: {exc}")
        db.rollback()


def _round_metric(value: float | None) -> float | None:
    if value is None:
        return None
    return round(float(value), 6)


def _normalize_architecture_metric_text(text: str) -> str:
    normalized = re.sub(r"\b\d+\b", "#", str(text or "").lower())
    return re.sub(r"\s+", " ", normalized).strip()


def _extract_history_actions(history_path: str | None) -> list[str]:
    if not history_path or not os.path.exists(history_path):
        return []
    history_text = Path(history_path).read_text()
    return [
        _normalize_architecture_metric_text(action)
        for action in re.findall(r"action: '([^']*)'\. observation: '", history_text)
        if action and action != "None"
    ]


def _extract_observation_history(agent) -> list[str]:
    observations: list[str] = []
    for item in getattr(agent, "curr_episodic_memory", []) or []:
        payload = None
        if isinstance(item, str):
            try:
                payload = json.loads(item)
            except json.JSONDecodeError:
                payload = None
        elif isinstance(item, dict):
            payload = item
        if not isinstance(payload, dict):
            continue
        observation = payload.get("resulting_observation")
        if observation:
            observations.append(_normalize_architecture_metric_text(observation))
    return observations


def _extract_message_names(agent, chat_text: str) -> list[str]:
    messages = getattr(getattr(agent, "group_chat", None), "messages", None) or []
    if messages:
        return [str(message.get("name") or "") for message in messages if message]
    return re.findall(r"^name: (.*)", chat_text or "", re.MULTILINE)


def _build_fallback_architecture_metrics(
    agent, *, chat_text: str, log_paths: dict
) -> dict:
    message_names = _extract_message_names(agent, chat_text)
    thinking_count = sum(1 for name in message_names if name == "Thinking_Agent")
    belief_update_count = sum(
        1 for name in message_names if name == "Belief_State_Agent"
    )

    actions = _extract_history_actions(log_paths.get("history_path"))
    observations = _extract_observation_history(agent)
    repeated_action_count = max(len(actions) - len(set(actions)), 0)
    repeated_state_count = max(len(observations) - len(set(observations)), 0)

    return {
        "version": 1,
        "thinking_count": thinking_count,
        "belief_update_count": belief_update_count,
        "deliberation_count": thinking_count + belief_update_count,
        "terminal_status_reason": getattr(agent, "task_status_reason", None),
        "repeated_action_density": (
            repeated_action_count / len(actions) if actions else None
        ),
        "repeated_state_density": (
            repeated_state_count / len(observations) if observations else None
        ),
        "observation_novelty_rate": (
            len(set(observations)) / len(observations) if observations else None
        ),
        "grounded_entity_growth_rate": None,
        "unique_grounded_entities": None,
        "burst_count": None,
        "mean_burst_length": None,
        "burst_length_histogram": {},
        "burst_stop_reasons": {},
    }


def get_agent_architecture_metrics(
    agent,
    *,
    log_paths: dict,
    chat_text: str,
    prompt_tokens: int,
    completion_tokens: int,
) -> dict:
    getter = getattr(agent, "get_architecture_metrics", None)
    metrics: dict | None = None
    if callable(getter):
        try:
            candidate = getter()
            if isinstance(candidate, dict):
                metrics = candidate
        except Exception as exc:
            print(f"⚠️ Architecture-metrics collection failed from agent runtime: {exc}")
    if metrics is None:
        metrics = _build_fallback_architecture_metrics(
            agent, chat_text=chat_text, log_paths=log_paths
        )

    total_tokens = int(prompt_tokens) + int(completion_tokens)
    actions_taken = int(getattr(agent, "num_actions_taken", 0) or 0)
    deliberation_count = int(metrics.get("deliberation_count") or 0)

    return {
        **metrics,
        "prompt_tokens": int(prompt_tokens),
        "completion_tokens": int(completion_tokens),
        "total_tokens": total_tokens,
        "tokens_per_action": (
            _round_metric(total_tokens / actions_taken) if actions_taken else None
        ),
        "tokens_per_deliberation": (
            _round_metric(total_tokens / deliberation_count)
            if deliberation_count
            else None
        ),
        "repeated_action_density": _round_metric(
            metrics.get("repeated_action_density")
        ),
        "repeated_state_density": _round_metric(metrics.get("repeated_state_density")),
        "observation_novelty_rate": _round_metric(
            metrics.get("observation_novelty_rate")
        ),
        "grounded_entity_growth_rate": _round_metric(
            metrics.get("grounded_entity_growth_rate")
        ),
        "mean_burst_length": _round_metric(metrics.get("mean_burst_length")),
    }


def aggregate_architecture_metrics(
    episode_metrics: list[dict | None],
) -> dict | None:
    valid_metrics = [
        metrics for metrics in episode_metrics if isinstance(metrics, dict)
    ]
    if not valid_metrics:
        return None

    hist_stop_reasons: Counter[str] = Counter()
    hist_burst_lengths: Counter[str] = Counter()
    hist_option_stop_reasons: Counter[str] = Counter()
    hist_option_interrupt_reasons: Counter[str] = Counter()
    hist_terminal_status_reasons: Counter[str] = Counter()
    for metrics in valid_metrics:
        hist_stop_reasons.update(metrics.get("burst_stop_reasons") or {})
        hist_burst_lengths.update(metrics.get("burst_length_histogram") or {})
        hist_option_stop_reasons.update(metrics.get("option_stop_reasons") or {})
        hist_option_interrupt_reasons.update(
            metrics.get("option_interrupt_reasons") or {}
        )
        terminal_reason = metrics.get("terminal_status_reason")
        if terminal_reason:
            hist_terminal_status_reasons.update([str(terminal_reason)])

    def _sum_int(key: str) -> int:
        return sum(int(metrics.get(key) or 0) for metrics in valid_metrics)

    def _mean_float(key: str) -> float | None:
        values = [
            metrics.get(key)
            for metrics in valid_metrics
            if metrics.get(key) is not None
        ]
        if not values:
            return None
        return _round_metric(sum(float(value) for value in values) / len(values))

    version = max(int(metrics.get("version") or 1) for metrics in valid_metrics)
    return {
        "version": version,
        "episode_count": len(valid_metrics),
        "thinking_count": _sum_int("thinking_count"),
        "belief_update_count": _sum_int("belief_update_count"),
        "deliberation_count": _sum_int("deliberation_count"),
        "burst_count": _sum_int("burst_count"),
        "v2_revision_required_count": _sum_int("v2_revision_required_count"),
        "v2_revision_action_count": _sum_int("v2_revision_action_count"),
        "v2_direct_action_count": _sum_int("v2_direct_action_count"),
        "mean_tokens_per_action": _mean_float("tokens_per_action"),
        "mean_tokens_per_deliberation": _mean_float("tokens_per_deliberation"),
        "mean_repeated_action_density": _mean_float("repeated_action_density"),
        "mean_repeated_state_density": _mean_float("repeated_state_density"),
        "mean_observation_novelty_rate": _mean_float("observation_novelty_rate"),
        "mean_grounded_entity_growth_rate": _mean_float("grounded_entity_growth_rate"),
        "mean_unique_grounded_entities": _mean_float("unique_grounded_entities"),
        "mean_burst_length": _mean_float("mean_burst_length"),
        "burst_stop_reasons": dict(sorted(hist_stop_reasons.items())),
        "burst_length_histogram": dict(sorted(hist_burst_lengths.items())),
        "option_count": _sum_int("option_count"),
        "mean_option_length": _mean_float("mean_option_length"),
        "mean_option_progress_debt": _mean_float("mean_option_progress_debt"),
        "mean_option_revisitation_count": _mean_float("mean_option_revisitation_count"),
        "mean_option_family_value": _mean_float("mean_option_family_value"),
        "option_stop_reasons": dict(sorted(hist_option_stop_reasons.items())),
        "option_interrupt_count": _sum_int("option_interrupt_count"),
        "option_interrupt_reasons": dict(sorted(hist_option_interrupt_reasons.items())),
        "terminal_status_reasons": dict(sorted(hist_terminal_status_reasons.items())),
    }


def persist_experiment_architecture_snapshot(
    db,
    experiment: ExperimentRun,
) -> None:
    try:
        episodes = (
            db.query(EpisodeRun)
            .filter(EpisodeRun.experiment_id == experiment.id)
            .order_by(EpisodeRun.game_number.asc(), EpisodeRun.id.asc())
            .all()
        )
        experiment.architecture_metrics = aggregate_architecture_metrics(
            [episode.architecture_metrics for episode in episodes]
        )
        db.commit()
    except Exception as exc:
        print(f"⚠️ Architecture-metrics aggregation failed: {exc}")
        db.rollback()


def load_analyst_trace_from_disk(analyst_trace_path: str | None) -> str:
    if not analyst_trace_path or not os.path.exists(analyst_trace_path):
        return ""
    return Path(analyst_trace_path).read_text()


def get_agent_analyst_trace_text(agent, *, prefer_existing: bool = False) -> str:
    log_paths = getattr(agent, "log_paths", {}) or {}
    analyst_trace_path = log_paths.get("analyst_trace_path")
    if prefer_existing:
        existing = load_analyst_trace_from_disk(analyst_trace_path)
        if existing:
            return existing

    getter = getattr(agent, "get_analyst_trace_text", None)
    if callable(getter):
        try:
            analyst_trace = getter()
            if analyst_trace:
                return analyst_trace
        except Exception as exc:
            print(f"⚠️ Analyst-trace collection failed from agent runtime: {exc}")

    return load_analyst_trace_from_disk(analyst_trace_path)


def persist_live_analyst_trace(experiment_id: int, analyst_trace: str) -> None:
    db = SessionLocal()
    try:
        experiment = db.get(ExperimentRun, experiment_id)
        if experiment is None:
            return
        experiment.current_analyst_trace = analyst_trace or None
        db.commit()
    except Exception as exc:
        db.rollback()
        print(f"⚠️ Failed to persist live analyst trace: {exc}")
    finally:
        db.close()


def configure_live_analyst_trace(agent, experiment_id: int) -> None:
    def _callback(analyst_trace: str) -> None:
        persist_live_analyst_trace(experiment_id, analyst_trace)

    setattr(agent, "analyst_trace_callback", _callback)
    analyst_trace = get_agent_analyst_trace_text(agent, prefer_existing=True)
    if analyst_trace:
        _callback(analyst_trace)


def upload_chat_history_artifact(
    s3, *, experiment_id: int, game_number: int, chat_text: str
) -> str | None:
    if not chat_text:
        return None

    s3_key = f"experiments/run_{experiment_id}/game_{game_number}_chat.txt"
    try:
        s3.put_object(
            Bucket=BUCKET_NAME,
            Key=s3_key,
            Body=chat_text.encode("utf-8"),
            ContentType="text/plain",
        )
        print(f"☁️ Uploaded chat history to S3: {s3_key}")
        return s3_key
    except Exception as exc:
        print(f"⚠️ S3 upload failed: {exc}")
        return None


def extract_concepts(chat_text: str) -> list[str]:
    concept_matches = re.findall(r"CONCEPT DISCOVERED: \[(.*?)\]", chat_text, re.DOTALL)
    concept_matches = [c.strip() for c in concept_matches]
    return [c for c in concept_matches if not c.upper().startswith("NO CONCEPT")]


def _safe_infer_episode_task_type(adapter) -> int | None:
    if not (adapter and hasattr(adapter, "infer_task_type")):
        return None
    try:
        return adapter.infer_task_type()
    except Exception as exc:
        print(f"⚠️ Task-type inference failed during episode persistence: {exc}")
        return None


def _safe_count_inadmissible_actions(adapter, history_path: str | None) -> int | None:
    if not history_path or not (
        adapter and hasattr(adapter, "count_inadmissible_actions")
    ):
        return None
    try:
        return adapter.count_inadmissible_actions(history_path)
    except Exception as exc:
        print(
            f"⚠️ Inadmissible-action counting failed during episode persistence: {exc}"
        )
        return None


def persist_episode_run(
    db,
    *,
    experiment: ExperimentRun,
    game_number: int,
    agent,
    log_paths: dict,
    success: bool,
    chat_rounds: int,
    runtime_minutes: float,
    error_message: str | None,
    transitions: list[dict],
    belief_matches: list[str],
    chat_text: str,
    prompt_tokens: int,
    completion_tokens: int,
    episode_cost: float,
    success_rate: float,
    error_adjusted_success_rate: float,
    chat_history_s3_key: str | None,
    analyst_trace: str | None,
) -> EpisodeRun:
    adapter = getattr(agent, "adapter", None)
    task_type = _safe_infer_episode_task_type(adapter)
    inadmissible_action_count = _safe_count_inadmissible_actions(
        adapter, log_paths.get("history_path")
    )
    concepts_learned = extract_concepts(chat_text)
    architecture_metrics = get_agent_architecture_metrics(
        agent,
        log_paths=log_paths,
        chat_text=chat_text,
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
    )

    episode = EpisodeRun(
        experiment_id=experiment.id,
        game_number=game_number,
        success=bool(success),
        actions_taken=agent.num_actions_taken,
        chat_rounds=chat_rounds,
        runtime_minutes=runtime_minutes,
        error_message=error_message,
        transitions={"transitions": transitions},
        belief_state={
            "memory": belief_matches
            if belief_matches
            else getattr(agent, "curr_episodic_memory", [])
        },
        task=getattr(agent, "task", None),
        task_type=task_type,
        inadmissible_action_count=inadmissible_action_count,
        concepts_learned=concepts_learned if concepts_learned else None,
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        episode_cost=episode_cost,
        success_rate=success_rate,
        error_adjusted_success_rate=error_adjusted_success_rate,
        chat_history_s3_key=chat_history_s3_key,
        analyst_trace=analyst_trace,
        architecture_metrics=architecture_metrics,
    )
    db.add(episode)
    db.commit()
    persist_experiment_architecture_snapshot(db, experiment)
    return episode


def persist_interrupted_episode_run(
    db,
    *,
    experiment: ExperimentRun,
    agent,
    game_number: int,
    s3,
    total_run_usage: dict[str, float],
    elapsed_minutes: float,
    success_rate: float,
    error_adjusted_success_rate: float,
    error_message: str,
) -> dict[str, float]:
    chat_artifacts = ensure_chat_artifacts(agent, prefer_existing=True)
    analyst_trace = get_agent_analyst_trace_text(agent, prefer_existing=True)
    current_usage_totals = get_agent_usage_totals(agent)
    game_usage = get_usage_delta(current_usage_totals, total_run_usage)
    updated_totals = {
        key: max(total_run_usage[key], current_usage_totals[key])
        for key in total_run_usage
    }
    persist_experiment_usage_snapshot(db, experiment, updated_totals)

    s3_key = upload_chat_history_artifact(
        s3,
        experiment_id=experiment.id,
        game_number=game_number,
        chat_text=chat_artifacts["chat_text"],
    )
    persist_episode_run(
        db,
        experiment=experiment,
        game_number=game_number,
        agent=agent,
        log_paths=getattr(agent, "log_paths", {}) or {},
        success=False,
        chat_rounds=chat_artifacts["chat_rounds"],
        runtime_minutes=elapsed_minutes,
        error_message=error_message,
        transitions=chat_artifacts["transitions"],
        belief_matches=chat_artifacts["belief_matches"],
        chat_text=chat_artifacts["chat_text"],
        prompt_tokens=int(game_usage["prompt_tokens"]),
        completion_tokens=int(game_usage["completion_tokens"]),
        episode_cost=float(game_usage["total_cost"]),
        success_rate=success_rate,
        error_adjusted_success_rate=error_adjusted_success_rate,
        chat_history_s3_key=s3_key,
        analyst_trace=analyst_trace,
    )
    return updated_totals


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
        "--max_actions", type=int, default=15, help="Max environment actions per game"
    )
    parser.add_argument(
        "--max_chat_rounds", type=int, default=75, help="Max chat rounds per game"
    )
    parser.add_argument(
        "--show_runtime_summary",
        action="store_true",
        default=False,
        help="Print a compact live runtime summary each step.",
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
        choices=[
            "alfworld",
            "scienceworld",
            "tales",
            "nethack",
            "webarena",
            "androidworld",
        ],
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
    parser.add_argument(
        "--tales-envs",
        nargs="+",
        default=None,
        dest="tales_envs",
        help="TALES environment names to run (default: all). E.g. --tales-envs JerichoEnvZork1",
    )
    parser.add_argument(
        "--nethack-variant",
        default="NetHackScore-v0",
        help="NLE variant to use (default: NetHackScore-v0).",
    )
    parser.add_argument(
        "--nethack-seeds",
        nargs="+",
        type=int,
        default=None,
        dest="nethack_seeds",
        help="Optional list of integer seeds for reproducible NetHack episodes.",
    )
    parser.add_argument(
        "--webarena-task-ids",
        nargs="+",
        type=int,
        default=None,
        dest="webarena_task_ids",
        help="BrowserGym WebArena task ids to run (default: all registered task ids, sampled by --num_games if set).",
    )
    parser.add_argument(
        "--webarena-env-id",
        default="browsergym/webarena",
        dest="webarena_env_id",
        help="Gymnasium environment id to use for WebArena-style tasks.",
    )
    parser.add_argument(
        "--androidworld-tasks",
        nargs="+",
        default=None,
        dest="androidworld_tasks",
        help="Specific AndroidWorld task types to run (default: all in the selected suite family).",
    )
    parser.add_argument(
        "--androidworld-smoke-suite",
        choices=sorted(_ANDROIDWORLD_SMOKE_TASK_GROUPS),
        default=None,
        dest="androidworld_smoke_suite",
        help="Named AndroidWorld smoke suite of transferable CIP tasks.",
    )
    parser.add_argument(
        "--androidworld-suite-family",
        default="android_world",
        dest="androidworld_suite_family",
        help="AndroidWorld suite family to use (default: android_world).",
    )
    parser.add_argument(
        "--androidworld-n-task-combinations",
        type=int,
        default=1,
        dest="androidworld_n_task_combinations",
        help="Number of task instances to materialize per AndroidWorld task template.",
    )
    parser.add_argument(
        "--androidworld-task-random-seed",
        type=int,
        default=30,
        dest="androidworld_task_random_seed",
        help="Seed used to materialize AndroidWorld task parameterizations.",
    )
    parser.add_argument(
        "--androidworld-perform-emulator-setup",
        action="store_true",
        default=False,
        dest="androidworld_perform_emulator_setup",
        help="Perform one-time AndroidWorld emulator app setup before evaluation.",
    )
    parser.add_argument(
        "--androidworld-console-port",
        type=int,
        default=5554,
        dest="androidworld_console_port",
        help="Android emulator console port (default: 5554).",
    )
    parser.add_argument(
        "--androidworld-grpc-port",
        type=int,
        default=8554,
        dest="androidworld_grpc_port",
        help="Android emulator gRPC port (default: 8554).",
    )
    parser.add_argument(
        "--androidworld-adb-path",
        default=None,
        dest="androidworld_adb_path",
        help="Explicit adb path for AndroidWorld.",
    )
    parser.add_argument(
        "--androidworld-adb-install-timeout",
        type=float,
        default=androidworld_install_timeout_from_env(),
        dest="androidworld_adb_install_timeout",
        help="Extended timeout in seconds for AndroidWorld APK installs during controller startup.",
    )
    parser.add_argument(
        "--render",
        action="store_true",
        default=False,
        help="Render the environment to the terminal each step (visual only, no extra LLM tokens).",
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
    candidate_games: list[tuple[int, str, int]] = []
    game_no = 0
    for task_name in task_names:
        total_vars = sw_env.get_max_variations(task_name)
        num_vars = (
            min(args.sw_variations, total_vars) if args.sw_variations else total_vars
        )
        for var_idx in range(num_vars):
            game_no += 1
            candidate_games.append((game_no, task_name, var_idx))

    if args.num_games > 0:
        selected_games = random.sample(
            candidate_games, k=min(args.num_games, len(candidate_games))
        )
        selected_games.sort(key=lambda game: game[0])
    else:
        selected_games = candidate_games

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
        status="RUNNING",
        split="scienceworld",
        num_games=0,  # updated after counting
    )
    db.add(experiment)
    db.commit()
    db.refresh(experiment)
    set_active_experiment(experiment.id)
    print(f"✅ Started DB Experiment Run ID: {experiment.id}")

    experiment.agents_config = agent.agents_info
    experiment.git_commit = get_git_commit()
    experiment.git_branch = get_git_branch()
    db.commit()

    total_games = len(selected_games)
    experiment.num_games = total_games
    experiment.selected_games = [game_no for game_no, _, _ in selected_games]
    experiment.selected_games_display = compress_game_list(experiment.selected_games)
    db.commit()

    try:
        for game_no, task_name, var_idx in selected_games:
            num_games_evaluated += 1
            update_experiment_runtime_state(
                db,
                experiment,
                current_game_number=game_no,
                current_game_label=f"Game #{game_no} | {task_name} var {var_idx}",
            )

            sw_env.load(task_name, var_idx)
            obs, info = sw_env.reset()
            adapter = ScienceWorldAdapter(sw_env, obs, info, task_name=task_name)
            agent.set_environment(sw_env, obs, info, game_no, adapter=adapter)
            configure_live_analyst_trace(agent, experiment.id)
            log_paths = agent.log_paths

            print(
                f"\n[Running Game #{game_no}] "
                f"({num_games_evaluated}/{total_games}) task={task_name} var={var_idx}"
            )
            try:
                game_start_time = time.time()
                chat_result, error_message, elapsed_minutes = run_game(agent, game_no)
                chat_artifacts = persist_chat_artifacts(agent, chat_result=chat_result)
                chat_text = chat_artifacts["chat_text"]
                transitions = chat_artifacts["transitions"]
                belief_matches = chat_artifacts["belief_matches"]
                chat_round_list.append(chat_artifacts["chat_rounds"])
                cumulative_runtime += elapsed_minutes

                current_usage_totals = get_agent_usage_totals(agent)
                game_usage = get_usage_delta(current_usage_totals, total_run_usage)
                game_prompt_tokens = int(game_usage["prompt_tokens"])
                game_completion_tokens = int(game_usage["completion_tokens"])
                game_total_tokens = int(game_usage["total_tokens"])
                game_total_cost = float(game_usage["total_cost"])
                total_run_usage = {
                    key: max(total_run_usage[key], current_usage_totals[key])
                    for key in total_run_usage
                }
                if any(value > 0 for value in game_usage.values()):
                    wandb.log(
                        {
                            "game/total_tokens": game_total_tokens,
                            "game/cost": game_total_cost,
                        },
                        step=num_games_evaluated,
                    )

                if error_message:
                    error_list.append(game_no)
                    _append_text_file(
                        log_paths["error_message_path"], f"Run Chat: {error_message}\n"
                    )

                agent.prev_episodic_memories.append(
                    {
                        "episode_number": num_games_evaluated,
                        "task_outcome": agent.task_status,
                        "memory": belief_matches
                        if belief_matches
                        else agent.curr_episodic_memory,
                    }
                )

                s3_key = upload_chat_history_artifact(
                    s3,
                    experiment_id=experiment.id,
                    game_number=game_no,
                    chat_text=chat_text,
                )

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
                    avg_runtime_per_failing_game = (
                        cumulative_failing_runtime / num_failures
                    )

                success_rate = num_successes / num_games_evaluated
                num_games_no_error = num_games_evaluated - len(
                    [g for g in error_list if g not in success_list]
                )
                error_adjusted_success_rate = (
                    num_successes / num_games_no_error
                    if num_games_no_error > 0
                    else 0.0
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

                persist_experiment_usage_snapshot(db, experiment, total_run_usage)
                persist_episode_run(
                    db,
                    experiment=experiment,
                    game_number=game_no,
                    agent=agent,
                    log_paths=log_paths,
                    success=success,
                    chat_rounds=chat_round_list[-1],
                    runtime_minutes=elapsed_minutes,
                    error_message=str(error_message) if error_message else None,
                    transitions=transitions,
                    belief_matches=belief_matches,
                    chat_text=chat_text,
                    prompt_tokens=game_prompt_tokens,
                    completion_tokens=game_completion_tokens,
                    episode_cost=game_total_cost,
                    success_rate=success_rate,
                    error_adjusted_success_rate=error_adjusted_success_rate,
                    chat_history_s3_key=s3_key,
                    analyst_trace=get_agent_analyst_trace_text(
                        agent, prefer_existing=True
                    ),
                )
                print(f"✅ Saved Game #{game_no} to PostgreSQL Database!")

                print(f"[Ran Game #{game_no}] task={task_name} var={var_idx}")
                print(
                    f"Success: {success} | Actions: {agent.num_actions_taken} | Runtime: {elapsed_minutes:.2f}m"
                )
                print(
                    f"Success Rate: {num_successes}/{num_games_evaluated} = {100 * success_rate:.2f}%"
                )
            except KeyboardInterrupt as exc:
                elapsed_minutes = (time.time() - game_start_time) / 60
                cumulative_runtime += elapsed_minutes
                if game_no not in error_list:
                    error_list.append(game_no)
                success_rate = (
                    num_successes / num_games_evaluated if num_games_evaluated else 0.0
                )
                num_games_no_error = num_games_evaluated - len(
                    [g for g in error_list if g not in success_list]
                )
                error_adjusted_success_rate = (
                    num_successes / num_games_no_error
                    if num_games_no_error > 0
                    else 0.0
                )
                total_run_usage = persist_interrupted_episode_run(
                    db,
                    experiment=experiment,
                    agent=agent,
                    game_number=game_no,
                    s3=s3,
                    total_run_usage=total_run_usage,
                    elapsed_minutes=elapsed_minutes,
                    success_rate=success_rate,
                    error_adjusted_success_rate=error_adjusted_success_rate,
                    error_message=f"Run interrupted: {exc}",
                )
                print(
                    f"⚠️ Saved partial interrupted Game #{game_no} to PostgreSQL Database."
                )
                raise
    except KeyboardInterrupt:
        finalize_experiment(
            db,
            experiment,
            cumulative_runtime=cumulative_runtime,
            success_rate=success_rate if num_games_evaluated else 0.0,
            error_adjusted_success_rate=error_adjusted_success_rate,
            error_count=len(error_list),
            avg_actions_per_successful_game=avg_actions_taken_per_successful_game,
            avg_chat_rounds_per_successful_game=avg_chat_rounds_per_successful_game,
            avg_runtime_per_successful_game=avg_runtime_per_successful_game,
            avg_actions_per_failing_game=avg_actions_taken_per_failing_game,
            avg_chat_rounds_per_failing_game=avg_chat_rounds_per_failing_game,
            avg_runtime_per_failing_game=avg_runtime_per_failing_game,
            status="CANCELLED",
        )
        print("⚠️ ScienceWorld experiment cancelled.")
        raise

    finalize_experiment(
        db,
        experiment,
        cumulative_runtime=cumulative_runtime,
        success_rate=success_rate if num_games_evaluated else 0.0,
        error_adjusted_success_rate=error_adjusted_success_rate,
        error_count=len(error_list),
        avg_actions_per_successful_game=avg_actions_taken_per_successful_game,
        avg_chat_rounds_per_successful_game=avg_chat_rounds_per_successful_game,
        avg_runtime_per_successful_game=avg_runtime_per_successful_game,
        avg_actions_per_failing_game=avg_actions_taken_per_failing_game,
        avg_chat_rounds_per_failing_game=avg_chat_rounds_per_failing_game,
        avg_runtime_per_failing_game=avg_runtime_per_failing_game,
        status="CONCLUDED",
    )
    print("✅ ScienceWorld experiment finalized in the database.")

    print(
        f"Final Success Rate: {num_successes}/{num_games_evaluated} = "
        f"{100 * (num_successes / num_games_evaluated if num_games_evaluated else 0):.2f}%"
    )


# ── TALES eval loop ────────────────────────────────────────────────────────────


def run_tales_eval(agent, agent_name, args, llm_profile_name, s3, db):
    import gymnasium
    import tales

    # tales.envs is a list of registered env name strings.
    # Each env is registered in gymnasium as "tales/{env_name}-v0".
    all_env_names = tales.envs
    if args.tales_envs:
        selected_env_names = list(args.tales_envs)
    else:
        selected_env_names = list(all_env_names)
        random.shuffle(selected_env_names)

    if args.num_games > 0:
        selected_env_names = selected_env_names[: args.num_games]

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
        eval_env_type="tales",
        max_actions_per_game=args.max_actions,
        max_chat_rounds=args.max_chat_rounds,
        start_time=datetime.now(UTC),
        status="RUNNING",
        split="tales",
        num_games=0,
    )
    db.add(experiment)
    db.commit()
    db.refresh(experiment)
    set_active_experiment(experiment.id)
    print(f"✅ Started DB Experiment Run ID: {experiment.id}")

    experiment.agents_config = agent.agents_info
    experiment.git_commit = get_git_commit()
    experiment.git_branch = get_git_branch()
    db.commit()

    total_games = len(selected_env_names)
    experiment.num_games = total_games
    experiment.selected_games = list(range(1, total_games + 1))
    experiment.selected_games_display = compress_game_list(experiment.selected_games)
    db.commit()

    try:
        for game_no, env_name in enumerate(selected_env_names, start=1):
            num_games_evaluated += 1
            update_experiment_runtime_state(
                db,
                experiment,
                current_game_number=game_no,
                current_game_label=f"Game #{game_no} | {env_name}",
            )

            family = tales.env2task.get(env_name, "unknown")
            make_kwargs: dict = {"disable_env_checker": True}
            if family in ("jericho", "textworld"):
                make_kwargs["admissible_commands"] = True

            env = gymnasium.make(f"tales/{env_name}-v0", **make_kwargs)
            obs, info = env.reset()
            adapter = TalesAdapter(env, obs, info, env_name=env_name)
            agent.set_environment(
                env, adapter.observation, info, game_no, adapter=adapter
            )
            configure_live_analyst_trace(agent, experiment.id)
            log_paths = agent.log_paths

            print(
                f"\n[Running Game #{game_no}] "
                f"({num_games_evaluated}/{total_games}) env={env_name}"
            )
            try:
                game_start_time = time.time()
                chat_result, error_message, elapsed_minutes = run_game(agent, game_no)
                chat_artifacts = persist_chat_artifacts(agent, chat_result=chat_result)
                chat_text = chat_artifacts["chat_text"]
                transitions = chat_artifacts["transitions"]
                belief_matches = chat_artifacts["belief_matches"]
                chat_round_list.append(chat_artifacts["chat_rounds"])
                cumulative_runtime += elapsed_minutes

                current_usage_totals = get_agent_usage_totals(agent)
                game_usage = get_usage_delta(current_usage_totals, total_run_usage)
                game_prompt_tokens = int(game_usage["prompt_tokens"])
                game_completion_tokens = int(game_usage["completion_tokens"])
                game_total_tokens = int(game_usage["total_tokens"])
                game_total_cost = float(game_usage["total_cost"])
                total_run_usage = {
                    key: max(total_run_usage[key], current_usage_totals[key])
                    for key in total_run_usage
                }
                if any(value > 0 for value in game_usage.values()):
                    wandb.log(
                        {
                            "game/total_tokens": game_total_tokens,
                            "game/cost": game_total_cost,
                        },
                        step=num_games_evaluated,
                    )

                if error_message:
                    error_list.append(game_no)
                    _append_text_file(
                        log_paths["error_message_path"], f"Run Chat: {error_message}\n"
                    )

                agent.prev_episodic_memories.append(
                    {
                        "episode_number": num_games_evaluated,
                        "task_outcome": agent.task_status,
                        "memory": belief_matches
                        if belief_matches
                        else agent.curr_episodic_memory,
                    }
                )

                s3_key = upload_chat_history_artifact(
                    s3,
                    experiment_id=experiment.id,
                    game_number=game_no,
                    chat_text=chat_text,
                )

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
                    avg_runtime_per_failing_game = (
                        cumulative_failing_runtime / num_failures
                    )

                success_rate = num_successes / num_games_evaluated
                num_games_no_error = num_games_evaluated - len(
                    [g for g in error_list if g not in success_list]
                )
                error_adjusted_success_rate = (
                    num_successes / num_games_no_error
                    if num_games_no_error > 0
                    else 0.0
                )

                wandb.log(
                    {
                        "env_name": env_name,
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

                persist_experiment_usage_snapshot(db, experiment, total_run_usage)
                agent.task = env_name
                persist_episode_run(
                    db,
                    experiment=experiment,
                    game_number=game_no,
                    agent=agent,
                    log_paths=log_paths,
                    success=success,
                    chat_rounds=chat_round_list[-1],
                    runtime_minutes=elapsed_minutes,
                    error_message=str(error_message) if error_message else None,
                    transitions=transitions,
                    belief_matches=belief_matches,
                    chat_text=chat_text,
                    prompt_tokens=game_prompt_tokens,
                    completion_tokens=game_completion_tokens,
                    episode_cost=game_total_cost,
                    success_rate=success_rate,
                    error_adjusted_success_rate=error_adjusted_success_rate,
                    chat_history_s3_key=s3_key,
                    analyst_trace=get_agent_analyst_trace_text(
                        agent, prefer_existing=True
                    ),
                )
                print(f"✅ Saved Game #{game_no} to PostgreSQL Database!")

                print(f"[Ran Game #{game_no}] env={env_name}")
                print(
                    f"Success: {success} | Actions: {agent.num_actions_taken} | Runtime: {elapsed_minutes:.2f}m"
                )
                print(
                    f"Success Rate: {num_successes}/{num_games_evaluated} = {100 * success_rate:.2f}%"
                )
            except KeyboardInterrupt as exc:
                elapsed_minutes = (time.time() - game_start_time) / 60
                cumulative_runtime += elapsed_minutes
                if game_no not in error_list:
                    error_list.append(game_no)
                success_rate = (
                    num_successes / num_games_evaluated if num_games_evaluated else 0.0
                )
                num_games_no_error = num_games_evaluated - len(
                    [g for g in error_list if g not in success_list]
                )
                error_adjusted_success_rate = (
                    num_successes / num_games_no_error
                    if num_games_no_error > 0
                    else 0.0
                )
                total_run_usage = persist_interrupted_episode_run(
                    db,
                    experiment=experiment,
                    agent=agent,
                    game_number=game_no,
                    s3=s3,
                    total_run_usage=total_run_usage,
                    elapsed_minutes=elapsed_minutes,
                    success_rate=success_rate,
                    error_adjusted_success_rate=error_adjusted_success_rate,
                    error_message=f"Run interrupted: {exc}",
                )
                print(
                    f"⚠️ Saved partial interrupted Game #{game_no} to PostgreSQL Database."
                )
                raise
    except KeyboardInterrupt:
        finalize_experiment(
            db,
            experiment,
            cumulative_runtime=cumulative_runtime,
            success_rate=success_rate if num_games_evaluated else 0.0,
            error_adjusted_success_rate=error_adjusted_success_rate,
            error_count=len(error_list),
            avg_actions_per_successful_game=avg_actions_taken_per_successful_game,
            avg_chat_rounds_per_successful_game=avg_chat_rounds_per_successful_game,
            avg_runtime_per_successful_game=avg_runtime_per_successful_game,
            avg_actions_per_failing_game=avg_actions_taken_per_failing_game,
            avg_chat_rounds_per_failing_game=avg_chat_rounds_per_failing_game,
            avg_runtime_per_failing_game=avg_runtime_per_failing_game,
            status="CANCELLED",
        )
        print("⚠️ TALES experiment cancelled.")
        raise

    finalize_experiment(
        db,
        experiment,
        cumulative_runtime=cumulative_runtime,
        success_rate=success_rate if num_games_evaluated else 0.0,
        error_adjusted_success_rate=error_adjusted_success_rate,
        error_count=len(error_list),
        avg_actions_per_successful_game=avg_actions_taken_per_successful_game,
        avg_chat_rounds_per_successful_game=avg_chat_rounds_per_successful_game,
        avg_runtime_per_successful_game=avg_runtime_per_successful_game,
        avg_actions_per_failing_game=avg_actions_taken_per_failing_game,
        avg_chat_rounds_per_failing_game=avg_chat_rounds_per_failing_game,
        avg_runtime_per_failing_game=avg_runtime_per_failing_game,
        status="CONCLUDED",
    )
    print("✅ TALES experiment finalized in the database.")

    print(
        f"Final Success Rate: {num_successes}/{num_games_evaluated} = "
        f"{100 * (num_successes / num_games_evaluated if num_games_evaluated else 0):.2f}%"
    )


# ── NetHack eval loop ─────────────────────────────────────────────────────────


def run_nethack_eval(agent, agent_name, args, llm_profile_name, s3, db):
    import gymnasium
    import nle  # noqa: F401 — triggers gymnasium.register() for NLE envs

    variant = getattr(args, "nethack_variant", "NetHackScore-v0")
    render = getattr(args, "render", False)
    num_episodes = max(args.num_games, 1)
    seeds = getattr(args, "nethack_seeds", None) or [None] * num_episodes

    # Pad seeds list to match num_episodes
    if len(seeds) < num_episodes:
        seeds = list(seeds) + [None] * (num_episodes - len(seeds))

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
    success_rate = 0.0
    total_run_usage: dict[str, float] = {
        "total_tokens": 0,
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "total_cost": 0.0,
    }

    experiment = ExperimentRun(
        agent_name=agent_name,
        llm_model=llm_profile_name,
        eval_env_type="nethack",
        max_actions_per_game=args.max_actions,
        max_chat_rounds=args.max_chat_rounds,
        start_time=datetime.now(UTC),
        status="RUNNING",
        split="nethack",
        num_games=num_episodes,
    )
    db.add(experiment)
    db.commit()
    db.refresh(experiment)
    set_active_experiment(experiment.id)
    print(f"✅ Started DB Experiment Run ID: {experiment.id}")

    experiment.agents_config = agent.agents_info
    experiment.git_commit = get_git_commit()
    experiment.git_branch = get_git_branch()
    experiment.selected_games = list(range(1, num_episodes + 1))
    experiment.selected_games_display = compress_game_list(experiment.selected_games)
    db.commit()

    try:
        for game_no in range(1, num_episodes + 1):
            num_games_evaluated += 1
            seed = seeds[game_no - 1]
            episode_label = f"Game #{game_no} | {variant}" + (
                f" seed={seed}" if seed is not None else ""
            )
            update_experiment_runtime_state(
                db,
                experiment,
                current_game_number=game_no,
                current_game_label=episode_label,
            )

            env = gymnasium.make(variant)
            reset_kwargs = {"seed": seed} if seed is not None else {}
            obs, info = env.reset(**reset_kwargs)
            adapter = NetHackAdapter(env, obs, info, variant=variant, render=render)
            agent.set_environment(
                env, adapter.observation, info, game_no, adapter=adapter
            )
            configure_live_analyst_trace(agent, experiment.id)
            log_paths = agent.log_paths

            print(
                f"\n[Running Game #{game_no}] "
                f"({num_games_evaluated}/{num_episodes}) variant={variant}"
                + (f" seed={seed}" if seed is not None else "")
            )
            try:
                game_start_time = time.time()
                chat_result, error_message, elapsed_minutes = run_game(agent, game_no)
                chat_artifacts = persist_chat_artifacts(agent, chat_result=chat_result)
                chat_text = chat_artifacts["chat_text"]
                transitions = chat_artifacts["transitions"]
                belief_matches = chat_artifacts["belief_matches"]
                chat_round_list.append(chat_artifacts["chat_rounds"])
                cumulative_runtime += elapsed_minutes

                current_usage_totals = get_agent_usage_totals(agent)
                game_usage = get_usage_delta(current_usage_totals, total_run_usage)
                game_prompt_tokens = int(game_usage["prompt_tokens"])
                game_completion_tokens = int(game_usage["completion_tokens"])
                game_total_tokens = int(game_usage["total_tokens"])
                game_total_cost = float(game_usage["total_cost"])
                total_run_usage = {
                    key: max(total_run_usage[key], current_usage_totals[key])
                    for key in total_run_usage
                }
                if any(value > 0 for value in game_usage.values()):
                    wandb.log(
                        {
                            "game/total_tokens": game_total_tokens,
                            "game/cost": game_total_cost,
                        },
                        step=num_games_evaluated,
                    )

                if error_message:
                    error_list.append(game_no)
                    _append_text_file(
                        log_paths["error_message_path"], f"Run Chat: {error_message}\n"
                    )

                agent.prev_episodic_memories.append(
                    {
                        "episode_number": num_games_evaluated,
                        "task_outcome": agent.task_status,
                        "memory": belief_matches
                        if belief_matches
                        else agent.curr_episodic_memory,
                    }
                )

                s3_key = upload_chat_history_artifact(
                    s3,
                    experiment_id=experiment.id,
                    game_number=game_no,
                    chat_text=chat_text,
                )

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
                    avg_runtime_per_failing_game = (
                        cumulative_failing_runtime / num_failures
                    )

                success_rate = num_successes / num_games_evaluated
                num_games_no_error = num_games_evaluated - len(
                    [g for g in error_list if g not in success_list]
                )
                error_adjusted_success_rate = (
                    num_successes / num_games_no_error
                    if num_games_no_error > 0
                    else 0.0
                )

                wandb.log(
                    {
                        "variant": variant,
                        "game_no": game_no,
                        "success": int(success),
                        "actions_taken": agent.num_actions_taken,
                        "success_rate": success_rate,
                        "runtime": elapsed_minutes,
                        "cumulative_runtime": cumulative_runtime,
                        "chat_rounds": chat_round_list[-1],
                        "error_adjusted_success_rate": error_adjusted_success_rate,
                        "nethack/cumulative_reward": adapter.cumulative_reward,
                        "final/total_tokens": total_run_usage["total_tokens"],
                        "final/total_cost": total_run_usage["total_cost"],
                    },
                    step=num_games_evaluated,
                )

                persist_experiment_usage_snapshot(db, experiment, total_run_usage)
                agent.task = f"{variant} (episode {game_no})"
                persist_episode_run(
                    db,
                    experiment=experiment,
                    game_number=game_no,
                    agent=agent,
                    log_paths=log_paths,
                    success=success,
                    chat_rounds=chat_round_list[-1],
                    runtime_minutes=elapsed_minutes,
                    error_message=str(error_message) if error_message else None,
                    transitions=transitions,
                    belief_matches=belief_matches,
                    chat_text=chat_text,
                    prompt_tokens=game_prompt_tokens,
                    completion_tokens=game_completion_tokens,
                    episode_cost=game_total_cost,
                    success_rate=success_rate,
                    error_adjusted_success_rate=error_adjusted_success_rate,
                    chat_history_s3_key=s3_key,
                    analyst_trace=get_agent_analyst_trace_text(
                        agent, prefer_existing=True
                    ),
                )
                print(f"✅ Saved Game #{game_no} to PostgreSQL Database!")

                print(
                    f"[Ran Game #{game_no}] variant={variant} reward={adapter.cumulative_reward:.1f}"
                )
                print(
                    f"Success: {success} | Actions: {agent.num_actions_taken} | Runtime: {elapsed_minutes:.2f}m"
                )
                print(
                    f"Success Rate: {num_successes}/{num_games_evaluated} = {100 * success_rate:.2f}%"
                )
            except KeyboardInterrupt as exc:
                elapsed_minutes = (time.time() - game_start_time) / 60
                cumulative_runtime += elapsed_minutes
                if game_no not in error_list:
                    error_list.append(game_no)
                success_rate = (
                    num_successes / num_games_evaluated if num_games_evaluated else 0.0
                )
                num_games_no_error = num_games_evaluated - len(
                    [g for g in error_list if g not in success_list]
                )
                error_adjusted_success_rate = (
                    num_successes / num_games_no_error
                    if num_games_no_error > 0
                    else 0.0
                )
                total_run_usage = persist_interrupted_episode_run(
                    db,
                    experiment=experiment,
                    agent=agent,
                    game_number=game_no,
                    s3=s3,
                    total_run_usage=total_run_usage,
                    elapsed_minutes=elapsed_minutes,
                    success_rate=success_rate,
                    error_adjusted_success_rate=error_adjusted_success_rate,
                    error_message=f"Run interrupted: {exc}",
                )
                print(
                    f"⚠️ Saved partial interrupted Game #{game_no} to PostgreSQL Database."
                )
                raise
            finally:
                env.close()
    except KeyboardInterrupt:
        finalize_experiment(
            db,
            experiment,
            cumulative_runtime=cumulative_runtime,
            success_rate=success_rate if num_games_evaluated else 0.0,
            error_adjusted_success_rate=error_adjusted_success_rate,
            error_count=len(error_list),
            avg_actions_per_successful_game=avg_actions_taken_per_successful_game,
            avg_chat_rounds_per_successful_game=avg_chat_rounds_per_successful_game,
            avg_runtime_per_successful_game=avg_runtime_per_successful_game,
            avg_actions_per_failing_game=avg_actions_taken_per_failing_game,
            avg_chat_rounds_per_failing_game=avg_chat_rounds_per_failing_game,
            avg_runtime_per_failing_game=avg_runtime_per_failing_game,
            status="CANCELLED",
        )
        print("⚠️ NetHack experiment cancelled.")
        raise

    finalize_experiment(
        db,
        experiment,
        cumulative_runtime=cumulative_runtime,
        success_rate=success_rate if num_games_evaluated else 0.0,
        error_adjusted_success_rate=error_adjusted_success_rate,
        error_count=len(error_list),
        avg_actions_per_successful_game=avg_actions_taken_per_successful_game,
        avg_chat_rounds_per_successful_game=avg_chat_rounds_per_successful_game,
        avg_runtime_per_successful_game=avg_runtime_per_successful_game,
        avg_actions_per_failing_game=avg_actions_taken_per_failing_game,
        avg_chat_rounds_per_failing_game=avg_chat_rounds_per_failing_game,
        avg_runtime_per_failing_game=avg_runtime_per_failing_game,
        status="CONCLUDED",
    )
    print("✅ NetHack experiment finalized in the database.")

    print(
        f"Final Success Rate: {num_successes}/{num_games_evaluated} = "
        f"{100 * (num_successes / num_games_evaluated if num_games_evaluated else 0):.2f}%"
    )


def _build_androidworld_task_suite(args):
    from android_world import registry as aw_registry_module
    from android_world import suite_utils as aw_suite_utils

    task_registry = aw_registry_module.TaskRegistry()
    suite_family = getattr(args, "androidworld_suite_family", "android_world")
    task_registry_for_family = task_registry.get_registry(family=suite_family)
    requested_tasks = args.androidworld_tasks or None
    smoke_suite = getattr(args, "androidworld_smoke_suite", None)
    if requested_tasks and smoke_suite:
        raise ValueError(
            "Specify either --androidworld-tasks or --androidworld-smoke-suite, not both."
        )
    if smoke_suite:
        requested_tasks = list(_ANDROIDWORLD_SMOKE_TASK_GROUPS[smoke_suite])
    if requested_tasks:
        available_task_names = sorted(task_registry_for_family.keys())
        available_by_lower = {name.lower(): name for name in available_task_names}
        normalized_requested: list[str] = []
        unknown_requested: list[str] = []
        for requested_task in requested_tasks:
            canonical_name = available_by_lower.get(requested_task.lower())
            if canonical_name is None:
                unknown_requested.append(requested_task)
            else:
                normalized_requested.append(canonical_name)
        if unknown_requested:
            available_preview = ", ".join(available_task_names[:20])
            raise ValueError(
                "Unknown AndroidWorld task name(s): "
                f"{', '.join(unknown_requested)}.\n"
                f"Suite family: {suite_family}\n"
                "Examples of valid task names: "
                f"{available_preview}"
            )
        requested_tasks = normalized_requested
    suite = aw_suite_utils.create_suite(
        task_registry=task_registry_for_family,
        n_task_combinations=max(1, args.androidworld_n_task_combinations),
        seed=args.androidworld_task_random_seed,
        tasks=requested_tasks,
    )
    task_items: list[tuple[str, int, object]] = []
    for task_type, task_instances in suite.items():
        for task_index, task in enumerate(task_instances):
            task_items.append((task_type, task_index, task))
    return suite_family, task_items


def run_androidworld_eval(agent, agent_name, args, llm_profile_name, s3, db):
    from android_world.env import env_launcher

    adb_path = validate_androidworld_runtime(
        adb_path=args.androidworld_adb_path,
        console_port=args.androidworld_console_port,
    )
    prepare_androidworld_runtime(
        install_timeout_sec=args.androidworld_adb_install_timeout
    )
    suite_family, task_items = _build_androidworld_task_suite(args)
    if not task_items:
        raise FileNotFoundError("No AndroidWorld tasks were discovered.")

    selected_task_items = (
        random.sample(task_items, k=min(args.num_games, len(task_items)))
        if args.num_games > 0
        else task_items
    )

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
    success_rate = 0.0
    total_run_usage: dict[str, float] = {
        "total_tokens": 0,
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "total_cost": 0.0,
    }

    selected_labels = [
        f"{task_type}[{task_index}]" for task_type, task_index, _ in selected_task_items
    ]
    experiment = ExperimentRun(
        agent_name=agent_name,
        llm_model=llm_profile_name,
        eval_env_type="androidworld",
        max_actions_per_game=args.max_actions,
        max_chat_rounds=args.max_chat_rounds,
        start_time=datetime.now(UTC),
        status="RUNNING",
        split=suite_family,
        num_games=len(selected_task_items),
    )
    db.add(experiment)
    db.commit()
    db.refresh(experiment)
    set_active_experiment(experiment.id)
    print(f"✅ Started DB Experiment Run ID: {experiment.id}")

    experiment.agents_config = agent.agents_info
    experiment.git_commit = get_git_commit()
    experiment.git_branch = get_git_branch()
    experiment.selected_games = selected_labels
    experiment.selected_games_display = ", ".join(selected_labels)
    db.commit()

    env = None
    try:
        env = env_launcher.load_and_setup_env(
            console_port=args.androidworld_console_port,
            emulator_setup=args.androidworld_perform_emulator_setup,
            adb_path=adb_path,
            grpc_port=args.androidworld_grpc_port,
        )
        for game_no, (task_type, task_index, task) in enumerate(
            selected_task_items, start=1
        ):
            num_games_evaluated += 1
            episode_persisted = False
            task_label = f"{task_type}[{task_index}]"
            episode_label = f"Game #{game_no} | {task_label}"
            update_experiment_runtime_state(
                db,
                experiment,
                current_game_number=game_no,
                current_game_label=episode_label,
            )

            _reset_androidworld_env(
                env,
                adb_path=adb_path,
                console_port=args.androidworld_console_port,
            )
            task.initialize_task(env)
            state = env.get_state(wait_to_stabilize=True)
            goal_text = _coerce_text(getattr(task, "goal", "")).strip() or task_label
            template_text = _coerce_text(getattr(task, "template", "")).strip() or None
            adapter = AndroidWorldAdapter(
                env,
                state,
                task_name=task_type,
                goal=goal_text,
                template=template_text,
                score_provider=lambda task=task, env=env: float(
                    task.is_successful(env) or 0.0
                ),
            )
            info = {
                "task": goal_text,
                "task_type": task_type,
                "task_index": task_index,
                "suite_family": suite_family,
                "template": template_text,
            }
            agent.set_environment(
                env, adapter.observation, info, game_no, adapter=adapter
            )
            configure_live_analyst_trace(agent, experiment.id)
            log_paths = agent.log_paths

            print(
                f"\n[Running Game #{game_no}] "
                f"({num_games_evaluated}/{len(selected_task_items)}) task={task_label}"
            )
            try:
                chat_result, error_message, elapsed_minutes = run_game(agent, game_no)
                chat_artifacts = persist_chat_artifacts(agent, chat_result=chat_result)
                chat_text = chat_artifacts["chat_text"]
                transitions = chat_artifacts["transitions"]
                belief_matches = chat_artifacts["belief_matches"]
                chat_round_list.append(chat_artifacts["chat_rounds"])
                cumulative_runtime += elapsed_minutes

                current_usage_totals = get_agent_usage_totals(agent)
                game_usage = get_usage_delta(current_usage_totals, total_run_usage)
                game_prompt_tokens = int(game_usage["prompt_tokens"])
                game_completion_tokens = int(game_usage["completion_tokens"])
                game_total_tokens = int(game_usage["total_tokens"])
                game_total_cost = float(game_usage["total_cost"])
                total_run_usage = {
                    key: max(total_run_usage[key], current_usage_totals[key])
                    for key in total_run_usage
                }
                if any(value > 0 for value in game_usage.values()):
                    wandb.log(
                        {
                            "game/total_tokens": game_total_tokens,
                            "game/cost": game_total_cost,
                        },
                        step=num_games_evaluated,
                    )

                if error_message:
                    error_list.append(game_no)
                    _append_text_file(
                        log_paths["error_message_path"], f"Run Chat: {error_message}\n"
                    )

                agent.prev_episodic_memories.append(
                    {
                        "episode_number": num_games_evaluated,
                        "task_outcome": agent.task_status,
                        "memory": belief_matches
                        if belief_matches
                        else agent.curr_episodic_memory,
                    }
                )

                s3_key = upload_chat_history_artifact(
                    s3,
                    experiment_id=experiment.id,
                    game_number=game_no,
                    chat_text=chat_text,
                )

                success = bool(float(task.is_successful(env) or 0.0) >= 1.0)
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
                    avg_runtime_per_failing_game = (
                        cumulative_failing_runtime / num_failures
                    )

                success_rate = num_successes / num_games_evaluated
                num_games_no_error = num_games_evaluated - len(
                    [g for g in error_list if g not in success_list]
                )
                error_adjusted_success_rate = (
                    num_successes / num_games_no_error
                    if num_games_no_error > 0
                    else 0.0
                )

                wandb.log(
                    {
                        "game_no": game_no,
                        "success": int(success),
                        "actions_taken": agent.num_actions_taken,
                        "success_rate": success_rate,
                        "runtime": elapsed_minutes,
                        "cumulative_runtime": cumulative_runtime,
                        "chat_rounds": chat_round_list[-1],
                        "error_adjusted_success_rate": error_adjusted_success_rate,
                        "androidworld/task": task_label,
                        "androidworld/task_goal": goal_text,
                        "final/total_tokens": total_run_usage["total_tokens"],
                        "final/total_cost": total_run_usage["total_cost"],
                    },
                    step=num_games_evaluated,
                )

                persist_experiment_usage_snapshot(db, experiment, total_run_usage)
                agent.task = f"AndroidWorld ({task_label})"
                persist_episode_run(
                    db,
                    experiment=experiment,
                    game_number=game_no,
                    agent=agent,
                    log_paths=log_paths,
                    success=success,
                    chat_rounds=chat_round_list[-1],
                    runtime_minutes=elapsed_minutes,
                    error_message=str(error_message) if error_message else None,
                    transitions=transitions,
                    belief_matches=belief_matches,
                    chat_text=chat_text,
                    prompt_tokens=game_prompt_tokens,
                    completion_tokens=game_completion_tokens,
                    episode_cost=game_total_cost,
                    success_rate=success_rate,
                    error_adjusted_success_rate=error_adjusted_success_rate,
                    chat_history_s3_key=s3_key,
                    analyst_trace=get_agent_analyst_trace_text(
                        agent, prefer_existing=True
                    ),
                )
                episode_persisted = True
                print(f"✅ Saved Game #{game_no} to PostgreSQL Database!")
            except Exception as exc:
                if game_no not in error_list:
                    error_list.append(game_no)
                if (
                    not episode_persisted
                    and getattr(agent, "adapter", None) is not None
                ):
                    try:
                        total_run_usage = persist_interrupted_episode_run(
                            db,
                            experiment=experiment,
                            agent=agent,
                            game_number=game_no,
                            s3=s3,
                            total_run_usage=total_run_usage,
                            elapsed_minutes=0.0,
                            success_rate=success_rate if num_games_evaluated else 0.0,
                            error_adjusted_success_rate=error_adjusted_success_rate,
                            error_message=str(exc),
                        )
                    except Exception as persist_exc:
                        print(
                            "⚠️ Failed to persist interrupted AndroidWorld episode "
                            f"{game_no}: {persist_exc}"
                        )
                raise
            finally:
                try:
                    task.tear_down(env)
                except Exception:
                    pass
                try:
                    env.reset(go_home=True)
                except Exception:
                    pass
    except KeyboardInterrupt:
        finalize_experiment(
            db,
            experiment,
            cumulative_runtime=cumulative_runtime,
            success_rate=success_rate if num_games_evaluated else 0.0,
            error_adjusted_success_rate=error_adjusted_success_rate,
            error_count=len(error_list),
            avg_actions_per_successful_game=avg_actions_taken_per_successful_game,
            avg_chat_rounds_per_successful_game=avg_chat_rounds_per_successful_game,
            avg_runtime_per_successful_game=avg_runtime_per_successful_game,
            avg_actions_per_failing_game=avg_actions_taken_per_failing_game,
            avg_chat_rounds_per_failing_game=avg_chat_rounds_per_failing_game,
            avg_runtime_per_failing_game=avg_runtime_per_failing_game,
            status="CANCELLED",
        )
        print("⚠️ AndroidWorld experiment cancelled.")
        raise
    except Exception as exc:
        finalize_experiment(
            db,
            experiment,
            cumulative_runtime=cumulative_runtime,
            success_rate=success_rate if num_games_evaluated else 0.0,
            error_adjusted_success_rate=error_adjusted_success_rate,
            error_count=len(error_list) or 1,
            avg_actions_per_successful_game=avg_actions_taken_per_successful_game,
            avg_chat_rounds_per_successful_game=avg_chat_rounds_per_successful_game,
            avg_runtime_per_successful_game=avg_runtime_per_successful_game,
            avg_actions_per_failing_game=avg_actions_taken_per_failing_game,
            avg_chat_rounds_per_failing_game=avg_chat_rounds_per_failing_game,
            avg_runtime_per_failing_game=avg_runtime_per_failing_game,
            status="FAILED",
        )
        print(f"⚠️ AndroidWorld experiment failed: {exc}")
        raise
    finally:
        if env is not None:
            env.close()

    finalize_experiment(
        db,
        experiment,
        cumulative_runtime=cumulative_runtime,
        success_rate=success_rate if num_games_evaluated else 0.0,
        error_adjusted_success_rate=error_adjusted_success_rate,
        error_count=len(error_list),
        avg_actions_per_successful_game=avg_actions_taken_per_successful_game,
        avg_chat_rounds_per_successful_game=avg_chat_rounds_per_successful_game,
        avg_runtime_per_successful_game=avg_runtime_per_successful_game,
        avg_actions_per_failing_game=avg_actions_taken_per_failing_game,
        avg_chat_rounds_per_failing_game=avg_chat_rounds_per_failing_game,
        avg_runtime_per_failing_game=avg_runtime_per_failing_game,
        status="CONCLUDED",
    )
    print("✅ AndroidWorld experiment finalized in the database.")


def _discover_webarena_task_ids(task_id_args: list[int] | None) -> list[int]:
    if task_id_args:
        return list(dict.fromkeys(task_id_args))
    try:
        import browsergym.webarena.config as webarena_config
    except ImportError as exc:
        raise ImportError(
            "WebArena task discovery requires browsergym-webarena to be installed."
        ) from exc
    return list(webarena_config.TASK_IDS)


def _make_webarena_env(task_id: int, *, render: bool, env_id: str):
    import gymnasium

    import_candidates = ("browsergym.webarena", "webarena")
    last_import_error: Exception | None = None
    for module_name in import_candidates:
        try:
            importlib.import_module(module_name)
            last_import_error = None
            break
        except ImportError as exc:
            last_import_error = exc

    if last_import_error is not None:
        raise ImportError(
            "WebArena integration requires an installed WebArena-compatible package "
            "(for example `browsergym.webarena` or `webarena`)."
        ) from last_import_error

    full_env_id = env_id if env_id.endswith(f".{task_id}") else f"{env_id}.{task_id}"
    make_attempts = [
        {"headless": not render},
        {"disable_viewport": False, "headless": not render},
    ]
    last_make_error: Exception | None = None
    for kwargs in make_attempts:
        try:
            return gymnasium.make(full_env_id, **kwargs)
        except Exception as exc:
            last_make_error = exc
    raise RuntimeError(
        f"Unable to construct WebArena env `{full_env_id}`."
    ) from last_make_error


def run_webarena_eval(agent, agent_name, args, llm_profile_name, s3, db):
    render = getattr(args, "render", False)
    env_id = getattr(args, "webarena_env_id", "browsergym/webarena")
    validate_webarena_instance_urls()
    task_ids = _discover_webarena_task_ids(args.webarena_task_ids)
    if not task_ids:
        raise FileNotFoundError("No WebArena task ids were discovered.")

    selected_task_ids = (
        random.sample(task_ids, k=min(args.num_games, len(task_ids)))
        if args.num_games > 0
        else task_ids
    )
    selected_task_ids = sorted(selected_task_ids)

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
    success_rate = 0.0
    total_run_usage: dict[str, float] = {
        "total_tokens": 0,
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "total_cost": 0.0,
    }

    experiment = ExperimentRun(
        agent_name=agent_name,
        llm_model=llm_profile_name,
        eval_env_type="webarena",
        max_actions_per_game=args.max_actions,
        max_chat_rounds=args.max_chat_rounds,
        start_time=datetime.now(UTC),
        status="RUNNING",
        split="webarena",
        num_games=len(selected_task_ids),
    )
    db.add(experiment)
    db.commit()
    db.refresh(experiment)
    set_active_experiment(experiment.id)
    print(f"✅ Started DB Experiment Run ID: {experiment.id}")

    experiment.agents_config = agent.agents_info
    experiment.git_commit = get_git_commit()
    experiment.git_branch = get_git_branch()
    experiment.selected_games = selected_task_ids
    experiment.selected_games_display = compress_game_list(selected_task_ids)
    db.commit()

    try:
        for game_no, task_id in enumerate(selected_task_ids, start=1):
            num_games_evaluated += 1
            task_label = f"task {task_id}"
            episode_label = f"Game #{game_no} | {task_label}"
            update_experiment_runtime_state(
                db,
                experiment,
                current_game_number=game_no,
                current_game_label=episode_label,
            )

            env = _make_webarena_env(task_id, render=render, env_id=env_id)
            reset_result = env.reset()
            if isinstance(reset_result, tuple) and len(reset_result) >= 2:
                obs, info = reset_result[:2]
            else:
                obs, info = reset_result, {}
            info = dict(info or {})
            info.setdefault("task", task_label)
            info.setdefault("task_id", task_id)
            adapter = WebArenaAdapter(
                env,
                obs,
                info,
                task_config_path=f"{env_id}.{task_id}",
            )
            agent.set_environment(
                env, adapter.observation, info, game_no, adapter=adapter
            )
            configure_live_analyst_trace(agent, experiment.id)
            log_paths = agent.log_paths

            print(
                f"\n[Running Game #{game_no}] "
                f"({num_games_evaluated}/{len(selected_task_ids)}) task={task_label}"
            )
            try:
                game_start_time = time.time()
                chat_result, error_message, elapsed_minutes = run_game(agent, game_no)
                chat_artifacts = persist_chat_artifacts(agent, chat_result=chat_result)
                chat_text = chat_artifacts["chat_text"]
                transitions = chat_artifacts["transitions"]
                belief_matches = chat_artifacts["belief_matches"]
                chat_round_list.append(chat_artifacts["chat_rounds"])
                cumulative_runtime += elapsed_minutes

                current_usage_totals = get_agent_usage_totals(agent)
                game_usage = get_usage_delta(current_usage_totals, total_run_usage)
                game_prompt_tokens = int(game_usage["prompt_tokens"])
                game_completion_tokens = int(game_usage["completion_tokens"])
                game_total_tokens = int(game_usage["total_tokens"])
                game_total_cost = float(game_usage["total_cost"])
                total_run_usage = {
                    key: max(total_run_usage[key], current_usage_totals[key])
                    for key in total_run_usage
                }
                if any(value > 0 for value in game_usage.values()):
                    wandb.log(
                        {
                            "game/total_tokens": game_total_tokens,
                            "game/cost": game_total_cost,
                        },
                        step=num_games_evaluated,
                    )

                if error_message:
                    error_list.append(game_no)
                    _append_text_file(
                        log_paths["error_message_path"], f"Run Chat: {error_message}\n"
                    )

                agent.prev_episodic_memories.append(
                    {
                        "episode_number": num_games_evaluated,
                        "task_outcome": agent.task_status,
                        "memory": belief_matches
                        if belief_matches
                        else agent.curr_episodic_memory,
                    }
                )

                s3_key = upload_chat_history_artifact(
                    s3,
                    experiment_id=experiment.id,
                    game_number=game_no,
                    chat_text=chat_text,
                )

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
                    avg_runtime_per_failing_game = (
                        cumulative_failing_runtime / num_failures
                    )

                success_rate = num_successes / num_games_evaluated
                num_games_no_error = num_games_evaluated - len(
                    [g for g in error_list if g not in success_list]
                )
                error_adjusted_success_rate = (
                    num_successes / num_games_no_error
                    if num_games_no_error > 0
                    else 0.0
                )

                wandb.log(
                    {
                        "game_no": game_no,
                        "success": int(success),
                        "actions_taken": agent.num_actions_taken,
                        "success_rate": success_rate,
                        "runtime": elapsed_minutes,
                        "cumulative_runtime": cumulative_runtime,
                        "chat_rounds": chat_round_list[-1],
                        "error_adjusted_success_rate": error_adjusted_success_rate,
                        "webarena/task": task_label,
                        "webarena/page_url": (info or {}).get("page_url")
                        or (info or {}).get("url"),
                        "final/total_tokens": total_run_usage["total_tokens"],
                        "final/total_cost": total_run_usage["total_cost"],
                    },
                    step=num_games_evaluated,
                )

                persist_experiment_usage_snapshot(db, experiment, total_run_usage)
                agent.task = f"WebArena ({task_label})"
                persist_episode_run(
                    db,
                    experiment=experiment,
                    game_number=game_no,
                    agent=agent,
                    log_paths=log_paths,
                    success=success,
                    chat_rounds=chat_round_list[-1],
                    runtime_minutes=elapsed_minutes,
                    error_message=str(error_message) if error_message else None,
                    transitions=transitions,
                    belief_matches=belief_matches,
                    chat_text=chat_text,
                    prompt_tokens=game_prompt_tokens,
                    completion_tokens=game_completion_tokens,
                    episode_cost=game_total_cost,
                    success_rate=success_rate,
                    error_adjusted_success_rate=error_adjusted_success_rate,
                    chat_history_s3_key=s3_key,
                    analyst_trace=get_agent_analyst_trace_text(
                        agent, prefer_existing=True
                    ),
                )
                print(f"✅ Saved Game #{game_no} to PostgreSQL Database!")
                print(
                    f"Success: {success} | Actions: {agent.num_actions_taken} | Runtime: {elapsed_minutes:.2f}m"
                )
            except KeyboardInterrupt as exc:
                elapsed_minutes = (time.time() - game_start_time) / 60
                cumulative_runtime += elapsed_minutes
                if game_no not in error_list:
                    error_list.append(game_no)
                success_rate = (
                    num_successes / num_games_evaluated if num_games_evaluated else 0.0
                )
                num_games_no_error = num_games_evaluated - len(
                    [g for g in error_list if g not in success_list]
                )
                error_adjusted_success_rate = (
                    num_successes / num_games_no_error
                    if num_games_no_error > 0
                    else 0.0
                )
                total_run_usage = persist_interrupted_episode_run(
                    db,
                    experiment=experiment,
                    agent=agent,
                    game_number=game_no,
                    s3=s3,
                    total_run_usage=total_run_usage,
                    elapsed_minutes=elapsed_minutes,
                    success_rate=success_rate,
                    error_adjusted_success_rate=error_adjusted_success_rate,
                    error_message=f"Run interrupted: {exc}",
                )
                print(
                    f"⚠️ Saved partial interrupted Game #{game_no} to PostgreSQL Database."
                )
                raise
            finally:
                env.close()
    except KeyboardInterrupt:
        finalize_experiment(
            db,
            experiment,
            cumulative_runtime=cumulative_runtime,
            success_rate=success_rate if num_games_evaluated else 0.0,
            error_adjusted_success_rate=error_adjusted_success_rate,
            error_count=len(error_list),
            avg_actions_per_successful_game=avg_actions_taken_per_successful_game,
            avg_chat_rounds_per_successful_game=avg_chat_rounds_per_successful_game,
            avg_runtime_per_successful_game=avg_runtime_per_successful_game,
            avg_actions_per_failing_game=avg_actions_taken_per_failing_game,
            avg_chat_rounds_per_failing_game=avg_chat_rounds_per_failing_game,
            avg_runtime_per_failing_game=avg_runtime_per_failing_game,
            status="CANCELLED",
        )
        print("⚠️ WebArena experiment cancelled.")
        raise

    finalize_experiment(
        db,
        experiment,
        cumulative_runtime=cumulative_runtime,
        success_rate=success_rate if num_games_evaluated else 0.0,
        error_adjusted_success_rate=error_adjusted_success_rate,
        error_count=len(error_list),
        avg_actions_per_successful_game=avg_actions_taken_per_successful_game,
        avg_chat_rounds_per_successful_game=avg_chat_rounds_per_successful_game,
        avg_runtime_per_successful_game=avg_runtime_per_successful_game,
        avg_actions_per_failing_game=avg_actions_taken_per_failing_game,
        avg_chat_rounds_per_failing_game=avg_chat_rounds_per_failing_game,
        avg_runtime_per_failing_game=avg_runtime_per_failing_game,
        status="CONCLUDED",
    )
    print("✅ WebArena experiment finalized in the database.")


# ── Main ───────────────────────────────────────────────────────────────────────


def main():
    args = parse_arguments()
    random.seed(args.seed)  # reproducible game selection (#10)
    require_current_schema(context="evaluation startup")

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
    wandb.init(
        project=WANDB_PROJECT,
        entity=WANDB_ENTITY,
        settings=wandb.Settings(console="off"),
    )

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
        except KeyboardInterrupt:
            wandb.finish()
            raise SystemExit(130)
        finally:
            db.close()
        wandb.finish()
        return

    if args.env_type == "tales":
        db = SessionLocal()
        try:
            run_tales_eval(agent, agent_name, args, llm_profile_name, s3, db)
        except KeyboardInterrupt:
            wandb.finish()
            raise SystemExit(130)
        finally:
            db.close()
        wandb.finish()
        return

    if args.env_type == "nethack":
        db = SessionLocal()
        try:
            run_nethack_eval(agent, agent_name, args, llm_profile_name, s3, db)
        except KeyboardInterrupt:
            wandb.finish()
            raise SystemExit(130)
        finally:
            db.close()
        wandb.finish()
        return

    if args.env_type == "webarena":
        db = SessionLocal()
        try:
            run_webarena_eval(agent, agent_name, args, llm_profile_name, s3, db)
        except KeyboardInterrupt:
            wandb.finish()
            raise SystemExit(130)
        finally:
            db.close()
        wandb.finish()
        return

    if args.env_type == "androidworld":
        db = SessionLocal()
        try:
            run_androidworld_eval(agent, agent_name, args, llm_profile_name, s3, db)
        except KeyboardInterrupt:
            wandb.finish()
            raise SystemExit(130)
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

    try:
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
                        selected_games = sorted(
                            int(g.strip())
                            for g in args.game_ids.split(",")
                            if g.strip()
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

                    chat_round_list: list[int] = []
                    error_list: list[int] = []
                    success_list: list[int] = []
                    failure_list: list[int] = []
                    cumulative_successful_actions = cumulative_failing_actions = 0
                    cumulative_successful_chat_rounds = (
                        cumulative_failing_chat_rounds
                    ) = 0
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
                    error_adjusted_success_rate = 0.0
                    total_run_usage: dict[str, float] = {
                        "total_tokens": 0,
                        "prompt_tokens": 0,
                        "completion_tokens": 0,
                        "total_cost": 0.0,
                    }
                    success_rate = 0.0
                    num_games_no_error = 0

                    db = SessionLocal()
                    try:
                        experiment = ExperimentRun(
                            agent_name=agent_name,
                            llm_model=llm_profile_name,
                            eval_env_type=eval_env_type,
                            max_actions_per_game=args.max_actions,
                            max_chat_rounds=args.max_chat_rounds,
                            start_time=datetime.now(UTC),
                            status="RUNNING",
                            split=split_name,
                            num_games=num_games_to_evaluate,
                            selected_games=selected_games,
                            selected_games_display=compress_game_list(selected_games),
                        )
                        db.add(experiment)
                        db.commit()
                        db.refresh(experiment)
                        set_active_experiment(experiment.id)
                        print(f"✅ Started DB Experiment Run ID: {experiment.id}")

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

                        try:
                            for i in range(1, total_num_games + 1):
                                obs, info = env.reset()
                                if i not in selected_games:
                                    continue

                                num_games_evaluated += 1
                                update_experiment_runtime_state(
                                    db,
                                    experiment,
                                    current_game_number=i,
                                    current_game_label=f"Game #{i}",
                                )
                                agent.set_environment(env, obs, info, i)
                                configure_live_analyst_trace(agent, experiment.id)
                                log_paths = agent.log_paths
                                print(
                                    f"\n[Running Game #{i}] ({num_games_evaluated}/{num_games_to_evaluate})"
                                )
                                try:
                                    game_start_time = time.time()
                                    chat_result, error_message, elapsed_minutes = (
                                        run_game(agent, i)
                                    )
                                    chat_artifacts = persist_chat_artifacts(
                                        agent, chat_result=chat_result
                                    )
                                    chat_text = chat_artifacts["chat_text"]
                                    transitions = chat_artifacts["transitions"]
                                    belief_matches = chat_artifacts["belief_matches"]
                                    chat_round_list.append(
                                        chat_artifacts["chat_rounds"]
                                    )
                                    cumulative_runtime += elapsed_minutes

                                    current_usage_totals = get_agent_usage_totals(agent)
                                    game_usage = get_usage_delta(
                                        current_usage_totals, total_run_usage
                                    )
                                    game_prompt_tokens = int(
                                        game_usage["prompt_tokens"]
                                    )
                                    game_completion_tokens = int(
                                        game_usage["completion_tokens"]
                                    )
                                    game_total_tokens = int(game_usage["total_tokens"])
                                    game_total_cost = float(game_usage["total_cost"])
                                    total_run_usage = {
                                        key: max(
                                            total_run_usage[key],
                                            current_usage_totals[key],
                                        )
                                        for key in total_run_usage
                                    }
                                    if any(value > 0 for value in game_usage.values()):
                                        wandb.log(
                                            {
                                                "game/total_tokens": game_total_tokens,
                                                "game/cost": game_total_cost,
                                            },
                                            step=num_games_evaluated,
                                        )

                                    if error_message:
                                        error_list.append(i)
                                        _append_text_file(
                                            log_paths["error_message_path"],
                                            f"Run Chat: {error_message}\n",
                                        )

                                    agent.prev_episodic_memories.append(
                                        {
                                            "episode_number": num_games_evaluated,
                                            "task_outcome": agent.task_status,
                                            "memory": belief_matches
                                            if belief_matches
                                            else agent.curr_episodic_memory,
                                        }
                                    )

                                    s3_key = upload_chat_history_artifact(
                                        s3,
                                        experiment_id=experiment.id,
                                        game_number=i,
                                        chat_text=chat_text,
                                    )

                                    success = agent.success
                                    if success:
                                        num_successes += 1
                                        success_list.append(i)
                                        cumulative_successful_actions += (
                                            agent.num_actions_taken
                                        )
                                        cumulative_successful_chat_rounds += (
                                            chat_round_list[-1]
                                        )
                                        cumulative_successful_runtime += elapsed_minutes
                                        avg_actions_taken_per_successful_game = (
                                            cumulative_successful_actions
                                            / num_successes
                                        )
                                        avg_chat_rounds_per_successful_game = (
                                            cumulative_successful_chat_rounds
                                            / num_successes
                                        )
                                        avg_runtime_per_successful_game = (
                                            cumulative_successful_runtime
                                            / num_successes
                                        )
                                    else:
                                        num_failures = (
                                            num_games_evaluated - num_successes
                                        )
                                        failure_list.append(i)
                                        cumulative_failing_actions += (
                                            agent.num_actions_taken
                                        )
                                        cumulative_failing_chat_rounds += (
                                            chat_round_list[-1]
                                        )
                                        cumulative_failing_runtime += elapsed_minutes
                                        avg_actions_taken_per_failing_game = (
                                            cumulative_failing_actions / num_failures
                                        )
                                        avg_chat_rounds_per_failing_game = (
                                            cumulative_failing_chat_rounds
                                            / num_failures
                                        )
                                        avg_runtime_per_failing_game = (
                                            cumulative_failing_runtime / num_failures
                                        )

                                    success_rate = num_successes / num_games_evaluated
                                    num_games_no_error = num_games_evaluated - len(
                                        [g for g in error_list if g not in success_list]
                                    )
                                    error_adjusted_success_rate = (
                                        num_successes / num_games_no_error
                                        if num_games_no_error > 0
                                        else 0.0
                                    )
                                    if num_games_no_error == 0:
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
                                            "final/total_tokens": total_run_usage[
                                                "total_tokens"
                                            ],
                                            "final/total_cost": total_run_usage[
                                                "total_cost"
                                            ],
                                            "final/prompt_tokens": total_run_usage[
                                                "prompt_tokens"
                                            ],
                                            "final/completion_tokens": total_run_usage[
                                                "completion_tokens"
                                            ],
                                        },
                                        step=num_games_evaluated,
                                    )

                                    persist_experiment_usage_snapshot(
                                        db, experiment, total_run_usage
                                    )
                                    persist_episode_run(
                                        db,
                                        experiment=experiment,
                                        game_number=i,
                                        agent=agent,
                                        log_paths=log_paths,
                                        success=success,
                                        chat_rounds=chat_round_list[-1],
                                        runtime_minutes=elapsed_minutes,
                                        error_message=str(error_message)
                                        if error_message
                                        else None,
                                        transitions=transitions,
                                        belief_matches=belief_matches,
                                        chat_text=chat_text,
                                        prompt_tokens=game_prompt_tokens,
                                        completion_tokens=game_completion_tokens,
                                        episode_cost=game_total_cost,
                                        success_rate=success_rate,
                                        error_adjusted_success_rate=error_adjusted_success_rate,
                                        chat_history_s3_key=s3_key,
                                        analyst_trace=get_agent_analyst_trace_text(
                                            agent, prefer_existing=True
                                        ),
                                    )
                                    print(f"✅ Saved Game #{i} to PostgreSQL Database!")

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
                                except KeyboardInterrupt as exc:
                                    elapsed_minutes = (
                                        time.time() - game_start_time
                                    ) / 60
                                    cumulative_runtime += elapsed_minutes
                                    if i not in error_list:
                                        error_list.append(i)
                                    success_rate = (
                                        num_successes / num_games_evaluated
                                        if num_games_evaluated
                                        else 0.0
                                    )
                                    num_games_no_error = num_games_evaluated - len(
                                        [g for g in error_list if g not in success_list]
                                    )
                                    error_adjusted_success_rate = (
                                        num_successes / num_games_no_error
                                        if num_games_no_error > 0
                                        else 0.0
                                    )
                                    total_run_usage = persist_interrupted_episode_run(
                                        db,
                                        experiment=experiment,
                                        agent=agent,
                                        game_number=i,
                                        s3=s3,
                                        total_run_usage=total_run_usage,
                                        elapsed_minutes=elapsed_minutes,
                                        success_rate=success_rate,
                                        error_adjusted_success_rate=error_adjusted_success_rate,
                                        error_message=f"Run interrupted: {exc}",
                                    )
                                    print(
                                        f"⚠️ Saved partial interrupted Game #{i} to PostgreSQL Database."
                                    )
                                    raise
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
                        except KeyboardInterrupt:
                            finalize_experiment(
                                db,
                                experiment,
                                cumulative_runtime=cumulative_runtime,
                                success_rate=(
                                    success_rate if num_games_evaluated else 0.0
                                ),
                                error_adjusted_success_rate=error_adjusted_success_rate,
                                error_count=len(error_list),
                                avg_actions_per_successful_game=avg_actions_taken_per_successful_game,
                                avg_chat_rounds_per_successful_game=avg_chat_rounds_per_successful_game,
                                avg_runtime_per_successful_game=avg_runtime_per_successful_game,
                                avg_actions_per_failing_game=avg_actions_taken_per_failing_game,
                                avg_chat_rounds_per_failing_game=avg_chat_rounds_per_failing_game,
                                avg_runtime_per_failing_game=avg_runtime_per_failing_game,
                                status="CANCELLED",
                            )
                            print("⚠️ Experiment cancelled.")
                            raise

                        finalize_experiment(
                            db,
                            experiment,
                            cumulative_runtime=cumulative_runtime,
                            success_rate=success_rate if num_games_evaluated else 0.0,
                            error_adjusted_success_rate=error_adjusted_success_rate,
                            error_count=len(error_list),
                            avg_actions_per_successful_game=avg_actions_taken_per_successful_game,
                            avg_chat_rounds_per_successful_game=avg_chat_rounds_per_successful_game,
                            avg_runtime_per_successful_game=avg_runtime_per_successful_game,
                            avg_actions_per_failing_game=avg_actions_taken_per_failing_game,
                            avg_chat_rounds_per_failing_game=avg_chat_rounds_per_failing_game,
                            avg_runtime_per_failing_game=avg_runtime_per_failing_game,
                            status="CONCLUDED",
                        )
                        print("✅ Experiment securely finalized in the database.")
                    finally:
                        db.close()

                    print(
                        f"Final Success Rate: {num_successes}/{num_games_evaluated} = "
                        f"{100 * (num_successes / num_games_evaluated if num_games_evaluated else 0):.2f}%"
                    )
                    print(
                        f"Final Error-Adjusted Success Rate: {num_successes}/{num_games_no_error} = "
                        f"{100 * error_adjusted_success_rate:.2f}%"
                    )
    except KeyboardInterrupt:
        wandb.finish()
        raise SystemExit(130)

    wandb.finish()


if __name__ == "__main__":
    main()
