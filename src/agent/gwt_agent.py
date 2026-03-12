import hashlib
import json
import os
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import umap
from autogen import ConversableAgent, GroupChat, GroupChatManager, register_function
from autogen.agentchat.contrib.capabilities import transform_messages
from autogen.agentchat.contrib.capabilities.transforms import MessageHistoryLimiter
from sklearn.cluster import KMeans

from src.agent.autogen_agent import AutogenAgent
from src.agent.env_adapter import ALFWorldAdapter, ScienceWorldAdapter
from src.agent.helpers import (
    ConvertOrphanedToolMessages,
    FlattenToolMessages,
    create_echo_agent,
    get_best_candidate,
    is_termination_msg_generic,
    sentence_transformer_model,
)
from src.agent.rag_memory import retrieve_relevant_concepts, retrieve_relevant_episodes


class GWTAutogenAgent(AutogenAgent):
    _AGENT_DIR = Path(__file__).resolve().parent
    _MEMORY_ROOT = _AGENT_DIR / "memory"
    _TASK_STOPWORDS = {
        "a",
        "about",
        "all",
        "an",
        "and",
        "around",
        "as",
        "at",
        "be",
        "box",
        "called",
        "cause",
        "change",
        "current",
        "determine",
        "do",
        "electrically",
        "first",
        "for",
        "from",
        "if",
        "in",
        "into",
        "is",
        "it",
        "its",
        "located",
        "matter",
        "of",
        "on",
        "or",
        "place",
        "red",
        "green",
        "state",
        "substance",
        "take",
        "task",
        "that",
        "the",
        "then",
        "to",
        "using",
        "whether",
        "with",
        "you",
        "your",
    }
    _RUNTIME_TOKEN_STOPWORDS = _TASK_STOPWORDS | {
        "action",
        "actions",
        "admissible",
        "agent",
        "also",
        "around",
        "available",
        "between",
        "called",
        "check",
        "command",
        "commands",
        "containing",
        "current",
        "describe",
        "described",
        "description",
        "door",
        "doors",
        "environment",
        "exact",
        "failed",
        "focus",
        "from",
        "goal",
        "inventory",
        "known",
        "latest",
        "list",
        "lists",
        "look",
        "move",
        "newly",
        "none",
        "observation",
        "observed",
        "previous",
        "resulting",
        "see",
        "sees",
        "room",
        "shortlist",
        "state",
        "status",
        "still",
        "summary",
        "take",
        "there",
        "these",
        "this",
        "through",
        "tool",
        "tooling",
        "timestep",
        "using",
        "valid",
        "visible",
        "with",
    }
    _ACTION_COMMAND_STOPWORDS = {
        "activate",
        "boil",
        "check",
        "close",
        "connect",
        "cook",
        "cool",
        "deactivate",
        "disconnect",
        "dunk",
        "eat",
        "empty",
        "enter",
        "examine",
        "fill",
        "focus",
        "go",
        "grab",
        "heat",
        "inspect",
        "insert",
        "inventory",
        "look",
        "mix",
        "move",
        "open",
        "pick",
        "place",
        "pour",
        "put",
        "read",
        "task",
        "take",
        "turn",
        "use",
        "wait",
    }
    _TASK_FAMILY_HINTS = {
        "focus": ("focus",),
        "inspect": ("inspect", "examine", "look at", "find", "identify", "locate"),
        "relation": ("connect", "link", "disconnect"),
        "relocation": ("go to", "enter", "move to", "bring", "carry", "transport"),
        "transfer_or_transform": (
            "fill",
            "pour",
            "mix",
            "insert",
            "place",
            "put",
            "transfer",
        ),
        "device_control": (
            "open",
            "close",
            "activate",
            "deactivate",
            "turn on",
            "turn off",
        ),
        "tool_application": ("use", "heat", "cool", "boil", "cook", "measure", "test"),
    }
    _TASK_ORDERING_HINTS = {
        "ordered_sequence": (
            "earliest",
            "latest",
            "first",
            "second",
            "third",
            "fourth",
            "before",
            "after",
            "order",
            "ordered",
            "sequence",
            "starting from",
        )
    }
    _TASK_ENTITY_STOPWORDS = _TASK_STOPWORDS | {
        "acceptable",
        "change",
        "compound",
        "compounds",
        "earliest",
        "electrically",
        "identify",
        "latest",
        "life",
        "locate",
        "matter",
        "point",
        "sequence",
        "stage",
        "stages",
        "starting",
        "whether",
    }
    _ACTION_FAMILY_PREFIXES = (
        (
            (
                "look around",
                "look in",
                "look at",
                "look ",
                "inspect ",
                "examine ",
                "check ",
                "read ",
                "inventory",
                "task",
            ),
            "inspect",
        ),
        (("focus on",), "focus"),
        (("connect ", "disconnect "), "relation"),
        (
            (
                "move ",
                "go to ",
                "enter ",
                "pick up ",
                "take ",
                "grab ",
                "put down ",
                "drop ",
            ),
            "relocation",
        ),
        (
            ("pour ", "dunk ", "mix ", "fill ", "empty ", "insert ", "place "),
            "transfer_or_transform",
        ),
        (
            (
                "open ",
                "close ",
                "activate ",
                "deactivate ",
                "turn on ",
                "turn off ",
            ),
            "device_control",
        ),
        (("use ", "heat ", "cool ", "boil ", "cook "), "tool_application"),
        (("eat ",), "consumption"),
        (("wait",), "idle"),
    )
    _DEPRIORITIZE_ELIGIBLE_FAMILIES = {
        "relation",
        "relocation",
        "transfer_or_transform",
        "device_control",
        "tool_application",
    }
    _UNCERTAINTY_RE = re.compile(
        r"\b(uncertain|unclear|unsure|unknown|conflicting|"
        r"contradictory|ambiguous|stalled|not\s+(?:sure|certain|confirmed|clear|visible|found)"
        r"|do\s+not\s+know|don't\s+know|need\s+to\s+(?:verify|check|confirm)"
        r"|may\s+(?:be|need)|might\s+(?:be|need))\b",
        re.IGNORECASE,
    )
    _HARD_FAILURE_RE = re.compile(
        r"(not in the list of admissible actions|not admissible|not recognized|"
        r"can't|cannot|failed|nothing happens|nothing is burning|not movable|"
        r"do not see|don't see|no such)",
        re.IGNORECASE,
    )
    _DIRECT_EFFECT_RE = re.compile(
        r"(heats up|cools down|produces|transforms?|changes? state|"
        r"temperature (?:of|reads|measures)|opens?|closes?|activates?|deactivates?|"
        r"moves?|picked up|put down|mixed?|created|appears?|disappears?|"
        r"filled?|emptied?|turned (?:on|off)|boils?|freezes?|burns?)",
        re.IGNORECASE,
    )
    _EFFECTLESS_RE = re.compile(
        r"(no effect|unchanged|still at|remains? |still |did not|does not|"
        r"no longer changes)",
        re.IGNORECASE,
    )

    def __init__(
        self,
        llm_profile,
        log_path,
        game_no=1,
        max_chat_round=150,
        max_actions=35,
        rounds_per_game=1,
        rag_episode_k=4,
        rag_concept_k=5,
        rag_episode_k_initial=4,
        rag_concept_k_initial=5,
        args=None,
        env=None,
        obs="",
        info=None,
    ):
        super().__init__(
            llm_profile,
            log_path,
            game_no,
            max_chat_round,
            max_actions,
            args,
            env,
            obs,
            info,
        )

        self.rag_episode_k = rag_episode_k
        self.rag_concept_k = rag_concept_k
        self.rag_episode_k_initial = rag_episode_k_initial
        self.rag_concept_k_initial = rag_concept_k_initial

        self.read_only_memory = False

        self.echo_agent = None
        self.action_agent = None
        self.thinking_agent = None
        self.external_perception_agent = None
        self.internal_perception_agent_1 = None
        self.internal_perception_agent_2 = None
        self.internal_perception_agent_3 = None
        self.belief_state_agent = None
        self.retrieve_memory_agent = None
        self.learning_agent = None
        self.record_long_term_memory_agent = None
        self.focus_agent = None
        self.agents_info = {}

        self._initial_max_actions = self.max_actions
        self.rounds = rounds_per_game
        self.max_round_actions = self.max_actions // self.rounds
        self.max_actions = self.max_actions - self.max_round_actions * (self.rounds - 1)

        self.allowed_transitions = None

        self.rounds_left = self.rounds
        self.task_failed = False
        self.task_success = False

        self.initialize_autogen()

        self.task = ""
        self.admissible_actions = []
        self.percept = {}
        self.curr_episodic_memory = []
        self.prev_episodic_memories = []
        self.knowledge = []
        self.task_status = "INCOMPLETE"
        self.initial_message = ""
        self.memory = ""
        self._episodic_rag_cache: dict = {}
        self._concept_rag_cache: dict = {}
        self._task_contract: dict = {}
        self._task_contract_source = ""
        self._reset_episode_reasoning_state()

    def _get_memory_environment_name(self) -> str:
        if isinstance(getattr(self, "adapter", None), ScienceWorldAdapter):
            return "scienceworld"
        if isinstance(getattr(self, "adapter", None), ALFWorldAdapter):
            return "alfworld"

        env_type = getattr(getattr(self, "args", None), "env_type", None)
        if env_type:
            return re.sub(r"[^a-z0-9_-]+", "_", env_type.lower())

        return "alfworld"

    def _get_memory_dir(self) -> Path:
        return self._MEMORY_ROOT / self._get_memory_environment_name()

    def _reset_episode_reasoning_state(self) -> None:
        self.episode_hypothesis_ledger: dict[str, dict] = {}
        self.recent_hypothesis_tests: list[dict] = []
        self._provider_fallbacks_applied: set[str] = set()
        self._last_action_shortlist: list[str] = []

    def _build_task_contract(self, task: str) -> dict:
        task_lower = (task or "").lower()
        required_families: list[str] = []
        family_hint_tokens: set[str] = set()
        for family, hints in self._TASK_FAMILY_HINTS.items():
            if any(self._task_contains_hint(task_lower, hint) for hint in hints):
                required_families.append(family)
                for hint in hints:
                    family_hint_tokens.update(
                        self._extract_runtime_tokens(
                            hint, stopwords=self._TASK_STOPWORDS | self._ACTION_COMMAND_STOPWORDS
                        )
                    )

        ordering_cues: list[str] = []
        ordering_tokens: set[str] = set()
        for cue, hints in self._TASK_ORDERING_HINTS.items():
            if any(self._task_contains_hint(task_lower, hint) for hint in hints):
                ordering_cues.append(cue)
                for hint in hints:
                    ordering_tokens.update(
                        self._extract_runtime_tokens(
                            hint,
                            stopwords=self._TASK_STOPWORDS | self._ACTION_COMMAND_STOPWORDS,
                        )
                    )

        target_entities = self._extract_runtime_tokens(
            task,
            stopwords=self._TASK_ENTITY_STOPWORDS
            | self._ACTION_COMMAND_STOPWORDS
            | family_hint_tokens
            | ordering_tokens,
            limit=8,
        )

        return {
            "required_families": required_families,
            "target_entities": target_entities,
            "ordering_cues": ordering_cues,
        }

    def _get_task_contract(self) -> dict:
        task = getattr(self, "task", "") or ""
        if self._task_contract_source != task:
            self._task_contract = self._build_task_contract(task)
            self._task_contract_source = task
        return self._task_contract

    @staticmethod
    def _task_contains_hint(task_text: str, hint: str) -> bool:
        normalized_task = re.sub(r"\s+", " ", task_text.strip().lower())
        normalized_hint = re.sub(r"\s+", " ", hint.strip().lower())
        if not normalized_hint:
            return False
        return bool(
            re.search(rf"(?<![a-z0-9]){re.escape(normalized_hint)}(?![a-z0-9])", normalized_task)
        )

    @staticmethod
    def _normalize_runtime_text(text: str) -> str:
        return re.sub(r"\s+", " ", (text or "").strip().lower())

    def _extract_task_keywords(self) -> list[str]:
        return self._extract_runtime_tokens(self.task, stopwords=self._TASK_STOPWORDS)

    def _extract_runtime_tokens(
        self,
        text: str,
        *,
        stopwords: set[str] | None = None,
        limit: int | None = None,
    ) -> list[str]:
        active_stopwords = stopwords or self._RUNTIME_TOKEN_STOPWORDS
        tokens = []
        for token in re.findall(r"[a-z0-9]+", (text or "").lower()):
            if len(token) < 3 or token in active_stopwords:
                continue
            if token not in tokens:
                tokens.append(token)
            if limit is not None and len(tokens) >= limit:
                break
        return tokens

    def _get_observation_grounded_tokens(self) -> list[str]:
        tokens: list[str] = []
        percept = getattr(self, "percept", {}) or {}

        for token in self._extract_runtime_tokens(
            percept.get("resulting_observation", ""), limit=10
        ):
            if token not in tokens:
                tokens.append(token)

        for test in reversed(self.recent_hypothesis_tests):
            if test["outcome"] not in {"observable_change", "evidence"}:
                continue
            for token in self._extract_runtime_tokens(test["action"], limit=6):
                if token not in tokens:
                    tokens.append(token)
            if len(tokens) >= 12:
                break

        return tokens[:12]

    def _extract_action_content_tokens(self, action: str, family: str | None = None) -> list[str]:
        stopwords = self._RUNTIME_TOKEN_STOPWORDS | self._ACTION_COMMAND_STOPWORDS
        content_tokens = self._extract_runtime_tokens(action, stopwords=stopwords)
        if family == "device_control":
            # Keeping "door" helps navigation/opening tasks stay grounded.
            return content_tokens
        return content_tokens

    def _get_recent_invalid_actions(self, *, limit: int = 4) -> list[str]:
        invalid_actions: list[str] = []
        for test in reversed(self.recent_hypothesis_tests):
            if test["outcome"] != "invalid":
                continue
            action = test["action"]
            if action and action not in invalid_actions:
                invalid_actions.append(action)
            if len(invalid_actions) >= limit:
                break
        return list(reversed(invalid_actions))

    @staticmethod
    def _get_shortlist_family_quotas(current_phase: str) -> dict[str, int]:
        quotas_by_phase = {
            "gather_evidence": {
                "inspect": 3,
                "device_control": 2,
                "relocation": 2,
                "transfer_or_transform": 2,
                "tool_application": 1,
                "focus": 1,
            },
            "test_mechanism": {
                "inspect": 2,
                "device_control": 1,
                "relocation": 1,
                "transfer_or_transform": 3,
                "tool_application": 3,
                "relation": 1,
                "focus": 1,
            },
            "commit_to_goal": {
                "inspect": 1,
                "device_control": 2,
                "relocation": 3,
                "transfer_or_transform": 3,
                "tool_application": 2,
                "focus": 1,
            },
            "act": {
                "inspect": 2,
                "device_control": 2,
                "relocation": 2,
                "transfer_or_transform": 2,
                "tool_application": 2,
                "focus": 1,
            },
        }
        return quotas_by_phase.get(current_phase, quotas_by_phase["act"])

    @staticmethod
    def _get_family_priority(current_phase: str, family: str) -> int:
        priorities_by_phase = {
            "gather_evidence": {
                "inspect": 6,
                "device_control": 5,
                "relocation": 5,
                "transfer_or_transform": 4,
                "tool_application": 3,
                "focus": 2,
                "relation": 1,
                "other": 0,
                "idle": -5,
            },
            "test_mechanism": {
                "tool_application": 6,
                "transfer_or_transform": 6,
                "inspect": 5,
                "device_control": 4,
                "relocation": 4,
                "relation": 3,
                "focus": 1,
                "other": 0,
                "idle": -5,
            },
            "commit_to_goal": {
                "relocation": 6,
                "transfer_or_transform": 5,
                "tool_application": 5,
                "device_control": 4,
                "inspect": 3,
                "focus": 1,
                "relation": 0,
                "other": 0,
                "idle": -5,
            },
            "act": {
                "inspect": 5,
                "device_control": 5,
                "relocation": 5,
                "transfer_or_transform": 5,
                "tool_application": 5,
                "focus": 1,
                "relation": 1,
                "other": 0,
                "idle": -5,
            },
        }
        return priorities_by_phase.get(current_phase, priorities_by_phase["act"]).get(
            family, 0
        )

    def _classify_action_family(self, action: str) -> str:
        normalized = self._normalize_runtime_text(action)
        if not normalized:
            return "unknown"

        for prefixes, family in self._ACTION_FAMILY_PREFIXES:
            if normalized.startswith(prefixes):
                return family
        return "other"

    def _canonicalize_suggested_action(
        self, suggested_action: str, admissible_commands: list[str]
    ) -> str:
        normalized = self._normalize_runtime_text(suggested_action)
        exact_matches = {
            self._normalize_runtime_text(command): command
            for command in admissible_commands
        }
        if normalized in exact_matches:
            return exact_matches[normalized]

        family = self._classify_action_family(suggested_action)
        suggested_tokens = set(
            self._extract_action_content_tokens(suggested_action, family=family)
        )
        if not suggested_tokens:
            return suggested_action

        best_command = suggested_action
        best_score = float("-inf")
        for command in admissible_commands:
            if self._classify_action_family(command) != family:
                continue
            command_tokens = set(
                self._extract_action_content_tokens(command, family=family)
            )
            overlap = len(suggested_tokens & command_tokens)
            if overlap == 0:
                continue

            missing = len(suggested_tokens - command_tokens)
            extra = len(command_tokens - suggested_tokens)
            score = overlap * 10 - missing * 5 - extra * 2
            if suggested_tokens == command_tokens:
                score += 10
            elif suggested_tokens.issubset(command_tokens):
                score += 6

            if score > best_score:
                best_score = score
                best_command = command

        return best_command if best_score >= 15 else suggested_action

    def _get_hypothesis_entry(self, family: str) -> dict:
        entry = self.episode_hypothesis_ledger.get(family)
        if entry is not None:
            return entry

        entry = {
            "family": family,
            "tests": 0,
            "invalid_attempts": 0,
            "stalled_attempts": 0,
            "observable_change_attempts": 0,
            "evidence_attempts": 0,
            "confidence": 0.5,
            "status": "unseen",
            "retired": False,
            "last_action": "",
            "last_evidence": "",
        }
        self.episode_hypothesis_ledger[family] = entry
        return entry

    def _classify_hypothesis_outcome(
        self,
        *,
        family: str,
        executed_action: str | None,
        observation: str,
        previous_observation: str,
    ) -> tuple[str, str]:
        normalized_observation = self._normalize_runtime_text(observation)
        observation_changed = (
            normalized_observation != self._normalize_runtime_text(previous_observation)
        )
        has_admissible_delta = bool(
            self.percept.get("newly_admissible_actions")
            or self.percept.get("no_longer_admissible_actions")
        )

        if executed_action is None or self._HARD_FAILURE_RE.search(observation):
            return "invalid", "The attempted action failed or was inadmissible."
        if family == "inspect" and observation_changed:
            return "evidence", "Inspection produced new evidence."
        if self.task_status == "COMPLETED" or self._DIRECT_EFFECT_RE.search(observation):
            return "observable_change", "The observation reported a state change."
        if has_admissible_delta:
            return "observable_change", "The action changed the available affordances."
        if self._EFFECTLESS_RE.search(observation) or not observation_changed:
            return "stalled", "No task-relevant change was observed."
        return (
            "uncertain",
            "The action changed context, but its causal value is still unclear.",
        )

    def _update_episode_hypothesis_ledger(
        self,
        *,
        suggested_action: str,
        executed_action: str | None,
        previous_observation: str,
    ) -> None:
        family = self._classify_action_family(executed_action or suggested_action)
        entry = self._get_hypothesis_entry(family)
        observation = self.percept.get("resulting_observation", "")
        outcome, evidence = self._classify_hypothesis_outcome(
            family=family,
            executed_action=executed_action,
            observation=observation,
            previous_observation=previous_observation,
        )

        entry["tests"] += 1
        entry["last_action"] = executed_action or suggested_action
        entry["last_evidence"] = evidence

        if outcome == "invalid":
            entry["invalid_attempts"] += 1
            entry["confidence"] = max(0.0, entry["confidence"] - 0.2)
        elif outcome == "observable_change":
            entry["observable_change_attempts"] += 1
            entry["confidence"] = min(1.0, entry["confidence"] + 0.1)
        elif outcome == "evidence":
            entry["evidence_attempts"] += 1
        elif outcome == "stalled":
            entry["stalled_attempts"] += 1
            entry["confidence"] = max(0.0, entry["confidence"] - 0.15)

        required_families = set(self._get_task_contract().get("required_families", []))
        if (
            family in self._DEPRIORITIZE_ELIGIBLE_FAMILIES
            and family not in required_families
            and entry["observable_change_attempts"] == 0
            and (entry["invalid_attempts"] >= 2 or entry["stalled_attempts"] >= 2)
        ):
            entry["status"] = "deprioritized"
            entry["retired"] = True
        elif entry["observable_change_attempts"] > 0:
            entry["status"] = "promising"
            entry["retired"] = False
        elif entry["tests"] > 0:
            entry["status"] = "uncertain"
            entry["retired"] = False

        self.recent_hypothesis_tests.append(
            {
                "family": family,
                "action": executed_action or suggested_action,
                "outcome": outcome,
                "evidence": evidence,
            }
        )
        self.recent_hypothesis_tests = self.recent_hypothesis_tests[-6:]

    def _get_episode_hypothesis_snapshot(
        self,
        *,
        max_families: int = 4,
        max_recent_tests: int = 4,
    ) -> dict:
        families = sorted(
            self.episode_hypothesis_ledger.values(),
            key=lambda entry: (
                entry["status"] != "deprioritized",
                -(entry["tests"]),
                entry["family"],
            ),
        )
        serialized_families = [
            {
                "family": entry["family"],
                "status": entry["status"],
                "confidence": round(entry["confidence"], 2),
                "tests": entry["tests"],
                "invalid_attempts": entry["invalid_attempts"],
                "stalled_attempts": entry["stalled_attempts"],
                "observable_change_attempts": entry["observable_change_attempts"],
                "last_evidence": entry["last_evidence"],
            }
            for entry in families[:max_families]
            if entry["tests"] > 0
        ]

        return {
            "mechanisms": serialized_families,
            "recent_tests": self.recent_hypothesis_tests[-max_recent_tests:],
        }

    def _get_current_phase(self) -> str:
        inspect_evidence = any(
            entry["evidence_attempts"] > 0
            for entry in self.episode_hypothesis_ledger.values()
            if entry["family"] == "inspect"
        )
        mechanism_progress = any(
            entry["observable_change_attempts"] > 0
            for entry in self.episode_hypothesis_ledger.values()
            if entry["family"] in {"relation", "tool_application", "transfer_or_transform"}
        )
        task_lower = self.task.lower()
        placement_task = any(
            token in task_lower for token in ("place", "put", "move", "deliver")
        )
        if not inspect_evidence:
            return "gather_evidence"
        if any(token in task_lower for token in ("determine", "test", "whether", "identify")) and not mechanism_progress:
            return "test_mechanism"
        if placement_task:
            return "commit_to_goal"
        return "act"

    def _score_action_for_shortlist(
        self,
        action: str,
        *,
        current_phase: str,
        task_keywords: list[str],
        grounded_tokens: list[str],
    ) -> tuple[int, int, int]:
        normalized = self._normalize_runtime_text(action)
        family = self._classify_action_family(action)
        action_tokens = self._extract_runtime_tokens(action)
        content_tokens = self._extract_action_content_tokens(action, family=family)
        action_token_set = set(action_tokens)
        content_token_set = set(content_tokens)
        grounded_token_set = set(grounded_tokens)
        task_keyword_set = set(task_keywords)
        task_contract = self._get_task_contract()
        required_families = set(task_contract.get("required_families", []))
        target_entity_set = set(task_contract.get("target_entities", []))
        score = 0

        keyword_hits = len(action_token_set & task_keyword_set)
        grounded_hits = len(content_token_set & grounded_token_set)
        target_hits = len(content_token_set & target_entity_set)
        score += keyword_hits * 5
        score += grounded_hits * 6
        score += target_hits * 5

        family_priority = self._get_family_priority(current_phase, family)
        score += family_priority

        if family in required_families:
            score += 7
            if grounded_hits:
                score += 4
            if target_hits:
                score += 6

        if family == "inspect" and (keyword_hits or grounded_hits):
            score += 5
        if family == "device_control" and grounded_hits:
            score += 4
        if family == "relocation" and grounded_hits:
            score += 4
        if family == "transfer_or_transform" and grounded_hits:
            score += 4
        if family == "tool_application" and (keyword_hits or grounded_hits):
            score += 4
        if family == "focus" and grounded_hits:
            score += 3
        elif family == "focus":
            score -= 3

        if current_phase == "gather_evidence":
            if family == "relation":
                score -= 2
        elif current_phase == "test_mechanism":
            if family == "focus":
                score -= 1
        elif current_phase == "commit_to_goal":
            if family == "relation":
                score -= 1

        entry = self.episode_hypothesis_ledger.get(family)
        if entry is not None:
            score += round((entry["confidence"] - 0.5) * 8)
            if entry["status"] == "promising":
                score += 2
            if entry["status"] == "deprioritized":
                score -= 8
            if entry["invalid_attempts"] >= 2:
                score -= 2
            if entry["stalled_attempts"] >= 2:
                score -= 2
        if family == "other":
            score -= 2

        if content_token_set and grounded_hits == 0:
            if target_hits:
                score -= 6
            else:
                score -= 10
        elif (
            target_entity_set
            and content_token_set
            and grounded_hits > 0
            and target_hits == 0
            and family not in {"inspect", "device_control", "relocation"}
        ):
            score -= 3

        if normalized.startswith("wait"):
            score -= 10

        return score, grounded_hits, family_priority

    def _summarize_admissible_actions(
        self,
        actions: list[str],
        *,
        shortlist_limit: int = 12,
    ) -> dict:
        family_counts: dict[str, int] = {}
        current_phase = self._get_current_phase()
        task_keywords = self._extract_task_keywords()
        grounded_tokens = self._get_observation_grounded_tokens()
        scored_actions = []

        for action in actions:
            family = self._classify_action_family(action)
            family_counts[family] = family_counts.get(family, 0) + 1
            score, grounded_hits, family_priority = self._score_action_for_shortlist(
                action,
                current_phase=current_phase,
                task_keywords=task_keywords,
                grounded_tokens=grounded_tokens,
            )
            scored_actions.append(
                {
                    "action": action,
                    "family": family,
                    "score": score,
                    "grounded_hits": grounded_hits,
                    "family_priority": family_priority,
                }
            )

        scored_actions.sort(
            key=lambda item: (
                -item["score"],
                -item["grounded_hits"],
                -item["family_priority"],
                len(item["action"]),
                item["action"],
            )
        )

        quotas = self._get_shortlist_family_quotas(current_phase)
        shortlist: list[str] = []
        selected_actions: set[str] = set()
        selected_by_family: dict[str, int] = {}

        for item in scored_actions:
            family = item["family"]
            quota = quotas.get(family, 0)
            if quota <= 0 or selected_by_family.get(family, 0) >= quota:
                continue
            shortlist.append(item["action"])
            selected_actions.add(item["action"])
            selected_by_family[family] = selected_by_family.get(family, 0) + 1
            if len(shortlist) >= shortlist_limit:
                break

        for item in scored_actions:
            if len(shortlist) >= shortlist_limit:
                break
            if item["action"] in selected_actions:
                continue
            shortlist.append(item["action"])
            selected_actions.add(item["action"])

        deprioritized_families = sorted(
            entry["family"]
            for entry in self.episode_hypothesis_ledger.values()
            if entry["status"] == "deprioritized"
        )
        task_contract = self._get_task_contract()

        return {
            "total_actions": len(actions),
            "current_phase": current_phase,
            "family_counts": family_counts,
            "salient_entities": grounded_tokens[:8],
            "task_relevant_action_shortlist": shortlist,
            "deprioritized_families": deprioritized_families,
            "task_contract": task_contract,
        }

    def _build_shared_action_context(self) -> dict:
        summary = self._summarize_admissible_actions(self.admissible_actions)
        previous_shortlist = self._last_action_shortlist
        current_shortlist = summary["task_relevant_action_shortlist"]
        self._last_action_shortlist = current_shortlist
        summary["newly_relevant_actions"] = [
            action for action in current_shortlist if action not in previous_shortlist
        ]
        summary["no_longer_relevant_actions"] = [
            action for action in previous_shortlist if action not in current_shortlist
        ]
        return summary

    def _refresh_action_agent_runtime_context(self) -> None:
        if self.action_agent is None:
            return

        summary = self._summarize_admissible_actions(self.admissible_actions, shortlist_limit=20)
        recent_invalid_actions = self._get_recent_invalid_actions()
        self._set_agent_system_message(
            self.action_agent,
            self._action_agent_base_prompt
            + "\n\n--- PRIVATE RUNTIME CONTEXT ---\n"
            + f"Current phase: {summary['current_phase']}\n"
            + f"Current exact task-relevant admissible shortlist: {json.dumps(summary['task_relevant_action_shortlist'])}\n"
            + f"Action family counts: {json.dumps(summary['family_counts'])}\n"
            + f"Salient grounded entities from the latest percept: {json.dumps(summary['salient_entities'])}\n"
            + f"Task contract: {json.dumps(summary['task_contract'])}\n"
            + f"Deprioritized mechanism families this episode: {json.dumps(summary['deprioritized_families'])}\n"
            + f"Recent invalid exact commands to avoid repeating: {json.dumps(recent_invalid_actions)}\n"
            + "Choose an exact shortlist string whenever possible. "
            + "If you go off-shortlist, stay lexically close to grounded entities and known admissible verb families instead of inventing a fresh command template. "
            + "The executor will only execute the action if it actually matches the environment's admissible commands.\n"
        )

    @staticmethod
    def _set_agent_system_message(agent, content: str) -> None:
        if hasattr(agent, "_oai_system_message") and getattr(agent, "_oai_system_message"):
            agent._oai_system_message[0]["content"] = content
            return
        setattr(agent, "system_message", content)

    def _synthesize_belief_state_fallback(self, malformed_content: str) -> str:
        percept = self.percept or {}
        timestep = percept.get("timestep", self.num_actions_taken)
        attempted_action = percept.get("attempted_action", "None")
        observation = percept.get("resulting_observation", "No observation available.")
        attempts_left = percept.get("action_attempts_left", self.max_actions - self.num_actions_taken)
        ledger = self._get_episode_hypothesis_snapshot(max_families=3, max_recent_tests=2)

        parts = [
            f"Timestep {timestep}: My previous belief-state output drifted out of format, so I discard any embedded action suggestion and re-anchor on the latest confirmed percept.",
            f"The last attempted action was {attempted_action!r}.",
            f"The latest confirmed observation is: {observation}",
            f"The task is still {percept.get('task_status', self.task_status)} with {attempts_left} action attempts left.",
        ]

        if ledger["mechanisms"]:
            mechanism_notes = ", ".join(
                f"{entry['family']}={entry['status']}({entry['confidence']})"
                for entry in ledger["mechanisms"]
            )
            parts.append(
                "My current episode-local mechanism confidence is: "
                + mechanism_notes
                + ". These are provisional beliefs about this episode only."
            )

        if malformed_content.strip():
            parts.append(
                "The malformed output should not be treated as world knowledge or a committed plan."
            )

        return "BELIEF STATE: [" + " ".join(parts) + "]"

    @staticmethod
    def _is_provider_quota_error(error: Exception) -> bool:
        message = str(error).upper()
        return "429" in message or "RESOURCE_EXHAUSTED" in message

    @staticmethod
    def _provider_from_error(error: Exception) -> str | None:
        message = str(error).lower()
        if "generativelanguage" in message or "gemini" in message:
            return "google"
        if "deepseek" in message:
            return "openai"
        return None

    @staticmethod
    def _deprioritize_provider(config: dict | None, provider: str) -> bool:
        if not config or "config_list" not in config:
            return False

        current = list(config["config_list"])
        if len(current) <= 1:
            return False

        preferred = [item for item in current if item.get("api_type") != provider]
        deprioritized = [item for item in current if item.get("api_type") == provider]
        reordered = preferred + deprioritized
        if reordered == current:
            return False
        config["config_list"] = reordered
        return True

    def _apply_provider_fallback(self, provider: str) -> bool:
        if provider in self._provider_fallbacks_applied:
            return False

        changed = False
        for config in (
            getattr(self, "support_config", None),
            getattr(self, "reasoner_config", None),
        ):
            changed = self._deprioritize_provider(config, provider) or changed

        if changed and self.group_chat_manager is not None:
            if getattr(self, "reasoner_config", None):
                self.group_chat_manager.llm_config = self.reasoner_config["config_list"][0]

        if changed:
            self._provider_fallbacks_applied.add(provider)
        return changed

    def recover_from_chat_error(self, *, error, initial_message_content, stage):
        if not self._is_provider_quota_error(error):
            return None

        provider = self._provider_from_error(error)
        if provider is None or not self._apply_provider_fallback(provider):
            return None

        print(
            f"⚠️ Quota error from provider '{provider}'. Reordering runtime configs and retrying the in-flight chat."
        )

        self._refresh_action_agent_runtime_context()
        try:
            if self.group_chat is not None and self.group_chat.messages:
                return self.resume_chat(self.group_chat.messages)
            assert self.start_agent is not None
            assert self.group_chat_manager is not None
            chat_result = self.start_agent.initiate_chat(
                self.group_chat_manager,
                message={"role": "system", "content": initial_message_content},
                summary_method="reflection_with_llm",
            )
            return chat_result, None
        except Exception as retry_error:
            print(f"⚠️ Provider fallback retry failed: {retry_error}")
            return None

    def _make_belief_state_termination_fn(self):
        """Returns a termination predicate for Belief_State_Agent.
        Terminates on STRAWBERRY/FLEECE OR after task_success is True
        and Belief_State_Agent has been visited twice (grace period for
        Learning_Agent to run one cycle).
        """
        self._belief_state_post_success_visits = 0

        def _check(msg):
            if is_termination_msg_generic(msg):
                return True
            if self.task_success or (self.task_failed and self.rounds_left == 0):
                self._belief_state_post_success_visits += 1
                return self._belief_state_post_success_visits >= 2
            return False

        return _check

    def set_environment(self, env, obs, info, game_no, adapter=None):
        self.adapter = (
            adapter if adapter is not None else ALFWorldAdapter(env, obs, info)
        )
        self.env = env
        self.game_no = game_no

        self.register_game_log_paths()
        self.cluster_knowledge()
        self._reset_episode_reasoning_state()

        self.num_actions_taken = 0
        self.max_actions = self._initial_max_actions - self.max_round_actions * (
            self.rounds - 1
        )
        self.rounds_left = self.rounds
        self.task_failed = False
        self.task_success = False
        self.success = False
        self._belief_state_post_success_visits = 0
        self._stale_action_count = 0
        self._last_seen_actions_taken = -1
        self._last_belief_content = ""
        self._consecutive_thinking_count = 0
        if hasattr(self, "_task_done_msg_count"):
            del self._task_done_msg_count
        # Reset echo agent relay state so stale_count from the previous game
        # doesn't fire false system errors at the start of a new game.
        if self.echo_agent is not None and hasattr(self.echo_agent, "_relay_state"):
            self.echo_agent._relay_state["stale_count"] = 0
            self.echo_agent._relay_state["last_obs"] = None
        # Clear all agent _oai_messages so game 1 context doesn't bleed into game 2.
        if self.group_chat is not None:
            for agent in self.group_chat.agents:
                agent.clear_history()
        # Recreate GroupChat and GroupChatManager so game 2 starts with a completely
        # clean slate.  GroupChatManager accumulates _oai_messages keyed by every
        # agent it spoke with; without reinit those 100+ rounds of stale history
        # bleed into game 2 and can cause the first LLM call to hang.  max_round is
        # set to self.max_chat_round inside _reinit_groupchat().
        self._reinit_groupchat()
        # Clear per-game RAG caches so stale embeddings don't pollute retrieval.
        # _cluster_cache is intentionally NOT cleared: it is keyed by the content
        # hash of memory1.txt and auto-invalidates whenever concepts change, so it
        # stays valid across games as long as memory is unchanged.
        self._episodic_rag_cache.clear()
        self._concept_rag_cache.clear()
        self.task = self.adapter.task
        self._task_contract = self._build_task_contract(self.task)
        self._task_contract_source = self.task
        self.admissible_actions = self.adapter.admissible_actions
        self.task_status = "INCOMPLETE"
        self.curr_episodic_memory = []
        self.retrieve_memory()

        self.update_percept(action="None")
        self.initial_message = self.generate_initial_message()

        with open(self.log_paths["task_path"], "w") as f:
            f.write(f"Task: {self.task}\n")

        initial_observation = self.adapter.initial_observation
        with open(self.log_paths["history_path"], "w") as f:
            f.write(f"action: 'None'. observation: '{initial_observation}'\n")

        with open(self.log_paths["admissible_commands_path"], "w") as f:
            f.write(f"{self.admissible_actions}\n")

    def update_percept(self, action):

        curr_admissible = self.adapter.admissible_actions
        no_longer = sorted(set(self.admissible_actions) - set(curr_admissible))
        newly_added = sorted(set(curr_admissible) - set(self.admissible_actions))
        self.admissible_actions = curr_admissible

        self.percept = {
            "timestep": self.num_actions_taken,
            "attempted_action": action,
            "resulting_observation": self.adapter.observation,
            "task_status": self.task_status,
            "action_attempts_left": self.max_actions - self.num_actions_taken,
        }
        task_contract = self._get_task_contract()
        if any(task_contract.values()):
            self.percept["task_contract"] = task_contract
        shared_action_context = self._build_shared_action_context()
        self.percept["admissible_action_summary"] = {
            "total_actions": shared_action_context["total_actions"],
            "current_phase": shared_action_context["current_phase"],
            "family_counts": shared_action_context["family_counts"],
            "salient_entities": shared_action_context["salient_entities"],
            "deprioritized_families": shared_action_context["deprioritized_families"],
            "required_families": shared_action_context["task_contract"]["required_families"],
        }
        self.percept["task_relevant_action_shortlist"] = shared_action_context[
            "task_relevant_action_shortlist"
        ]
        if self.num_actions_taken > 0:
            if shared_action_context["newly_relevant_actions"]:
                self.percept["newly_relevant_actions"] = shared_action_context[
                    "newly_relevant_actions"
                ]
            if shared_action_context["no_longer_relevant_actions"]:
                self.percept["no_longer_relevant_actions"] = shared_action_context[
                    "no_longer_relevant_actions"
                ]
            if not newly_added and not no_longer:
                self.percept["admissible_actions_unchanged"] = True
        self._refresh_action_agent_runtime_context()

        keys_to_extract = ["timestep", "attempted_action", "resulting_observation"]
        summary_json = json.dumps(
            {k: self.percept[k] for k in keys_to_extract if k in self.percept}
        )
        self.curr_episodic_memory.append(summary_json)

    def get_curr_episodic_memory_str(self):
        return json.dumps(self.curr_episodic_memory, indent=2)

    def initialize_agents(self):
        from pathlib import Path

        import yaml

        _prompts_path = Path(__file__).parent / "configs" / "prompts.yaml"
        with _prompts_path.open() as _f:
            _PROMPTS = yaml.safe_load(_f)

        self._action_agent_base_prompt = _PROMPTS["action_agent"]
        self.llm_config_list = self.llm_profile.get("config_list", [])

        # Standard priority: Gemini -> Chat -> Reasoner
        self.standard_config = {
            "config_list": list(self.llm_config_list),
            "temperature": 0.0,
            "max_tokens": 200,
        }
        self.support_config = {
            "config_list": [
                *[
                    cfg
                    for cfg in self.llm_config_list
                    if cfg.get("api_type") != "google"
                ],
                *[
                    cfg
                    for cfg in self.llm_config_list
                    if cfg.get("api_type") == "google"
                ],
            ],
            "temperature": 0.0,
            "max_tokens": 200,
        }

        # Reasoner priority: Reasoner -> Chat -> Gemini (reversed)
        self.reasoner_config = {
            "config_list": list(reversed(self.llm_config_list)),
            "temperature": 1.0,  # Reasoners need higher temp for R1/o1
        }

        self.echo_agent = create_echo_agent()
        self.agents_info[self.echo_agent.name] = {
            "Prompt": self.echo_agent.system_message,
            "Description": self.echo_agent.description,
        }
        # 2. Initialize Infrastructure Agents
        self.focus_agent = ConversableAgent(
            name="Focus_Agent",
            system_message=_PROMPTS["focus_agent"],
            description="Focus_Agent calls the 'focus' function whenever Belief_State_Agent fails to state a BELIEF STATE until Belief_State_Agent outputs a BELIEF STATE.",
            llm_config=self.support_config,
            is_termination_msg=lambda msg: False,
            human_input_mode="NEVER",
        )
        self.agents_info[self.focus_agent.name] = {
            "Prompt": self.focus_agent.system_message,
            "Description": self.focus_agent.description,
        }

        self.retrieve_memory_agent = ConversableAgent(
            name="Retrieve_Memory_Agent",
            system_message=_PROMPTS["retrieve_memory_agent"],
            description="Retrieve_Memory_Agent calls the 'retrieve_memory' function to help recall and process useful knowledge and information to solve the task.",
            llm_config=self.support_config,
            human_input_mode="NEVER",
            is_termination_msg=lambda msg: False,
        )
        self.agents_info[self.retrieve_memory_agent.name] = {
            "Prompt": self.retrieve_memory_agent.system_message,
            "Description": self.retrieve_memory_agent.description,
        }

        self.action_agent = ConversableAgent(
            name="Action_Agent",
            system_message=self._action_agent_base_prompt,
            description="Action_Agent calls the 'execute_action' function with the best admissible action as the argument.",
            llm_config=self.reasoner_config,
            human_input_mode="NEVER",
            is_termination_msg=lambda msg: False,
        )
        self.agents_info[self.action_agent.name] = {
            "Prompt": self.action_agent.system_message,
            "Description": self.action_agent.description,
        }

        self.thinking_agent = ConversableAgent(
            name="Thinking_Agent",
            system_message=_PROMPTS["thinking_agent"],
            description="Thinking_Agent integrates all available information from the ongoing conversation in order to construct new ideas.",
            llm_config=self.reasoner_config,
            human_input_mode="NEVER",
            is_termination_msg=lambda msg: False,
        )
        self.agents_info[self.thinking_agent.name] = {
            "Prompt": self.thinking_agent.system_message,
            "Description": self.thinking_agent.description,
        }

        self.belief_state_agent = ConversableAgent(
            name="Belief_State_Agent",
            system_message=_PROMPTS["belief_state_agent"],
            description="Belief_State_Agent interprets the latest percept and refines an evolving first-person belief state of the environment. Never suggests next actions.",
            llm_config=self.reasoner_config,
            human_input_mode="NEVER",
            is_termination_msg=self._make_belief_state_termination_fn(),
        )
        self.agents_info[self.belief_state_agent.name] = {
            "Prompt": self.belief_state_agent.system_message,
            "Description": self.belief_state_agent.description,
        }

        self.external_perception_agent = ConversableAgent(
            name="External_Perception_Agent",
            description="External_Perception_Agent executes the proposed 'execute_action' function call given by 'Action_Agent' and then parrots the resulting output as feedback.",
            llm_config=None,
            human_input_mode="NEVER",
            is_termination_msg=lambda msg: False,
        )
        self.agents_info[self.external_perception_agent.name] = {
            "Prompt": self.external_perception_agent.system_message,
            "Description": self.external_perception_agent.description,
        }

        self.internal_perception_agent_1 = ConversableAgent(
            name="Internal_Perception_Agent_1",
            description="Internal_Perception_Agent_1 executes the 'record_long_term_memory' function and then parrots the resulting output.",
            llm_config=None,
            human_input_mode="NEVER",
            is_termination_msg=lambda msg: False,
        )
        self.agents_info[self.internal_perception_agent_1.name] = {
            "Prompt": None,
            "Description": self.internal_perception_agent_1.description,
        }

        self.internal_perception_agent_2 = ConversableAgent(
            name="Internal_Perception_Agent_2",
            description="Internal_Perception_Agent_2 executes the 'focus' function and then parrots the resulting output.",
            llm_config=None,
            human_input_mode="NEVER",
            is_termination_msg=lambda msg: False,
        )
        self.agents_info[self.internal_perception_agent_2.name] = {
            "Prompt": None,
            "Description": self.internal_perception_agent_2.description,
        }

        self.internal_perception_agent_3 = ConversableAgent(
            name="Internal_Perception_Agent_3",
            description="Internal_Perception_Agent_3 executes the 'retrieve_memory' function and then parrots the resulting output.",
            llm_config=None,
            human_input_mode="NEVER",
            is_termination_msg=lambda msg: False,
        )
        self.agents_info[self.internal_perception_agent_3.name] = {
            "Prompt": None,
            "Description": self.internal_perception_agent_3.description,
        }

        self.learning_agent = ConversableAgent(
            name="Learning_Agent",
            system_message=_PROMPTS["learning_agent"],
            description="Learning_Agent forms or reinforces generalizable concepts only after successful, observed actions or contrastive outcomes. Prioritizes novel discovery and integrates belief state-based abstraction.",
            llm_config=self.reasoner_config,
            human_input_mode="NEVER",
            is_termination_msg=lambda msg: False,
        )
        self.agents_info[self.learning_agent.name] = {
            "Prompt": self.learning_agent.system_message,
            "Description": self.learning_agent.description,
        }

        self.record_long_term_memory_agent = ConversableAgent(
            name="Record_Long_Term_Memory_Agent",
            system_message=_PROMPTS["record_long_term_memory_agent"],
            description="Record_Long_Term_Memory_Agent calls the 'record_long_term_memory' function with the concept given by 'Learning_Agent' as the argument.",
            llm_config=self.support_config,
            human_input_mode="NEVER",
            is_termination_msg=lambda msg: False,
        )
        self.agents_info[self.record_long_term_memory_agent.name] = {
            "Prompt": self.record_long_term_memory_agent.system_message,
            "Description": self.record_long_term_memory_agent.description,
        }

        self.start_agent = self.external_perception_agent

        self.allowed_transitions = {
            self.action_agent: [self.external_perception_agent],
            # Route through Echo_Agent to broadcast tool results as plain text.
            self.external_perception_agent: [self.echo_agent],
            self.echo_agent: [self.belief_state_agent],
            self.belief_state_agent: [
                self.action_agent,
                self.retrieve_memory_agent,
                self.focus_agent,
                self.learning_agent,
                self.thinking_agent,
            ],
            self.retrieve_memory_agent: [self.internal_perception_agent_3],
            self.internal_perception_agent_3: [
                self.thinking_agent,
                self.learning_agent,
            ],
            self.thinking_agent: [self.action_agent],
            self.learning_agent: [self.record_long_term_memory_agent],
            self.record_long_term_memory_agent: [self.internal_perception_agent_1],
            self.internal_perception_agent_1: [self.thinking_agent],
            self.internal_perception_agent_2: [self.belief_state_agent],
            self.focus_agent: [self.internal_perception_agent_2],
        }

        for fromAgent, toAgents in self.allowed_transitions.items():
            self.agents_info[fromAgent.name]["Allowed Transitions"] = [
                a.name for a in toAgents
            ]

        with open(self.log_paths["agents_info_path"], "w") as f:
            json.dump(self.agents_info, f, indent=4)

    def initialize_groupchat(self):
        # Define Active Agents (Exclude Echo_Agent from selection list)
        active_agents = [
            self.action_agent,
            self.thinking_agent,
            self.internal_perception_agent_1,
            self.internal_perception_agent_2,
            self.internal_perception_agent_3,
            self.belief_state_agent,
            self.retrieve_memory_agent,
            self.learning_agent,
            self.record_long_term_memory_agent,
            self.focus_agent,
            self.external_perception_agent,
            self.echo_agent,
        ]
        active_agents = [a for a in active_agents if a is not None]
        active_agent_names = [a.name for a in active_agents]

        # 3. Filter Transitions to match active agents
        filtered_transitions = {}
        for speaker, next_speakers in self.allowed_transitions.items():
            if speaker.name in active_agent_names:
                valid_next = [s for s in next_speakers if s.name in active_agent_names]
                filtered_transitions[speaker] = valid_next

        # Store topology so _reinit_groupchat() can rebuild GroupChat/Manager each game.
        self._active_agents = active_agents
        self._filtered_transitions = filtered_transitions

        # 5 JIT SCRUBBING (Input Protection for LLMs)
        # Use TransformMessages (applied just before the API call, on the correct
        # _oai_messages data) instead of a register_reply hook (which only modifies
        # groupchat.messages and is ignored by _generate_oai_reply).
        #
        # IMPORTANT: Action_Agent must NOT have FlattenToolMessages applied.
        # FlattenToolMessages strips 'tool_calls' keys from all prior messages,
        # so after a few rounds Action_Agent's LLM context contains no tool_call
        # examples and the model switches to plain-text output instead of calling
        # execute_action() — breaking the observation relay for the rest of the game.
        # Action_Agent only needs ConvertOrphanedToolMessages; see action_scrubber below.
        cognitive_agents_to_scrub = [
            self.thinking_agent,
            self.belief_state_agent,
            self.retrieve_memory_agent,
            self.learning_agent,
            self.record_long_term_memory_agent,
            self.focus_agent,
        ]

        self._full_scrubber = transform_messages.TransformMessages(
            transforms=[
                MessageHistoryLimiter(
                    max_messages=60
                ),  # was 50; raised to reduce cliff-pruning
                FlattenToolMessages(),
            ]
        )
        for agent in cognitive_agents_to_scrub:
            if agent is not None:
                self._full_scrubber.add_to_agent(agent)

        # Action_Agent must NOT have MessageHistoryLimiter: trimming its context
        # causes ConvertOrphanedToolMessages to strip the tool_call/response pairs,
        # leaving only plain-text "[Calling execute_action]" examples — after which
        # the model outputs text instead of JSON tool calls, stalling the game.
        # Action_Agent messages are tiny (one tool call each) so no limiter is needed.
        action_scrubber = transform_messages.TransformMessages(
            transforms=[ConvertOrphanedToolMessages()]
        )
        if self.action_agent is not None:
            action_scrubber.add_to_agent(self.action_agent)

        # 4. Initialize GroupChat and Manager (also called each game via _reinit_groupchat).
        self._reinit_groupchat()

    def _reinit_groupchat(self):
        """Create a fresh GroupChat and GroupChatManager for the upcoming game.

        Called once at startup (from initialize_groupchat) and again at the start
        of every subsequent game (from set_environment).  Recreating these objects
        gives each game a completely clean slate — GroupChatManager accumulates
        _oai_messages keyed by every agent it spoke with, so without reinit the
        second game inherits 100+ rounds of stale history that can cause the first
        LLM call to hang or produce corrupted responses.

        Cognitive-agent transforms (_full_scrubber, motor_scrubber) are registered
        directly on agent objects and survive across games; only the GroupChatManager's
        scrubber registration must be repeated here for the new manager instance.
        """
        self.group_chat = GroupChat(
            agents=self._active_agents,
            messages=[],
            allowed_or_disallowed_speaker_transitions=self._filtered_transitions,
            speaker_transitions_type="allowed",
            max_round=self.max_chat_round,
            speaker_selection_method=self.custom_speaker_selection,
        )

        self.group_chat_manager = GroupChatManager(
            groupchat=self.group_chat,
            llm_config=self.llm_config_list[0],
        )

        # Register the full scrubber on the new GroupChatManager instance.
        # (GroupChatManager is not in _active_agents so it was excluded from the
        # cognitive-agent scrubber loop above; it needs the scrubber for the rare
        # cases where speaker_selection_method falls back to "auto".)
        assert hasattr(self, "_full_scrubber") and self._full_scrubber is not None, (
            "_reinit_groupchat() called before initialize_groupchat() completed"
        )
        self._full_scrubber.add_to_agent(self.group_chat_manager)

    def register_log_paths(self):

        # Ensure memory directory and memory files exist
        memory_path = self._get_memory_dir()
        memory_path.mkdir(parents=True, exist_ok=True)

        memory1_path = str(memory_path / "memory1.txt")
        memory2_path = str(memory_path / "memory2.txt")
        result_dict_path = os.path.join(self.log_path, "result_dict.txt")
        agents_info_path = os.path.join(self.log_path, "agents_info.json")
        start_memory1_path = os.path.join(self.log_path, "start_memory1.txt")
        end_memory1_path = os.path.join(self.log_path, "end_memory1.txt")
        start_memory2_path = os.path.join(self.log_path, "start_memory2.txt")
        end_memory2_path = os.path.join(self.log_path, "end_memory2.txt")

        self.log_paths = {
            "memory_dir": str(memory_path),
            "memory1_path": memory1_path,
            "memory2_path": memory2_path,
            "result_dict_path": result_dict_path,
            "agents_info_path": agents_info_path,
            "start_memory1_path": start_memory1_path,
            "end_memory1_path": end_memory1_path,
            "start_memory2_path": start_memory2_path,
            "end_memory2_path": end_memory2_path,
        }

        for path in self.log_paths.values():
            if not os.path.exists(path):
                open(path, "w").close()  # Create an empty file

        with (
            open(self.log_paths["memory1_path"]) as src,
            open(self.log_paths["start_memory1_path"], "w") as dst,
        ):
            content = src.read()
            dst.write(content)

        with (
            open(self.log_paths["memory2_path"]) as src,
            open(self.log_paths["start_memory2_path"], "w") as dst,
        ):
            content = src.read()
            dst.write(content)

    def register_game_log_paths(self):

        game_path = os.path.join(self.log_path, f"game_{self.game_no}")
        os.makedirs(game_path, exist_ok=True)

        task_path = os.path.join(game_path, "task.txt")
        history_path = os.path.join(game_path, "history.txt")
        concept_path = os.path.join(game_path, "concepts.txt")
        admissible_commands_path = os.path.join(game_path, "admissible_commands.txt")
        chat_history_path = os.path.join(game_path, "chat_history.txt")
        result_path = os.path.join(game_path, "result.txt")
        error_message_path = os.path.join(game_path, "error_message.txt")

        self.log_paths.update(
            {
                "task_path": task_path,
                "history_path": history_path,
                "concept_path": concept_path,
                "admissible_commands_path": admissible_commands_path,
                "chat_history_path": chat_history_path,
                "result_path": result_path,
                "error_message_path": error_message_path,
            }
        )

        for path in self.log_paths.values():
            if not os.path.exists(path):
                open(path, "w").close()  # Create an empty file

        if self.task_status != "INCOMPLETE":
            with (
                open(self.log_paths["memory1_path"]) as src,
                open(self.log_paths["end_memory1_path"], "w") as dst,
            ):
                content = src.read()
                dst.write(content)

            with (
                open(self.log_paths["memory2_path"]) as src,
                open(self.log_paths["end_memory2_path"], "w") as dst,
            ):
                content = src.read()
                dst.write(content)

    def register_functions(self):

        def execute_action1(suggested_action: str) -> str:
            # Check terminal states first — before any early return — so that
            # empty/text calls after task completion still emit the stop signal.
            if self.task_success:
                self.result_dict[self.game_no] = "SUCCESS"
                with open(self.log_paths["result_path"], "w") as f:
                    f.write(f"Success: {self.success}\n")
                return "STRAWBERRY"

            if self.task_failed and self.rounds_left == 0:
                self.result_dict[self.game_no] = "FAILURE"
                with open(self.log_paths["result_path"], "w") as f:
                    f.write(f"Success: {self.success}\n")
                return "FLEECE"

            if not suggested_action or suggested_action == "do nothing":
                return "NO ACTION EXECUTED. " + focus()

            if self.task_failed:
                self.max_actions += self.max_round_actions
                self.task_failed = False
                return "YOU GET ONE MORE CHANCE! DON'T GIVE UP! " + focus()

            admissible_commands = self.adapter.admissible_actions
            assert admissible_commands, "No admissible commands found."

            previous_observation = self.percept.get("resulting_observation", "")
            canonical_suggested_action = self._canonicalize_suggested_action(
                suggested_action, admissible_commands
            )
            action, action_score = get_best_candidate(
                canonical_suggested_action, admissible_commands
            )
            executed_action = None
            if action_score < 0.98:
                self.adapter.set_observation(
                    f"The action '{suggested_action}' is not in the list of admissible actions for the current timestep."
                )
                # Inadmissible actions don't consume the action budget
            else:
                executed_action = action
                self.adapter.step(action)
                self.success = self.adapter.has_won
                self.num_actions_taken += 1

            reflection = ""
            self.task_status = (
                "COMPLETED"
                if self.success
                else "FAILED"
                if self.num_actions_taken >= self.max_actions
                else "INCOMPLETE"
            )
            if self.task_status == "COMPLETED":
                self.task_success = True
                self.rounds_left -= 1
                reflection = "\nTask COMPLETED. Reflect on your actions and reasoning. Identify what went right and what good decisions led to success. Have Learning_Agent extract any generalizable insights. Action_Agent will close the session automatically."
            elif self.task_status == "FAILED":
                self.task_failed = True
                self.rounds_left -= 1
                reflection = "\nTask FAILED. Reflect on your actions and reasoning. Identify what went wrong and what mistakes led to failure. Have Learning_Agent extract any generalizable insights. Action_Agent will close the session automatically."

            attempted_action = canonical_suggested_action if executed_action else suggested_action
            self.update_percept(attempted_action)
            if canonical_suggested_action != suggested_action:
                self.percept["requested_action"] = suggested_action
                self.percept["canonicalized_action"] = canonical_suggested_action
            self._update_episode_hypothesis_ledger(
                suggested_action=suggested_action,
                executed_action=executed_action,
                previous_observation=previous_observation,
            )
            hypothesis_snapshot = self._get_episode_hypothesis_snapshot(
                max_families=3, max_recent_tests=2
            )
            if hypothesis_snapshot["mechanisms"] or hypothesis_snapshot["recent_tests"]:
                self.percept["episode_hypothesis_ledger"] = hypothesis_snapshot

            with open(self.log_paths["admissible_commands_path"], "a+") as f:
                f.write(f"{self.admissible_actions}\n")
            with open(self.log_paths["history_path"], "a+") as f:
                f.write(
                    f"action: '{suggested_action}'. observation: '{self.adapter.observation}'\n"
                )

            return json.dumps(self.percept) + reflection

        # Register the WRAPPER instead of the method
        assert self.action_agent is not None
        assert self.external_perception_agent is not None
        register_function(
            execute_action1,
            caller=self.action_agent,
            executor=self.external_perception_agent,
            name="execute_action",
            description="Execute an action in the ALFWorld environment and return a structured percept JSON.",
        )

        def record_long_term_memory(concept: str) -> str:

            _, score = get_best_candidate(concept, ["NO CONCEPT at this time."])
            if (
                concept == "NO CONCEPT at this time."
                or len(concept) <= 30
                or score >= 0.7
            ):
                return "I attempted to learn something, but I couldn't formulate any concept."

            concept = concept.replace("\n", " ").replace("\r", " ").strip()

            existing = []
            if os.path.exists(self.log_paths["memory1_path"]):
                with open(self.log_paths["memory1_path"]) as f:
                    existing = [ln.lstrip("- ").strip() for ln in f if ln.strip()]
            if existing:
                _, dup_score = get_best_candidate(concept, existing)
                if dup_score >= 0.85:
                    return "I attempted to learn something, but I couldn't formulate any concept."

            if self.read_only_memory:
                return (
                    f"I learned that {concept}. (memory write skipped — read-only mode)"
                )

            with open(self.log_paths["concept_path"], "a+") as f:
                f.write(f"- {concept}\n")

            with open(self.log_paths["memory1_path"], "a+") as f:
                f.write(f"- {concept}\n")

            self.cluster_knowledge()

            return f"I learned that {concept}."

        def retrieve_memory() -> str:
            return self.retrieve_memory()

        def focus() -> str:
            shared_action_context = self._summarize_admissible_actions(
                self.admissible_actions
            )
            return (
                f"TASK: {self.task}\n"
                f"REPEATING LAST PERCEPT TO HELP CONSTRUCT BELIEF STATE:\n{json.dumps(self.percept)}\n"
                f"EPISODE HYPOTHESIS LEDGER: {json.dumps(self._get_episode_hypothesis_snapshot())}\n"
                f"ACTION SPACE SUMMARY: {json.dumps(shared_action_context)}"
            )

        assert self.focus_agent is not None
        assert self.internal_perception_agent_2 is not None
        register_function(
            focus,
            caller=self.focus_agent,
            executor=self.internal_perception_agent_2,
            description="Resets focus.",
        )

        assert self.record_long_term_memory_agent is not None
        assert self.internal_perception_agent_1 is not None
        register_function(
            record_long_term_memory,
            caller=self.record_long_term_memory_agent,
            executor=self.internal_perception_agent_1,
            description="Records new concept in long-term memory.",
        )

        assert self.retrieve_memory_agent is not None
        assert self.internal_perception_agent_3 is not None
        register_function(
            retrieve_memory,
            caller=self.retrieve_memory_agent,
            executor=self.internal_perception_agent_3,
            description="Retrieves Memory.",
        )

    def cluster_knowledge(self, plot_clusters=False, save_dir="."):
        """
        Get representative concepts using KMeans clustering and optionally save cluster plot.

        Args:
            plot_clusters (bool): Whether to save UMAP cluster visualization.
            save_dir (str): Directory to save plot (if applicable).

        Returns:
            dict: Representative concepts, cluster sizes, cluster members, and chosen_k.
        """

        concept_text = ""
        if os.path.exists(self.log_paths["memory1_path"]):
            with open(self.log_paths["memory1_path"]) as file:
                concept_text = file.read()

        concept_lines = [
            line.strip() for line in concept_text.split("\n") if line.strip()
        ]
        num_concepts = len(concept_lines)
        print(f"🧠 Clustering {num_concepts} concept(s)...")

        empty_result = {
            "representative_concepts": [],
            "cluster_sizes": {},
            "cluster_members": {},
            "chosen_k": 0,
        }
        if num_concepts == 0:
            self._cluster_cache = ("", empty_result)
            return empty_result

        content_hash = hashlib.md5(concept_text.encode()).hexdigest()
        if getattr(self, "_cluster_cache", (None, None))[0] == content_hash:
            return self._cluster_cache[1]

        embeddings = (
            sentence_transformer_model.encode(concept_lines, convert_to_tensor=True)
            .cpu()
            .numpy()
        )

        # Calculate k (clusters) using capped growth function to prevent over-clustering
        chosen_k = max(1, min(num_concepts, int(num_concepts ** (1 / 2))))

        kmeans = KMeans(n_clusters=chosen_k, random_state=42, n_init=10)
        labels = kmeans.fit_predict(embeddings)

        unique_labels, counts = np.unique(labels, return_counts=True)
        cluster_sizes = {label: count for label, count in zip(unique_labels, counts)}
        cluster_members = {i: [] for i in range(chosen_k)}
        for i, label in enumerate(labels):
            cluster_members[label].append(concept_lines[i])

        representative_concepts = []
        self.knowledge = []
        if os.path.exists(self.log_paths["memory2_path"]):
            with open(self.log_paths["memory2_path"], "w") as file:
                for i in range(chosen_k):
                    cluster_indices = [
                        j for j, label in enumerate(labels) if label == i
                    ]
                    center = kmeans.cluster_centers_[i]
                    cluster_embeddings = embeddings[cluster_indices]
                    distances = np.linalg.norm(cluster_embeddings - center, axis=1)

                    if len(distances) == 0:
                        continue  # Skip empty cluster

                    closest_idx = np.argmin(distances)
                    closest_concept_idx = cluster_indices[closest_idx]
                    representative_concept = concept_lines[closest_concept_idx]
                    confidence_score = cluster_sizes[i]

                    # Avoid stripping leading char if it's not needed
                    clean_concept = (
                        representative_concept[1:]
                        if representative_concept.startswith("[")
                        else representative_concept
                    )

                    file.write(
                        f"Cluster {i + 1}; Confidence Score = {confidence_score}; Concept: {clean_concept}\n"
                    )

                    self.knowledge.append(
                        json.dumps(
                            {
                                "cluster_id": int(i + 1),
                                "confidence_score": int(confidence_score),
                                "general_concept": clean_concept,
                            }
                        )
                    )
                    representative_concepts.append(representative_concept)

        if plot_clusters:
            reducer = umap.UMAP(random_state=42)
            embedding_2d = reducer.fit_transform(embeddings)

            plt.figure(figsize=(10, 6))
            for i in range(chosen_k):
                points = embedding_2d[np.array(labels) == i]
                plt.scatter(
                    points[:, 0],
                    points[:, 1],
                    label=f"Cluster {i} ({cluster_sizes[i]})",
                    alpha=0.7,
                )

            plt.title("2D Visualization of Clusters (UMAP)")
            plt.xlabel("UMAP-1")
            plt.ylabel("UMAP-2")
            plt.legend()
            plt.grid(True)
            plt.tight_layout()

            cluster_path = os.path.join(save_dir, "cluster_plot.png")
            plt.savefig(cluster_path)
            plt.close()

        result = {
            "representative_concepts": representative_concepts,
            "cluster_sizes": cluster_sizes,
            "cluster_members": cluster_members,
            "chosen_k": chosen_k,
        }
        self._cluster_cache = (content_hash, result)
        return result

    def generate_initial_message(self):
        """
        Generate the initial message sent to the group of agents, summarizing their purpose, constraints,
        roles, prior concept, and the current task state.
        """
        intro = (
            "You and all other Agents are collectively a unified cognitive system named ALFRED. "
            "Each of you plays a distinct role in perception, memory, planning, reasoning, or action execution. "
            "Together, your goal is to solve the following task as efficiently and intelligently as possible.\n\n"
        )

        task_section = f"--- TASK DESCRIPTION ---\n{self.task}\n\n"

        constraints_section = (
            f"--- ENVIRONMENTAL CONSTRAINTS ---\n"
            f"- Max chat rounds allowed: {self.max_chat_round} (represents internal cognitive transitions).\n"
            f"- Max environment actions allowed: {self.max_actions} (physical interactions only).\n\n"
        )

        relevant_knowledge = retrieve_relevant_concepts(
            self.knowledge,
            self.task,
            k=self.rag_concept_k_initial,
            cache=self._concept_rag_cache,
        )
        relevant_episodes = retrieve_relevant_episodes(
            self.prev_episodic_memories,
            self.task,
            k=self.rag_episode_k_initial,
            cache=self._episodic_rag_cache,
        )
        memory_section = "--- PRIOR KNOWLEDGE & EPISODIC MEMORY ---\n"
        memory_section += (
            json.dumps(
                {
                    "knowledge": relevant_knowledge,
                    "recent_episodic_memories": relevant_episodes,
                    "current_episode_memory": self.curr_episodic_memory,
                    "episode_hypothesis_ledger": self._get_episode_hypothesis_snapshot(),
                },
            )
            + "\n\n"
        )

        state_section = "--- CURRENT STATE ---\n" + json.dumps(self.percept) + "\n"

        final_prompt = (
            "Begin cognitive deliberation. Coordinate through structured, grounded reasoning. "
            "Use prior knowledge when relevant, minimize communication and actions, and confirm task completion explicitly through perceptual feedback."
        )
        return (
            intro
            + task_section
            + constraints_section
            + memory_section
            + state_section
            + final_prompt
        )

    def retrieve_memory(self):
        query = self.task
        if self.curr_episodic_memory:
            last = self.curr_episodic_memory[-1]
            query += " " + (last if isinstance(last, str) else json.dumps(last))

        print(
            f"🔍 Retrieving memory: {len(self.prev_episodic_memories)} episode(s), "
            f"{len(self.knowledge)} concept(s) → top-{self.rag_episode_k} episodes, "
            f"top-{self.rag_concept_k} concepts"
        )

        relevant_episodes = retrieve_relevant_episodes(
            self.prev_episodic_memories,
            query,
            k=self.rag_episode_k,
            cache=self._episodic_rag_cache,
        )
        relevant_knowledge = retrieve_relevant_concepts(
            self.knowledge, query, k=self.rag_concept_k, cache=self._concept_rag_cache
        )

        self.memory = json.dumps(
            {
                "knowledge": relevant_knowledge,
                "previous_episodic_memories": relevant_episodes,
                "current_episode_memory": self.curr_episodic_memory,
                "episode_hypothesis_ledger": self._get_episode_hypothesis_snapshot(),
            },
        )
        return self.memory

    def custom_speaker_selection(self, last_speaker, groupchat):
        messages = groupchat.messages
        if not messages:
            return self.external_perception_agent

        last_msg = messages[-1]

        # Route tool calls to the executor defined in the transition graph.
        # Using allowed_transitions instead of hardcoding external_perception_agent
        # ensures retrieve_memory/focus/record_long_term_memory are sent to the
        # correct Internal_Perception_Agent, not to external_perception_agent which
        # only knows about execute_action.
        if "tool_calls" in last_msg:
            executors = self.allowed_transitions.get(last_speaker, [])
            if executors:
                return executors[0]
            return self.external_perception_agent

        # Route tool responses via the transition graph (e.g. Internal_Perception_Agent_1
        # → Thinking_Agent, Internal_Perception_Agent_2 → Belief_State_Agent, etc.).
        # Only fall back to echo_agent for external_perception_agent responses.
        if last_msg.get("role") == "tool":
            executors = self.allowed_transitions.get(last_speaker, [])
            if executors:
                return executors[0]
            return self.echo_agent

        # Stuck-state early termination: tick _stale_action_count exactly once
        # per complete observation relay — i.e. only when echo_agent just spoke.
        # That way the count equals the number of full action cycles (Action →
        # External_Perception → Echo) during which no new execute_action call
        # was made, giving a clean semantic: "8 stale rounds = 8 full cycles
        # with no action executed."
        #
        # Note: echo_agent's own _relay_state["stale_count"] independently fires
        # a STRAWBERRY termination after 6 consecutive relays with NO tool message
        # at all (Action_Agent output plain text instead of calling execute_action).
        # That guard fires first in a full stall; this counter covers the separate
        # (unlikely) case where execute_action IS called but num_actions_taken stalls.
        if last_speaker is self.echo_agent:
            if self.num_actions_taken == self._last_seen_actions_taken:
                self._stale_action_count += 1
            else:
                self._stale_action_count = 0
                self._last_seen_actions_taken = self.num_actions_taken
            if self._stale_action_count >= 8:
                # +1: the AutoGen termination check (i == max_round - 1) runs
                # BEFORE select_speaker, so we target the NEXT iteration.
                # Guard: only reduce max_round, never increase it.
                new_cap = len(messages) + 1
                if self.group_chat.max_round > new_cap:
                    self.group_chat.max_round = new_cap

        # Standard Graph Transitions
        possible_speakers = self.allowed_transitions.get(last_speaker, [])

        # Gate Learning_Agent: only invoke it after a task outcome (success/failure).
        if (
            self.learning_agent in possible_speakers
            and not self.task_success
            and not self.task_failed
        ):
            possible_speakers = [
                s for s in possible_speakers if s is not self.learning_agent
            ]

        # Gate Thinking_Agent: only invoke when Belief_State_Agent signals uncertainty.
        # Also skip if the belief state content is identical to the previous round —
        # repeating Thinking_Agent on an unchanged belief wastes tokens with no gain.
        if (
            last_speaker is self.belief_state_agent
            and self.thinking_agent in possible_speakers
            and not self.task_success
            and not self.task_failed
        ):
            current_content = last_msg.get("content") or ""
            if not current_content.lstrip().upper().startswith("BELIEF STATE:"):
                repaired = self._synthesize_belief_state_fallback(current_content)
                last_msg["content"] = repaired
                current_content = repaired
                self._last_belief_content = ""
                self._consecutive_thinking_count = 0
            if (
                self._UNCERTAINTY_RE.search(current_content.lower())
                or "no observation" in current_content.lower()
            ):
                if (
                    current_content == self._last_belief_content
                    and self._consecutive_thinking_count >= 1
                ):
                    # Identical belief state fired again — skip Thinking_Agent
                    self._consecutive_thinking_count = 0
                    self._last_belief_content = ""
                    return self.action_agent
                self._last_belief_content = current_content
                self._consecutive_thinking_count += 1
                return self.thinking_agent
            self._last_belief_content = ""
            self._consecutive_thinking_count = 0
            return self.action_agent

        # Safety valve: if the task is done and the conversation is still
        # running (e.g. Action_Agent stuck outputting text instead of calling
        # execute_action), cap max_round to force termination.
        if self.task_success or (self.task_failed and self.rounds_left == 0):
            if not hasattr(self, "_task_done_msg_count"):
                self._task_done_msg_count = len(messages)
            elif len(messages) > self._task_done_msg_count + 12:
                new_cap = len(messages) + 1
                if self.group_chat.max_round > new_cap:
                    self.group_chat.max_round = new_cap

        if len(possible_speakers) == 1:
            return possible_speakers[0]

        return "auto"
