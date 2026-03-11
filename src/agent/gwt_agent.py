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
        "are",
        "around",
        "as",
        "at",
        "be",
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
        "inspect": ("inspect", "examine", "look at"),
        "relation": (
            "connect",
            "link",
            "disconnect",
            "circuit",
            "electrical circuit",
            "wire",
            "anode",
            "cathode",
        ),
        "relocation": ("go to", "enter", "move", "bring", "carry", "transport"),
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
    _TASK_SEARCH_HINTS = (
        "find",
        "locate",
        "identify",
        "discover",
        "determine whether",
    )
    _TASK_ARTIFACT_CREATION_HINTS = (
        "create",
        "make",
        "produce",
        "synthesize",
    )
    _TASK_PROCEDURAL_HINTS = (
        "first",
        "then",
        "and then",
        "next",
    )
    _TASK_ORDERING_HINTS = {
        "ordered_sequence": (
            "earliest",
            "latest",
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
    _LIFECYCLE_TASK_HINTS = (
        "life stage",
        "life stages",
        "lifecycle",
        "life cycle",
    )
    _TASK_ROLE_PATTERNS = {
        "primary_targets": (
            r"\bfocus on\s+(.+?)(?=$|[.;,\n]|\bthen\b|\band then\b)",
            r"\bturn on\s+(.+?)(?=$|[.;,\n]|\bby\b|\busing\b|\bwith\b|\bthen\b|\band then\b)",
            r"\bactivate\s+(.+?)(?=$|[.;,\n]|\bby\b|\busing\b|\bwith\b|\bthen\b|\band then\b)",
            r"\bmove\s+(.+?)\s+\bto\b",
            r"\b(?:put|place)\s+(.+?)\s+\b(?:in|into|on)\b",
        ),
        "supporting_targets": (
            r"\busing\s+(.+?)(?=$|[.;,\n]|\bthen\b|\band then\b)",
            r"\bwith\s+(.+?)(?=$|[.;,\n]|\bthen\b|\band then\b)",
            r"\bmove\s+.+?\s+\bto\b\s+(.+?)(?=$|[.;,\n]|\bthen\b|\band then\b)",
            r"\b(?:put|place)\s+.+?\s+\b(?:in|into|on)\b\s+(.+?)(?=$|[.;,\n]|\bthen\b|\band then\b)",
            r"\bconnect\s+.+?\s+\bto\b\s+(.+?)(?=$|[.;,\n]|\bthen\b|\band then\b)",
        ),
        "required_relations": (
            r"\bcreate\s+(.+?)(?=$|[.;,\n]|\bthen\b|\band then\b)",
            r"\bconnect\s+(.+?)(?=$|[.;,\n]|\bthen\b|\band then\b)",
            r"\b(an electrical circuit)\b",
        ),
    }
    _TASK_STATE_CHANGE_HINTS = {
        "melt": {"direction": "warm"},
        "freeze": {"direction": "cool"},
        "boil": {"direction": "warm"},
        "heat": {"direction": "warm"},
        "warm": {"direction": "warm"},
        "cool": {"direction": "cool"},
        "chill": {"direction": "cool"},
    }
    _STATE_CHANGE_TASK_STOPWORDS = {
        "actions",
        "also",
        "boiling",
        "combusting",
        "combustion",
        "without",
        "will",
    }
    _MEASUREMENT_TASK_STOPWORDS = {
        "above",
        "below",
        "celsius",
        "degrees",
        "degree",
        "melting",
        "next",
        "point",
        "which",
    }
    _ARTIFACT_CREATION_TASK_STOPWORDS = {
        "chemistry",
        "completely",
        "done",
        "part",
        "way",
        "when",
    }
    _TASK_ENTITY_STOPWORDS = _TASK_STOPWORDS | {
        "acceptable",
        "action",
        "actions",
        "change",
        "compound",
        "compounds",
        "create",
        "creating",
        "earliest",
        "electrically",
        "identify",
        "latest",
        "life",
        "locate",
        "matter",
        "point",
        "powered",
        "powering",
        "powers",
        "sequence",
        "stage",
        "stages",
        "starting",
        "transform",
        "transformation",
        "whether",
        "will",
    }
    _GENERIC_PRIMARY_TARGET_TOKENS = {
        "thing",
        "object",
        "item",
        "target",
        "entity",
    }
    _NON_CANDIDATE_REFERENT_TOKENS = {
        "agent",
        "air",
        "inventory",
        "hallway",
        "kitchen",
        "bathroom",
        "bedroom",
        "greenhouse",
        "outside",
        "workshop",
        "living",
        "room",
        "studio",
        "door",
    }
    _THINKING_PREFIXES = (
        "IDEA:",
        "STRATEGY:",
        "HYPOTHESIS:",
        "INSIGHT:",
        "QUESTION:",
        "THEORY:",
        "EXPLANATION:",
        "TACTIC:",
    )
    _LIFECYCLE_STAGE_PATTERNS = (
        ("egg", (r"\begg\b",)),
        ("seed", (r"\bseed\b",)),
        (
            "germinating",
            (r"\bgerminat(?:e|es|ed|ing|ion)\b", r"\bsprout(?:s|ed|ing)?\b"),
        ),
        ("seedling", (r"\bseedling\b", r"\bsapling\b", r"\bshoot\b")),
        (
            "juvenile",
            (
                r"\bhatchling\b",
                r"\bjuvenile\b",
                r"\blarva\b",
                r"\bnymph\b",
                r"\btadpole\b",
                r"\bpupa\b",
                r"\bfledgling\b",
            ),
        ),
        (
            "flowering",
            (
                r"\bflower(?:ing)?\b",
                r"\bblossom(?:ing)?\b",
                r"\bpollen\b",
                r"\breproducing\b",
            ),
        ),
        ("fruiting", (r"\bfruit(?:ing)?\b",)),
        ("adult", (r"\badult\b", r"\bmature\b", r"\bfull[- ]grown\b")),
        (
            "dead",
            (
                r"\bdead\b",
                r"\bdecay(?:ing)?\b",
                r"\bdying\b",
                r"\bwither(?:ed|ing)?\b",
                r"\brotten\b",
            ),
        ),
    )
    _LIFECYCLE_STAGE_ORDER = {
        "egg": 0,
        "seed": 0,
        "germinating": 1,
        "seedling": 2,
        "juvenile": 2,
        "flowering": 3,
        "fruiting": 4,
        "adult": 5,
        "dead": 6,
    }
    _CONTAINER_REFERENT_TOKENS = {
        "basket",
        "bin",
        "bottle",
        "bowl",
        "box",
        "bucket",
        "cabinet",
        "chest",
        "closet",
        "container",
        "cupboard",
        "drawer",
        "freezer",
        "fridge",
        "jar",
        "locker",
        "pot",
        "refrigerator",
        "shelf",
        "tray",
    }
    _STAGE_EVIDENCE_FILTER_TOKENS = _CONTAINER_REFERENT_TOKENS | {
        "containing",
        "flower",
        "self",
        "soil",
        "substance",
        "water",
        "watering",
    }
    _REFERENT_SIGNATURE_STOPWORDS = {
        "self",
        "watering",
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
        self._completed_focus_targets: list[str] = []
        self._focused_stage_labels: list[str] = []
        self._observed_stage_labels: list[str] = []
        self._stage_evidence_by_referent: dict[str, list[str]] = {}
        self._referent_resolution_events: list[dict] = []
        self._active_candidate: str | None = None
        self._candidate_states: dict[str, dict] = {}
        self._rejected_candidates: list[str] = []
        self._action_observation_signatures: dict[tuple[str, str, str], int] = {}
        self._grounded_artifacts: dict[str, dict] = {}
        self._grounded_substances: dict[str, dict] = {}
        self._exhausted_container_targets: list[str] = []
        self._measurement_observations: list[dict] = []
        self._selected_measurement_branch_target: str | None = None
        self._containment_by_object: dict[str, str] = {}
        self._invalid_exact_actions: dict[str, int] = {}

    def _build_task_contract(self, task: str) -> dict:
        task_lower = (task or "").lower()
        lifecycle_sequence = any(
            self._task_contains_hint(task_lower, hint)
            for hint in self._LIFECYCLE_TASK_HINTS
        )
        procedural_sequence = any(
            self._task_contains_hint(task_lower, hint)
            for hint in self._TASK_PROCEDURAL_HINTS
        )
        search_mode = any(
            self._task_contains_hint(task_lower, hint)
            for hint in self._TASK_SEARCH_HINTS
        )
        (
            target_substances,
            desired_transformation,
            transformation_direction,
        ) = self._extract_state_change_goal(task)
        state_change_task = bool(desired_transformation)
        (
            measurement_property,
            measurement_target,
            measurement_instrument,
            measurement_branch_targets,
            measurement_branches,
        ) = self._extract_measurement_contract(task)
        measurement_task = bool(measurement_property and measurement_target)
        role_phrases = self._extract_task_role_phrases(task)
        (
            artifact_type,
            artifact_intermediate_targets,
            artifact_final_targets,
            artifact_descriptor_tokens,
        ) = self._extract_artifact_creation_contract(task, role_phrases)
        artifact_creation_task = bool(
            artifact_type and not state_change_task and not measurement_task
        )
        measurement_branch_tokens: set[str] = set()
        for branch_target in measurement_branch_targets:
            measurement_branch_tokens.update(
                self._extract_runtime_tokens(
                    branch_target,
                    stopwords=self._TASK_ENTITY_STOPWORDS
                    | self._ACTION_COMMAND_STOPWORDS,
                    limit=6,
                )
            )
        measurement_numeric_tokens = {
            token
            for branch in measurement_branches
            for token in re.findall(r"[0-9]+", str(branch["threshold"]))
        }
        transformation_tokens = set(
            self._extract_runtime_tokens(
                desired_transformation,
                stopwords=self._TASK_STOPWORDS | self._ACTION_COMMAND_STOPWORDS,
            )
        )
        required_families: list[str] = []
        family_hint_tokens: set[str] = set()
        for family, hints in self._TASK_FAMILY_HINTS.items():
            if any(self._task_contains_hint(task_lower, hint) for hint in hints):
                required_families.append(family)
                for hint in hints:
                    family_hint_tokens.update(
                        self._extract_runtime_tokens(
                            hint,
                            stopwords=self._TASK_STOPWORDS
                            | self._ACTION_COMMAND_STOPWORDS,
                        )
                    )

        support_families: list[str] = []
        if (search_mode or state_change_task) and "inspect" not in required_families:
            support_families.append("inspect")

        ordering_cues: list[str] = []
        ordering_tokens: set[str] = set()
        for cue, hints in self._TASK_ORDERING_HINTS.items():
            if any(self._task_contains_hint(task_lower, hint) for hint in hints):
                ordering_cues.append(cue)
                for hint in hints:
                    ordering_tokens.update(
                        self._extract_runtime_tokens(
                            hint,
                            stopwords=self._TASK_STOPWORDS
                            | self._ACTION_COMMAND_STOPWORDS,
                        )
                    )

        if measurement_task:
            role_phrases["primary_targets"] = [measurement_target]
            role_phrases["supporting_targets"] = []
            if measurement_instrument:
                role_phrases["supporting_targets"].append(measurement_instrument)
        if target_substances and (
            not role_phrases["primary_targets"]
            or all(
                not self._extract_runtime_tokens(
                    phrase,
                    stopwords=self._TASK_ENTITY_STOPWORDS
                    | self._ACTION_COMMAND_STOPWORDS,
                )
                for phrase in role_phrases["primary_targets"]
            )
        ):
            for substance in target_substances:
                if substance not in role_phrases["primary_targets"]:
                    role_phrases["primary_targets"].append(substance)
        candidate_classes = self._extract_candidate_classes(task)
        destination_container, destination_room = self._extract_destination_roles(task)
        if (
            destination_container
            and destination_container not in role_phrases["supporting_targets"]
        ):
            role_phrases["supporting_targets"].append(destination_container)
        if (
            destination_room
            and destination_room not in role_phrases["supporting_targets"]
        ):
            role_phrases["supporting_targets"].append(destination_room)
        target_entities: list[str] = []
        for role in (
            "primary_targets",
            "supporting_targets",
            "required_relations",
            "candidate_classes",
            "target_substances",
            "artifact_type",
            "artifact_intermediate_targets",
            "artifact_final_targets",
            "measurement_target",
            "measurement_instrument",
            "destination_container",
            "destination_room",
        ):
            phrases = role_phrases.get(role, [])
            if role == "candidate_classes":
                phrases = candidate_classes
            elif role == "target_substances":
                phrases = target_substances
            elif role == "artifact_type":
                phrases = [artifact_type] if artifact_type else []
            elif role == "artifact_intermediate_targets":
                phrases = artifact_intermediate_targets
            elif role == "artifact_final_targets":
                phrases = artifact_final_targets
            elif role == "measurement_target":
                phrases = [measurement_target] if measurement_target else []
            elif role == "measurement_instrument":
                phrases = [measurement_instrument] if measurement_instrument else []
            elif role == "destination_container":
                phrases = [destination_container] if destination_container else []
            elif role == "destination_room":
                phrases = [destination_room] if destination_room else []
            for phrase in phrases:
                for token in self._extract_runtime_tokens(
                    phrase,
                    stopwords=self._TASK_ENTITY_STOPWORDS
                    | self._ACTION_COMMAND_STOPWORDS,
                    limit=6,
                ):
                    if token not in target_entities:
                        target_entities.append(token)

        raw_target_entities = self._extract_runtime_tokens(
            task,
            stopwords=(
                self._TASK_ENTITY_STOPWORDS
                | self._ACTION_COMMAND_STOPWORDS
                | family_hint_tokens
                | ordering_tokens
                | transformation_tokens
                | (self._STATE_CHANGE_TASK_STOPWORDS if state_change_task else set())
                | (
                    self._ARTIFACT_CREATION_TASK_STOPWORDS
                    if artifact_creation_task
                    else set()
                )
                | (self._MEASUREMENT_TASK_STOPWORDS if measurement_task else set())
                | (measurement_branch_tokens if measurement_task else set())
                | (measurement_numeric_tokens if measurement_task else set())
            ),
            limit=8,
        )
        raw_target_set = set(raw_target_entities)
        for token in raw_target_entities:
            if token.endswith("s") and token[:-1] in raw_target_set:
                continue
            if token not in target_entities:
                target_entities.append(token)

        return {
            "required_families": required_families,
            "support_families": support_families,
            "target_entities": target_entities,
            "ordering_cues": ordering_cues,
            "procedural_sequence": procedural_sequence,
            "lifecycle_sequence": lifecycle_sequence,
            "state_change_task": state_change_task,
            "artifact_creation_task": artifact_creation_task,
            "measurement_task": measurement_task,
            "search_mode": search_mode,
            "candidate_classes": candidate_classes,
            "primary_targets": role_phrases["primary_targets"],
            "supporting_targets": role_phrases["supporting_targets"],
            "target_substances": target_substances,
            "artifact_type": [artifact_type] if artifact_type else [],
            "artifact_intermediate_targets": artifact_intermediate_targets,
            "artifact_final_targets": artifact_final_targets,
            "artifact_descriptor_tokens": artifact_descriptor_tokens,
            "measurement_property": measurement_property,
            "measurement_target": [measurement_target] if measurement_target else [],
            "measurement_instrument": (
                [measurement_instrument] if measurement_instrument else []
            ),
            "measurement_branch_targets": measurement_branch_targets,
            "measurement_branches": measurement_branches,
            "destination_container": (
                [destination_container] if destination_container else []
            ),
            "destination_room": [destination_room] if destination_room else [],
            "required_relations": role_phrases["required_relations"],
            "desired_transformation": desired_transformation,
            "transformation_direction": transformation_direction,
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
            re.search(
                rf"(?<![a-z0-9]){re.escape(normalized_hint)}(?![a-z0-9])",
                normalized_task,
            )
        )

    @staticmethod
    def _normalize_runtime_text(text: str) -> str:
        return re.sub(r"\s+", " ", (text or "").strip().lower())

    def _extract_task_keywords(self) -> list[str]:
        task_contract = self._get_task_contract()
        keywords: list[str] = []
        for token in task_contract.get("target_entities", []):
            if token not in keywords:
                keywords.append(token)
        transformation = task_contract.get("desired_transformation")
        if transformation and transformation not in keywords:
            keywords.append(transformation)
        measurement_property = task_contract.get("measurement_property")
        if measurement_property and measurement_property not in keywords:
            keywords.append(measurement_property)
        for artifact_type in task_contract.get("artifact_type", []):
            if artifact_type and artifact_type not in keywords:
                keywords.append(artifact_type)
        if self._selected_measurement_branch_target:
            for token in self._extract_runtime_tokens(
                self._selected_measurement_branch_target,
                stopwords=self._TASK_ENTITY_STOPWORDS | self._ACTION_COMMAND_STOPWORDS,
                limit=4,
            ):
                if token not in keywords:
                    keywords.append(token)
        if keywords:
            return keywords
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
            if (len(token) < 3 and not token.isdigit()) or token in active_stopwords:
                continue
            if token not in tokens:
                tokens.append(token)
            if limit is not None and len(tokens) >= limit:
                break
        return tokens

    def _is_lifecycle_task(self, task_contract: dict | None = None) -> bool:
        contract = task_contract or self._get_task_contract()
        return bool(contract.get("lifecycle_sequence"))

    def _extract_state_change_goal(self, task: str) -> tuple[list[str], str, str]:
        normalized_task = self._normalize_runtime_text(task).replace("a(n)", "an")
        desired_transformation = ""
        transformation_direction = ""

        for verb, metadata in self._TASK_STATE_CHANGE_HINTS.items():
            if self._task_contains_hint(normalized_task, verb):
                desired_transformation = verb
                transformation_direction = metadata["direction"]
                break

        if not desired_transformation:
            return [], "", ""

        target_substances: list[str] = []
        patterns = (
            rf"\b{re.escape(desired_transformation)}(?:\s+up|\s+down)?\s+(?:an?\s+|the\s+)?(.+?)(?=$|[.;,\n]|\bfirst\b|\bthen\b|\band then\b)",
        )
        for pattern in patterns:
            for match in re.finditer(pattern, normalized_task):
                phrase = self._normalize_task_phrase(
                    match.group(1), role="primary_targets"
                )
                if phrase and phrase not in target_substances:
                    target_substances.append(phrase)
                if len(target_substances) >= 2:
                    break
            if target_substances:
                break

        return target_substances[:2], desired_transformation, transformation_direction

    def _extract_artifact_type_from_phrase(self, phrase: str) -> str:
        tokens = self._extract_runtime_tokens(
            phrase,
            stopwords=(
                self._TASK_ENTITY_STOPWORDS
                | self._ACTION_COMMAND_STOPWORDS
                | self._ARTIFACT_CREATION_TASK_STOPWORDS
            ),
            limit=6,
        )
        for token in reversed(tokens):
            if token not in {"color", "intermediate", "secondary"}:
                return token
        return ""

    def _extract_artifact_creation_contract(
        self, task: str, role_phrases: dict[str, list[str]]
    ) -> tuple[str, list[str], list[str], list[str]]:
        normalized_task = self._normalize_runtime_text(task).replace("a(n)", "an")
        if not any(
            self._task_contains_hint(normalized_task, hint)
            for hint in self._TASK_ARTIFACT_CREATION_HINTS
        ):
            return "", [], [], []

        created_phrase = ""
        for pattern in (
            r"\b(?:create|make|produce|synthesize)\s+(.+?)(?=$|[.;,\n]|\bwhen\b|\bfirst\b|\bthen\b|\band then\b)",
        ):
            match = re.search(pattern, normalized_task)
            if not match:
                continue
            created_phrase = self._normalize_task_phrase(
                match.group(1), role="required_relations"
            )
            if created_phrase:
                break

        artifact_type = self._extract_artifact_type_from_phrase(created_phrase)
        if not artifact_type:
            return "", [], [], []

        product_targets = [
            phrase
            for phrase in role_phrases.get("primary_targets", [])
            if artifact_type
            in self._extract_runtime_tokens(
                phrase,
                stopwords=self._TASK_ENTITY_STOPWORDS | self._ACTION_COMMAND_STOPWORDS,
                limit=6,
            )
        ]
        if not product_targets:
            return "", [], [], []

        intermediate_targets = product_targets[:-1]
        final_targets = product_targets[-1:]
        descriptor_tokens: list[str] = []
        for phrase in [created_phrase, *product_targets]:
            for token in self._extract_runtime_tokens(
                phrase,
                stopwords=(
                    self._TASK_ENTITY_STOPWORDS
                    | self._ACTION_COMMAND_STOPWORDS
                    | self._ARTIFACT_CREATION_TASK_STOPWORDS
                    | {artifact_type}
                ),
                limit=6,
            ):
                if token not in descriptor_tokens:
                    descriptor_tokens.append(token)

        return (
            artifact_type,
            intermediate_targets[:2],
            final_targets[:2],
            descriptor_tokens[:6],
        )

    def _normalize_measurement_phrase(self, phrase: str) -> str:
        normalized = self._normalize_runtime_text(phrase)
        if not normalized:
            return ""

        normalized = re.split(
            r"\b(?:that|which|where|because|since|until|while|then|and then)\b",
            normalized,
            maxsplit=1,
        )[0]
        normalized = re.sub(r"^(?:a|an|the)\s+", "", normalized)
        normalized = normalized.strip(" .,:;")
        tokens = self._extract_runtime_tokens(
            normalized,
            stopwords=self._TASK_STOPWORDS | self._ACTION_COMMAND_STOPWORDS,
            limit=6,
        )
        return " ".join(tokens[:4])

    def _normalize_measurement_target_phrase(self, phrase: str) -> str:
        normalized = self._normalize_runtime_text(phrase)
        if not normalized:
            return ""

        normalized = re.split(
            r"\b(?:that|which|where|because|since|until|while|then|and then)\b",
            normalized,
            maxsplit=1,
        )[0]
        normalized = re.sub(r"^(?:a|an|the)\s+", "", normalized)
        normalized = normalized.strip(" .,:;")
        active_stopwords = (self._TASK_STOPWORDS | self._ACTION_COMMAND_STOPWORDS) - {
            "substance"
        }
        raw_tokens = re.findall(r"[a-z0-9]+", normalized)
        tokens: list[str] = []
        for index, token in enumerate(raw_tokens):
            if token in active_stopwords:
                continue
            if len(token) < 3 and not token.isdigit():
                if index > 0 and raw_tokens[index - 1] == "substance":
                    tokens.append(token)
                continue
            tokens.append(token)
            if len(tokens) >= 5:
                break
        return " ".join(tokens[:5])

    def _extract_measurement_contract(
        self, task: str
    ) -> tuple[str, str, str, list[str], list[dict]]:
        normalized_task = self._normalize_runtime_text(task).replace("a(n)", "an")
        measurement_property = ""
        measurement_target = ""
        measurement_instrument = ""
        measurement_branch_targets: list[str] = []
        measurement_branches: list[dict] = []

        measure_patterns = (
            r"\bmeasure\s+(?:the\s+)?(.+?)\s+of\s+(?:the\s+)?(.+?)(?=$|[.;,\n]|\bfirst\b|\bthen\b|\band then\b|\bif\b)",
            r"\bdetermine\s+(?:the\s+)?(.+?)\s+of\s+(?:the\s+)?(.+?)(?=$|[.;,\n]|\bfirst\b|\bthen\b|\band then\b|\bif\b)",
        )
        for pattern in measure_patterns:
            match = re.search(pattern, normalized_task)
            if not match:
                continue
            measurement_property = self._normalize_measurement_phrase(match.group(1))
            measurement_target = self._normalize_measurement_target_phrase(
                match.group(2)
            )
            if measurement_property and measurement_target:
                break

        if not (measurement_property and measurement_target):
            return "", "", "", [], []

        branch_pattern = re.compile(
            r"\bif\s+(?:the\s+)?(.+?)\s+is\s+(above|below)\s+"
            r"(-?\d+(?:\.\d+)?)\s*(?:degrees?\s+celsius)?"
            r",?\s*focus on\s+(?:the\s+)?(.+?)(?=$|[.;,\n])"
        )
        for match in branch_pattern.finditer(normalized_task):
            branch_target = self._normalize_task_phrase(
                match.group(4), role="supporting_targets"
            )
            if not branch_target:
                continue
            measurement_branches.append(
                {
                    "condition": self._normalize_measurement_phrase(match.group(1)),
                    "operator": match.group(2),
                    "threshold": float(match.group(3)),
                    "target": branch_target,
                }
            )
            if branch_target not in measurement_branch_targets:
                measurement_branch_targets.append(branch_target)

        focus_targets = [
            self._normalize_task_phrase(match.group(1), role="primary_targets")
            for match in re.finditer(
                r"\bfocus on\s+(.+?)(?=$|[.;,\n]|\bthen\b|\band then\b)",
                normalized_task,
            )
        ]
        for focus_target in focus_targets:
            if not focus_target:
                continue
            if focus_target == measurement_target:
                continue
            if focus_target in measurement_branch_targets:
                continue
            measurement_instrument = focus_target
            break

        using_match = re.search(
            r"\b(?:using|with)\s+(?:the\s+)?(.+?)(?=$|[.;,\n]|\bthen\b|\band then\b)",
            normalized_task,
        )
        if not measurement_instrument and using_match:
            measurement_instrument = self._normalize_task_phrase(
                using_match.group(1), role="supporting_targets"
            )

        return (
            measurement_property,
            measurement_target,
            measurement_instrument,
            measurement_branch_targets[:2],
            measurement_branches[:2],
        )

    def _is_state_change_task(self, task_contract: dict | None = None) -> bool:
        contract = task_contract or self._get_task_contract()
        return bool(contract.get("state_change_task"))

    def _is_artifact_creation_task(self, task_contract: dict | None = None) -> bool:
        contract = task_contract or self._get_task_contract()
        return bool(contract.get("artifact_creation_task"))

    def _is_measurement_task(self, task_contract: dict | None = None) -> bool:
        contract = task_contract or self._get_task_contract()
        return bool(contract.get("measurement_task"))

    def _state_change_target_is_grounded(
        self,
        task_contract: dict | None = None,
        grounded_tokens: set[str] | None = None,
    ) -> bool:
        contract = task_contract or self._get_task_contract()
        if not self._is_state_change_task(contract):
            return False

        grounded_token_set = grounded_tokens or set(
            self._get_observation_grounded_tokens()
        )
        role_token_sets = self._get_task_role_token_sets(contract)
        return self._role_is_grounded(
            grounded_token_set, role_token_sets["target_substances"]
        ) or self._role_is_grounded(
            grounded_token_set, role_token_sets["primary_targets"]
        )

    def _normalize_grounded_substance_label(self, phrase: str) -> str:
        normalized = self._normalize_runtime_text(phrase)
        if not normalized:
            return ""

        normalized = re.split(
            r"\b(?:that|which|where|because|since|until|while|then|and then)\b",
            normalized,
            maxsplit=1,
        )[0]
        normalized = re.sub(r"^(?:a|an|the|some)\s+", "", normalized)
        normalized = normalized.strip(" .,:;()")
        if " and " in normalized:
            return ""

        tokens = self._extract_runtime_tokens(
            normalized,
            stopwords=self._TASK_STOPWORDS | self._ACTION_COMMAND_STOPWORDS,
            limit=4,
        )
        if not tokens:
            return ""
        if set(tokens).issubset(self._CONTAINER_REFERENT_TOKENS):
            return ""
        return " ".join(tokens[:4])

    def _extract_grounded_substance_mentions(self, observation: str) -> list[str]:
        normalized_observation = self._normalize_runtime_text(observation)
        if not normalized_observation:
            return []

        grounded_labels: list[str] = []
        explicit_patterns = (
            r"\b(?:a|an)\s+substance called\s+([a-z0-9][a-z0-9\s-]{0,40}?)(?=$|[).,;\n])",
            r"\bcontaining\s+(?:a|an)\s+substance called\s+([a-z0-9][a-z0-9\s-]{0,40}?)(?=$|[).,;\n])",
        )
        contextual_patterns = (
            r"\bfilled with\s+([a-z0-9][a-z0-9\s-]{0,30}?)(?=$|[).,;\n])",
            r"\bcontaining\s+([a-z0-9][a-z0-9\s-]{0,30}?)(?=$|[).,;\n])",
            r"\bcontains?\s+([a-z0-9][a-z0-9\s-]{0,30}?)(?=$|[).,;\n])",
        )
        for pattern in explicit_patterns:
            for match in re.finditer(pattern, normalized_observation):
                label = self._normalize_grounded_substance_label(match.group(1))
                if label and label not in grounded_labels:
                    grounded_labels.append(label)

        target_token_sets = self._get_task_role_token_sets().get(
            "target_substances", []
        )
        for pattern in contextual_patterns:
            for match in re.finditer(pattern, normalized_observation):
                label = self._normalize_grounded_substance_label(match.group(1))
                if not label or label in grounded_labels:
                    continue
                label_tokens = set(
                    self._extract_runtime_tokens(
                        label,
                        stopwords=self._TASK_STOPWORDS | self._ACTION_COMMAND_STOPWORDS,
                    )
                )
                if not label_tokens:
                    continue
                if not self._matches_any_role(label_tokens, target_token_sets):
                    continue
                grounded_labels.append(label)
        return grounded_labels[:6]

    def _update_grounded_substances(self, observation: str) -> None:
        if not self._is_state_change_task():
            return
        for label in self._extract_grounded_substance_mentions(observation):
            entry = self._grounded_substances.setdefault(
                label,
                {
                    "label": label,
                    "last_seen_timestep": self.num_actions_taken,
                },
            )
            entry["last_seen_timestep"] = self.num_actions_taken

    def _normalize_grounded_artifact_label(self, phrase: str) -> str:
        normalized = self._normalize_runtime_text(phrase)
        if not normalized:
            return ""

        normalized = re.split(
            r"\b(?:containing|filled with|contains?)\b",
            normalized,
            maxsplit=1,
        )[-1]
        normalized = re.split(
            r"\b(?:that|which|where|because|since|until|while|then|and then)\b",
            normalized,
            maxsplit=1,
        )[0]
        normalized = re.sub(r"^(?:a|an|the|some)\s+", "", normalized)
        normalized = normalized.strip(" .,:;()")
        tokens = self._extract_runtime_tokens(
            normalized,
            stopwords=self._TASK_STOPWORDS | self._ACTION_COMMAND_STOPWORDS,
            limit=5,
        )
        return " ".join(tokens[:5])

    def _extract_grounded_artifact_mentions(self, observation: str) -> list[str]:
        task_contract = self._get_task_contract()
        if not self._is_artifact_creation_task(task_contract):
            return []

        artifact_types = task_contract.get("artifact_type", [])
        if not artifact_types:
            return []
        artifact_type = artifact_types[0]
        normalized_observation = self._normalize_runtime_text(observation)
        if not normalized_observation:
            return []

        grounded_labels: list[str] = []
        patterns = (
            rf"\b(?:a|an|the|some)\s+([a-z0-9][a-z0-9\s-]{{0,40}}?\b{re.escape(artifact_type)}\b)(?=$|[).,;\n])",
            rf"\bcontaining\s+([a-z0-9][a-z0-9\s-]{{0,40}}?\b{re.escape(artifact_type)}\b)(?=$|[).,;\n])",
            rf"\byou\s+focus\s+on\s+(?:the\s+)?([a-z0-9][a-z0-9\s-]{{0,40}}?\b{re.escape(artifact_type)}\b)(?=$|[).,;\n])",
        )
        for pattern in patterns:
            for match in re.finditer(pattern, normalized_observation):
                label = self._normalize_grounded_artifact_label(match.group(1))
                if label and label not in grounded_labels:
                    grounded_labels.append(label)
        return grounded_labels[:6]

    def _update_artifact_creation_tracking(self, observation: str) -> None:
        if not self._is_artifact_creation_task():
            return
        for label in self._extract_grounded_artifact_mentions(observation):
            entry = self._grounded_artifacts.setdefault(
                label,
                {
                    "label": label,
                    "last_seen_timestep": self.num_actions_taken,
                },
            )
            entry["last_seen_timestep"] = self.num_actions_taken

    def _get_grounded_artifact_labels(self, *, limit: int = 4) -> list[str]:
        if not self._grounded_artifacts:
            return []
        labels = sorted(
            self._grounded_artifacts.values(),
            key=lambda entry: (-entry["last_seen_timestep"], entry["label"]),
        )
        return [entry["label"] for entry in labels[:limit]]

    def _get_grounded_artifact_token_sets(self) -> list[set[str]]:
        token_sets: list[set[str]] = []
        for label in self._get_grounded_artifact_labels(limit=6):
            tokens = set(
                self._extract_runtime_tokens(
                    label,
                    stopwords=self._TASK_STOPWORDS | self._ACTION_COMMAND_STOPWORDS,
                )
            )
            if tokens:
                token_sets.append(tokens)
        return token_sets

    def _artifact_role_is_grounded(self, role_token_sets: list[set[str]]) -> bool:
        grounded_artifact_token_sets = self._get_grounded_artifact_token_sets()
        return any(
            role_tokens and role_tokens.issubset(artifact_tokens)
            for role_tokens in role_token_sets
            for artifact_tokens in grounded_artifact_token_sets
        )

    def _get_artifact_creation_snapshot(self) -> dict:
        if not self._is_artifact_creation_task():
            return {}
        task_contract = self._get_task_contract()
        grounded_artifacts = self._get_grounded_artifact_labels(limit=4)
        snapshot = {
            "artifact_type": task_contract.get("artifact_type", [])[:1],
            "grounded_artifacts": grounded_artifacts,
            "missing_ingredient_gap": len(grounded_artifacts) <= 1,
        }
        if task_contract.get("artifact_intermediate_targets"):
            snapshot["intermediate_targets"] = task_contract[
                "artifact_intermediate_targets"
            ][:2]
        if task_contract.get("artifact_final_targets"):
            snapshot["final_targets"] = task_contract["artifact_final_targets"][:2]
        return snapshot

    @staticmethod
    def _artifact_creation_has_signal(snapshot: dict) -> bool:
        return any(bool(value) for value in snapshot.values())

    def _get_grounded_substance_labels(self, *, limit: int = 6) -> list[str]:
        labels = sorted(
            self._grounded_substances.values(),
            key=lambda item: (-item["last_seen_timestep"], item["label"]),
        )
        return [item["label"] for item in labels[:limit]]

    def _get_grounded_substance_token_sets(self) -> list[set[str]]:
        token_sets: list[set[str]] = []
        for label in self._get_grounded_substance_labels():
            tokens = set(
                self._extract_runtime_tokens(
                    label,
                    stopwords=self._TASK_STOPWORDS | self._ACTION_COMMAND_STOPWORDS,
                )
            )
            if tokens:
                token_sets.append(tokens)
        return token_sets

    def _extract_action_source_signature(
        self, action: str, *, family: str | None = None
    ) -> str:
        normalized = self._normalize_runtime_text(action)
        if not normalized or " from " not in normalized:
            return ""

        source_segment = normalized.rsplit(" from ", maxsplit=1)[-1]
        source_tokens = self._extract_runtime_tokens(
            source_segment,
            stopwords=self._RUNTIME_TOKEN_STOPWORDS | self._ACTION_COMMAND_STOPWORDS,
            limit=6,
        )
        return " ".join(source_tokens[:6])

    def _update_state_change_search_tracking(
        self, *, action: str | None, observation: str
    ) -> None:
        if not self._is_state_change_task():
            return
        self._update_grounded_substances(observation)
        if not action or action == "None" or self._HARD_FAILURE_RE.search(observation):
            return

        family = self._classify_action_family(action)
        if family not in {"inspect", "device_control"}:
            return

        if not self._is_container_like_action(action, family=family):
            return

        grounded_tokens = set(self._get_observation_grounded_tokens())
        if self._state_change_target_is_grounded(
            self._get_task_contract(), grounded_tokens
        ):
            return

        normalized = self._normalize_runtime_text(action)
        if family != "inspect" and not normalized.startswith("open "):
            return

        referent = self._get_action_referent_signature(action, family=family)
        if referent and referent not in self._exhausted_container_targets:
            self._exhausted_container_targets.append(referent)
            self._exhausted_container_targets = self._exhausted_container_targets[-8:]

    def _infer_source_candidates(self, actions: list[str] | None = None) -> list[str]:
        profiles: dict[str, dict[str, int | bool]] = {}
        for action in actions or self.admissible_actions:
            family = self._classify_action_family(action)
            referent = self._get_action_referent_signature(action, family=family)
            if referent:
                profile = profiles.setdefault(
                    referent,
                    {
                        "inspect": 0,
                        "device_control": 0,
                        "relocation": 0,
                        "from_refs": 0,
                        "container_like": False,
                    },
                )
                if family in {"inspect", "device_control", "relocation"}:
                    profile[family] += 1
                if self._is_container_like_action(action, family=family):
                    profile["container_like"] = True

            source_referent = self._extract_action_source_signature(
                action, family=family
            )
            if source_referent:
                profile = profiles.setdefault(
                    source_referent,
                    {
                        "inspect": 0,
                        "device_control": 0,
                        "relocation": 0,
                        "from_refs": 0,
                        "container_like": False,
                    },
                )
                profile["from_refs"] += 1

        ranked_candidates: list[tuple[int, str]] = []
        exhausted_set = set(self._exhausted_container_targets)
        grounded_substance_tokens = self._get_grounded_substance_token_sets()
        for referent, profile in profiles.items():
            referent_tokens = set(
                self._extract_runtime_tokens(
                    referent,
                    stopwords=self._TASK_STOPWORDS | self._ACTION_COMMAND_STOPWORDS,
                    limit=6,
                )
            )
            if not referent_tokens:
                continue
            if referent_tokens.issubset(self._NON_CANDIDATE_REFERENT_TOKENS):
                continue
            if any(
                tokens and tokens.issubset(referent_tokens)
                for tokens in grounded_substance_tokens
            ):
                continue

            score = 0
            if profile["from_refs"]:
                score += 8 + int(profile["from_refs"]) * 3
            if profile["inspect"]:
                score += 2
            if profile["device_control"]:
                score += 2
            if profile["relocation"]:
                score -= 4
            if profile["container_like"]:
                score -= 5
            if referent in exhausted_set:
                score -= 6

            if score > 0:
                ranked_candidates.append((score, referent))

        ranked_candidates.sort(key=lambda item: (-item[0], len(item[1]), item[1]))
        return [referent for _, referent in ranked_candidates[:4]]

    def _get_substance_search_snapshot(self, actions: list[str] | None = None) -> dict:
        if not self._is_state_change_task():
            return {}
        snapshot = {
            "grounded_substances": self._get_grounded_substance_labels(limit=4),
            "exhausted_containers": self._exhausted_container_targets[-4:],
        }
        source_candidates = self._infer_source_candidates(actions)
        if source_candidates:
            snapshot["source_candidates"] = source_candidates
        return snapshot

    @staticmethod
    def _parse_numeric_measurement(observation: str) -> tuple[float | None, str]:
        normalized = (observation or "").lower()
        if not normalized:
            return None, ""
        match = re.search(
            r"\b(?:temperature of|measures? a temperature of|reading a temperature of|reads?)\s+"
            r"(-?\d+(?:\.\d+)?)\s*(degrees?\s+celsius|celsius|degrees?)?",
            normalized,
        )
        if not match:
            return None, ""
        unit = re.sub(r"\s+", " ", (match.group(2) or "").strip())
        return float(match.group(1)), unit

    def _extract_action_destination_signature(
        self, action: str, *, family: str | None = None
    ) -> str:
        family = family or self._classify_action_family(action)
        if family not in {"relocation", "transfer_or_transform"}:
            return ""

        normalized = self._normalize_runtime_text(action)
        if not normalized:
            return ""

        destination_match = re.search(
            r"\b(?:to|into|in|on)\b\s+([a-z0-9][a-z0-9\s-]{0,60})$",
            normalized,
        )
        if not destination_match:
            return ""
        destination_tokens = self._extract_runtime_tokens(
            destination_match.group(1),
            stopwords=self._RUNTIME_TOKEN_STOPWORDS | self._ACTION_COMMAND_STOPWORDS,
            limit=6,
        )
        return " ".join(destination_tokens[:6])

    def _referent_tokens(self, referent: str) -> set[str]:
        return set(
            self._extract_runtime_tokens(
                referent,
                stopwords=self._TASK_ENTITY_STOPWORDS | self._ACTION_COMMAND_STOPWORDS,
                limit=8,
            )
        )

    def _phrase_to_referent_signature(self, phrase: str) -> str:
        tokens = self._extract_runtime_tokens(
            phrase,
            stopwords=self._TASK_ENTITY_STOPWORDS | self._ACTION_COMMAND_STOPWORDS,
            limit=6,
        )
        return " ".join(tokens[:6])

    def _get_measurement_target_signature(
        self, task_contract: dict | None = None
    ) -> str:
        contract = task_contract or self._get_task_contract()
        measurement_target = contract.get("measurement_target", [])
        if not measurement_target:
            return ""
        return self._phrase_to_referent_signature(measurement_target[0])

    def _referent_is_visible(
        self, referent: str, observation: str | None = None
    ) -> bool:
        referent_tokens = self._referent_tokens(referent)
        if not referent_tokens:
            return False
        observation_tokens = set(
            self._extract_runtime_tokens(
                observation
                if observation is not None
                else (self.percept or {}).get("resulting_observation", ""),
                limit=48,
            )
        )
        return referent_tokens.issubset(observation_tokens)

    def _resolve_enclosing_referent(self, referent: str) -> str:
        current = referent
        seen: set[str] = set()
        while current and current not in seen:
            seen.add(current)
            parent = self._containment_by_object.get(current)
            if not parent or parent == current:
                break
            current = parent
        return current if current != referent else ""

    def _update_containment_tracking(
        self, action: str | None, observation: str
    ) -> None:
        if not action or self._HARD_FAILURE_RE.search(observation):
            return
        family = self._classify_action_family(action)
        primary_object = self._extract_action_primary_object_signature(
            action, family=family
        )
        destination = self._extract_action_destination_signature(action, family=family)
        if primary_object and destination and primary_object != destination:
            self._containment_by_object[primary_object] = destination

    def _extract_measurement_subject_signature(
        self, action: str, *, family: str | None = None
    ) -> str:
        family = family or self._classify_action_family(action)
        normalized = self._normalize_runtime_text(action)
        if family != "tool_application" or not normalized:
            return ""
        if normalized.startswith("use ") and " on " in normalized:
            subject_segment = normalized.split(" on ", maxsplit=1)[-1]
        elif normalized.startswith("measure ") and " with " in normalized:
            subject_segment = normalized.split(" with ", maxsplit=1)[0][
                len("measure ") :
            ]
        else:
            return ""
        subject_tokens = self._extract_runtime_tokens(
            subject_segment,
            stopwords=self._RUNTIME_TOKEN_STOPWORDS | self._ACTION_COMMAND_STOPWORDS,
            limit=6,
        )
        return " ".join(subject_tokens[:6])

    def _measurement_property_requires_event(
        self, task_contract: dict | None = None
    ) -> bool:
        contract = task_contract or self._get_task_contract()
        property_text = self._normalize_runtime_text(
            contract.get("measurement_property", "")
        )
        return "point" in property_text

    def _measurement_property_event_detected(
        self, observation: str, task_contract: dict | None = None
    ) -> bool:
        contract = task_contract or self._get_task_contract()
        if not self._measurement_property_requires_event(contract):
            return True

        observation_lower = self._normalize_runtime_text(observation)
        property_text = self._normalize_runtime_text(
            contract.get("measurement_property", "")
        )
        event_patterns = [r"\bchanges? state\b", r"\btransforms?\b"]
        if "melt" in property_text:
            event_patterns.extend((r"\bmelt(?:s|ed|ing)?\b", r"\bliquid\b"))
        if "boil" in property_text:
            event_patterns.extend((r"\bboil(?:s|ed|ing)?\b", r"\bsteam\b"))
        if "freeze" in property_text:
            event_patterns.extend((r"\bfreez(?:e|es|ing|ed)\b", r"\bice\b"))
        return any(re.search(pattern, observation_lower) for pattern in event_patterns)

    def _update_measurement_tracking(
        self, *, action: str | None, observation: str
    ) -> None:
        self._update_containment_tracking(action, observation)
        if not self._is_measurement_task() or not action or action == "None":
            return

        family = self._classify_action_family(action)
        measurement_value, measurement_unit = self._parse_numeric_measurement(
            observation
        )
        if measurement_value is None or family != "tool_application":
            return

        task_contract = self._get_task_contract()
        role_token_sets = self._get_task_role_token_sets(task_contract)
        subject_signature = self._extract_measurement_subject_signature(
            action, family=family
        )
        subject_tokens = self._referent_tokens(subject_signature)
        direct_measurement = bool(
            subject_tokens
            and self._matches_any_role(
                subject_tokens, role_token_sets["measurement_target"]
            )
        )
        active_enclosure = self._resolve_enclosing_referent(
            self._get_measurement_target_signature(task_contract)
        )
        proxy_measurement = bool(
            subject_signature
            and not direct_measurement
            and active_enclosure
            and subject_signature == active_enclosure
        )
        event_confirmed = self._measurement_property_event_detected(
            observation, task_contract
        )
        measurement_entry = {
            "value": measurement_value,
            "unit": measurement_unit,
            "subject": subject_signature,
            "direct": direct_measurement,
            "proxy": proxy_measurement,
            "event_confirmed": event_confirmed,
            "timestep": self.num_actions_taken,
        }
        self._measurement_observations.append(measurement_entry)
        self._measurement_observations = self._measurement_observations[-8:]

        if not direct_measurement:
            return
        if self._selected_measurement_branch_target:
            return

        for branch in task_contract.get("measurement_branches", []):
            operator = branch["operator"]
            threshold = branch["threshold"]
            if (
                self._measurement_property_requires_event(task_contract)
                and not event_confirmed
            ):
                continue
            if operator == "above" and measurement_value > threshold:
                self._selected_measurement_branch_target = branch["target"]
                break
            if operator == "below" and measurement_value < threshold:
                self._selected_measurement_branch_target = branch["target"]
                break

    def _get_latest_measurement(self, *, direct: bool | None = None) -> dict | None:
        for entry in reversed(self._measurement_observations):
            if direct is None or entry["direct"] is direct:
                return entry
        return None

    def _get_measurement_tracking_snapshot(self) -> dict:
        if not self._is_measurement_task():
            return {}
        task_contract = self._get_task_contract()
        snapshot = {
            "measurement_target": task_contract.get("measurement_target", [])[:1],
            "measurement_instrument": task_contract.get("measurement_instrument", [])[
                :1
            ],
            "branch_target": (
                [self._selected_measurement_branch_target]
                if self._selected_measurement_branch_target
                else []
            ),
        }
        latest_direct = self._get_latest_measurement(direct=True)
        latest_proxy = self._get_latest_measurement(direct=False)
        if latest_direct:
            snapshot["latest_direct_measurement"] = {
                "value": latest_direct["value"],
                "unit": latest_direct["unit"],
                "subject": latest_direct["subject"],
            }
        if latest_proxy:
            snapshot["latest_proxy_measurement"] = {
                "value": latest_proxy["value"],
                "unit": latest_proxy["unit"],
                "subject": latest_proxy["subject"],
            }
        measurement_target = self._get_measurement_target_signature(task_contract)
        active_enclosure = self._resolve_enclosing_referent(measurement_target)
        if active_enclosure and not self._referent_is_visible(measurement_target):
            snapshot["active_enclosure"] = [active_enclosure]
        if task_contract.get("measurement_branch_targets"):
            snapshot["branch_ready"] = bool(self._selected_measurement_branch_target)
        return snapshot

    @staticmethod
    def _measurement_tracking_has_signal(snapshot: dict) -> bool:
        return any(bool(value) for value in snapshot.values())

    @staticmethod
    def _substance_search_has_signal(snapshot: dict) -> bool:
        return any(bool(value) for value in snapshot.values())

    def _should_probe_sources(self, task_contract: dict | None = None) -> bool:
        contract = task_contract or self._get_task_contract()
        if not self._is_state_change_task(contract):
            return False
        grounded_tokens = set(self._get_observation_grounded_tokens())
        if self._state_change_target_is_grounded(contract, grounded_tokens):
            return False
        snapshot = self._get_substance_search_snapshot()
        return len(snapshot.get("exhausted_containers", [])) >= 2 and bool(
            snapshot.get("source_candidates")
        )

    def _canonicalize_unsupported_substance_action(
        self, suggested_action: str, admissible_commands: list[str]
    ) -> str | None:
        if not self._is_state_change_task():
            return None

        family = self._classify_action_family(suggested_action)
        if family not in {
            "focus",
            "inspect",
            "tool_application",
            "transfer_or_transform",
        }:
            return None

        suggested_tokens = set(
            self._extract_action_content_tokens(suggested_action, family=family)
        )
        if not suggested_tokens or not (
            suggested_tokens & self._CONTAINER_REFERENT_TOKENS
        ):
            return None

        grounded_tokens = set(self._get_observation_grounded_tokens())
        grounded_tokens.update(*self._get_grounded_substance_token_sets())
        unsupported_tokens = {
            token
            for token in suggested_tokens
            if token not in grounded_tokens
            and token not in self._CONTAINER_REFERENT_TOKENS
        }
        if not unsupported_tokens:
            return None

        reduced_tokens = suggested_tokens - unsupported_tokens
        if not reduced_tokens:
            return None

        candidates: list[tuple[int, int, str]] = []
        for command in admissible_commands:
            if self._classify_action_family(command) != family:
                continue
            command_tokens = set(
                self._extract_action_content_tokens(command, family=family)
            )
            if not reduced_tokens.issubset(command_tokens):
                continue
            if not (command_tokens & self._CONTAINER_REFERENT_TOKENS):
                continue
            extra = len(command_tokens - reduced_tokens)
            candidates.append((extra, len(command_tokens), command))

        if not candidates:
            return None

        candidates.sort(key=lambda item: (item[0], item[1], len(item[2]), item[2]))
        if len(candidates) > 1 and candidates[0][:2] == candidates[1][:2]:
            return None
        return candidates[0][2]

    def _canonicalize_room_transition_action(
        self, suggested_action: str, admissible_commands: list[str]
    ) -> str | None:
        family = self._classify_action_family(suggested_action)
        normalized = self._normalize_runtime_text(suggested_action)
        if family != "relocation" or not normalized.startswith(("go to ", "enter ")):
            return None

        room_signature = self._extract_action_primary_object_signature(
            suggested_action, family=family
        )
        room_tokens = set(
            self._extract_runtime_tokens(
                room_signature,
                stopwords=self._TASK_STOPWORDS | self._ACTION_COMMAND_STOPWORDS,
                limit=4,
            )
        )
        if not room_tokens:
            return None

        candidates: list[tuple[int, int, str]] = []
        for command in admissible_commands:
            if self._classify_action_family(command) != "device_control":
                continue
            command_tokens = set(
                self._extract_action_content_tokens(command, family="device_control")
            )
            if "door" not in command_tokens or not room_tokens.issubset(command_tokens):
                continue
            bonus = (
                4 if self._normalize_runtime_text(command).startswith("open ") else 0
            )
            extra = len(command_tokens - room_tokens)
            candidates.append((bonus - extra, extra, command))

        if not candidates:
            return None

        candidates.sort(key=lambda item: (-item[0], item[1], len(item[2]), item[2]))
        if len(candidates) > 1 and candidates[0][:2] == candidates[1][:2]:
            return None
        return candidates[0][2]

    def _extract_stage_labels(self, text: str) -> list[str]:
        normalized = self._normalize_runtime_text(text)
        normalized = re.sub(r"\b(?:self watering )?flower pot\b", "pot", normalized)
        labels: list[str] = []
        for label, patterns in self._LIFECYCLE_STAGE_PATTERNS:
            if any(re.search(pattern, normalized) for pattern in patterns):
                labels.append(label)
        return labels

    def _get_stage_rank(self, label: str) -> int:
        return self._LIFECYCLE_STAGE_ORDER.get(label, 999)

    def _merge_stage_labels(self, labels: list[str]) -> list[str]:
        ordered: list[str] = []
        for label in sorted(set(labels), key=self._get_stage_rank):
            if label not in ordered:
                ordered.append(label)
        return ordered

    def _is_container_like_action(
        self, action: str, *, family: str | None = None
    ) -> bool:
        content_tokens = set(self._extract_action_content_tokens(action, family=family))
        return bool(content_tokens & self._CONTAINER_REFERENT_TOKENS)

    def _build_stage_evidence_keys(
        self, action: str, observation: str, *, family: str | None = None
    ) -> list[str]:
        keys: list[str] = []
        referent_signature = self._get_action_referent_signature(action, family=family)
        if referent_signature:
            keys.append(referent_signature)

        observation_lower = self._normalize_runtime_text(observation)
        pot_match = re.search(r"\b(?:flower pot|pot)\s+(\d+)\b", observation_lower)
        if pot_match:
            content_tokens = [
                token
                for token in self._extract_action_content_tokens(action, family=family)
                if token not in self._STAGE_EVIDENCE_FILTER_TOKENS
                and not token.isdigit()
            ]
            if content_tokens:
                observation_key = " ".join(
                    content_tokens[:3] + ["flower", "pot", pot_match.group(1)]
                )
                if observation_key not in keys:
                    keys.append(observation_key)

        return keys

    def _record_stage_evidence(
        self, referent_keys: list[str], stage_labels: list[str]
    ) -> None:
        merged_labels = self._merge_stage_labels(stage_labels)
        if not merged_labels:
            return

        for key in referent_keys:
            if not key:
                continue
            existing = self._stage_evidence_by_referent.get(key, [])
            self._stage_evidence_by_referent[key] = self._merge_stage_labels(
                existing + merged_labels
            )

        for label in merged_labels:
            if label not in self._observed_stage_labels:
                self._observed_stage_labels.append(label)
        self._observed_stage_labels = self._observed_stage_labels[-8:]

    def _get_stage_evidence_for_referent(self, referent_signature: str) -> list[str]:
        if not referent_signature:
            return []

        direct = self._stage_evidence_by_referent.get(referent_signature, [])
        if direct:
            return direct

        referent_tokens = set(self._extract_runtime_tokens(referent_signature, limit=8))
        referent_digits = {token for token in referent_tokens if token.isdigit()}
        inferred: list[str] = []
        for key, labels in self._stage_evidence_by_referent.items():
            key_tokens = set(self._extract_runtime_tokens(key, limit=8))
            if referent_digits and not (referent_digits & key_tokens):
                continue
            overlap = len(
                (referent_tokens & key_tokens) - self._CONTAINER_REFERENT_TOKENS
            )
            if overlap >= 2:
                inferred.extend(labels)
        return self._merge_stage_labels(inferred)

    def _update_lifecycle_stage_state(
        self, *, action: str | None, observation: str
    ) -> None:
        if not action or not self._is_lifecycle_task():
            return

        family = self._classify_action_family(action)
        action_stage_labels = self._extract_stage_labels(action)
        observation_stage_labels = self._extract_stage_labels(observation)
        stage_labels = list(observation_stage_labels)
        if action_stage_labels and (
            family == "inspect"
            or not self._is_container_like_action(action, family=family)
        ):
            stage_labels.extend(action_stage_labels)

        merged_labels = self._merge_stage_labels(stage_labels)
        if not merged_labels:
            return

        referent_keys = self._build_stage_evidence_keys(
            action, observation, family=family
        )
        self._record_stage_evidence(referent_keys, merged_labels)

    def _get_focus_stage_labels(self, action: str, observation: str) -> list[str]:
        family = self._classify_action_family(action)
        if family != "focus":
            return []

        referent_signature = self._get_action_referent_signature(action, family=family)
        evidence_labels = self._get_stage_evidence_for_referent(referent_signature)
        observation_labels = self._extract_stage_labels(observation)
        action_labels = self._extract_stage_labels(action)
        if observation_labels:
            return self._merge_stage_labels(observation_labels)
        if evidence_labels:
            return evidence_labels
        if self._is_container_like_action(action, family=family):
            return []
        return self._merge_stage_labels(action_labels)

    def _get_next_expected_stage_label(self) -> str | None:
        focused_ranks = {
            self._get_stage_rank(label) for label in self._focused_stage_labels
        }
        if focused_ranks:
            highest_focused_rank = max(focused_ranks)
        else:
            highest_focused_rank = -1

        for label in self._observed_stage_labels:
            rank = self._get_stage_rank(label)
            if rank > highest_focused_rank and label not in self._focused_stage_labels:
                return label
        return None

    @staticmethod
    def _parse_ambiguity_options(observation: str) -> dict[str, str]:
        options: dict[str, str] = {}
        for line in (observation or "").splitlines():
            match = re.match(r"\s*(\d+):\s*(.+?)\s*$", line)
            if match:
                options[match.group(1)] = match.group(2)
        return options

    def _resolve_reasoning_action(
        self, action: str | None, previous_observation: str
    ) -> tuple[str | None, dict | None]:
        if not action:
            return action, None

        normalized_action = self._normalize_runtime_text(action)
        if not re.fullmatch(r"\d+", normalized_action):
            return action, None
        if "ambiguous request" not in self._normalize_runtime_text(
            previous_observation
        ):
            return action, None

        options = self._parse_ambiguity_options(previous_observation)
        resolved_action = options.get(normalized_action)
        if not resolved_action:
            return action, None

        return resolved_action, {
            "status": "resolved",
            "choice": normalized_action,
            "resolved_action": resolved_action,
        }

    def _normalize_task_phrase(self, phrase: str, *, role: str) -> str:
        normalized = self._normalize_runtime_text(phrase)
        if not normalized:
            return ""

        normalized = re.split(
            r"\b(?:that|which|who|where|because|since|until|while|so that)\b",
            normalized,
            maxsplit=1,
        )[0]
        if role == "primary_targets":
            normalized = re.split(
                r"\b(?:by|using|with|after|before|then|and then)\b",
                normalized,
                maxsplit=1,
            )[0]
        elif role == "supporting_targets":
            normalized = re.split(
                r"\b(?:after|before|then|and then)\b",
                normalized,
                maxsplit=1,
            )[0]
        elif role == "required_relations":
            normalized = re.split(
                r"\b(?:using|with|after|before|then|and then)\b",
                normalized,
                maxsplit=1,
            )[0]

        normalized = re.sub(r"^(?:a|an|the)\s+", "", normalized)
        normalized = normalized.strip(" .,:;")
        phrase_tokens = self._extract_runtime_tokens(
            normalized,
            stopwords=self._TASK_ENTITY_STOPWORDS | self._ACTION_COMMAND_STOPWORDS,
            limit=6,
        )
        return " ".join(phrase_tokens[:4])

    def _extract_candidate_classes(self, task: str) -> list[str]:
        normalized_task = self._normalize_runtime_text(task).replace("a(n)", "an")
        candidate_classes: list[str] = []
        patterns = (
            r"\b(?:find|locate|identify|discover)\s+(?:an?\s+|the\s+)?(.+?)(?=$|[.;,\n]|\bfirst\b|\bthen\b|\band then\b)",
        )
        for pattern in patterns:
            for match in re.finditer(pattern, normalized_task):
                phrase = self._normalize_task_phrase(
                    match.group(1), role="primary_targets"
                )
                if phrase and phrase not in candidate_classes:
                    candidate_classes.append(phrase)
        return candidate_classes[:3]

    def _extract_destination_roles(self, task: str) -> tuple[str, str]:
        normalized_task = self._normalize_runtime_text(task).replace("a(n)", "an")
        patterns = (
            r"\b(?:move|bring|carry)\s+.+?\s+\bto\b\s+(?:the\s+)?(.+?)(?:\s+\bin\b\s+(?:the\s+)?(.+?))?(?=$|[.;,\n]|\bthen\b|\band then\b)",
            r"\b(?:put|place)\s+.+?\s+\b(?:in|into|on)\b\s+(?:the\s+)?(.+?)(?:\s+\bin\b\s+(?:the\s+)?(.+?))?(?=$|[.;,\n]|\bthen\b|\band then\b)",
        )
        for pattern in patterns:
            match = re.search(pattern, normalized_task)
            if not match:
                continue
            container = self._normalize_task_phrase(
                match.group(1), role="supporting_targets"
            )
            room = self._normalize_task_phrase(
                match.group(2) if match.lastindex and match.group(2) else "",
                role="supporting_targets",
            )
            return container, room
        return "", ""

    def _extract_task_role_phrases(self, task: str) -> dict[str, list[str]]:
        normalized_task = self._normalize_runtime_text(task)
        role_phrases = {
            "primary_targets": [],
            "supporting_targets": [],
            "required_relations": [],
        }
        for role, patterns in self._TASK_ROLE_PATTERNS.items():
            for pattern in patterns:
                for match in re.finditer(pattern, normalized_task):
                    phrase = self._normalize_task_phrase(match.group(1), role=role)
                    if phrase and phrase not in role_phrases[role]:
                        role_phrases[role].append(phrase)
                    if len(role_phrases[role]) >= 3:
                        break
                if len(role_phrases[role]) >= 3:
                    break
        return role_phrases

    def _get_task_role_token_sets(self, task_contract: dict | None = None) -> dict:
        contract = task_contract or self._get_task_contract()
        role_token_sets: dict[str, list[set[str]]] = {}
        for role in (
            "primary_targets",
            "supporting_targets",
            "required_relations",
            "candidate_classes",
            "target_substances",
            "artifact_type",
            "artifact_intermediate_targets",
            "artifact_final_targets",
            "measurement_target",
            "measurement_instrument",
            "measurement_branch_targets",
            "destination_container",
            "destination_room",
        ):
            role_token_sets[role] = []
            for phrase in contract.get(role, []):
                phrase_tokens = set(
                    self._extract_runtime_tokens(
                        phrase,
                        stopwords=self._TASK_ENTITY_STOPWORDS
                        | self._ACTION_COMMAND_STOPWORDS,
                    )
                )
                if phrase_tokens:
                    role_token_sets[role].append(phrase_tokens)
        return role_token_sets

    @staticmethod
    def _best_role_overlap(
        action_tokens: set[str], role_token_sets: list[set[str]]
    ) -> tuple[int, int]:
        best_hits = 0
        best_size = 0
        for token_set in role_token_sets:
            hits = len(action_tokens & token_set)
            if hits > best_hits or (hits == best_hits and len(token_set) > best_size):
                best_hits = hits
                best_size = len(token_set)
        return best_hits, best_size

    @staticmethod
    def _has_full_role_match(
        action_tokens: set[str], role_token_sets: list[set[str]]
    ) -> bool:
        return any(
            token_set and token_set.issubset(action_tokens)
            for token_set in role_token_sets
        )

    @staticmethod
    def _role_is_grounded(
        grounded_tokens: set[str], role_token_sets: list[set[str]]
    ) -> bool:
        return any(
            token_set and token_set.issubset(grounded_tokens)
            for token_set in role_token_sets
        )

    def _role_focus_completed(self, role_token_sets: list[set[str]]) -> bool:
        for completed_target in self._completed_focus_targets:
            completed_tokens = set(
                self._extract_runtime_tokens(
                    completed_target,
                    stopwords=self._TASK_ENTITY_STOPWORDS
                    | self._ACTION_COMMAND_STOPWORDS,
                )
            )
            if not completed_tokens:
                continue
            for role_tokens in role_token_sets:
                if role_tokens and role_tokens.issubset(completed_tokens):
                    return True
        return False

    def _count_recent_referent_repeats(
        self, *, family: str, referent_signature: str
    ) -> int:
        if not referent_signature:
            return 0
        repeats = 0
        for test in self.recent_hypothesis_tests:
            if test["family"] != family:
                continue
            prior_referent = self._get_action_referent_signature(
                test["action"], family=family
            )
            if prior_referent == referent_signature:
                repeats += 1
        return repeats

    def _extract_current_location_tokens(self, observation: str) -> list[str]:
        patterns = (
            r"\b(?:room|location)\s+is\s+called\s+(?:the\s+)?([a-z0-9][a-z0-9\s-]{0,40}?)(?:[.,\n]|$)",
            r"\byou\s+are\s+in\s+(?:the\s+)?([a-z0-9][a-z0-9\s-]{0,40}?)(?:[.,\n]|$)",
        )
        location_tokens: list[str] = []
        for pattern in patterns:
            match = re.search(pattern, (observation or "").lower())
            if not match:
                continue
            for token in self._extract_runtime_tokens(match.group(1), limit=4):
                if token not in location_tokens:
                    location_tokens.append(token)
            if location_tokens:
                break
        return location_tokens

    def _get_observation_grounded_tokens(self) -> list[str]:
        tokens: list[str] = []
        percept = getattr(self, "percept", {}) or {}
        observation = percept.get("resulting_observation", "")
        observation_tokens = self._extract_runtime_tokens(observation, limit=36)
        target_entities = set(self._get_task_contract().get("target_entities", []))

        for token in self._extract_current_location_tokens(observation):
            if token not in tokens:
                tokens.append(token)

        for label in self._observed_stage_labels[-4:]:
            if label not in tokens:
                tokens.append(label)

        for label in self._get_grounded_substance_labels(limit=4):
            for token in self._extract_runtime_tokens(
                label,
                stopwords=self._TASK_STOPWORDS | self._ACTION_COMMAND_STOPWORDS,
                limit=4,
            ):
                if token not in tokens:
                    tokens.append(token)

        for label in self._get_grounded_artifact_labels(limit=4):
            for token in self._extract_runtime_tokens(
                label,
                stopwords=self._TASK_STOPWORDS | self._ACTION_COMMAND_STOPWORDS,
                limit=4,
            ):
                if token not in tokens:
                    tokens.append(token)

        for token in observation_tokens:
            if token not in target_entities or token in tokens:
                continue
            tokens.append(token)

        for test in reversed(self.recent_hypothesis_tests):
            if test["outcome"] not in {"observable_change", "evidence"}:
                continue
            for token in self._extract_runtime_tokens(test["action"], limit=6):
                if token not in tokens:
                    tokens.append(token)
            if len(tokens) >= 12:
                break

        for token in observation_tokens:
            if token not in tokens:
                tokens.append(token)
            if len(tokens) >= 12:
                break

        return tokens[:12]

    def _is_candidate_search_task(self, task_contract: dict | None = None) -> bool:
        contract = task_contract or self._get_task_contract()
        return bool(contract.get("search_mode") or contract.get("candidate_classes"))

    def _matches_any_role(
        self, token_set: set[str], role_token_sets: list[set[str]]
    ) -> bool:
        return any(tokens and tokens.issubset(token_set) for tokens in role_token_sets)

    def _signature_matches_role(
        self, signature: str, role_token_sets: list[set[str]]
    ) -> bool:
        signature_tokens = self._referent_tokens(signature)
        if not signature_tokens:
            return False
        return self._matches_any_role(signature_tokens, role_token_sets)

    def _is_support_referent(
        self, referent: str, task_contract: dict | None = None
    ) -> bool:
        if not referent:
            return False
        contract = task_contract or self._get_task_contract()
        role_token_sets = self._get_task_role_token_sets(contract)
        referent_tokens = set(
            self._extract_runtime_tokens(
                referent,
                stopwords=self._TASK_ENTITY_STOPWORDS | self._ACTION_COMMAND_STOPWORDS,
            )
        )
        if not referent_tokens:
            return False
        return any(
            self._matches_any_role(referent_tokens, role_token_sets[role])
            for role in (
                "supporting_targets",
                "destination_container",
                "destination_room",
            )
        )

    def _extract_action_primary_object_signature(
        self, action: str, *, family: str | None = None
    ) -> str:
        family = family or self._classify_action_family(action)
        normalized = self._normalize_runtime_text(action)
        if not normalized:
            return ""

        content = normalized
        for prefix in (
            "focus on ",
            "look at ",
            "look in ",
            "inspect ",
            "examine ",
            "check ",
            "read ",
            "move ",
            "bring ",
            "carry ",
            "pick up ",
            "take ",
            "grab ",
            "put down ",
            "drop ",
            "pour ",
            "fill ",
            "mix ",
            "insert ",
            "place ",
            "open ",
            "close ",
            "activate ",
            "deactivate ",
            "turn on ",
            "turn off ",
            "use ",
            "heat ",
            "cool ",
            "boil ",
            "cook ",
            "connect ",
            "disconnect ",
            "go to ",
            "enter ",
        ):
            if normalized.startswith(prefix):
                content = normalized[len(prefix) :]
                break

        primary_segment = re.split(
            r"\b(?:to|into|in|on|with|from)\b", content, maxsplit=1
        )[0]
        primary_tokens = self._extract_runtime_tokens(
            primary_segment,
            stopwords=self._RUNTIME_TOKEN_STOPWORDS | self._ACTION_COMMAND_STOPWORDS,
            limit=6,
        )
        return " ".join(primary_tokens[:6])

    def _get_candidate_action_target(
        self,
        action: str,
        *,
        family: str | None = None,
        task_contract: dict | None = None,
    ) -> str:
        contract = task_contract or self._get_task_contract()
        if not self._is_candidate_search_task(contract):
            return ""

        family = family or self._classify_action_family(action)
        if family not in {"focus", "inspect", "relocation", "transfer_or_transform"}:
            return ""

        candidate = self._extract_action_primary_object_signature(action, family=family)
        if not candidate:
            candidate = self._get_action_referent_signature(action, family=family)
        if not candidate:
            return ""

        candidate_tokens = set(self._extract_runtime_tokens(candidate, limit=6))
        if not candidate_tokens or candidate_tokens.issubset(
            self._NON_CANDIDATE_REFERENT_TOKENS
        ):
            return ""
        if self._is_support_referent(candidate, contract):
            return ""
        return candidate

    def _get_candidate_state(self, candidate: str) -> dict:
        state = self._candidate_states.get(candidate)
        if state is not None:
            return state
        state = {
            "focused": False,
            "relocated": False,
            "support_confirmed": False,
            "stalled_confirmations": 0,
            "status": "active",
        }
        self._candidate_states[candidate] = state
        return state

    @staticmethod
    def _normalize_observation_signature(observation: str) -> str:
        return re.sub(r"\s+", " ", (observation or "").strip().lower())

    def _get_action_observation_signature(
        self, action: str | None, observation: str, *, family: str | None = None
    ) -> tuple[str, str, str] | None:
        if not action:
            return None
        family = family or self._classify_action_family(action)
        referent = self._get_action_referent_signature(action, family=family)
        observation_signature = self._normalize_observation_signature(observation)
        if not referent or not observation_signature:
            return None
        return family, referent, observation_signature

    def _is_repeated_action_observation(
        self, action: str | None, observation: str, *, family: str | None = None
    ) -> bool:
        signature = self._get_action_observation_signature(
            action, observation, family=family
        )
        if signature is None:
            return False
        return self._action_observation_signatures.get(signature, 0) > 0

    def _record_action_observation_signature(
        self, action: str | None, observation: str, *, family: str | None = None
    ) -> None:
        signature = self._get_action_observation_signature(
            action, observation, family=family
        )
        if signature is None:
            return
        self._action_observation_signatures[signature] = (
            self._action_observation_signatures.get(signature, 0) + 1
        )

    def _observation_confirms_candidate_destination(
        self, candidate: str, observation: str, task_contract: dict | None = None
    ) -> bool:
        if not candidate:
            return False
        contract = task_contract or self._get_task_contract()
        observation_tokens = set(self._extract_runtime_tokens(observation, limit=32))
        candidate_tokens = set(self._extract_runtime_tokens(candidate, limit=6))
        if not candidate_tokens or not candidate_tokens.issubset(observation_tokens):
            return False

        role_token_sets = self._get_task_role_token_sets(contract)
        if self._matches_any_role(
            observation_tokens, role_token_sets["destination_container"]
        ):
            return True
        current_location_tokens = set(
            self._extract_current_location_tokens(observation)
        )
        return self._matches_any_role(
            current_location_tokens, role_token_sets["destination_room"]
        )

    def _update_candidate_tracking(
        self,
        *,
        executed_action: str | None,
        observation: str,
        previous_observation: str,
    ) -> None:
        if not executed_action or not self._is_candidate_search_task():
            return

        family = self._classify_action_family(executed_action)
        referent = self._get_action_referent_signature(executed_action, family=family)
        candidate = self._get_candidate_action_target(executed_action, family=family)
        normalized_observation = self._normalize_observation_signature(observation)
        previous_signature = self._normalize_observation_signature(previous_observation)
        observation_stalled = normalized_observation == previous_signature

        if family == "focus" and candidate:
            state = self._get_candidate_state(candidate)
            state["focused"] = True
            if candidate not in self._rejected_candidates:
                self._active_candidate = candidate

        active_candidate = self._active_candidate or candidate
        if not active_candidate:
            return

        active_state = self._get_candidate_state(active_candidate)
        active_candidate_tokens = set(
            self._extract_runtime_tokens(active_candidate, limit=6)
        )
        action_tokens = set(self._extract_runtime_tokens(executed_action, limit=12))
        if (
            family in {"relocation", "transfer_or_transform"}
            and active_candidate_tokens
            and active_candidate_tokens.issubset(action_tokens)
        ):
            if self._observation_confirms_candidate_destination(
                active_candidate, observation
            ):
                active_state["relocated"] = True

        if self._observation_confirms_candidate_destination(
            active_candidate, observation
        ):
            active_state["support_confirmed"] = True

        repeated_confirmation = self._is_repeated_action_observation(
            executed_action, observation, family=family
        )
        support_referent = self._is_support_referent(referent)
        if family in {"focus", "inspect"} and (
            referent == active_candidate or support_referent
        ):
            if active_state["support_confirmed"] and (
                repeated_confirmation or observation_stalled
            ):
                active_state["stalled_confirmations"] += 1

        if (
            active_state["focused"]
            and active_state["support_confirmed"]
            and self.task_status != "COMPLETED"
            and active_state["stalled_confirmations"] >= 2
        ):
            active_state["status"] = "rejected"
            if active_candidate not in self._rejected_candidates:
                self._rejected_candidates.append(active_candidate)
                self._rejected_candidates = self._rejected_candidates[-4:]
            if self._active_candidate == active_candidate:
                self._active_candidate = None

    def _get_candidate_tracking_snapshot(self) -> dict:
        task_contract = self._get_task_contract()
        snapshot = {}
        if task_contract.get("candidate_classes"):
            snapshot["candidate_classes"] = task_contract["candidate_classes"][:2]
        if task_contract.get("destination_container"):
            snapshot["destination_container"] = task_contract["destination_container"][
                :1
            ]
        if task_contract.get("destination_room"):
            snapshot["destination_room"] = task_contract["destination_room"][:1]
        if self._active_candidate:
            snapshot["active_candidate"] = self._active_candidate
        if self._rejected_candidates:
            snapshot["rejected_candidates"] = self._rejected_candidates[-3:]
        return snapshot

    @staticmethod
    def _candidate_tracking_has_signal(snapshot: dict) -> bool:
        return any(bool(value) for value in snapshot.values())

    def _extract_action_content_tokens(
        self, action: str, family: str | None = None
    ) -> list[str]:
        stopwords = self._RUNTIME_TOKEN_STOPWORDS | self._ACTION_COMMAND_STOPWORDS
        content_tokens = self._extract_runtime_tokens(action, stopwords=stopwords)
        if family == "device_control":
            # Keep door tokens for hierarchical navigation and opening tasks.
            return self._extract_runtime_tokens(
                action,
                stopwords=(stopwords - {"door", "doors"}),
            )
        return content_tokens

    def _get_action_referent_signature(
        self, action: str, *, family: str | None = None
    ) -> str:
        referent_tokens = [
            token
            for token in self._extract_action_content_tokens(action, family=family)
            if token not in self._REFERENT_SIGNATURE_STOPWORDS
        ]
        if not referent_tokens:
            referent_tokens = self._extract_action_content_tokens(action, family=family)
        return " ".join(referent_tokens[:6])

    def _record_referent_resolution(
        self, *, suggested_action: str, canonical_action: str
    ) -> dict | None:
        suggested_family = self._classify_action_family(suggested_action)
        canonical_family = self._classify_action_family(canonical_action)
        if suggested_family != canonical_family:
            return None

        requested_target = self._get_action_referent_signature(
            suggested_action, family=suggested_family
        )
        resolved_target = self._get_action_referent_signature(
            canonical_action, family=canonical_family
        )
        if (
            not requested_target
            or not resolved_target
            or requested_target == resolved_target
        ):
            return None

        resolution = {
            "status": "ambiguous",
            "requested_target": requested_target,
            "resolved_target": resolved_target,
        }
        if resolution not in self._referent_resolution_events:
            self._referent_resolution_events.append(resolution)
            self._referent_resolution_events = self._referent_resolution_events[-4:]
        return resolution

    def _update_ordered_target_progress(
        self, *, executed_action: str | None, observation: str
    ) -> None:
        if executed_action is None:
            return

        family = self._classify_action_family(executed_action)
        if family != "focus":
            return

        normalized_observation = self._normalize_runtime_text(observation)
        if not normalized_observation.startswith("you focus on"):
            return

        referent = self._get_action_referent_signature(executed_action, family=family)
        if self._is_lifecycle_task():
            stage_labels = self._get_focus_stage_labels(executed_action, observation)
            if not stage_labels:
                return
            if referent and referent not in self._completed_focus_targets:
                self._completed_focus_targets.append(referent)
                self._completed_focus_targets = self._completed_focus_targets[-6:]
            for label in stage_labels:
                if label not in self._focused_stage_labels:
                    self._focused_stage_labels.append(label)
            self._focused_stage_labels = self._focused_stage_labels[-6:]
            return

        if self._is_candidate_search_task() and self._is_support_referent(referent):
            return

        if self._is_measurement_task():
            task_contract = self._get_task_contract()
            role_token_sets = self._get_task_role_token_sets(task_contract)
            referent_tokens = self._referent_tokens(referent)
            if (
                referent_tokens
                and self._matches_any_role(
                    referent_tokens, role_token_sets["measurement_branch_targets"]
                )
                and referent != self._selected_measurement_branch_target
            ):
                return

        if referent and referent not in self._completed_focus_targets:
            self._completed_focus_targets.append(referent)
            self._completed_focus_targets = self._completed_focus_targets[-6:]

    def _get_ordered_target_snapshot(self) -> dict:
        snapshot = {
            "completed_focus_targets": self._completed_focus_targets[-4:],
            "ambiguous_focus_targets": self._referent_resolution_events[-3:],
        }
        if self._is_lifecycle_task():
            pending_stage_candidates = [
                label
                for label in self._observed_stage_labels
                if label not in self._focused_stage_labels
            ]
            snapshot["focused_stage_labels"] = self._focused_stage_labels[-4:]
            snapshot["observed_stage_labels"] = self._observed_stage_labels[-6:]
            snapshot["pending_stage_candidates"] = pending_stage_candidates[:4]
        return snapshot

    @staticmethod
    def _ordered_progress_has_signal(snapshot: dict) -> bool:
        return any(bool(value) for value in snapshot.values())

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

    def _record_invalid_exact_action(self, action: str | None) -> None:
        normalized = self._normalize_runtime_text(action or "")
        if not normalized:
            return
        self._invalid_exact_actions[normalized] = (
            self._invalid_exact_actions.get(normalized, 0) + 1
        )

    @staticmethod
    def _get_shortlist_family_quotas(current_phase: str) -> dict[str, int]:
        quotas_by_phase = {
            "locate_base_artifact": {
                "inspect": 4,
                "device_control": 3,
                "relocation": 2,
                "focus": 1,
                "transfer_or_transform": 1,
            },
            "find_missing_ingredient_or_reagent": {
                "inspect": 4,
                "device_control": 3,
                "relocation": 2,
                "focus": 1,
                "transfer_or_transform": 1,
                "tool_application": 1,
            },
            "combine_or_transform": {
                "transfer_or_transform": 3,
                "tool_application": 3,
                "inspect": 2,
                "device_control": 2,
                "focus": 2,
                "relocation": 1,
            },
            "verify_intermediate": {
                "focus": 2,
                "inspect": 3,
                "transfer_or_transform": 2,
                "tool_application": 2,
                "device_control": 1,
                "relocation": 1,
            },
            "verify_final": {
                "focus": 2,
                "inspect": 3,
                "transfer_or_transform": 2,
                "tool_application": 2,
                "device_control": 1,
                "relocation": 1,
            },
            "locate_instrument": {
                "inspect": 3,
                "focus": 2,
                "device_control": 2,
                "relocation": 2,
                "tool_application": 1,
            },
            "locate_measured_target": {
                "inspect": 3,
                "focus": 2,
                "device_control": 2,
                "relocation": 2,
                "tool_application": 1,
            },
            "measure_target": {
                "tool_application": 3,
                "inspect": 2,
                "focus": 2,
                "device_control": 2,
                "relocation": 1,
            },
            "resolve_branch": {
                "tool_application": 3,
                "inspect": 3,
                "device_control": 2,
                "focus": 1,
                "transfer_or_transform": 1,
                "relocation": 1,
            },
            "execute_branch": {
                "focus": 2,
                "inspect": 2,
                "relocation": 2,
                "transfer_or_transform": 1,
                "device_control": 1,
            },
            "locate_substance": {
                "inspect": 4,
                "device_control": 3,
                "relocation": 2,
                "focus": 1,
                "transfer_or_transform": 1,
            },
            "probe_sources": {
                "inspect": 4,
                "device_control": 4,
                "focus": 1,
                "relocation": 1,
                "transfer_or_transform": 1,
            },
            "confirm_referent": {
                "inspect": 3,
                "focus": 2,
                "device_control": 2,
                "tool_application": 1,
                "transfer_or_transform": 1,
                "relocation": 1,
            },
            "test_transformation": {
                "inspect": 2,
                "device_control": 3,
                "tool_application": 3,
                "transfer_or_transform": 2,
                "focus": 1,
                "relocation": 1,
            },
            "verify_outcome": {
                "inspect": 3,
                "focus": 2,
                "device_control": 2,
                "tool_application": 2,
                "transfer_or_transform": 1,
                "relocation": 1,
            },
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
            "locate_base_artifact": {
                "inspect": 7,
                "device_control": 6,
                "relocation": 5,
                "focus": 2,
                "transfer_or_transform": -3,
                "tool_application": -3,
                "relation": -4,
                "other": -2,
                "idle": -5,
            },
            "find_missing_ingredient_or_reagent": {
                "inspect": 8,
                "device_control": 7,
                "relocation": 5,
                "focus": 2,
                "transfer_or_transform": -4,
                "tool_application": -4,
                "relation": -4,
                "other": -2,
                "idle": -5,
            },
            "combine_or_transform": {
                "transfer_or_transform": 8,
                "tool_application": 7,
                "inspect": 5,
                "device_control": 4,
                "focus": 4,
                "relocation": 1,
                "relation": -1,
                "other": -2,
                "idle": -5,
            },
            "verify_intermediate": {
                "focus": 8,
                "inspect": 7,
                "transfer_or_transform": 4,
                "tool_application": 4,
                "device_control": 2,
                "relocation": 1,
                "relation": -2,
                "other": -2,
                "idle": -5,
            },
            "verify_final": {
                "focus": 8,
                "inspect": 7,
                "transfer_or_transform": 4,
                "tool_application": 4,
                "device_control": 2,
                "relocation": 1,
                "relation": -2,
                "other": -2,
                "idle": -5,
            },
            "locate_instrument": {
                "focus": 7,
                "inspect": 6,
                "device_control": 5,
                "relocation": 5,
                "tool_application": 2,
                "transfer_or_transform": -2,
                "relation": -4,
                "other": -2,
                "idle": -5,
            },
            "locate_measured_target": {
                "focus": 7,
                "inspect": 6,
                "device_control": 5,
                "relocation": 5,
                "tool_application": 3,
                "transfer_or_transform": -2,
                "relation": -4,
                "other": -2,
                "idle": -5,
            },
            "measure_target": {
                "tool_application": 8,
                "inspect": 6,
                "focus": 5,
                "device_control": 4,
                "relocation": 2,
                "transfer_or_transform": 1,
                "relation": -3,
                "other": -2,
                "idle": -5,
            },
            "resolve_branch": {
                "tool_application": 8,
                "inspect": 7,
                "device_control": 6,
                "focus": 2,
                "transfer_or_transform": 4,
                "relocation": 1,
                "relation": -3,
                "other": -2,
                "idle": -5,
            },
            "execute_branch": {
                "focus": 7,
                "inspect": 6,
                "relocation": 5,
                "transfer_or_transform": 4,
                "device_control": 3,
                "tool_application": 2,
                "relation": -3,
                "other": -2,
                "idle": -5,
            },
            "locate_substance": {
                "inspect": 7,
                "device_control": 7,
                "relocation": 4,
                "focus": 1,
                "transfer_or_transform": -2,
                "tool_application": -2,
                "relation": -4,
                "other": -2,
                "idle": -5,
            },
            "probe_sources": {
                "inspect": 8,
                "device_control": 8,
                "focus": 1,
                "relocation": 1,
                "transfer_or_transform": -4,
                "tool_application": -4,
                "relation": -5,
                "other": -3,
                "idle": -5,
            },
            "confirm_referent": {
                "focus": 7,
                "inspect": 6,
                "device_control": 4,
                "transfer_or_transform": 2,
                "tool_application": 2,
                "relocation": 2,
                "relation": -2,
                "other": -2,
                "idle": -5,
            },
            "test_transformation": {
                "tool_application": 7,
                "device_control": 6,
                "transfer_or_transform": 6,
                "inspect": 5,
                "focus": 2,
                "relocation": 1,
                "relation": -1,
                "other": -2,
                "idle": -5,
            },
            "verify_outcome": {
                "inspect": 7,
                "focus": 5,
                "device_control": 4,
                "tool_application": 3,
                "transfer_or_transform": 2,
                "relocation": 1,
                "relation": -2,
                "other": -2,
                "idle": -5,
            },
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

        candidates: list[tuple[int, int, str]] = []
        for command in admissible_commands:
            if self._classify_action_family(command) != family:
                continue
            command_tokens = set(
                self._extract_action_content_tokens(command, family=family)
            )
            if not suggested_tokens.issubset(command_tokens):
                continue

            extra = len(command_tokens - suggested_tokens)
            score = len(suggested_tokens) * 10 - extra * 2
            if suggested_tokens == command_tokens:
                score += 8
            if self._get_action_referent_signature(
                command, family=family
            ) == self._get_action_referent_signature(suggested_action, family=family):
                score += 4
            candidates.append((score, extra, command))

        if not candidates:
            room_transition_fallback = self._canonicalize_room_transition_action(
                suggested_action, admissible_commands
            )
            if room_transition_fallback is not None:
                return room_transition_fallback
            unsupported_substance_fallback = (
                self._canonicalize_unsupported_substance_action(
                    suggested_action, admissible_commands
                )
            )
            if unsupported_substance_fallback is not None:
                return unsupported_substance_fallback
            return suggested_action

        candidates.sort(key=lambda item: (-item[0], item[1], len(item[2]), item[2]))
        best_score, _, best_command = candidates[0]
        if best_score < 18:
            room_transition_fallback = self._canonicalize_room_transition_action(
                suggested_action, admissible_commands
            )
            if room_transition_fallback is not None:
                return room_transition_fallback
            unsupported_substance_fallback = (
                self._canonicalize_unsupported_substance_action(
                    suggested_action, admissible_commands
                )
            )
            if unsupported_substance_fallback is not None:
                return unsupported_substance_fallback
            return suggested_action
        if len(candidates) > 1 and best_score - candidates[1][0] < 4:
            room_transition_fallback = self._canonicalize_room_transition_action(
                suggested_action, admissible_commands
            )
            if room_transition_fallback is not None:
                return room_transition_fallback
            unsupported_substance_fallback = (
                self._canonicalize_unsupported_substance_action(
                    suggested_action, admissible_commands
                )
            )
            if unsupported_substance_fallback is not None:
                return unsupported_substance_fallback
            return suggested_action
        return best_command

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
        observation_changed = normalized_observation != self._normalize_runtime_text(
            previous_observation
        )
        has_admissible_delta = bool(
            self.percept.get("newly_admissible_actions")
            or self.percept.get("no_longer_admissible_actions")
        )
        progress_signal = self._estimate_task_progress_signal(
            family=family,
            executed_action=executed_action,
            observation=observation,
        )

        if executed_action is None or self._HARD_FAILURE_RE.search(observation):
            return "invalid", "The attempted action failed or was inadmissible."
        if family in {"focus", "inspect"} and self._is_repeated_action_observation(
            executed_action, observation, family=family
        ):
            return (
                "stalled",
                "Repeated confirmation produced no new task-relevant evidence.",
            )
        if family == "inspect" and observation_changed:
            if progress_signal >= 2:
                return "evidence", "Inspection produced task-relevant evidence."
            return "evidence", "Inspection produced new evidence."
        if self.task_status == "COMPLETED" or self._DIRECT_EFFECT_RE.search(
            observation
        ):
            if progress_signal >= 2 or self.task_status == "COMPLETED":
                return (
                    "observable_change",
                    "The observation reported a task-relevant state change.",
                )
            return (
                "evidence",
                "The observation reported a state change, but its task value is still uncertain.",
            )
        if has_admissible_delta:
            if progress_signal >= 2:
                return (
                    "observable_change",
                    "The action changed task-relevant affordances.",
                )
            return (
                "evidence",
                "The action changed the available affordances, but not yet in a task-relevant way.",
            )
        if self._EFFECTLESS_RE.search(observation) or not observation_changed:
            return "stalled", "No task-relevant change was observed."
        if progress_signal >= 1:
            return (
                "evidence",
                "The action changed nearby context, but its task relevance still needs confirmation.",
            )
        return (
            "uncertain",
            "The action changed context, but its causal value is still unclear.",
        )

    def _estimate_task_progress_signal(
        self,
        *,
        family: str,
        executed_action: str | None,
        observation: str,
    ) -> int:
        task_contract = self._get_task_contract()
        role_token_sets = self._get_task_role_token_sets(task_contract)
        action_tokens = set(
            self._extract_action_content_tokens(executed_action or "", family=family)
        )
        observation_tokens = set(self._extract_runtime_tokens(observation, limit=24))
        combined_tokens = action_tokens | observation_tokens
        target_entity_set = set(task_contract.get("target_entities", []))
        required_families = set(task_contract.get("required_families", []))
        support_families = set(task_contract.get("support_families", []))
        referent_signature = self._get_action_referent_signature(
            executed_action or "", family=family
        )
        support_referent = self._is_support_referent(referent_signature, task_contract)
        candidate_target = self._get_candidate_action_target(
            executed_action or "", family=family, task_contract=task_contract
        )
        measurement_subject = self._extract_measurement_subject_signature(
            executed_action or "", family=family
        )

        signal = 0
        if family in required_families and not (
            self._is_candidate_search_task(task_contract)
            and family == "focus"
            and support_referent
        ):
            signal += 1
        elif family in support_families:
            signal += 1
        if combined_tokens & target_entity_set:
            signal += 1
        if self._is_lifecycle_task():
            lifecycle_focus_labels = self._get_focus_stage_labels(
                executed_action or "", observation
            )
            if lifecycle_focus_labels:
                signal += 2
            elif family == "focus":
                signal = max(signal - 1, 0)
        if self._has_full_role_match(
            combined_tokens, role_token_sets["primary_targets"]
        ):
            signal += 2
        elif self._best_role_overlap(
            combined_tokens, role_token_sets["primary_targets"]
        )[0]:
            signal += 1
        if self._has_full_role_match(
            combined_tokens, role_token_sets["required_relations"]
        ):
            signal += 2
        elif self._best_role_overlap(
            combined_tokens, role_token_sets["required_relations"]
        )[0]:
            signal += 1
        if self._best_role_overlap(
            combined_tokens, role_token_sets["supporting_targets"]
        )[0]:
            signal += 1
        if (
            self._is_candidate_search_task(task_contract)
            and candidate_target
            and candidate_target in self._rejected_candidates
        ):
            signal = 0
        elif (
            self._is_candidate_search_task(task_contract)
            and support_referent
            and family in {"focus", "inspect"}
        ):
            signal = max(signal - 2, 0)

        if self._is_artifact_creation_task(task_contract):
            artifact_type_match = self._has_full_role_match(
                combined_tokens, role_token_sets["artifact_type"]
            )
            intermediate_match = self._has_full_role_match(
                combined_tokens, role_token_sets["artifact_intermediate_targets"]
            )
            final_match = self._has_full_role_match(
                combined_tokens, role_token_sets["artifact_final_targets"]
            )
            descriptor_only = bool(
                combined_tokens
                & set(task_contract.get("artifact_descriptor_tokens", []))
            ) and not (artifact_type_match or intermediate_match or final_match)
            if artifact_type_match:
                signal += 1
            if intermediate_match or final_match:
                signal += 2
            if (
                len(self._get_grounded_artifact_labels(limit=6)) <= 1
                and family
                in {"transfer_or_transform", "tool_application", "relocation"}
                and artifact_type_match
                and not (intermediate_match or final_match)
            ):
                signal = min(signal, 1)
            if descriptor_only:
                signal = min(signal, 1)

        if self._is_measurement_task(task_contract):
            direct_measurement = self._signature_matches_role(
                measurement_subject, role_token_sets["measurement_target"]
            )
            branch_target_action = self._signature_matches_role(
                referent_signature, role_token_sets["measurement_branch_targets"]
            )
            selected_branch_action = (
                referent_signature == self._selected_measurement_branch_target
                and bool(self._selected_measurement_branch_target)
            )
            direct_value_entry = self._get_latest_measurement(direct=True)
            if family == "tool_application":
                if direct_measurement:
                    signal += 2
                elif measurement_subject:
                    signal = min(signal, 1)
            if branch_target_action and not self._selected_measurement_branch_target:
                signal = 0
            elif branch_target_action and selected_branch_action:
                signal += 2
            if (
                direct_value_entry is not None
                and direct_value_entry.get("subject") != measurement_subject
                and family == "tool_application"
                and measurement_subject
            ):
                signal = min(signal, 1)

        for command in self.percept.get("newly_admissible_actions", []):
            command_tokens = set(
                self._extract_action_content_tokens(
                    command, family=self._classify_action_family(command)
                )
            )
            if (
                command_tokens & target_entity_set
                or self._has_full_role_match(
                    command_tokens, role_token_sets["primary_targets"]
                )
                or self._has_full_role_match(
                    command_tokens, role_token_sets["required_relations"]
                )
            ):
                signal += 1
                break

        return signal

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
            self._record_invalid_exact_action(executed_action or suggested_action)
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
        self._record_action_observation_signature(
            executed_action or suggested_action,
            observation,
            family=family,
        )

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

    def _score_state_change_action(
        self,
        *,
        action: str,
        family: str,
        current_phase: str,
        content_token_set: set[str],
        grounded_token_set: set[str],
        task_contract: dict,
        role_token_sets: dict[str, list[set[str]]],
        grounded_hits: int,
        target_hits: int,
        primary_role_hits: int,
        support_role_hits: int,
    ) -> int:
        if not self._is_state_change_task(task_contract):
            return 0

        normalized = self._normalize_runtime_text(action)
        substance_role_hits, _ = self._best_role_overlap(
            content_token_set, role_token_sets["target_substances"]
        )
        substance_full_match = self._has_full_role_match(
            content_token_set, role_token_sets["target_substances"]
        )
        substance_grounded = self._state_change_target_is_grounded(
            task_contract, grounded_token_set
        )
        grounded_substance_token_sets = self._get_grounded_substance_token_sets()
        grounded_substance_hits, _ = self._best_role_overlap(
            content_token_set, grounded_substance_token_sets
        )
        grounded_substance_match = self._has_full_role_match(
            content_token_set, grounded_substance_token_sets
        )
        search_snapshot = self._get_substance_search_snapshot()
        source_candidates = set(search_snapshot.get("source_candidates", []))
        exhausted_containers = set(search_snapshot.get("exhausted_containers", []))
        referent_signature = self._get_action_referent_signature(action, family=family)
        source_signature = self._extract_action_source_signature(action, family=family)
        source_candidate_match = bool(
            (referent_signature and referent_signature in source_candidates)
            or (source_signature and source_signature in source_candidates)
        )
        unsupported_substance_tokens = {
            token
            for token in content_token_set
            if token not in grounded_token_set
            and token not in self._CONTAINER_REFERENT_TOKENS
            and not any(
                token in token_set for token_set in grounded_substance_token_sets
            )
        }
        score = 0

        if substance_role_hits:
            score += substance_role_hits * 8
            if substance_full_match:
                score += 6
        if grounded_substance_hits:
            score += grounded_substance_hits * 6
            if grounded_substance_match:
                score += 4

        if current_phase == "locate_substance":
            if family == "inspect":
                score += 10
                if normalized.startswith("look in "):
                    score += 6
                elif normalized.startswith(("look at ", "look around", "inspect ")):
                    score += 3
            elif family == "device_control":
                score += 8
                if normalized.startswith("open "):
                    score += 4
            elif family == "focus":
                score += 6 if substance_role_hits and substance_grounded else -14
            elif family in {"tool_application", "transfer_or_transform", "relation"}:
                score += 4 if substance_role_hits and substance_grounded else -16
            elif family == "relocation" and not (
                substance_role_hits or support_role_hits
            ):
                score -= 12
                if normalized.startswith(("move ", "pick up ", "take ", "grab ")):
                    score -= 10

            if (
                referent_signature
                and referent_signature in exhausted_containers
                and family in {"inspect", "device_control"}
            ):
                score -= 6

            if (
                content_token_set
                and grounded_hits > 0
                and target_hits == 0
                and family not in {"inspect", "device_control", "relocation"}
            ):
                score -= 8
            if (
                family == "relocation"
                and grounded_hits > 0
                and target_hits == 0
                and not substance_role_hits
            ):
                score -= 8

        elif current_phase == "probe_sources":
            if family in {"inspect", "device_control"}:
                score += 10 if source_candidate_match else -4
                if family == "inspect" and normalized.startswith("look at "):
                    score += 3
                if family == "device_control" and normalized.startswith("open "):
                    score += 2
            elif family == "focus":
                score += 3 if substance_full_match or grounded_substance_match else -14
            elif family in {"tool_application", "transfer_or_transform", "relation"}:
                score -= 16
            elif family == "relocation":
                score -= 10

        elif current_phase == "confirm_referent":
            if family == "focus":
                score += 18 if (substance_full_match or primary_role_hits) else -12
            elif family == "inspect" and (substance_role_hits or primary_role_hits):
                score += 10
            elif family in {"tool_application", "transfer_or_transform"}:
                score -= 10 if not (substance_role_hits or primary_role_hits) else 4
            elif family == "device_control" and not (
                substance_role_hits or primary_role_hits or support_role_hits
            ):
                score -= 6

        elif current_phase == "test_transformation":
            if family in {
                "tool_application",
                "device_control",
                "transfer_or_transform",
            }:
                score += 12 if (substance_role_hits or primary_role_hits) else -8
            elif family == "inspect" and (substance_role_hits or primary_role_hits):
                score += 8
            elif family == "focus":
                score += 4 if (substance_role_hits or primary_role_hits) else -8
            elif family == "relocation" and not (
                substance_role_hits or primary_role_hits or support_role_hits
            ):
                score -= 8

        elif current_phase == "verify_outcome":
            if family == "inspect" and (substance_role_hits or primary_role_hits):
                score += 12
            elif family == "focus" and (substance_role_hits or primary_role_hits):
                score += 8
            elif family in {
                "tool_application",
                "device_control",
                "transfer_or_transform",
            }:
                score += 2 if (substance_role_hits or primary_role_hits) else -10

        if not substance_grounded:
            if family == "focus" and not substance_role_hits:
                score -= 8
            if (
                family in {"tool_application", "transfer_or_transform"}
                and not substance_role_hits
            ):
                score -= 6
        if (
            family in {"focus", "inspect", "tool_application", "transfer_or_transform"}
            and unsupported_substance_tokens
            and not grounded_substance_match
            and self._is_container_like_action(action, family=family)
        ):
            score -= 14
        if (
            source_candidates
            and current_phase == "locate_substance"
            and source_candidate_match
        ):
            score += 5

        return score

    def _score_artifact_creation_action(
        self,
        *,
        action: str,
        family: str,
        current_phase: str,
        content_token_set: set[str],
        grounded_token_set: set[str],
        task_contract: dict,
        role_token_sets: dict[str, list[set[str]]],
        grounded_hits: int,
        support_role_hits: int,
    ) -> int:
        if not self._is_artifact_creation_task(task_contract):
            return 0

        normalized = self._normalize_runtime_text(action)
        artifact_role_hits, _ = self._best_role_overlap(
            content_token_set, role_token_sets["artifact_type"]
        )
        artifact_full_match = self._has_full_role_match(
            content_token_set, role_token_sets["artifact_type"]
        )
        intermediate_hits, _ = self._best_role_overlap(
            content_token_set, role_token_sets["artifact_intermediate_targets"]
        )
        intermediate_full_match = self._has_full_role_match(
            content_token_set, role_token_sets["artifact_intermediate_targets"]
        )
        final_hits, _ = self._best_role_overlap(
            content_token_set, role_token_sets["artifact_final_targets"]
        )
        final_full_match = self._has_full_role_match(
            content_token_set, role_token_sets["artifact_final_targets"]
        )
        grounded_artifact_token_sets = self._get_grounded_artifact_token_sets()
        grounded_artifact_hits, _ = self._best_role_overlap(
            content_token_set, grounded_artifact_token_sets
        )
        grounded_artifact_match = self._has_full_role_match(
            content_token_set, grounded_artifact_token_sets
        )
        grounded_artifact_count = len(self._get_grounded_artifact_labels(limit=6))
        descriptor_hits = len(
            content_token_set & set(task_contract.get("artifact_descriptor_tokens", []))
        )
        score = 0

        if artifact_full_match:
            score += 10
        elif artifact_role_hits:
            score += artifact_role_hits * 4
        if grounded_artifact_match:
            score += 8
        elif grounded_artifact_hits:
            score += grounded_artifact_hits * 3
        if intermediate_full_match:
            score += 10
        elif intermediate_hits:
            score += intermediate_hits * 4
        if final_full_match:
            score += 12
        elif final_hits:
            score += final_hits * 5

        if current_phase == "locate_base_artifact":
            if family == "inspect":
                score += 10
                if normalized.startswith(("look in ", "look around")):
                    score += 3
            elif family == "device_control":
                score += 8
                if normalized.startswith("open "):
                    score += 2
            elif family == "relocation":
                score += 6
            elif family == "focus":
                score += (
                    10
                    if (
                        artifact_full_match
                        or grounded_artifact_match
                        or intermediate_full_match
                        or final_full_match
                    )
                    else -10
                )
            elif family in {
                "transfer_or_transform",
                "tool_application",
                "relation",
            }:
                score -= 14

        elif current_phase == "find_missing_ingredient_or_reagent":
            if family in {"inspect", "device_control"}:
                score += 8
            elif family == "relocation":
                score += 8
                if grounded_artifact_count <= 1 and (
                    artifact_full_match or grounded_artifact_match
                ):
                    score -= 78
            elif family == "focus":
                score += (
                    10
                    if (
                        artifact_full_match
                        or grounded_artifact_match
                        or intermediate_full_match
                        or final_full_match
                    )
                    else -12
                )
            elif family in {"transfer_or_transform", "tool_application"}:
                if grounded_artifact_count <= 1 and (
                    artifact_full_match or grounded_artifact_match
                ):
                    score -= 62
                elif (
                    artifact_full_match
                    or grounded_artifact_match
                    or intermediate_hits
                    or final_hits
                ):
                    score += 6
                else:
                    score -= 10
            elif family == "relation":
                score -= 10

        elif current_phase == "combine_or_transform":
            if family in {"transfer_or_transform", "tool_application"}:
                score += (
                    12
                    if (
                        artifact_full_match
                        or grounded_artifact_match
                        or intermediate_hits
                        or final_hits
                    )
                    else -8
                )
            elif family == "inspect":
                score += 8 if artifact_role_hits or grounded_artifact_hits else 2
            elif family == "focus":
                score += (
                    10
                    if (
                        grounded_artifact_match
                        or intermediate_full_match
                        or final_full_match
                    )
                    else -8
                )
            elif family == "device_control":
                score += (
                    4
                    if artifact_role_hits or grounded_hits or support_role_hits
                    else -4
                )
            elif family == "relocation" and not (
                artifact_full_match or grounded_artifact_match or support_role_hits
            ):
                score -= 6

        elif current_phase == "verify_intermediate":
            if family == "focus":
                score += (
                    18 if intermediate_full_match else -8 if final_full_match else -10
                )
            elif family == "inspect":
                score += 10 if intermediate_hits or grounded_artifact_hits else 2
            elif family in {"transfer_or_transform", "tool_application"}:
                score += 4 if grounded_artifact_match else -8

        elif current_phase == "verify_final":
            if family == "focus":
                score += (
                    18 if final_full_match else -8 if intermediate_full_match else -10
                )
            elif family == "inspect":
                score += 10 if final_hits or grounded_artifact_hits else 2
            elif family in {"transfer_or_transform", "tool_application"}:
                score += 4 if grounded_artifact_match else -8

        if descriptor_hits and not (
            artifact_full_match
            or grounded_artifact_match
            or intermediate_hits
            or final_hits
        ):
            score -= 18

        if (
            current_phase
            in {"locate_base_artifact", "find_missing_ingredient_or_reagent"}
            and grounded_artifact_count <= 1
            and family in {"relocation", "transfer_or_transform"}
            and grounded_artifact_match
        ):
            score -= 10

        if (
            family == "device_control"
            and descriptor_hits
            and not artifact_role_hits
            and not grounded_artifact_hits
            and not support_role_hits
        ):
            score -= 6

        return score

    def _score_measurement_action(
        self,
        *,
        action: str,
        family: str,
        current_phase: str,
        content_token_set: set[str],
        grounded_token_set: set[str],
        task_contract: dict,
        role_token_sets: dict[str, list[set[str]]],
    ) -> int:
        if not self._is_measurement_task(task_contract):
            return 0

        normalized = self._normalize_runtime_text(action)
        referent_signature = self._get_action_referent_signature(action, family=family)
        primary_signature = self._extract_action_primary_object_signature(
            action, family=family
        )
        measurement_subject = self._extract_measurement_subject_signature(
            action, family=family
        )
        measurement_target = task_contract.get("measurement_target", [""])
        measurement_target = (
            self._get_measurement_target_signature(task_contract)
            if measurement_target
            else ""
        )
        active_enclosure = self._resolve_enclosing_referent(measurement_target)
        target_hidden = bool(active_enclosure) and not self._referent_is_visible(
            measurement_target
        )

        referent_matches_target = self._signature_matches_role(
            referent_signature, role_token_sets["measurement_target"]
        )
        primary_matches_target = self._signature_matches_role(
            primary_signature, role_token_sets["measurement_target"]
        )
        subject_matches_target = self._signature_matches_role(
            measurement_subject, role_token_sets["measurement_target"]
        )
        instrument_primary_match = self._signature_matches_role(
            primary_signature, role_token_sets["measurement_instrument"]
        )
        instrument_referent_match = self._signature_matches_role(
            referent_signature, role_token_sets["measurement_instrument"]
        )
        branch_referent_match = self._signature_matches_role(
            referent_signature, role_token_sets["measurement_branch_targets"]
        )
        selected_branch_match = bool(
            self._selected_measurement_branch_target
            and referent_signature == self._selected_measurement_branch_target
        )
        touches_active_enclosure = bool(
            active_enclosure
            and active_enclosure
            in {
                referent_signature,
                measurement_subject,
                primary_signature,
            }
        )
        latest_direct_measurement = self._get_latest_measurement(direct=True)
        direct_measurement_ready = latest_direct_measurement is not None
        score = 0

        if current_phase == "locate_instrument":
            if family == "focus":
                score += (
                    18 if referent_matches_target or instrument_referent_match else -8
                )
            elif family == "inspect":
                score += 10 if instrument_referent_match else 2
            elif family == "relocation" and instrument_referent_match:
                score += 6
            elif family == "tool_application" and instrument_primary_match:
                score += 4

        elif current_phase == "locate_measured_target":
            if family == "focus":
                score += 18 if referent_matches_target else -8
            elif family == "inspect":
                score += (
                    10 if (referent_matches_target or primary_matches_target) else 2
                )
            elif family == "tool_application" and subject_matches_target:
                score += 6

        elif current_phase == "measure_target":
            if family == "tool_application":
                if subject_matches_target and instrument_primary_match:
                    score += 22
                    if target_hidden:
                        score -= 30
                elif touches_active_enclosure and instrument_primary_match:
                    score += 8
                else:
                    score -= 10
            elif family == "focus":
                if referent_matches_target or instrument_referent_match:
                    score += 8
            elif family == "inspect":
                if referent_matches_target or touches_active_enclosure:
                    score += 8

        elif current_phase == "resolve_branch":
            if family == "tool_application":
                if subject_matches_target and instrument_primary_match:
                    score += 16
                    if target_hidden:
                        score -= 30
                elif touches_active_enclosure and instrument_primary_match:
                    score += 6
                else:
                    score -= 8
            elif family in {"inspect", "device_control"}:
                if touches_active_enclosure:
                    score += 12
                elif referent_matches_target:
                    score += 8
            elif family in {"transfer_or_transform", "relocation"}:
                if referent_matches_target or touches_active_enclosure:
                    score += 6
                else:
                    score -= 8
            elif family == "focus":
                score += 4 if referent_matches_target else -6

        elif current_phase == "execute_branch":
            if family == "focus":
                score += (
                    18 if selected_branch_match else -10 if branch_referent_match else 0
                )
            elif family in {"inspect", "relocation", "transfer_or_transform"}:
                score += (
                    12 if selected_branch_match else -8 if branch_referent_match else 0
                )
            elif branch_referent_match:
                score -= 6

        if branch_referent_match and not self._selected_measurement_branch_target:
            score -= 18
        if branch_referent_match and not selected_branch_match:
            score -= 10

        if (
            target_hidden
            and not touches_active_enclosure
            and (
                referent_matches_target
                or primary_matches_target
                or subject_matches_target
            )
        ):
            score -= 24
            if family == "tool_application" and subject_matches_target:
                score -= 72
            elif family in {"focus", "inspect"} and referent_matches_target:
                score -= 48
        if (
            target_hidden
            and touches_active_enclosure
            and family
            in {
                "inspect",
                "device_control",
                "tool_application",
            }
        ):
            score += 14
        if (
            direct_measurement_ready
            and family == "tool_application"
            and not subject_matches_target
            and not touches_active_enclosure
        ):
            score -= 10
        if (
            normalized in self._invalid_exact_actions
            and self._invalid_exact_actions[normalized] > 0
        ):
            score -= 18 + self._invalid_exact_actions[normalized] * 4

        return score

    def _get_current_phase(self) -> str:
        task_contract = self._get_task_contract()
        if self._is_candidate_search_task(task_contract) and self._rejected_candidates:
            return "gather_evidence"
        if self._is_measurement_task(task_contract):
            role_token_sets = self._get_task_role_token_sets(task_contract)
            if task_contract.get(
                "measurement_instrument"
            ) and not self._role_focus_completed(
                role_token_sets["measurement_instrument"]
            ):
                return "locate_instrument"
            if task_contract.get(
                "measurement_target"
            ) and not self._role_focus_completed(role_token_sets["measurement_target"]):
                return "locate_measured_target"
            if self._get_latest_measurement(direct=True) is None:
                return "measure_target"
            if task_contract.get("measurement_branch_targets"):
                if self._selected_measurement_branch_target:
                    return "execute_branch"
                return "resolve_branch"
            return "measure_target"
        if self._is_state_change_task(task_contract):
            grounded_tokens = set(self._get_observation_grounded_tokens())
            role_token_sets = self._get_task_role_token_sets(task_contract)
            if not self._state_change_target_is_grounded(
                task_contract, grounded_tokens
            ):
                if self._should_probe_sources(task_contract):
                    return "probe_sources"
                return "locate_substance"
            if "focus" in task_contract.get(
                "required_families", []
            ) and not self._role_focus_completed(role_token_sets["primary_targets"]):
                return "confirm_referent"
            mechanism_progress = any(
                entry["observable_change_attempts"] > 0
                for entry in self.episode_hypothesis_ledger.values()
                if entry["family"]
                in {"device_control", "tool_application", "transfer_or_transform"}
            )
            if not mechanism_progress:
                return "test_transformation"
            return "verify_outcome"
        if self._is_artifact_creation_task(task_contract):
            role_token_sets = self._get_task_role_token_sets(task_contract)
            grounded_artifacts = self._get_grounded_artifact_labels(limit=6)
            intermediate_targets = task_contract.get(
                "artifact_intermediate_targets", []
            )
            final_targets = task_contract.get("artifact_final_targets", [])
            intermediate_focused = self._role_focus_completed(
                role_token_sets["artifact_intermediate_targets"]
            )
            final_focused = self._role_focus_completed(
                role_token_sets["artifact_final_targets"]
            )
            intermediate_grounded = self._artifact_role_is_grounded(
                role_token_sets["artifact_intermediate_targets"]
            )
            final_grounded = self._artifact_role_is_grounded(
                role_token_sets["artifact_final_targets"]
            )
            mechanism_progress = any(
                entry["observable_change_attempts"] > 0
                for entry in self.episode_hypothesis_ledger.values()
                if entry["family"]
                in {"tool_application", "transfer_or_transform", "device_control"}
            )
            if not grounded_artifacts:
                return "locate_base_artifact"
            if intermediate_targets and not intermediate_focused:
                if len(grounded_artifacts) <= 1 and not intermediate_grounded:
                    return "find_missing_ingredient_or_reagent"
                if intermediate_grounded:
                    return "verify_intermediate"
                return "combine_or_transform"
            if final_targets and not final_focused:
                if final_grounded:
                    return "verify_final"
                if len(grounded_artifacts) <= 1 and not mechanism_progress:
                    return "find_missing_ingredient_or_reagent"
                return "combine_or_transform"
            return "combine_or_transform"
        inspect_evidence = any(
            entry["evidence_attempts"] > 0
            for entry in self.episode_hypothesis_ledger.values()
            if entry["family"] == "inspect"
        )
        mechanism_progress = any(
            entry["observable_change_attempts"] > 0
            for entry in self.episode_hypothesis_ledger.values()
            if entry["family"]
            in {"relation", "tool_application", "transfer_or_transform"}
        )
        task_lower = self.task.lower()
        placement_task = any(
            token in task_lower for token in ("place", "put", "move", "deliver")
        )
        if not inspect_evidence:
            return "gather_evidence"
        if (
            any(
                token in task_lower
                for token in ("determine", "test", "whether", "identify")
            )
            and not mechanism_progress
        ):
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
        role_token_sets = self._get_task_role_token_sets(task_contract)
        required_families = set(task_contract.get("required_families", []))
        support_families = set(task_contract.get("support_families", []))
        target_entity_set = set(task_contract.get("target_entities", []))
        current_location_tokens = set(
            self._extract_current_location_tokens(
                (self.percept or {}).get("resulting_observation", "")
            )
        )
        visible_nonlocation_targets = (
            grounded_token_set & target_entity_set
        ) - current_location_tokens
        ordered_sequence = "ordered_sequence" in task_contract.get("ordering_cues", [])
        referent_signature = self._get_action_referent_signature(action, family=family)
        candidate_search_task = self._is_candidate_search_task(task_contract)
        candidate_target = self._get_candidate_action_target(
            action, family=family, task_contract=task_contract
        )
        support_referent = self._is_support_referent(referent_signature, task_contract)
        recent_repeat_count = self._count_recent_referent_repeats(
            family=family, referent_signature=referent_signature
        )
        primary_role_hits, _ = self._best_role_overlap(
            content_token_set, role_token_sets["primary_targets"]
        )
        relation_role_hits, _ = self._best_role_overlap(
            content_token_set, role_token_sets["required_relations"]
        )
        support_role_hits, _ = self._best_role_overlap(
            content_token_set, role_token_sets["supporting_targets"]
        )
        primary_full_match = self._has_full_role_match(
            content_token_set, role_token_sets["primary_targets"]
        )
        relation_full_match = self._has_full_role_match(
            content_token_set, role_token_sets["required_relations"]
        )
        primary_target_grounded = self._role_is_grounded(
            grounded_token_set, role_token_sets["primary_targets"]
        )
        primary_target_focused = self._role_focus_completed(
            role_token_sets["primary_targets"]
        )
        state_change_task = self._is_state_change_task(task_contract)
        artifact_creation_task = self._is_artifact_creation_task(task_contract)
        measurement_task = self._is_measurement_task(task_contract)
        lifecycle_task = self._is_lifecycle_task(task_contract)
        lifecycle_targets_visible = bool(visible_nonlocation_targets) or bool(
            self._observed_stage_labels
        )
        stage_labels = (
            self._get_focus_stage_labels(action, "")
            if lifecycle_task and family == "focus"
            else self._merge_stage_labels(self._extract_stage_labels(action))
            if lifecycle_task
            else []
        )
        next_expected_stage = (
            self._get_next_expected_stage_label() if lifecycle_task else None
        )
        seen_stage_labels = [
            label for label in stage_labels if label in self._focused_stage_labels
        ]
        unseen_stage_labels = [
            label for label in stage_labels if label not in self._focused_stage_labels
        ]
        score = 0

        keyword_hits = len(action_token_set & task_keyword_set)
        grounded_hits = len(content_token_set & grounded_token_set)
        target_hits = len(content_token_set & target_entity_set)
        score += keyword_hits * 5
        score += grounded_hits * 6
        score += target_hits * 5
        if primary_full_match:
            score += 12
        elif primary_role_hits:
            score += primary_role_hits * 3
        if relation_full_match:
            score += 9
        elif relation_role_hits:
            score += relation_role_hits * 3
        if support_role_hits:
            score += support_role_hits * 2

        family_priority = self._get_family_priority(current_phase, family)
        score += family_priority

        if family in required_families and not (
            candidate_search_task and family == "focus" and support_referent
        ):
            score += 7
            if grounded_hits:
                score += 4
            if target_hits:
                score += 6
            if family == "relation" and (
                primary_target_grounded or primary_target_focused
            ):
                score += 8
            if family == "focus" and primary_target_grounded:
                score += 3
        elif family in support_families:
            score += 4

        if family == "inspect" and (keyword_hits or grounded_hits):
            score += 5
            if primary_full_match or relation_full_match:
                score += 5
        if family == "device_control" and grounded_hits:
            score += 4
            if not (
                target_hits
                or primary_role_hits
                or relation_role_hits
                or support_role_hits
            ):
                score -= 4
        if family == "relocation" and grounded_hits:
            score += 4
        if family == "transfer_or_transform" and grounded_hits:
            score += 4
            if primary_target_focused and not (
                primary_full_match or relation_full_match or support_role_hits
            ):
                score -= 5
        if family == "tool_application" and (keyword_hits or grounded_hits):
            score += 4
            if primary_full_match or relation_role_hits:
                score += 4
        if family == "focus" and grounded_hits:
            score += 3
        elif family == "focus":
            score -= 3
        if (
            family == "focus"
            and visible_nonlocation_targets
            and content_token_set
            and content_token_set.issubset(current_location_tokens)
        ):
            score -= 8

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
        invalid_repeat_count = self._invalid_exact_actions.get(normalized, 0)
        if invalid_repeat_count:
            score -= 12 + invalid_repeat_count * 4
        if recent_repeat_count >= 2 and family in {"focus", "inspect"}:
            score -= 10
        elif recent_repeat_count >= 1 and family in {"focus", "inspect"}:
            score -= 4

        if lifecycle_task:
            if family == "focus":
                if grounded_hits == 0:
                    score -= 18
                if unseen_stage_labels:
                    score += 14
                    if (
                        next_expected_stage
                        and next_expected_stage in unseen_stage_labels
                    ):
                        score += 10
                    if grounded_hits == 0:
                        score -= 24
                elif seen_stage_labels:
                    score -= 12
                else:
                    score -= 18
            elif family == "inspect" and (target_hits or primary_role_hits):
                score += 10
                if next_expected_stage and next_expected_stage in stage_labels:
                    score += 6

            if lifecycle_targets_visible and family in {
                "relation",
                "transfer_or_transform",
                "tool_application",
            }:
                score -= 30
            if (
                lifecycle_targets_visible
                and family == "device_control"
                and not (target_hits or primary_role_hits or relation_role_hits)
            ):
                score -= 8
            if (
                lifecycle_targets_visible
                and family == "relocation"
                and not (target_hits or primary_role_hits)
            ):
                score -= 4
        elif candidate_search_task:
            active_candidate_state = (
                self._candidate_states.get(self._active_candidate, {})
                if self._active_candidate
                else {}
            )
            if family == "focus" and support_referent:
                score -= 24
            elif family == "inspect" and support_referent:
                score -= 8

            if candidate_target:
                if candidate_target in self._rejected_candidates:
                    score -= 18
                elif (
                    self._rejected_candidates
                    and candidate_target != self._active_candidate
                ):
                    if family in {"inspect", "focus", "relocation"}:
                        score += 10
                    if family == "focus":
                        score += 6
                elif family == "inspect" and grounded_hits:
                    score += 3

            if (
                self._active_candidate
                and active_candidate_state.get("support_confirmed")
                and family in {"focus", "inspect"}
            ):
                if candidate_target == self._active_candidate:
                    score -= 12
                if support_referent:
                    score -= 10
        elif measurement_task:
            score += self._score_measurement_action(
                action=action,
                family=family,
                current_phase=current_phase,
                content_token_set=content_token_set,
                grounded_token_set=grounded_token_set,
                task_contract=task_contract,
                role_token_sets=role_token_sets,
            )
        elif artifact_creation_task:
            score += self._score_artifact_creation_action(
                action=action,
                family=family,
                current_phase=current_phase,
                content_token_set=content_token_set,
                grounded_token_set=grounded_token_set,
                task_contract=task_contract,
                role_token_sets=role_token_sets,
                grounded_hits=grounded_hits,
                support_role_hits=support_role_hits,
            )
        elif state_change_task:
            score += self._score_state_change_action(
                action=action,
                family=family,
                current_phase=current_phase,
                content_token_set=content_token_set,
                grounded_token_set=grounded_token_set,
                task_contract=task_contract,
                role_token_sets=role_token_sets,
                grounded_hits=grounded_hits,
                target_hits=target_hits,
                primary_role_hits=primary_role_hits,
                support_role_hits=support_role_hits,
            )

        if primary_target_focused and family == "relation":
            score += 8
        if primary_target_focused and relation_role_hits and family == "inspect":
            score += 4
        if (
            primary_target_focused
            and family == "device_control"
            and not (primary_role_hits or relation_role_hits or support_role_hits)
        ):
            score -= 6
        if primary_target_focused and family == "focus" and not primary_full_match:
            score -= 6

        if referent_signature and referent_signature in self._completed_focus_targets:
            if family == "focus":
                score -= 10 if ordered_sequence else 4
            else:
                score -= 2

        if (
            (primary_target_grounded or primary_target_focused)
            and primary_role_hits
            and not primary_full_match
        ):
            if family == "inspect":
                score -= 4
            else:
                score -= 8

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

        if visible_nonlocation_targets and target_hits == 0:
            if family == "inspect":
                score -= 2
            elif family not in {"device_control", "relocation"}:
                score -= 6

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
        if "ordered_sequence" in self._get_task_contract().get(
            "ordering_cues", []
        ) and set(grounded_tokens) & set(
            self._get_task_contract().get("target_entities", [])
        ):
            quotas = dict(quotas)
            quotas["focus"] = max(2, quotas.get("focus", 0))
            if quotas.get("inspect", 0) > 0:
                quotas["inspect"] -= 1
        task_contract = self._get_task_contract()
        role_token_sets = self._get_task_role_token_sets(task_contract)
        if "relation" in task_contract.get(
            "required_families", []
        ) and self._role_focus_completed(role_token_sets["primary_targets"]):
            quotas = dict(quotas)
            quotas["relation"] = max(2, quotas.get("relation", 0))
            if quotas.get("focus", 0) > 0:
                quotas["focus"] -= 1
        if self._is_candidate_search_task(task_contract):
            quotas = dict(quotas)
            quotas["inspect"] = max(2, quotas.get("inspect", 0))
            if self._rejected_candidates:
                quotas["focus"] = max(2, quotas.get("focus", 0))
                quotas["relocation"] = max(2, quotas.get("relocation", 0))
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
        artifact_creation = self._get_artifact_creation_snapshot()
        substance_search = self._get_substance_search_snapshot(actions)
        measurement_tracking = self._get_measurement_tracking_snapshot()

        return {
            "total_actions": len(actions),
            "current_phase": current_phase,
            "family_counts": family_counts,
            "salient_entities": grounded_tokens[:8],
            "task_relevant_action_shortlist": shortlist,
            "deprioritized_families": deprioritized_families,
            "candidate_tracking": self._get_candidate_tracking_snapshot(),
            "task_contract": task_contract,
            "artifact_creation": artifact_creation,
            "substance_search": substance_search,
            "measurement_tracking": measurement_tracking,
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

        summary = self._summarize_admissible_actions(
            self.admissible_actions, shortlist_limit=20
        )
        recent_invalid_actions = self._get_recent_invalid_actions()
        ordered_progress = self._get_ordered_target_snapshot()
        extra_context = ""
        if self._ordered_progress_has_signal(ordered_progress):
            extra_context += (
                "Ordered focus progress this episode: "
                + json.dumps(ordered_progress)
                + "\n"
            )
        candidate_tracking = summary.get("candidate_tracking", {})
        if self._candidate_tracking_has_signal(candidate_tracking):
            extra_context += (
                "Candidate tracking this episode: "
                + json.dumps(candidate_tracking)
                + "\n"
            )
        substance_search = summary.get("substance_search", {})
        if self._substance_search_has_signal(substance_search):
            extra_context += (
                "Substance search state this episode: "
                + json.dumps(substance_search)
                + "\n"
            )
        artifact_creation = summary.get("artifact_creation", {})
        if self._artifact_creation_has_signal(artifact_creation):
            extra_context += (
                "Artifact creation state this episode: "
                + json.dumps(artifact_creation)
                + "\n"
            )
        measurement_tracking = summary.get("measurement_tracking", {})
        if self._measurement_tracking_has_signal(measurement_tracking):
            extra_context += (
                "Measurement state this episode: "
                + json.dumps(measurement_tracking)
                + "\n"
            )
        if self.percept.get("referent_resolution"):
            extra_context += (
                "Latest referent resolution warning: "
                + json.dumps(self.percept["referent_resolution"])
                + "\n"
            )
        self._set_agent_system_message(
            self.action_agent,
            self._action_agent_base_prompt
            + "\n\n--- PRIVATE RUNTIME CONTEXT ---\n"
            + f"Current phase: {summary['current_phase']}\n"
            + f"Current exact task-relevant admissible shortlist: {json.dumps(summary['task_relevant_action_shortlist'])}\n"
            + f"Action family counts: {json.dumps(summary['family_counts'])}\n"
            + f"Salient grounded entities from the latest percept: {json.dumps(summary['salient_entities'])}\n"
            + f"Task contract: {json.dumps(summary['task_contract'])}\n"
            + f"Artifact creation snapshot: {json.dumps(summary['artifact_creation'])}\n"
            + f"Substance search snapshot: {json.dumps(summary['substance_search'])}\n"
            + f"Measurement tracking snapshot: {json.dumps(summary['measurement_tracking'])}\n"
            + f"Deprioritized mechanism families this episode: {json.dumps(summary['deprioritized_families'])}\n"
            + f"Recent invalid exact commands to avoid repeating: {json.dumps(recent_invalid_actions)}\n"
            + extra_context
            + "Choose an exact shortlist string whenever possible. "
            + "If you go off-shortlist, stay lexically close to grounded entities and known admissible verb families instead of inventing a fresh command template. "
            + "The executor will only execute the action if it actually matches the environment's admissible commands.\n",
        )

    @staticmethod
    def _set_agent_system_message(agent, content: str) -> None:
        if hasattr(agent, "_oai_system_message") and getattr(
            agent, "_oai_system_message"
        ):
            agent._oai_system_message[0]["content"] = content
            return
        setattr(agent, "system_message", content)

    def _synthesize_belief_state_fallback(self, malformed_content: str) -> str:
        percept = self.percept or {}
        timestep = percept.get("timestep", self.num_actions_taken)
        attempted_action = percept.get("attempted_action", "None")
        observation = percept.get("resulting_observation", "No observation available.")
        attempts_left = percept.get(
            "action_attempts_left", self.max_actions - self.num_actions_taken
        )
        ledger = self._get_episode_hypothesis_snapshot(
            max_families=3, max_recent_tests=2
        )

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

    def _is_valid_thinking_output(self, content: str) -> bool:
        normalized = (content or "").lstrip().upper()
        if not normalized:
            return False
        if normalized.startswith("[OBSERVATION]:") or normalized.startswith("ACTION:"):
            return False
        return normalized.startswith(self._THINKING_PREFIXES)

    def _synthesize_thinking_fallback(self, malformed_content: str) -> str:
        candidate_tracking = self._get_candidate_tracking_snapshot()
        active_candidate = candidate_tracking.get("active_candidate")
        rejected_candidates = candidate_tracking.get("rejected_candidates", [])
        task_contract = self._get_task_contract()
        if active_candidate and not rejected_candidates:
            idea = (
                f"STRATEGY: Re-anchor on the explicit task contract. Treat {active_candidate!r} "
                "as the current candidate only, not the destination support entities. "
                "If one grounded confirmation step still leaves the task incomplete, pivot to a new grounded candidate instead of repeating support focus or inspection."
            )
        elif rejected_candidates:
            idea = (
                "STRATEGY: The latest grounded candidate has already satisfied the visible task semantics without completing the task. "
                "Pivot to a different grounded candidate from the same task class and avoid repeating support-entity focus loops."
            )
        elif task_contract.get("destination_container") or task_contract.get(
            "destination_room"
        ):
            idea = (
                "STRATEGY: Separate the primary candidate from destination support entities. "
                "Use focus on the candidate only when required by the task, and treat the destination room/container as placement context rather than extra focus targets."
            )
        else:
            idea = (
                "STRATEGY: Re-anchor on the latest percept and task contract. "
                "Avoid repeating unchanged confirmation steps; prefer one grounded diagnostic or a pivot to a distinct grounded candidate."
            )
        if malformed_content.strip():
            idea += " Ignore the malformed prior reasoning and continue from confirmed perceptual state only."
        return idea

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
                self.group_chat_manager.llm_config = self.reasoner_config[
                    "config_list"
                ][0]

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
        self._update_state_change_search_tracking(
            action=action,
            observation=self.adapter.observation,
        )
        self._update_artifact_creation_tracking(self.adapter.observation)
        self._update_measurement_tracking(
            action=action,
            observation=self.adapter.observation,
        )
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
            "required_families": shared_action_context["task_contract"][
                "required_families"
            ],
        }
        substance_search = shared_action_context.get("substance_search", {})
        if self._substance_search_has_signal(substance_search):
            self.percept["substance_search"] = substance_search
        artifact_creation = shared_action_context.get("artifact_creation", {})
        if self._artifact_creation_has_signal(artifact_creation):
            self.percept["artifact_creation"] = artifact_creation
        measurement_tracking = shared_action_context.get("measurement_tracking", {})
        if self._measurement_tracking_has_signal(measurement_tracking):
            self.percept["measurement_tracking"] = measurement_tracking
        self.percept["task_relevant_action_shortlist"] = shared_action_context[
            "task_relevant_action_shortlist"
        ]
        ordered_progress = self._get_ordered_target_snapshot()
        if self._ordered_progress_has_signal(ordered_progress):
            self.percept["ordered_target_progress"] = ordered_progress
        candidate_tracking = self._get_candidate_tracking_snapshot()
        if self._candidate_tracking_has_signal(candidate_tracking):
            self.percept["candidate_tracking"] = candidate_tracking
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

            reasoning_action = executed_action or canonical_suggested_action
            ambiguity_resolution = None
            if reasoning_action:
                reasoning_action, ambiguity_resolution = self._resolve_reasoning_action(
                    reasoning_action, previous_observation
                )

            if reasoning_action:
                self._update_lifecycle_stage_state(
                    action=reasoning_action, observation=self.adapter.observation
                )
                self._update_ordered_target_progress(
                    executed_action=reasoning_action,
                    observation=self.adapter.observation,
                )
                self._update_candidate_tracking(
                    executed_action=reasoning_action,
                    observation=self.adapter.observation,
                    previous_observation=previous_observation,
                )

            attempted_action = (
                canonical_suggested_action if executed_action else suggested_action
            )
            self.update_percept(attempted_action)
            if canonical_suggested_action != suggested_action:
                self.percept["requested_action"] = suggested_action
                self.percept["canonicalized_action"] = canonical_suggested_action
                referent_resolution = self._record_referent_resolution(
                    suggested_action=suggested_action,
                    canonical_action=canonical_suggested_action,
                )
                if referent_resolution is not None:
                    self.percept["referent_resolution"] = referent_resolution
            if ambiguity_resolution is not None:
                self.percept["ambiguity_resolution"] = ambiguity_resolution
                self.percept["resolved_action"] = reasoning_action
            self._update_episode_hypothesis_ledger(
                suggested_action=reasoning_action or suggested_action,
                executed_action=reasoning_action
                if executed_action is not None
                else None,
                previous_observation=previous_observation,
            )
            hypothesis_snapshot = self._get_episode_hypothesis_snapshot(
                max_families=3, max_recent_tests=2
            )
            if hypothesis_snapshot["mechanisms"] or hypothesis_snapshot["recent_tests"]:
                self.percept["episode_hypothesis_ledger"] = hypothesis_snapshot
            ordered_progress = self._get_ordered_target_snapshot()
            if self._ordered_progress_has_signal(ordered_progress):
                self.percept["ordered_target_progress"] = ordered_progress
            self._refresh_action_agent_runtime_context()

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
                    "ordered_target_progress": self._get_ordered_target_snapshot(),
                    "candidate_tracking": self._get_candidate_tracking_snapshot(),
                },
            )
            + "\n\n"
        )

        state_section = "--- CURRENT STATE ---\n" + json.dumps(self.percept) + "\n"

        final_prompt = (
            "Begin cognitive deliberation. Coordinate through structured, grounded reasoning. "
            "Use prior knowledge when relevant, maximize communication efficiency, and confirm task completion explicitly through perceptual feedback."
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
                "ordered_target_progress": self._get_ordered_target_snapshot(),
                "candidate_tracking": self._get_candidate_tracking_snapshot(),
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

        if (
            last_speaker is self.thinking_agent
            and self.action_agent in possible_speakers
            and not self._is_valid_thinking_output(last_msg.get("content") or "")
        ):
            last_msg["content"] = self._synthesize_thinking_fallback(
                last_msg.get("content") or ""
            )
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
