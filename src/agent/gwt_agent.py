import hashlib
import io
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
from rich.console import Console, Group
from rich.panel import Panel
from rich.table import Table
from rich.text import Text
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
    _ANALYST_AGENT_GUIDE = {
        "Belief_State_Agent": (
            "Maintains the first-person world model from the latest percept. "
            "It should explain what the agent currently believes is true, uncertain, "
            "or newly discovered."
        ),
        "Thinking_Agent": (
            "Turns the current belief state into a tactic or hypothesis. "
            "It proposes what kind of evidence or action family should come next."
        ),
        "Action_Agent": (
            "Chooses the exact environment action to try next, ideally from the "
            "task-relevant admissible shortlist."
        ),
        "External_Perception_Agent": (
            "Executes the chosen action and returns the raw environment-facing result."
        ),
        "Focus_Agent": (
            "Recovery path used when the belief-state format or internal reasoning "
            "drifts. It helps re-anchor the cognitive loop."
        ),
        "Learning_Agent": (
            "Summarizes reusable lessons after success or failure without turning them "
            "into environment-specific hardcoded rules."
        ),
        "Retrieve_Memory_Agent": (
            "Fetches relevant prior concepts or episodic traces to support the current task."
        ),
    }
    _ANALYST_TERM_GLOSSARY = {
        "task_contract": (
            "Runtime decomposition of the task into goals, roles, and control signals.",
            "required_families, support_families, primary_targets, supporting_targets",
            "This is the architecture's high-level interpretation of what must be solved.",
        ),
        "required_families": (
            "Action families the task explicitly requires, such as focus or connect.",
            "task_contract, support_families",
            "These stay protected from premature retirement because the task text named them.",
        ),
        "support_families": (
            "Action families that usually help gather evidence, such as inspect.",
            "task_contract, required_families",
            "These support the main task but are not themselves the end goal.",
        ),
        "target_entities": (
            "General task-relevant nouns or phrases extracted from the instruction.",
            "primary_targets, supporting_targets, target_substances",
            "These help shortlist scoring stay semantically tied to the task.",
        ),
        "ordering_cues": (
            "Signals that the task has an order, such as earliest-to-latest or before/after.",
            "procedural_sequence, lifecycle_sequence, ordered_target_progress",
            "These determine whether the architecture should sequence actions or evidence.",
        ),
        "procedural_sequence": (
            "Task-level cue that one step must happen before another, without implying an ordered object list.",
            "ordering_cues, required_families",
            "Used to preserve task step order without confusing it with lifecycle progression.",
        ),
        "lifecycle_sequence": (
            "Task-level cue that the agent is reasoning over ordered stages of a lifecycle.",
            "ordering_cues, ordered_target_progress, growth_task",
            "Used for stage-aware focus tasks where the order is over biological or process stages.",
        ),
        "growth_task": (
            "Flag that the task likely requires creating, growing, or revealing a living/process outcome first.",
            "lifecycle_sequence, conditional_branch_task, evidence_target",
            "Keeps precursor-generation steps active before the agent branches or commits.",
        ),
        "state_change_task": (
            "Flag that the task is about changing the state of a target substance or object.",
            "target_substances, desired_transformation, transformation_direction, substance_search",
            "Routes the controller into substance search and transformation phases.",
        ),
        "artifact_creation_task": (
            "Flag that the task is about constructing or mixing a target artifact from ingredients.",
            "artifact_creation, artifact_type, artifact_intermediate_targets, artifact_final_targets",
            "Keeps artifact type and ingredient gaps central to planning.",
        ),
        "measurement_task": (
            "Flag that the task is about measuring a target property before acting on it.",
            "measurement_tracking, measurement_property, measurement_target, measurement_instrument",
            "Routes the controller into instrument search, measurement, and branch gating.",
        ),
        "comparison_task": (
            "Flag that the task requires comparing multiple targets before choosing or acting.",
            "comparison_tracking, comparison_subject, comparison_targets",
            "Prevents the agent from collapsing compared targets into a single winner too early.",
        ),
        "search_mode": (
            "Explicit task-level instruction that the agent should search for something first.",
            "inferred_search_mode, primary_targets, candidate_classes",
            "This comes directly from the task text.",
        ),
        "inferred_search_mode": (
            "Architecture inference that search is needed because the named target is not yet grounded.",
            "search_mode, primary_targets, remote_room_signal",
            "Lets the agent search even when the task never literally says find or locate.",
        ),
        "relation_mechanism_task": (
            "Flag that the task depends on building or testing a local relation mechanism, such as a circuit or support graph.",
            "relation_frontier, required_relations, supporting_targets",
            "Prunes huge relation spaces down to the active grounded mechanism.",
        ),
        "conditional_branch_task": (
            "Flag that the task has multiple possible outcomes and requires evidence before choosing one.",
            "conditional_branch_tracking, evidence_target, branch_ready, destination_container",
            "Keeps branch-target actions suppressed until the evidence actually resolves the branch.",
        ),
        "candidate_classes": (
            "Generic class labels for search tasks, such as living thing or non-living thing.",
            "candidate_tracking, primary_targets, inferred_search_mode",
            "These let the agent pivot between grounded candidates without hardcoded object lists.",
        ),
        "primary_targets": (
            "The main objects or entities the task is directly about.",
            "target_entities, supporting_targets, candidate_classes",
            "The search, focus, and branch logic should stay anchored to these.",
        ),
        "supporting_targets": (
            "Objects or locations that enable the main task without being the final subject.",
            "primary_targets, destination_container, destination_room, measurement_instrument",
            "Examples include destination containers, rooms, tools, or evidence sources.",
        ),
        "target_substances": (
            "Task-compatible substances the state-change controller is searching for or manipulating.",
            "state_change_task, desired_transformation, substance_search",
            "These keep the search focused on the right matter instead of unrelated visible substances.",
        ),
        "artifact_type": (
            "The type of artifact the task is trying to create, such as paint or mixture.",
            "artifact_creation_task, artifact_intermediate_targets, artifact_final_targets",
            "This protects the agent from adjective-only distractors of the wrong type.",
        ),
        "artifact_intermediate_targets": (
            "Intermediate artifact states or products that may need to exist before the final artifact.",
            "artifact_type, artifact_final_targets, artifact_creation",
            "These make multi-step creation tasks explicit instead of collapsing them into one jump.",
        ),
        "artifact_final_targets": (
            "The final artifact outcome the task wants produced or verified.",
            "artifact_type, artifact_intermediate_targets, supporting_targets",
            "This is the creation-task end state after ingredients and transformations are resolved.",
        ),
        "artifact_descriptor_tokens": (
            "Descriptors such as colors or modifiers attached to the artifact target.",
            "artifact_type, target_entities",
            "These should refine the artifact target, not replace the artifact type itself.",
        ),
        "measurement_property": (
            "The property the task wants measured or inferred, such as temperature or melting point.",
            "measurement_task, measurement_property_type, measurement_target",
            "This is the semantic property that must be resolved before acting on measurement branches.",
        ),
        "measurement_property_type": (
            "Architecture classification of the property, such as instantaneous state or stable threshold property.",
            "measurement_property, branch_ready, measurement_tracking",
            "This controls whether a raw measurement is enough or whether more evidence is required.",
        ),
        "measurement_target": (
            "The entity whose property is being measured.",
            "measurement_task, measurement_instrument, measurement_tracking",
            "This is the true evidence-bearing object for measurement tasks.",
        ),
        "measurement_instrument": (
            "The tool or device used to obtain measurement evidence.",
            "measurement_target, measurement_tracking, supporting_targets",
            "This can become its own search frontier when the instrument is not yet grounded.",
        ),
        "measurement_branch_targets": (
            "Branch outcomes that become relevant after the measured property crosses or resolves a threshold.",
            "measurement_branches, branch_ready, destination_container",
            "These stay suppressed until the measurement controller resolves the relevant property.",
        ),
        "measurement_branches": (
            "Explicit branch rules derived from the task for measurement outcomes.",
            "measurement_branch_targets, measurement_property, branch_ready",
            "These connect evidence about a property to the actions that should follow.",
        ),
        "comparison_subject": (
            "The shared attribute or substance being compared across multiple targets.",
            "comparison_targets, comparison_property, comparison_tracking",
            "This anchors the comparison to a common basis instead of letting targets drift apart semantically.",
        ),
        "comparison_property": (
            "The property used to decide which compared target wins.",
            "comparison_subject, comparison_direction, comparison_tracking",
            "This tells the architecture what evidence should settle the comparison.",
        ),
        "comparison_direction": (
            "The direction of the comparison, such as hotter, larger, or more conductive.",
            "comparison_property, comparison_targets",
            "This determines what kind of evidence counts as better or worse in the comparison.",
        ),
        "comparison_targets": (
            "The entities being compared before one is selected.",
            "comparison_subject, comparison_tracking, selected_target",
            "These remain separate until the comparison evidence resolves the winner.",
        ),
        "conditional_branch_subject": (
            "The entity whose trait or property determines which branch should be chosen.",
            "conditional_branch_evidence_target, conditional_branch_targets, evidence_subject",
            "This is the thing being tested before the architecture commits to a branch target.",
        ),
        "conditional_branch_evidence_target": (
            "The evidence-bearing object or phenomenon used to resolve a conditional branch.",
            "conditional_branch_subject, evidence_target, conditional_branch_tracking",
            "This keeps branch reasoning tied to what must actually be observed.",
        ),
        "conditional_branch_targets": (
            "The possible task outcomes that become available once branch evidence resolves.",
            "conditional_branch_tracking, selected_branch, branch_ready",
            "These remain inactive until the evidence target resolves the branch.",
        ),
        "conditional_branches": (
            "The parsed mapping from evidence condition to branch outcome.",
            "conditional_branch_targets, evidence_target, destination_container",
            "This is the explicit branch logic extracted from the task.",
        ),
        "destination_container": (
            "A container destination the task wants the target placed into after evidence or search is resolved.",
            "supporting_targets, conditional_branch_targets, candidate_tracking",
            "This is a support role, not normally a primary focus target.",
        ),
        "destination_room": (
            "A room destination the task wants the target moved to or verified in.",
            "supporting_targets, remote_room_signal, candidate_tracking",
            "This stays a support location unless the task explicitly makes it the main target.",
        ),
        "required_relations": (
            "Named relations or connectors the task implies, such as circuit or anode/cathode links.",
            "relation_mechanism_task, relation_frontier",
            "These constrain relation-building to the intended mechanism family.",
        ),
        "desired_transformation": (
            "The state or process change the target substance or object should undergo.",
            "state_change_task, transformation_direction, target_substances",
            "This is the transformation goal that guides probe and verification phases.",
        ),
        "transformation_direction": (
            "The directional hint for a state change, such as heat up, cool down, melt, or freeze.",
            "desired_transformation, state_change_task, measurement_tracking",
            "This biases which tools, devices, or interventions look plausible.",
        ),
        "candidate_tracking": (
            "Episode-local memory of which candidate object is currently being pursued.",
            "candidate_classes, destination_container, destination_room",
            "Used for search-and-place tasks so the agent can reacquire or pivot cleanly.",
        ),
        "ordered_target_progress": (
            "Episode-local progress tracker for ordered focus tasks.",
            "lifecycle_sequence, growth_task, primary_targets",
            "This tracks what ordered stage or target has already been resolved.",
        ),
        "substance_search": (
            "State-change search controller for finding and probing the target substance.",
            "target_substances, desired_transformation, source_candidates",
            "It separates locating the substance from changing or verifying its state.",
        ),
        "artifact_creation": (
            "Creation-task controller for building a target artifact from ingredients or reagents.",
            "artifact_creation_task, artifact_type, grounded_artifacts",
            "It keeps artifact type distinct from distracting descriptors like color.",
        ),
        "measurement_tracking": (
            "Measurement-task controller for instrument search, target measurement, and branch gating.",
            "measurement_property, measurement_target, branch_ready",
            "It distinguishes raw measurements from the task property that must be inferred.",
        ),
        "comparison_tracking": (
            "Comparison-task controller for evidence gathering across multiple targets.",
            "comparison_subject, comparison_targets, selected_target",
            "It preserves compared subjects separately from the final target to act on.",
        ),
        "conditional_branch_tracking": (
            "Branch controller for tasks that require evidence before committing to one outcome.",
            "conditional_branch_subject, evidence_target, selected_branch, branch_ready",
            "It keeps the branch target inactive until evidence resolves the branch.",
        ),
        "evidence_target": (
            "The object or phenomenon the agent must observe to resolve a branch or comparison.",
            "conditional_branch_evidence_target, comparison_subject, branch_ready",
            "The architecture should gather evidence here before committing downstream actions.",
        ),
        "evidence_subject": (
            "Human-readable alias for the evidence-bearing target in a branching or comparison task.",
            "conditional_branch_subject, evidence_target",
            "If this appears, read it as the thing whose observed property decides the next step.",
        ),
        "relation_frontier": (
            "Local mechanism graph for relation-building tasks such as circuits or support structures.",
            "relation_mechanism_task, required_relations, control_candidates",
            "It prunes the enormous relation space down to the currently grounded mechanism.",
        ),
        "remote_room_signal": (
            "Memory that a remote inspection revealed task-relevant evidence in another room.",
            "destination_room, inferred_search_mode, candidate_tracking",
            "Used to bias travel/opening decisions without confusing remote evidence with current location.",
        ),
        "referent_resolution": (
            "Record of how a suggested action was canonicalized or ambiguously mapped to an environment referent.",
            "requested_action, canonicalized_action",
            "This prevents the architecture from silently learning the wrong object identity.",
        ),
        "branch_ready": (
            "Whether the evidence required to commit to a branch has actually been resolved.",
            "conditional_branch_tracking, measurement_tracking, selected_branch",
            "If false, branch-target actions should stay suppressed.",
        ),
        "selected_branch": (
            "The currently resolved branch target after evidence has been interpreted.",
            "conditional_branch_targets, branch_ready, destination_container",
            "This is the branch the agent should now commit to.",
        ),
        "selected_target": (
            "The currently resolved winner of a comparison task.",
            "comparison_targets, comparison_tracking",
            "This is the target that comparison evidence selected.",
        ),
        "source_candidates": (
            "Objects or fixtures that look like plausible sources for a substance or state-change interaction.",
            "substance_search, target_substances, desired_transformation",
            "These are the next likely places to probe when the target substance is still missing.",
        ),
        "grounded_substances": (
            "Substance labels explicitly grounded in observations during the current episode.",
            "target_substances, substance_search",
            "These prevent the agent from inventing ungrounded substance names.",
        ),
        "grounded_artifacts": (
            "Artifact labels explicitly grounded in observations during the current episode.",
            "artifact_type, artifact_creation",
            "These anchor creation logic to real observed objects instead of lexical lookalikes.",
        ),
        "control_candidates": (
            "Objects in a local mechanism that might mediate activation or state changes, such as switches or powered devices.",
            "relation_frontier, measurement_tracking",
            "These become important after a direct mechanism setup still fails to produce the target effect.",
        ),
    }
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
    _RUNTIME_PERCEPT_KEYS: tuple[tuple[str, int], ...] = (
        ("ordered_target_progress", 3),
        ("candidate_tracking", 3),
        ("substance_search", 3),
        ("artifact_creation", 3),
        ("measurement_tracking", 3),
        ("comparison_tracking", 3),
        ("conditional_branch_tracking", 3),
        ("relation_frontier", 3),
        ("remote_room_signal", 3),
        ("episode_hypothesis_ledger", 2),
        ("referent_resolution", 3),
        ("ambiguity_resolution", 3),
    )
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
    _GROWTH_TASK_HINTS = (
        "grow",
        "growing",
        "grown",
        "pollinate",
        "pollinated",
        "crosspollinate",
        "crosspollinated",
        "cross pollinate",
        "cross pollinated",
        "produce fruit",
        "bear fruit",
        "bearing fruit",
    )
    _GROWTH_TASK_TARGET_HINTS = (
        "seed",
        "germinating",
        "seedling",
        "sapling",
        "flower",
        "blossom",
        "pollen",
        "fruit",
        "grown",
    )
    _GENETIC_TRAIT_HINTS = (
        "dominant",
        "recessive",
        "trait",
        "inherit",
        "inherited",
        "inheritance",
        "phenotype",
        "genotype",
    )
    _GROWTH_EVIDENCE_TARGET_TOKENS = {
        "plant",
        "flower",
        "fruit",
        "tree",
        "sapling",
        "seedling",
        "vine",
        "crop",
    }
    _GROWTH_PRECURSOR_TOKENS = {
        "seed",
        "pollen",
        "spore",
        "fruit",
        "flower",
        "blossom",
    }
    _RECIPE_DIRECTIONS_RE = re.compile(
        r"^-\s+(chop|slice|dice|grill|fry|roast|bake|cook|heat|boil|steam)\b",
        re.IGNORECASE | re.MULTILINE,
    )
    _RECIPE_INGREDIENTS_RE = re.compile(
        r"^-\s+(?:a\s+|an\s+|the\s+)?(.+?)$",
        re.IGNORECASE | re.MULTILINE,
    )
    _RECIPE_SECTION_RE = re.compile(
        r"ingredients?.*?:(.*?)(?:directions?|steps?).*?:(.*?)(?:\n\n|\Z)",
        re.IGNORECASE | re.DOTALL,
    )

    _TASK_ROLE_PATTERNS = {
        "primary_targets": (
            r"\bfocus on\s+(.+?)(?=$|[.;,\n]|\bthen\b|\band then\b)",
            r"\bturn on\s+(.+?)(?=$|[.;,\n]|\bby\b|\busing\b|\bwith\b|\bthen\b|\band then\b)",
            r"\bactivate\s+(.+?)(?=$|[.;,\n]|\bby\b|\busing\b|\bwith\b|\bthen\b|\band then\b)",
            r"\bmove\s+(.+?)\s+\bto\b",
            r"\b(?:put|place)\s+(.+?)\s+\b(?:in|into|on)\b",
            r"\b(?:check|read|examine)\s+(.+?)(?=$|[.;,\n]|\bfor\b|\bin\b|\bthen\b|\band then\b)",
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
    _THERMAL_CONTROL_DIRECTION_HINTS = {
        "warm": {
            "boiler",
            "burner",
            "furnace",
            "heater",
            "hotplate",
            "kiln",
            "microwave",
            "oven",
            "stove",
        },
        "cool": {
            "chiller",
            "cooler",
            "freezer",
            "fridge",
            "refrigerator",
        },
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
    _ARTIFACT_CREATION_GENERIC_TOOL_PREAMBLES = (
        "use chemistry to create",
        "use chemistry to make",
        "use chemistry to produce",
        "use chemistry to synthesize",
    )
    _CONDITIONAL_BRANCH_TASK_STOPWORDS = {
        "answer",
        "branch",
        "condition",
        "conditions",
        "outcome",
        "property",
        "result",
        "trait",
        "value",
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
    _CONTROL_COMPONENT_HINTS = {
        "button",
        "dial",
        "knob",
        "lever",
        "switch",
        "toggle",
        "trigger",
    }
    _ABSTRACT_SUPPORT_ROLE_HINTS = {
        "electric",
        "electrical",
        "electricity",
        "energy",
        "power",
        "renewable",
        "source",
        "supply",
    }
    _POWER_SOURCE_HINTS = {
        "battery",
        "cell",
        "charger",
        "dynamo",
        "generator",
        "outlet",
        "panel",
        "photovoltaic",
        "reactor",
        "socket",
        "solar",
        "supply",
        "turbine",
        "wind",
    }
    _RENEWABLE_SOURCE_HINTS = {
        "hydro",
        "photovoltaic",
        "renewable",
        "solar",
        "sun",
        "sunlight",
        "turbine",
        "water",
        "waterwheel",
        "wind",
    }
    _NON_RENEWABLE_SOURCE_HINTS = {
        "coal",
        "diesel",
        "fossil",
        "fuel",
        "gas",
        "petrol",
    }
    _POWER_SINK_HINTS = {
        "anode",
        "bulb",
        "buzzer",
        "cathode",
        "component",
        "light",
        "motor",
        "terminal",
        "wire",
    }
    _RELATION_BRIDGE_HINTS = {
        "adapter",
        "cable",
        "connector",
        "cord",
        "lead",
        "terminal",
        "wire",
    }
    _FRONTIER_GENERIC_TOKENS = {
        "anode",
        "bulb",
        "cathode",
        "circuit",
        "electrical",
        "light",
        "panel",
        "power",
        "source",
        "terminal",
        "wire",
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
        "studio",
        "door",
        "doors",
    }
    _GENERIC_LOCATION_TOKENS = {
        "area",
        "hall",
        "hallway",
        "inside",
        "location",
        "outside",
        "place",
        "room",
        "space",
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
                r"\bflower(?!\s+pot\b)(?:ing)?\b",
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
        r"no known action matches(?: that input)?|"
        r"can't|cannot|failed|nothing happens|nothing is burning|not movable|"
        r"do not see|don't see|no such)",
        re.IGNORECASE,
    )
    _DIRECT_EFFECT_RE = re.compile(
        r"(heats up|cools down|produces|transforms?|changes? state|"
        r"temperature (?:of|reads|measures)|opens?|closes?|activates?|deactivates?|"
        r"moves?|picked up|put down|mixed?|created|appears?|disappears?|"
        r"filled?|emptied?|turned (?:on|off)|boils?|freezes?|burns?|"
        r"roasted?|grilled?|fried|baked|chopped?|sliced?|diced?|cooked?|"
        r"cut|prepared?|seasoned?|dropped?|taken|removed?)",
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
        max_chat_round=75,
        max_actions=15,
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
        self._measurement_property_event_observed = False
        self._comparison_observations: dict[str, dict] = {}
        self._selected_comparison_target: str | None = None
        self._comparison_resolution: dict[str, object] = {}
        self._selected_conditional_branch_target: str | None = None
        self._conditional_branch_resolution: dict[str, object] = {}
        self._containment_by_object: dict[str, str] = {}
        self._invalid_exact_actions: dict[str, int] = {}
        self._invalid_referent_attempts: dict[tuple[str, str], int] = {}
        self._relation_frontier_referents: list[str] = []
        self._remote_room_signals: dict[str, dict] = {}
        self._search_location_states: dict[str, dict] = {}
        self._target_status_by_referent: dict[str, str] = {}
        self._admissible_summary_cache: dict[tuple[tuple[str, ...], int], dict] = {}
        self._analyst_trace_entries: list[dict] = []
        self._last_analyst_trace_text = ""
        self._last_analyst_trace_ansi_text = ""
        self._analyst_trace_message_cursor = 0

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
        explicit_search_mode = any(
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
        measurement_property_type = self._get_measurement_property_type(
            measurement_property
        )
        (
            comparison_subject,
            comparison_property,
            comparison_direction,
            comparison_targets,
        ) = self._extract_comparison_contract(task)
        role_phrases = self._extract_task_role_phrases(task)
        candidate_classes = self._extract_candidate_classes(task)
        (
            conditional_branch_subject,
            conditional_branch_evidence_target,
            conditional_branch_targets,
            conditional_branches,
        ) = self._extract_conditional_branch_contract(task)
        (
            artifact_type,
            artifact_intermediate_targets,
            artifact_final_targets,
            artifact_descriptor_tokens,
        ) = self._extract_artifact_creation_contract(task, role_phrases)
        artifact_creation_task = bool(
            artifact_type and not state_change_task and not measurement_task
        )
        comparison_task = bool(
            comparison_property
            and len(comparison_targets) >= 2
            and not state_change_task
            and not artifact_creation_task
            and not measurement_task
        )
        conditional_branch_task = bool(
            conditional_branch_targets
            and conditional_branches
            and not state_change_task
            and not artifact_creation_task
            and not measurement_task
            and not comparison_task
        )
        growth_from_conditional_branch = (
            self._conditional_branch_requires_growth_precursor(
                task_lower=task_lower,
                conditional_branch_task=conditional_branch_task,
                conditional_branch_subject=conditional_branch_subject,
                conditional_branch_evidence_target=conditional_branch_evidence_target,
            )
        )
        if not conditional_branch_task:
            conditional_branch_subject = ""
            conditional_branch_evidence_target = ""
            conditional_branch_targets = []
            conditional_branches = []
        growth_task = bool(
            (
                any(
                    self._task_contains_hint(task_lower, hint)
                    for hint in self._GROWTH_TASK_HINTS
                )
                or growth_from_conditional_branch
            )
            and not lifecycle_sequence
            and not state_change_task
            and not artifact_creation_task
            and not measurement_task
        )
        inferred_search_mode = bool(
            role_phrases["primary_targets"]
            and not explicit_search_mode
            and not candidate_classes
            and not lifecycle_sequence
            and not growth_task
            and not state_change_task
            and not artifact_creation_task
            and not measurement_task
            and not comparison_task
            and not conditional_branch_task
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
        generic_artifact_use_preamble_only = (
            artifact_creation_task
            and self._artifact_creation_preamble_uses_generic_tool_hint_only(task_lower)
        )
        for family, hints in self._TASK_FAMILY_HINTS.items():
            matched_hints = [
                hint
                for hint in hints
                if self._task_contains_hint(task_lower, hint)
                and not (
                    family == "tool_application"
                    and hint == "use"
                    and generic_artifact_use_preamble_only
                )
            ]
            if matched_hints:
                required_families.append(family)
                for hint in matched_hints:
                    family_hint_tokens.update(
                        self._extract_runtime_tokens(
                            hint,
                            stopwords=self._TASK_STOPWORDS
                            | self._ACTION_COMMAND_STOPWORDS,
                        )
                    )

        support_families: list[str] = []
        if (
            explicit_search_mode
            or inferred_search_mode
            or growth_task
            or state_change_task
            or comparison_task
            or conditional_branch_task
        ) and "inspect" not in required_families:
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
        if comparison_task:
            role_phrases["primary_targets"] = (
                [comparison_subject]
                if comparison_subject
                else role_phrases["primary_targets"]
            )
            role_phrases["supporting_targets"] = [
                phrase
                for phrase in role_phrases["supporting_targets"]
                if phrase not in comparison_targets
            ]
        if conditional_branch_task:
            role_phrases["primary_targets"] = []
            if conditional_branch_evidence_target:
                role_phrases["primary_targets"].append(
                    conditional_branch_evidence_target
                )
            if growth_task:
                if (
                    conditional_branch_subject
                    and conditional_branch_subject
                    not in role_phrases["supporting_targets"]
                ):
                    role_phrases["supporting_targets"].append(
                        conditional_branch_subject
                    )
            elif (
                conditional_branch_subject
                and conditional_branch_subject not in role_phrases["primary_targets"]
            ):
                role_phrases["primary_targets"].append(conditional_branch_subject)
            role_phrases["supporting_targets"] = [
                phrase
                for phrase in role_phrases["supporting_targets"]
                if phrase not in conditional_branch_targets
            ]
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
        destination_container, destination_room = self._extract_destination_roles(task)
        if conditional_branch_task:
            if destination_container in conditional_branch_targets:
                destination_container = ""
            if destination_room in conditional_branch_targets:
                destination_room = ""
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
        relation_mechanism_task = bool(
            role_phrases["required_relations"]
            and role_phrases["primary_targets"]
            and role_phrases["supporting_targets"]
            and not candidate_classes
            and not lifecycle_sequence
            and not state_change_task
            and not artifact_creation_task
            and not measurement_task
            and not comparison_task
            and not conditional_branch_task
        )
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
            "comparison_subject",
            "comparison_targets",
            "conditional_branch_subject",
            "conditional_branch_evidence_target",
            "conditional_branch_targets",
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
            elif role == "comparison_subject":
                phrases = [comparison_subject] if comparison_subject else []
            elif role == "comparison_targets":
                phrases = comparison_targets
            elif role == "conditional_branch_subject":
                phrases = (
                    [conditional_branch_subject] if conditional_branch_subject else []
                )
            elif role == "conditional_branch_evidence_target":
                phrases = (
                    [conditional_branch_evidence_target]
                    if conditional_branch_evidence_target
                    else []
                )
            elif role == "conditional_branch_targets":
                phrases = conditional_branch_targets
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
        if growth_task:
            for token in self._GROWTH_TASK_TARGET_HINTS:
                if token not in target_entities:
                    target_entities.append(token)

        return {
            "required_families": required_families,
            "support_families": support_families,
            "target_entities": target_entities,
            "ordering_cues": ordering_cues,
            "procedural_sequence": procedural_sequence,
            "lifecycle_sequence": lifecycle_sequence,
            "growth_task": growth_task,
            "state_change_task": state_change_task,
            "artifact_creation_task": artifact_creation_task,
            "measurement_task": measurement_task,
            "comparison_task": comparison_task,
            "search_mode": explicit_search_mode,
            "inferred_search_mode": inferred_search_mode,
            "relation_mechanism_task": relation_mechanism_task,
            "conditional_branch_task": conditional_branch_task,
            "candidate_classes": candidate_classes,
            "primary_targets": role_phrases["primary_targets"],
            "supporting_targets": role_phrases["supporting_targets"],
            "target_substances": target_substances,
            "artifact_type": [artifact_type] if artifact_type else [],
            "artifact_intermediate_targets": artifact_intermediate_targets,
            "artifact_final_targets": artifact_final_targets,
            "artifact_descriptor_tokens": artifact_descriptor_tokens,
            "measurement_property": measurement_property,
            "measurement_property_type": measurement_property_type,
            "measurement_target": [measurement_target] if measurement_target else [],
            "measurement_instrument": (
                [measurement_instrument] if measurement_instrument else []
            ),
            "measurement_branch_targets": measurement_branch_targets,
            "measurement_branches": measurement_branches,
            "comparison_subject": ([comparison_subject] if comparison_subject else []),
            "comparison_property": comparison_property,
            "comparison_direction": comparison_direction,
            "comparison_targets": comparison_targets,
            "conditional_branch_subject": (
                [conditional_branch_subject] if conditional_branch_subject else []
            ),
            "conditional_branch_evidence_target": (
                [conditional_branch_evidence_target]
                if conditional_branch_evidence_target
                else []
            ),
            "conditional_branch_targets": conditional_branch_targets,
            "conditional_branches": conditional_branches,
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

    def _artifact_creation_preamble_uses_generic_tool_hint_only(
        self, task_text: str
    ) -> bool:
        normalized_task = re.sub(r"\s+", " ", task_text.strip().lower())
        if not normalized_task:
            return False

        stripped_task = normalized_task
        matched_preamble = False
        for preamble in self._ARTIFACT_CREATION_GENERIC_TOOL_PREAMBLES:
            stripped_task, replacements = re.subn(
                rf"(?<![a-z0-9]){re.escape(preamble)}(?![a-z0-9])",
                " ",
                stripped_task,
            )
            if replacements:
                matched_preamble = True
        return matched_preamble and not self._task_contains_hint(stripped_task, "use")

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
        if self._selected_comparison_target:
            for token in self._extract_runtime_tokens(
                self._selected_comparison_target,
                stopwords=self._TASK_ENTITY_STOPWORDS | self._ACTION_COMMAND_STOPWORDS,
                limit=4,
            ):
                if token not in keywords:
                    keywords.append(token)
        if self._selected_conditional_branch_target:
            for token in self._extract_runtime_tokens(
                self._selected_conditional_branch_target,
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

    def _extract_comparison_tokens(
        self, text: str, *, limit: int | None = None
    ) -> list[str]:
        tokens: list[str] = []
        raw_tokens = re.findall(r"[a-z0-9]+", (text or "").lower())
        stopwords = self._TASK_ENTITY_STOPWORDS | self._ACTION_COMMAND_STOPWORDS
        for index, token in enumerate(raw_tokens):
            previous_token = raw_tokens[index - 1] if index > 0 else ""
            if token in stopwords:
                continue
            if (
                len(token) < 3
                and not token.isdigit()
                and previous_token not in {"material"}
            ):
                continue
            if token not in tokens:
                tokens.append(token)
            if limit is not None and len(tokens) >= limit:
                break
        return tokens

    def _normalize_comparison_target_phrase(self, phrase: str) -> str:
        normalized = self._normalize_runtime_text(phrase)
        if not normalized:
            return ""

        normalized = re.split(
            r"\b(?:that|which|who|where|because|since|until|while|so that)\b",
            normalized,
            maxsplit=1,
        )[0]
        normalized = re.sub(r"^(?:a|an|the)\s+", "", normalized)
        normalized = normalized.strip(" .,:;")
        tokens = self._extract_comparison_tokens(normalized, limit=6)
        return " ".join(tokens[:4])

    def _is_lifecycle_task(self, task_contract: dict | None = None) -> bool:
        contract = task_contract or self._get_task_contract()
        return bool(contract.get("lifecycle_sequence"))

    def _is_growth_task(self, task_contract: dict | None = None) -> bool:
        contract = task_contract or self._get_task_contract()
        return bool(contract.get("growth_task"))

    def _is_growth_conditional_branch_task(
        self, task_contract: dict | None = None
    ) -> bool:
        contract = task_contract or self._get_task_contract()
        return bool(
            contract.get("growth_task") and contract.get("conditional_branch_task")
        )

    def _is_comparison_task(self, task_contract: dict | None = None) -> bool:
        contract = task_contract or self._get_task_contract()
        return bool(contract.get("comparison_task"))

    def _extract_state_change_goal(self, task: str) -> tuple[list[str], str, str]:
        normalized_task = self._normalize_runtime_text(task).replace("a(n)", "an")
        desired_transformation = ""
        transformation_direction = ""
        target_substances: list[str] = []

        for verb, metadata in self._TASK_STATE_CHANGE_HINTS.items():
            if self._task_contains_hint(normalized_task, verb):
                desired_transformation = verb
                transformation_direction = metadata["direction"]
                break

        patterns: tuple[str, ...] = ()
        if desired_transformation:
            patterns = (
                rf"\b{re.escape(desired_transformation)}(?:\s+up|\s+down)?\s+(?:an?\s+|the\s+)?(.+?)(?=$|[.;,\n]|\bfirst\b|\bthen\b|\band then\b)",
            )
        else:
            generic_patterns: tuple[tuple[str, str], ...] = (
                (
                    r"\bchange\s+(?:the\s+)?state\s+of\s+matter\s+of\s+(?:an?\s+|the\s+)?(.+?)(?=$|[.;,\n]|\bfirst\b|\bthen\b|\band then\b)",
                    "change state of matter",
                ),
                (
                    r"\bchange\s+(?:the\s+)?phase\s+of\s+(?:an?\s+|the\s+)?(.+?)(?=$|[.;,\n]|\bfirst\b|\bthen\b|\band then\b)",
                    "change phase",
                ),
                (
                    r"\bcause\s+(?:an?\s+|the\s+)?(.+?)\s+to\s+change\s+(?:its|the)\s+state\s+of\s+matter(?=$|[.;,\n]|\bfirst\b|\bthen\b|\band then\b)",
                    "change state of matter",
                ),
                (
                    r"\bcause\s+(?:an?\s+|the\s+)?(.+?)\s+to\s+change\s+(?:its|the)\s+phase(?=$|[.;,\n]|\bfirst\b|\bthen\b|\band then\b)",
                    "change phase",
                ),
            )
            for pattern, transformation in generic_patterns:
                matches = list(re.finditer(pattern, normalized_task))
                if not matches:
                    continue
                desired_transformation = transformation
                patterns = (pattern,)
                break

        if not desired_transformation:
            return [], "", ""

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

    def _get_measurement_property_type(
        self, property_text: str | None = None, task_contract: dict | None = None
    ) -> str:
        if property_text is None:
            contract = task_contract or self._get_task_contract()
            property_text = contract.get("measurement_property", "")
        normalized = self._normalize_runtime_text(property_text or "")
        if not normalized:
            return ""
        if "point" in normalized:
            return "stable_threshold_property"
        return "instantaneous_state"

    def _get_measurement_property_direction(
        self, task_contract: dict | None = None
    ) -> str:
        contract = task_contract or self._get_task_contract()
        property_text = self._normalize_runtime_text(
            contract.get("measurement_property", "")
        )
        if not property_text:
            return ""
        for hint, metadata in self._TASK_STATE_CHANGE_HINTS.items():
            if hint in property_text:
                return metadata.get("direction", "")
        return ""

    def _is_measurement_thermal_control_signature(
        self, signature: str, *, direction: str
    ) -> bool:
        if not signature or not direction:
            return False
        hint_tokens = self._THERMAL_CONTROL_DIRECTION_HINTS.get(direction, set())
        return bool(self._referent_tokens(signature) & hint_tokens)

    def _get_measurement_control_candidate_signatures(
        self,
        actions: list[str] | None = None,
        task_contract: dict | None = None,
    ) -> list[str]:
        contract = task_contract or self._get_task_contract()
        if not self._is_measurement_task(contract):
            return []

        direction = self._get_measurement_property_direction(contract)
        measurement_target = self._get_measurement_target_signature(contract)
        if not direction or not measurement_target:
            return []

        active_actions = actions if actions is not None else self.admissible_actions
        move_destinations: set[str] = set()
        control_candidates: list[str] = []
        for action in active_actions:
            family = self._classify_action_family(action)
            if family in {"relocation", "transfer_or_transform"}:
                primary_signature = self._extract_action_primary_object_signature(
                    action, family=family
                )
                destination_signature = self._extract_action_destination_signature(
                    action, family=family
                )
                if primary_signature == measurement_target and destination_signature:
                    move_destinations.add(destination_signature)
                continue

            if family != "device_control" or self._action_mentions_door(
                action, family=family
            ):
                continue
            primary_signature = self._extract_action_primary_object_signature(
                action, family=family
            )
            if not self._is_measurement_thermal_control_signature(
                primary_signature, direction=direction
            ):
                continue
            if primary_signature not in control_candidates:
                control_candidates.append(primary_signature)

        return [
            candidate
            for candidate in control_candidates
            if candidate in move_destinations
        ][:4]

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

    def _extract_comparison_contract(
        self, task: str
    ) -> tuple[str, str, str, list[str]]:
        normalized_task = self._normalize_runtime_text(task).replace("a(n)", "an")
        patterns = (
            r"\b(?:determine|identify|find)\s+which\s+of\s+(?:the\s+)?(?:two\s+)?"
            r"(.+?)\s*\((.+?)\)\s+has\s+(?:the\s+)?"
            r"(most|least|highest|lowest)\s+(.+?)(?=$|[.;,\n])",
        )
        direction_map = {
            "most": "max",
            "highest": "max",
            "least": "min",
            "lowest": "min",
        }

        for pattern in patterns:
            match = re.search(pattern, normalized_task)
            if not match:
                continue
            subject = self._normalize_task_phrase(
                match.group(1), role="primary_targets"
            )
            candidate_block = match.group(2)
            property_name = self._normalize_measurement_phrase(match.group(4))
            direction = direction_map.get(match.group(3), "")
            raw_candidates = re.split(
                r"\s*,\s*|\s+\band\b\s+|\s+\bor\b\s+",
                candidate_block,
            )
            comparison_targets: list[str] = []
            for candidate in raw_candidates:
                normalized_candidate = self._normalize_comparison_target_phrase(
                    candidate
                )
                if (
                    normalized_candidate
                    and normalized_candidate not in comparison_targets
                ):
                    comparison_targets.append(normalized_candidate)
                if len(comparison_targets) >= 2:
                    break
            if subject and property_name and len(comparison_targets) >= 2 and direction:
                return subject, property_name, direction, comparison_targets[:2]

        return "", "", "", []

    def _normalize_conditional_branch_subject(self, phrase: str) -> str:
        if re.search(r"\bsubstance\b", phrase, flags=re.IGNORECASE):
            normalized_substance = self._normalize_measurement_target_phrase(phrase)
            if normalized_substance:
                return normalized_substance
        normalized = self._normalize_task_phrase(phrase, role="primary_targets")
        if not normalized:
            return ""
        tokens = normalized.split()
        if tokens and tokens[-1] in {"color", "pattern", "shape", "size", "trait"}:
            tokens = tokens[:-1]
        return " ".join(tokens[:4])

    def _normalize_conditional_branch_condition(self, phrase: str) -> str:
        normalized = self._normalize_runtime_text(phrase)
        if not normalized:
            return ""
        normalized = re.split(
            r"\b(?:focus on|then|and then|otherwise)\b",
            normalized,
            maxsplit=1,
        )[0]
        normalized = re.sub(r"^(?:if\s+)?", "", normalized)
        normalized = re.sub(
            r"^(?:the\s+)?(?:trait|property|value|condition)\s+is\s+", "", normalized
        )
        normalized = re.sub(r"^(?:it|this|that)\s+is\s+", "", normalized)
        normalized = normalized.strip(" .,:;")
        return normalized

    def _extract_conditional_branch_target(self, branch_action: str) -> tuple[str, str]:
        normalized_action = self._normalize_runtime_text(branch_action)
        if not normalized_action:
            return "", ""

        focus_match = re.match(r"^focus on\s+(?:the\s+)?(.+?)$", normalized_action)
        if focus_match:
            branch_target = self._normalize_task_phrase(
                focus_match.group(1), role="supporting_targets"
            )
            if branch_target:
                return branch_target, "focus"
            return "", ""

        destination_container, destination_room = self._extract_destination_roles(
            normalized_action
        )
        if destination_container:
            return destination_container, "destination"
        if destination_room:
            return destination_room, "destination"
        return "", ""

    def _extract_conditional_branch_condition_tokens(
        self,
        condition_phrase: str,
        *,
        branch_subject: str = "",
        branch_target: str = "",
        evidence_target: str = "",
    ) -> list[str]:
        raw_tokens = self._extract_runtime_tokens(
            condition_phrase,
            stopwords=self._TASK_STOPWORDS | self._ACTION_COMMAND_STOPWORDS,
            limit=5,
        )
        ignored_tokens = (
            set(
                self._extract_runtime_tokens(
                    branch_subject,
                    stopwords=self._TASK_STOPWORDS | self._ACTION_COMMAND_STOPWORDS,
                    limit=5,
                )
            )
            | set(
                self._extract_runtime_tokens(
                    branch_target,
                    stopwords=self._TASK_STOPWORDS | self._ACTION_COMMAND_STOPWORDS,
                    limit=4,
                )
            )
            | set(
                self._extract_runtime_tokens(
                    evidence_target,
                    stopwords=self._TASK_STOPWORDS | self._ACTION_COMMAND_STOPWORDS,
                    limit=4,
                )
            )
            | self._CONDITIONAL_BRANCH_TASK_STOPWORDS
        )
        filtered_tokens = [token for token in raw_tokens if token not in ignored_tokens]
        return (filtered_tokens or raw_tokens)[:3]

    def _extract_conditional_branch_contract(
        self, task: str
    ) -> tuple[str, str, list[str], list[dict]]:
        normalized_task = self._normalize_runtime_text(task).replace("a(n)", "an")
        conditional_branch_targets: list[str] = []
        conditional_branches: list[dict] = []

        branch_pattern = re.compile(r"(?:^|[.;\n])\s*if\s+(.+?),\s*(.+?)(?=$|[.;\n])")
        raw_branches: list[tuple[str, str, str]] = []
        for match in branch_pattern.finditer(normalized_task):
            branch_target, branch_mode = self._extract_conditional_branch_target(
                match.group(2)
            )
            if not branch_target or not branch_mode:
                continue
            raw_branches.append((match.group(1), branch_target, branch_mode))
            if branch_target not in conditional_branch_targets:
                conditional_branch_targets.append(branch_target)
            if len(conditional_branch_targets) >= 2:
                break

        if len(conditional_branch_targets) < 2:
            return "", "", [], []

        branch_subject = ""
        evidence_target = ""
        inference_patterns = (
            (
                r"\b(?:determine|identify)\s+(?:whether|if)\s+(?:the\s+)?(.+?)\s+is\s+.+?\s+or\s+.+?\s+in\s+(?:an?\s+|the\s+)?(.+?)(?=$|[.;,\n]|\bif\b)",
                1,
                2,
            ),
            (
                r"\b(?:determine|identify)\s+(?:whether|if)\s+(?:the\s+)?(.+?)\s+of\s+(?:an?\s+|the\s+)?(.+?)\s+is\s+.+?\s+or\s+.+?(?=$|[.;,\n]|\bif\b)",
                1,
                2,
            ),
            (
                r"\b(?:determine|identify)\s+(?:whether|if)\s+(?:an?\s+|the\s+)?(.+?)\s+is\s+.+?\s+or\s+.+?(?=$|[.;,\n]|\bif\b)",
                1,
                0,
            ),
            (
                r"\b(?:determine|identify)\s+(?:whether|if)\s+(?:an?\s+|the\s+)?(.+?)\s+is\s+.+?(?=$|[.;,\n]|\bif\b)",
                1,
                0,
            ),
        )
        for pattern, subject_index, target_index in inference_patterns:
            match = re.search(pattern, normalized_task)
            if not match:
                continue
            branch_subject = self._normalize_conditional_branch_subject(
                match.group(subject_index)
            )
            if target_index:
                evidence_target = self._normalize_task_phrase(
                    match.group(target_index), role="primary_targets"
                )
            break

        if not evidence_target:
            evidence_target = branch_subject

        for raw_condition, branch_target, branch_mode in raw_branches[:2]:
            condition = self._normalize_conditional_branch_condition(raw_condition)
            condition_tokens = self._extract_conditional_branch_condition_tokens(
                condition,
                branch_subject=branch_subject,
                branch_target=branch_target,
                evidence_target=evidence_target,
            )
            if not condition_tokens:
                continue
            conditional_branches.append(
                {
                    "condition": condition,
                    "condition_tokens": condition_tokens,
                    "mode": branch_mode,
                    "target": branch_target,
                }
            )

        if len(conditional_branches) < 2:
            return "", "", [], []

        return (
            branch_subject,
            evidence_target,
            conditional_branch_targets[:2],
            conditional_branches[:2],
        )

    def _conditional_branch_requires_growth_precursor(
        self,
        *,
        task_lower: str,
        conditional_branch_task: bool,
        conditional_branch_subject: str,
        conditional_branch_evidence_target: str,
    ) -> bool:
        if not conditional_branch_task:
            return False

        evidence_tokens = set(
            self._extract_runtime_tokens(
                conditional_branch_evidence_target,
                stopwords=self._TASK_ENTITY_STOPWORDS | self._ACTION_COMMAND_STOPWORDS,
                limit=6,
            )
        )
        subject_tokens = set(
            self._extract_runtime_tokens(
                conditional_branch_subject,
                stopwords=self._TASK_ENTITY_STOPWORDS | self._ACTION_COMMAND_STOPWORDS,
                limit=6,
            )
        )
        if not evidence_tokens:
            return False

        evidence_is_growth_target = bool(
            evidence_tokens & self._GROWTH_EVIDENCE_TARGET_TOKENS
        )
        subject_is_growth_precursor = bool(
            subject_tokens & self._GROWTH_PRECURSOR_TOKENS
        )
        genetics_signal = any(
            self._task_contains_hint(task_lower, hint)
            for hint in self._GENETIC_TRAIT_HINTS
        )
        return evidence_is_growth_target and (
            subject_is_growth_precursor or genetics_signal
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

    def _is_conditional_branch_task(self, task_contract: dict | None = None) -> bool:
        contract = task_contract or self._get_task_contract()
        return bool(contract.get("conditional_branch_task"))

    def _get_conditional_branch_metadata(
        self,
        *,
        target: str | None = None,
        task_contract: dict | None = None,
    ) -> dict:
        contract = task_contract or self._get_task_contract()
        branch_target = target or self._selected_conditional_branch_target or ""
        if not branch_target:
            return {}
        for branch in contract.get("conditional_branches", []):
            if branch.get("target") == branch_target:
                return branch
        return {}

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

    def _extract_grounded_artifact_mentions_from_action(self, action: str) -> list[str]:
        task_contract = self._get_task_contract()
        if not self._is_artifact_creation_task(task_contract):
            return []

        artifact_types = task_contract.get("artifact_type", [])
        if not artifact_types:
            return []

        artifact_type = artifact_types[0]
        normalized_action = self._normalize_runtime_text(action)
        if not normalized_action:
            return []

        grounded_labels: list[str] = []
        patterns = (
            rf"\b(?:focus on|look in|look at|move|pick up|take|grab|pour|dunk|mix|fill|empty|insert|place|use|heat|cool|boil|cook)\s+([a-z0-9][a-z0-9\s-]{{0,50}}?\b{re.escape(artifact_type)}\b)(?=$|\s+\b(?:to|into|in|on|with|using|from)\b|[).,;])",
            rf"\bcontaining\s+([a-z0-9][a-z0-9\s-]{{0,40}}?\b{re.escape(artifact_type)}\b)(?=$|\s+\b(?:to|into|in|on|with|using|from)\b|[).,;])",
            rf"\bfilled with\s+([a-z0-9][a-z0-9\s-]{{0,40}}?\b{re.escape(artifact_type)}\b)(?=$|\s+\b(?:to|into|in|on|with|using|from)\b|[).,;])",
        )
        for pattern in patterns:
            for match in re.finditer(pattern, normalized_action):
                label = self._normalize_grounded_artifact_label(match.group(1))
                if label and label not in grounded_labels:
                    grounded_labels.append(label)
        return grounded_labels[:6]

    def _record_grounded_artifact_labels(self, labels: list[str]) -> None:
        if not self._is_artifact_creation_task():
            return

        for label in labels:
            entry = self._grounded_artifacts.setdefault(
                label,
                {
                    "label": label,
                    "last_seen_timestep": self.num_actions_taken,
                },
            )
            entry["last_seen_timestep"] = self.num_actions_taken

    def _update_artifact_creation_tracking(
        self, observation: str, *, action: str | None = None
    ) -> None:
        if not self._is_artifact_creation_task():
            return

        self._record_grounded_artifact_labels(
            self._extract_grounded_artifact_mentions(observation)
        )
        if not action or action == "None":
            return

        normalized_observation = self._normalize_runtime_text(observation)
        if self._HARD_FAILURE_RE.search(observation) or "ambiguous request" in (
            normalized_observation
        ):
            return

        self._record_grounded_artifact_labels(
            self._extract_grounded_artifact_mentions_from_action(action)
        )

    def _update_artifact_creation_search_tracking(
        self, *, action: str | None, observation: str
    ) -> None:
        if not self._is_artifact_creation_task():
            return

        room_signature = self._get_current_location_signature(observation)
        target_grounded = bool(self._get_grounded_artifact_labels(limit=1))
        if room_signature:
            room_state = self._search_location_states.setdefault(
                room_signature,
                {
                    "local_exploration": 0,
                    "target_grounded": False,
                },
            )
            if target_grounded:
                room_state["target_grounded"] = True

        if not action or action == "None" or self._HARD_FAILURE_RE.search(observation):
            return
        if not room_signature or target_grounded:
            return

        family = self._classify_action_family(action)
        if family in {"inspect", "device_control"} and not (
            self._action_targets_room_frontier(action, observation, family=family)
        ):
            room_state["local_exploration"] += 1

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
    def _snapshot_has_signal(snapshot: dict, *, key: str | None = None) -> bool:
        if key is not None:
            return bool(snapshot.get(key))
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

    def _get_task_grounded_substance_labels(
        self,
        *,
        limit: int = 6,
        task_contract: dict | None = None,
    ) -> list[str]:
        contract = task_contract or self._get_task_contract()
        if not self._is_state_change_task(contract):
            return []

        role_token_sets = self._get_task_role_token_sets(contract)
        target_roles = (
            role_token_sets["target_substances"] or role_token_sets["primary_targets"]
        )
        if not target_roles:
            return self._get_grounded_substance_labels(limit=limit)

        compatible_labels: list[str] = []
        for label in self._get_grounded_substance_labels(limit=max(limit * 3, 12)):
            label_tokens = set(
                self._extract_runtime_tokens(
                    label,
                    stopwords=self._TASK_STOPWORDS | self._ACTION_COMMAND_STOPWORDS,
                )
            )
            if not label_tokens or not self._matches_any_role(
                label_tokens, target_roles
            ):
                continue
            compatible_labels.append(label)
            if len(compatible_labels) >= limit:
                break
        return compatible_labels

    def _get_task_grounded_substance_token_sets(
        self,
        *,
        limit: int = 6,
        task_contract: dict | None = None,
    ) -> list[set[str]]:
        token_sets: list[set[str]] = []
        for label in self._get_task_grounded_substance_labels(
            limit=limit, task_contract=task_contract
        ):
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
        task_contract = self._get_task_contract()
        grounded_tokens = set(self._get_observation_grounded_tokens())
        room_signature = self._get_current_location_signature(observation)
        target_grounded = self._state_change_target_is_grounded(
            task_contract, grounded_tokens
        )
        if room_signature:
            room_state = self._search_location_states.setdefault(
                room_signature,
                {
                    "local_exploration": 0,
                    "target_grounded": False,
                },
            )
            if target_grounded:
                room_state["target_grounded"] = True
        if not action or action == "None" or self._HARD_FAILURE_RE.search(observation):
            return

        family = self._classify_action_family(action)
        if room_signature and not target_grounded:
            room_state = self._search_location_states.setdefault(
                room_signature,
                {
                    "local_exploration": 0,
                    "target_grounded": False,
                },
            )
            if family in {"inspect", "device_control"} and not (
                self._action_targets_room_frontier(action, observation, family=family)
            ):
                room_state["local_exploration"] += 1
        if family not in {"inspect", "device_control"}:
            return

        if not self._is_container_like_action(action, family=family):
            return

        if target_grounded:
            return

        if not self._observation_confirms_container_probe(
            action, observation, family=family
        ):
            return

        referent = self._get_action_referent_signature(action, family=family)
        if referent and referent not in self._exhausted_container_targets:
            self._exhausted_container_targets.append(referent)
            self._exhausted_container_targets = self._exhausted_container_targets[-8:]

    def _observation_confirms_container_probe(
        self,
        action: str,
        observation: str,
        *,
        family: str | None = None,
    ) -> bool:
        family = family or self._classify_action_family(action)
        if family not in {"inspect", "device_control"}:
            return False
        if not self._is_container_like_action(action, family=family):
            return False

        normalized_action = self._normalize_runtime_text(action)
        if family == "device_control" and not normalized_action.startswith("open "):
            return False

        referent = self._get_action_referent_signature(action, family=family)
        normalized_observation = self._normalize_runtime_text(observation)
        if not referent or not normalized_observation:
            return False

        referent_pattern = re.escape(referent)
        if re.search(
            rf"\b(?:the )?{referent_pattern}\b[^.\n]*\bclosed\b",
            normalized_observation,
        ):
            return False

        if family == "inspect" and re.search(
            rf"\b(?:the )?{referent_pattern}\b[^.\n]*\bopen\b",
            normalized_observation,
        ):
            return True

        reveal_patterns = (
            rf"\binside (?:the )?{referent_pattern}\b",
            rf"\bin (?:the )?{referent_pattern}\b",
            rf"\b(?:the )?{referent_pattern}\b[^.\n]*\bcontains?\b",
            rf"\b(?:the )?{referent_pattern}\b[^.\n]*\bcontaining\b",
        )
        return any(
            re.search(pattern, normalized_observation) for pattern in reveal_patterns
        )

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
        current_observation = (self.percept or {}).get("resulting_observation", "")
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
            if self._signature_looks_structural_noncandidate(
                referent, observation=current_observation
            ):
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
            "grounded_substances": self._get_task_grounded_substance_labels(limit=4),
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

        destination_matches = list(
            re.finditer(
                r"\b(?:to|into|in|on)\b\s+([a-z0-9][a-z0-9\s-]{0,60}?)(?=\s+\b(?:to|into|in|on)\b|$)",
                normalized,
            )
        )
        if not destination_matches:
            return ""
        destination_tokens = self._extract_runtime_tokens(
            destination_matches[-1].group(1),
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
        return (
            self._get_measurement_property_type(task_contract=contract)
            == "stable_threshold_property"
        )

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

    def _measurement_event_relates_to_target(
        self, *, action: str | None, observation: str, task_contract: dict | None = None
    ) -> bool:
        contract = task_contract or self._get_task_contract()
        measurement_target = self._get_measurement_target_signature(contract)
        active_enclosure = self._resolve_enclosing_referent(measurement_target)
        target_tokens = self._referent_tokens(measurement_target)
        observation_tokens = set(self._extract_runtime_tokens(observation, limit=48))
        if target_tokens and target_tokens.issubset(observation_tokens):
            return True
        if not action:
            return False
        family = self._classify_action_family(action)
        touched_signatures = {
            self._get_action_referent_signature(action, family=family),
            self._extract_action_primary_object_signature(action, family=family),
            self._extract_action_destination_signature(action, family=family),
            self._extract_measurement_subject_signature(action, family=family),
        }
        touched_signatures.discard("")
        if measurement_target and measurement_target in touched_signatures:
            return True
        return bool(active_enclosure and active_enclosure in touched_signatures)

    def _measurement_property_is_resolved(
        self, task_contract: dict | None = None
    ) -> bool:
        contract = task_contract or self._get_task_contract()
        latest_direct = self._get_latest_measurement(direct=True)
        if latest_direct is None:
            return False
        if not self._measurement_property_requires_event(contract):
            return True
        return any(
            entry["direct"] and entry.get("property_relevant")
            for entry in self._measurement_observations
        )

    def _get_measurement_property_resolution_status(
        self, task_contract: dict | None = None
    ) -> str:
        contract = task_contract or self._get_task_contract()
        if not self._is_measurement_task(contract):
            return ""
        if self._measurement_property_is_resolved(contract):
            return "resolved"
        if self._get_latest_measurement(direct=True) is None:
            return "needs_direct_measurement"
        if self._measurement_property_requires_event(contract):
            if self._measurement_property_event_observed:
                return "needs_confirming_measurement"
            return "needs_transition_evidence"
        return "needs_direct_measurement"

    def _update_measurement_search_tracking(
        self, *, action: str | None, observation: str
    ) -> None:
        if not self._is_measurement_task():
            return
        task_contract = self._get_task_contract()
        role_token_sets = self._get_task_role_token_sets(task_contract)
        grounded_tokens = set(self._get_observation_grounded_tokens())
        instrument_grounded = self._role_is_grounded(
            grounded_tokens, role_token_sets["measurement_instrument"]
        )

        previous_observation = ""
        if self.curr_episodic_memory:
            try:
                previous_observation = json.loads(self.curr_episodic_memory[-1]).get(
                    "resulting_observation", ""
                )
            except (TypeError, ValueError, json.JSONDecodeError):
                previous_observation = ""

        room_signature = self._get_current_location_signature(
            observation
        ) or self._get_current_location_signature(previous_observation)
        if room_signature:
            room_state = self._search_location_states.setdefault(
                room_signature,
                {
                    "local_exploration": 0,
                    "target_grounded": False,
                },
            )
            if instrument_grounded:
                room_state["target_grounded"] = True

        if not action or action == "None" or self._HARD_FAILURE_RE.search(observation):
            return
        if not room_signature or instrument_grounded:
            return

        family = self._classify_action_family(action)
        if family == "inspect":
            if self._action_targets_room_frontier(action, observation, family=family):
                return
            room_state["local_exploration"] += 1
        elif family == "device_control":
            if self._action_targets_room_frontier(action, observation, family=family):
                return
            if self._action_mentions_door(action, family=family):
                return
            room_state["local_exploration"] += 1

    def _update_growth_search_tracking(
        self, *, action: str | None, observation: str
    ) -> None:
        task_contract = self._get_task_contract()
        if not self._is_growth_task(task_contract):
            return

        grounded_tokens = set(self._extract_runtime_tokens(observation, limit=48))
        primary_target_grounded = self._primary_target_is_grounded(
            task_contract, grounded_tokens
        )

        previous_observation = ""
        if self.curr_episodic_memory:
            try:
                previous_observation = json.loads(self.curr_episodic_memory[-1]).get(
                    "resulting_observation", ""
                )
            except (TypeError, ValueError, json.JSONDecodeError):
                previous_observation = ""

        room_signature = self._get_current_location_signature(
            observation
        ) or self._get_current_location_signature(previous_observation)
        if room_signature:
            room_state = self._search_location_states.setdefault(
                room_signature,
                {
                    "local_exploration": 0,
                    "target_grounded": False,
                },
            )
            if primary_target_grounded:
                room_state["target_grounded"] = True

        if not action or action == "None" or self._HARD_FAILURE_RE.search(observation):
            return
        if not room_signature or primary_target_grounded:
            return

        family = self._classify_action_family(action)
        if family == "inspect":
            if self._action_targets_room_frontier(action, observation, family=family):
                return
            room_state["local_exploration"] += 1
        elif family == "device_control":
            if self._action_targets_room_frontier(action, observation, family=family):
                return
            if self._action_mentions_door(action, family=family):
                return
            room_state["local_exploration"] += 1

    def _update_measurement_tracking(
        self, *, action: str | None, observation: str
    ) -> None:
        self._update_containment_tracking(action, observation)
        if not self._is_measurement_task():
            return
        task_contract = self._get_task_contract()
        if (
            self._measurement_property_requires_event(task_contract)
            and self._measurement_property_event_detected(observation, task_contract)
            and self._measurement_event_relates_to_target(
                action=action, observation=observation, task_contract=task_contract
            )
        ):
            self._measurement_property_event_observed = True
        if not action or action == "None":
            return

        family = self._classify_action_family(action)
        measurement_value, measurement_unit = self._parse_numeric_measurement(
            observation
        )
        if measurement_value is None or family != "tool_application":
            return

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
        event_confirmed = bool(
            self._measurement_property_event_detected(observation, task_contract)
            and self._measurement_event_relates_to_target(
                action=action, observation=observation, task_contract=task_contract
            )
        )
        property_relevant = bool(
            direct_measurement
            and (
                not self._measurement_property_requires_event(task_contract)
                or event_confirmed
                or self._measurement_property_event_observed
            )
        )
        measurement_entry = {
            "value": measurement_value,
            "unit": measurement_unit,
            "subject": subject_signature,
            "direct": direct_measurement,
            "proxy": proxy_measurement,
            "event_confirmed": event_confirmed,
            "property_relevant": property_relevant,
            "timestep": self.num_actions_taken,
        }
        self._measurement_observations.append(measurement_entry)
        self._measurement_observations = self._measurement_observations[-8:]

        if not direct_measurement:
            return
        if self._selected_measurement_branch_target:
            return
        if not property_relevant:
            return

        for branch in task_contract.get("measurement_branches", []):
            operator = branch["operator"]
            threshold = branch["threshold"]
            if operator == "above" and measurement_value > threshold:
                self._selected_measurement_branch_target = branch["target"]
                break
            if operator == "below" and measurement_value < threshold:
                self._selected_measurement_branch_target = branch["target"]
                break

    def _get_latest_measurement(
        self,
        *,
        direct: bool | None = None,
        property_relevant: bool | None = None,
    ) -> dict | None:
        for entry in reversed(self._measurement_observations):
            if direct is not None and entry["direct"] is not direct:
                continue
            if (
                property_relevant is not None
                and entry.get("property_relevant") is not property_relevant
            ):
                continue
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
            "measurement_property_type": task_contract.get(
                "measurement_property_type", ""
            ),
            "property_resolution_status": self._get_measurement_property_resolution_status(
                task_contract
            ),
            "branch_target": (
                [self._selected_measurement_branch_target]
                if self._selected_measurement_branch_target
                else []
            ),
        }
        latest_direct = self._get_latest_measurement(direct=True)
        latest_proxy = self._get_latest_measurement(direct=False)
        latest_property = self._get_latest_measurement(
            direct=True, property_relevant=True
        )
        if latest_direct:
            snapshot["latest_direct_measurement"] = {
                "value": latest_direct["value"],
                "unit": latest_direct["unit"],
                "subject": latest_direct["subject"],
            }
        if latest_property:
            snapshot["latest_property_measurement"] = {
                "value": latest_property["value"],
                "unit": latest_property["unit"],
                "subject": latest_property["subject"],
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
        if self._measurement_property_requires_event(task_contract):
            snapshot["transition_observed"] = self._measurement_property_event_observed
        if task_contract.get("measurement_branch_targets"):
            snapshot["branch_ready"] = bool(self._selected_measurement_branch_target)
        return snapshot

    def _extract_comparison_primary_signature(self, action: str) -> str:
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
        primary_tokens = self._extract_comparison_tokens(primary_segment, limit=6)
        return " ".join(primary_tokens[:6])

    def _extract_comparison_destination_signature(self, action: str) -> str:
        normalized = self._normalize_runtime_text(action)
        if not normalized:
            return ""

        destination_matches = list(
            re.finditer(
                r"\b(?:to|into|in|on)\b\s+([a-z0-9][a-z0-9\s-]{0,60}?)(?=\s+\b(?:to|into|in|on)\b|$)",
                normalized,
            )
        )
        if not destination_matches:
            return ""
        destination_tokens = self._extract_comparison_tokens(
            destination_matches[-1].group(1),
            limit=6,
        )
        return " ".join(destination_tokens[:6])

    def _get_comparison_target_signatures(
        self, task_contract: dict | None = None
    ) -> list[str]:
        contract = task_contract or self._get_task_contract()
        signatures: list[str] = []
        for target in contract.get("comparison_targets", []):
            signature = self._normalize_comparison_target_phrase(target)
            if signature and signature not in signatures:
                signatures.append(signature)
        return signatures

    def _resolve_comparison_target_from_context(
        self,
        *,
        action: str | None,
        observation: str,
        task_contract: dict | None = None,
    ) -> str:
        contract = task_contract or self._get_task_contract()
        comparison_targets = contract.get("comparison_targets", [])
        if not comparison_targets:
            return ""

        action_tokens: set[str] = set()
        if action and action != "None":
            action_tokens = set(self._extract_comparison_tokens(action, limit=24))

        observation_tokens = set(self._extract_comparison_tokens(observation, limit=36))
        matches: list[str] = []
        for target in comparison_targets:
            target_tokens = set(self._extract_comparison_tokens(target, limit=6))
            if not target_tokens:
                continue
            if action_tokens and target_tokens.issubset(action_tokens):
                matches.append(target)
                continue
            if target_tokens.issubset(observation_tokens):
                matches.append(target)

        if len(matches) == 1:
            return matches[0]
        return ""

    def _parse_comparison_observation(
        self, observation: str, *, task_contract: dict | None = None
    ) -> tuple[float | None, str]:
        contract = task_contract or self._get_task_contract()
        property_text = self._normalize_runtime_text(
            contract.get("comparison_property", "")
        )
        observation_lower = self._normalize_runtime_text(observation)

        if "friction" in property_text:
            match = re.search(
                r"\bapproximately\s+(-?\d+(?:\.\d+)?)%\s+down\s+the\s+plane\b",
                observation_lower,
            )
            if match:
                return float(match.group(1)), "percent_down_plane"

        return None, ""

    def _comparison_prefers_lower_value(
        self, *, property_text: str, metric: str, direction: str
    ) -> bool | None:
        normalized_property = self._normalize_runtime_text(property_text)
        if metric == "percent_down_plane" and "friction" in normalized_property:
            return direction == "max"
        return None

    def _update_comparison_search_tracking(
        self, *, action: str | None, observation: str
    ) -> None:
        task_contract = self._get_task_contract()
        if not self._is_comparison_task(task_contract):
            return

        grounded_tokens = set(self._extract_runtime_tokens(observation, limit=48))
        target_grounded = self._primary_target_is_grounded(
            task_contract, grounded_tokens
        ) or self._comparison_target_is_grounded(task_contract, observation=observation)

        previous_observation = ""
        if self.curr_episodic_memory:
            try:
                previous_observation = json.loads(self.curr_episodic_memory[-1]).get(
                    "resulting_observation", ""
                )
            except (TypeError, ValueError, json.JSONDecodeError):
                previous_observation = ""

        room_signature = self._get_current_location_signature(
            observation
        ) or self._get_current_location_signature(previous_observation)
        if room_signature:
            room_state = self._search_location_states.setdefault(
                room_signature,
                {
                    "local_exploration": 0,
                    "target_grounded": False,
                },
            )
            if target_grounded:
                room_state["target_grounded"] = True

        if not action or action == "None" or self._HARD_FAILURE_RE.search(observation):
            return
        if not room_signature or target_grounded:
            return

        family = self._classify_action_family(action)
        if family == "inspect":
            if self._action_targets_room_frontier(action, observation, family=family):
                return
            room_state["local_exploration"] += 1
        elif family == "device_control":
            if self._action_targets_room_frontier(action, observation, family=family):
                return
            if self._action_mentions_door(action, family=family):
                return
            room_state["local_exploration"] += 1

    def _update_comparison_tracking(
        self, *, action: str | None, observation: str
    ) -> None:
        if not self._is_comparison_task():
            return

        task_contract = self._get_task_contract()
        target = self._resolve_comparison_target_from_context(
            action=action,
            observation=observation,
            task_contract=task_contract,
        )
        if not target:
            return

        value, metric = self._parse_comparison_observation(
            observation, task_contract=task_contract
        )
        if value is None or not metric:
            return

        self._comparison_observations[target] = {
            "value": value,
            "metric": metric,
            "evidence_action": action or "None",
            "timestep": self.num_actions_taken,
        }

        if len(self._comparison_observations) < len(
            task_contract.get("comparison_targets", [])
        ):
            return

        direction = task_contract.get("comparison_direction", "")
        property_text = task_contract.get("comparison_property", "")
        prefer_lower = self._comparison_prefers_lower_value(
            property_text=property_text,
            metric=metric,
            direction=direction,
        )
        if prefer_lower is None:
            return

        ranked = sorted(
            (
                (
                    target_name,
                    observation_entry["value"],
                    observation_entry["metric"],
                )
                for target_name, observation_entry in self._comparison_observations.items()
                if observation_entry.get("metric") == metric
            ),
            key=lambda item: item[1],
            reverse=not prefer_lower,
        )
        if len(ranked) < 2 or ranked[0][1] == ranked[1][1]:
            return

        selected_target = ranked[0][0]
        self._selected_comparison_target = selected_target
        self._comparison_resolution = {
            "comparison_property": property_text,
            "comparison_direction": direction,
            "branch_target": selected_target,
            "evidence_metric": metric,
            "evidence_action": action or "None",
            "timestep": self.num_actions_taken,
        }

    def _get_comparison_tracking_snapshot(self) -> dict:
        if not self._is_comparison_task():
            return {}

        task_contract = self._get_task_contract()
        snapshot = {
            "comparison_subject": task_contract.get("comparison_subject", [])[:1],
            "comparison_property": task_contract.get("comparison_property", ""),
            "comparison_direction": task_contract.get("comparison_direction", ""),
            "comparison_targets": task_contract.get("comparison_targets", [])[:2],
            "branch_target": (
                [self._selected_comparison_target]
                if self._selected_comparison_target
                else []
            ),
            "branch_ready": bool(self._selected_comparison_target),
        }
        if self._comparison_observations:
            snapshot["observations"] = {
                target: {
                    "value": entry["value"],
                    "metric": entry["metric"],
                }
                for target, entry in self._comparison_observations.items()
            }
        return snapshot

    def _update_conditional_branch_tracking(
        self, *, action: str | None, observation: str
    ) -> None:
        if not self._is_conditional_branch_task():
            return
        if self._selected_conditional_branch_target:
            return
        observation_tokens = set(
            self._extract_runtime_tokens(
                observation,
                stopwords=self._TASK_STOPWORDS | self._ACTION_COMMAND_STOPWORDS,
            )
        )
        if not observation_tokens:
            return

        matches: list[dict] = []
        for branch in self._get_task_contract().get("conditional_branches", []):
            condition_tokens = set(branch.get("condition_tokens", []))
            if condition_tokens and condition_tokens.issubset(observation_tokens):
                matches.append(branch)

        if len(matches) != 1:
            return

        resolved_branch = matches[0]
        self._selected_conditional_branch_target = resolved_branch["target"]
        self._conditional_branch_resolution = {
            "condition": resolved_branch["condition"],
            "branch_target": resolved_branch["target"],
            "evidence_action": action or "None",
            "evidence_tokens": resolved_branch.get("condition_tokens", []),
            "timestep": self.num_actions_taken,
        }

    def _get_conditional_branch_tracking_snapshot(self) -> dict:
        if not self._is_conditional_branch_task():
            return {}
        task_contract = self._get_task_contract()
        snapshot = {
            "evidence_target": task_contract.get(
                "conditional_branch_evidence_target", []
            )[:1],
            "evidence_subject": task_contract.get("conditional_branch_subject", [])[:1],
            "branch_targets": task_contract.get("conditional_branch_targets", [])[:2],
            "branch_target": (
                [self._selected_conditional_branch_target]
                if self._selected_conditional_branch_target
                else []
            ),
            "branch_ready": bool(self._selected_conditional_branch_target),
        }
        if self._conditional_branch_resolution:
            snapshot["resolved_condition"] = self._conditional_branch_resolution.get(
                "condition", ""
            )
        return snapshot

    def _action_targets_conditional_branch(
        self,
        action: str,
        *,
        family: str,
        task_contract: dict | None = None,
        role_token_sets: dict[str, list[set[str]]] | None = None,
    ) -> bool:
        contract = task_contract or self._get_task_contract()
        if not self._is_conditional_branch_task(contract):
            return False
        role_sets = role_token_sets or self._get_task_role_token_sets(contract)
        branch_roles = role_sets["conditional_branch_targets"]
        if not branch_roles:
            return False
        content_token_set = set(
            self._extract_action_content_tokens(action, family=family)
        )
        if self._has_full_role_match(content_token_set, branch_roles):
            return True
        referent_signature = self._get_action_referent_signature(action, family=family)
        primary_signature = self._extract_action_primary_object_signature(
            action, family=family
        )
        destination_signature = self._extract_action_destination_signature(
            action, family=family
        )
        return any(
            self._signature_matches_role(signature, branch_roles)
            for signature in (
                referent_signature,
                primary_signature,
                destination_signature,
            )
            if signature
        )

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

    def _state_change_room_search_stalled(
        self,
        *,
        current_room: str,
        visible_doors: int,
        task_contract: dict | None = None,
        grounded_tokens: set[str] | None = None,
    ) -> bool:
        contract = task_contract or self._get_task_contract()
        if not self._is_state_change_task(contract):
            return False
        grounded_token_set = grounded_tokens or set(
            self._get_observation_grounded_tokens()
        )
        if self._state_change_target_is_grounded(contract, grounded_token_set):
            return False
        if visible_doors < 2 or not current_room:
            return False
        room_state = self._search_location_states.get(current_room, {})
        return bool(
            not room_state.get("target_grounded")
            and room_state.get("local_exploration", 0) >= 2
        )

    def _measurement_instrument_search_stalled(
        self,
        *,
        current_room: str,
        visible_doors: int,
        task_contract: dict | None = None,
        grounded_tokens: set[str] | None = None,
    ) -> bool:
        contract = task_contract or self._get_task_contract()
        if not self._is_measurement_task(contract):
            return False
        grounded_token_set = grounded_tokens or set(
            self._get_observation_grounded_tokens()
        )
        role_token_sets = self._get_task_role_token_sets(contract)
        if self._role_is_grounded(
            grounded_token_set, role_token_sets["measurement_instrument"]
        ):
            return False
        if visible_doors < 1 or not current_room:
            return False
        room_state = self._search_location_states.get(current_room, {})
        return bool(room_state.get("local_exploration", 0) >= 2)

    def _artifact_creation_room_search_stalled(
        self,
        *,
        current_room: str,
        visible_doors: int,
        task_contract: dict | None = None,
    ) -> bool:
        contract = task_contract or self._get_task_contract()
        if not self._is_artifact_creation_task(contract):
            return False
        if self._get_grounded_artifact_labels(limit=1):
            return False
        if visible_doors < 2 or not current_room:
            return False
        room_state = self._search_location_states.get(current_room, {})
        return bool(
            not room_state.get("target_grounded")
            and room_state.get("local_exploration", 0) >= 2
        )

    def _growth_room_search_stalled(
        self,
        *,
        current_room: str,
        visible_doors: int,
        task_contract: dict | None = None,
        grounded_tokens: set[str] | None = None,
    ) -> bool:
        contract = task_contract or self._get_task_contract()
        if not self._is_growth_task(contract):
            return False

        grounded_token_set = grounded_tokens or set(
            self._get_observation_grounded_tokens()
        )
        if self._primary_target_is_grounded(contract, grounded_token_set):
            return False
        if visible_doors < 1 or not current_room:
            return False

        current_observation = (self.percept or {}).get("resulting_observation", "")
        visible_stage_labels = self._get_visible_lifecycle_stage_labels(
            current_observation
        )
        if any(label not in {"seed"} for label in visible_stage_labels):
            return False

        room_state = self._search_location_states.get(current_room, {})
        if (
            not room_state.get("target_grounded")
            and room_state.get("local_exploration", 0) >= 2
        ):
            return True

        nonproductive_growth_tests = 0
        stalled_growth_tests = 0
        for family in {
            "focus",
            "inspect",
            "device_control",
            "transfer_or_transform",
            "tool_application",
            "relocation",
        }:
            entry = self.episode_hypothesis_ledger.get(family)
            if not entry or entry["observable_change_attempts"] > 0:
                continue
            nonproductive_growth_tests += entry["tests"]
            stalled_growth_tests += (
                entry["stalled_attempts"] + entry["invalid_attempts"]
            )

        return bool(
            nonproductive_growth_tests >= 3
            and stalled_growth_tests >= 2
            and self._growth_task_has_precursor_signal(
                current_observation,
                task_contract=contract,
                grounded_tokens=grounded_token_set,
            )
        )

    def _comparison_room_search_stalled(
        self,
        *,
        current_room: str,
        visible_doors: int,
        task_contract: dict | None = None,
        grounded_tokens: set[str] | None = None,
    ) -> bool:
        contract = task_contract or self._get_task_contract()
        if not self._is_comparison_task(contract):
            return False

        grounded_token_set = grounded_tokens or set(
            self._get_observation_grounded_tokens()
        )
        if self._primary_target_is_grounded(contract, grounded_token_set) or (
            self._comparison_target_is_grounded(contract)
        ):
            return False
        if visible_doors < 1 or not current_room:
            return False

        room_state = self._search_location_states.get(current_room, {})
        return bool(
            not room_state.get("target_grounded")
            and room_state.get("local_exploration", 0) >= 2
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

    def _get_visible_lifecycle_stage_labels(
        self, observation: str | None = None
    ) -> list[str]:
        observation_text = (
            observation
            if observation is not None
            else (self.percept or {}).get("resulting_observation", "")
        )
        return self._merge_stage_labels(
            self._extract_stage_labels(observation_text) + self._observed_stage_labels
        )

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

    def _update_lifecycle_search_tracking(
        self, *, action: str | None, observation: str
    ) -> None:
        task_contract = self._get_task_contract()
        if not self._is_lifecycle_task(task_contract):
            return

        room_signature = self._get_current_location_signature(observation)
        if not room_signature:
            return

        grounded_tokens = set(self._extract_runtime_tokens(observation, limit=40))
        current_location_tokens = set(
            self._extract_current_location_tokens(observation)
        )
        room_state = self._search_location_states.setdefault(
            room_signature,
            {
                "local_exploration": 0,
                "target_grounded": False,
            },
        )
        if self._get_visible_lifecycle_stage_labels(observation) or bool(
            (grounded_tokens & set(task_contract.get("target_entities", [])))
            - current_location_tokens
        ):
            room_state["target_grounded"] = True

        if room_state["target_grounded"]:
            return
        if not action or action == "None" or self._HARD_FAILURE_RE.search(observation):
            return

        family = self._classify_action_family(action)
        if family not in {"inspect", "device_control"}:
            return
        if self._action_targets_room_frontier(action, observation, family=family):
            return
        room_state["local_exploration"] += 1

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

    def _update_task_contract_from_recipe_observation(
        self, action: str, observation: str
    ) -> None:
        """If a cookbook/recipe was just read, extract ingredients and inject
        them as discovered primary_targets in the task contract cache."""
        if not re.search(r"\b(?:read|check|examine)\b", action, re.IGNORECASE):
            return
        match = self._RECIPE_SECTION_RE.search(observation)
        if not match:
            return
        ingredients_block, directions_block = match.group(1), match.group(2)
        ingredients = [
            self._normalize_task_phrase(m.group(1), role="primary_targets")
            for m in self._RECIPE_INGREDIENTS_RE.finditer(ingredients_block)
            if m.group(1).strip()
        ]
        verbs_found = bool(self._RECIPE_DIRECTIONS_RE.search(directions_block))
        if not ingredients and not verbs_found:
            return
        contract = self._get_task_contract()
        existing = set(contract.get("primary_targets", []))
        new_targets = [t for t in ingredients if t and t not in existing]
        if new_targets:
            contract["primary_targets"] = list(existing) + new_targets
            contract["target_entities"] = list(
                set(contract.get("target_entities", [])) | set(new_targets)
            )
        if verbs_found and "tool_application" not in contract.get(
            "required_families", []
        ):
            contract.setdefault("required_families", []).append("tool_application")

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
            "comparison_subject",
            "comparison_targets",
            "conditional_branch_subject",
            "conditional_branch_evidence_target",
            "conditional_branch_targets",
            "destination_container",
            "destination_room",
        ):
            role_token_sets[role] = []
            for phrase in contract.get(role, []):
                if role in {"comparison_subject", "comparison_targets"}:
                    phrase_tokens = set(
                        self._extract_comparison_tokens(phrase, limit=6)
                    )
                else:
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
            r"\byou\s+move\s+to\s+(?:the\s+)?([a-z0-9][a-z0-9\s-]{0,40}?)(?:[.,\n]|$)",
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
        return bool(
            (contract.get("search_mode") or contract.get("candidate_classes"))
            and not contract.get("conditional_branch_task")
        )

    def _is_inferred_target_search_task(
        self, task_contract: dict | None = None
    ) -> bool:
        contract = task_contract or self._get_task_contract()
        return bool(contract.get("inferred_search_mode"))

    def _is_relation_mechanism_task(self, task_contract: dict | None = None) -> bool:
        contract = task_contract or self._get_task_contract()
        return bool(contract.get("relation_mechanism_task"))

    def _primary_target_is_grounded(
        self,
        task_contract: dict | None = None,
        grounded_tokens: set[str] | None = None,
    ) -> bool:
        contract = task_contract or self._get_task_contract()
        grounded_token_set = grounded_tokens or set(
            self._get_observation_grounded_tokens()
        )
        role_token_sets = self._get_task_role_token_sets(contract)
        return self._role_is_grounded(
            grounded_token_set, role_token_sets["primary_targets"]
        )

    def _comparison_target_is_grounded(
        self,
        task_contract: dict | None = None,
        observation: str | None = None,
    ) -> bool:
        contract = task_contract or self._get_task_contract()
        if not self._is_comparison_task(contract):
            return False

        observed_text = observation
        if observed_text is None:
            observed_text = ((self.percept or {}) or {}).get(
                "resulting_observation", ""
            )

        observation_tokens = set(
            self._extract_comparison_tokens(observed_text, limit=48)
        )
        role_token_sets = self._get_task_role_token_sets(contract)
        return any(
            target_tokens and target_tokens.issubset(observation_tokens)
            for target_tokens in role_token_sets["comparison_targets"]
        )

    def _growth_task_has_precursor_signal(
        self,
        observation: str,
        *,
        task_contract: dict | None = None,
        grounded_tokens: set[str] | None = None,
    ) -> bool:
        contract = task_contract or self._get_task_contract()
        if not self._is_growth_task(contract):
            return False

        grounded_token_set = grounded_tokens or set(
            self._get_observation_grounded_tokens()
        )
        if self._primary_target_is_grounded(contract, grounded_token_set):
            return True

        role_token_sets = self._get_task_role_token_sets(contract)
        precursor_support_roles = [
            token_set
            for token_set in role_token_sets["supporting_targets"]
            if token_set & self._GROWTH_PRECURSOR_TOKENS
        ]
        if self._role_is_grounded(grounded_token_set, precursor_support_roles):
            return True

        if self._get_visible_lifecycle_stage_labels(observation):
            return True

        return bool(grounded_token_set & set(self._GROWTH_TASK_TARGET_HINTS))

    def _supporting_target_is_grounded(
        self,
        task_contract: dict | None = None,
        grounded_tokens: set[str] | None = None,
    ) -> bool:
        contract = task_contract or self._get_task_contract()
        grounded_token_set = grounded_tokens or set(
            self._get_observation_grounded_tokens()
        )
        role_token_sets = self._get_task_role_token_sets(contract)
        if self._role_is_grounded(
            grounded_token_set, role_token_sets["supporting_targets"]
        ):
            return True
        if not self._relation_support_role_uses_candidate_inference(
            task_contract=contract
        ):
            return False

        for candidate in self._get_relation_support_candidate_signatures(
            task_contract=contract
        ):
            if not self._relation_support_candidate_can_ground(
                candidate,
                task_contract=contract,
                role_token_sets=role_token_sets,
            ):
                continue
            if self._relation_support_candidate_is_grounded(
                candidate, grounded_tokens=grounded_token_set
            ):
                return True
        return False

    def _relation_support_role_uses_candidate_inference(
        self, *, task_contract: dict | None = None
    ) -> bool:
        contract = task_contract or self._get_task_contract()
        if not (
            self._is_relation_mechanism_task(contract)
            or self._is_inferred_target_search_task(contract)
        ):
            return False
        role_token_sets = self._get_task_role_token_sets(contract)
        return any(
            token_set and bool(token_set & self._ABSTRACT_SUPPORT_ROLE_HINTS)
            for token_set in role_token_sets["supporting_targets"]
        )

    def _score_relation_support_candidate_signature(
        self,
        signature: str,
        *,
        task_contract: dict | None = None,
        role_token_sets: dict[str, list[set[str]]] | None = None,
    ) -> int:
        contract = task_contract or self._get_task_contract()
        role_sets = role_token_sets or self._get_task_role_token_sets(contract)
        signature_tokens = self._referent_tokens(signature)
        current_observation = (self.percept or {}).get("resulting_observation", "")
        if not signature_tokens:
            return -24
        if self._signature_looks_structural_noncandidate(
            signature, observation=current_observation
        ):
            return -24
        if self._signature_matches_role(signature, role_sets["primary_targets"]):
            return -24

        support_tokens = {
            token
            for token_set in role_sets["supporting_targets"]
            for token in token_set
        }
        support_match = self._signature_matches_role(
            signature, role_sets["supporting_targets"]
        )
        power_like = bool(signature_tokens & self._POWER_SOURCE_HINTS)
        renewable_requested = "renewable" in support_tokens
        renewable_like = bool(signature_tokens & self._RENEWABLE_SOURCE_HINTS)
        nonrenewable_like = bool(signature_tokens & self._NON_RENEWABLE_SOURCE_HINTS)
        control_like = bool(signature_tokens & self._CONTROL_COMPONENT_HINTS)
        bridge_like = bool(signature_tokens & self._RELATION_BRIDGE_HINTS)
        sink_like = bool(signature_tokens & self._POWER_SINK_HINTS)

        score = 0
        if support_match:
            score += 24
        if power_like:
            score += 12
        if support_tokens & self._ABSTRACT_SUPPORT_ROLE_HINTS and power_like:
            score += 6
        if renewable_requested:
            if renewable_like:
                score += 12
            elif nonrenewable_like:
                score -= 12
            elif power_like:
                score -= 4
        if control_like:
            score -= 12
        if bridge_like:
            score -= 10
        if sink_like:
            score -= 10
        if not (support_match or power_like or renewable_like):
            score -= 6
        return score

    def _relation_support_candidate_can_ground(
        self,
        signature: str,
        *,
        task_contract: dict | None = None,
        role_token_sets: dict[str, list[set[str]]] | None = None,
    ) -> bool:
        contract = task_contract or self._get_task_contract()
        role_sets = role_token_sets or self._get_task_role_token_sets(contract)
        if self._signature_matches_role(signature, role_sets["supporting_targets"]):
            return True

        signature_tokens = self._referent_tokens(signature)
        if not signature_tokens:
            return False

        support_tokens = {
            token
            for token_set in role_sets["supporting_targets"]
            for token in token_set
        }
        renewable_requested = "renewable" in support_tokens
        power_like = bool(signature_tokens & self._POWER_SOURCE_HINTS)
        renewable_like = bool(signature_tokens & self._RENEWABLE_SOURCE_HINTS)
        nonrenewable_like = bool(signature_tokens & self._NON_RENEWABLE_SOURCE_HINTS)
        if renewable_requested:
            return renewable_like and not nonrenewable_like
        if support_tokens & self._ABSTRACT_SUPPORT_ROLE_HINTS:
            return power_like and not nonrenewable_like
        return False

    def _relation_support_candidate_is_grounded(
        self, signature: str, *, grounded_tokens: set[str] | None = None
    ) -> bool:
        grounded_token_set = grounded_tokens or set(
            self._get_observation_grounded_tokens()
        )
        signature_tokens = self._referent_tokens(signature)
        if not signature_tokens:
            return False
        if signature_tokens.issubset(grounded_token_set):
            return True
        if any(
            self._signature_matches_frontier(signature, [completed_target])
            for completed_target in self._completed_focus_targets
        ):
            return True
        return any(
            self._signature_matches_frontier(signature, [referent])
            and self._referent_is_visible(referent)
            for referent in self._relation_frontier_referents[-6:]
        )

    def _get_relation_support_candidate_signatures(
        self,
        actions: list[str] | None = None,
        task_contract: dict | None = None,
    ) -> list[str]:
        contract = task_contract or self._get_task_contract()
        if not contract.get("supporting_targets"):
            return []

        role_token_sets = self._get_task_role_token_sets(contract)
        candidate_profiles: dict[str, dict[str, int]] = {}
        for action in actions if actions is not None else self.admissible_actions:
            family = self._classify_action_family(action)
            if family not in {"inspect", "focus", "device_control"}:
                continue
            if self._action_mentions_door(action, family=family):
                continue
            referent = self._extract_action_primary_object_signature(
                action, family=family
            )
            if not referent:
                continue
            profile = candidate_profiles.setdefault(
                referent,
                {"inspect": 0, "focus": 0, "device_control": 0},
            )
            profile[family] += 1
        for referent in self._relation_frontier_referents[-6:]:
            if not referent:
                continue
            candidate_profiles.setdefault(
                referent,
                {"inspect": 0, "focus": 0, "device_control": 0},
            )

        ranked_candidates: list[tuple[int, str]] = []
        for candidate, profile in candidate_profiles.items():
            score = self._score_relation_support_candidate_signature(
                candidate,
                task_contract=contract,
                role_token_sets=role_token_sets,
            )
            if score <= 0:
                continue
            score += profile["inspect"] * 4
            score += profile["focus"] * 2
            score += profile["device_control"] * 2
            ranked_candidates.append((score, candidate))

        ranked_candidates.sort(key=lambda item: (-item[0], len(item[1]), item[1]))
        return [candidate for _, candidate in ranked_candidates[:4]]

    def _get_current_location_signature(self, observation: str) -> str:
        return " ".join(self._extract_current_location_tokens(observation)[:4])

    def _count_visible_doors(self, observation: str) -> int:
        return len(
            re.findall(r"\bdoor to\b|\bdoor \(that is\b|\bdoor\b", observation.lower())
        )

    def _is_agent_navigation_action(
        self, action: str, *, family: str | None = None
    ) -> bool:
        family = family or self._classify_action_family(action)
        if family != "relocation":
            return False
        normalized = self._normalize_runtime_text(action)
        return normalized.startswith(("go to ", "enter "))

    def _action_mentions_door(self, action: str, *, family: str | None = None) -> bool:
        action_tokens = set(self._extract_action_content_tokens(action, family=family))
        return "door" in action_tokens or "doors" in action_tokens

    def _extract_visible_room_targets(self, observation: str) -> list[str]:
        room_targets: list[str] = []
        for match in re.finditer(
            r"\bdoor to\s+(?:the\s+)?([a-z0-9][a-z0-9\s-]{0,40}?)(?=\s*\()",
            (observation or "").lower(),
        ):
            signature = self._phrase_to_referent_signature(match.group(1))
            if signature and signature not in room_targets:
                room_targets.append(signature)
        return room_targets[:6]

    def _get_room_frontier_target(
        self, action: str, observation: str, *, family: str | None = None
    ) -> str:
        family = family or self._classify_action_family(action)
        if family not in {"inspect", "device_control", "relocation"}:
            return ""
        if family == "relocation" and not self._is_agent_navigation_action(
            action, family=family
        ):
            return ""

        room_targets = self._extract_visible_room_targets(observation)
        if not room_targets:
            return ""

        referent_candidates = []
        referent = self._get_action_referent_signature(action, family=family)
        if referent:
            referent_candidates.append(referent)
        primary_object = self._extract_action_primary_object_signature(
            action, family=family
        )
        if primary_object and primary_object not in referent_candidates:
            referent_candidates.append(primary_object)

        for candidate in referent_candidates:
            for room_target in room_targets:
                if self._signature_matches_frontier(candidate, [room_target]):
                    return room_target
        return ""

    def _action_targets_room_frontier(
        self, action: str, observation: str, *, family: str | None = None
    ) -> bool:
        family = family or self._classify_action_family(action)
        if family not in {"inspect", "device_control", "relocation"}:
            return False
        if family == "relocation" and not self._is_agent_navigation_action(
            action, family=family
        ):
            return False
        if self._action_mentions_door(action, family=family):
            return True
        return bool(self._get_room_frontier_target(action, observation, family=family))

    def _extract_relation_endpoint_signatures(
        self, action: str, *, family: str | None = None
    ) -> tuple[str, str]:
        family = family or self._classify_action_family(action)
        if family != "relation":
            return "", ""

        normalized = self._normalize_runtime_text(action)
        if normalized.startswith("connect ") and " to " in normalized:
            lhs, rhs = normalized[len("connect ") :].split(" to ", maxsplit=1)
        elif normalized.startswith("disconnect ") and " from " in normalized:
            lhs, rhs = normalized[len("disconnect ") :].split(" from ", maxsplit=1)
        else:
            return "", ""

        stopwords = self._RUNTIME_TOKEN_STOPWORDS | self._ACTION_COMMAND_STOPWORDS
        left_tokens = self._extract_runtime_tokens(lhs, stopwords=stopwords, limit=6)
        right_tokens = self._extract_runtime_tokens(rhs, stopwords=stopwords, limit=6)
        return " ".join(left_tokens[:6]), " ".join(right_tokens[:6])

    def _is_control_candidate_signature(self, signature: str) -> bool:
        signature_tokens = self._referent_tokens(signature)
        return bool(signature_tokens & self._CONTROL_COMPONENT_HINTS)

    def _is_relation_bridge_signature(self, signature: str) -> bool:
        signature_tokens = self._referent_tokens(signature)
        return bool(signature_tokens & self._RELATION_BRIDGE_HINTS)

    def _signature_matches_frontier(
        self, signature: str, frontier_referents: list[str]
    ) -> bool:
        signature_tokens = self._referent_tokens(signature)
        if not signature_tokens:
            return False

        for frontier_referent in frontier_referents:
            frontier_tokens = self._referent_tokens(frontier_referent)
            if not frontier_tokens:
                continue
            overlap = signature_tokens & frontier_tokens
            if not overlap:
                continue
            if frontier_tokens.issubset(signature_tokens) or signature_tokens.issubset(
                frontier_tokens
            ):
                return True
            discriminative_overlap = overlap - self._FRONTIER_GENERIC_TOKENS
            if discriminative_overlap:
                return True
        return False

    def _get_control_candidate_signatures(
        self,
        actions: list[str] | None = None,
        task_contract: dict | None = None,
    ) -> list[str]:
        contract = task_contract or self._get_task_contract()
        candidates: list[str] = []
        active_actions = actions if actions is not None else self.admissible_actions
        role_token_sets = self._get_task_role_token_sets(contract)
        for action in active_actions:
            family = self._classify_action_family(action)
            if family != "device_control":
                continue
            referent = self._extract_action_primary_object_signature(
                action, family=family
            )
            if (
                not referent
                or self._action_mentions_door(action, family=family)
                or self._signature_matches_role(
                    referent, role_token_sets["destination_room"]
                )
            ):
                continue
            if (
                self._is_control_candidate_signature(referent)
                or self._signature_matches_role(
                    referent, role_token_sets["supporting_targets"]
                )
            ) and referent not in candidates:
                candidates.append(referent)
        return candidates[:4]

    def _update_relation_task_tracking(
        self, *, action: str | None, observation: str
    ) -> None:
        task_contract = self._get_task_contract()
        if not (
            self._is_relation_mechanism_task(task_contract)
            or self._is_inferred_target_search_task(task_contract)
        ):
            return

        room_signature = self._get_current_location_signature(observation)
        grounded_tokens = set(self._extract_runtime_tokens(observation, limit=40))
        primary_grounded = self._primary_target_is_grounded(
            task_contract, grounded_tokens
        )
        if room_signature:
            room_state = self._search_location_states.setdefault(
                room_signature,
                {
                    "local_exploration": 0,
                    "target_grounded": False,
                },
            )
            if primary_grounded:
                room_state["target_grounded"] = True

        if not action or action == "None" or self._HARD_FAILURE_RE.search(observation):
            return

        family = self._classify_action_family(action)
        primary_signature = self._extract_action_primary_object_signature(
            action, family=family
        )
        secondary_signature = ""
        if family == "relation":
            _, secondary_signature = self._extract_relation_endpoint_signatures(
                action, family=family
            )

        if room_signature and not primary_grounded:
            room_state = self._search_location_states.setdefault(
                room_signature,
                {
                    "local_exploration": 0,
                    "target_grounded": False,
                },
            )
            if family == "inspect" or (
                family == "device_control"
                and not self._action_mentions_door(action, family=family)
            ):
                room_state["local_exploration"] += 1

        for signature in (primary_signature, secondary_signature):
            if signature and signature not in self._relation_frontier_referents:
                self._relation_frontier_referents.append(signature)
        self._relation_frontier_referents = self._relation_frontier_referents[-8:]

        if family in {"inspect", "focus"} and primary_signature:
            role_token_sets = self._get_task_role_token_sets(task_contract)
            if self._signature_matches_role(
                primary_signature, role_token_sets["primary_targets"]
            ):
                normalized_observation = self._normalize_runtime_text(observation)
                if re.search(r"\bwhich is on\b|\bit is on\b", normalized_observation):
                    self._target_status_by_referent[primary_signature] = "on"
                elif re.search(
                    r"\bwhich is off\b|\bit is off\b", normalized_observation
                ):
                    self._target_status_by_referent[primary_signature] = "off"

    def _get_relation_frontier_snapshot(
        self, actions: list[str] | None = None, task_contract: dict | None = None
    ) -> dict:
        contract = task_contract or self._get_task_contract()
        if not self._is_relation_mechanism_task(contract):
            return {}

        grounded_tokens = set(self._get_observation_grounded_tokens())
        role_token_sets = self._get_task_role_token_sets(contract)
        frontier_referents: list[str] = []
        support_candidates = self._get_relation_support_candidate_signatures(
            actions=actions, task_contract=contract
        )
        for phrase in contract.get("primary_targets", [])[:2]:
            if phrase and phrase not in frontier_referents:
                frontier_referents.append(phrase)
        if self._supporting_target_is_grounded(contract, grounded_tokens):
            for phrase in contract.get("supporting_targets", [])[:2]:
                if phrase and phrase not in frontier_referents:
                    frontier_referents.append(phrase)
        for candidate in support_candidates[:3]:
            if candidate not in frontier_referents:
                frontier_referents.append(candidate)
        for referent in self._relation_frontier_referents[-6:]:
            if not referent:
                continue
            if (
                self._signature_matches_role(
                    referent, role_token_sets["primary_targets"]
                )
                or self._signature_matches_role(
                    referent, role_token_sets["supporting_targets"]
                )
                or self._is_control_candidate_signature(referent)
            ):
                if referent not in frontier_referents:
                    frontier_referents.append(referent)

        control_candidates = self._get_control_candidate_signatures(actions, contract)
        for candidate in control_candidates:
            if candidate not in frontier_referents:
                frontier_referents.append(candidate)

        primary_target = contract.get("primary_targets", [""])
        primary_signature = primary_target[0] if primary_target else ""
        target_status = ""
        if primary_signature:
            target_status = self._target_status_by_referent.get(primary_signature, "")

        return {
            "frontier_entities": frontier_referents[:6],
            "support_candidates": support_candidates[:3],
            "control_candidates": control_candidates[:3],
            "primary_target_grounded": self._primary_target_is_grounded(
                contract, grounded_tokens
            ),
            "supporting_source_grounded": self._supporting_target_is_grounded(
                contract, grounded_tokens
            ),
            "target_status": target_status,
        }

    def _primary_relation_component_is_grounded(
        self,
        *,
        task_contract: dict | None = None,
        role_token_sets: dict[str, list[set[str]]] | None = None,
    ) -> bool:
        contract = task_contract or self._get_task_contract()
        role_sets = role_token_sets or self._get_task_role_token_sets(contract)
        primary_roles = role_sets["primary_targets"]
        if not primary_roles:
            return False

        inspect_actions: list[str] = []
        inspect_entry = self.episode_hypothesis_ledger.get("inspect", {})
        if inspect_entry.get("last_action"):
            inspect_actions.append(inspect_entry["last_action"])
        attempted_action = (self.percept or {}).get("attempted_action", "")
        if attempted_action:
            inspect_actions.append(attempted_action)

        for action in inspect_actions:
            family = self._classify_action_family(action)
            if family not in {"inspect", "focus"}:
                continue
            action_tokens = set(
                self._extract_action_content_tokens(action, family=family)
            )
            if not action_tokens:
                continue
            for primary_tokens in primary_roles:
                if not primary_tokens or not primary_tokens.issubset(action_tokens):
                    continue
                extra_tokens = action_tokens - primary_tokens
                if extra_tokens:
                    return True

        for referent in self._relation_frontier_referents[-6:]:
            referent_tokens = self._referent_tokens(referent)
            if not referent_tokens:
                continue
            for primary_tokens in primary_roles:
                if not primary_tokens or not primary_tokens.issubset(referent_tokens):
                    continue
                extra_tokens = referent_tokens - primary_tokens
                if extra_tokens:
                    return True
        return False

    def _is_primary_relation_inspect_action(
        self,
        action: str,
        *,
        task_contract: dict | None = None,
        role_token_sets: dict[str, list[set[str]]] | None = None,
    ) -> bool:
        family = self._classify_action_family(action)
        if family != "inspect":
            return False

        contract = task_contract or self._get_task_contract()
        role_sets = role_token_sets or self._get_task_role_token_sets(contract)
        referent_tokens = set(
            self._extract_action_content_tokens(action, family=family)
        )
        if not referent_tokens:
            referent = self._extract_action_primary_object_signature(
                action, family=family
            )
            if not referent:
                referent = self._get_action_referent_signature(action, family=family)
            if not referent:
                return False
            referent_tokens = self._referent_tokens(referent)
        return any(
            primary_tokens and primary_tokens.issubset(referent_tokens)
            for primary_tokens in role_sets["primary_targets"]
        )

    def _is_relation_commit_candidate_action(
        self,
        action: str,
        *,
        family: str | None = None,
        task_contract: dict | None = None,
        role_token_sets: dict[str, list[set[str]]] | None = None,
        control_candidates: list[str] | None = None,
    ) -> bool:
        family = family or self._classify_action_family(action)
        if family not in {"relation", "device_control"}:
            return False

        contract = task_contract or self._get_task_contract()
        role_sets = role_token_sets or self._get_task_role_token_sets(contract)
        content_token_set = set(
            self._extract_action_content_tokens(action, family=family)
        )
        primary_role_hits, _ = self._best_role_overlap(
            content_token_set, role_sets["primary_targets"]
        )
        support_role_hits, _ = self._best_role_overlap(
            content_token_set, role_sets["supporting_targets"]
        )
        if primary_role_hits or support_role_hits:
            return True

        if family != "device_control":
            return False

        referent = self._extract_action_primary_object_signature(action, family=family)
        if not referent or self._action_mentions_door(action, family=family):
            return False
        return bool(
            self._is_control_candidate_signature(referent)
            or referent in (control_candidates or [])
        )

    def _extract_remote_inspected_room_signature(
        self,
        *,
        action: str | None,
        observation: str,
        previous_observation: str,
    ) -> str:
        if not action:
            return ""

        family = self._classify_action_family(action)
        normalized = self._normalize_runtime_text(action)
        if family != "inspect" or not normalized.startswith("look in "):
            return ""

        referent = self._get_action_referent_signature(action, family=family)
        previous_room = self._get_current_location_signature(previous_observation)
        if not referent or referent == previous_room or referent == "inventory":
            return ""

        if not self._signature_looks_like_room(
            referent, observation, previous_observation
        ):
            return ""

        return referent

    def _update_remote_room_signal(
        self,
        *,
        action: str | None,
        observation: str,
        previous_observation: str,
    ) -> None:
        family = self._classify_action_family(action) if action else ""
        current_room = self._get_current_location_signature(observation)
        if (
            current_room
            and action
            and family == "relocation"
            and self._is_agent_navigation_action(action, family=family)
        ):
            self._remote_room_signals.pop(current_room, None)

        if not action or action == "None" or self._HARD_FAILURE_RE.search(observation):
            return

        inspected_room = self._extract_remote_inspected_room_signature(
            action=action,
            observation=observation,
            previous_observation=previous_observation,
        )
        if not inspected_room:
            return

        task_contract = self._get_task_contract()
        observation_tokens = set(self._extract_runtime_tokens(observation, limit=40))
        signal_tokens = sorted(
            (
                observation_tokens
                & self._get_remote_room_signal_match_tokens(task_contract)
            )
            - self._referent_tokens(inspected_room)
        )
        if not signal_tokens:
            return

        self._remote_room_signals[inspected_room] = {
            "signal_tokens": signal_tokens[:4],
            "last_seen_timestep": self.num_actions_taken,
        }

    def _get_remote_room_signal_snapshot(self) -> dict:
        if not self._remote_room_signals:
            return {}

        room, entry = max(
            self._remote_room_signals.items(),
            key=lambda item: (
                item[1].get("last_seen_timestep", -1),
                len(item[1].get("signal_tokens", [])),
                item[0],
            ),
        )
        snapshot = {"room": room}
        signal_tokens = entry.get("signal_tokens", [])
        if signal_tokens:
            snapshot["signal_tokens"] = signal_tokens[:4]
        return snapshot

    def _get_remote_room_signal_match_tokens(
        self, task_contract: dict | None = None
    ) -> set[str]:
        contract = task_contract or self._get_task_contract()
        match_tokens = set(contract.get("target_entities", []))
        role_token_sets = self._get_task_role_token_sets(contract)
        for tokens in role_token_sets.get("candidate_classes", []):
            match_tokens.update(tokens)
        return match_tokens

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
        current_observation = (self.percept or {}).get("resulting_observation", "")
        if not candidate_tokens or self._signature_looks_structural_noncandidate(
            candidate, observation=current_observation
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
            "container_confirmed": False,
            "room_confirmed": False,
            "stalled_confirmations": 0,
            "post_goal_confirmations": 0,
            "status": "active",
            "aliases": [],
            "last_seen_room": "",
        }
        self._candidate_states[candidate] = state
        return state

    def _get_candidate_labels(self, candidate: str) -> list[str]:
        if not candidate:
            return []
        labels = [candidate]
        state = self._candidate_states.get(candidate, {})
        for alias in state.get("aliases", []):
            if alias and alias not in labels:
                labels.append(alias)
        return labels

    def _record_candidate_alias(
        self, candidate: str, alias: str, task_contract: dict | None = None
    ) -> None:
        if not candidate or not alias:
            return
        contract = task_contract or self._get_task_contract()
        normalized_alias = self._phrase_to_referent_signature(alias)
        if (
            not normalized_alias
            or normalized_alias == candidate
            or self._is_support_referent(normalized_alias, contract)
        ):
            return
        alias_tokens = set(self._extract_runtime_tokens(normalized_alias, limit=6))
        if not alias_tokens:
            return
        state = self._get_candidate_state(candidate)
        aliases = state.setdefault("aliases", [])
        if normalized_alias not in aliases:
            aliases.append(normalized_alias)
            state["aliases"] = aliases[-6:]

    def _candidate_signature_matches(self, candidate: str, signature: str) -> bool:
        signature_tokens = self._referent_tokens(signature)
        if not signature_tokens:
            return False
        for label in self._get_candidate_labels(candidate):
            label_tokens = self._referent_tokens(label)
            if not label_tokens:
                continue
            overlap = signature_tokens & label_tokens
            if signature_tokens == label_tokens:
                return True
            if len(signature_tokens) >= 2 and signature_tokens.issubset(label_tokens):
                return True
            if len(label_tokens) >= 2 and label_tokens.issubset(signature_tokens):
                return True
            if len(overlap) >= 2:
                return True
        return False

    def _candidate_observed(self, candidate: str, observation: str) -> bool:
        observation_tokens = set(self._extract_runtime_tokens(observation, limit=36))
        if not observation_tokens:
            return False
        for label in self._get_candidate_labels(candidate):
            label_tokens = self._referent_tokens(label)
            if label_tokens and label_tokens.issubset(observation_tokens):
                return True
        return False

    def _extract_observation_candidate_alias(
        self, observation: str, *, family: str | None = None
    ) -> str:
        normalized = self._normalize_runtime_text(observation)
        if not normalized:
            return ""

        patterns = []
        if family == "focus":
            patterns.append(r"\byou\s+focus\s+on\s+(?:the\s+)?(.+?)(?:[.!?,]|$)")
        if family in {"relocation", "transfer_or_transform"}:
            patterns.append(
                r"\byou\s+move\s+(?:the\s+)?(.+?)\s+to\s+(?:the\s+)?[a-z0-9][a-z0-9\s-]{0,60}(?:[.!?,]|$)"
            )
        patterns.append(
            r"^\s*(?:a|an|the)\s+(.+?)(?=\s+(?:in|with|that|which|currently|containing|on)\b|[.,\n]|$)"
        )

        for pattern in patterns:
            match = re.search(pattern, normalized)
            if not match:
                continue
            alias = self._phrase_to_referent_signature(match.group(1))
            if alias:
                return alias
        return ""

    def _resolve_candidate_room_signature(
        self, observation: str, previous_observation: str
    ) -> str:
        return self._get_current_location_signature(
            observation
        ) or self._get_current_location_signature(previous_observation)

    def _signature_looks_like_room(
        self, signature: str, observation: str, previous_observation: str
    ) -> bool:
        if not signature:
            return False
        signature_tokens = self._referent_tokens(signature)
        if not signature_tokens:
            return False
        if signature_tokens.issubset(self._GENERIC_LOCATION_TOKENS):
            return True
        if (
            signature_tokens & {"room", "location", "area"}
            and len(signature_tokens) <= 3
        ):
            return True
        known_rooms = set(self._extract_visible_room_targets(observation))
        known_rooms.update(self._extract_visible_room_targets(previous_observation))
        current_room = self._get_current_location_signature(observation)
        previous_room = self._get_current_location_signature(previous_observation)
        if current_room:
            known_rooms.add(current_room)
        if previous_room:
            known_rooms.add(previous_room)
        known_rooms.update(self._get_task_contract().get("destination_room", []))
        return signature in known_rooms

    def _signature_looks_structural_noncandidate(
        self,
        signature: str,
        *,
        observation: str = "",
        previous_observation: str = "",
    ) -> bool:
        if not signature:
            return False
        signature_tokens = self._referent_tokens(signature)
        if not signature_tokens:
            return False
        if signature_tokens.issubset(self._NON_CANDIDATE_REFERENT_TOKENS):
            return True
        return self._signature_looks_like_room(
            signature,
            observation,
            previous_observation,
        )

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
        return self._observation_confirms_candidate_container(
            candidate, observation, task_contract
        ) or self._observation_confirms_candidate_room(
            candidate, observation, task_contract
        )

    def _observation_confirms_candidate_container(
        self, candidate: str, observation: str, task_contract: dict | None = None
    ) -> bool:
        if not candidate:
            return False
        contract = task_contract or self._get_task_contract()
        observation_tokens = set(self._extract_runtime_tokens(observation, limit=32))
        candidate_visible = any(
            label_tokens and label_tokens.issubset(observation_tokens)
            for label_tokens in (
                self._referent_tokens(label)
                for label in self._get_candidate_labels(candidate)
            )
        )
        if not candidate_visible:
            return False

        role_token_sets = self._get_task_role_token_sets(contract)
        return self._matches_any_role(
            observation_tokens, role_token_sets["destination_container"]
        )

    def _observation_confirms_candidate_room(
        self, candidate: str, observation: str, task_contract: dict | None = None
    ) -> bool:
        if not candidate:
            return False
        contract = task_contract or self._get_task_contract()
        observation_tokens = set(self._extract_runtime_tokens(observation, limit=32))
        candidate_visible = any(
            label_tokens and label_tokens.issubset(observation_tokens)
            for label_tokens in (
                self._referent_tokens(label)
                for label in self._get_candidate_labels(candidate)
            )
        )
        if not candidate_visible:
            return False

        role_token_sets = self._get_task_role_token_sets(contract)
        current_location_tokens = set(
            self._extract_current_location_tokens(observation)
        )
        return self._matches_any_role(
            current_location_tokens, role_token_sets["destination_room"]
        )

    def _candidate_goal_visibly_satisfied(
        self, candidate_state: dict, task_contract: dict | None = None
    ) -> bool:
        if not candidate_state:
            return False

        contract = task_contract or self._get_task_contract()
        required_families = set(contract.get("required_families", []))
        if "focus" in required_families and not candidate_state.get("focused"):
            return False
        if contract.get("destination_container") and not candidate_state.get(
            "container_confirmed"
        ):
            return False
        if (
            not contract.get("destination_container")
            and contract.get("destination_room")
            and not candidate_state.get("room_confirmed")
        ):
            return False
        if "relocation" in required_families and not (
            candidate_state.get("relocated") or candidate_state.get("support_confirmed")
        ):
            return False
        return bool(
            candidate_state.get("focused")
            or candidate_state.get("support_confirmed")
            or candidate_state.get("relocated")
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
        destination_signature = self._extract_action_destination_signature(
            executed_action, family=family
        )
        observation_alias = self._extract_observation_candidate_alias(
            observation, family=family
        )
        candidate_room = self._resolve_candidate_room_signature(
            observation, previous_observation
        )
        normalized_observation = self._normalize_observation_signature(observation)
        previous_signature = self._normalize_observation_signature(previous_observation)
        observation_stalled = normalized_observation == previous_signature
        pre_update_candidate = self._active_candidate or candidate
        pre_update_state = (
            dict(self._candidate_states.get(pre_update_candidate, {}))
            if pre_update_candidate
            else {}
        )

        if family == "focus" and candidate:
            state = self._get_candidate_state(candidate)
            state["focused"] = True
            if candidate_room:
                state["last_seen_room"] = candidate_room
            self._record_candidate_alias(candidate, observation_alias)
            if candidate not in self._rejected_candidates:
                self._active_candidate = candidate

        active_candidate = self._active_candidate or candidate
        if not active_candidate:
            return

        active_state = self._get_candidate_state(active_candidate)
        goal_was_satisfied = (
            self._candidate_goal_visibly_satisfied(pre_update_state)
            if active_candidate == pre_update_candidate
            else False
        )
        candidate_matches_active = bool(
            candidate and self._candidate_signature_matches(active_candidate, candidate)
        )
        if not candidate_matches_active and observation_alias:
            candidate_matches_active = self._candidate_signature_matches(
                active_candidate, observation_alias
            )
        if candidate_matches_active:
            if candidate:
                self._record_candidate_alias(active_candidate, candidate)
            if family in {"focus", "inspect"} and referent:
                self._record_candidate_alias(active_candidate, referent)
            self._record_candidate_alias(active_candidate, observation_alias)
            if family in {"focus", "inspect"} and candidate_room:
                active_state["last_seen_room"] = candidate_room
            elif (
                family in {"relocation", "transfer_or_transform"}
                and destination_signature
                and self._signature_looks_like_room(
                    destination_signature, observation, previous_observation
                )
            ):
                active_state["last_seen_room"] = destination_signature
            elif family in {"relocation", "transfer_or_transform"} and candidate_room:
                active_state["last_seen_room"] = candidate_room

        if (
            family in {"relocation", "transfer_or_transform"}
            and candidate_matches_active
        ):
            if self._observation_confirms_candidate_destination(
                active_candidate, observation
            ):
                active_state["relocated"] = True

        container_confirmed = self._observation_confirms_candidate_container(
            active_candidate, observation
        )
        room_confirmed = self._observation_confirms_candidate_room(
            active_candidate, observation
        )
        if container_confirmed:
            active_state["container_confirmed"] = True
        if room_confirmed:
            active_state["room_confirmed"] = True
        if container_confirmed or room_confirmed:
            active_state["support_confirmed"] = True

        goal_is_satisfied = self._candidate_goal_visibly_satisfied(active_state)
        repeated_confirmation = self._is_repeated_action_observation(
            executed_action, observation, family=family
        )
        support_referent = self._is_support_referent(referent)
        if family in {"focus", "inspect"} and (
            candidate_matches_active or support_referent
        ):
            if goal_was_satisfied and self.task_status != "COMPLETED":
                active_state["post_goal_confirmations"] += 1
            if goal_is_satisfied and (repeated_confirmation or observation_stalled):
                active_state["stalled_confirmations"] += 1

        if (
            goal_is_satisfied
            and self.task_status != "COMPLETED"
            and (
                active_state["post_goal_confirmations"] >= 1
                or active_state["stalled_confirmations"] >= 2
            )
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
            active_state = self._candidate_states.get(self._active_candidate, {})
            if active_state.get("last_seen_room"):
                snapshot["active_candidate_room"] = active_state["last_seen_room"]
            aliases = [
                alias
                for alias in active_state.get("aliases", [])
                if alias != self._active_candidate
            ]
            if aliases:
                snapshot["active_candidate_aliases"] = aliases[-2:]
        if self._rejected_candidates:
            snapshot["rejected_candidates"] = self._rejected_candidates[-3:]
        return snapshot

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
        task_contract = self._get_task_contract()
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

        if self._is_candidate_search_task(task_contract):
            candidate_target = self._get_candidate_action_target(
                executed_action,
                family=family,
                task_contract=task_contract,
            )
            observation_alias = self._extract_observation_candidate_alias(
                observation, family=family
            )
            if self._signature_looks_structural_noncandidate(
                referent, observation=observation
            ) or self._signature_looks_structural_noncandidate(
                observation_alias, observation=observation
            ):
                return
            if (
                "focus" in task_contract.get("required_families", [])
                and not candidate_target
            ):
                return
            if self._is_support_referent(referent, task_contract):
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

    def _record_invalid_referent_attempt(
        self, *, family: str, action: str | None
    ) -> None:
        if not action:
            return
        referent = self._extract_action_primary_object_signature(action, family=family)
        if not referent:
            referent = self._get_action_referent_signature(action, family=family)
        if not referent:
            return
        key = (family, referent)
        self._invalid_referent_attempts[key] = (
            self._invalid_referent_attempts.get(key, 0) + 1
        )

    def _get_invalid_referent_attempts(self, family: str, *referents: str) -> int:
        max_attempts = 0
        for referent in referents:
            if not referent:
                continue
            max_attempts = max(
                max_attempts,
                self._invalid_referent_attempts.get((family, referent), 0),
            )
        return max_attempts

    def _invalidate_action_summary_cache(self) -> None:
        self._admissible_summary_cache.clear()

    @staticmethod
    def _get_shortlist_family_quotas(current_phase: str) -> dict[str, int]:
        quotas_by_phase = {
            "locate_primary_target": {
                "inspect": 4,
                "device_control": 3,
                "relocation": 3,
                "focus": 1,
                "relation": 1,
            },
            "confirm_primary_target": {
                "focus": 3,
                "inspect": 3,
                "device_control": 1,
                "relocation": 1,
            },
            "locate_supporting_source": {
                "inspect": 4,
                "device_control": 3,
                "relocation": 2,
                "focus": 2,
                "relation": 1,
            },
            "inspect_target_mechanism": {
                "inspect": 3,
                "relation": 3,
                "device_control": 2,
                "focus": 2,
                "relocation": 1,
            },
            "integrate_control_or_verify": {
                "device_control": 3,
                "relation": 3,
                "inspect": 3,
                "focus": 1,
                "relocation": 1,
            },
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
            "induce_property_change": {
                "tool_application": 3,
                "device_control": 3,
                "inspect": 2,
                "transfer_or_transform": 2,
                "relocation": 1,
                "focus": 1,
            },
            "verify_transition": {
                "tool_application": 3,
                "inspect": 3,
                "device_control": 2,
                "focus": 1,
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
            "gather_branch_evidence": {
                "inspect": 4,
                "focus": 2,
                "device_control": 2,
                "relocation": 2,
                "transfer_or_transform": 1,
                "tool_application": 1,
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
            "locate_primary_target": {
                "inspect": 8,
                "device_control": 7,
                "relocation": 7,
                "focus": 3,
                "relation": -6,
                "transfer_or_transform": -6,
                "tool_application": -6,
                "other": -3,
                "idle": -5,
            },
            "confirm_primary_target": {
                "focus": 9,
                "inspect": 7,
                "device_control": 3,
                "relocation": 2,
                "relation": -4,
                "transfer_or_transform": -4,
                "tool_application": -4,
                "other": -2,
                "idle": -5,
            },
            "locate_supporting_source": {
                "inspect": 8,
                "device_control": 7,
                "relocation": 6,
                "focus": 4,
                "relation": -2,
                "transfer_or_transform": -4,
                "tool_application": -4,
                "other": -2,
                "idle": -5,
            },
            "inspect_target_mechanism": {
                "inspect": 8,
                "relation": 8,
                "device_control": 6,
                "focus": 4,
                "relocation": 1,
                "transfer_or_transform": -4,
                "tool_application": -4,
                "other": -2,
                "idle": -5,
            },
            "integrate_control_or_verify": {
                "device_control": 9,
                "relation": 8,
                "inspect": 7,
                "focus": 3,
                "relocation": 1,
                "transfer_or_transform": -4,
                "tool_application": -4,
                "other": -2,
                "idle": -5,
            },
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
            "induce_property_change": {
                "tool_application": 7,
                "device_control": 7,
                "inspect": 5,
                "transfer_or_transform": 5,
                "relocation": 2,
                "focus": 1,
                "relation": -3,
                "other": -2,
                "idle": -5,
            },
            "verify_transition": {
                "tool_application": 8,
                "inspect": 7,
                "device_control": 5,
                "focus": 2,
                "relocation": 1,
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
            "gather_branch_evidence": {
                "inspect": 8,
                "focus": 5,
                "device_control": 5,
                "relocation": 5,
                "transfer_or_transform": -2,
                "tool_application": -2,
                "relation": -6,
                "other": -1,
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
        destination_signature = self._extract_action_destination_signature(
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
            ) or self._signature_matches_role(
                destination_signature,
                role_token_sets["measurement_branch_targets"],
            )
            selected_branch_action = bool(
                self._selected_measurement_branch_target
                and self._selected_measurement_branch_target
                in {referent_signature, destination_signature}
            )
            direct_value_entry = self._get_latest_measurement(direct=True)
            if family == "tool_application":
                if direct_measurement:
                    signal += 2
                elif measurement_subject:
                    signal = min(signal, 1)
            if (
                self._measurement_property_requires_event(task_contract)
                and direct_measurement
                and not self._measurement_property_is_resolved(task_contract)
            ):
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
            self._record_invalid_referent_attempt(
                family=family,
                action=executed_action or suggested_action,
            )
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
        room_frontier_action: bool,
        room_search_stalled: bool,
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
        visible_grounded_substance_token_sets = (
            self._get_grounded_substance_token_sets()
        )
        grounded_substance_token_sets = self._get_task_grounded_substance_token_sets(
            task_contract=task_contract
        )
        grounded_substance_hits, _ = self._best_role_overlap(
            content_token_set, grounded_substance_token_sets
        )
        grounded_substance_match = self._has_full_role_match(
            content_token_set, grounded_substance_token_sets
        )
        visible_grounded_substance_match = self._has_full_role_match(
            content_token_set, visible_grounded_substance_token_sets
        )
        non_target_grounded_substance_match = bool(
            visible_grounded_substance_match and not grounded_substance_match
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
        invalid_referent_attempts = self._get_invalid_referent_attempts(
            family,
            referent_signature,
            source_signature,
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
        if non_target_grounded_substance_match:
            if family == "inspect":
                score -= 10
            elif family == "focus":
                score -= 18
            elif family in {"tool_application", "transfer_or_transform", "relation"}:
                score -= 18
            elif family == "relocation":
                score -= 12

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

        if (
            room_search_stalled
            and not substance_grounded
            and current_phase in {"locate_substance", "probe_sources"}
        ):
            if room_frontier_action:
                if family == "inspect":
                    score += 10
                    if normalized.startswith("look around"):
                        score += 4
                elif family == "device_control":
                    score += 14
                elif family == "relocation":
                    score += 22
            elif family == "inspect":
                if source_candidate_match:
                    score += 3
                elif self._is_container_like_action(action, family=family):
                    score -= 12
            elif family == "device_control":
                if source_candidate_match:
                    score += 4
                elif self._is_container_like_action(action, family=family):
                    score -= 14
            elif family == "relocation":
                score -= 6
            elif family == "focus" and not (
                substance_full_match or grounded_substance_match
            ):
                score -= 10

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
        if (
            invalid_referent_attempts
            and current_phase
            in {"locate_substance", "probe_sources", "confirm_referent"}
            and not (substance_role_hits or grounded_substance_match)
        ):
            if family in {"inspect", "device_control"}:
                score -= 10 + invalid_referent_attempts * 6
            elif family == "focus":
                score -= 8 + invalid_referent_attempts * 4

        return score

    def _score_artifact_creation_action(
        self,
        *,
        action: str,
        family: str,
        current_phase: str,
        room_frontier_action: bool,
        room_search_stalled: bool,
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

        if (
            room_search_stalled
            and grounded_artifact_count == 0
            and current_phase
            in {"locate_base_artifact", "find_missing_ingredient_or_reagent"}
        ):
            if room_frontier_action:
                if family == "inspect":
                    score += 16
                    if self._action_mentions_door(action, family=family):
                        score += 6
                    if normalized.startswith("look around"):
                        score += 4
                elif family == "device_control":
                    score += 18
                    if self._action_mentions_door(action, family=family):
                        score += 8
                elif family == "relocation":
                    score += 10
            elif family == "inspect" and self._is_container_like_action(
                action, family=family
            ):
                score -= 18
            elif family == "device_control" and self._is_container_like_action(
                action, family=family
            ):
                score -= 18
            elif family == "relocation":
                score -= 10

        return score

    def _score_relation_mechanism_action(
        self,
        *,
        action: str,
        family: str,
        current_phase: str,
        content_token_set: set[str],
        task_contract: dict,
        role_token_sets: dict[str, list[set[str]]],
        primary_role_hits: int,
        support_role_hits: int,
    ) -> int:
        if not (
            self._is_relation_mechanism_task(task_contract)
            or self._is_inferred_target_search_task(task_contract)
        ):
            return 0

        normalized = self._normalize_runtime_text(action)
        referent_signature = self._get_action_referent_signature(action, family=family)
        primary_signature = self._extract_action_primary_object_signature(
            action, family=family
        )
        relation_lhs, relation_rhs = self._extract_relation_endpoint_signatures(
            action, family=family
        )
        relation_snapshot = self._get_relation_frontier_snapshot(
            task_contract=task_contract
        )
        frontier_referents = relation_snapshot.get("frontier_entities", [])
        support_candidates = relation_snapshot.get("support_candidates", [])
        control_candidates = relation_snapshot.get("control_candidates", [])
        current_observation = (self.percept or {}).get("resulting_observation", "")
        current_room = self._get_current_location_signature(current_observation)
        room_state = self._search_location_states.get(current_room, {})
        visible_doors = self._count_visible_doors(current_observation)

        primary_match = self._signature_matches_role(
            primary_signature, role_token_sets["primary_targets"]
        )
        support_match = self._signature_matches_role(
            primary_signature, role_token_sets["supporting_targets"]
        )
        support_candidate_match = self._signature_matches_frontier(
            primary_signature or referent_signature, support_candidates
        )
        frontier_match = self._signature_matches_frontier(
            primary_signature or referent_signature, frontier_referents
        )
        lhs_frontier_match = self._signature_matches_frontier(
            relation_lhs, frontier_referents
        )
        rhs_frontier_match = self._signature_matches_frontier(
            relation_rhs, frontier_referents
        )
        lhs_support_candidate_match = self._signature_matches_frontier(
            relation_lhs, support_candidates
        )
        rhs_support_candidate_match = self._signature_matches_frontier(
            relation_rhs, support_candidates
        )
        lhs_control_match = (
            relation_lhs in control_candidates
            or self._is_control_candidate_signature(relation_lhs)
        )
        rhs_control_match = (
            relation_rhs in control_candidates
            or self._is_control_candidate_signature(relation_rhs)
        )
        lhs_bridge_match = self._is_relation_bridge_signature(relation_lhs)
        rhs_bridge_match = self._is_relation_bridge_signature(relation_rhs)
        referent_control_match = (
            primary_signature in control_candidates
            or referent_signature in control_candidates
            or self._is_control_candidate_signature(
                primary_signature or referent_signature
            )
        )
        invalid_referent_attempts = self._get_invalid_referent_attempts(
            family, primary_signature, referent_signature
        )
        score = 0

        if current_phase == "locate_primary_target":
            if family == "focus":
                score += 22 if primary_match or primary_role_hits else -14
            elif family == "inspect":
                if normalized.startswith("look around"):
                    score += 10
                elif self._action_mentions_door(action, family=family):
                    score += 8
                elif primary_match or primary_role_hits:
                    score += 12
                else:
                    score += 1
            elif family == "device_control":
                if self._action_mentions_door(action, family=family):
                    score += 14
                else:
                    score -= 10
            elif family == "relocation":
                if primary_match or primary_role_hits:
                    score += 12
                else:
                    score += 6
            elif family in {
                "relation",
                "tool_application",
                "transfer_or_transform",
            }:
                score -= 18

            if (
                visible_doors <= 1
                and room_state.get("local_exploration", 0) >= 1
                and family in {"inspect", "device_control"}
                and not self._action_mentions_door(action, family=family)
                and not (primary_match or primary_role_hits)
            ):
                score -= 16

        elif current_phase == "confirm_primary_target":
            if family == "focus":
                score += 22 if primary_match or primary_role_hits else -14
            elif family == "inspect":
                score += 12 if primary_match or primary_role_hits else -6
            elif family in {
                "relation",
                "tool_application",
                "transfer_or_transform",
            }:
                score -= 14

        elif current_phase == "locate_supporting_source":
            if family == "focus":
                score += (
                    20
                    if support_match or support_candidate_match or support_role_hits
                    else -10
                )
            elif family == "inspect":
                if support_match or support_candidate_match or support_role_hits:
                    score += 18
                    if normalized.startswith(("look at ", "inspect ", "examine ")):
                        score += 4
                else:
                    score += 4
            elif family == "device_control":
                if self._action_mentions_door(action, family=family):
                    score += 12
                elif support_candidate_match:
                    score += 6
                else:
                    score -= 6
            elif family == "relocation":
                score += 6
            elif family == "relation":
                if (lhs_support_candidate_match and rhs_frontier_match) or (
                    rhs_support_candidate_match and lhs_frontier_match
                ):
                    score += 6
                else:
                    score -= 12

            if support_candidates and family in {"inspect", "device_control"}:
                if not (
                    support_match
                    or support_candidate_match
                    or support_role_hits
                    or frontier_match
                ):
                    score -= 6

        elif current_phase == "inspect_target_mechanism":
            if family == "inspect":
                if primary_match or support_match or frontier_match:
                    score += 14
                elif self._action_mentions_door(action, family=family):
                    score -= 8
            elif family == "relation":
                if (
                    (lhs_frontier_match and rhs_frontier_match)
                    or (lhs_control_match and rhs_frontier_match)
                    or (rhs_control_match and lhs_frontier_match)
                ):
                    score += 18
                elif (lhs_frontier_match and rhs_bridge_match) or (
                    rhs_frontier_match and lhs_bridge_match
                ):
                    score += 12
                else:
                    score -= 36
            elif family == "device_control":
                if referent_control_match:
                    if self._is_control_candidate_signature(
                        primary_signature or referent_signature
                    ):
                        score += 18
                    elif support_match:
                        score += 6
                    else:
                        score += 10
                else:
                    score -= 12
            elif family == "focus":
                score += 8 if primary_match or support_match or frontier_match else -8
            elif family in {"relocation", "transfer_or_transform", "tool_application"}:
                score -= 12

        elif current_phase == "integrate_control_or_verify":
            if family == "device_control":
                if referent_control_match:
                    if self._is_control_candidate_signature(
                        primary_signature or referent_signature
                    ):
                        score += 36
                    elif support_match:
                        score += 6
                    else:
                        score += 12
                    if invalid_referent_attempts:
                        score -= 24 + invalid_referent_attempts * 10
                        if support_match and not self._is_control_candidate_signature(
                            primary_signature or referent_signature
                        ):
                            score -= 36
                else:
                    score -= 14
            elif family == "relation":
                if (lhs_control_match and rhs_frontier_match) or (
                    rhs_control_match and lhs_frontier_match
                ):
                    score += 18
                elif (lhs_frontier_match and rhs_bridge_match) or (
                    rhs_frontier_match and lhs_bridge_match
                ):
                    score += 10
                elif lhs_frontier_match and rhs_frontier_match:
                    score += 8
                    if normalized.startswith("disconnect ") and primary_match:
                        score -= 10
                else:
                    score -= 52
            elif family == "inspect":
                score += 12 if referent_control_match or frontier_match else -6
            elif family == "focus":
                score += 4 if primary_match or referent_control_match else -8
            elif family in {"relocation", "transfer_or_transform", "tool_application"}:
                score -= 14

        if (
            family == "device_control"
            and invalid_referent_attempts
            and not referent_control_match
        ):
            score -= 8 + invalid_referent_attempts * 4

        return score

    def _score_comparison_action(
        self,
        *,
        action: str,
        family: str,
        current_phase: str,
        current_room: str,
        room_frontier_action: bool,
        room_search_stalled: bool,
        content_token_set: set[str],
        task_contract: dict,
        role_token_sets: dict[str, list[set[str]]],
        primary_role_hits: int,
        support_role_hits: int,
    ) -> int:
        if not self._is_comparison_task(task_contract):
            return 0

        normalized = self._normalize_runtime_text(action)
        comparison_action_token_set = set(self._extract_comparison_tokens(action))
        referent_signature = self._get_action_referent_signature(action, family=family)
        primary_signature = self._extract_comparison_primary_signature(action)
        destination_signature = self._extract_comparison_destination_signature(action)
        primary_token_set = set(self._extract_comparison_tokens(primary_signature))
        destination_token_set = set(
            self._extract_comparison_tokens(destination_signature)
        )
        target_role_hits, _ = self._best_role_overlap(
            comparison_action_token_set, role_token_sets["comparison_targets"]
        )
        action_matches_target = self._matches_any_role(
            comparison_action_token_set, role_token_sets["comparison_targets"]
        )
        primary_matches_target = self._matches_any_role(
            primary_token_set, role_token_sets["comparison_targets"]
        )
        destination_matches_target = self._matches_any_role(
            destination_token_set, role_token_sets["comparison_targets"]
        )
        selected_target_match = bool(
            self._selected_comparison_target
            and set(
                self._extract_comparison_tokens(self._selected_comparison_target)
            ).issubset(comparison_action_token_set)
        )
        selected_primary_match = bool(
            self._selected_comparison_target
            and set(
                self._extract_comparison_tokens(self._selected_comparison_target)
            ).issubset(primary_token_set)
        )
        room_referent_match = bool(current_room and referent_signature == current_room)
        score = 0

        if current_phase == "locate_primary_target":
            if family == "inspect":
                if target_role_hits or primary_role_hits or support_role_hits:
                    score += 12
                elif room_referent_match:
                    score += 18
                    if normalized.startswith(("look at ", "look in ")):
                        score += 4
                elif normalized.startswith("look around"):
                    score += 10
                elif room_frontier_action:
                    score += 8
                elif self._is_container_like_action(action, family=family):
                    score -= 14
                else:
                    score -= 8
            elif family == "device_control":
                if room_frontier_action:
                    score += 16
                elif self._is_container_like_action(action, family=family):
                    score -= 18
                else:
                    score -= 12
            elif family == "relocation":
                if self._is_agent_navigation_action(action, family=family):
                    score += 18 if room_frontier_action else 4
                else:
                    score -= 24
            elif family == "focus":
                score -= 18
            elif family in {"relation", "transfer_or_transform", "tool_application"}:
                score -= 20

            if room_search_stalled:
                if room_frontier_action:
                    if family == "inspect":
                        score += 8
                    elif family == "device_control":
                        score += 10
                    elif family == "relocation" and self._is_agent_navigation_action(
                        action, family=family
                    ):
                        score += 8
                elif family == "inspect" and self._is_container_like_action(
                    action, family=family
                ):
                    score -= 12
                elif family == "device_control" and self._is_container_like_action(
                    action, family=family
                ):
                    score -= 12

        elif current_phase == "gather_branch_evidence":
            if family == "inspect":
                if action_matches_target or primary_matches_target or target_role_hits:
                    if normalized.startswith("look at "):
                        score += 24
                    elif normalized.startswith("look in "):
                        score -= 30
                    else:
                        score += 8
                elif normalized.startswith("look around"):
                    score += 10
            elif family == "relocation":
                if destination_matches_target and not (
                    action_matches_target and primary_matches_target
                ):
                    score += 22
                elif primary_matches_target:
                    score -= 28
            elif family == "transfer_or_transform":
                if destination_matches_target and not (
                    action_matches_target and primary_matches_target
                ):
                    score += 14
                elif primary_matches_target:
                    score -= 16
            elif family == "focus":
                if action_matches_target or primary_matches_target or target_role_hits:
                    score -= 120
                else:
                    score -= 12
            elif family == "device_control":
                if self._action_mentions_door(action, family=family):
                    score += 8
                else:
                    score -= 6
            elif family in {"relation", "tool_application"}:
                score -= 14

        elif current_phase == "execute_branch":
            if family == "focus":
                score += (
                    150
                    if selected_target_match
                    else 142
                    if selected_primary_match
                    else -90
                    if action_matches_target or primary_matches_target
                    else -18
                )
            elif (
                action_matches_target
                or primary_matches_target
                or destination_matches_target
            ):
                score -= 20

        if (
            self._selected_comparison_target
            and (
                action_matches_target
                or primary_matches_target
                or destination_matches_target
            )
            and not (selected_target_match or selected_primary_match)
        ):
            score -= 36

        return score

    def _score_growth_action(
        self,
        *,
        action: str,
        family: str,
        current_phase: str,
        current_room: str,
        room_frontier_action: bool,
        room_search_stalled: bool,
        grounded_hits: int,
        target_hits: int,
        primary_full_match: bool,
        task_contract: dict,
    ) -> int:
        if not self._is_growth_task(task_contract) or self._is_conditional_branch_task(
            task_contract
        ):
            return 0

        if current_phase not in {"test_mechanism", "commit_to_goal"}:
            return 0

        normalized = self._normalize_runtime_text(action)
        referent_signature = self._get_action_referent_signature(action, family=family)
        primary_signature = self._extract_action_primary_object_signature(
            action, family=family
        )
        destination_signature = self._extract_action_destination_signature(
            action, family=family
        )
        precursor_tokens = set(self._GROWTH_PRECURSOR_TOKENS)
        referent_tokens = self._referent_tokens(referent_signature)
        primary_tokens = self._referent_tokens(primary_signature)
        destination_tokens = self._referent_tokens(destination_signature)
        precursor_referent_match = bool(referent_tokens & precursor_tokens)
        precursor_primary_match = bool(primary_tokens & precursor_tokens)
        precursor_destination_match = bool(destination_tokens & precursor_tokens)
        precursor_to_growth_locus = bool(
            family in {"relocation", "transfer_or_transform"}
            and self._is_container_like_action(action, family=family)
            and (precursor_referent_match or precursor_primary_match)
            and not precursor_destination_match
        )
        room_referent_match = bool(current_room and referent_signature == current_room)

        if not room_search_stalled:
            if (
                family == "focus"
                and precursor_referent_match
                and not primary_full_match
            ):
                return -10
            return 0

        score = 0
        if family == "focus":
            score += (
                10 if primary_full_match else -50 if precursor_referent_match else -32
            )
        elif family == "inspect":
            if room_frontier_action:
                score += 12
                if normalized.startswith("look around"):
                    score += 4
            elif room_referent_match:
                score += 8
            elif self._is_container_like_action(action, family=family):
                score -= 18
            else:
                score -= 10
        elif family == "device_control":
            if room_frontier_action:
                score += 16
            elif self._is_container_like_action(action, family=family):
                score -= 20
            else:
                score -= 10
        elif family == "relocation":
            if room_frontier_action and self._is_agent_navigation_action(
                action, family=family
            ):
                score += 22
            elif precursor_to_growth_locus:
                score += 14
            else:
                score -= 18
        elif family == "transfer_or_transform":
            score += 12 if precursor_to_growth_locus else -24
        elif family == "tool_application":
            score += 4 if (target_hits or grounded_hits) else -12
        elif family == "relation":
            score -= 14

        return score

    def _score_conditional_branch_action(
        self,
        *,
        action: str,
        family: str,
        current_phase: str,
        content_token_set: set[str],
        task_contract: dict,
        role_token_sets: dict[str, list[set[str]]],
        primary_role_hits: int,
        support_role_hits: int,
    ) -> int:
        if not self._is_conditional_branch_task(task_contract):
            return 0

        normalized = self._normalize_runtime_text(action)
        referent_signature = self._get_action_referent_signature(action, family=family)
        primary_signature = self._extract_action_primary_object_signature(
            action, family=family
        )
        destination_signature = self._extract_action_destination_signature(
            action, family=family
        )
        branch_role_hits, _ = self._best_role_overlap(
            content_token_set, role_token_sets["conditional_branch_targets"]
        )
        branch_full_match = self._has_full_role_match(
            content_token_set, role_token_sets["conditional_branch_targets"]
        )
        branch_referent_match = self._signature_matches_role(
            referent_signature, role_token_sets["conditional_branch_targets"]
        )
        branch_primary_match = self._signature_matches_role(
            primary_signature, role_token_sets["conditional_branch_targets"]
        )
        branch_destination_match = self._signature_matches_role(
            destination_signature, role_token_sets["conditional_branch_targets"]
        )
        selected_branch_match = bool(
            self._selected_conditional_branch_target
            and referent_signature == self._selected_conditional_branch_target
        )
        selected_branch_primary_match = bool(
            self._selected_conditional_branch_target
            and primary_signature == self._selected_conditional_branch_target
        )
        selected_branch_destination_match = bool(
            self._selected_conditional_branch_target
            and destination_signature == self._selected_conditional_branch_target
        )
        selected_branch_metadata = self._get_conditional_branch_metadata(
            target=self._selected_conditional_branch_target,
            task_contract=task_contract,
        )
        selected_branch_mode = selected_branch_metadata.get("mode", "focus")
        primary_target_referent_match = self._signature_matches_role(
            referent_signature, role_token_sets["primary_targets"]
        )
        primary_target_primary_match = self._signature_matches_role(
            primary_signature, role_token_sets["primary_targets"]
        )
        selected_branch_destination_commit = bool(
            selected_branch_destination_match
            and (primary_target_referent_match or primary_target_primary_match)
        )
        branch_action = bool(
            branch_full_match
            or branch_referent_match
            or branch_primary_match
            or branch_destination_match
            or branch_role_hits
        )
        growth_branch_task = self._is_growth_conditional_branch_task(task_contract)
        score = 0

        if current_phase == "locate_primary_target":
            if family == "inspect":
                if normalized.startswith("look around"):
                    score += 10
                elif self._action_mentions_door(action, family=family):
                    score += 8
                elif primary_role_hits or support_role_hits:
                    score += 12
            elif family == "device_control":
                score += 16 if self._action_mentions_door(action, family=family) else -8
            elif family == "relocation":
                score += 10 if (primary_role_hits or support_role_hits) else 6
            elif family == "focus":
                score += 8 if (primary_role_hits or support_role_hits) else -8
            elif family in {
                "relation",
                "transfer_or_transform",
                "tool_application",
            }:
                score -= 16
            if branch_action:
                score -= 120

        elif growth_branch_task and current_phase in {
            "test_mechanism",
            "commit_to_goal",
            "confirm_primary_target",
            "verify_outcome",
            "gather_branch_evidence",
        }:
            precursor_referent_match = self._signature_matches_role(
                referent_signature, role_token_sets["supporting_targets"]
            )
            precursor_primary_match = self._signature_matches_role(
                primary_signature, role_token_sets["supporting_targets"]
            )
            precursor_destination_match = self._signature_matches_role(
                destination_signature, role_token_sets["supporting_targets"]
            )
            precursor_action = bool(
                support_role_hits
                or precursor_referent_match
                or precursor_primary_match
                or precursor_destination_match
            )
            container_like_action = self._is_container_like_action(
                action, family=family
            )
            if current_phase == "gather_branch_evidence":
                if family == "inspect":
                    if primary_role_hits:
                        score += 18
                    elif support_role_hits:
                        score += 8
                    elif normalized.startswith("look around"):
                        score += 6
                elif family == "focus":
                    if primary_role_hits:
                        score += 14
                    elif support_role_hits:
                        score += 2
                    else:
                        score -= 6
                elif family == "device_control":
                    if self._action_mentions_door(action, family=family):
                        score += 4
                    elif primary_role_hits or support_role_hits:
                        score += 4
                    else:
                        score -= 6
                elif family == "relocation":
                    score += 8 if primary_role_hits else 2 if support_role_hits else -6
                elif family == "tool_application":
                    score += 8 if (primary_role_hits or support_role_hits) else -10
                elif family == "transfer_or_transform":
                    score -= 14 if branch_action else 4 if primary_role_hits else -18
            else:
                if family == "inspect":
                    if primary_role_hits:
                        score += 14
                    elif support_role_hits or normalized.startswith("look around"):
                        score += 8
                elif family == "focus":
                    if current_phase == "confirm_primary_target" and primary_role_hits:
                        score += 24
                    elif primary_role_hits:
                        score += 8
                    elif support_role_hits:
                        score -= 6
                    else:
                        score -= 10
                elif family == "device_control":
                    if support_role_hits or primary_role_hits or container_like_action:
                        score += 8
                    elif self._action_mentions_door(action, family=family):
                        score += 4
                    else:
                        score -= 10
                elif family == "tool_application":
                    score += 10 if (primary_role_hits or precursor_action) else -12
                elif family == "relation":
                    score += 8 if (primary_role_hits or precursor_action) else -12
                elif family == "relocation":
                    if primary_role_hits:
                        score += 12
                    elif precursor_action and container_like_action:
                        score += 10
                    elif precursor_action:
                        score -= 20
                    else:
                        score -= 10
                elif family == "transfer_or_transform":
                    if primary_role_hits:
                        score += 14
                    elif precursor_action and container_like_action:
                        score += 12
                    elif precursor_action:
                        score -= 24
                    else:
                        score -= 12

            if branch_action:
                score -= 180
                if family == "focus":
                    score -= 120
                elif family in {"relocation", "transfer_or_transform"}:
                    score -= 80

        elif current_phase == "gather_branch_evidence":
            if family == "inspect":
                if primary_role_hits or support_role_hits:
                    score += 18
                elif normalized.startswith("look around"):
                    score += 8
                elif self._action_mentions_door(action, family=family):
                    score += 3
            elif family == "focus":
                if primary_role_hits or support_role_hits:
                    score += 10
                else:
                    score -= 4
            elif family == "device_control":
                if self._action_mentions_door(action, family=family):
                    score += 8
                elif primary_role_hits or support_role_hits:
                    score += 4
                else:
                    score -= 4
            elif family == "relocation":
                if support_role_hits and not primary_role_hits:
                    score += 4
                elif primary_role_hits:
                    score -= (
                        14
                        if self._is_container_like_action(action, family=family)
                        else 8
                    )
                else:
                    score -= 8
            elif family == "relation":
                if "relation" in task_contract.get("required_families", []):
                    score += 10 if (primary_role_hits or support_role_hits) else -8
                else:
                    score += 6 if support_role_hits else -12
            elif family == "tool_application":
                if "tool_application" in task_contract.get("required_families", []):
                    score += 8 if (primary_role_hits or support_role_hits) else -8
                else:
                    score += 6 if support_role_hits else -10
            elif family == "transfer_or_transform":
                score -= 90

            if branch_action:
                score -= 180
                if family == "focus":
                    score -= 120
                elif (
                    family in {"relocation", "transfer_or_transform"}
                    and branch_destination_match
                ):
                    score -= 160

        elif current_phase == "execute_branch":
            if selected_branch_mode == "destination":
                if family in {"relocation", "transfer_or_transform"}:
                    if selected_branch_destination_commit:
                        score += 160
                    elif selected_branch_destination_match:
                        score += 24
                    elif branch_action:
                        score -= 80
                elif family == "focus":
                    if selected_branch_match or selected_branch_primary_match:
                        score -= 18
                    elif branch_action:
                        score -= 82
                    elif primary_role_hits:
                        score += 6
                    else:
                        score -= 12
                elif family == "inspect":
                    if selected_branch_destination_commit:
                        score += 18
                    elif branch_action:
                        score -= 24
                    elif primary_role_hits or support_role_hits:
                        score += 2
                elif branch_action:
                    if selected_branch_destination_match:
                        score -= 8
                    else:
                        score -= 40
            else:
                if family == "focus":
                    score += (
                        140
                        if selected_branch_match
                        else 132
                        if selected_branch_primary_match
                        else -90
                        if branch_action
                        else -20
                    )
                elif family == "inspect" and not branch_action:
                    score -= 30
                elif branch_action:
                    if selected_branch_match or selected_branch_primary_match:
                        score += 20
                    elif selected_branch_destination_match:
                        score += 14
                    else:
                        score -= 40

        if branch_action and not self._selected_conditional_branch_target:
            score -= 40
        if (
            branch_action
            and self._selected_conditional_branch_target
            and not (
                selected_branch_match
                or selected_branch_primary_match
                or selected_branch_destination_match
            )
        ):
            score -= 42

        return score

    def _score_measurement_action(
        self,
        *,
        action: str,
        available_actions: list[str],
        family: str,
        current_phase: str,
        current_room: str,
        room_frontier_action: bool,
        room_search_stalled: bool,
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
        destination_signature = self._extract_action_destination_signature(
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
        branch_destination_match = self._signature_matches_role(
            destination_signature, role_token_sets["measurement_branch_targets"]
        )
        selected_branch_match = bool(
            self._selected_measurement_branch_target
            and referent_signature == self._selected_measurement_branch_target
        )
        selected_branch_destination_match = bool(
            self._selected_measurement_branch_target
            and destination_signature == self._selected_measurement_branch_target
        )
        touches_active_enclosure = bool(
            active_enclosure
            and active_enclosure
            in {
                referent_signature,
                measurement_subject,
                primary_signature,
                destination_signature,
            }
        )
        activates_active_enclosure = bool(
            active_enclosure
            and primary_signature == active_enclosure
            and family == "device_control"
            and normalized.startswith(("activate ", "turn on "))
        )
        deactivates_active_enclosure = bool(
            active_enclosure
            and primary_signature == active_enclosure
            and family == "device_control"
            and normalized.startswith(("deactivate ", "turn off "))
        )
        latest_direct_measurement = self._get_latest_measurement(direct=True)
        direct_measurement_ready = latest_direct_measurement is not None
        property_resolved = self._measurement_property_is_resolved(task_contract)
        control_candidates = set(
            self._get_measurement_control_candidate_signatures(
                actions=available_actions, task_contract=task_contract
            )
        )
        setup_control_candidate = bool(
            self._measurement_property_requires_event(task_contract)
            and direct_measurement_ready
            and not self._measurement_property_event_observed
            and not active_enclosure
            and control_candidates
        )
        control_candidate_action = bool(
            control_candidates
            and (
                referent_signature in control_candidates
                or primary_signature in control_candidates
                or destination_signature in control_candidates
            )
        )
        opens_control_candidate = bool(
            family == "device_control"
            and primary_signature in control_candidates
            and normalized.startswith("open ")
        )
        activates_control_candidate = bool(
            family == "device_control"
            and primary_signature in control_candidates
            and normalized.startswith(("activate ", "turn on "))
        )
        relocates_target_to_control_candidate = bool(
            family in {"relocation", "transfer_or_transform"}
            and primary_signature == measurement_target
            and destination_signature in control_candidates
        )
        room_referent_match = bool(current_room and referent_signature == current_room)
        referent_room_state = self._search_location_states.get(referent_signature, {})
        exhausted_room_referent = bool(
            referent_signature
            and referent_signature != current_room
            and referent_room_state.get("local_exploration", 0) >= 2
            and not referent_room_state.get("target_grounded")
        )
        score = 0

        if current_phase == "locate_instrument":
            if family == "focus":
                if instrument_referent_match:
                    score += 18
                elif referent_matches_target:
                    score -= 48
                else:
                    score -= 32
                if exhausted_room_referent:
                    score -= 18
            elif family == "inspect":
                if instrument_referent_match:
                    score += 12
                elif referent_matches_target or primary_matches_target:
                    score -= 30
                elif exhausted_room_referent:
                    score -= 18
                elif room_referent_match:
                    score += -4 if room_search_stalled else 8
                elif normalized.startswith(
                    "look in "
                ) and self._is_container_like_action(action, family=family):
                    score += -4 if room_search_stalled else 2
                elif self._is_container_like_action(action, family=family):
                    score -= 8 if room_search_stalled else 4
                elif room_frontier_action:
                    score += 8 if room_search_stalled else 0
                else:
                    score -= 12
            elif family == "device_control":
                if instrument_referent_match:
                    score += 10
                elif self._action_mentions_door(action, family=family):
                    score += (
                        -10
                        if exhausted_room_referent
                        else 12
                        if room_search_stalled
                        else 2
                    )
                elif normalized.startswith("open ") and self._is_container_like_action(
                    action, family=family
                ):
                    score += 4 if room_search_stalled else 20
                elif self._is_container_like_action(action, family=family):
                    score += -4 if room_search_stalled else 4
                else:
                    score -= 8
            elif family == "relocation":
                if instrument_referent_match:
                    score += 6
                elif referent_matches_target or primary_matches_target:
                    score -= 24
                elif room_frontier_action:
                    score += (
                        -18
                        if exhausted_room_referent
                        else 14
                        if room_search_stalled
                        else 0
                    )
                else:
                    score -= 14
                    if normalized.startswith(("move ", "pick up ", "take ", "grab ")):
                        score -= 8
            elif family == "tool_application":
                if subject_matches_target:
                    score -= 24
                else:
                    score += 4 if instrument_primary_match else -12
            elif family in {"transfer_or_transform", "relation"}:
                score -= (
                    24 if (referent_matches_target or primary_matches_target) else -16
                )

            if room_search_stalled:
                if room_frontier_action:
                    if exhausted_room_referent:
                        if family == "inspect":
                            score -= 18
                        elif family == "device_control":
                            score -= 18
                        elif family == "relocation":
                            score -= 24
                    elif family == "inspect":
                        score += 10
                    elif family == "device_control":
                        score += 12
                    elif family == "relocation":
                        score += 14
                elif family == "inspect" and not instrument_referent_match:
                    if room_referent_match:
                        score -= 18
                    elif self._is_container_like_action(action, family=family):
                        score -= 10
                    else:
                        score -= 24
                elif family == "focus" and not instrument_referent_match:
                    score -= (
                        36 if room_referent_match or exhausted_room_referent else 24
                    )
                elif family == "device_control" and not instrument_referent_match:
                    if self._is_container_like_action(action, family=family):
                        score -= 14
                    else:
                        score -= 8

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

        elif current_phase == "induce_property_change":
            if family == "tool_application":
                if subject_matches_target and instrument_primary_match:
                    score += 4
                    if target_hidden:
                        score -= 30
                elif touches_active_enclosure and instrument_primary_match:
                    score -= 18 if target_hidden else 8
                elif subject_matches_target or touches_active_enclosure:
                    score += 14
                else:
                    score -= 8
                if setup_control_candidate and subject_matches_target:
                    score -= 42 if instrument_primary_match else 32
            elif family == "device_control":
                if self._action_mentions_door(action, family=family):
                    score -= 8
                elif deactivates_active_enclosure:
                    score -= 50 if target_hidden else 30
                elif activates_active_enclosure:
                    score += 36 if target_hidden else 22
                elif referent_matches_target or touches_active_enclosure:
                    score += 12
                else:
                    score += 6
                if setup_control_candidate:
                    if opens_control_candidate:
                        score += 62
                    elif activates_control_candidate:
                        score += 54
                    elif control_candidate_action:
                        score += 18
            elif family in {"transfer_or_transform", "relocation"}:
                if (
                    referent_matches_target
                    or primary_matches_target
                    or touches_active_enclosure
                ):
                    score += 10
                else:
                    score -= 6
                if setup_control_candidate and relocates_target_to_control_candidate:
                    score += 34
            elif family == "inspect":
                if referent_matches_target or touches_active_enclosure:
                    score += 10
                else:
                    score += 2
                if setup_control_candidate and referent_signature in control_candidates:
                    score += 20
            elif family == "focus":
                score += (
                    4 if referent_matches_target or instrument_referent_match else -8
                )

        elif current_phase == "verify_transition":
            if family == "tool_application":
                if subject_matches_target and instrument_primary_match:
                    score += 24
                    if target_hidden:
                        score -= 30
                elif touches_active_enclosure and instrument_primary_match:
                    score += 10
                else:
                    score -= 8
            elif family == "inspect":
                if referent_matches_target or touches_active_enclosure:
                    score += 10
            elif family in {"device_control", "transfer_or_transform", "relocation"}:
                if (
                    referent_matches_target
                    or primary_matches_target
                    or touches_active_enclosure
                ):
                    score += 6
                else:
                    score -= 6
            elif family == "focus":
                score += (
                    4 if referent_matches_target or instrument_referent_match else -8
                )

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
                if selected_branch_match or selected_branch_destination_match:
                    score += 12
                elif branch_referent_match or branch_destination_match:
                    score -= 8
            elif branch_referent_match or branch_destination_match:
                score -= 6

        if (
            branch_referent_match or branch_destination_match
        ) and not self._selected_measurement_branch_target:
            score -= 18
        if (branch_referent_match or branch_destination_match) and not (
            selected_branch_match or selected_branch_destination_match
        ):
            score -= 10
        if (
            self._measurement_property_requires_event(task_contract)
            and not property_resolved
            and (branch_referent_match or branch_destination_match)
        ):
            score -= 36

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
            self._measurement_property_requires_event(task_contract)
            and not property_resolved
            and family == "tool_application"
            and instrument_primary_match
            and touches_active_enclosure
            and not subject_matches_target
        ):
            score -= 28 if not self._measurement_property_event_observed else 18
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
        if self._is_growth_conditional_branch_task(task_contract):
            current_observation = (self.percept or {}).get("resulting_observation", "")
            grounded_tokens = set(self._get_observation_grounded_tokens())
            role_token_sets = self._get_task_role_token_sets(task_contract)
            if self._selected_conditional_branch_target:
                return "execute_branch"
            if self._primary_target_is_grounded(task_contract, grounded_tokens):
                if "focus" in task_contract.get(
                    "required_families", []
                ) and not self._role_focus_completed(
                    role_token_sets["primary_targets"]
                ):
                    return "confirm_primary_target"
                return "gather_branch_evidence"
            if not self._growth_task_has_precursor_signal(
                current_observation,
                task_contract=task_contract,
                grounded_tokens=grounded_tokens,
            ):
                return "locate_primary_target"
            mechanism_progress = any(
                entry["observable_change_attempts"] > 0
                for entry in self.episode_hypothesis_ledger.values()
                if entry["family"]
                in {
                    "relation",
                    "tool_application",
                    "transfer_or_transform",
                    "device_control",
                    "relocation",
                }
            )
            if mechanism_progress:
                return "commit_to_goal"
            return "test_mechanism"
        if self._is_comparison_task(task_contract):
            grounded_tokens = set(self._get_observation_grounded_tokens())
            if not (
                self._primary_target_is_grounded(task_contract, grounded_tokens)
                or self._comparison_target_is_grounded(task_contract)
            ):
                return "locate_primary_target"
            if self._selected_comparison_target:
                return "execute_branch"
            return "gather_branch_evidence"
        if self._is_conditional_branch_task(task_contract):
            grounded_tokens = set(self._get_observation_grounded_tokens())
            if task_contract.get(
                "primary_targets"
            ) and not self._primary_target_is_grounded(task_contract, grounded_tokens):
                return "locate_primary_target"
            if self._selected_conditional_branch_target:
                return "execute_branch"
            return "gather_branch_evidence"
        if self._is_relation_mechanism_task(task_contract):
            grounded_tokens = set(self._get_observation_grounded_tokens())
            role_token_sets = self._get_task_role_token_sets(task_contract)
            relation_snapshot = self._get_relation_frontier_snapshot(
                task_contract=task_contract
            )
            relation_entry = self.episode_hypothesis_ledger.get("relation", {})
            relation_tests = relation_entry.get("tests", 0)
            if not self._primary_target_is_grounded(task_contract, grounded_tokens):
                return "locate_primary_target"
            if "focus" in task_contract.get(
                "required_families", []
            ) and not self._role_focus_completed(role_token_sets["primary_targets"]):
                return "confirm_primary_target"
            if task_contract.get(
                "supporting_targets"
            ) and not self._supporting_target_is_grounded(
                task_contract, grounded_tokens
            ):
                return "locate_supporting_source"
            if relation_tests > 0 and relation_snapshot.get("target_status") == "off":
                return "integrate_control_or_verify"
            return "inspect_target_mechanism"
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
            if self._measurement_property_requires_event(task_contract):
                if not self._measurement_property_event_observed:
                    return "induce_property_change"
                if not self._measurement_property_is_resolved(task_contract):
                    return "verify_transition"
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
        if self._is_lifecycle_task(task_contract):
            current_observation = (self.percept or {}).get("resulting_observation", "")
            grounded_tokens = set(self._get_observation_grounded_tokens())
            current_location_tokens = set(
                self._extract_current_location_tokens(current_observation)
            )
            visible_target_tokens = (
                grounded_tokens & set(task_contract.get("target_entities", []))
            ) - current_location_tokens
            visible_stage_labels = self._get_visible_lifecycle_stage_labels(
                current_observation
            )
            visible_unfocused_labels = [
                label
                for label in visible_stage_labels
                if label not in self._focused_stage_labels
            ]
            if visible_unfocused_labels or (
                visible_target_tokens
                and not visible_stage_labels
                and not self._focused_stage_labels
            ):
                return "confirm_primary_target"
            return "locate_primary_target"
        if self._is_growth_task(task_contract):
            current_observation = (self.percept or {}).get("resulting_observation", "")
            grounded_tokens = set(self._get_observation_grounded_tokens())
            role_token_sets = self._get_task_role_token_sets(task_contract)
            if self._primary_target_is_grounded(task_contract, grounded_tokens):
                if "focus" in task_contract.get(
                    "required_families", []
                ) and not self._role_focus_completed(
                    role_token_sets["primary_targets"]
                ):
                    return "confirm_primary_target"
                return "verify_outcome"
            if not self._growth_task_has_precursor_signal(
                current_observation,
                task_contract=task_contract,
                grounded_tokens=grounded_tokens,
            ):
                return "locate_primary_target"
            mechanism_progress = any(
                entry["observable_change_attempts"] > 0
                for entry in self.episode_hypothesis_ledger.values()
                if entry["family"]
                in {
                    "relation",
                    "tool_application",
                    "transfer_or_transform",
                    "device_control",
                }
            )
            if mechanism_progress:
                return "commit_to_goal"
            return "test_mechanism"
        if self._is_inferred_target_search_task(task_contract):
            grounded_tokens = set(self._get_observation_grounded_tokens())
            role_token_sets = self._get_task_role_token_sets(task_contract)
            if not self._primary_target_is_grounded(task_contract, grounded_tokens):
                return "locate_primary_target"
            if "focus" in task_contract.get(
                "required_families", []
            ) and not self._role_focus_completed(role_token_sets["primary_targets"]):
                return "confirm_primary_target"
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
        available_actions: list[str],
        current_phase: str,
        task_keywords: list[str],
        grounded_tokens: list[str],
        scoring_context: dict | None = None,
    ) -> tuple[int, int, int]:
        scoring_context = (
            scoring_context
            if scoring_context is not None
            else self._build_shortlist_scoring_context(
                current_phase=current_phase,
                task_keywords=task_keywords,
                grounded_tokens=grounded_tokens,
            )
        )
        normalized = self._normalize_runtime_text(action)
        family = self._classify_action_family(action)
        action_tokens = self._extract_runtime_tokens(action)
        content_tokens = self._extract_action_content_tokens(action, family=family)
        action_token_set = set(action_tokens)
        content_token_set = set(content_tokens)
        grounded_token_set = scoring_context["grounded_token_set"]
        task_keyword_set = scoring_context["task_keyword_set"]
        task_contract = scoring_context["task_contract"]
        role_token_sets = scoring_context["role_token_sets"]
        required_families = scoring_context["required_families"]
        support_families = scoring_context["support_families"]
        target_entity_set = scoring_context["target_entity_set"]
        current_observation = scoring_context["current_observation"]
        current_location_tokens = scoring_context["current_location_tokens"]
        current_room = scoring_context["current_room"]
        remote_room_signal = scoring_context["remote_room_signal"]
        remote_room_tokens = scoring_context["remote_room_tokens"]
        room_state = scoring_context["room_state"]
        visible_doors = scoring_context["visible_doors"]
        visible_nonlocation_targets = scoring_context["visible_nonlocation_targets"]
        ordered_sequence = scoring_context["ordered_sequence"]
        referent_signature = self._get_action_referent_signature(action, family=family)
        candidate_search_task = scoring_context["candidate_search_task"]
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
        primary_target_grounded = scoring_context["primary_target_grounded"]
        primary_target_focused = scoring_context["primary_target_focused"]
        growth_task = scoring_context["growth_task"]
        state_change_task = scoring_context["state_change_task"]
        artifact_creation_task = scoring_context["artifact_creation_task"]
        relation_mechanism_task = scoring_context["relation_mechanism_task"]
        inferred_target_search_task = scoring_context["inferred_target_search_task"]
        measurement_task = scoring_context["measurement_task"]
        comparison_task = scoring_context["comparison_task"]
        conditional_branch_task = scoring_context["conditional_branch_task"]
        lifecycle_task = scoring_context["lifecycle_task"]
        unresolved_conditional_branch_evidence = bool(
            conditional_branch_task
            and current_phase == "gather_branch_evidence"
            and not self._selected_conditional_branch_target
        )
        lifecycle_targets_visible = scoring_context["lifecycle_targets_visible"]
        room_frontier_action = self._action_targets_room_frontier(
            action, current_observation, family=family
        )
        agent_navigation_action = self._is_agent_navigation_action(
            action, family=family
        )
        remote_room_match = bool(
            remote_room_tokens and remote_room_tokens.issubset(content_token_set)
        )
        remote_room_navigation_action = bool(
            remote_room_match and agent_navigation_action
        )
        remote_room_door_action = bool(
            remote_room_match
            and family == "device_control"
            and self._action_mentions_door(action, family=family)
        )
        frontier_room_target = (
            self._get_room_frontier_target(action, current_observation, family=family)
            if room_frontier_action
            else ""
        )
        frontier_room_exhausted = bool(
            frontier_room_target
            and frontier_room_target != current_room
            and (self._search_location_states.get(frontier_room_target, {})).get(
                "local_exploration", 0
            )
            >= 1
            and not (self._search_location_states.get(frontier_room_target, {})).get(
                "target_grounded"
            )
        )
        state_change_room_search_stalled = scoring_context[
            "state_change_room_search_stalled"
        ]
        measurement_instrument_search_stalled = scoring_context[
            "measurement_instrument_search_stalled"
        ]
        artifact_room_search_stalled = scoring_context["artifact_room_search_stalled"]
        growth_room_search_stalled = scoring_context["growth_room_search_stalled"]
        comparison_room_search_stalled = scoring_context[
            "comparison_room_search_stalled"
        ]
        stage_labels = (
            self._get_focus_stage_labels(action, "")
            if lifecycle_task and family == "focus"
            else self._merge_stage_labels(self._extract_stage_labels(action))
            if lifecycle_task
            else []
        )
        next_expected_stage = scoring_context["next_expected_stage"]
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
        if (
            family == "relocation"
            and grounded_hits
            and not (unresolved_conditional_branch_evidence and primary_role_hits)
        ):
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
            if not lifecycle_targets_visible:
                if family == "inspect":
                    if normalized.startswith("look around"):
                        score += 10
                    elif room_frontier_action:
                        score += 8
                    elif self._is_container_like_action(action, family=family):
                        score -= 14
                    else:
                        score -= 6
                elif family == "device_control":
                    if room_frontier_action:
                        score += 14
                    elif self._is_container_like_action(action, family=family):
                        score -= 18
                    else:
                        score -= 10
                elif family == "relocation":
                    if agent_navigation_action:
                        score += 12 if room_frontier_action else 2
                    else:
                        score -= 40
                elif family in {
                    "relation",
                    "transfer_or_transform",
                    "tool_application",
                }:
                    score -= 20

                if frontier_room_exhausted:
                    if family == "relocation":
                        score -= 24
                    elif family in {"inspect", "device_control"}:
                        score -= 16

                if (
                    visible_doors <= 1
                    and room_state.get("local_exploration", 0) >= 1
                    and family in {"inspect", "device_control"}
                    and not room_frontier_action
                ):
                    score -= 18

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
            active_candidate_room = active_candidate_state.get("last_seen_room", "")
            active_candidate_room_tokens = (
                self._referent_tokens(active_candidate_room)
                if active_candidate_room
                else set()
            )
            active_candidate_visible = bool(
                self._active_candidate
                and self._candidate_observed(
                    self._active_candidate, current_observation
                )
            )
            active_candidate_match = bool(
                self._active_candidate
                and (
                    self._candidate_signature_matches(
                        self._active_candidate, candidate_target
                    )
                    or self._candidate_signature_matches(
                        self._active_candidate, referent_signature
                    )
                )
            )
            reacquire_active_candidate = (
                bool(self._active_candidate)
                and bool(active_candidate_room)
                and current_room != active_candidate_room
                and not active_candidate_visible
                and not active_candidate_state.get("relocated")
            )
            room_reacquisition_action = bool(
                active_candidate_room_tokens
                and active_candidate_room_tokens.issubset(content_token_set)
            )
            destination_container_referent = self._signature_matches_role(
                referent_signature, role_token_sets["destination_container"]
            )
            destination_room_referent = self._signature_matches_role(
                referent_signature, role_token_sets["destination_room"]
            )
            candidate_reset_required = bool(
                self._rejected_candidates and not self._active_candidate
            )
            candidate_identification_pending = bool(
                "focus" in required_families and not self._active_candidate
            )
            room_like_referent = self._signature_looks_like_room(
                referent_signature,
                current_observation,
                current_observation,
            )
            if self._snapshot_has_signal(remote_room_signal, key="room"):
                if family == "relocation":
                    if remote_room_navigation_action:
                        score += 40
                    elif agent_navigation_action:
                        score -= 12
                    else:
                        score -= 16
                elif family == "device_control":
                    if remote_room_door_action:
                        score += 22
                    elif self._action_mentions_door(action, family=family):
                        score -= 10
                elif family == "inspect":
                    if normalized.startswith("look in ") and remote_room_match:
                        score -= 14
                    elif remote_room_match:
                        score -= 10
                    elif not support_referent:
                        score -= 10
                elif family == "focus":
                    score -= 12 if remote_room_match else 18
                elif family in {
                    "relation",
                    "transfer_or_transform",
                    "tool_application",
                }:
                    score -= 16
            if candidate_identification_pending:
                if family == "focus":
                    if candidate_target:
                        score += 22
                    elif support_referent or room_like_referent:
                        score -= 28
                    elif referent_signature:
                        score -= 12
                elif family == "inspect" and candidate_target:
                    score += 12
                elif (
                    family in {"relocation", "transfer_or_transform"}
                    and not agent_navigation_action
                ):
                    if candidate_target:
                        score -= 48
                    elif support_referent:
                        score -= 18
                    elif referent_signature:
                        score -= 22
            if family == "focus" and support_referent:
                score -= 24
            elif family == "inspect" and support_referent:
                score -= 8

            if active_candidate_match:
                if family == "relocation":
                    score += 14
                    if support_role_hits:
                        score += 6
                elif family == "transfer_or_transform":
                    score += 8
                elif family == "inspect" and not active_candidate_state.get(
                    "support_confirmed"
                ):
                    score += 4

            if reacquire_active_candidate:
                if family == "relocation" and room_reacquisition_action:
                    score += 18
                elif family == "device_control" and room_reacquisition_action:
                    score += 14
                elif family == "inspect" and room_reacquisition_action:
                    score += 8
                elif support_role_hits and family in {"relocation", "device_control"}:
                    score -= 8

            if candidate_reset_required:
                if candidate_target:
                    if candidate_target in self._rejected_candidates:
                        if family in {
                            "focus",
                            "inspect",
                            "relocation",
                            "transfer_or_transform",
                        }:
                            score -= 50
                    elif family == "focus":
                        score += 18
                    elif family == "inspect":
                        score += 12
                    elif family == "relocation":
                        if "focus" in required_families:
                            score -= 60
                        else:
                            score += 4
                elif destination_container_referent:
                    if family in {"focus", "inspect"}:
                        score -= 80
                    elif family in {
                        "relocation",
                        "transfer_or_transform",
                        "tool_application",
                        "relation",
                    }:
                        score -= 20
                elif destination_room_referent:
                    if family == "inspect":
                        score += 10
                    elif family == "focus":
                        score -= 12

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
        elif growth_task and not conditional_branch_task:
            score += self._score_growth_action(
                action=action,
                family=family,
                current_phase=current_phase,
                current_room=current_room,
                room_frontier_action=room_frontier_action,
                room_search_stalled=growth_room_search_stalled,
                grounded_hits=grounded_hits,
                target_hits=target_hits,
                primary_full_match=primary_full_match,
                task_contract=task_contract,
            )
        elif comparison_task:
            score += self._score_comparison_action(
                action=action,
                family=family,
                current_phase=current_phase,
                current_room=current_room,
                room_frontier_action=room_frontier_action,
                room_search_stalled=comparison_room_search_stalled,
                content_token_set=content_token_set,
                task_contract=task_contract,
                role_token_sets=role_token_sets,
                primary_role_hits=primary_role_hits,
                support_role_hits=support_role_hits,
            )
        elif conditional_branch_task:
            score += self._score_conditional_branch_action(
                action=action,
                family=family,
                current_phase=current_phase,
                content_token_set=content_token_set,
                task_contract=task_contract,
                role_token_sets=role_token_sets,
                primary_role_hits=primary_role_hits,
                support_role_hits=support_role_hits,
            )
            if unresolved_conditional_branch_evidence:
                if (
                    family == "relocation"
                    and primary_role_hits
                    and not support_role_hits
                ):
                    score -= 36
                    if self._is_container_like_action(action, family=family):
                        score -= 8
                elif (
                    family == "relation"
                    and "relation" not in required_families
                    and primary_role_hits
                    and not support_role_hits
                ):
                    score -= 30
                elif (
                    family == "tool_application"
                    and "tool_application" not in required_families
                    and primary_role_hits
                    and not support_role_hits
                ):
                    score -= 24
        elif measurement_task:
            score += self._score_measurement_action(
                action=action,
                available_actions=available_actions,
                family=family,
                current_phase=current_phase,
                current_room=current_room,
                room_frontier_action=room_frontier_action,
                room_search_stalled=measurement_instrument_search_stalled,
                content_token_set=content_token_set,
                grounded_token_set=grounded_token_set,
                task_contract=task_contract,
                role_token_sets=role_token_sets,
            )
        elif relation_mechanism_task or inferred_target_search_task:
            score += self._score_relation_mechanism_action(
                action=action,
                family=family,
                current_phase=current_phase,
                content_token_set=content_token_set,
                task_contract=task_contract,
                role_token_sets=role_token_sets,
                primary_role_hits=primary_role_hits,
                support_role_hits=support_role_hits,
            )
        elif artifact_creation_task:
            score += self._score_artifact_creation_action(
                action=action,
                family=family,
                current_phase=current_phase,
                room_frontier_action=room_frontier_action,
                room_search_stalled=artifact_room_search_stalled,
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
                room_frontier_action=room_frontier_action,
                room_search_stalled=state_change_room_search_stalled,
                content_token_set=content_token_set,
                grounded_token_set=grounded_token_set,
                task_contract=task_contract,
                role_token_sets=role_token_sets,
                grounded_hits=grounded_hits,
                target_hits=target_hits,
                primary_role_hits=primary_role_hits,
                support_role_hits=support_role_hits,
            )

        if (
            primary_target_focused
            and family == "relation"
            and not (unresolved_conditional_branch_evidence)
        ):
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

    def _build_shortlist_scoring_context(
        self,
        *,
        current_phase: str,
        task_keywords: list[str],
        grounded_tokens: list[str],
    ) -> dict:
        task_contract = self._get_task_contract()
        role_token_sets = self._get_task_role_token_sets(task_contract)
        grounded_token_set = set(grounded_tokens)
        task_keyword_set = set(task_keywords)
        target_entity_set = set(task_contract.get("target_entities", []))
        current_observation = (self.percept or {}).get("resulting_observation", "")
        current_location_tokens = set(
            self._extract_current_location_tokens(current_observation)
        )
        current_room = self._get_current_location_signature(current_observation)
        remote_room_signal = self._get_remote_room_signal_snapshot()
        remote_room = remote_room_signal.get("room", "")
        remote_room_tokens = (
            self._referent_tokens(remote_room) if remote_room else set()
        )
        visible_doors = self._count_visible_doors(current_observation)
        visible_nonlocation_targets = (
            grounded_token_set & target_entity_set
        ) - current_location_tokens
        candidate_search_task = self._is_candidate_search_task(task_contract)
        state_change_task = self._is_state_change_task(task_contract)
        artifact_creation_task = self._is_artifact_creation_task(task_contract)
        relation_mechanism_task = self._is_relation_mechanism_task(task_contract)
        inferred_target_search_task = self._is_inferred_target_search_task(
            task_contract
        )
        measurement_task = self._is_measurement_task(task_contract)
        comparison_task = self._is_comparison_task(task_contract)
        conditional_branch_task = self._is_conditional_branch_task(task_contract)
        lifecycle_task = self._is_lifecycle_task(task_contract)
        growth_task = self._is_growth_task(task_contract)
        return {
            "current_phase": current_phase,
            "task_keyword_set": task_keyword_set,
            "grounded_token_set": grounded_token_set,
            "task_contract": task_contract,
            "role_token_sets": role_token_sets,
            "required_families": set(task_contract.get("required_families", [])),
            "support_families": set(task_contract.get("support_families", [])),
            "target_entity_set": target_entity_set,
            "current_observation": current_observation,
            "current_location_tokens": current_location_tokens,
            "current_room": current_room,
            "remote_room_signal": remote_room_signal,
            "remote_room_tokens": remote_room_tokens,
            "room_state": self._search_location_states.get(current_room, {}),
            "visible_doors": visible_doors,
            "visible_nonlocation_targets": visible_nonlocation_targets,
            "ordered_sequence": "ordered_sequence"
            in task_contract.get("ordering_cues", []),
            "candidate_search_task": candidate_search_task,
            "growth_task": growth_task,
            "state_change_task": state_change_task,
            "artifact_creation_task": artifact_creation_task,
            "relation_mechanism_task": relation_mechanism_task,
            "inferred_target_search_task": inferred_target_search_task,
            "measurement_task": measurement_task,
            "comparison_task": comparison_task,
            "conditional_branch_task": conditional_branch_task,
            "lifecycle_task": lifecycle_task,
            "lifecycle_targets_visible": bool(visible_nonlocation_targets)
            or bool(self._get_visible_lifecycle_stage_labels(current_observation)),
            "primary_target_grounded": self._role_is_grounded(
                grounded_token_set, role_token_sets["primary_targets"]
            ),
            "primary_target_focused": self._role_focus_completed(
                role_token_sets["primary_targets"]
            ),
            "state_change_room_search_stalled": self._state_change_room_search_stalled(
                current_room=current_room,
                visible_doors=visible_doors,
                task_contract=task_contract,
                grounded_tokens=grounded_token_set,
            )
            if state_change_task
            else False,
            "measurement_instrument_search_stalled": self._measurement_instrument_search_stalled(
                current_room=current_room,
                visible_doors=visible_doors,
                task_contract=task_contract,
                grounded_tokens=grounded_token_set,
            )
            if measurement_task
            else False,
            "artifact_room_search_stalled": self._artifact_creation_room_search_stalled(
                current_room=current_room,
                visible_doors=visible_doors,
                task_contract=task_contract,
            )
            if artifact_creation_task
            else False,
            "growth_room_search_stalled": self._growth_room_search_stalled(
                current_room=current_room,
                visible_doors=visible_doors,
                task_contract=task_contract,
                grounded_tokens=grounded_token_set,
            )
            if growth_task
            else False,
            "comparison_room_search_stalled": self._comparison_room_search_stalled(
                current_room=current_room,
                visible_doors=visible_doors,
                task_contract=task_contract,
                grounded_tokens=grounded_token_set,
            )
            if comparison_task
            else False,
            "next_expected_stage": self._get_next_expected_stage_label()
            if lifecycle_task
            else None,
        }

    def _summarize_admissible_actions_uncached(
        self,
        actions: list[str],
        *,
        shortlist_limit: int = 12,
    ) -> dict:
        family_counts: dict[str, int] = {}
        current_phase = self._get_current_phase()
        task_keywords = self._extract_task_keywords()
        grounded_tokens = self._get_observation_grounded_tokens()
        task_contract = self._get_task_contract()
        remote_room_signal = self._get_remote_room_signal_snapshot()
        role_token_sets = self._get_task_role_token_sets(task_contract)
        scoring_context = self._build_shortlist_scoring_context(
            current_phase=current_phase,
            task_keywords=task_keywords,
            grounded_tokens=grounded_tokens,
        )
        scored_actions = []

        for action in actions:
            family = self._classify_action_family(action)
            family_counts[family] = family_counts.get(family, 0) + 1
            content_token_set = set(
                self._extract_action_content_tokens(action, family=family)
            )
            primary_role_hits, _ = self._best_role_overlap(
                content_token_set, role_token_sets["primary_targets"]
            )
            support_role_hits, _ = self._best_role_overlap(
                content_token_set, role_token_sets["supporting_targets"]
            )
            score, grounded_hits, family_priority = self._score_action_for_shortlist(
                action,
                available_actions=actions,
                current_phase=current_phase,
                task_keywords=task_keywords,
                grounded_tokens=grounded_tokens,
                scoring_context=scoring_context,
            )
            scored_actions.append(
                {
                    "action": action,
                    "family": family,
                    "score": score,
                    "grounded_hits": grounded_hits,
                    "family_priority": family_priority,
                    "primary_role_hits": primary_role_hits,
                    "support_role_hits": support_role_hits,
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

        relation_frontier = self._get_relation_frontier_snapshot(actions, task_contract)
        relation_commit_candidates: list[str] = []
        relation_commit_ready = bool(
            "relation" in task_contract.get("required_families", [])
            and self._role_focus_completed(role_token_sets["primary_targets"])
            and self._primary_relation_component_is_grounded(
                task_contract=task_contract, role_token_sets=role_token_sets
            )
        )
        if relation_commit_ready:
            for item in scored_actions:
                if item["score"] <= 0:
                    continue
                if not self._is_relation_commit_candidate_action(
                    item["action"],
                    family=item["family"],
                    task_contract=task_contract,
                    role_token_sets=role_token_sets,
                    control_candidates=relation_frontier.get("control_candidates", []),
                ):
                    continue
                relation_commit_candidates.append(item["action"])
                if len(relation_commit_candidates) >= 4:
                    break
            relation_commit_ready = bool(relation_commit_candidates)
            if relation_commit_ready:
                relation_frontier = dict(relation_frontier)
                relation_frontier["commit_ready"] = True
                relation_frontier["commit_candidates"] = relation_commit_candidates[:4]

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
        if "relation" in task_contract.get(
            "required_families", []
        ) and self._role_focus_completed(role_token_sets["primary_targets"]):
            quotas = dict(quotas)
            quotas["relation"] = max(2, quotas.get("relation", 0))
            if quotas.get("focus", 0) > 0:
                quotas["focus"] -= 1
        if relation_commit_ready:
            quotas = dict(quotas)
            quotas["relation"] = max(3, quotas.get("relation", 0))
            quotas["device_control"] = max(2, quotas.get("device_control", 0))
            quotas["inspect"] = min(1, quotas.get("inspect", 0))
        if self._is_candidate_search_task(task_contract):
            quotas = dict(quotas)
            quotas["inspect"] = max(2, quotas.get("inspect", 0))
            if self._rejected_candidates:
                quotas["focus"] = max(2, quotas.get("focus", 0))
                quotas["relocation"] = max(2, quotas.get("relocation", 0))
        if self._snapshot_has_signal(remote_room_signal, key="room"):
            quotas = dict(quotas)
            quotas["relocation"] = max(1, quotas.get("relocation", 0))
            quotas["device_control"] = max(1, quotas.get("device_control", 0))
            if quotas.get("focus", 0) > 0:
                quotas["focus"] -= 1
        unresolved_conditional_branch = bool(
            self._is_conditional_branch_task(task_contract)
            and not self._selected_conditional_branch_target
            and current_phase != "execute_branch"
        )
        lifecycle_search_mode = False
        if self._is_lifecycle_task(task_contract):
            current_observation = (self.percept or {}).get("resulting_observation", "")
            grounded_token_set = set(grounded_tokens)
            current_location_tokens = set(
                self._extract_current_location_tokens(current_observation)
            )
            target_entity_set = set(task_contract.get("target_entities", []))
            lifecycle_targets_visible = bool(
                (grounded_token_set & target_entity_set) - current_location_tokens
            ) or bool(self._get_visible_lifecycle_stage_labels(current_observation))
            if not lifecycle_targets_visible:
                lifecycle_search_mode = True
                quotas = dict(quotas)
                quotas["relocation"] = (
                    1
                    if any(
                        self._is_agent_navigation_action(
                            action, family=self._classify_action_family(action)
                        )
                        for action in actions
                    )
                    else 0
                )
        blocked_actions = {
            item["action"]
            for item in scored_actions
            if lifecycle_search_mode
            and item["family"] == "relocation"
            and not self._is_agent_navigation_action(
                item["action"], family=item["family"]
            )
        }
        if unresolved_conditional_branch:
            blocked_actions.update(
                item["action"]
                for item in scored_actions
                if self._action_targets_conditional_branch(
                    item["action"],
                    family=item["family"],
                    task_contract=task_contract,
                    role_token_sets=role_token_sets,
                )
            )
            if current_phase == "gather_branch_evidence":
                blocked_actions.update(
                    item["action"]
                    for item in scored_actions
                    if item["family"] in {"relocation", "relation", "tool_application"}
                    and item["family"] not in task_contract.get("required_families", [])
                    and item["primary_role_hits"] > 0
                    and item["support_role_hits"] == 0
                )
        if relation_commit_ready:
            blocked_actions.update(
                item["action"]
                for item in scored_actions
                if self._is_primary_relation_inspect_action(
                    item["action"],
                    task_contract=task_contract,
                    role_token_sets=role_token_sets,
                )
            )
        shortlist: list[str] = []
        selected_actions: set[str] = set()
        selected_by_family: dict[str, int] = {}

        for item in scored_actions:
            if item["action"] in blocked_actions:
                continue
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
            if item["action"] in blocked_actions:
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
        comparison_tracking = self._get_comparison_tracking_snapshot()
        conditional_branch_tracking = self._get_conditional_branch_tracking_snapshot()

        return {
            "total_actions": len(actions),
            "current_phase": current_phase,
            "family_counts": family_counts,
            "salient_entities": grounded_tokens[:8],
            "task_relevant_action_shortlist": shortlist,
            "deprioritized_families": deprioritized_families,
            "candidate_tracking": self._get_candidate_tracking_snapshot(),
            "task_contract": task_contract,
            "remote_room_signal": remote_room_signal,
            "relation_frontier": relation_frontier,
            "artifact_creation": artifact_creation,
            "substance_search": substance_search,
            "measurement_tracking": measurement_tracking,
            "comparison_tracking": comparison_tracking,
            "conditional_branch_tracking": conditional_branch_tracking,
        }

    def _summarize_admissible_actions(
        self,
        actions: list[str],
        *,
        shortlist_limit: int = 12,
    ) -> dict:
        cache_key = (tuple(actions), shortlist_limit)
        cached = self._admissible_summary_cache.get(cache_key)
        if cached is not None:
            return cached
        summary = self._summarize_admissible_actions_uncached(
            actions, shortlist_limit=shortlist_limit
        )
        self._admissible_summary_cache[cache_key] = summary
        return summary

    @staticmethod
    def _prune_empty_runtime_fields(data: dict) -> dict:
        return {
            key: value for key, value in data.items() if value not in ({}, [], "", None)
        }

    def _limit_runtime_payload(
        self, value, *, list_limit: int = 4, filter_false: bool = False
    ):
        if isinstance(value, dict):
            limited = {
                key: self._limit_runtime_payload(
                    item, list_limit=list_limit, filter_false=filter_false
                )
                for key, item in value.items()
            }
            exclude = ({}, [], "", None, False) if filter_false else ({}, [], "", None)
            return {key: item for key, item in limited.items() if item not in exclude}
        if isinstance(value, list):
            return [
                self._limit_runtime_payload(
                    item, list_limit=list_limit, filter_false=filter_false
                )
                for item in value[:list_limit]
            ]
        return value

    def _get_compact_task_contract_snapshot(
        self, task_contract: dict | None, *, list_limit: int = 4
    ) -> dict:
        contract = task_contract if isinstance(task_contract, dict) else {}
        compact_contract = self._limit_runtime_payload(
            contract, list_limit=list_limit, filter_false=True
        )
        compact_contract.pop("measurement_branches", None)
        compact_contract.pop("conditional_branches", None)
        return compact_contract

    def _get_action_agent_task_contract_snapshot(self, summary: dict) -> dict:
        return self._get_compact_task_contract_snapshot(
            summary.get("task_contract", {})
        )

    def _get_agent_facing_action_summary_snapshot(self, summary: dict | None) -> dict:
        summary = summary if isinstance(summary, dict) else {}
        return self._prune_empty_runtime_fields(
            {
                "total_actions": summary.get("total_actions"),
                "current_phase": summary.get("current_phase"),
                "salient_entities": list(summary.get("salient_entities", [])[:6]),
                "deprioritized_families": list(
                    summary.get("deprioritized_families", [])[:4]
                ),
                "required_families": list(summary.get("required_families", [])[:4]),
            }
        )

    def _get_agent_facing_runtime_snapshot(self) -> dict:
        return self._prune_empty_runtime_fields(
            {
                key: self._limit_runtime_payload(
                    self.percept.get(key, {}), list_limit=limit
                )
                for key, limit in self._RUNTIME_PERCEPT_KEYS
            }
        )

    def _get_agent_facing_percept_snapshot(self) -> dict:
        percept_snapshot = self._snapshot_analyst_payload(self.percept)
        if not isinstance(percept_snapshot, dict):
            return {}
        compact_task_contract = self._get_compact_task_contract_snapshot(
            percept_snapshot.get("task_contract", {})
        )
        if compact_task_contract:
            percept_snapshot["task_contract"] = compact_task_contract
        else:
            percept_snapshot.pop("task_contract", None)
        compact_action_summary = self._get_agent_facing_action_summary_snapshot(
            percept_snapshot.get("admissible_action_summary", {})
        )
        if compact_action_summary:
            percept_snapshot["admissible_action_summary"] = compact_action_summary
        else:
            percept_snapshot.pop("admissible_action_summary", None)
        _runtime_key_set = {key for key, _ in self._RUNTIME_PERCEPT_KEYS}
        percept_snapshot = {
            k: v for k, v in percept_snapshot.items() if k not in _runtime_key_set
        }
        percept_snapshot.update(self._get_agent_facing_runtime_snapshot())
        if "task_relevant_action_shortlist" in percept_snapshot:
            percept_snapshot["task_relevant_action_shortlist"] = list(
                percept_snapshot.get("task_relevant_action_shortlist", [])[:8]
            )
        for key in ("newly_relevant_actions", "no_longer_relevant_actions"):
            if key in percept_snapshot:
                percept_snapshot[key] = list(percept_snapshot.get(key, [])[:4])
        return percept_snapshot

    def _get_action_agent_runtime_snapshots(self, summary: dict) -> dict:
        snapshots = {}
        ordered_progress = self._get_ordered_target_snapshot()
        if self._snapshot_has_signal(ordered_progress):
            snapshots["ordered_target_progress"] = ordered_progress
        candidate_tracking = summary.get("candidate_tracking", {})
        if self._snapshot_has_signal(candidate_tracking):
            snapshots["candidate_tracking"] = candidate_tracking
        relation_frontier = summary.get("relation_frontier", {})
        if self._snapshot_has_signal(relation_frontier):
            snapshots["relation_frontier"] = relation_frontier
        substance_search = summary.get("substance_search", {})
        if self._snapshot_has_signal(substance_search):
            snapshots["substance_search"] = substance_search
        artifact_creation = summary.get("artifact_creation", {})
        if self._snapshot_has_signal(artifact_creation):
            snapshots["artifact_creation"] = artifact_creation
        measurement_tracking = summary.get("measurement_tracking", {})
        if self._snapshot_has_signal(measurement_tracking):
            snapshots["measurement_tracking"] = measurement_tracking
        comparison_tracking = summary.get("comparison_tracking", {})
        if self._snapshot_has_signal(comparison_tracking):
            snapshots["comparison_tracking"] = comparison_tracking
        conditional_branch_tracking = summary.get("conditional_branch_tracking", {})
        if self._snapshot_has_signal(conditional_branch_tracking):
            snapshots["conditional_branch_tracking"] = conditional_branch_tracking
        remote_room_signal = summary.get("remote_room_signal", {})
        if self._snapshot_has_signal(remote_room_signal, key="room"):
            snapshots["remote_room_signal"] = remote_room_signal
        if self.percept.get("referent_resolution"):
            snapshots["referent_resolution"] = self.percept["referent_resolution"]
        return snapshots

    def _get_focus_agent_action_summary(self, summary: dict | None) -> dict:
        summary = summary if isinstance(summary, dict) else {}
        return self._prune_empty_runtime_fields(
            {
                "total_actions": summary.get("total_actions"),
                "current_phase": summary.get("current_phase"),
                "salient_entities": list(summary.get("salient_entities", [])[:6]),
                "task_relevant_action_shortlist": list(
                    summary.get("task_relevant_action_shortlist", [])[:8]
                ),
                "newly_relevant_actions": list(
                    summary.get("newly_relevant_actions", [])[:4]
                ),
                "no_longer_relevant_actions": list(
                    summary.get("no_longer_relevant_actions", [])[:4]
                ),
                "deprioritized_families": list(
                    summary.get("deprioritized_families", [])[:4]
                ),
                "task_contract": self._get_compact_task_contract_snapshot(
                    summary.get("task_contract", {})
                ),
                "runtime_snapshots": self._limit_runtime_payload(
                    self._get_action_agent_runtime_snapshots(summary),
                    list_limit=3,
                ),
            }
        )

    @staticmethod
    def _snapshot_analyst_payload(value):
        return json.loads(json.dumps(value, default=str))

    @staticmethod
    def _indent_analyst_text(text: str, *, prefix: str = "  ") -> str:
        if not text:
            return f"{prefix}[empty]"
        return "\n".join(
            f"{prefix}{line}" if line else prefix.rstrip()
            for line in str(text).splitlines()
        )

    def _render_analyst_section(self, title: str, body) -> str:
        lines = [title, "-" * len(title)]
        if isinstance(body, str):
            rendered = body
        elif isinstance(body, (dict, list)):
            rendered = json.dumps(body, indent=2, sort_keys=True)
        elif body is None:
            rendered = "[empty]"
        else:
            rendered = str(body)
        lines.append(self._indent_analyst_text(rendered))
        return "\n".join(lines)

    def _render_analyst_list_section(self, title: str, items: list[str]) -> str:
        lines = [title, "-" * len(title)]
        if not items:
            lines.append("  [none]")
            return "\n".join(lines)
        lines.extend(f"  {idx}. {item}" for idx, item in enumerate(items, start=1))
        return "\n".join(lines)

    def _get_recent_analyst_messages(self) -> list[dict]:
        group_chat = getattr(self, "group_chat", None)
        messages = getattr(group_chat, "messages", None) or []
        if self._analyst_trace_message_cursor >= len(messages):
            self._analyst_trace_message_cursor = len(messages)
            return []
        recent_messages = [
            self._snapshot_analyst_payload(message)
            for message in messages[self._analyst_trace_message_cursor :]
        ]
        self._analyst_trace_message_cursor = len(messages)
        return recent_messages

    def _render_analyst_messages_section(self, title: str, messages: list[dict]) -> str:
        lines = [title, "-" * len(title)]
        if not messages:
            lines.append("  [none]")
            return "\n".join(lines)

        for idx, message in enumerate(messages, start=1):
            name = str(message.get("name") or message.get("role") or "unknown")
            role = str(message.get("role") or "unknown")
            content = message.get("content")
            metadata = {
                key: value
                for key, value in message.items()
                if key not in {"name", "role", "content"}
            }
            if isinstance(content, str):
                rendered_content = content or "[empty]"
            elif content is None:
                rendered_content = "[empty]"
            else:
                rendered_content = json.dumps(
                    content, indent=2, sort_keys=True, default=str
                )

            lines.extend(
                [
                    f"  [{idx}] {name}",
                    f"    role: {role}",
                    "    content:",
                    *(
                        f"      {line}" if line else ""
                        for line in rendered_content.splitlines()
                    ),
                ]
            )
            if metadata:
                lines.extend(
                    [
                        "    metadata:",
                        *(
                            f"      {line}"
                            for line in json.dumps(
                                metadata, indent=2, sort_keys=True, default=str
                            ).splitlines()
                        ),
                    ]
                )
        return "\n".join(lines)

    def _get_analyst_runtime_snapshots(self, summary: dict) -> dict:
        snapshots = {}

        ordered_progress = self._get_ordered_target_snapshot()
        if self._snapshot_has_signal(ordered_progress):
            snapshots["ordered_progress"] = {
                "focused": ordered_progress.get("focused_stage_labels", [])[:3],
                "pending": ordered_progress.get("pending_stage_candidates", [])[:3],
            }

        candidate_tracking = summary.get("candidate_tracking", {})
        if self._snapshot_has_signal(candidate_tracking):
            snapshots["candidate"] = {
                "active": candidate_tracking.get("active_candidate"),
                "last_seen_room": candidate_tracking.get("last_seen_room"),
                "rejected": candidate_tracking.get("rejected_candidates", [])[:2],
            }

        measurement_tracking = summary.get("measurement_tracking", {})
        if self._snapshot_has_signal(measurement_tracking):
            snapshots["measurement"] = {
                "target": measurement_tracking.get("measurement_target"),
                "property": measurement_tracking.get("measurement_property"),
                "resolved": measurement_tracking.get("property_resolved"),
                "branch_ready": measurement_tracking.get("branch_ready"),
            }

        comparison_tracking = summary.get("comparison_tracking", {})
        if self._snapshot_has_signal(comparison_tracking):
            snapshots["comparison"] = {
                "targets": comparison_tracking.get("comparison_targets", [])[:2],
                "resolved_target": comparison_tracking.get("selected_target"),
            }

        conditional_branch_tracking = summary.get("conditional_branch_tracking", {})
        if self._snapshot_has_signal(conditional_branch_tracking):
            snapshots["conditional_branch"] = {
                "evidence_target": conditional_branch_tracking.get("evidence_target"),
                "resolved_target": conditional_branch_tracking.get("selected_branch"),
            }

        relation_frontier = summary.get("relation_frontier", {})
        if self._snapshot_has_signal(relation_frontier):
            snapshots["relation_frontier"] = {
                "referents": relation_frontier.get("frontier_referents", [])[:4],
                "control_candidates": relation_frontier.get("control_candidates", [])[
                    :2
                ],
            }

        remote_room_signal = summary.get("remote_room_signal", {})
        if self._snapshot_has_signal(remote_room_signal, key="room"):
            snapshots["remote_room"] = {
                "room": remote_room_signal.get("room"),
                "reason": remote_room_signal.get("reason"),
            }

        substance_search = summary.get("substance_search", {})
        if self._snapshot_has_signal(substance_search):
            snapshots["substance_search"] = {
                "phase": substance_search.get("phase"),
                "grounded_substances": substance_search.get("grounded_substances", [])[
                    :3
                ],
                "source_candidates": substance_search.get("source_candidates", [])[:3],
            }

        artifact_creation = summary.get("artifact_creation", {})
        if self._snapshot_has_signal(artifact_creation):
            snapshots["artifact_creation"] = {
                "artifact_type": artifact_creation.get("artifact_type"),
                "grounded_artifacts": artifact_creation.get("grounded_artifacts", [])[
                    :3
                ],
            }

        return self._limit_runtime_payload(snapshots, list_limit=3)

    @staticmethod
    def _render_analyst_value(value) -> str:
        if isinstance(value, str):
            return value or "[empty]"
        if value in (None, [], {}):
            return "[empty]"
        return json.dumps(value, indent=2, sort_keys=True, default=str)

    def _build_analyst_architecture_overview(self) -> Panel:
        table = Table(expand=True, box=None, show_header=True, header_style="bold cyan")
        table.add_column("Component", style="bold yellow", ratio=1)
        table.add_column("Purpose In The Cognitive Loop", style="white", ratio=3)
        for component, description in self._ANALYST_AGENT_GUIDE.items():
            table.add_row(component, description)
        return Panel(
            table,
            title="[bold bright_blue]How To Read The Cognitive Architecture[/bold bright_blue]",
            border_style="bright_blue",
        )

    def _collect_analyst_glossary_terms(self) -> list[str]:
        return sorted(self._ANALYST_TERM_GLOSSARY)

    def _build_analyst_glossary(self) -> Panel | None:
        terms = self._collect_analyst_glossary_terms()
        if not terms:
            return None
        table = Table(
            expand=True,
            show_lines=True,
            header_style="bold magenta",
        )
        table.add_column("Term", style="bold yellow", width=28)
        table.add_column("Meaning", style="white", ratio=2)
        table.add_column("Related Terms", style="green", ratio=2)
        table.add_column("How It Fits The Architecture", style="cyan", ratio=3)
        for term in terms:
            meaning, related_terms, role = self._ANALYST_TERM_GLOSSARY[term]
            table.add_row(term, meaning, related_terms, role)
        return Panel(
            table,
            title="[bold magenta]Runtime Glossary[/bold magenta]",
            border_style="magenta",
        )

    def _build_analyst_key_value_panel(
        self,
        *,
        title: str,
        rows: list[tuple[str, str]],
        border_style: str = "cyan",
    ) -> Panel:
        table = Table(expand=True, box=None, show_header=False)
        table.add_column("key", style="bold yellow", width=24)
        table.add_column("value", style="white")
        for key, value in rows:
            table.add_row(Text(str(key)), Text(str(value)))
        return Panel(table, title=title, border_style=border_style)

    def _build_analyst_message_panel(
        self,
        *,
        title: str,
        messages: list[dict],
        border_style: str,
    ) -> Panel:
        if not messages:
            return Panel("[dim][none][/dim]", title=title, border_style=border_style)
        table = Table(expand=True, show_lines=True)
        table.add_column("Agent", style="bold yellow", width=24)
        table.add_column("Role", style="cyan", width=14)
        table.add_column("Output", style="white", ratio=4)
        for message in messages:
            name = str(message.get("name") or message.get("role") or "unknown")
            role = str(message.get("role") or "unknown")
            content = self._render_analyst_value(message.get("content"))
            metadata = {
                key: value
                for key, value in message.items()
                if key not in {"name", "role", "content"}
            }
            if metadata:
                content = f"{content}\n\nMetadata:\n" + json.dumps(
                    metadata, indent=2, sort_keys=True, default=str
                )
            table.add_row(Text(name), Text(role), Text(content))
        return Panel(table, title=title, border_style=border_style)

    def _build_analyst_list_panel(
        self,
        *,
        title: str,
        items: list[str],
        border_style: str,
    ) -> Panel:
        table = Table(expand=True, box=None, show_header=False)
        table.add_column("#", style="bold cyan", width=4)
        table.add_column("Item", style="white")
        if not items:
            table.add_row(Text("-"), Text("[none]"))
        else:
            for idx, item in enumerate(items, start=1):
                table.add_row(Text(str(idx)), Text(item))
        return Panel(table, title=title, border_style=border_style)

    def _build_analyst_trace_entry_renderable(self, entry: dict) -> Panel:
        belief_messages = [
            message
            for message in entry["agent_messages"]
            if str(message.get("name") or "") == "Belief_State_Agent"
        ]
        other_agent_messages = [
            message
            for message in entry["agent_messages"]
            if str(message.get("name") or "") != "Belief_State_Agent"
        ]
        status_style = {
            "COMPLETED": "green",
            "FAILED": "red",
            "INCOMPLETE": "bright_blue",
        }.get(entry["task_status"], "white")

        renderables = [
            self._build_analyst_key_value_panel(
                title="[bold]Action + Observation[/bold]",
                border_style=status_style,
                rows=[
                    ("timestep", str(entry["timestep"])),
                    ("phase", entry["phase"]),
                    ("task_status", entry["task_status"]),
                    ("attempted_action", entry["attempted_action"]),
                    ("observation", entry["observation"]),
                ],
            )
        ]

        if entry.get("task_contract"):
            renderables.append(
                Panel(
                    Text(self._render_analyst_value(entry["task_contract"])),
                    title="[bold]Task Contract[/bold]",
                    border_style="yellow",
                )
            )

        renderables.extend(
            [
                Panel(
                    Text(self._render_analyst_value(entry["agent_facing_percept"])),
                    title="[bold]Agent-Facing Percept Snapshot[/bold]",
                    border_style="cyan",
                ),
                Panel(
                    Text(self._render_analyst_value(entry["percept"])),
                    title="[bold]Execute Action Percept (Full)[/bold]",
                    border_style="bright_blue",
                ),
                self._build_analyst_message_panel(
                    title="[bold]Belief State[/bold]",
                    messages=belief_messages,
                    border_style="green",
                ),
                self._build_analyst_message_panel(
                    title="[bold]Other Agent Outputs[/bold]",
                    messages=other_agent_messages,
                    border_style="cyan",
                ),
                self._build_analyst_list_panel(
                    title="[bold]Salient Entities[/bold]",
                    items=entry["salient_entities"],
                    border_style="magenta",
                ),
                self._build_analyst_list_panel(
                    title="[bold]Task-Relevant Action Shortlist[/bold]",
                    items=entry["shortlist"],
                    border_style="yellow",
                ),
                Panel(
                    Text(self._render_analyst_value(entry["runtime"])),
                    title="[bold]Runtime Snapshots[/bold]",
                    border_style="magenta",
                ),
            ]
        )

        if entry["newly_relevant_actions"]:
            renderables.append(
                self._build_analyst_list_panel(
                    title="[bold]Newly Relevant Actions[/bold]",
                    items=entry["newly_relevant_actions"],
                    border_style="green",
                )
            )
        if entry["no_longer_relevant_actions"]:
            renderables.append(
                self._build_analyst_list_panel(
                    title="[bold]No Longer Relevant Actions[/bold]",
                    items=entry["no_longer_relevant_actions"],
                    border_style="red",
                )
            )

        return Panel(
            Group(*renderables),
            title=(
                f"[bold {status_style}]T{entry['timestep']} | "
                f"{entry['phase']} | {entry['task_status']}[/bold {status_style}]"
            ),
            border_style=status_style,
        )

    def _render_analyst_trace(self, *, styles: bool = False) -> str:
        if not self._analyst_trace_entries:
            return ""

        console = Console(
            record=True,
            width=140,
            soft_wrap=True,
            force_terminal=styles,
            color_system="truecolor" if styles else None,
            file=io.StringIO(),
        )
        console.print(
            Panel(
                Text(
                    "This trace is written for a human analyst. Read top-to-bottom: "
                    "task interpretation, belief updates, supporting agent reasoning, "
                    "the compact agent-facing percept, the full execute_action "
                    "observation/percept, and the shortlist/runtime state that shaped "
                    "the next move."
                ),
                title="[bold bright_white]Analyst Trace[/bold bright_white]",
                border_style="bright_white",
            )
        )
        console.print(self._build_analyst_architecture_overview())
        glossary_panel = self._build_analyst_glossary()
        if glossary_panel is not None:
            console.print(glossary_panel)
        for entry in self._analyst_trace_entries:
            console.print(self._build_analyst_trace_entry_renderable(entry))
        return console.export_text(styles=styles)

    def _persist_analyst_trace(self, *, summary: dict) -> None:
        recent_messages = self._get_recent_analyst_messages()
        entry = {
            "timestep": self.percept.get("timestep", self.num_actions_taken),
            "phase": summary.get("current_phase", "act"),
            "task_status": self.percept.get("task_status", self.task_status),
            "attempted_action": self.percept.get("attempted_action", "None"),
            "observation": self.percept.get("resulting_observation", ""),
            "agent_facing_percept": self._snapshot_analyst_payload(
                self._get_agent_facing_percept_snapshot()
            ),
            "percept": self._snapshot_analyst_payload(self.percept),
            "task_contract": self._snapshot_analyst_payload(
                self._get_compact_task_contract_snapshot(
                    self.percept.get("task_contract", {})
                )
            ),
            "agent_messages": recent_messages,
            "salient_entities": list(summary.get("salient_entities", [])[:6]),
            "shortlist": list(summary.get("task_relevant_action_shortlist", [])[:8]),
            "runtime": self._snapshot_analyst_payload(
                self._get_analyst_runtime_snapshots(summary)
            ),
            "newly_relevant_actions": list(
                summary.get("newly_relevant_actions", [])[:6]
            ),
            "no_longer_relevant_actions": list(
                summary.get("no_longer_relevant_actions", [])[:6]
            ),
        }

        if (
            self._analyst_trace_entries
            and self._analyst_trace_entries[-1]["timestep"] == entry["timestep"]
        ):
            self._analyst_trace_entries[-1] = entry
        else:
            self._analyst_trace_entries.append(entry)

        text = self._render_analyst_trace(styles=False)
        ansi_text = self._render_analyst_trace(styles=True)
        self._last_analyst_trace_text = text
        self._last_analyst_trace_ansi_text = ansi_text

        analyst_trace_path = self.log_paths.get("analyst_trace_path")
        if analyst_trace_path:
            with open(analyst_trace_path, "w") as f:
                f.write(text)
        analyst_trace_ansi_path = self.log_paths.get("analyst_trace_ansi_path")
        if analyst_trace_ansi_path:
            with open(analyst_trace_ansi_path, "w") as f:
                f.write(ansi_text)

        callback = getattr(self, "analyst_trace_callback", None)
        if callable(callback):
            callback(text)

    def _build_shared_action_context(self, *, summary: dict | None = None) -> dict:
        summary = dict(
            summary
            if summary is not None
            else self._summarize_admissible_actions(
                self.admissible_actions, shortlist_limit=20
            )
        )
        summary["task_relevant_action_shortlist"] = summary[
            "task_relevant_action_shortlist"
        ][:12]
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

    def get_analyst_trace_text(self) -> str:
        if self._last_analyst_trace_text:
            return self._last_analyst_trace_text
        analyst_trace_path = self.log_paths.get("analyst_trace_path")
        if analyst_trace_path and os.path.exists(analyst_trace_path):
            with open(analyst_trace_path) as f:
                return f.read()
        return ""

    def get_analyst_trace_ansi_text(self) -> str:
        if self._last_analyst_trace_ansi_text:
            return self._last_analyst_trace_ansi_text
        analyst_trace_ansi_path = self.log_paths.get("analyst_trace_ansi_path")
        if analyst_trace_ansi_path and os.path.exists(analyst_trace_ansi_path):
            with open(analyst_trace_ansi_path) as f:
                return f.read()
        return ""

    def _refresh_action_agent_runtime_context(
        self, *, summary: dict | None = None
    ) -> None:
        if self.action_agent is None:
            return

        summary = (
            summary
            if summary is not None
            else self._summarize_admissible_actions(
                self.admissible_actions, shortlist_limit=20
            )
        )
        recent_invalid_actions = self._get_recent_invalid_actions()
        compact_task_contract = self._get_action_agent_task_contract_snapshot(summary)
        runtime_snapshots = self._limit_runtime_payload(
            self._get_action_agent_runtime_snapshots(summary)
        )
        self._set_agent_system_message(
            self.action_agent,
            self._action_agent_base_prompt
            + "\n\n--- PRIVATE RUNTIME CONTEXT ---\n"
            + f"Current phase: {summary['current_phase']}\n"
            + f"Current exact task-relevant admissible shortlist: {json.dumps(summary['task_relevant_action_shortlist'])}\n"
            + f"Salient grounded entities from the latest percept: {json.dumps(summary['salient_entities'])}\n"
            + f"Task contract: {json.dumps(compact_task_contract)}\n"
            + f"Episode runtime snapshots: {json.dumps(runtime_snapshots)}\n"
            + f"Deprioritized mechanism families this episode: {json.dumps(summary['deprioritized_families'])}\n"
            + f"Recent invalid exact commands to avoid repeating: {json.dumps(recent_invalid_actions)}\n"
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
        ledger = self._get_episode_hypothesis_snapshot(
            max_families=3, max_recent_tests=2
        )

        parts = [
            f"Timestep {timestep}: My previous belief-state output drifted out of format, so I discard any embedded action suggestion and re-anchor on the latest confirmed percept.",
            f"The last attempted action was {attempted_action!r}.",
            f"The latest confirmed observation is: {observation}",
            f"The task is still {percept.get('task_status', self.task_status)}.",
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

    def update_percept(self, action, executed: bool = True):
        previous_observation = (self.percept or {}).get("resulting_observation", "")
        curr_admissible = self.adapter.admissible_actions
        no_longer = sorted(set(self.admissible_actions) - set(curr_admissible))
        newly_added = sorted(set(curr_admissible) - set(self.admissible_actions))
        self.admissible_actions = curr_admissible

        self.percept = {
            "timestep": self.num_actions_taken,
            "attempted_action": action,
            "action_executed": executed,
            "resulting_observation": self.adapter.observation,
            "task_status": self.task_status,
        }
        self._update_state_change_search_tracking(
            action=action,
            observation=self.adapter.observation,
        )
        self._update_artifact_creation_tracking(
            self.adapter.observation,
            action=action,
        )
        self._update_artifact_creation_search_tracking(
            action=action,
            observation=self.adapter.observation,
        )
        self._update_growth_search_tracking(
            action=action,
            observation=self.adapter.observation,
        )
        self._update_measurement_search_tracking(
            action=action,
            observation=self.adapter.observation,
        )
        self._update_measurement_tracking(
            action=action,
            observation=self.adapter.observation,
        )
        self._update_comparison_search_tracking(
            action=action,
            observation=self.adapter.observation,
        )
        self._update_comparison_tracking(
            action=action,
            observation=self.adapter.observation,
        )
        self._update_conditional_branch_tracking(
            action=action,
            observation=self.adapter.observation,
        )
        self._update_relation_task_tracking(
            action=action,
            observation=self.adapter.observation,
        )
        self._update_lifecycle_search_tracking(
            action=action,
            observation=self.adapter.observation,
        )
        self._update_remote_room_signal(
            action=action,
            observation=self.adapter.observation,
            previous_observation=previous_observation,
        )
        self._invalidate_action_summary_cache()
        self._update_task_contract_from_recipe_observation(
            action=action, observation=self.adapter.observation
        )
        task_contract = self._get_task_contract()
        if any(task_contract.values()):
            self.percept["task_contract"] = task_contract
        action_runtime_summary = self._summarize_admissible_actions(
            self.admissible_actions, shortlist_limit=20
        )
        shared_action_context = self._build_shared_action_context(
            summary=action_runtime_summary
        )
        self.percept["admissible_action_summary"] = {
            "total_actions": shared_action_context["total_actions"],
            "current_phase": shared_action_context["current_phase"],
            "salient_entities": shared_action_context["salient_entities"],
            "deprioritized_families": shared_action_context["deprioritized_families"],
            "required_families": shared_action_context["task_contract"][
                "required_families"
            ],
        }
        substance_search = shared_action_context.get("substance_search", {})
        if self._snapshot_has_signal(substance_search):
            self.percept["substance_search"] = substance_search
        artifact_creation = shared_action_context.get("artifact_creation", {})
        if self._snapshot_has_signal(artifact_creation):
            self.percept["artifact_creation"] = artifact_creation
        measurement_tracking = shared_action_context.get("measurement_tracking", {})
        if self._snapshot_has_signal(measurement_tracking):
            self.percept["measurement_tracking"] = measurement_tracking
        comparison_tracking = shared_action_context.get("comparison_tracking", {})
        if self._snapshot_has_signal(comparison_tracking):
            self.percept["comparison_tracking"] = comparison_tracking
        conditional_branch_tracking = shared_action_context.get(
            "conditional_branch_tracking", {}
        )
        if self._snapshot_has_signal(conditional_branch_tracking):
            self.percept["conditional_branch_tracking"] = conditional_branch_tracking
        relation_frontier = shared_action_context.get("relation_frontier", {})
        if self._snapshot_has_signal(relation_frontier):
            self.percept["relation_frontier"] = relation_frontier
        remote_room_signal = shared_action_context.get("remote_room_signal", {})
        if self._snapshot_has_signal(remote_room_signal, key="room"):
            self.percept["remote_room_signal"] = remote_room_signal
        self.percept["task_relevant_action_shortlist"] = shared_action_context[
            "task_relevant_action_shortlist"
        ]
        ordered_progress = self._get_ordered_target_snapshot()
        if self._snapshot_has_signal(ordered_progress):
            self.percept["ordered_target_progress"] = ordered_progress
        candidate_tracking = self._get_candidate_tracking_snapshot()
        if self._snapshot_has_signal(candidate_tracking):
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
        self._persist_analyst_trace(summary=shared_action_context)
        self._refresh_action_agent_runtime_context(summary=action_runtime_summary)

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
        analyst_trace_path = os.path.join(game_path, "analyst_trace.txt")
        analyst_trace_ansi_path = os.path.join(game_path, "analyst_trace.ansi")
        result_path = os.path.join(game_path, "result.txt")
        error_message_path = os.path.join(game_path, "error_message.txt")

        self.log_paths.update(
            {
                "task_path": task_path,
                "history_path": history_path,
                "concept_path": concept_path,
                "admissible_commands_path": admissible_commands_path,
                "chat_history_path": chat_history_path,
                "analyst_trace_path": analyst_trace_path,
                "analyst_trace_ansi_path": analyst_trace_ansi_path,
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
            self.update_percept(attempted_action, executed=executed_action is not None)
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
            if self._snapshot_has_signal(ordered_progress):
                self.percept["ordered_target_progress"] = ordered_progress
            self._refresh_action_agent_runtime_context()

            with open(self.log_paths["admissible_commands_path"], "a+") as f:
                f.write(f"{self.admissible_actions}\n")
            with open(self.log_paths["history_path"], "a+") as f:
                f.write(
                    f"action: '{suggested_action}'. observation: '{self.adapter.observation}'\n"
                )

            return json.dumps(self._get_agent_facing_percept_snapshot()) + reflection

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
            focus_summary = self._get_focus_agent_action_summary(shared_action_context)
            return (
                f"TASK: {self.task}\n"
                "REPEATING LAST PERCEPT TO HELP CONSTRUCT BELIEF STATE:\n"
                f"{json.dumps(self._get_agent_facing_percept_snapshot())}\n"
                f"EPISODE HYPOTHESIS LEDGER: {json.dumps(self._get_episode_hypothesis_snapshot())}\n"
                f"ACTION SPACE SUMMARY: {json.dumps(focus_summary)}"
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
            "--- ENVIRONMENTAL CONSTRAINTS ---\n"
            "- Internal deliberation and physical actions are scarce resources. Stay efficient without letting budget pressure override grounded reasoning.\n\n"
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

        state_section = (
            "--- CURRENT STATE ---\n"
            + json.dumps(self._get_agent_facing_percept_snapshot())
            + "\n"
        )

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
