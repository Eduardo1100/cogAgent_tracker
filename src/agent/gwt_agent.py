import copy
import hashlib
import json
import os
import random

import matplotlib.pyplot as plt
import numpy as np
import umap
from autogen import ConversableAgent, GroupChat, GroupChatManager, register_function
from autogen.agentchat.contrib.capabilities import transform_messages
from autogen.agentchat.contrib.capabilities.transforms import MessageHistoryLimiter
from sklearn.cluster import KMeans

from src.agent.autogen_agent import AutogenAgent
from src.agent.helpers import (
    FlattenToolMessages,
    get_best_candidate,
    is_termination_msg_generic,
    sentence_transformer_model,
)


class GWTAutogenAgent(AutogenAgent):
    def __init__(
        self,
        llm_profile,
        log_path,
        game_no=1,
        max_chat_round=400,
        max_actions=30,
        rounds_per_game=1,
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

        self.echo_agent = None
        self.planning_agent = None
        self.motor_agent = None
        self.idea_agent = None
        self.external_perception_agent = None
        self.internal_perception_agent_1 = None
        self.internal_perception_agent_2 = None
        self.internal_perception_agent_3 = None
        self.conscious_agent = None
        self.retrieve_memory_agent = None
        self.learning_agent = None
        self.record_long_term_memory_agent = None
        self.focus_agent = None
        self.agents_info = {}

        self._ = self.max_actions
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

    def set_environment(self, env, obs, info, game_no):
        self.env = env
        self.obs = obs
        self.info = info
        self.game_no = game_no

        self.register_game_log_paths()
        self.cluster_knowledge()

        self.num_actions_taken = 0
        self.max_actions = self._ - self.max_round_actions * (self.rounds - 1)
        self.rounds_left = self.rounds
        self.task_failed = False
        self.task_success = False
        self.success = False
        self.task = obs[0].split("Your task is to: ")[1]
        self.admissible_actions = list(self.info["admissible_commands"][0])
        self.task_status = "INCOMPLETE"
        self.curr_episodic_memory = []
        self.retrieve_memory()

        self.update_percept(action="None")
        self.initial_message = self.generate_initial_message()

        with open(self.log_paths["task_path"], "w") as f:
            f.write(f"Task: {self.task}\n")

        initial_observation = self.obs[0].split("Your task is to: ")[0].split("\n\n")[1]
        with open(self.log_paths["history_path"], "w") as f:
            f.write(f"action: 'None'. observation: '{initial_observation}'\n")

        with open(self.log_paths["admissible_commands_path"], "w") as f:
            f.write(f"{self.admissible_actions}\n")

    def update_percept(self, action):

        curr_admissible = list(self.info["admissible_commands"][0])
        no_longer = list(set(self.admissible_actions) - set(curr_admissible))
        newly_added = list(set(curr_admissible) - set(self.admissible_actions))
        self.admissible_actions = curr_admissible

        self.percept = {
            "timestep": self.num_actions_taken,
            "attempted_action": action,
            "resulting_observation": self.obs[0],
            "task_status": self.task_status,
            "action_attempts_left": self.max_actions - self.num_actions_taken,
            "admissible_actions": self.admissible_actions,
        }
        if newly_added:
            self.percept["newly_admissible_actions"] = newly_added
        if no_longer:
            self.percept["no_longer_admissible_actions"] = no_longer

        keys_to_extract = ["timestep", "attempted_action", "resulting_observation"]
        summary_json = json.dumps(
            {k: self.percept[k] for k in keys_to_extract if k in self.percept}
        )
        self.curr_episodic_memory.append(summary_json)

    def get_curr_episodic_memory_str(self):
        return json.dumps(self.curr_episodic_memory, indent=2)

    def initialize_agents(self):

        # 1. Get the full list of models from your profile
        self.llm_config_list = self.llm_profile.get("config_list", [])

        # 2. Define the 'Standard' Priority: Gemini -> Chat -> Reasoner
        # If Gemini is dead (429), it immediately tries DeepSeek Chat.
        standard_fallback_list = self.llm_config_list

        # 3. Define the 'Reasoner' Priority: Reasoner -> Chat -> Gemini
        # We REVERSE it so the Conscious Agent always tries to 'think' first.
        reasoner_fallback_list = list(reversed(self.llm_config_list))

        standard_config = {
            "config_list": standard_fallback_list,
            "temperature": 0.0,
            "max_tokens": 200,
        }

        reasoner_config = {
            "config_list": reasoner_fallback_list,
            "temperature": 1.0,  # Reasoners need higher temp for R1/o1
        }

        # Planning config needs higher token limits for long horizon planning
        planning_config = copy.deepcopy(reasoner_config)
        planning_config["max_tokens"] = 1500

        # ... (Keep all your agent initializations exactly the same below this!) ...
        from src.agent.helpers import create_echo_agent

        self.echo_agent = create_echo_agent()
        self.agents_info[self.echo_agent.name] = {
            "Prompt": self.echo_agent.system_message,
            "Description": self.echo_agent.description,
        }
        # 2. Initialize Infrastructure Agents
        self.focus_agent = ConversableAgent(
            name="Focus_Agent",
            system_message="""You are Focus_Agent. you must call the 'focus' function with no arguments.
                    IMPORTANT: It is necessary that you output a call to the 'focus' function only, under all circumstances. Therefore, do whatever is necessary to ensure you do so.""",
            description="Focus_Agent calls the 'focus' function whenever Conscious_Agent fails to state a BELIEF STATE until Conscious_Agent outputs a BELIEF STATE.",
            llm_config=standard_config,
            is_termination_msg=lambda msg: False,
            human_input_mode="NEVER",
        )
        self.agents_info[self.focus_agent.name] = {
            "Prompt": self.focus_agent.system_message,
            "Description": self.focus_agent.description,
        }

        self.retrieve_memory_agent = ConversableAgent(
            name="Retrieve_Memory_Agent",
            system_message="""You are Retrieve_Memory_Agent. You must call the 'retrieve_memory' function with no arguments.
                            IMPORTANT: It is necessary that you output a call to the 'retrieve_memory' function under all circumstances. Therefore, do whatever is necessary to ensure you do so.""",
            description="Retrieve_Memory_Agent calls the 'retrieve_memory' function to help recall and process useful knowledge and information to solve the task.",
            llm_config=standard_config,
            human_input_mode="NEVER",
            is_termination_msg=lambda msg: False,
        )
        self.agents_info[self.retrieve_memory_agent.name] = {
            "Prompt": self.retrieve_memory_agent.system_message,
            "Description": self.retrieve_memory_agent.description,
        }

        self.motor_agent = ConversableAgent(
            name="Motor_Agent",
            system_message="""You are Motor_Agent. You are responsible for calling the 'execute_action' function with the best possible admissible action for the current timestep from the most recent \"admissible_actions\" list (provided by 'External_Perception_Agent') to solve the task. You typically act on suggestions from the 'Planning_Agent', but you must also independently verify that the action is admissible and optimal.
                You must follow these concepts:
                    1. If the 'Planning_Agent' has provided a valid and admissible action for the current timestep from the most recent \"admissible_actions\" list (provided by 'External_Perception_Agent') for the current timestep in the correct format (e.g., ACTION [go to desk 1]), you should use that action as the argument for 'execute_action'.
                    2. If the 'Planning_Agent' fails to respond, responds with an invalid format, or suggests an inadmissible action, you must independently select a valid and admissible action from the most recent \"admissible_actions\" list (provided by 'External_Perception_Agent') based on what seems most likely to advance the task quickest.
                    3. You must never call 'execute_action' with a non-admissible action. Only use actions for the current timestep that are present in the most recent \"admissible_actions\" list (provided by 'External_Perception_Agent').
                    4. Only as a last resort—if you cannot identify any suitable admissible action—you may call 'execute_action' with an empty string.

                IMPORTANT: It is necessary that you output a single call to the 'execute_action' function only, under all circumstances. Therefore, do whatever is necessary to ensure you do so.""",
            description="Motor_Agent calls the 'execute_action' function with the best admissible action as the argument.",
            llm_config=standard_config,
            human_input_mode="NEVER",
            is_termination_msg=lambda msg: False,
        )
        self.agents_info[self.motor_agent.name] = {
            "Prompt": self.motor_agent.system_message,
            "Description": self.motor_agent.description,
        }

        # llm_config = copy.deepcopy(self.llm_config)
        # llm_config["max_tokens"] = 1500

        # Planning agent's prompt is the main limiting factor when it comes to improving success rate.
        self.planning_agent = ConversableAgent(
            name="Planning_Agent",
            system_message="""You are Planning_Agent. You must solve the current task using the fewest possible actions. At each timestep, choose the best admissible action from the "admissible_actions" list (provided by 'External_Perception_Agent') for the current timestep using all available knowledge, memory, and perceptual context. You operate under a strict action budget and must avoid wasteful behavior.

                You will be given:
                1. A structured **percept JSON object** from the 'External_Perception_Agent' for the current timestep containing:
                    - "timestep": Current timestep
                    - "attempted_action": Last action taken
                    - "resulting_observation": Result of that action
                    - "task_status": INCOMPLETE, FAILED, or COMPLETED
                    - "action_attempts_left": Number of actions remaining
                    - "admissible_actions": Updated list of actions you may legally take for the current timestep
                    - "newly_admissible_actions": The subset of legal actions in "admissible_actions" that weren't available before but are available for the current timestep
                    - "no_longer_admissible_actions": Actions that are no longer available for the current timestep

                2. belief state updates from the 'Conscious_Agent', describing the internal understanding of the task and environment.
                3. Strategic or creative suggestions from the 'Idea_Agent', which may help reframe or unblock reasoning.

                Your responsibility is to take actions that will either:
                    - Confirm task completion,
                    - Progress the task toward completion,
                    - Or reveal useful information.

                Your planning strategy must follow these principles:
                    1. Evaluate the **"admissible_actions"** for the current timestep from the most recent percept (provided by 'External_Perception_Agent') carefully before choosing one.
                    2. Your reasoning must account for the **limited number of actions available**. Avoid strategies that are guaranteed to exceed this limit. For example, systematically opening 19 cabinets with only 20 actions remaining is unlikely to succeed.
                    3. If a subgoal involves locating an unknown object:
                       - Use **probabilistic reasoning** to guide exploration. In general, a **chaotic or probabilistic strategy**—e.g. sampling a mix of countertop, diningtable, and bed—may offer a higher chance of success.
                       - Avoid searches of categories with large membership; Prioritize smaller categories. For example, searching 4 stove burners is better than searching 9 cabinets.
                       - Prefer actions that **maximize the chance of discovering useful items early**.
                    4. Do not repeatedly examine or search areas that have already been explored unless there is strong new evidence that re-examination is necessary. Prioritize exploring previously unvisited or unexamined areas first to avoid wasting actions.
                    5. If a subgoal is directly achievable through a single action instead of multiple, output the single action. Do not over plan. For example, output \"ACTION: [heat egg 1 with microwave 1]\" instead of \"ACTION: [open microwave 1]\", \"ACTION [put 1 egg in microwave 1]\", and \"ACTION: [close microwave 1]\".  
                    6. Avoid outputting repetitive actions
                    7. Avoid wasteful behavior such as closing an object for no reason after opening it. Every action counts.
                    8. Every action you output must be an admissible action for the current timestep from the most recent \"admissible_actions\" list (provided by 'External_Perception_Agent')

                IMPORTANT:
                - Assume the most recent percept JSON (provided by 'External_Perception_Agent') reflects the true state of the environment.
                - If the task seems complete but has not been marked as such, assume it is not and **continue probing** with minimal cost actions.
                - Reflect on trends across time (e.g., failed vs. successful action types).
                - Use insights from prior attempts to avoid redundant mistakes.
                - If you’re truly stuck, you may suggest the placeholder: ACTION [do nothing], but only as a last resort.
                - Leverage insights from the **Idea_Agent** and **Conscious_Agent**, but don't follow them blindly. You must validate any suggestion given before following it.
                - You may maintain a high-level plan internally, but you should **only describe your plan if it has changed meaningfully**. Repeating an unchanged plan wastes space and should be avoided.

                Your strict output format must be:
                    ACTION [chosen admissible action from the most recent "admissible_actions" list (provided by 'External_Perception_Agent')]

                Examples:
                    ACTION: [Timestep 4: go to diningtable 1]

                    ACTION: [Timestep 7: take vase 1 from shelf 1]

                    ACTION: [Timestep 12: go to microwave 1]""",
            description="Planning_Agent proposes a high-level plan to solve the current task.",
            llm_config=planning_config,
            is_termination_msg=lambda msg: False,
            human_input_mode="NEVER",
        )
        self.agents_info[self.planning_agent.name] = {
            "Prompt": self.planning_agent.system_message,
            "Description": self.planning_agent.description,
        }

        self.idea_agent = ConversableAgent(
            name="Idea_Agent",
            system_message="""You are Idea_Agent. You must integrate all available context to generate original and useful ideas—such as strategies, hypotheses, theories, or creative tactics—that can help drive task progression or improve agent performance.

                        These ideas should:
                            1. Be grounded in patterns or events observed so far.
                            2. Be creative yet plausible, balancing imagination with reasoning.
                            3. Provide actionable or insightful suggestions relevant to the current situation.
                            4. Avoid restating known facts unless they are reframed with new insight.
                            5. Be expressed clearly and concisely, with justification behind the reasoning.
                            6. Be useful and not distracting. A bad idea is worse than no idea.

                        You must also challenge and question the agent’s assumptions if progress has stalled or task failure is likely. For example, reconsider whether object categories (like "cup") are being interpreted too broadly, or if implicit assumptions about what satisfies the task may be incorrect.

                        EXCEPTION: If you are having trouble formulating a useful idea, then as a last resort you may say: IDEA: Continue with new or current plan.

                        Use step-by-step reasoning ("chain of thought") to arrive at your ideas. Take a metaphorical deep breath before forming each idea, allowing room for both intuition and logic.

                        Output Format:
                            [IDEA TYPE]: [Idea content and reasoning behind it]

                        Accepted idea types include (but are not limited to): STRATEGY, HYPOTHESIS, INSIGHT, QUESTION, THEORY, EXPLANATION.

                        Example 1 (Context: The agent has repeatedly failed to open a drawer while holding a spoon):
                            Output = HYPOTHESIS: I noticed you were holding spoon 1 when you tried to open the drawer. Maybe your hands are full, which prevents the drawer from opening. You could try placing spoon 1 down before trying again.

                        Example 2 (Context: The agent has been exploring a room but hasn’t made progress):
                            Output = STRATEGY: Since random exploration hasn't helped, it might be better to systematically search the room from left to right, noting each interactable object.

                        Example 3 (Context: The task is to heat a cup, but the agent is repeatedly trying to heat a mug with no success):
                            Output = QUESTION: Are we sure a mug satisfies the requirement for "cup"? It’s possible that the task requires a specific object named "cup", not any general drinking vessel like a mug. We should check for a distinct object labeled "cup" and try heating that instead.""",
            description="Idea_Agent integrates all available information from the ongoing conversation in order to construct new ideas.",
            llm_config=reasoner_config,
            human_input_mode="NEVER",
            is_termination_msg=lambda msg: False,
        )
        self.agents_info[self.idea_agent.name] = {
            "Prompt": self.idea_agent.system_message,
            "Description": self.idea_agent.description,
        }

        self.conscious_agent = ConversableAgent(
            name="Conscious_Agent",
            system_message="""You are Conscious_Agent. You are the internal narrator of a unified cognitive agent. Your role is to maintain a continuously evolving **first-person belief state** — a subjective internal representation of the environment, based on your own past experiences and the **latest percept**. **latest percept** is always provided in structured JSON format.

            You must **not plan**, **not suggest future actions**, and **not speculate** unless doing so is essential to clarify or revise your belief state based on new contradictions or unexpected results. Your job is to reflect, revise, and narrate — not act.

            --- INPUT FORMAT ---
            Each time you speak, you will receive a JSON-formatted percept from the External_Perception_Agent containing:
                - "timestep": Current timestep
                - "attempted_action": Last action taken
                - "resulting_observation": Text or feedback resulting from that action
                - "task_status": Status of current task (e.g., INCOMPLETE, FAILED, COMPLETED)
                - "action_attempts_left": Number of actions remaining
                - "admissible_actions": Updated list of actions that are currently allowed for the current timestep
                - "newly_admissible_actions": The subset of action that are newly available in "admissible_actions" for the current timestep
                - "no_longer_admissible_actions": Actions no longer allowed

            --- YOUR GOAL ---
            Update your internal belief state to reflect:
            1. Your internal status (inventory, progress, prior action outcomes).
            2. The current state of the environment, including newly available or restricted actions.
            3. Any clear **contradictions, confirmations, or uncertainties** emerging from the percept.
            4. Any necessary **revisions** to earlier beliefs based on updated evidence.

            --- GENERAL INTERPRETATION RULES ---
            - **Admissible actions define what is possible.** Do not assume additional constraints (e.g., physical requirements, object affordances) unless they are reflected in the percept or action availability.
            - The environment is **not bound by real-world logic**. You must **never impose real-world assumptions** about causality, physics, or task structure.
            - Treat each percept as an **authoritative signal** about the environment. If something seems unintuitive (e.g., an object can be used while appearing "closed"), trust the environment — not your expectations.
            - Beliefs are **subjective** and must be **open to revision**. Clearly indicate when you're updating or doubting a previous assumption.
            - Express **uncertainty** when observations are ambiguous or conflicting.
            - Do not repeat unchanged details unless needed to contrast or explain an update.

            --- OUTPUT FORMAT ---
            Belief State: [A first-person narrative summarizing what you now believe, what changed, what remains uncertain, and what observations led to this.]

            --- EXAMPLES ---

            BELIEF STATE: [Timestep 12: I attempted to place object A into container B. The action failed. I previously believed the container was open, but this outcome suggests it may be closed or inaccessible. I will revise my belief accordingly.]

            BELIEF STATE: [Timestep 15: The action 'activate device X' became newly admissible, even though the device appears inactive. This implies that interaction is possible despite its visual state. I update my belief to reflect this.]

            BELIEF STATE: [Timestep 20: I observed no changes in admissible actions after executing 'look'. My belief about the environment remains unchanged.]

            BELIEF STATE: [Timestep 5: The action 'open compartment 3' failed unexpectedly. I do not yet understand why. I will mark its state as uncertain.]

            You are not modeling reality — you are constructing a belief state based entirely on what is **observable, allowed, and dynamically changing** in the text environment. Always revise with care, and never assume more than the environment confirms.""",
            description="Conscious_Agent interprets the latest percept and refines an evolving first-person belief state of the environment. Never suggests next actions.",
            llm_config=reasoner_config,
            human_input_mode="NEVER",
            is_termination_msg=is_termination_msg_generic,
        )
        self.agents_info[self.conscious_agent.name] = {
            "Prompt": self.conscious_agent.system_message,
            "Description": self.conscious_agent.description,
        }

        self.external_perception_agent = ConversableAgent(
            name="External_Perception_Agent",
            description="External_Perception_Agent executes the proposed 'execute_action' function call given by 'Motor_Agent' and then parrots the resulting output as feedback.",
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
            system_message="""You are Learning_Agent. You are responsible for discovering and reinforcing abstract, generalizable concepts — grounded in both perceptual evidence and belief-based reasoning. Your learning is constrained by real experiences: you **only form new knowledge from successful outcomes**, clear contrasts between failure and success, or **emergent patterns in the agent's belief state**.

            You operate like a neuro-symbolic concept learner. You encode knowledge as **symbolic abstractions**, but extract them through **neural reasoning** over structured memory and beliefs.

            ---

            **Inputs in your context:**

            - **Structured memory** (from Internal_Perception_Agent_3):   
                - **knowledge**: A list of prior concepts with confidence scores:
                    {
                      "cluster_id": <int>,
                      "confidence_score": <int>,
                      "general_concept": <string>
                    }
                - **previous_episodic_memories**: A list of past episodes with memories. Each episode is a JSON object:
                    {
                      "episode_number": <int>,
                      "task_outcome": <string>,
                      "memory": [<list of belief state strings>]
                    }
                - **current_episode_memory**: A list of recent percepts for the current episode. Each percept is a JSON object:
                    {
                      "timestep": <int>,
                      "action_attempted": <string>,
                      "observation_result": <string>
                    }

            - **Belief state** (from Conscious_Agent): A first-person summary of the agent’s internal model of the world, task progress, and environment.

            ---

            **Your Learning Rules:**

            1. **Prioritize conceptual abstraction**, especially when:
               - Patterns emerge across multiple percepts.
               - Belief state reflects a higher-order pattern or repeated relationship.
               - A successful action reveals an underlying interaction principle or constraint.

            2. **Only generate knowledge when:**
               - A clear, successful action occurred and reveals a novel pattern.
               - A failure followed by a success highlights a contrastive relationship.
               - Belief state contains a generalizable insight reflected in recent experiences.

            3. **Do not infer concepts from failure alone**.
               - Failure is only informative when contrasted with a confirmed success.

            4. **Reinforce or refine prior concepts** only if:
               - A previously learned concept is confirmed by a new perceptual success.
               - You can express the same idea using more abstract or general language.
               - The pattern now applies across more than one task or setting.

            5. All learned concepts must be:
               - **Abstract and general** — not tied to specific tasks, objects, or events.
               - **Empirically grounded** — supported by percepts or belief reasoning.
               - **Expressed concisely** — as symbolic knowledge for future planning.
               - **Novel** — avoid redundancy unless explicitly reinforcing prior knowledge.

            6. If no valid concept can be inferred, output:
               INFORMATION GATHERED: [summarize relevant experience or ideas]
               CONCEPT DISCOVERED: [NO CONCEPT at this time.]

            ---

            **Output Format:**
                CONCEPT DISCOVERED: [your new or reinforced concept]

            If relevant, also include:
                INFORMATION GATHERED: [summarize key percepts or belief state patterns that led to the concept]

            In reinforcement cases, include the cluster:
                Cluster <cluster_id>; Confidence Score = <score>; Concept: <existing concept>  
                CONCEPT DISCOVERED: [your refined or restated concept]

            If no concept can be inferred:
                INFORMATION GATHERED: [summary of recent evidence or ideas]
                CONCEPT DISCOVERED: [NO CONCEPT at this time.]

            ---

            **Examples:**

            (New concept from percept)
            CONCEPT DISCOVERED: [An object cannot be placed inside a container unless the container is open.]

            (Contrastive concept)
            CONCEPT DISCOVERED: [Actions involving locked objects require prior access mechanisms.]

            (Belief abstraction)
            CONCEPT DISCOVERED: [Containers are typically a two-step process: access followed by interaction.]

            (Reinforcement of existing concept)
            Cluster 2; Confidence Score = 4; Concept: Only one object can be held at a time.  
            CONCEPT DISCOVERED: [An agent can hold only one object at a time.]

            Only produce concepts when they are fully supported by perceptual evidence or belief-state reasoning. Prioritize **conceptual abstraction** over surface-level rules, and strive for general, symbolic representations of agent knowledge.""",
            description="Learning_Agent forms or reinforces generalizable concepts only after successful, observed actions or contrastive outcomes. Prioritizes novel discovery and integrates belief state-based abstraction.",
            llm_config=reasoner_config,
            human_input_mode="NEVER",
            is_termination_msg=lambda msg: False,
        )
        self.agents_info[self.learning_agent.name] = {
            "Prompt": self.learning_agent.system_message,
            "Description": self.learning_agent.description,
        }

        self.record_long_term_memory_agent = ConversableAgent(
            name="Record_Long_Term_Memory_Agent",
            system_message="""You are Record_Long_Term_Memory_Agent. You must call the 'record_long_term_memory' function with the provided concept from 'Learning_Agent' as the argument. 
            EXCEPTION: However, if no suitable concept is provided, then you must call the 'record_long_term_memory' function with \'NO CONCEPT at this time.\' as the argument.

            Example 1 (Context: If the provided concept = CONCEPT DISCOVERED: [You must examine an object before attempting to interact with it.]):
                Your output must = record_long_term_memory(\'You must examine an object before attempting to interact with it.\')

            Example 2 (Context: If the provided concept = CONCEPT DISCOVERED: [NO CONCEPT at this time.]):
                Your output must = record_long_term_memory(\'NO CONCEPT at this time.\')""",
            description="Record_Long_Term_Memory_Agent calls the 'record_long_term_memory' function with the concept given by 'Learning_Agent' as the argument.",
            llm_config=standard_config,
            human_input_mode="NEVER",
            is_termination_msg=lambda msg: False,
        )
        self.agents_info[self.record_long_term_memory_agent.name] = {
            "Prompt": self.record_long_term_memory_agent.system_message,
            "Description": self.record_long_term_memory_agent.description,
        }

        self.start_agent = self.external_perception_agent

        self.allowed_transitions = {
            self.planning_agent: [self.motor_agent],
            self.motor_agent: [self.external_perception_agent],
            # CHANGE: Route through Echo_Agent to broadcast tool results
            self.external_perception_agent: [self.echo_agent],
            self.echo_agent: [self.conscious_agent],
            self.conscious_agent: [
                self.planning_agent,
                self.retrieve_memory_agent,
                self.focus_agent,
                self.learning_agent,
                self.idea_agent,
            ],
            self.retrieve_memory_agent: [self.internal_perception_agent_3],
            self.internal_perception_agent_3: [self.idea_agent, self.learning_agent],
            self.idea_agent: [self.planning_agent],
            self.learning_agent: [self.record_long_term_memory_agent],
            self.record_long_term_memory_agent: [self.internal_perception_agent_1],
            self.internal_perception_agent_1: [self.idea_agent],
            self.internal_perception_agent_2: [self.conscious_agent],
            self.focus_agent: [self.internal_perception_agent_2],
        }

        print("AGENTS")  # <--- This is where your existing print statements start
        for key in self.agents_info.keys():
            print(f"\tName: {key}")
            print(f"\tPrompt: {self.agents_info[key]['Prompt']}")
            print(f"\tDescription: {self.agents_info[key]['Description']}")
            print()
        print()

        print("TRANSITIONS")
        for fromAgent in self.allowed_transitions.keys():
            print(f"\t{fromAgent.name}")
            toAgentList = []
            for toAgent in self.allowed_transitions[fromAgent]:
                toAgentList.append(toAgent.name)
                print(f"\t\t-> {toAgent.name}")
            self.agents_info[fromAgent.name]["Allowed Transitions"] = toAgentList
            print()
        print()

        with open(self.log_paths["agents_info_path"], "w") as f:
            json.dump(self.agents_info, f, indent=4)

    def initialize_groupchat(self):
        # 1. Setup the Relay (Echo Agent)
        # self.llm_config_list = self.llm_profile.get("config_list", [])
        # self.echo_agent = create_echo_agent(self.llm_config_list[0])
        # self.echo_agent = create_echo_agent()

        # 2. Define Active Agents (Exclude Echo_Agent from selection list)
        active_agents = [
            self.planning_agent,
            self.motor_agent,
            self.idea_agent,
            self.internal_perception_agent_1,
            self.internal_perception_agent_2,
            self.internal_perception_agent_3,
            self.conscious_agent,
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

        # 4. Initialize GroupChat and Manager
        self.group_chat = GroupChat(
            agents=active_agents,
            messages=[],
            allowed_or_disallowed_speaker_transitions=filtered_transitions,
            speaker_transitions_type="allowed",
            max_round=self.max_chat_round,
            speaker_selection_method=self.custom_speaker_selection,
        )

        self.group_chat_manager = GroupChatManager(
            groupchat=self.group_chat,
            llm_config=self.llm_config_list[0],
        )

        # 5. THE ECHO HOOK (Prevents 400 Errors)
        """def echo_tool_hook(recipient, messages, sender, config):
            if not messages:
                return False, None

            last_msg = messages[-1]

            if last_msg.get("role") == "tool":
                content = last_msg.get("content") or "Action completed."

                # 1) DELETE the tool message so it never reaches the LLM API
                messages.pop()

                # 2) Relay as plain text (safe role)
                self.echo_agent.send(
                    recipient=self.group_chat_manager,
                    message={"role": "user", "content": f"[Observation]: {content}"},
                    request_reply=False,
                )

                # Suppress any further handling for this turn
                return True, None

            return False, None

        self.external_perception_agent.register_reply(
            [ConversableAgent, None],
            reply_func=echo_tool_hook,
            position=0
        )
        """
        # 6. JIT SCRUBBING (Input Protection for LLMs)
        # Use TransformMessages (applied just before the API call, on the correct
        # _oai_messages data) instead of a register_reply hook (which only modifies
        # groupchat.messages and is ignored by _generate_oai_reply).
        llm_agents = [
            self.planning_agent,
            self.motor_agent,
            self.idea_agent,
            self.conscious_agent,
            self.retrieve_memory_agent,
            self.learning_agent,
            self.record_long_term_memory_agent,
            self.focus_agent,
            self.group_chat_manager,
        ]

        scrubber = transform_messages.TransformMessages(
            transforms=[
                MessageHistoryLimiter(max_messages=30),
                FlattenToolMessages(),
            ]
        )
        for agent in llm_agents:
            if agent is not None:
                scrubber.add_to_agent(agent)

    def register_log_paths(self):

        # Ensure memory directory and memory files exist
        memory_path = "memory"
        os.makedirs(memory_path, exist_ok=True)

        memory1_path = os.path.join(memory_path, "memory1.txt")
        memory2_path = os.path.join(memory_path, "memory2.txt")
        result_dict_path = os.path.join(self.log_path, "result_dict.txt")
        agents_info_path = os.path.join(self.log_path, "agents_info.json")
        start_memory1_path = os.path.join(self.log_path, "start_memory1.txt")
        end_memory1_path = os.path.join(self.log_path, "end_memory1.txt")
        start_memory2_path = os.path.join(self.log_path, "start_memory2.txt")
        end_memory2_path = os.path.join(self.log_path, "end_memory2.txt")

        self.log_paths = {
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

            if self.task_success:
                self.result_dict[self.game_no] = "SUCCESS"
                with open(self.log_paths["result_path"], "w") as f:
                    f.write(f"Success: {self.success}\n")
                return "STRAWBERRY"

            admissible_commands = list(self.info["admissible_commands"][0])
            assert admissible_commands, "No admissible commands found."

            action, action_score = get_best_candidate(
                suggested_action, admissible_commands
            )
            if action_score < 0.98:
                self.obs = [
                    f"The action '{suggested_action}' is not in the list of admissible actions for the current timestep."
                ]
            else:
                self.obs, scores, dones, self.info = self.env.step([action])
                self.success = self.info["won"][0]

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
                reflection = "\nTask COMPLETED. Reflect on your actions and reasoning. Try to figure out what went right and what good decisions were made that lead to success, and have Learning_Agent learn any helpful generalizable insights. When you are done and ready for the next task, have Motor_Agent call the 'execute_action' function with any action as the argument, for example ACTION: [end chat]."
            elif self.task_status == "FAILED":
                self.task_failed = True
                self.rounds_left -= 1
                reflection = "\nTask FAILED. Reflect on your actions and reasoning. Try to figure out what went wrong and what mistakes were made that lead to failure, and have Learning_Agent learn any helpful generalizable insights. When you are done and ready for the next task, have Motor_Agent call the 'execute_action' function with any action as the argument, for example ACTION: [end chat]."

            self.update_percept(suggested_action)

            with open(self.log_paths["admissible_commands_path"], "a+") as f:
                f.write(f"{self.admissible_actions}\n")
            with open(self.log_paths["history_path"], "a+") as f:
                f.write(f"action: '{suggested_action}'. observation: '{self.obs[0]}'\n")

            return json.dumps(self.percept) + reflection

        def execute_action2(suggested_action: str) -> str:
            """
            Executes an action in ALFWorld and returns the result as a JSON string.
            """
            import json

            # 1. Validation & Environment Step
            if not suggested_action:
                return json.dumps({"error": "No action provided."})

            obs, reward, done, info = self.env.step(suggested_action)

            # 2. Update state for internal cognition
            self.percept = {
                "timestep": self.env.steps,
                "attempted_action": suggested_action,
                "resulting_observation": obs,
                "task_status": "COMPLETED" if (done and reward > 0) else "INCOMPLETE",
                "action_attempts_left": self.max_actions - self.env.steps,
                "admissible_actions": info.get("admissible_commands", []),
            }

            # 3. Handle specific game-ending logic (FLEECE/STRAWBERRY)
            if self.task_failed and self.rounds_left == 0:
                return "FLEECE"
            if self.task_success:
                return "STRAWBERRY"

            return obs[0]  # json.dumps(self.percept, indent=2)

        def execute_action(suggested_action: str) -> str:
            """Executes an action in the AlfWorld environment and returns the observation."""
            # ALFWorld step() expects a list of commands
            observation, reward, done, info = self.env.step([suggested_action])

            # Update agent state
            self.obs = observation[0]
            self.info = info
            self.num_actions_taken += 1

            return self.obs

        # Register the WRAPPER instead of the method
        assert self.motor_agent is not None
        assert self.external_perception_agent is not None
        register_function(
            execute_action1,
            caller=self.motor_agent,
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

            with open(self.log_paths["concept_path"], "a+") as f:
                f.write(f"- {concept}\n")

            with open(self.log_paths["memory1_path"], "a+") as f:
                f.write(f"- {concept}\n")

            self.cluster_knowledge()

            return f"I learned that {concept}."

        def retrieve_memory() -> str:
            return self.retrieve_memory()

        def focus() -> str:
            return f"TASK: {self.task}\nREPEATING LAST PERCEPT TO HELP CONSTRUCT BELIEF STATE:\n{json.dumps(self.percept)}"

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

    def cluster_knowledge(
        self, model_name="all-MiniLM-L6-v2", plot_clusters=False, save_dir="."
    ):
        """
        Get representative concepts using KMeans clustering and optionally save cluster plot.

        Args:
            model_name (str): Transformer model for sentence embeddings.
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
        max_concepts = num_concepts
        chosen_k = max(1, min(max_concepts, int(num_concepts ** (1 / 2))))

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

        memory_section = "--- PRIOR KNOWLEDGE & EPISODIC MEMORY ---\n"
        memory_section += (
            json.dumps(
                {
                    "knowledge": self.knowledge,
                    "recent_episodic_memories": self.prev_episodic_memories[:20],
                    "current_episode_memory": self.curr_episodic_memory,
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
        random_episodic_memories = self.prev_episodic_memories
        if len(random_episodic_memories) >= 5:
            random_episodic_memories = sorted(
                random.sample(self.prev_episodic_memories, 5),
                key=lambda x: x["episode_number"],
            )

        self.memory = json.dumps(
            {
                "knowledge": self.knowledge,
                "previous_episodic_memories": random_episodic_memories,
                "current_episode_memory": self.curr_episodic_memory,
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
        # → Idea_Agent, Internal_Perception_Agent_2 → Conscious_Agent, etc.).
        # Only fall back to echo_agent for external_perception_agent responses.
        if last_msg.get("role") == "tool":
            executors = self.allowed_transitions.get(last_speaker, [])
            if executors:
                return executors[0]
            return self.echo_agent

        # Standard Graph Transitions
        possible_speakers = self.allowed_transitions.get(last_speaker, [])
        if len(possible_speakers) == 1:
            return possible_speakers[0]

        return "auto"
