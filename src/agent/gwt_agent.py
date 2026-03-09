import hashlib
import json
import os
import re

import matplotlib.pyplot as plt
import numpy as np
import umap
from autogen import ConversableAgent, GroupChat, GroupChatManager, register_function
from autogen.agentchat.contrib.capabilities import transform_messages
from autogen.agentchat.contrib.capabilities.transforms import MessageHistoryLimiter
from sklearn.cluster import KMeans

from src.agent.autogen_agent import AutogenAgent
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
    _UNCERTAINTY_RE = re.compile(
        r"\b(uncertain|unclear|unsure|unknown|conflicting|"
        r"contradictory|ambiguous|stalled)\b"
    )
    def __init__(
        self,
        llm_profile,
        log_path,
        game_no=1,
        max_chat_round=400,
        max_actions=30,
        rounds_per_game=1,
        rag_episode_k=5,
        rag_concept_k=5,
        rag_episode_k_initial=10,
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
        self.planning_agent = None
        self.motor_agent = None
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

    def set_environment(self, env, obs, info, game_no):
        self.env = env
        self.obs = obs
        self.info = info
        self.game_no = game_no

        self.register_game_log_paths()
        self.cluster_knowledge()

        self.num_actions_taken = 0
        self.max_actions = self._initial_max_actions - self.max_round_actions * (self.rounds - 1)
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
        # Reset GroupChat max_round in case stale-termination capped it last game.
        if self.group_chat is not None:
            self.group_chat.max_round = self.max_chat_round
        # Reset echo agent relay state so stale_count from the previous game
        # doesn't fire false system errors at the start of a new game.
        if self.echo_agent is not None and hasattr(self.echo_agent, "_relay_state"):
            self.echo_agent._relay_state["stale_count"] = 0
            self.echo_agent._relay_state["last_obs"] = None
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
        no_longer = sorted(set(self.admissible_actions) - set(curr_admissible))
        newly_added = sorted(set(curr_admissible) - set(self.admissible_actions))
        self.admissible_actions = curr_admissible

        self.percept = {
            "timestep": self.num_actions_taken,
            "attempted_action": action,
            "resulting_observation": self.obs[0],
            "task_status": self.task_status,
            "action_attempts_left": self.max_actions - self.num_actions_taken,
        }
        if self.num_actions_taken == 0:
            # First percept: send the full admissible list as the baseline.
            self.percept["admissible_actions"] = sorted(self.admissible_actions)
        else:
            # Subsequent percepts: send only what changed to reduce prompt tokens.
            # Agents should track the running list: initial + added - removed.
            if newly_added:
                self.percept["newly_admissible_actions"] = newly_added
            if no_longer:
                self.percept["no_longer_admissible_actions"] = no_longer
            if not newly_added and not no_longer:
                self.percept["admissible_actions_unchanged"] = True

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

        self.llm_config_list = self.llm_profile.get("config_list", [])

        # Standard priority: Gemini -> Chat -> Reasoner
        standard_config = {
            "config_list": self.llm_config_list,
            "temperature": 0.0,
            "max_tokens": 200,
        }

        # Reasoner priority: Reasoner -> Chat -> Gemini (reversed)
        reasoner_config = {
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
            system_message=_PROMPTS["retrieve_memory_agent"],
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
            system_message=_PROMPTS["motor_agent"],
            description="Motor_Agent calls the 'execute_action' function with the best admissible action as the argument.",
            llm_config=standard_config,
            human_input_mode="NEVER",
            is_termination_msg=lambda msg: False,
        )
        self.agents_info[self.motor_agent.name] = {
            "Prompt": self.motor_agent.system_message,
            "Description": self.motor_agent.description,
        }

        # Planning agent's prompt is the main limiting factor when it comes to improving success rate.
        self.planning_agent = ConversableAgent(
            name="Planning_Agent",
            system_message=_PROMPTS["planning_agent"],
            description="Planning_Agent proposes a high-level plan to solve the current task.",
            llm_config=standard_config,
            is_termination_msg=lambda msg: False,
            human_input_mode="NEVER",
        )
        self.agents_info[self.planning_agent.name] = {
            "Prompt": self.planning_agent.system_message,
            "Description": self.planning_agent.description,
        }

        self.thinking_agent = ConversableAgent(
            name="Thinking_Agent",
            system_message=_PROMPTS["thinking_agent"],
            description="Thinking_Agent integrates all available information from the ongoing conversation in order to construct new ideas.",
            llm_config=reasoner_config,
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
            llm_config=standard_config,
            human_input_mode="NEVER",
            is_termination_msg=self._make_belief_state_termination_fn(),
        )
        self.agents_info[self.belief_state_agent.name] = {
            "Prompt": self.belief_state_agent.system_message,
            "Description": self.belief_state_agent.description,
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
            system_message=_PROMPTS["learning_agent"],
            description="Learning_Agent forms or reinforces generalizable concepts only after successful, observed actions or contrastive outcomes. Prioritizes novel discovery and integrates belief state-based abstraction.",
            llm_config=standard_config,
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
            # Route through Echo_Agent to broadcast tool results as plain text
            self.external_perception_agent: [self.echo_agent],
            self.echo_agent: [self.belief_state_agent],
            self.belief_state_agent: [
                self.planning_agent,
                self.retrieve_memory_agent,
                self.focus_agent,
                self.learning_agent,
                self.thinking_agent,
            ],
            self.retrieve_memory_agent: [self.internal_perception_agent_3],
            self.internal_perception_agent_3: [self.thinking_agent, self.learning_agent],
            self.thinking_agent: [self.planning_agent],
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
            self.planning_agent,
            self.motor_agent,
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

        # 5 JIT SCRUBBING (Input Protection for LLMs)
        # Use TransformMessages (applied just before the API call, on the correct
        # _oai_messages data) instead of a register_reply hook (which only modifies
        # groupchat.messages and is ignored by _generate_oai_reply).
        #
        # IMPORTANT: Motor_Agent must NOT have FlattenToolMessages applied.
        # FlattenToolMessages strips 'tool_calls' keys from all prior messages,
        # so after a few rounds Motor_Agent's LLM context contains no tool_call
        # examples and the model switches to plain-text output instead of calling
        # execute_action() — breaking the observation relay for the rest of the game.
        # Motor_Agent only needs the history limiter; it uses a standard
        # OpenAI-compatible model that handles tool_calls natively.
        scrubbed_agents = [
            self.planning_agent,
            self.thinking_agent,
            self.belief_state_agent,
            self.retrieve_memory_agent,
            self.learning_agent,
            self.record_long_term_memory_agent,
            self.focus_agent,
            self.group_chat_manager,
        ]

        full_scrubber = transform_messages.TransformMessages(
            transforms=[
                MessageHistoryLimiter(max_messages=30),
                FlattenToolMessages(),
            ]
        )
        for agent in scrubbed_agents:
            if agent is not None:
                full_scrubber.add_to_agent(agent)

        # Motor_Agent must NOT have MessageHistoryLimiter: trimming its context
        # causes ConvertOrphanedToolMessages to strip the tool_call/response pairs,
        # leaving only plain-text "[Calling execute_action]" examples — after which
        # the model outputs text instead of JSON tool calls, stalling the game.
        # Motor_Agent messages are tiny (one tool call each) so no limiter is needed.
        motor_scrubber = transform_messages.TransformMessages(
            transforms=[ConvertOrphanedToolMessages()]
        )
        if self.motor_agent is not None:
            motor_scrubber.add_to_agent(self.motor_agent)

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

            admissible_commands = list(self.info["admissible_commands"][0])
            assert admissible_commands, "No admissible commands found."

            action, action_score = get_best_candidate(
                suggested_action, admissible_commands
            )
            if action_score < 0.98:
                self.obs = [
                    f"The action '{suggested_action}' is not in the list of admissible actions for the current timestep."
                ]
                # Inadmissible actions don't consume the action budget
            else:
                self.obs, _, __, self.info = self.env.step([action])
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
            return (
                f"TASK: {self.task}\n"
                f"REPEATING LAST PERCEPT TO HELP CONSTRUCT BELIEF STATE:\n{json.dumps(self.percept)}\n"
                f"CURRENT ADMISSIBLE ACTIONS: {json.dumps(sorted(self.admissible_actions))}"
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
            self.knowledge, self.task, k=self.rag_concept_k_initial, cache=self._concept_rag_cache
        )
        relevant_episodes = retrieve_relevant_episodes(
            self.prev_episodic_memories, self.task, k=self.rag_episode_k_initial, cache=self._episodic_rag_cache
        )
        memory_section = "--- PRIOR KNOWLEDGE & EPISODIC MEMORY ---\n"
        memory_section += (
            json.dumps(
                {
                    "knowledge": relevant_knowledge,
                    "recent_episodic_memories": relevant_episodes,
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
            self.prev_episodic_memories, query, k=self.rag_episode_k, cache=self._episodic_rag_cache
        )
        relevant_knowledge = retrieve_relevant_concepts(
            self.knowledge, query, k=self.rag_concept_k, cache=self._concept_rag_cache
        )

        self.memory = json.dumps(
            {
                "knowledge": relevant_knowledge,
                "previous_episodic_memories": relevant_episodes,
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
        # → Thinking_Agent, Internal_Perception_Agent_2 → Belief_State_Agent, etc.).
        # Only fall back to echo_agent for external_perception_agent responses.
        if last_msg.get("role") == "tool":
            executors = self.allowed_transitions.get(last_speaker, [])
            if executors:
                return executors[0]
            return self.echo_agent

        # Stuck-state early termination: if num_actions_taken hasn't changed for
        # 8 consecutive speaker-selection calls, force termination to avoid burning
        # tokens on a stalled game (e.g. Motor_Agent outputting plain text).
        if self.num_actions_taken == self._last_seen_actions_taken:
            self._stale_action_count += 1
        else:
            self._stale_action_count = 0
            self._last_seen_actions_taken = self.num_actions_taken
        if self._stale_action_count >= 8:
            self.group_chat.max_round = len(messages)

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
            if self._UNCERTAINTY_RE.search(current_content.lower()) or "no observation" in current_content.lower():
                if current_content == self._last_belief_content and self._consecutive_thinking_count >= 1:
                    # Identical belief state fired again — skip Thinking_Agent
                    self._consecutive_thinking_count = 0
                    self._last_belief_content = ""
                    return self.planning_agent
                self._last_belief_content = current_content
                self._consecutive_thinking_count += 1
                return self.thinking_agent
            self._last_belief_content = ""
            self._consecutive_thinking_count = 0
            return self.planning_agent

        # Safety valve: if the task is done and the conversation is still
        # running (e.g. Motor_Agent stuck outputting text instead of calling
        # execute_action), cap max_round to force termination.
        if self.task_success or (self.task_failed and self.rounds_left == 0):
            if not hasattr(self, "_task_done_msg_count"):
                self._task_done_msg_count = len(messages)
            elif len(messages) > self._task_done_msg_count + 12:
                self.group_chat.max_round = len(messages)

        if len(possible_speakers) == 1:
            return possible_speakers[0]

        return "auto"
