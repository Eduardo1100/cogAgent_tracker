import os
from pathlib import Path
from typing import Any


# This is a template class for the Autogen Agent
class AutogenAgent:
    def __init__(
        self,
        llm_profile,
        log_path,
        game_no=1,
        max_chat_round=400,
        max_actions=50,
        args=None,
        env=None,
        obs="",
        info=None,
    ):
        self.env: Any = env
        self.obs: Any = obs
        self.info: Any = info
        self.game_no = game_no
        self.max_chat_round = max_chat_round
        self.llm_profile = llm_profile
        self.llm_config = llm_profile
        self.llm_config_list = []
        self.log_path = log_path
        self.num_actions_taken = 0
        self.max_actions = max_actions
        self.success = False
        self.args = args
        self.start_agent: Any = None
        self.log_paths: dict[str, Any] = {}
        self.result_dict = {}
        self.allowed_transitions: Any = None

        self.group_chat: Any = None
        self.group_chat_manager: Any = None
        self.analyst_trace_callback: Any = None

        # Shared attributes used by the eval loop
        self.task_status: str = "INCOMPLETE"
        self.rounds_left: int = 1
        self.initial_message: str = ""
        self.curr_episodic_memory: list = []
        self.prev_episodic_memories: list = []

    def set_environment(self, env, obs, info, game_no):
        self.env = env
        self.obs = obs
        self.info = info
        self.game_no = game_no
        self.register_log_paths()

    def initialize_autogen(self):
        self.register_log_paths()
        self.initialize_agents()
        self.register_functions()
        self.initialize_groupchat()

    def initialize_agents(self):
        raise NotImplementedError

    def register_functions(self):
        raise NotImplementedError

    def initialize_groupchat(self):
        raise NotImplementedError

    def _execute_chat_operation(self, fn):
        chat_result = None
        error_message = None
        try:
            chat_result = fn()
        except Exception as e:
            error_message = self._recover_chat_error(e)
        return chat_result, error_message

    def run_chat(self, initial_message_content):
        assert self.start_agent is not None, "self.start_agent must be defined"
        assert self.group_chat_manager is not None, (
            "self.group_chat_manager must be defined"
        )
        assert self.group_chat is not None, "self.group_chat must be defined"

        self.num_actions_taken = 0
        self.success = False

        return self._execute_chat_operation(
            lambda: self.start_agent.initiate_chat(
                self.group_chat_manager,
                message={"role": "system", "content": initial_message_content},
                summary_method="reflection_with_llm",
            )
        )

    def resume_chat(self, last_message):
        def _resume():
            assert self.group_chat_manager is not None
            last_agent, message = self.group_chat_manager.resume(messages=last_message)
            # Resume the chat using the last agent and message
            return last_agent.initiate_chat(
                recipient=self.group_chat_manager,
                message=message,
                clear_history=False,
            )

        return self._execute_chat_operation(_resume)

    def _recover_chat_error(self, error):
        recover = getattr(self, "recover_from_chat_error", None)
        if callable(recover):
            return recover(error=error)
        return None

    def register_log_paths(self):

        game_path = os.path.join(self.log_path, f"game_{self.game_no}")
        os.makedirs(game_path, exist_ok=True)

        task_path = os.path.join(game_path, "task.txt")
        history_path = os.path.join(game_path, "history.txt")
        rule_path = os.path.join(game_path, "rules.txt")
        admissible_commands_path = os.path.join(game_path, "admissible_commands.txt")
        chat_history_path = os.path.join(game_path, "chat_history.txt")
        analyst_trace_path = os.path.join(game_path, "analyst_trace.txt")
        analyst_trace_ansi_path = os.path.join(game_path, "analyst_trace.ansi")
        message_path = os.path.join(game_path, "last_message.pkl")
        result_path = os.path.join(game_path, "result.txt")
        error_message_path = os.path.join(game_path, "error_message.txt")

        # get all the previous game pathF
        previous_game_path = [
            os.path.join(self.log_path, f"game_{i}") for i in range(self.game_no)
        ]
        previous_rule_path = [
            os.path.join(game_path, "rules.txt") for game_path in previous_game_path
        ]

        self.log_paths = {
            "task_path": task_path,
            "history_path": history_path,
            "rule_path": rule_path,
            "admissible_commands_path": admissible_commands_path,
            "chat_history_path": chat_history_path,
            "analyst_trace_path": analyst_trace_path,
            "analyst_trace_ansi_path": analyst_trace_ansi_path,
            "message_path": message_path,
            "result_path": result_path,
            "error_message_path": error_message_path,
            "previous_rule_path": previous_rule_path,
        }

    def get_log_paths(self):
        return self.log_paths

    def _read_log_file(self, key: str) -> str:
        path = self.log_paths.get(key)
        if path and os.path.exists(path):
            return Path(path).read_text()
        return ""

    def get_analyst_trace_text(self) -> str:
        return self._read_log_file("analyst_trace_path")

    def get_analyst_trace_ansi_text(self) -> str:
        return self._read_log_file("analyst_trace_ansi_path")
