import importlib
import json
import sys
import types
from pathlib import Path
from types import SimpleNamespace


def _install_gwt_import_stubs() -> None:
    matplotlib_module = types.ModuleType("matplotlib")
    pyplot_module = types.ModuleType("matplotlib.pyplot")
    pyplot_module.figure = lambda *args, **kwargs: None
    pyplot_module.scatter = lambda *args, **kwargs: None
    pyplot_module.title = lambda *args, **kwargs: None
    pyplot_module.xlabel = lambda *args, **kwargs: None
    pyplot_module.ylabel = lambda *args, **kwargs: None
    pyplot_module.legend = lambda *args, **kwargs: None
    pyplot_module.grid = lambda *args, **kwargs: None
    pyplot_module.tight_layout = lambda *args, **kwargs: None
    pyplot_module.savefig = lambda *args, **kwargs: None
    pyplot_module.close = lambda *args, **kwargs: None
    matplotlib_module.pyplot = pyplot_module
    sys.modules["matplotlib"] = matplotlib_module
    sys.modules["matplotlib.pyplot"] = pyplot_module

    numpy_module = types.ModuleType("numpy")
    numpy_module.unique = lambda *args, **kwargs: ([], [])
    numpy_module.array = lambda value, *args, **kwargs: value
    numpy_module.argmin = lambda values, *args, **kwargs: 0
    numpy_module.linalg = types.SimpleNamespace(norm=lambda *args, **kwargs: 0)
    sys.modules["numpy"] = numpy_module

    umap_module = types.ModuleType("umap")

    class UMAP:
        def __init__(self, *args, **kwargs):
            pass

        def fit_transform(self, embeddings):
            return embeddings

    umap_module.UMAP = UMAP
    sys.modules["umap"] = umap_module

    sklearn_module = types.ModuleType("sklearn")
    sklearn_cluster_module = types.ModuleType("sklearn.cluster")

    class KMeans:
        def __init__(self, *args, **kwargs):
            self.cluster_centers_ = []

        def fit_predict(self, embeddings):
            return []

    sklearn_cluster_module.KMeans = KMeans
    sklearn_module.cluster = sklearn_cluster_module
    sys.modules["sklearn"] = sklearn_module
    sys.modules["sklearn.cluster"] = sklearn_cluster_module

    autogen_module = types.ModuleType("autogen")

    class ConversableAgent:
        def __init__(self, *args, **kwargs):
            self.name = kwargs.get("name", "agent")
            self.system_message = kwargs.get("system_message")
            self.description = kwargs.get("description")

        def clear_history(self):
            return None

    class GroupChat:
        def __init__(self, *args, **kwargs):
            self.agents = kwargs.get("agents", [])
            self.messages = kwargs.get("messages", [])

    class GroupChatManager:
        def __init__(self, *args, **kwargs):
            pass

    autogen_module.ConversableAgent = ConversableAgent
    autogen_module.GroupChat = GroupChat
    autogen_module.GroupChatManager = GroupChatManager
    autogen_module.register_function = lambda *args, **kwargs: None
    sys.modules["autogen"] = autogen_module

    agentchat_module = types.ModuleType("autogen.agentchat")
    contrib_module = types.ModuleType("autogen.agentchat.contrib")
    capabilities_module = types.ModuleType("autogen.agentchat.contrib.capabilities")
    transform_messages_module = types.ModuleType(
        "autogen.agentchat.contrib.capabilities.transform_messages"
    )
    transforms_module = types.ModuleType(
        "autogen.agentchat.contrib.capabilities.transforms"
    )

    class TransformMessages:
        def __init__(self, *args, **kwargs):
            pass

        def add_to_agent(self, agent):
            return None

    class MessageHistoryLimiter:
        def __init__(self, *args, **kwargs):
            pass

    transform_messages_module.TransformMessages = TransformMessages
    transforms_module.MessageHistoryLimiter = MessageHistoryLimiter
    capabilities_module.transform_messages = transform_messages_module
    sys.modules["autogen.agentchat"] = agentchat_module
    sys.modules["autogen.agentchat.contrib"] = contrib_module
    sys.modules["autogen.agentchat.contrib.capabilities"] = capabilities_module
    sys.modules["autogen.agentchat.contrib.capabilities.transform_messages"] = (
        transform_messages_module
    )
    sys.modules["autogen.agentchat.contrib.capabilities.transforms"] = transforms_module

    helpers_module = types.ModuleType("src.agent.helpers")

    class FlattenToolMessages:
        pass

    class ConvertOrphanedToolMessages:
        pass

    class _SentenceTransformerModel:
        def encode(self, *args, **kwargs):
            return []

    helpers_module.ConvertOrphanedToolMessages = ConvertOrphanedToolMessages
    helpers_module.FlattenToolMessages = FlattenToolMessages
    helpers_module.create_echo_agent = lambda: ConversableAgent(name="Echo_Agent")
    helpers_module.get_best_candidate = lambda text, candidates: (
        candidates[0],
        1.0,
    )
    helpers_module.is_termination_msg_generic = lambda msg: False
    helpers_module.sentence_transformer_model = _SentenceTransformerModel()
    sys.modules["src.agent.helpers"] = helpers_module

    rag_memory_module = types.ModuleType("src.agent.rag_memory")
    rag_memory_module.retrieve_relevant_concepts = (
        lambda knowledge, query, k=5, cache=None: knowledge[:k]
    )
    rag_memory_module.retrieve_relevant_episodes = (
        lambda episodes, query, k=5, cache=None: episodes[:k]
    )
    sys.modules["src.agent.rag_memory"] = rag_memory_module


def _load_gwt_agent_module():
    _install_gwt_import_stubs()
    sys.modules.pop("src.agent.gwt_agent", None)
    return importlib.import_module("src.agent.gwt_agent")


def _build_agent(tmp_path: Path, env_type: str):
    gwt_module = _load_gwt_agent_module()
    agent = gwt_module.GWTAutogenAgent.__new__(gwt_module.GWTAutogenAgent)
    agent.args = SimpleNamespace(env_type=env_type)
    agent.log_path = str(tmp_path / "logs")
    Path(agent.log_path).mkdir(parents=True, exist_ok=True)
    agent.log_paths = {}
    agent.adapter = None
    agent.task = ""
    agent.admissible_actions = []
    agent.percept = {}
    agent.curr_episodic_memory = []
    agent.prev_episodic_memories = []
    agent.knowledge = []
    agent.task_status = "INCOMPLETE"
    agent.max_actions = 35
    agent.num_actions_taken = 0
    agent._task_contract = {}
    agent._task_contract_source = ""
    agent._last_action_shortlist = []
    agent.group_chat_manager = None
    agent.group_chat = None
    agent.action_agent = None
    agent._action_agent_base_prompt = ""
    agent.support_config = None
    agent.reasoner_config = None
    agent._episodic_rag_cache = {}
    agent._concept_rag_cache = {}
    agent._reset_episode_reasoning_state()
    return agent, gwt_module


def test_register_log_paths_uses_env_type_before_adapter_is_set(tmp_path):
    agent, gwt_module = _build_agent(tmp_path, env_type="scienceworld")

    memory_root = tmp_path / "agent-memory"
    gwt_module.GWTAutogenAgent._MEMORY_ROOT = memory_root
    scienceworld_dir = memory_root / "scienceworld"
    scienceworld_dir.mkdir(parents=True, exist_ok=True)
    (scienceworld_dir / "memory1.txt").write_text("science concept\n")
    (scienceworld_dir / "memory2.txt").write_text("science cluster\n")

    agent.register_log_paths()

    assert Path(agent.log_paths["memory_dir"]) == scienceworld_dir
    assert (
        Path(agent.log_paths["start_memory1_path"]).read_text() == "science concept\n"
    )
    assert (
        Path(agent.log_paths["start_memory2_path"]).read_text() == "science cluster\n"
    )


def test_register_log_paths_switches_to_adapter_environment(tmp_path):
    agent, gwt_module = _build_agent(tmp_path, env_type="scienceworld")
    env_adapter = importlib.import_module("src.agent.env_adapter")

    memory_root = tmp_path / "agent-memory"
    gwt_module.GWTAutogenAgent._MEMORY_ROOT = memory_root
    alfworld_dir = memory_root / "alfworld"
    alfworld_dir.mkdir(parents=True, exist_ok=True)
    (alfworld_dir / "memory1.txt").write_text("alf concept\n")
    (alfworld_dir / "memory2.txt").write_text("alf cluster\n")

    obs = ["intro\n\nYou are in a kitchen.\n\nYour task is to: put apple on table."]
    info = {"admissible_commands": [["look"]], "won": [False]}
    agent.adapter = env_adapter.ALFWorldAdapter(env=None, obs=obs, info=info)

    agent.register_log_paths()

    assert Path(agent.log_paths["memory_dir"]) == alfworld_dir
    assert Path(agent.log_paths["memory1_path"]) == alfworld_dir / "memory1.txt"
    assert Path(agent.log_paths["memory2_path"]) == alfworld_dir / "memory2.txt"


def test_episode_hypothesis_ledger_is_episode_local_and_deprioritizes_repeats(
    tmp_path,
):
    agent, _ = _build_agent(tmp_path, env_type="scienceworld")
    agent.task_status = "INCOMPLETE"
    agent._reset_episode_reasoning_state()

    agent.percept = {
        "resulting_observation": "The action 'connect sink to stove' is not in the list of admissible actions for the current timestep.",
    }
    agent._update_episode_hypothesis_ledger(
        suggested_action="connect sink to stove",
        executed_action=None,
        previous_observation="",
    )
    agent._update_episode_hypothesis_ledger(
        suggested_action="connect sink to stove",
        executed_action=None,
        previous_observation="",
    )

    snapshot = agent._get_episode_hypothesis_snapshot()
    relation_entry = snapshot["mechanisms"][0]
    assert relation_entry["family"] == "relation"
    assert relation_entry["status"] == "deprioritized"
    assert relation_entry["invalid_attempts"] == 2

    agent._reset_episode_reasoning_state()
    assert agent._get_episode_hypothesis_snapshot() == {
        "mechanisms": [],
        "recent_tests": [],
    }


def test_retrieve_memory_serializes_episode_hypothesis_ledger(tmp_path):
    agent, _ = _build_agent(tmp_path, env_type="scienceworld")
    agent.task = "Heat the water."
    agent.curr_episodic_memory = []
    agent.prev_episodic_memories = []
    agent.knowledge = []
    agent.rag_episode_k = 4
    agent.rag_concept_k = 5
    agent._episodic_rag_cache = {}
    agent._concept_rag_cache = {}
    agent.task_status = "INCOMPLETE"
    agent._reset_episode_reasoning_state()
    agent.percept = {
        "resulting_observation": "The lighter heats up the water a small amount.",
    }

    agent._update_episode_hypothesis_ledger(
        suggested_action="use lighter on water",
        executed_action="use lighter on water",
        previous_observation="The water is cold.",
    )

    payload = json.loads(agent.retrieve_memory())
    mechanisms = payload["episode_hypothesis_ledger"]["mechanisms"]
    assert mechanisms[0]["family"] == "tool_application"
    assert mechanisms[0]["status"] == "promising"


def test_update_percept_adds_action_summary_and_refreshes_private_shortlist(tmp_path):
    agent, _ = _build_agent(tmp_path, env_type="scienceworld")
    agent.task = "Determine whether aluminum foil is electrically conductive."
    agent.task_status = "INCOMPLETE"
    agent.curr_episodic_memory = []
    agent.num_actions_taken = 0
    agent.max_actions = 35
    agent.action_agent = SimpleNamespace(system_message="")
    agent._action_agent_base_prompt = "BASE ACTION PROMPT"
    agent._reset_episode_reasoning_state()
    agent.admissible_actions = ["look around"]
    agent.adapter = SimpleNamespace(
        admissible_actions=[
            "look around",
            "look at aluminum foil",
            "pick up aluminum foil",
            "connect aluminum foil to battery",
            "open drawer",
        ],
        observation="You are in the workshop with aluminum foil, a battery, and a drawer.",
    )

    agent.update_percept("go to workshop")

    summary = agent.percept["admissible_action_summary"]
    shortlist = agent.percept["task_relevant_action_shortlist"]

    assert summary["total_actions"] == 5
    assert summary["current_phase"] == "gather_evidence"
    assert summary["family_counts"]["inspect"] >= 2
    assert "aluminum" in summary["salient_entities"]
    assert "look at aluminum foil" in shortlist
    assert "pick up aluminum foil" in shortlist
    assert (
        "Current exact task-relevant admissible shortlist"
        in agent.action_agent.system_message
    )
    assert "Salient grounded entities" in agent.action_agent.system_message
    assert "connect aluminum foil to battery" in agent.action_agent.system_message


def test_custom_speaker_selection_repairs_malformed_belief_state_and_continues(
    tmp_path,
):
    agent, _ = _build_agent(tmp_path, env_type="scienceworld")
    agent.focus_agent = object()
    agent.belief_state_agent = object()
    agent.thinking_agent = object()
    agent.action_agent = object()
    agent.echo_agent = object()
    agent.learning_agent = object()
    agent.retrieve_memory_agent = object()
    agent.task_success = False
    agent.task_failed = False
    agent.task_status = "INCOMPLETE"
    agent.max_actions = 35
    agent.num_actions_taken = 3
    agent.percept = {
        "timestep": 3,
        "attempted_action": "look at aluminum foil",
        "resulting_observation": "The aluminum foil is a thin sheet of metal.",
        "task_status": "INCOMPLETE",
        "action_attempts_left": 32,
    }
    agent._reset_episode_reasoning_state()
    agent._last_belief_content = "stale"
    agent._consecutive_thinking_count = 2
    agent.allowed_transitions = {
        agent.belief_state_agent: [
            agent.action_agent,
            agent.thinking_agent,
            agent.focus_agent,
        ]
    }

    groupchat = SimpleNamespace(
        messages=[{"content": '{"attempted_action": "use lighter on water"}'}]
    )

    next_speaker = agent.custom_speaker_selection(agent.belief_state_agent, groupchat)

    assert next_speaker is agent.action_agent
    assert groupchat.messages[-1]["content"].startswith("BELIEF STATE:")
    assert "latest confirmed percept" in groupchat.messages[-1]["content"]
    assert agent._last_belief_content == ""
    assert agent._consecutive_thinking_count == 0


def test_recover_from_chat_error_reorders_provider_and_retries(tmp_path):
    agent, _ = _build_agent(tmp_path, env_type="scienceworld")
    agent.task = "Determine whether aluminum foil is electrically conductive."
    agent.admissible_actions = [
        "look at aluminum foil",
        "pick up aluminum foil",
        "connect aluminum foil to battery",
    ]
    agent.action_agent = SimpleNamespace(system_message="")
    agent._action_agent_base_prompt = "BASE ACTION PROMPT"
    agent.support_config = {
        "config_list": [
            {"api_type": "google", "model": "gemini-2.0-flash"},
            {"api_type": "openai", "model": "deepseek-chat"},
        ]
    }
    agent.reasoner_config = {
        "config_list": [
            {"api_type": "google", "model": "gemini-2.0-flash"},
            {"api_type": "openai", "model": "deepseek-reasoner"},
        ]
    }
    agent.group_chat_manager = SimpleNamespace(
        llm_config=agent.support_config["config_list"][0]
    )
    agent.group_chat = SimpleNamespace(messages=[{"content": "in-flight"}])
    agent._reset_episode_reasoning_state()
    agent.resume_chat = lambda messages: ("recovered", None)

    recovered = agent.recover_from_chat_error(
        error=RuntimeError(
            "429 RESOURCE_EXHAUSTED from generativelanguage.googleapis.com for gemini"
        ),
        initial_message_content="retry",
        stage="resume",
    )

    assert recovered == ("recovered", None)
    assert [cfg["api_type"] for cfg in agent.support_config["config_list"]] == [
        "openai",
        "google",
    ]
    assert [cfg["api_type"] for cfg in agent.reasoner_config["config_list"]] == [
        "openai",
        "google",
    ]
    assert agent.group_chat_manager.llm_config["api_type"] == "openai"
    assert (
        "Current exact task-relevant admissible shortlist"
        in agent.action_agent.system_message
    )


def test_set_agent_system_message_supports_read_only_autogen_property(tmp_path):
    agent, gwt_module = _build_agent(tmp_path, env_type="scienceworld")

    class ReadOnlyAgent:
        def __init__(self):
            self._oai_system_message = [{"role": "system", "content": "old"}]

        @property
        def system_message(self):
            return self._oai_system_message[0]["content"]

    read_only_agent = ReadOnlyAgent()

    gwt_module.GWTAutogenAgent._set_agent_system_message(read_only_agent, "new prompt")

    assert read_only_agent.system_message == "new prompt"


def test_action_family_classification_covers_common_general_verbs(tmp_path):
    agent, _ = _build_agent(tmp_path, env_type="scienceworld")

    assert agent._classify_action_family("inspect sink") == "inspect"
    assert agent._classify_action_family("take ceramic cup") == "relocation"
    assert agent._classify_action_family("fill kettle from tap") == (
        "transfer_or_transform"
    )
    assert agent._classify_action_family("turn on stove") == "device_control"


def test_shortlist_balances_families_and_grounds_on_observation(tmp_path):
    agent, _ = _build_agent(tmp_path, env_type="scienceworld")
    agent.task = "Boil water in the kitchen."
    agent.task_status = "INCOMPLETE"
    agent._reset_episode_reasoning_state()
    agent.percept = {
        "resulting_observation": (
            "This room is called the hallway. "
            "You also see a door to the kitchen and a door to the workshop."
        )
    }

    summary = agent._summarize_admissible_actions(
        [
            "focus on air",
            "focus on door to kitchen",
            "focus on workshop",
            "look at door to kitchen",
            "open door to kitchen",
            "go to kitchen",
            "wait",
        ],
        shortlist_limit=4,
    )

    shortlist = summary["task_relevant_action_shortlist"]
    assert summary["salient_entities"][:2] == ["hallway", "kitchen"]
    assert "open door to kitchen" in shortlist
    assert "go to kitchen" in shortlist
    assert "look at door to kitchen" in shortlist
    assert sum(action.startswith("focus on") for action in shortlist) <= 1

