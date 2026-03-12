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


def test_task_contract_extracts_required_family_and_target_entities(tmp_path):
    agent, _ = _build_agent(tmp_path, env_type="scienceworld")
    agent.task = (
        "Focus on the 4 life stages of the turtle, starting from earliest to latest."
    )

    contract = agent._get_task_contract()

    assert contract["required_families"] == ["focus"]
    assert "turtle" in contract["target_entities"]
    assert contract["ordering_cues"] == ["ordered_sequence"]


def test_task_contract_drops_stopwords_and_plural_duplicates(tmp_path):
    agent, _ = _build_agent(tmp_path, env_type="scienceworld")
    agent.task = (
        "Focus on the life stages of the apple plant, starting from earliest to latest. "
        "The plants are located outside."
    )

    contract = agent._get_task_contract()

    assert contract["required_families"] == ["focus"]
    assert "apple" in contract["target_entities"]
    assert "plant" in contract["target_entities"]
    assert "plants" not in contract["target_entities"]
    assert "are" not in contract["target_entities"]
    assert contract["ordering_cues"] == ["ordered_sequence"]


def test_canonicalize_suggested_action_snaps_to_exact_admissible_command(tmp_path):
    agent, _ = _build_agent(tmp_path, env_type="scienceworld")

    canonical = agent._canonicalize_suggested_action(
        "move to art studio",
        [
            "go to art studio",
            "open art studio door",
            "look at picture",
        ],
    )

    assert canonical == "go to art studio"


def test_canonicalize_suggested_action_refuses_ambiguous_sibling_targets(tmp_path):
    agent, _ = _build_agent(tmp_path, env_type="scienceworld")
    suggested_action = "focus on apple tree in self watering flower pot 6"

    canonical = agent._canonicalize_suggested_action(
        suggested_action,
        [
            "focus on apple tree in self watering flower pot 4",
            "focus on apple tree in self watering flower pot 7",
            "look at apple tree in self watering flower pot 6",
        ],
    )

    assert canonical == suggested_action


def test_required_task_family_is_not_deprioritized_after_repeated_failures(tmp_path):
    agent, _ = _build_agent(tmp_path, env_type="scienceworld")
    agent.task = "Connect the red wire to the battery."
    agent.task_status = "INCOMPLETE"
    agent._reset_episode_reasoning_state()
    agent.percept = {
        "resulting_observation": (
            "The action 'connect red wire to battery' is not in the list of admissible actions for the current timestep."
        ),
    }

    agent._update_episode_hypothesis_ledger(
        suggested_action="connect red wire to battery",
        executed_action=None,
        previous_observation="",
    )
    agent._update_episode_hypothesis_ledger(
        suggested_action="connect red wire to battery",
        executed_action=None,
        previous_observation="",
    )

    entry = agent.episode_hypothesis_ledger["relation"]
    assert entry["invalid_attempts"] == 2
    assert entry["status"] == "uncertain"
    assert entry["retired"] is False


def test_required_focus_family_survives_shortlist_without_hidden_entity_drift(tmp_path):
    agent, _ = _build_agent(tmp_path, env_type="scienceworld")
    agent.task = (
        "Focus on the 4 life stages of the turtle, starting from earliest to latest."
    )
    agent.task_status = "INCOMPLETE"
    agent._reset_episode_reasoning_state()
    agent.percept = {
        "resulting_observation": (
            "You are in the hallway. You see a picture, an art studio door, and a greenhouse door."
        )
    }

    summary = agent._summarize_admissible_actions(
        [
            "look at picture",
            "open art studio door",
            "go to art studio",
            "focus on picture",
            "focus on turtle egg",
            "focus on turtle",
        ],
        shortlist_limit=4,
    )

    shortlist = summary["task_relevant_action_shortlist"]
    assert "focus on picture" in shortlist
    assert "focus on turtle egg" not in shortlist
    assert "open art studio door" in shortlist
    assert "go to art studio" in shortlist


def test_lifecycle_task_uses_search_phase_until_stage_signal_is_visible(tmp_path):
    agent, _ = _build_agent(tmp_path, env_type="scienceworld")
    agent.task = (
        "Focus on the 4 life stages of the turtle, starting from earliest to latest."
    )
    agent.task_status = "INCOMPLETE"
    agent._reset_episode_reasoning_state()
    agent.percept = {
        "resulting_observation": (
            "This room is called the hallway. You see a picture. "
            "You also see: A door to the art studio (that is closed) and "
            "A door to the greenhouse (that is closed)."
        )
    }

    assert agent._get_current_phase() == "locate_primary_target"

    agent.percept = {
        "resulting_observation": (
            "This room is called the greenhouse. Here you see a turtle egg, "
            "a juvenile turtle, and an adult turtle."
        )
    }

    assert agent._get_current_phase() == "confirm_primary_target"


def test_lifecycle_search_frontier_prefers_room_exit_over_local_container_digging(
    tmp_path,
):
    agent, _ = _build_agent(tmp_path, env_type="scienceworld")
    agent.task = (
        "Focus on the 4 life stages of the turtle, starting from earliest to latest."
    )
    agent.task_status = "INCOMPLETE"
    agent._reset_episode_reasoning_state()
    observation = (
        "Inside the art studio is: the agent, a large cupboard. "
        "The large cupboard door is closed. "
        "a table. On the table is: a jug (containing nothing). "
        "You also see: A door to the hallway (that is open)."
    )
    agent.percept = {"resulting_observation": observation}
    agent._update_lifecycle_search_tracking(
        action="look in art studio",
        observation=observation,
    )

    summary = agent._summarize_admissible_actions(
        [
            "open cupboard",
            "look at jug",
            "look in hallway",
            "go to hallway",
            "focus on art studio",
        ],
        shortlist_limit=2,
    )

    shortlist = summary["task_relevant_action_shortlist"]
    assert summary["current_phase"] == "locate_primary_target"
    assert "go to hallway" in shortlist
    assert "look in hallway" in shortlist
    assert "open cupboard" not in shortlist


def test_room_frontier_relocation_requires_agent_navigation(tmp_path):
    agent, _ = _build_agent(tmp_path, env_type="scienceworld")
    observation = (
        "This room is called the hallway. You also see: "
        "A door to the greenhouse (that is closed)."
    )

    assert (
        agent._action_targets_room_frontier(
            "go to greenhouse",
            observation,
            family="relocation",
        )
        is True
    )
    assert (
        agent._action_targets_room_frontier(
            "move picture to greenhouse",
            observation,
            family="relocation",
        )
        is False
    )


def test_lifecycle_search_prefers_doors_over_object_transport_before_entry(tmp_path):
    agent, _ = _build_agent(tmp_path, env_type="scienceworld")
    agent.task = (
        "Focus on the 4 life stages of the turtle, starting from earliest to latest."
    )
    agent.task_status = "INCOMPLETE"
    agent._reset_episode_reasoning_state()
    agent.percept = {
        "resulting_observation": (
            "This room is called the hallway. In it, you see a picture. "
            "You also see: A door to the art studio (that is closed), "
            "A door to the bedroom (that is closed), "
            "A door to the greenhouse (that is closed), "
            "A door to the kitchen (that is closed)."
        )
    }

    summary = agent._summarize_admissible_actions(
        [
            "move picture to art studio",
            "move picture to bedroom",
            "move picture to greenhouse",
            "open art studio door",
            "open bedroom door",
            "open door to greenhouse",
            "open door to kitchen",
            "look at art studio",
            "look at greenhouse",
            "look around",
        ],
        shortlist_limit=8,
    )

    shortlist = summary["task_relevant_action_shortlist"]
    assert "open art studio door" in shortlist
    assert "open bedroom door" in shortlist
    assert "open door to greenhouse" in shortlist
    assert "move picture to art studio" not in shortlist
    assert "move picture to bedroom" not in shortlist
    assert "move picture to greenhouse" not in shortlist


def test_lifecycle_search_avoids_reentering_exhausted_room_from_hallway(tmp_path):
    agent, _ = _build_agent(tmp_path, env_type="scienceworld")
    agent.task = (
        "Focus on the 4 life stages of the turtle, starting from earliest to latest."
    )
    agent.task_status = "INCOMPLETE"
    agent._reset_episode_reasoning_state()
    art_studio_observation = (
        "This room is called the art studio. In it, you see: a large cupboard. "
        "The large cupboard door is closed. a table with a jug. "
        "You also see: A door to the hallway (that is open)."
    )
    agent.percept = {"resulting_observation": art_studio_observation}
    agent._update_lifecycle_search_tracking(
        action="look at art studio",
        observation=art_studio_observation,
    )
    agent.percept = {
        "resulting_observation": (
            "This room is called the hallway. In it, you see a picture. "
            "You also see: A door to the art studio (that is open), "
            "A door to the bedroom (that is closed), "
            "A door to the greenhouse (that is closed), "
            "A door to the kitchen (that is closed)."
        )
    }

    summary = agent._summarize_admissible_actions(
        [
            "go to art studio",
            "look at art studio",
            "open art studio door",
            "move picture to art studio",
            "open bedroom door",
            "open door to greenhouse",
            "open door to kitchen",
            "look at bedroom",
            "look at greenhouse",
            "look around",
        ],
        shortlist_limit=6,
    )

    shortlist = summary["task_relevant_action_shortlist"]
    assert "open bedroom door" in shortlist
    assert "open door to greenhouse" in shortlist
    assert "go to art studio" not in shortlist
    assert "look at art studio" not in shortlist
    assert "move picture to art studio" not in shortlist


def test_shortlist_prefers_grounded_stage_targets_over_area_focus_and_distractors(
    tmp_path,
):
    agent, _ = _build_agent(tmp_path, env_type="scienceworld")
    agent.task = (
        "Focus on the life stages of the apple plant, starting from earliest to latest. "
        "The plants are located outside."
    )
    agent.task_status = "INCOMPLETE"
    agent._reset_episode_reasoning_state()
    agent.percept = {
        "resulting_observation": (
            "This outside location is called the outside. Here you see a blue jay egg, "
            "a self watering flower pot 4 containing a apple seed, and a self watering "
            "flower pot 6 containing a apple tree in the seedling stage."
        )
    }

    summary = agent._summarize_admissible_actions(
        [
            "focus on outside",
            "look at blue jay egg",
            "focus on apple seed",
            "focus on apple tree in self watering flower pot 6",
        ],
        shortlist_limit=2,
    )

    shortlist = summary["task_relevant_action_shortlist"]
    assert "focus on apple seed" in shortlist
    assert "focus on apple tree in self watering flower pot 6" in shortlist
    assert "focus on outside" not in shortlist
    assert "look at blue jay egg" not in shortlist


def test_ordered_target_progress_penalizes_repeat_focus_on_completed_target(tmp_path):
    agent, _ = _build_agent(tmp_path, env_type="scienceworld")
    agent.task = (
        "Focus on the life stages of the apple plant, starting from earliest to latest."
    )
    agent.task_status = "INCOMPLETE"
    agent._reset_episode_reasoning_state()
    agent.percept = {
        "resulting_observation": (
            "This outside location is called the outside. Here you see a apple seed and "
            "a apple tree in the seedling stage."
        )
    }
    agent._update_ordered_target_progress(
        executed_action="focus on apple seed",
        observation="You focus on the apple seed.",
    )

    summary = agent._summarize_admissible_actions(
        [
            "focus on apple seed",
            "focus on apple tree in self watering flower pot 6",
        ],
        shortlist_limit=1,
    )

    assert summary["task_relevant_action_shortlist"] == [
        "focus on apple tree in self watering flower pot 6"
    ]


def test_referent_resolution_records_ambiguity_without_collapsing_targets(tmp_path):
    agent, _ = _build_agent(tmp_path, env_type="scienceworld")
    agent._reset_episode_reasoning_state()

    resolution = agent._record_referent_resolution(
        suggested_action="focus on apple tree in self watering flower pot 6",
        canonical_action="focus on apple tree in self watering flower pot 4",
    )

    assert resolution == {
        "status": "ambiguous",
        "requested_target": "apple tree flower pot 6",
        "resolved_target": "apple tree flower pot 4",
    }
    assert agent._get_ordered_target_snapshot()["ambiguous_focus_targets"] == [
        resolution
    ]


def test_task_contract_preserves_primary_target_and_relation_roles(tmp_path):
    agent, _ = _build_agent(tmp_path, env_type="scienceworld")
    agent.task = (
        "Turn on the red light bulb by powering it using a renewable power source. "
        "First, focus on the red light bulb. Then, create an electrical circuit that powers it on."
    )

    contract = agent._get_task_contract()

    assert "focus" in contract["required_families"]
    assert "relation" in contract["required_families"]
    assert contract["primary_targets"] == ["red light bulb"]
    assert contract["supporting_targets"] == ["renewable power source"]
    assert contract["required_relations"] == ["electrical circuit"]
    assert contract["inferred_search_mode"] is True
    assert contract["relation_mechanism_task"] is True
    assert "red" in contract["target_entities"]
    assert "circuit" in contract["target_entities"]
    assert "create" not in contract["target_entities"]


def test_ordered_target_progress_preserves_discriminative_target_modifiers(tmp_path):
    agent, _ = _build_agent(tmp_path, env_type="scienceworld")
    agent.task = (
        "Turn on the red light bulb by powering it using a renewable power source. "
        "First, focus on the red light bulb. Then, create an electrical circuit that powers it on."
    )
    agent._reset_episode_reasoning_state()

    agent._update_ordered_target_progress(
        executed_action="focus on red light bulb",
        observation="You focus on the red light bulb.",
    )

    assert agent._get_ordered_target_snapshot()["completed_focus_targets"] == [
        "red light bulb"
    ]


def test_shortlist_prefers_primary_target_and_relation_actions_after_focus(tmp_path):
    agent, _ = _build_agent(tmp_path, env_type="scienceworld")
    agent.task = (
        "Turn on the red light bulb by powering it using the solar panel. "
        "First, focus on the red light bulb. Then, create an electrical circuit that powers it on."
    )
    agent.task_status = "INCOMPLETE"
    agent._reset_episode_reasoning_state()
    observation = (
        "You are in the workshop. You see a red light bulb, a red light bulb anode, "
        "a red light bulb cathode, a red wire, a solar panel, a blue light bulb, "
        "a green light bulb, and a freezer."
    )
    agent.percept = {"resulting_observation": observation}
    agent._update_ordered_target_progress(
        executed_action="focus on red light bulb",
        observation="You focus on the red light bulb.",
    )
    agent._update_relation_task_tracking(
        action="look at solar panel",
        observation=observation,
    )

    summary = agent._summarize_admissible_actions(
        [
            "open freezer",
            "activate freezer",
            "look at blue light bulb",
            "look at red light bulb cathode",
            "connect red wire to red light bulb cathode",
            "connect solar panel to red wire",
        ],
        shortlist_limit=3,
    )

    shortlist = summary["task_relevant_action_shortlist"]
    assert "look at red light bulb cathode" in shortlist
    assert "connect red wire to red light bulb cathode" in shortlist
    assert "connect solar panel to red wire" in shortlist
    assert "open freezer" not in shortlist
    assert "look at blue light bulb" not in shortlist


def test_irrelevant_device_control_does_not_become_promising_from_state_change(
    tmp_path,
):
    agent, _ = _build_agent(tmp_path, env_type="scienceworld")
    agent.task = (
        "Turn on the red light bulb by powering it using a renewable power source. "
        "First, focus on the red light bulb. Then, create an electrical circuit that powers it on."
    )
    agent.task_status = "INCOMPLETE"
    agent._reset_episode_reasoning_state()
    agent.percept = {
        "resulting_observation": "The freezer is now open.",
        "newly_admissible_actions": [],
        "no_longer_admissible_actions": [],
    }

    agent._update_episode_hypothesis_ledger(
        suggested_action="open freezer",
        executed_action="open freezer",
        previous_observation="The freezer is closed.",
    )

    device_control_entry = agent.episode_hypothesis_ledger["device_control"]
    assert device_control_entry["observable_change_attempts"] == 0
    assert device_control_entry["evidence_attempts"] == 1
    assert device_control_entry["status"] == "uncertain"
    assert agent.recent_hypothesis_tests[-1]["outcome"] == "evidence"


def test_lifecycle_task_contract_sets_lifecycle_sequence_mode(tmp_path):
    agent, _ = _build_agent(tmp_path, env_type="scienceworld")
    agent.task = (
        "Focus on the life stages of the apple plant, starting from earliest to latest. "
        "The plants are located outside."
    )

    contract = agent._get_task_contract()

    assert contract["lifecycle_sequence"] is True
    assert contract["required_families"] == ["focus"]


def test_container_focus_does_not_advance_lifecycle_progress_without_stage_evidence(
    tmp_path,
):
    agent, _ = _build_agent(tmp_path, env_type="scienceworld")
    agent.task = (
        "Focus on the life stages of the apple plant, starting from earliest to latest. "
        "The plants are located outside."
    )
    agent._reset_episode_reasoning_state()

    agent._update_lifecycle_stage_state(
        action="focus on flower pot containing apple seed and soil and water",
        observation="You focus on the self watering flower pot 4.",
    )
    agent._update_ordered_target_progress(
        executed_action="focus on flower pot containing apple seed and soil and water",
        observation="You focus on the self watering flower pot 4.",
    )

    snapshot = agent._get_ordered_target_snapshot()
    assert snapshot["completed_focus_targets"] == []
    assert snapshot["focused_stage_labels"] == []


def test_inspection_evidence_turns_into_lifecycle_stage_progress(tmp_path):
    agent, _ = _build_agent(tmp_path, env_type="scienceworld")
    agent.task = (
        "Focus on the life stages of the apple plant, starting from earliest to latest. "
        "The plants are located outside."
    )
    agent._reset_episode_reasoning_state()

    agent._update_lifecycle_stage_state(
        action="look at apple tree in self watering flower pot 4",
        observation="a apple tree in the seedling stage",
    )
    agent._update_ordered_target_progress(
        executed_action="focus on apple tree in self watering flower pot 4",
        observation="You focus on the apple tree.",
    )

    snapshot = agent._get_ordered_target_snapshot()
    assert snapshot["completed_focus_targets"] == ["apple tree flower pot 4"]
    assert snapshot["focused_stage_labels"] == ["seedling"]
    assert snapshot["observed_stage_labels"] == ["seedling"]


def test_container_inspection_maps_stage_evidence_to_focusable_referent(tmp_path):
    agent, _ = _build_agent(tmp_path, env_type="scienceworld")
    agent.task = (
        "Focus on the life stages of the apple plant, starting from earliest to latest. "
        "The plants are located outside."
    )
    agent._reset_episode_reasoning_state()

    agent._update_lifecycle_stage_state(
        action="look at flower pot containing apple tree and soil and water",
        observation=(
            "a self watering flower pot 5 (containing a dead apple tree, soil, "
            "a substance called water)"
        ),
    )
    agent._update_ordered_target_progress(
        executed_action="focus on apple tree in self watering flower pot 5",
        observation="You focus on the apple tree.",
    )

    snapshot = agent._get_ordered_target_snapshot()
    assert snapshot["completed_focus_targets"] == ["apple tree flower pot 5"]
    assert snapshot["focused_stage_labels"] == ["dead"]


def test_lifecycle_shortlist_prefers_inspection_and_stage_bearing_focus(tmp_path):
    agent, _ = _build_agent(tmp_path, env_type="scienceworld")
    agent.task = (
        "Focus on the life stages of the apple plant, starting from earliest to latest. "
        "The plants are located outside."
    )
    agent.task_status = "INCOMPLETE"
    agent._reset_episode_reasoning_state()
    agent._update_ordered_target_progress(
        executed_action="focus on apple seed",
        observation="You focus on the apple seed.",
    )
    agent.percept = {
        "resulting_observation": (
            "This outside location is called the outside. Here you see an apple seed, "
            "an apple tree in self watering flower pot 4, an apple tree in self watering "
            "flower pot 5, and a adult apple tree."
        )
    }

    summary = agent._summarize_admissible_actions(
        [
            "focus on apple tree in self watering flower pot 4",
            "look at apple tree in self watering flower pot 4",
            "focus on adult apple tree",
            "pour apple tree in self watering flower pot 4 into outside",
            "use axe on apple tree in self watering flower pot 5",
        ],
        shortlist_limit=3,
    )

    shortlist = summary["task_relevant_action_shortlist"]
    assert "look at apple tree in self watering flower pot 4" in shortlist
    assert "focus on adult apple tree" in shortlist
    assert "pour apple tree in self watering flower pot 4 into outside" not in shortlist
    assert "use axe on apple tree in self watering flower pot 5" not in shortlist


def test_numeric_ambiguity_choice_resolves_to_semantic_action(tmp_path):
    agent, _ = _build_agent(tmp_path, env_type="scienceworld")

    resolved_action, resolution = agent._resolve_reasoning_action(
        "0",
        (
            "Ambiguous request: Please enter the number for the action you intended "
            "(or blank to cancel):\n"
            "0:\tlook at flower (in apple tree, in self watering flower pot 7, in outside)\n"
            "1:\tlook at flower (in apple tree, in self watering flower pot 9, in outside)\n"
        ),
    )

    assert (
        resolved_action
        == "look at flower (in apple tree, in self watering flower pot 7, in outside)"
    )
    assert resolution == {
        "status": "resolved",
        "choice": "0",
        "resolved_action": "look at flower (in apple tree, in self watering flower pot 7, in outside)",
    }
    assert agent._classify_action_family(resolved_action) == "inspect"


def test_task_contract_separates_candidate_class_and_destination_roles(tmp_path):
    agent, _ = _build_agent(tmp_path, env_type="scienceworld")
    agent.task = (
        "Your task is to find a(n) non-living thing. First, focus on the thing. "
        "Then, move it to the red box in the kitchen."
    )

    contract = agent._get_task_contract()

    assert "focus" in contract["required_families"]
    assert "relocation" in contract["required_families"]
    assert "inspect" not in contract["required_families"]
    assert contract["support_families"] == ["inspect"]
    assert contract["candidate_classes"] == ["non living thing"]
    assert contract["destination_container"] == ["red box"]
    assert contract["destination_room"] == ["kitchen"]


def test_support_entities_do_not_count_as_completed_focus_targets(tmp_path):
    agent, _ = _build_agent(tmp_path, env_type="scienceworld")
    agent.task = (
        "Your task is to find a(n) non-living thing. First, focus on the thing. "
        "Then, move it to the red box in the kitchen."
    )
    agent._reset_episode_reasoning_state()

    agent._update_ordered_target_progress(
        executed_action="focus on picture",
        observation="You focus on the picture.",
    )
    agent._update_ordered_target_progress(
        executed_action="focus on red box",
        observation="You focus on the red box.",
    )
    agent._update_ordered_target_progress(
        executed_action="focus on kitchen",
        observation="You focus on the kitchen.",
    )

    assert agent._get_ordered_target_snapshot()["completed_focus_targets"] == [
        "picture"
    ]


def test_repeated_support_confirmation_becomes_stalled_not_new_evidence(tmp_path):
    agent, _ = _build_agent(tmp_path, env_type="scienceworld")
    agent.task = (
        "Your task is to find a(n) non-living thing. First, focus on the thing. "
        "Then, move it to the red box in the kitchen."
    )
    observation = "a red box (containing a picture)"
    agent.percept = {
        "resulting_observation": observation,
        "newly_admissible_actions": [],
        "no_longer_admissible_actions": [],
    }
    agent._record_action_observation_signature(
        "look at red box", observation, family="inspect"
    )

    outcome, evidence = agent._classify_hypothesis_outcome(
        family="inspect",
        executed_action="look at red box",
        observation=observation,
        previous_observation="You focus on the red box.",
    )

    assert outcome == "stalled"
    assert "Repeated confirmation" in evidence


def test_no_known_action_match_is_classified_as_invalid(tmp_path):
    agent, _ = _build_agent(tmp_path, env_type="scienceworld")
    agent.task = (
        "Your task is to melt water. First, focus on the substance. "
        "Then, take actions that will cause it to change its state of matter."
    )
    agent.percept = {
        "resulting_observation": "No known action matches that input.",
        "newly_admissible_actions": [],
        "no_longer_admissible_actions": [],
    }

    outcome, evidence = agent._classify_hypothesis_outcome(
        family="inspect",
        executed_action="look at tin cup",
        observation="No known action matches that input.",
        previous_observation="You move to the kitchen.",
    )

    assert outcome == "invalid"
    assert "failed" in evidence.lower() or "inadmissible" in evidence.lower()


def test_candidate_tracking_rejects_stale_candidate_after_repeated_confirmation(
    tmp_path,
):
    agent, _ = _build_agent(tmp_path, env_type="scienceworld")
    agent.task = (
        "Your task is to find a(n) non-living thing. First, focus on the thing. "
        "Then, move it to the red box in the kitchen."
    )
    agent.task_status = "INCOMPLETE"
    agent._reset_episode_reasoning_state()

    agent._update_candidate_tracking(
        executed_action="focus on picture",
        observation="You focus on the picture.",
        previous_observation="a picture of a farm.",
    )

    for previous_observation in (
        "You focus on the red box.",
        "a red box (containing a picture)",
        "a red box (containing a picture)",
    ):
        observation = "a red box (containing a picture)"
        agent._update_candidate_tracking(
            executed_action="look at red box",
            observation=observation,
            previous_observation=previous_observation,
        )
        agent.percept = {
            "resulting_observation": observation,
            "newly_admissible_actions": [],
            "no_longer_admissible_actions": [],
        }
        agent._update_episode_hypothesis_ledger(
            suggested_action="look at red box",
            executed_action="look at red box",
            previous_observation=previous_observation,
        )

    snapshot = agent._get_candidate_tracking_snapshot()
    assert snapshot["rejected_candidates"] == ["picture"]
    assert "active_candidate" not in snapshot


def test_shortlist_pivots_to_new_grounded_candidate_after_rejection(tmp_path):
    agent, _ = _build_agent(tmp_path, env_type="scienceworld")
    agent.task = (
        "Your task is to find a(n) non-living thing. First, focus on the thing. "
        "Then, move it to the red box in the kitchen."
    )
    agent.task_status = "INCOMPLETE"
    agent._reset_episode_reasoning_state()
    agent._rejected_candidates = ["picture"]
    agent.percept = {
        "resulting_observation": (
            "This room is called the kitchen. In it, you see a picture, a painting, "
            "a lighter, and a red box."
        )
    }

    summary = agent._summarize_admissible_actions(
        [
            "focus on picture",
            "look at picture",
            "focus on red box",
            "look at painting",
            "focus on painting",
            "look at lighter",
        ],
        shortlist_limit=3,
    )

    shortlist = summary["task_relevant_action_shortlist"]
    assert "look at painting" in shortlist
    assert "focus on painting" in shortlist
    assert "focus on red box" not in shortlist
    assert "focus on picture" not in shortlist


def test_candidate_tracking_preserves_room_and_aliases_across_referent_shift(
    tmp_path,
):
    agent, _ = _build_agent(tmp_path, env_type="scienceworld")
    agent.task = (
        "Your task is to find a(n) plant. First, focus on the thing. "
        "Then, move it to the red box in the kitchen."
    )
    agent.task_status = "INCOMPLETE"
    agent._reset_episode_reasoning_state()

    agent._update_candidate_tracking(
        executed_action="focus on adult pea plant",
        observation="You focus on the pea plant.",
        previous_observation="You move to the greenhouse.",
    )
    first_snapshot = agent._get_candidate_tracking_snapshot()
    assert first_snapshot["active_candidate"] == "adult pea plant"
    assert first_snapshot["active_candidate_room"] == "greenhouse"
    assert "pea plant" in first_snapshot["active_candidate_aliases"]

    agent._update_candidate_tracking(
        executed_action="look at living thing in flower pot 3",
        observation="a pea plant in the reproducing stage with a tall height.",
        previous_observation="You move to the greenhouse.",
    )
    agent._update_candidate_tracking(
        executed_action="move living thing in flower pot 3 to hallway",
        observation="You move the pea plant to the hallway.",
        previous_observation="a pea plant in the reproducing stage with a tall height.",
    )

    final_snapshot = agent._get_candidate_tracking_snapshot()
    assert final_snapshot["active_candidate"] == "adult pea plant"
    assert final_snapshot["active_candidate_room"] == "hallway"
    assert any(
        "flower pot 3" in alias for alias in final_snapshot["active_candidate_aliases"]
    )


def test_shortlist_reacquires_last_seen_candidate_room_before_support_room_drift(
    tmp_path,
):
    agent, _ = _build_agent(tmp_path, env_type="scienceworld")
    agent.task = (
        "Your task is to find a(n) plant. First, focus on the thing. "
        "Then, move it to the red box in the kitchen."
    )
    agent.task_status = "INCOMPLETE"
    agent._reset_episode_reasoning_state()
    agent._update_candidate_tracking(
        executed_action="focus on adult pea plant",
        observation="You focus on the pea plant.",
        previous_observation="You move to the greenhouse.",
    )
    inspect_entry = agent._get_hypothesis_entry("inspect")
    inspect_entry["tests"] = 1
    inspect_entry["evidence_attempts"] = 1
    inspect_entry["status"] = "uncertain"
    agent.percept = {
        "resulting_observation": (
            "This room is called the hallway. In it, you see the agent. "
            "You also see: A door to the greenhouse (that is open) "
            "A door to the kitchen (that is open)"
        )
    }

    summary = agent._summarize_admissible_actions(
        [
            "go to greenhouse",
            "go to kitchen",
            "look at kitchen",
            "open door to greenhouse",
        ],
        shortlist_limit=1,
    )

    assert summary["candidate_tracking"]["active_candidate_room"] == "greenhouse"
    assert summary["task_relevant_action_shortlist"] == ["go to greenhouse"]


def test_shortlist_prefers_alias_matched_candidate_relocation(tmp_path):
    agent, _ = _build_agent(tmp_path, env_type="scienceworld")
    agent.task = (
        "Your task is to find a(n) plant. First, focus on the thing. "
        "Then, move it to the red box in the kitchen."
    )
    agent.task_status = "INCOMPLETE"
    agent._reset_episode_reasoning_state()
    agent._active_candidate = "adult pea plant"
    state = agent._get_candidate_state("adult pea plant")
    state["focused"] = True
    state["support_confirmed"] = True
    state["last_seen_room"] = "hallway"
    state["aliases"] = ["pea plant", "living thing flower pot 3"]
    inspect_entry = agent._get_hypothesis_entry("inspect")
    inspect_entry["tests"] = 1
    inspect_entry["evidence_attempts"] = 1
    inspect_entry["status"] = "uncertain"
    agent.percept = {
        "resulting_observation": (
            "This room is called the hallway. In it, you see a living thing in "
            "flower pot 3 and a picture."
        )
    }

    summary = agent._summarize_admissible_actions(
        [
            "move living thing to kitchen",
            "go to kitchen",
            "look at picture",
        ],
        shortlist_limit=1,
    )

    assert summary["task_relevant_action_shortlist"] == ["move living thing to kitchen"]


def test_custom_speaker_selection_repairs_malformed_thinking_output(tmp_path):
    agent, _ = _build_agent(tmp_path, env_type="scienceworld")
    agent.thinking_agent = object()
    agent.action_agent = object()
    agent.echo_agent = object()
    agent.learning_agent = object()
    agent.belief_state_agent = object()
    agent.task_success = False
    agent.task_failed = False
    agent.task = (
        "Your task is to find a(n) non-living thing. First, focus on the thing. "
        "Then, move it to the red box in the kitchen."
    )
    agent.allowed_transitions = {agent.thinking_agent: [agent.action_agent]}
    agent._reset_episode_reasoning_state()
    agent._active_candidate = "picture"

    groupchat = SimpleNamespace(messages=[{"content": "[Observation]: stale replay"}])

    next_speaker = agent.custom_speaker_selection(agent.thinking_agent, groupchat)

    assert next_speaker is agent.action_agent
    assert groupchat.messages[-1]["content"].startswith("STRATEGY:")
    assert "support entities" in groupchat.messages[-1]["content"]


def test_state_change_task_contract_separates_substance_from_transformation(tmp_path):
    agent, _ = _build_agent(tmp_path, env_type="scienceworld")
    agent.task = (
        "Your task is to melt water. First, focus on the substance. "
        "Then, take actions that will cause it to change its state of matter "
        "without boiling or combusting it. Also, keep the substance intact."
    )

    contract = agent._get_task_contract()

    assert contract["state_change_task"] is True
    assert contract["target_substances"] == ["water"]
    assert contract["desired_transformation"] == "melt"
    assert contract["transformation_direction"] == "warm"
    assert contract["procedural_sequence"] is True
    assert contract["ordering_cues"] == []
    assert "water" in contract["primary_targets"]
    assert "water" in contract["target_entities"]
    assert "melt" not in contract["target_entities"]
    assert "actions" not in contract["target_entities"]
    assert "will" not in contract["target_entities"]
    assert "without" not in contract["target_entities"]
    assert "boiling" not in contract["target_entities"]
    assert "combusting" not in contract["target_entities"]
    assert "also" not in contract["target_entities"]


def test_generic_state_change_task_contract_parses_state_of_matter_goal(tmp_path):
    agent, _ = _build_agent(tmp_path, env_type="scienceworld")
    agent.task = (
        "Your task is to change the state of matter of water. First, focus on the "
        "substance. Then, take actions that will cause it to change its state of matter."
    )

    contract = agent._get_task_contract()

    assert contract["state_change_task"] is True
    assert contract["target_substances"] == ["water"]
    assert contract["desired_transformation"] == "change state of matter"
    assert contract["transformation_direction"] == ""
    assert "water" in contract["primary_targets"]
    assert "water" in contract["target_entities"]


def test_state_change_phase_moves_from_search_to_focus_to_transformation(tmp_path):
    agent, _ = _build_agent(tmp_path, env_type="scienceworld")
    agent.task = (
        "Your task is to melt water. First, focus on the substance. "
        "Then, take actions that will cause it to change its state of matter."
    )
    agent._reset_episode_reasoning_state()

    agent.percept = {
        "resulting_observation": (
            "This room is called the hallway. In it, you see a picture and doors."
        )
    }
    assert agent._get_current_phase() == "locate_substance"

    agent.percept = {
        "resulting_observation": (
            "This room is called the kitchen. In it, you see water in a kettle and a stove."
        )
    }
    assert agent._get_current_phase() == "confirm_referent"

    agent._completed_focus_targets = ["water"]
    assert agent._get_current_phase() == "test_transformation"


def test_generic_state_change_task_enters_locate_substance_phase(tmp_path):
    agent, _ = _build_agent(tmp_path, env_type="scienceworld")
    agent.task = (
        "Your task is to change the state of matter of water. First, focus on the "
        "substance. Then, take actions that will cause it to change its state of matter."
    )
    agent._reset_episode_reasoning_state()
    agent.percept = {
        "resulting_observation": (
            "This room is called the hallway. In it, you see the agent, a picture, "
            "and doors to the art studio and the kitchen."
        )
    }

    summary = agent._summarize_admissible_actions(
        [
            "open art studio door",
            "open door to kitchen",
            "look around",
            "focus on agent",
            "mix agent",
        ],
        shortlist_limit=3,
    )

    assert summary["current_phase"] == "locate_substance"
    assert "open door to kitchen" in summary["task_relevant_action_shortlist"]
    assert "focus on agent" not in summary["task_relevant_action_shortlist"]
    assert "mix agent" not in summary["task_relevant_action_shortlist"]


def test_state_change_shortlist_suppresses_irrelevant_object_manipulation(tmp_path):
    agent, _ = _build_agent(tmp_path, env_type="scienceworld")
    agent.task = (
        "Your task is to melt water. First, focus on the substance. "
        "Then, take actions that will cause it to change its state of matter."
    )
    agent.task_status = "INCOMPLETE"
    agent._reset_episode_reasoning_state()
    agent.percept = {
        "resulting_observation": (
            "This room is called the hallway. In it, you see the agent, a picture, "
            "and doors to the art studio and the kitchen."
        )
    }

    summary = agent._summarize_admissible_actions(
        [
            "move picture to art studio",
            "focus on agent",
            "mix agent",
            "open art studio door",
            "open door to kitchen",
            "look around",
            "go to art studio",
        ],
        shortlist_limit=4,
    )

    shortlist = summary["task_relevant_action_shortlist"]
    assert "open door to kitchen" in shortlist
    assert "move picture to art studio" not in shortlist
    assert "mix agent" not in shortlist
    assert "focus on agent" not in shortlist


def test_state_change_shortlist_avoids_wrong_visible_substance(tmp_path):
    agent, _ = _build_agent(tmp_path, env_type="scienceworld")
    agent.task = (
        "Your task is to melt water. First, focus on the substance. "
        "Then, take actions that will cause it to change its state of matter."
    )
    agent.task_status = "INCOMPLETE"
    agent._reset_episode_reasoning_state()
    agent.percept = {
        "resulting_observation": (
            "This room is called the kitchen. In it, you see a fridge and a cupboard. "
            "In the fridge is a cup containing orange juice."
        )
    }

    summary = agent._summarize_admissible_actions(
        [
            "look at fridge",
            "look in fridge",
            "focus on cup containing orange juice",
            "use lighter on cup containing orange juice",
            "open cupboard",
            "look at cupboard",
        ],
        shortlist_limit=4,
    )

    shortlist = summary["task_relevant_action_shortlist"]
    assert "open cupboard" in shortlist
    assert "look at cupboard" in shortlist
    assert "use lighter on cup containing orange juice" not in shortlist
    assert "focus on cup containing orange juice" not in shortlist


def test_state_change_locate_substance_filters_explicit_non_target_substances(tmp_path):
    agent, _ = _build_agent(tmp_path, env_type="scienceworld")
    agent.task = (
        "Your task is to boil water. First, focus on the substance. "
        "Then, take actions that will cause it to change its state of matter."
    )
    agent.task_status = "INCOMPLETE"
    agent._reset_episode_reasoning_state()
    observation = (
        "This room is called the kitchen. In it, you see a substance called air, "
        "a fridge, a freezer, a sink, and a glass jar (containing a substance "
        "called sodium chloride)."
    )
    agent.percept = {"resulting_observation": observation}
    agent._update_state_change_search_tracking(action=None, observation=observation)

    summary = agent._summarize_admissible_actions(
        [
            "look at sodium chloride",
            "focus on sodium chloride",
            "mix sodium chloride",
            "move sodium chloride to sink",
            "open fridge",
            "open freezer",
            "look at sink",
            "activate sink",
        ],
        shortlist_limit=5,
    )

    shortlist = summary["task_relevant_action_shortlist"]
    assert summary["substance_search"]["grounded_substances"] == []
    assert "open fridge" in shortlist
    assert "open freezer" in shortlist
    assert "activate sink" in shortlist
    assert "focus on sodium chloride" not in shortlist
    assert "mix sodium chloride" not in shortlist


def test_state_change_shortlist_prefers_grounded_transformation_after_focus(tmp_path):
    agent, _ = _build_agent(tmp_path, env_type="scienceworld")
    agent.task = (
        "Your task is to melt water. First, focus on the substance. "
        "Then, take actions that will cause it to change its state of matter."
    )
    agent.task_status = "INCOMPLETE"
    agent._reset_episode_reasoning_state()
    agent._completed_focus_targets = ["water"]
    agent.percept = {
        "resulting_observation": (
            "This room is called the kitchen. In it, you see water in a kettle, "
            "a lighter, and a metal pot."
        )
    }

    summary = agent._summarize_admissible_actions(
        [
            "use lighter on water",
            "look at water",
            "pour water into metal pot",
            "go to hallway",
            "move picture to kitchen",
            "focus on picture",
        ],
        shortlist_limit=3,
    )

    shortlist = summary["task_relevant_action_shortlist"]
    assert "use lighter on water" in shortlist
    assert "look at water" in shortlist
    assert "move picture to kitchen" not in shortlist
    assert "focus on picture" not in shortlist


def test_state_change_canonicalizes_unsupported_substance_alias_to_grounded_container(
    tmp_path,
):
    agent, _ = _build_agent(tmp_path, env_type="scienceworld")
    agent.task = (
        "Your task is to melt water. First, focus on the substance. "
        "Then, take actions that will cause it to change its state of matter."
    )
    agent._reset_episode_reasoning_state()
    observation = (
        "This room is called the kitchen. In it, you see a glass jar "
        "(containing a substance called sodium chloride) and a cupboard."
    )
    agent.percept = {"resulting_observation": observation}
    agent._update_state_change_search_tracking(action=None, observation=observation)

    canonical = agent._canonicalize_suggested_action(
        "focus on jar containing salt water",
        [
            "focus on glass jar",
            "focus on cupboard",
            "look at glass jar",
        ],
    )

    assert canonical == "focus on glass jar"


def test_state_change_phase_switches_to_probe_sources_after_container_search(tmp_path):
    agent, _ = _build_agent(tmp_path, env_type="scienceworld")
    agent.task = (
        "Your task is to melt water. First, focus on the substance. "
        "Then, take actions that will cause it to change its state of matter."
    )
    agent._reset_episode_reasoning_state()
    agent.percept = {
        "resulting_observation": (
            "This room is called the kitchen. In it, you see a fridge, a cupboard, "
            "a sink, and a kettle."
        )
    }
    agent.admissible_actions = [
        "look in fridge",
        "look in cupboard",
        "look at sink",
        "activate sink",
        "fill kettle from sink",
    ]

    agent._update_state_change_search_tracking(
        action="look in fridge",
        observation=(
            "This room is called the kitchen. In it, you see a fridge, a cupboard, "
            "a sink, and a kettle. In the fridge is a cup containing orange juice."
        ),
    )
    agent.percept = {
        "resulting_observation": (
            "This room is called the kitchen. In it, you see a fridge, a cupboard, "
            "a sink, and a kettle. In the fridge is a cup containing orange juice."
        )
    }
    agent._update_state_change_search_tracking(
        action="look in cupboard",
        observation=(
            "This room is called the kitchen. In it, you see a fridge, a cupboard, "
            "a sink, and a kettle. The cupboard contains a bowl."
        ),
    )
    agent.percept = {
        "resulting_observation": (
            "This room is called the kitchen. In it, you see a fridge, a cupboard, "
            "a sink, and a kettle. The cupboard contains a bowl."
        )
    }

    snapshot = agent._get_substance_search_snapshot()

    assert agent._get_current_phase() == "probe_sources"
    assert snapshot["exhausted_containers"] == ["fridge", "cupboard"]
    assert "sink" in snapshot["source_candidates"]


def test_state_change_tracking_records_local_room_search_before_target_grounding(
    tmp_path,
):
    agent, _ = _build_agent(tmp_path, env_type="scienceworld")
    agent.task = (
        "Your task is to freeze water. First, focus on the substance. "
        "Then, take actions that will cause it to change its state of matter."
    )
    agent._reset_episode_reasoning_state()
    observation = (
        "This room is called the kitchen. In it, you see a sink, a cupboard, "
        "and a fridge. You also see: A door to the hallway (that is open), "
        "A door to the bathroom (that is closed), A door to the outside "
        "(that is closed)."
    )
    agent.percept = {"resulting_observation": observation}
    agent._update_state_change_search_tracking(action=None, observation=observation)

    agent.percept = {"resulting_observation": observation}
    agent._update_state_change_search_tracking(
        action="look in kitchen",
        observation=observation,
    )
    agent.percept = {"resulting_observation": observation}
    agent._update_state_change_search_tracking(
        action="open cupboard",
        observation=observation,
    )
    agent.percept = {"resulting_observation": observation}
    agent._update_state_change_search_tracking(
        action="open door to bathroom",
        observation=observation,
    )

    assert agent._search_location_states["kitchen"] == {
        "local_exploration": 2,
        "target_grounded": False,
    }


def test_state_change_shortlist_prefers_source_candidates_after_container_exhaustion(
    tmp_path,
):
    agent, _ = _build_agent(tmp_path, env_type="scienceworld")
    agent.task = (
        "Your task is to melt water. First, focus on the substance. "
        "Then, take actions that will cause it to change its state of matter."
    )
    agent.task_status = "INCOMPLETE"
    agent._reset_episode_reasoning_state()
    agent._exhausted_container_targets = ["fridge", "cupboard"]
    agent.percept = {
        "resulting_observation": (
            "This room is called the kitchen. In it, you see a sink, a kettle, "
            "a cupboard, and a fridge."
        )
    }
    agent.admissible_actions = [
        "look at sink",
        "activate sink",
        "fill kettle from sink",
        "look in cupboard",
        "open cupboard",
        "move bowl to kitchen",
        "focus on kitchen",
    ]

    summary = agent._summarize_admissible_actions(
        agent.admissible_actions,
        shortlist_limit=4,
    )

    shortlist = summary["task_relevant_action_shortlist"]
    assert summary["current_phase"] == "probe_sources"
    assert "look at sink" in shortlist
    assert "activate sink" in shortlist
    assert "move bowl to kitchen" not in shortlist
    assert "focus on kitchen" not in shortlist


def test_state_change_shortlist_reopens_room_frontier_after_local_probe_stall(
    tmp_path,
):
    agent, _ = _build_agent(tmp_path, env_type="scienceworld")
    agent.task = (
        "Your task is to freeze water. First, focus on the substance. "
        "Then, take actions that will cause it to change its state of matter."
    )
    agent.task_status = "INCOMPLETE"
    agent._reset_episode_reasoning_state()
    agent._search_location_states["kitchen"] = {
        "local_exploration": 3,
        "target_grounded": False,
    }
    observation = (
        "This room is called the kitchen. In it, you see a substance called air, "
        "a freezer, a fridge, and a sink. You also see: A door to the hallway "
        "(that is open), A door to the bathroom (that is closed), A door to the "
        "outside (that is closed)."
    )
    agent.percept = {"resulting_observation": observation}
    agent._update_state_change_search_tracking(action=None, observation=observation)
    agent.admissible_actions = [
        "focus on air",
        "open freezer",
        "open fridge",
        "look in sink",
        "activate sink",
        "go to hallway",
        "open door to bathroom",
        "look at door to bathroom",
    ]

    summary = agent._summarize_admissible_actions(
        agent.admissible_actions,
        shortlist_limit=4,
    )

    shortlist = summary["task_relevant_action_shortlist"]
    assert summary["current_phase"] == "locate_substance"
    assert "look at door to bathroom" in shortlist
    assert "open door to bathroom" in shortlist
    assert "focus on air" not in shortlist
    assert "open freezer" not in shortlist


def test_state_change_probe_shortlist_ignores_non_target_grounded_substances(tmp_path):
    agent, _ = _build_agent(tmp_path, env_type="scienceworld")
    agent.task = (
        "Your task is to boil water. First, focus on the substance. "
        "Then, take actions that will cause it to change its state of matter."
    )
    agent.task_status = "INCOMPLETE"
    agent._reset_episode_reasoning_state()
    agent._exhausted_container_targets = ["fridge", "cupboard"]
    observation = (
        "This room is called the kitchen. In it, you see a substance called air, "
        "a sink, a stove, and a glass jar (containing a substance called sodium chloride)."
    )
    agent.percept = {"resulting_observation": observation}
    agent._update_state_change_search_tracking(action=None, observation=observation)
    agent.admissible_actions = [
        "look at sodium chloride",
        "focus on sodium chloride",
        "move sodium chloride to sink",
        "mix sodium chloride",
        "look at sink",
        "look in sink",
        "activate sink",
        "look at stove",
        "activate stove",
    ]

    summary = agent._summarize_admissible_actions(
        agent.admissible_actions,
        shortlist_limit=5,
    )

    shortlist = summary["task_relevant_action_shortlist"]
    assert summary["current_phase"] == "probe_sources"
    assert summary["substance_search"]["grounded_substances"] == []
    assert "look at sink" in shortlist
    assert "activate sink" in shortlist
    assert "activate stove" in shortlist
    assert "focus on sodium chloride" not in shortlist
    assert "move sodium chloride to sink" not in shortlist


def test_state_change_shortlist_avoids_invalid_same_referent_probe_retries(tmp_path):
    agent, _ = _build_agent(tmp_path, env_type="scienceworld")
    agent.task = (
        "Your task is to melt water. First, focus on the substance. "
        "Then, take actions that will cause it to change its state of matter."
    )
    agent.task_status = "INCOMPLETE"
    agent._reset_episode_reasoning_state()
    agent._exhausted_container_targets = ["drawer", "cupboard"]

    agent.percept = {
        "resulting_observation": "No known action matches that input.",
        "newly_admissible_actions": [],
        "no_longer_admissible_actions": [],
    }
    agent._update_episode_hypothesis_ledger(
        suggested_action="look at tin cup",
        executed_action="look at tin cup",
        previous_observation="You move to the kitchen.",
    )

    agent.percept = {
        "resulting_observation": (
            "This room is called the kitchen. In it, you see a tin cup, a sink, "
            "a drain, a stove, and an oven."
        )
    }
    agent.admissible_actions = [
        "look at tin cup",
        "look in tin cup",
        "look at sink",
        "look in sink",
        "activate sink",
        "close drain",
        "look at stove",
        "activate oven",
    ]

    summary = agent._summarize_admissible_actions(
        agent.admissible_actions,
        shortlist_limit=5,
    )

    shortlist = summary["task_relevant_action_shortlist"]
    assert summary["current_phase"] == "probe_sources"
    assert "look at tin cup" not in shortlist
    assert "look in tin cup" not in shortlist
    assert "look at sink" in shortlist
    assert "close drain" in shortlist
    assert any("sink" in action for action in shortlist)


def test_measurement_task_contract_extracts_measurement_roles_and_cleans_tokens(
    tmp_path,
):
    agent, _ = _build_agent(tmp_path, env_type="scienceworld")
    agent.task = (
        "Your task is to measure the melting point of solid unknown substance C, "
        "which is located around the kitchen. First, focus on the thermometer. "
        "Next, focus on the solid unknown substance C. If the melting point of "
        "solid unknown substance C is above 150.0 degrees celsius, focus on the "
        "orange box. If the melting point of solid unknown substance C is below "
        "150.0 degrees celsius, focus on the yellow box."
    )

    contract = agent._get_task_contract()

    assert contract["measurement_task"] is True
    assert contract["measurement_property"] == "melting point"
    assert contract["measurement_property_type"] == "stable_threshold_property"
    assert contract["measurement_target"] == ["solid unknown substance c"]
    assert contract["measurement_instrument"] == ["thermometer"]
    assert contract["measurement_branch_targets"] == ["orange box", "yellow box"]
    assert contract["primary_targets"] == ["solid unknown substance c"]
    assert contract["supporting_targets"] == ["thermometer"]
    assert "which" not in contract["target_entities"]
    assert "next" not in contract["target_entities"]
    assert "above" not in contract["target_entities"]
    assert "orange" not in contract["target_entities"]


def test_measurement_task_does_not_emit_substance_search_snapshot(tmp_path):
    agent, _ = _build_agent(tmp_path, env_type="scienceworld")
    agent.task = (
        "Your task is to measure the melting point of solid unknown substance C. "
        "First, focus on the thermometer. Next, focus on the solid unknown substance C."
    )
    observation = (
        "This room is called the kitchen. In it, you see a glass jar "
        "(containing a substance called sodium chloride), a thermometer, "
        "and solid unknown substance C."
    )
    agent._reset_episode_reasoning_state()
    agent.percept = {"resulting_observation": observation}

    agent._update_state_change_search_tracking(action=None, observation=observation)

    assert agent._grounded_substances == {}
    assert agent._get_substance_search_snapshot() == {}


def test_measurement_proxy_reading_does_not_resolve_branch_target(tmp_path):
    agent, _ = _build_agent(tmp_path, env_type="scienceworld")
    agent.task = (
        "Your task is to measure the melting point of solid unknown substance C. "
        "First, focus on the thermometer. Next, focus on the solid unknown substance C. "
        "If the melting point of solid unknown substance C is above 150.0 degrees celsius, "
        "focus on the orange box. If the melting point of solid unknown substance C is below "
        "150.0 degrees celsius, focus on the yellow box."
    )
    agent._reset_episode_reasoning_state()
    agent._containment_by_object["solid unknown substance c"] = "orange box"

    agent._update_measurement_tracking(
        action="use thermometer on orange box",
        observation="the thermometer measures a temperature of 153 degrees celsius",
    )

    snapshot = agent._get_measurement_tracking_snapshot()

    assert agent._selected_measurement_branch_target is None
    assert "latest_proxy_measurement" in snapshot
    assert "latest_direct_measurement" not in snapshot
    assert snapshot["branch_ready"] is False


def test_measurement_shortlist_gates_branch_targets_until_branch_is_resolved(tmp_path):
    agent, _ = _build_agent(tmp_path, env_type="scienceworld")
    agent.task = (
        "Your task is to measure the melting point of solid unknown substance C, "
        "which is located around the kitchen. First, focus on the thermometer. "
        "Next, focus on the solid unknown substance C. If the melting point of "
        "solid unknown substance C is above 150.0 degrees celsius, focus on the "
        "orange box. If the melting point of solid unknown substance C is below "
        "150.0 degrees celsius, focus on the yellow box."
    )
    agent.task_status = "INCOMPLETE"
    agent._reset_episode_reasoning_state()
    agent.percept = {
        "resulting_observation": (
            "This room is called the kitchen. In it, you see a thermometer, "
            "solid unknown substance C, an orange box, and a yellow box."
        )
    }
    agent._update_ordered_target_progress(
        executed_action="focus on thermometer",
        observation="You focus on the thermometer.",
    )
    agent._update_ordered_target_progress(
        executed_action="focus on solid unknown substance c",
        observation="You focus on the solid unknown substance C.",
    )
    agent._update_measurement_tracking(
        action="use thermometer on solid unknown substance c",
        observation="the thermometer measures a temperature of 10 degrees celsius",
    )

    summary = agent._summarize_admissible_actions(
        [
            "focus on orange box",
            "focus on yellow box",
            "use thermometer on solid unknown substance c",
            "look at solid unknown substance c",
        ],
        shortlist_limit=2,
    )

    shortlist = summary["task_relevant_action_shortlist"]
    assert summary["current_phase"] == "induce_property_change"
    assert summary["measurement_tracking"]["property_resolution_status"] == (
        "needs_transition_evidence"
    )
    assert "use thermometer on solid unknown substance c" in shortlist
    assert "focus on orange box" not in shortlist
    assert "focus on yellow box" not in shortlist


def test_measurement_branch_destination_is_blocked_until_property_is_resolved(
    tmp_path,
):
    agent, _ = _build_agent(tmp_path, env_type="scienceworld")
    agent.task = (
        "Your task is to measure the melting point of solid unknown substance C, "
        "which is located around the kitchen. First, focus on the thermometer. "
        "Next, focus on the solid unknown substance C. If the melting point of "
        "solid unknown substance C is above 150.0 degrees celsius, focus on the "
        "orange box. If the melting point of solid unknown substance C is below "
        "150.0 degrees celsius, focus on the yellow box."
    )
    agent.task_status = "INCOMPLETE"
    agent._reset_episode_reasoning_state()
    agent.percept = {
        "resulting_observation": (
            "This room is called the kitchen. In it, you see a thermometer, "
            "solid unknown substance C, an orange box, a yellow box, and a lighter."
        )
    }
    agent._update_ordered_target_progress(
        executed_action="focus on thermometer",
        observation="You focus on the thermometer.",
    )
    agent._update_ordered_target_progress(
        executed_action="focus on solid unknown substance c",
        observation="You focus on the solid unknown substance C.",
    )
    agent._update_measurement_tracking(
        action="use thermometer on solid unknown substance c",
        observation="the thermometer measures a temperature of 8 degrees celsius",
    )

    summary = agent._summarize_admissible_actions(
        [
            "move solid unknown substance c to orange box",
            "use lighter on solid unknown substance c",
            "use thermometer on solid unknown substance c",
            "look at solid unknown substance c",
        ],
        shortlist_limit=2,
    )

    shortlist = summary["task_relevant_action_shortlist"]
    assert summary["current_phase"] == "induce_property_change"
    assert "move solid unknown substance c to orange box" not in shortlist
    assert "use lighter on solid unknown substance c" in shortlist


def test_measurement_locate_instrument_shortlist_prefers_local_search_actions(
    tmp_path,
):
    agent, _ = _build_agent(tmp_path, env_type="scienceworld")
    agent.task = (
        "Your task is to measure the temperature of unknown substance B, "
        "which is located around the living room. First, focus on the "
        "thermometer. Next, focus on the unknown substance B. If the unknown "
        "substance B temperature is above 100.0 degrees celsius, place it in "
        "the red box. If the unknown substance B temperature is below 100.0 "
        "degrees celsius, place it in the green box. The boxes are located "
        "around the bathroom."
    )
    agent.task_status = "INCOMPLETE"
    agent._reset_episode_reasoning_state()
    agent.percept = {
        "resulting_observation": (
            "This room is called the living room. In it, you see the agent, "
            "a substance called air, a chair. On the chair is: nothing. "
            "a couch. On the couch is: a white pillow. a desk. On the desk "
            "is: a drawer. a painting. unknown substance B. You also see: "
            "A door to the hallway (that is open)"
        )
    }

    summary = agent._summarize_admissible_actions(
        [
            "focus on unknown substance",
            "move unknown substance to living room",
            "look at unknown substance",
            "look at living room",
            "look in living room",
            "open drawer",
        ],
        shortlist_limit=3,
    )

    shortlist = summary["task_relevant_action_shortlist"]
    assert summary["current_phase"] == "locate_instrument"
    assert "open drawer" in shortlist
    assert "focus on unknown substance" not in shortlist
    assert "move unknown substance to living room" not in shortlist


def test_measurement_search_frontier_reappears_after_local_instrument_search(
    tmp_path,
):
    agent, _ = _build_agent(tmp_path, env_type="scienceworld")
    agent.task = (
        "Your task is to measure the temperature of unknown substance B, "
        "which is located around the living room. First, focus on the "
        "thermometer. Next, focus on the unknown substance B. If the unknown "
        "substance B temperature is above 100.0 degrees celsius, place it in "
        "the red box. If the unknown substance B temperature is below 100.0 "
        "degrees celsius, place it in the green box. The boxes are located "
        "around the bathroom."
    )
    agent.task_status = "INCOMPLETE"
    agent._reset_episode_reasoning_state()
    art_studio_observation = (
        "This room is called the art studio. In it, you see the agent, a "
        "substance called air, a large cupboard. The large cupboard door is "
        "open. a table. On the table is: a jug (containing nothing). a wood "
        "cup (containing red paint). a wood cup (containing yellow paint). "
        "a wood cup (containing blue paint). You also see: A door to the "
        "hallway (that is open)"
    )
    agent.percept = {"resulting_observation": art_studio_observation}
    agent._update_measurement_search_tracking(
        action="look at art studio",
        observation=art_studio_observation,
    )
    agent.curr_episodic_memory.append(
        json.dumps(
            {
                "timestep": 10,
                "attempted_action": "look at art studio",
                "resulting_observation": art_studio_observation,
            }
        )
    )
    agent.percept = {"resulting_observation": "The large cupboard door is now open."}
    agent._update_measurement_search_tracking(
        action="open cupboard",
        observation="The large cupboard door is now open.",
    )
    agent.percept = {"resulting_observation": art_studio_observation}

    summary = agent._summarize_admissible_actions(
        [
            "focus on cup containing red paint in art studio",
            "look at cup containing red paint in art studio",
            "look in cup containing red paint in art studio",
            "look at art studio",
            "open cupboard",
            "go to hallway",
        ],
        shortlist_limit=3,
    )

    shortlist = summary["task_relevant_action_shortlist"]
    assert agent._search_location_states["art studio"]["local_exploration"] == 2
    assert summary["current_phase"] == "locate_instrument"
    assert "open cupboard" in shortlist
    assert "go to hallway" in shortlist
    assert "focus on cup containing red paint in art studio" not in shortlist


def test_measurement_transition_event_requires_confirming_measurement(tmp_path):
    agent, _ = _build_agent(tmp_path, env_type="scienceworld")
    agent.task = (
        "Your task is to measure the melting point of solid unknown substance C. "
        "First, focus on the thermometer. Next, focus on the solid unknown substance C. "
        "If the melting point of solid unknown substance C is above 150.0 degrees celsius, "
        "focus on the orange box. If the melting point of solid unknown substance C is below "
        "150.0 degrees celsius, focus on the yellow box."
    )
    agent.task_status = "INCOMPLETE"
    agent._reset_episode_reasoning_state()
    agent.percept = {
        "resulting_observation": (
            "This room is called the kitchen. In it, you see a thermometer, "
            "solid unknown substance C, an orange box, a yellow box, and a lighter."
        )
    }
    agent._update_ordered_target_progress(
        executed_action="focus on thermometer",
        observation="You focus on the thermometer.",
    )
    agent._update_ordered_target_progress(
        executed_action="focus on solid unknown substance c",
        observation="You focus on the solid unknown substance C.",
    )
    agent._update_measurement_tracking(
        action="use thermometer on solid unknown substance c",
        observation="the thermometer measures a temperature of 8 degrees celsius",
    )
    agent._update_measurement_tracking(
        action="use lighter on solid unknown substance c",
        observation="The solid unknown substance C melts into a liquid.",
    )

    snapshot = agent._get_measurement_tracking_snapshot()

    assert snapshot["transition_observed"] is True
    assert snapshot["property_resolution_status"] == "needs_confirming_measurement"
    assert agent._selected_measurement_branch_target is None
    assert agent._get_current_phase() == "verify_transition"


def test_measurement_event_confirmed_direct_reading_resolves_branch_target(tmp_path):
    agent, _ = _build_agent(tmp_path, env_type="scienceworld")
    agent.task = (
        "Your task is to measure the melting point of solid unknown substance C. "
        "First, focus on the thermometer. Next, focus on the solid unknown substance C. "
        "If the melting point of solid unknown substance C is above 150.0 degrees celsius, "
        "focus on the orange box. If the melting point of solid unknown substance C is below "
        "150.0 degrees celsius, focus on the yellow box."
    )
    agent.task_status = "INCOMPLETE"
    agent._reset_episode_reasoning_state()
    agent.percept = {
        "resulting_observation": (
            "This room is called the kitchen. In it, you see a thermometer, "
            "solid unknown substance C, an orange box, and a yellow box."
        )
    }
    agent._update_ordered_target_progress(
        executed_action="focus on thermometer",
        observation="You focus on the thermometer.",
    )
    agent._update_ordered_target_progress(
        executed_action="focus on solid unknown substance c",
        observation="You focus on the solid unknown substance C.",
    )
    agent._update_measurement_tracking(
        action="use thermometer on solid unknown substance c",
        observation=(
            "The solid unknown substance C melts and the thermometer measures "
            "a temperature of 151 degrees celsius"
        ),
    )

    snapshot = agent._get_measurement_tracking_snapshot()

    assert snapshot["property_resolution_status"] == "resolved"
    assert snapshot["branch_ready"] is True
    assert snapshot["branch_target"] == ["orange box"]
    assert agent._get_current_phase() == "execute_branch"


def test_conditional_branch_task_contract_preserves_evidence_target_and_branch_targets(
    tmp_path,
):
    agent, _ = _build_agent(tmp_path, env_type="scienceworld")
    agent.task = (
        "Your task is to determine whether blue seed color is a dominant or "
        "recessive trait in the unknown E plant. If the trait is dominant, "
        "focus on the red box. If the trait is recessive, focus on the green box."
    )

    contract = agent._get_task_contract()

    assert contract["conditional_branch_task"] is True
    assert contract["measurement_task"] is False
    assert contract["conditional_branch_evidence_target"] == ["unknown plant"]
    assert contract["conditional_branch_subject"] == ["blue seed"]
    assert contract["conditional_branch_targets"] == ["red box", "green box"]
    assert contract["primary_targets"] == ["unknown plant", "blue seed"]
    assert contract["conditional_branches"][0]["condition_tokens"] == ["dominant"]
    assert contract["conditional_branches"][1]["condition_tokens"] == ["recessive"]


def test_conditional_branch_search_prefers_evidence_location_over_branch_boxes(
    tmp_path,
):
    agent, _ = _build_agent(tmp_path, env_type="scienceworld")
    agent.task = (
        "Your task is to determine whether blue seed color is a dominant or "
        "recessive trait in the unknown E plant. If the trait is dominant, "
        "focus on the red box. If the trait is recessive, focus on the green box."
    )
    agent.task_status = "INCOMPLETE"
    agent._reset_episode_reasoning_state()
    agent.percept = {
        "resulting_observation": (
            "This room is called the hallway. In it, you see a picture. "
            "You also see a door to the greenhouse and a door to the workshop."
        )
    }

    summary = agent._summarize_admissible_actions(
        [
            "focus on red box",
            "focus on green box",
            "look at picture",
            "open door to greenhouse",
            "go to greenhouse",
        ],
        shortlist_limit=3,
    )

    shortlist = summary["task_relevant_action_shortlist"]
    assert summary["current_phase"] == "locate_primary_target"
    assert "open door to greenhouse" in shortlist
    assert "go to greenhouse" in shortlist
    assert "focus on red box" not in shortlist
    assert "focus on green box" not in shortlist


def test_conditional_branch_shortlist_gathers_evidence_before_branch_actions(tmp_path):
    agent, _ = _build_agent(tmp_path, env_type="scienceworld")
    agent.task = (
        "Your task is to determine whether blue seed color is a dominant or "
        "recessive trait in the unknown E plant. If the trait is dominant, "
        "focus on the red box. If the trait is recessive, focus on the green box."
    )
    agent.task_status = "INCOMPLETE"
    agent._reset_episode_reasoning_state()
    agent.percept = {
        "resulting_observation": (
            "This room is called the greenhouse. In it, you see a seed square blue "
            "unknown E seed, a round brown unknown E seed, a red box, a green box, "
            "a shovel, and a drain."
        )
    }

    summary = agent._summarize_admissible_actions(
        [
            "look at seed square blue unknown e seed",
            "look at round brown unknown e seed",
            "look in seed square blue unknown e seed",
            "look around",
            "look at shovel",
            "focus on red box",
            "connect red box to seed square blue unknown e seed",
            "pour seed square blue unknown e seed into red box",
        ],
        shortlist_limit=4,
    )

    shortlist = summary["task_relevant_action_shortlist"]
    assert summary["current_phase"] == "gather_branch_evidence"
    assert "look at seed square blue unknown e seed" in shortlist
    assert "look at round brown unknown e seed" in shortlist
    assert "connect red box to seed square blue unknown e seed" not in shortlist
    assert "pour seed square blue unknown e seed into red box" not in shortlist
    assert "focus on red box" not in shortlist


def test_conditional_branch_resolution_promotes_selected_branch_target(tmp_path):
    agent, _ = _build_agent(tmp_path, env_type="scienceworld")
    agent.task = (
        "Your task is to determine whether blue seed color is a dominant or "
        "recessive trait in the unknown E plant. If the trait is dominant, "
        "focus on the red box. If the trait is recessive, focus on the green box."
    )
    agent.task_status = "INCOMPLETE"
    agent._reset_episode_reasoning_state()
    agent.percept = {
        "resulting_observation": (
            "This room is called the greenhouse. In it, you see a seed square blue "
            "unknown E seed, a red box, and a green box."
        )
    }

    agent._update_conditional_branch_tracking(
        action="look at genetics chart",
        observation=(
            "The genetics chart states that blue seed color is dominant in the "
            "unknown E plant."
        ),
    )

    summary = agent._summarize_admissible_actions(
        [
            "focus on red box",
            "focus on green box",
            "look at seed square blue unknown e seed",
        ],
        shortlist_limit=1,
    )

    snapshot = agent._get_conditional_branch_tracking_snapshot()

    assert snapshot["branch_ready"] is True
    assert snapshot["branch_target"] == ["red box"]
    assert summary["current_phase"] == "execute_branch"
    assert summary["task_relevant_action_shortlist"] == ["focus on red box"]


def test_admissible_action_summary_is_cached_until_invalidated(tmp_path, monkeypatch):
    agent, _ = _build_agent(tmp_path, env_type="scienceworld")
    agent.task = (
        "Your task is to freeze water. First, focus on the substance. "
        "Then, take actions that will cause it to change its state of matter."
    )
    agent.task_status = "INCOMPLETE"
    agent._reset_episode_reasoning_state()
    agent.percept = {
        "resulting_observation": (
            "This room is called the kitchen. In it, you see a sink, a freezer, "
            "and a fridge."
        )
    }
    actions = ["look at sink", "open freezer", "go to hallway"]

    original = agent._summarize_admissible_actions_uncached
    call_count = {"value": 0}

    def wrapped(admissible_actions, *, shortlist_limit=12):
        call_count["value"] += 1
        return original(admissible_actions, shortlist_limit=shortlist_limit)

    monkeypatch.setattr(agent, "_summarize_admissible_actions_uncached", wrapped)

    first = agent._summarize_admissible_actions(actions, shortlist_limit=20)
    second = agent._summarize_admissible_actions(actions, shortlist_limit=20)

    assert first is second
    assert call_count["value"] == 1

    agent._invalidate_action_summary_cache()
    third = agent._summarize_admissible_actions(actions, shortlist_limit=20)

    assert third is not first
    assert call_count["value"] == 2


def test_measurement_shortlist_prefers_active_enclosure_for_hidden_target(tmp_path):
    agent, _ = _build_agent(tmp_path, env_type="scienceworld")
    agent.task = (
        "Your task is to measure the melting point of solid unknown substance C, "
        "which is located around the kitchen. First, focus on the thermometer. "
        "Next, focus on the solid unknown substance C. If the melting point of "
        "solid unknown substance C is above 150.0 degrees celsius, focus on the "
        "orange box. If the melting point of solid unknown substance C is below "
        "150.0 degrees celsius, focus on the yellow box."
    )
    agent.task_status = "INCOMPLETE"
    agent._reset_episode_reasoning_state()
    agent.percept = {
        "resulting_observation": (
            "This room is called the kitchen. In it, you see the agent, a oven, "
            "a thermometer, and a yellow box."
        )
    }
    agent._update_ordered_target_progress(
        executed_action="focus on thermometer",
        observation="You focus on the thermometer.",
    )
    agent._update_ordered_target_progress(
        executed_action="focus on solid unknown substance c",
        observation="You focus on the solid unknown substance C.",
    )
    agent._update_measurement_tracking(
        action="move solid unknown substance c to orange box",
        observation="You move the solid unknown substance C to the orange box.",
    )
    agent._update_measurement_tracking(
        action="move orange box to oven",
        observation="You move the orange box to the oven.",
    )
    agent._update_measurement_tracking(
        action="use thermometer on solid unknown substance c",
        observation="the thermometer measures a temperature of 10 degrees celsius",
    )

    summary = agent._summarize_admissible_actions(
        [
            "open oven",
            "look at oven",
            "use thermometer on solid unknown substance c",
            "look at solid unknown substance c",
        ],
        shortlist_limit=2,
    )

    shortlist = summary["task_relevant_action_shortlist"]
    assert "open oven" in shortlist
    assert "look at oven" in shortlist
    assert "use thermometer on solid unknown substance c" not in shortlist
    assert "look at solid unknown substance c" not in shortlist


def test_invalid_measurement_exact_action_is_retired_from_shortlist(tmp_path):
    agent, _ = _build_agent(tmp_path, env_type="scienceworld")
    agent.task = (
        "Your task is to measure the melting point of solid unknown substance C. "
        "First, focus on the thermometer. Next, focus on the solid unknown substance C."
    )
    agent.task_status = "INCOMPLETE"
    agent._reset_episode_reasoning_state()
    agent.percept = {
        "resulting_observation": (
            "This room is called the kitchen. In it, you see a thermometer, "
            "solid unknown substance C, and a oven."
        )
    }
    agent._update_ordered_target_progress(
        executed_action="focus on thermometer",
        observation="You focus on the thermometer.",
    )
    agent._update_ordered_target_progress(
        executed_action="focus on solid unknown substance c",
        observation="You focus on the solid unknown substance C.",
    )
    agent._record_invalid_exact_action("use thermometer on orange box")

    summary = agent._summarize_admissible_actions(
        [
            "use thermometer on orange box",
            "use thermometer on solid unknown substance c",
            "look at oven",
        ],
        shortlist_limit=2,
    )

    shortlist = summary["task_relevant_action_shortlist"]
    assert "use thermometer on solid unknown substance c" in shortlist
    assert "use thermometer on orange box" not in shortlist


def test_artifact_creation_task_contract_tracks_artifact_roles_and_cleans_tokens(
    tmp_path,
):
    agent, _ = _build_agent(tmp_path, env_type="scienceworld")
    agent.task = (
        "Use chemistry to create green-blue paint. When part-way done, focus on the "
        "intermediate (secondary color) paint. When completely done, focus on the "
        "green paint."
    )

    contract = agent._get_task_contract()

    assert contract["artifact_creation_task"] is True
    assert contract["artifact_type"] == ["paint"]
    assert contract["artifact_intermediate_targets"] == [
        "intermediate secondary color paint"
    ]
    assert contract["artifact_final_targets"] == ["green paint"]
    assert "paint" in contract["target_entities"]
    assert "green" in contract["target_entities"]
    assert "when" not in contract["target_entities"]
    assert "part" not in contract["target_entities"]
    assert "way" not in contract["target_entities"]
    assert "done" not in contract["target_entities"]


def test_artifact_creation_shortlist_prefers_missing_ingredient_search_to_empty_transfer(
    tmp_path,
):
    agent, _ = _build_agent(tmp_path, env_type="scienceworld")
    agent.task = (
        "Use chemistry to create green-blue paint. When part-way done, focus on the "
        "intermediate (secondary color) paint. When completely done, focus on the "
        "green paint."
    )
    agent.task_status = "INCOMPLETE"
    agent._reset_episode_reasoning_state()
    observation = (
        "This room is called the art studio. In it, you see a cup containing blue "
        "paint, an empty wood cup, a drawer, and the door to the workshop."
    )
    agent.percept = {"resulting_observation": observation}
    agent._update_artifact_creation_tracking(observation)

    summary = agent._summarize_admissible_actions(
        [
            "look in drawer",
            "open workshop door",
            "look around",
            "pour blue paint into wood cup",
            "move blue paint to wood cup",
            "focus on blue paint",
        ],
        shortlist_limit=4,
    )

    shortlist = summary["task_relevant_action_shortlist"]
    assert summary["current_phase"] == "find_missing_ingredient_or_reagent"
    assert "look in drawer" in shortlist
    assert "open workshop door" in shortlist
    assert "pour blue paint into wood cup" not in shortlist
    assert "move blue paint to wood cup" not in shortlist


def test_artifact_creation_tracking_records_local_room_search_before_grounding(
    tmp_path,
):
    agent, _ = _build_agent(tmp_path, env_type="scienceworld")
    agent.task = (
        "Your task is to use chemistry to create the substance 'salt water'. "
        "A recipe and some of the ingredients might be found near the kitchen. "
        "When you are done, focus on the salt water."
    )
    agent._reset_episode_reasoning_state()
    observation = (
        "This room is called the kitchen. In it, you see a sink, a fridge, a "
        "freezer, a glass jar (containing a substance called sodium chloride), "
        "and a recipe titled instructions to make salt water. You also see: "
        "A door to the hallway (that is open), A door to the bathroom "
        "(that is closed), A door to the outside (that is closed)."
    )

    agent._update_artifact_creation_search_tracking(
        action=None,
        observation=observation,
    )
    agent._update_artifact_creation_search_tracking(
        action="look in kitchen",
        observation=observation,
    )
    agent._update_artifact_creation_search_tracking(
        action="open fridge",
        observation=observation,
    )
    agent._update_artifact_creation_search_tracking(
        action="open door to bathroom",
        observation=observation,
    )

    assert agent._search_location_states["kitchen"] == {
        "local_exploration": 2,
        "target_grounded": False,
    }


def test_artifact_creation_shortlist_reopens_room_frontier_after_local_search_stall(
    tmp_path,
):
    agent, _ = _build_agent(tmp_path, env_type="scienceworld")
    agent.task = (
        "Your task is to use chemistry to create the substance 'salt water'. "
        "A recipe and some of the ingredients might be found near the kitchen. "
        "When you are done, focus on the salt water."
    )
    agent.task_status = "INCOMPLETE"
    agent._reset_episode_reasoning_state()
    agent._search_location_states["kitchen"] = {
        "local_exploration": 3,
        "target_grounded": False,
    }
    observation = (
        "This room is called the kitchen. In it, you see a sink, a fridge, a "
        "freezer, a glass cup (containing nothing), a glass jar (containing a "
        "substance called sodium chloride), and a recipe titled instructions to "
        "make salt water. You also see: A door to the hallway (that is open), "
        "A door to the bathroom (that is closed), A door to the outside "
        "(that is closed)."
    )
    agent.percept = {"resulting_observation": observation}
    agent._update_artifact_creation_search_tracking(
        action=None,
        observation=observation,
    )
    agent.admissible_actions = [
        "read instructions to make salt water",
        "look in sink",
        "open fridge",
        "open freezer",
        "look in cup containing nothing",
        "open door to bathroom",
        "look at door to bathroom",
        "go to hallway",
    ]

    summary = agent._summarize_admissible_actions(
        agent.admissible_actions,
        shortlist_limit=4,
    )

    shortlist = summary["task_relevant_action_shortlist"]
    assert summary["current_phase"] == "locate_base_artifact"
    assert "open door to bathroom" in shortlist
    assert "open fridge" not in shortlist
    assert shortlist.index("open door to bathroom") < shortlist.index("look in sink")


def test_artifact_creation_shortlist_downranks_wrong_type_color_distractors(tmp_path):
    agent, _ = _build_agent(tmp_path, env_type="scienceworld")
    agent.task = (
        "Use chemistry to create green-blue paint. When part-way done, focus on the "
        "intermediate (secondary color) paint. When completely done, focus on the "
        "green paint."
    )
    agent.task_status = "INCOMPLETE"
    agent._reset_episode_reasoning_state()
    observation = (
        "This room is called the workshop. In it, you see a green light bulb, "
        "a green paint can, and a freezer."
    )
    agent.percept = {"resulting_observation": observation}
    agent._update_artifact_creation_tracking(observation)

    summary = agent._summarize_admissible_actions(
        [
            "look at green light bulb",
            "focus on green light bulb",
            "look at green paint can",
            "focus on green paint can",
        ],
        shortlist_limit=2,
    )

    shortlist = summary["task_relevant_action_shortlist"]
    assert "look at green paint can" in shortlist
    assert "focus on green paint can" in shortlist
    assert "look at green light bulb" not in shortlist
    assert "focus on green light bulb" not in shortlist


def test_canonicalize_room_transition_prefers_opening_required_door(tmp_path):
    agent, _ = _build_agent(tmp_path, env_type="scienceworld")

    canonical = agent._canonicalize_suggested_action(
        "go to workshop",
        [
            "open workshop door",
            "look at workshop",
            "go to hallway",
        ],
    )

    assert canonical == "open workshop door"


def test_relation_task_phase_locates_primary_target_before_generic_act(tmp_path):
    agent, _ = _build_agent(tmp_path, env_type="scienceworld")
    agent.task = (
        "Turn on the red light bulb by powering it using a renewable power source. "
        "First, focus on the red light bulb. Then, create an electrical circuit that powers it on."
    )
    agent.task_status = "INCOMPLETE"
    agent._reset_episode_reasoning_state()
    agent.percept = {
        "resulting_observation": (
            "This room is called the hallway. Here you see the door to the art studio, "
            "the door to the bedroom, and the door to the workshop."
        )
    }

    assert agent._get_current_phase() == "locate_primary_target"


def test_relation_search_prefers_exit_from_local_dead_end(tmp_path):
    agent, _ = _build_agent(tmp_path, env_type="scienceworld")
    agent.task = (
        "Turn on the red light bulb by powering it using a renewable power source. "
        "First, focus on the red light bulb. Then, create an electrical circuit that powers it on."
    )
    agent.task_status = "INCOMPLETE"
    agent._reset_episode_reasoning_state()
    observation = (
        "This room is called the art studio. Here you see blue paint, an empty cup, "
        "and the door to the hallway."
    )
    agent.percept = {"resulting_observation": observation}
    agent._update_relation_task_tracking(
        action="look at blue paint",
        observation=observation,
    )

    summary = agent._summarize_admissible_actions(
        [
            "look around",
            "look at blue paint",
            "open cupboard",
            "go to hallway",
        ],
        shortlist_limit=2,
    )

    shortlist = summary["task_relevant_action_shortlist"]
    assert summary["current_phase"] == "locate_primary_target"
    assert "go to hallway" in shortlist
    assert "open cupboard" not in shortlist


def test_relation_frontier_prefers_local_mechanism_over_unrelated_pairs(tmp_path):
    agent, _ = _build_agent(tmp_path, env_type="scienceworld")
    agent.task = (
        "Turn on the red light bulb by powering it using the solar panel. "
        "First, focus on the red light bulb. Then, create an electrical circuit that powers it on."
    )
    agent.task_status = "INCOMPLETE"
    agent._reset_episode_reasoning_state()
    observation = (
        "You are in the workshop. You see a red light bulb, a red light bulb cathode, "
        "a solar panel, a switch, a blue light bulb, and a buzzer."
    )
    agent.percept = {"resulting_observation": observation}
    agent._update_ordered_target_progress(
        executed_action="focus on red light bulb",
        observation="You focus on the red light bulb.",
    )
    agent._update_relation_task_tracking(
        action="look at solar panel",
        observation=observation,
    )
    agent._target_status_by_referent["red light bulb"] = "off"
    agent._get_hypothesis_entry("relation")["tests"] = 1

    summary = agent._summarize_admissible_actions(
        [
            "look at solar panel",
            "look at red light bulb cathode",
            "activate switch",
            "connect anode in buzzer to blue light bulb cathode",
        ],
        shortlist_limit=3,
    )

    shortlist = summary["task_relevant_action_shortlist"]
    assert summary["current_phase"] == "integrate_control_or_verify"
    assert summary["relation_frontier"]["frontier_entities"]
    assert "look at red light bulb cathode" in shortlist
    assert "activate switch" in shortlist
    assert "connect anode in buzzer to blue light bulb cathode" not in shortlist


def test_invalid_control_guess_shifts_relation_shortlist_to_adjacent_control(tmp_path):
    agent, _ = _build_agent(tmp_path, env_type="scienceworld")
    agent.task = (
        "Turn on the red light bulb by powering it using the solar panel. "
        "First, focus on the red light bulb. Then, create an electrical circuit that powers it on."
    )
    agent.task_status = "INCOMPLETE"
    agent._reset_episode_reasoning_state()
    observation = "You are in the workshop. You see a red light bulb, a solar panel, and a switch."
    agent.percept = {"resulting_observation": observation}
    agent._update_ordered_target_progress(
        executed_action="focus on red light bulb",
        observation="You focus on the red light bulb.",
    )
    agent._target_status_by_referent["red light bulb"] = "off"
    agent._get_hypothesis_entry("relation")["tests"] = 1
    agent._record_invalid_referent_attempt(
        family="device_control",
        action="activate solar panel",
    )

    summary = agent._summarize_admissible_actions(
        [
            "activate solar panel",
            "activate switch",
            "look at switch",
        ],
        shortlist_limit=2,
    )

    shortlist = summary["task_relevant_action_shortlist"]
    assert summary["current_phase"] == "integrate_control_or_verify"
    assert shortlist[0] == "activate switch"


def test_update_percept_exposes_relation_frontier_runtime_context(tmp_path):
    agent, _ = _build_agent(tmp_path, env_type="scienceworld")
    agent.task = (
        "Turn on the red light bulb by powering it using the solar panel. "
        "First, focus on the red light bulb. Then, create an electrical circuit that powers it on."
    )
    agent.task_status = "INCOMPLETE"
    agent.curr_episodic_memory = []
    agent.num_actions_taken = 1
    agent.max_actions = 35
    agent.action_agent = SimpleNamespace(system_message="")
    agent._action_agent_base_prompt = "BASE ACTION PROMPT"
    agent._reset_episode_reasoning_state()
    agent.admissible_actions = ["look around"]
    agent.adapter = SimpleNamespace(
        admissible_actions=[
            "look around",
            "look at solar panel",
            "focus on red light bulb",
            "activate switch",
        ],
        observation=(
            "You are in the workshop. You see a red light bulb, a solar panel, "
            "and a switch."
        ),
    )

    agent.update_percept("look at solar panel")

    assert "relation_frontier" in agent.percept
    assert agent.percept["relation_frontier"]["primary_target_grounded"] is True
    assert "Episode runtime snapshots" in agent.action_agent.system_message
    assert "relation_frontier" in agent.action_agent.system_message
