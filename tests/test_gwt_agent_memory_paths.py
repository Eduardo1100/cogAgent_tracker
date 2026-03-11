import importlib
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
