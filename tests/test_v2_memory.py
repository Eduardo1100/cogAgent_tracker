from src.agent.v2.memory import (
    EpisodicMemoryStore,
    RuntimeMemoryManager,
    SemanticMemoryStore,
)
from src.agent.v2.types import (
    ActionResult,
    AdapterEvent,
    FrontierDelta,
    GoalRecord,
    OperatorCandidate,
    OptionContract,
    WorldModelSnapshot,
)


def _grid_snapshot() -> WorldModelSnapshot:
    return WorldModelSnapshot(
        observation_mode="grid",
        goals=[GoalRecord(goal_id="primary", description="Explore the dungeon")],
        operator_candidates=[
            OperatorCandidate(
                operator_id="move-north",
                family="relocation",
                action_label="move north",
            ),
            OperatorCandidate(
                operator_id="search",
                family="inspect",
                action_label="search",
            ),
        ],
        frontier_summary={"current_region": "Dlvl:1", "frontier_nodes": ["north"]},
    )


def _grid_event(step_index: int = 1) -> AdapterEvent:
    return AdapterEvent(
        step_index=step_index,
        task_text="Explore the dungeon",
        raw_observation="Dlvl:1",
        normalized_observation="Dlvl:1",
        action_result=ActionResult(
            action_text="move north",
            action_executed=True,
            operator_family="relocation",
        ),
        reward_delta=2.0,
        status_delta={"T": step_index, "S": 2},
        frontier_delta=FrontierDelta(opened=["north"]),
        novelty_signals=["north"],
    )


def test_runtime_memory_records_episodic_state_option_outcome():
    manager = RuntimeMemoryManager()
    snapshot = _grid_snapshot()
    option = OptionContract(
        family="explore_frontier",
        objective="Explore the dungeon",
        target_signature="north",
        expected_outcomes=["frontier_expansion"],
    )
    record = manager.record_event(snapshot, _grid_event(), current_option=option)

    assert record.state_signature.startswith("grid|Dlvl:1|Explore the dungeon|")
    assert record.option_signature == "explore_frontier|north"
    assert "frontier-opened" in record.outcome_signature
    assert record.option_family == "explore_frontier"
    assert len(manager.episodic_store.records) == 1


def test_semantic_memory_retrieves_relevant_cues():
    episodic = EpisodicMemoryStore()
    semantic = SemanticMemoryStore()
    manager = RuntimeMemoryManager(episodic_store=episodic, semantic_store=semantic)
    option = OptionContract(
        family="explore_frontier",
        objective="Explore the dungeon",
        target_signature="north",
    )
    manager.record_event(
        _grid_snapshot(), _grid_event(step_index=1), current_option=option
    )
    manager.record_event(
        _grid_snapshot(), _grid_event(step_index=2), current_option=option
    )

    cues = manager.retrieve_cues(_grid_snapshot(), preferred_option=option)

    assert cues
    assert cues[0].source_scope == "semantic"
    assert "explore_frontier" in cues[0].applicability


def test_runtime_memory_falls_back_to_recent_episodic_cues_when_semantic_empty():
    manager = RuntimeMemoryManager()
    cue_snapshot = WorldModelSnapshot(
        observation_mode="ui",
        goals=[GoalRecord(goal_id="primary", description="Checkout")],
        operator_candidates=[],
    )
    manager.episodic_store.add(
        manager.record_event(
            _grid_snapshot(),
            _grid_event(step_index=3),
            current_option=OptionContract(
                family="inspect_novelty",
                objective="Explore the dungeon",
                target_signature="north",
            ),
        )
    )
    manager.semantic_store.records = {}

    cues = manager.retrieve_cues(cue_snapshot, limit=1)

    assert len(cues) == 1
    assert cues[0].source_scope == "episodic"
