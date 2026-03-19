from src.agent.v2.controller import V2Controller
from src.agent.v2.executor import DeterministicExecutor
from src.agent.v2.planner import SparsePlanner, build_option_contract
from src.agent.v2.types import (
    ActionResult,
    AdapterCapabilities,
    AdapterEvent,
    FrontierDelta,
    OperatorCandidate,
    SpatialContext,
    UIContext,
    UIElementRecord,
    WorldModelSnapshot,
)
from src.agent.v2.world_model import WorldModel


def test_sparse_planner_prefers_frontier_exploration_for_grid_state():
    planner = SparsePlanner()
    snapshot = WorldModelSnapshot(
        observation_mode="grid",
        goals=[],
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
        frontier_summary={"frontier_nodes": ["north", "east"]},
    )

    directive = planner.plan(snapshot)

    assert directive.option is not None
    assert directive.option.family == "explore_frontier"
    assert directive.option.target_signature == "north"
    assert directive.continue_current_option is False


def test_sparse_planner_prefers_ui_manipulation_for_webarena_style_state():
    planner = SparsePlanner()
    snapshot = WorldModelSnapshot(
        observation_mode="ui",
        goals=[],
        operator_candidates=[
            OperatorCandidate(
                operator_id="click-submit",
                family="ui_click",
                action_label="click submit",
                target_ids=["submit"],
            )
        ],
        metadata={
            "ui_context": UIContext(
                page_url="https://example.test/checkout",
                page_title="Checkout",
                focused_element_id="submit",
                visible_elements=[
                    UIElementRecord(
                        element_id="submit",
                        role="button",
                        text="Place order",
                    )
                ],
            ).to_dict()
        },
    )

    directive = planner.plan(snapshot)

    assert directive.option is not None
    assert directive.option.family == "manipulate_target"
    assert directive.option.target_signature == "submit"


def test_executor_selects_matching_operator_and_avoids_failed_actions():
    executor = DeterministicExecutor()
    option = build_option_contract(
        family="explore_frontier",
        objective="Explore the dungeon",
        target_signature="north",
    )
    snapshot = WorldModelSnapshot(
        observation_mode="grid",
        operator_candidates=[
            OperatorCandidate(
                operator_id="move-north",
                family="relocation",
                action_label="move north",
            ),
            OperatorCandidate(
                operator_id="move-east",
                family="relocation",
                action_label="move east",
            ),
            OperatorCandidate(
                operator_id="search",
                family="inspect",
                action_label="search",
            ),
        ],
    )

    step = executor.select_next_step(
        snapshot,
        option,
        failed_actions={"move north"},
        executed_actions=["search"],
    )

    assert step.action_label == "move east"
    assert step.stop_reason == "operator_selected"


def test_controller_continues_active_option_without_replanning():
    capabilities = AdapterCapabilities(
        adapter_name="nethack",
        observation_mode="grid",
        supports_spatial_context=True,
        supports_status_delta=True,
    )
    world_model = WorldModel(capabilities=capabilities, task_text="Explore the dungeon")
    controller = V2Controller()
    first_event = AdapterEvent(
        step_index=1,
        task_text="Explore the dungeon",
        raw_observation="Dlvl:1 T:1",
        normalized_observation="Dlvl:1 T:1",
        action_result=ActionResult(action_text="move north", action_executed=True),
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
        status_delta={"Dlvl": 1, "T": 1},
        frontier_delta=FrontierDelta(opened=["north"]),
        spatial_context=SpatialContext(
            topology="grid",
            current_region="Dlvl:1",
            current_node_id="Dlvl:1:T:1",
            visible_nodes=["north", "east"],
            frontier_nodes=["north", "east"],
            passable_directions=["north", "east"],
            blocked_directions=["west"],
        ),
    )
    controller.observe(world_model, first_event)

    first_step = controller.step(world_model)
    assert first_step.planner_directive.option is not None
    assert first_step.planner_directive.option.family == "explore_frontier"
    assert first_step.execution_step.action_label == "move north"

    second_event = AdapterEvent(
        step_index=2,
        task_text="Explore the dungeon",
        raw_observation="Dlvl:1 T:2",
        normalized_observation="Dlvl:1 T:2",
        action_result=ActionResult(action_text="move north", action_executed=True),
        operator_candidates=[
            OperatorCandidate(
                operator_id="move-east",
                family="relocation",
                action_label="move east",
            ),
            OperatorCandidate(
                operator_id="search",
                family="inspect",
                action_label="search",
            ),
        ],
        status_delta={"T": 2},
        frontier_delta=FrontierDelta(opened=["east"]),
        spatial_context=SpatialContext(
            topology="grid",
            current_region="Dlvl:1",
            current_node_id="Dlvl:1:T:2",
            visible_nodes=["east"],
            frontier_nodes=["east"],
            passable_directions=["east"],
            blocked_directions=["west", "north"],
        ),
    )
    controller.observe(world_model, second_event)
    second_step = controller.step(world_model)

    assert second_step.planner_directive.continue_current_option is True
    assert second_step.planner_directive.option is not None
    assert second_step.planner_directive.option.family == "explore_frontier"
    assert second_step.execution_step.action_label == "move east"


def test_controller_injects_memory_cues_into_future_planning():
    capabilities = AdapterCapabilities(
        adapter_name="webarena",
        observation_mode="ui",
        supports_ui_context=True,
        supports_status_delta=True,
    )
    world_model = WorldModel(capabilities=capabilities, task_text="Complete checkout")
    controller = V2Controller()
    controller.current_option = build_option_contract(
        family="manipulate_target",
        objective="Complete checkout",
        target_signature="submit",
    )
    event = AdapterEvent(
        step_index=1,
        task_text="Complete checkout",
        raw_observation="Checkout page",
        normalized_observation="Checkout page",
        action_result=ActionResult(
            action_text="click submit",
            action_executed=True,
            operator_family="ui_click",
        ),
        operator_candidates=[
            OperatorCandidate(
                operator_id="click-submit",
                family="ui_click",
                action_label="click submit",
                target_ids=["submit"],
            )
        ],
        ui_context=UIContext(
            page_url="https://example.test/checkout",
            page_title="Checkout",
            focused_element_id="submit",
            visible_elements=[
                UIElementRecord(
                    element_id="submit",
                    role="button",
                    text="Place order",
                )
            ],
        ),
        status_delta={"form_completion": 1},
        novelty_signals=["button_enabled"],
    )
    controller.observe(world_model, event)
    controller.current_option = None

    step = controller.step(world_model)

    assert world_model.memory_cues
    assert any(
        "manipulate_target" in cue.applicability for cue in world_model.memory_cues
    )
    assert step.planner_directive.option is not None
    assert step.planner_directive.option.family == "manipulate_target"


def test_controller_revises_before_act_after_repeated_contradiction():
    capabilities = AdapterCapabilities(
        adapter_name="nethack",
        observation_mode="grid",
        supports_spatial_context=True,
        supports_status_delta=True,
    )
    world_model = WorldModel(capabilities=capabilities, task_text="Explore the dungeon")
    controller = V2Controller()

    for step_index in (1, 2):
        controller.observe(
            world_model,
            AdapterEvent(
                step_index=step_index,
                task_text="Explore the dungeon",
                raw_observation="It's a wall.",
                normalized_observation="It's a wall.",
                action_result=ActionResult(
                    action_text="move north",
                    action_executed=True,
                    operator_family="relocation",
                ),
                operator_candidates=[
                    OperatorCandidate(
                        operator_id="move-north",
                        family="relocation",
                        action_label="move north",
                    ),
                    OperatorCandidate(
                        operator_id="move-east",
                        family="relocation",
                        action_label="move east",
                    ),
                    OperatorCandidate(
                        operator_id="search",
                        family="inspect",
                        action_label="search",
                    ),
                ],
                status_delta={"T": step_index},
                spatial_context=SpatialContext(
                    topology="grid",
                    current_region="Dlvl:1",
                    current_node_id=f"Dlvl:1:T:{step_index}",
                    visible_nodes=["east", "north"],
                    frontier_nodes=["east"],
                    passable_directions=["east"],
                    blocked_directions=["north", "west"],
                ),
            ),
        )

    step = controller.step(world_model)

    assert step.planner_directive.stop_reason == "revise_before_act"
    assert step.planner_directive.option is not None
    assert step.planner_directive.option.family == "recover_from_failure"
    assert step.execution_step.action_label != "move north"
