from src.agent.decision_state import (
    ActionSurfaceState,
    DecisionState,
    GoalState,
    GroundingState,
    OptionState,
    ProgressState,
    UncertaintyState,
)
from src.agent.v2.types import (
    ActionResult,
    AdapterCapabilities,
    AdapterEvent,
    FrontierDelta,
    MemoryCue,
    OperatorCandidate,
    OptionContract,
    SpatialContext,
    UIContext,
    UIElementRecord,
)
from src.agent.v2.world_model import (
    WorldModel,
    build_world_model_snapshot_from_decision_state,
)


def test_world_model_applies_spatial_event_deterministically():
    capabilities = AdapterCapabilities(
        adapter_name="nethack",
        observation_mode="grid",
        supports_spatial_context=True,
        supports_status_delta=True,
        supports_reward_delta=True,
    )
    model = WorldModel(capabilities=capabilities, task_text="Explore the dungeon")
    model.set_memory_cues(
        [
            MemoryCue(
                cue_id="frontier",
                summary="Prefer unexplored frontier before repeated local loops.",
                source_scope="semantic",
            )
        ]
    )
    model.set_active_option(
        OptionContract(
            family="explore_frontier",
            objective="Reach a new frontier node",
            target_signature="north",
            expected_outcomes=["frontier_expansion", "novel_observation"],
            progress_budget=4,
            failure_budget=1,
            termination_conditions=["frontier_expanded"],
            interrupt_conditions=["status_contradiction"],
            reasoning_budget="light",
        )
    )
    event = AdapterEvent(
        step_index=5,
        task_text="Explore the dungeon",
        raw_observation="Dlvl:1 T:5",
        normalized_observation="Dlvl:1 T:5",
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
                operator_id="search",
                family="inspect",
                action_label="search",
            ),
        ],
        entity_updates=[
            {
                "entity_id": "adjacent:north",
                "label": "floor",
                "entity_type": "adjacent_tile",
                "properties": {"direction": "north"},
            }
        ],
        reward_delta=3.0,
        status_delta={"Dlvl": 1, "T": 5, "S": 3},
        frontier_delta=FrontierDelta(opened=["north"], blocked=["west"]),
        novelty_signals=["north"],
        spatial_context=SpatialContext(
            topology="grid",
            current_region="Dlvl:1",
            current_node_id="Dlvl:1:T:5",
            visible_nodes=["north", "east", "west"],
            frontier_nodes=["north", "east"],
            passable_directions=["north", "east"],
            blocked_directions=["west"],
        ),
    )

    model.apply_event(event)
    snapshot = model.to_snapshot()
    analyst = model.to_analyst_view()

    assert snapshot.observation_mode == "grid"
    assert snapshot.metadata["total_reward"] == 3.0
    assert snapshot.metadata["status_snapshot"]["S"] == 3
    assert snapshot.frontier_summary["current_region"] == "Dlvl:1"
    assert snapshot.frontier_summary["opened_nodes"] == ["north"]
    assert snapshot.frontier_summary["visit_counts"]["Dlvl:1:T:5"] == 1
    assert snapshot.entities[0]["entity_id"] == "adjacent:north"
    assert snapshot.active_option is not None
    assert snapshot.active_option.family == "explore_frontier"
    assert snapshot.revision_required is False
    assert snapshot.action_outcomes
    assert snapshot.memory_cues[0].cue_id == "frontier"
    assert analyst["operator_families"] == ["inspect", "relocation"]
    assert analyst["memory_cue_count"] == 1


def test_world_model_preserves_ui_context_for_webarena_style_event():
    capabilities = AdapterCapabilities(
        adapter_name="webarena",
        observation_mode="ui",
        supports_ui_context=True,
        supports_status_delta=True,
    )
    event = AdapterEvent(
        step_index=2,
        task_text="Complete the checkout flow",
        raw_observation="Checkout page",
        normalized_observation="Checkout page",
        action_result=ActionResult(action_text="click submit", action_executed=True),
        operator_candidates=[
            OperatorCandidate(
                operator_id="click-submit",
                family="ui_click",
                action_label="click submit",
                target_ids=["submit"],
            )
        ],
        status_delta={"form_completion": 1},
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
            action_scope=["click", "type"],
        ),
        novelty_signals=["button_enabled"],
    )

    model = WorldModel.from_event(event, capabilities=capabilities)
    snapshot = model.to_snapshot()
    analyst = model.to_analyst_view()

    assert snapshot.observation_mode == "ui"
    assert snapshot.metadata["ui_context"]["page_title"] == "Checkout"
    assert snapshot.metadata["status_snapshot"]["form_completion"] == 1
    assert analyst["ui"]["focused_element_id"] == "submit"
    assert analyst["ui"]["visible_element_count"] == 1


def test_build_world_model_snapshot_from_decision_state_bridges_legacy_state():
    decision_state = DecisionState(
        action_surface=ActionSurfaceState(
            total_actions=7,
            shortlist=["move north", "search"],
            required_families=["relocation", "inspect"],
        ),
        goal_state=GoalState(
            task_status="INCOMPLETE",
            current_phase="act",
            task_contract={"search_mode": True},
        ),
        grounding_state=GroundingState(
            candidate_tracking={"active_candidate": "seed", "last_seen_room": "lab"},
            relation_frontier={"frontier_referents": ["switch", "wire"]},
            substance_search={"grounded_substances": ["water"]},
            artifact_creation={"grounded_artifacts": ["mixture"]},
            measurement_tracking={"measurement_target": "thermometer"},
            referent_resolution={
                "requested_target": "it",
                "resolved_target": "seed",
            },
        ),
        option_state=OptionState(),
        progress_state=ProgressState(
            task_relevant_affordance_delta_count=2,
            option_progress_debt=1,
            option_family_value=3,
        ),
        uncertainty_state=UncertaintyState(
            referent_resolution={
                "requested_target": "it",
                "resolved_target": "seed",
            }
        ),
    )

    snapshot = build_world_model_snapshot_from_decision_state(
        decision_state,
        task_text="Grow the plant",
    )
    compact = snapshot.to_compact_dict()

    assert compact["goals"][0]["description"] == "Grow the plant"
    assert compact["operator_candidates"][0]["action_label"] == "move north"
    assert "candidate:seed" in [entity["entity_id"] for entity in compact["entities"]]
    assert compact["unresolved_referents"] == ["it"]
    assert compact["uncertainty"][0]["category"] == "referent_resolution"
    assert compact["metadata"]["legacy_progress"]["option_family_value"] == 3


def test_world_model_marks_revision_required_after_repeated_blocked_action():
    capabilities = AdapterCapabilities(
        adapter_name="nethack",
        observation_mode="grid",
        supports_spatial_context=True,
    )
    model = WorldModel(capabilities=capabilities, task_text="Explore the dungeon")

    for step_index in (1, 2):
        model.apply_event(
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
                entity_updates=[
                    {
                        "entity_id": "adjacent:north",
                        "label": "wall",
                        "entity_type": "adjacent_tile",
                        "properties": {"direction": "north"},
                    }
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
            )
        )

    snapshot = model.to_snapshot()

    assert snapshot.revision_required is True
    assert snapshot.repair_directive is not None
    assert snapshot.repair_directive.operator == "repair_topology"
    assert snapshot.metadata["revision_reason"] == "repeated_contradiction"
    assert snapshot.metadata["contradiction_debt"] >= 2
    assert snapshot.metadata["repair_pending"] is True
    assert snapshot.metadata["escalation_required"] is False
    assert "move north" in snapshot.metadata["blocked_action_labels"]
    assert any(
        contradiction.category == "blocked_path"
        for contradiction in snapshot.contradictions
    )


def test_world_model_keeps_repair_cycle_active_until_structural_progress_returns():
    capabilities = AdapterCapabilities(
        adapter_name="nethack",
        observation_mode="grid",
        supports_spatial_context=True,
        supports_status_delta=True,
    )
    model = WorldModel(capabilities=capabilities, task_text="Explore the dungeon")

    for step_index in (1, 2):
        model.apply_event(
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
                    visible_nodes=["north", "east"],
                    frontier_nodes=["east"],
                    passable_directions=["east"],
                    blocked_directions=["north"],
                ),
            )
        )

    blocked_snapshot = model.to_snapshot()
    blocked_debt = blocked_snapshot.metadata["contradiction_debt"]
    assert blocked_snapshot.revision_required is True
    assert blocked_snapshot.metadata["repair_cycle_active"] is True

    model.set_active_option(
        OptionContract(
            family="recover_from_failure",
            objective="Explore the dungeon",
            reasoning_budget="none",
        )
    )
    model.apply_event(
        AdapterEvent(
            step_index=3,
            task_text="Explore the dungeon",
            raw_observation="The goblin throws a crude dagger! You are hit.",
            normalized_observation="The goblin throws a crude dagger! You are hit.",
            action_result=ActionResult(
                action_text="search",
                action_executed=True,
                operator_family="inspect",
            ),
            operator_candidates=[
                OperatorCandidate(
                    operator_id="search",
                    family="inspect",
                    action_label="search",
                ),
                OperatorCandidate(
                    operator_id="move-east",
                    family="relocation",
                    action_label="move east",
                ),
            ],
            status_delta={"HP": -3, "T": 3},
            spatial_context=SpatialContext(
                topology="grid",
                current_region="Dlvl:1",
                current_node_id="Dlvl:1:T:3",
                visible_nodes=["east"],
                frontier_nodes=["east"],
                passable_directions=["east"],
                blocked_directions=["north"],
            ),
        )
    )

    repair_snapshot = model.to_snapshot()

    assert repair_snapshot.revision_required is True
    assert repair_snapshot.metadata["revision_reason"] == "repair_incomplete"
    assert repair_snapshot.metadata["repair_cycle_active"] is True
    assert repair_snapshot.metadata["repair_pending"] is True
    assert repair_snapshot.metadata["escalation_required"] is False
    assert repair_snapshot.metadata["repair_attempt_count"] == 1
    assert repair_snapshot.metadata["contradiction_debt"] > blocked_debt
    assert any(
        contradiction.category == "negative_status_change"
        for contradiction in repair_snapshot.contradictions
    )


def test_world_model_escalates_after_bounded_repair_attempts_are_exhausted():
    capabilities = AdapterCapabilities(
        adapter_name="nethack",
        observation_mode="grid",
        supports_spatial_context=True,
        supports_status_delta=True,
    )
    model = WorldModel(capabilities=capabilities, task_text="Explore the dungeon")

    for step_index in (1, 2):
        model.apply_event(
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
                    visible_nodes=["north", "east"],
                    frontier_nodes=["east"],
                    passable_directions=["east"],
                    blocked_directions=["north"],
                ),
            )
        )

    model.set_active_option(
        OptionContract(
            family="recover_from_failure",
            objective="Explore the dungeon",
            reasoning_budget="none",
        )
    )
    for step_index in (3, 4):
        model.apply_event(
            AdapterEvent(
                step_index=step_index,
                task_text="Explore the dungeon",
                raw_observation="Nothing happens.",
                normalized_observation="Nothing happens.",
                action_result=ActionResult(
                    action_text="search",
                    action_executed=True,
                    operator_family="inspect",
                ),
                operator_candidates=[
                    OperatorCandidate(
                        operator_id="search",
                        family="inspect",
                        action_label="search",
                    ),
                    OperatorCandidate(
                        operator_id="move-east",
                        family="relocation",
                        action_label="move east",
                    ),
                ],
                status_delta={"T": step_index},
                spatial_context=SpatialContext(
                    topology="grid",
                    current_region="Dlvl:1",
                    current_node_id=f"Dlvl:1:T:{step_index}",
                    visible_nodes=["east"],
                    frontier_nodes=["east"],
                    passable_directions=["east"],
                    blocked_directions=["north"],
                ),
            )
        )

    snapshot = model.to_snapshot()

    assert snapshot.revision_required is True
    assert snapshot.metadata["repair_pending"] is False
    assert snapshot.metadata["escalation_required"] is True
    assert snapshot.metadata["revision_reason"] == "repair_budget_exhausted"
    assert snapshot.metadata["repair_attempt_count"] == 2


def test_world_model_detects_ui_target_binding_contradiction_and_requests_repair():
    capabilities = AdapterCapabilities(
        adapter_name="webarena",
        observation_mode="ui",
        supports_ui_context=True,
        supports_status_delta=True,
    )
    model = WorldModel(capabilities=capabilities, task_text="Complete checkout")

    model.apply_event(
        AdapterEvent(
            step_index=1,
            task_text="Complete checkout",
            raw_observation="Checkout page.",
            normalized_observation="Checkout page.",
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
                ),
                OperatorCandidate(
                    operator_id="scroll-down",
                    family="ui_navigation",
                    action_label="scroll down",
                ),
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
                action_scope=["click", "scroll"],
            ),
        )
    )
    model.apply_event(
        AdapterEvent(
            step_index=2,
            task_text="Complete checkout",
            raw_observation="Element submit is not visible anymore.",
            normalized_observation="Element submit is not visible anymore.",
            action_result=ActionResult(
                action_text="click submit",
                action_executed=True,
                operator_family="ui_click",
                failure_reason="element not visible",
            ),
            operator_candidates=[
                OperatorCandidate(
                    operator_id="click-cart",
                    family="ui_click",
                    action_label="click cart",
                    target_ids=["cart"],
                ),
                OperatorCandidate(
                    operator_id="scroll-down",
                    family="ui_navigation",
                    action_label="scroll down",
                ),
                OperatorCandidate(
                    operator_id="type-search",
                    family="ui_type",
                    action_label="type search",
                    target_ids=["search"],
                ),
            ],
            ui_context=UIContext(
                page_url="https://example.test/checkout",
                page_title="Checkout",
                focused_element_id="cart",
                visible_elements=[
                    UIElementRecord(
                        element_id="cart",
                        role="button",
                        text="Cart",
                    ),
                    UIElementRecord(
                        element_id="search",
                        role="textbox",
                        name="Search",
                    ),
                ],
                action_scope=["click", "type", "scroll"],
            ),
        )
    )

    snapshot = model.to_snapshot()

    assert snapshot.revision_required is True
    assert snapshot.repair_directive is not None
    assert snapshot.repair_directive.operator == "repair_target_binding"
    assert snapshot.metadata["repair_pending"] is True
    assert snapshot.metadata["escalation_required"] is False
    assert any(
        contradiction.category == "ui_target_binding"
        for contradiction in snapshot.contradictions
    )
