from src.agent.env_adapter import V2EnvironmentAdapter, build_text_only_adapter_event
from src.agent.v2.types import (
    AdapterCapabilities,
    AdapterEvent,
    MemoryCue,
    OperatorCandidate,
    OptionContract,
    PlannerDirective,
    UIContext,
    UIElementRecord,
    WorldModelSnapshot,
)


def test_build_text_only_adapter_event_wraps_legacy_adapter_state():
    event = build_text_only_adapter_event(
        adapter_name="nethack",
        task_text="Explore the dungeon",
        observation="You are in a room with stairs.",
        admissible_actions=["move north", "go down", "search"],
        step_index=3,
        action_text="search",
        action_executed=True,
        reward_delta=1.0,
        status_delta={"score": 4},
        metadata={"channel": "tty"},
    )

    assert isinstance(event, AdapterEvent)
    assert event.task_text == "Explore the dungeon"
    assert event.action_result.action_text == "search"
    assert event.action_result.action_executed is True
    assert [op.action_label for op in event.operator_candidates] == [
        "move north",
        "go down",
        "search",
    ]
    assert event.metadata["adapter_name"] == "nethack"
    assert event.metadata["channel"] == "tty"
    compact = event.to_compact_dict()
    assert compact["status_delta"] == {"score": 4}
    assert "ui_context" not in compact
    assert "spatial_context" not in compact


def test_v2_environment_adapter_protocol_accepts_structured_adapter():
    class FakeAdapter:
        observation = "Current page state"
        task = "Complete the checkout flow"

        def get_v2_capabilities(self) -> AdapterCapabilities:
            return AdapterCapabilities(
                adapter_name="webarena",
                observation_mode="ui",
                supports_ui_context=True,
                supports_status_delta=True,
            )

        def build_v2_event(
            self,
            *,
            step_index: int,
            action_text: str | None = None,
            action_executed: bool | None = None,
            reward_delta: float | None = None,
            status_delta=None,
            metadata=None,
        ) -> AdapterEvent:
            return AdapterEvent(
                step_index=step_index,
                task_text=self.task,
                raw_observation=self.observation,
                normalized_observation=self.observation,
            )

    assert isinstance(FakeAdapter(), V2EnvironmentAdapter)


def test_world_model_snapshot_preserves_ui_context_for_webarena_style_state():
    page = UIContext(
        page_url="https://example.test/checkout",
        page_title="Checkout",
        focused_element_id="submit",
        visible_elements=[
            UIElementRecord(
                element_id="email",
                role="textbox",
                name="Email",
                interactable=True,
            ),
            UIElementRecord(
                element_id="submit",
                role="button",
                text="Place order",
                interactable=True,
            ),
        ],
        action_scope=["click", "type", "scroll"],
    )
    option = OptionContract(
        family="manipulate_target",
        objective="Submit the checkout form",
        target_signature="submit",
        expected_outcomes=["ui_state_change", "goal_progress"],
        progress_budget=3,
        failure_budget=1,
        termination_conditions=["order_submitted"],
        interrupt_conditions=["modal_changed", "validation_error"],
        reasoning_budget="light",
    )
    snapshot = WorldModelSnapshot(
        observation_mode="ui",
        operator_candidates=[
            OperatorCandidate(
                operator_id="click-submit",
                family="ui_click",
                action_label="click submit",
                target_ids=["submit"],
            )
        ],
        active_option=option,
        memory_cues=[
            MemoryCue(
                cue_id="checkout-form",
                summary="Forms often require required fields before submit succeeds.",
                source_scope="semantic",
                applicability=["ui_form", "submit_flow"],
            )
        ],
        metadata={"ui_context": page.to_dict()},
    )
    directive = PlannerDirective(
        option=option,
        reasoning_tier="light",
        planner_notes=["Prefer the primary call-to-action after required fields."],
        memory_queries=["required-field validation patterns"],
    )

    compact_snapshot = snapshot.to_compact_dict()
    compact_directive = directive.to_compact_dict()

    assert compact_snapshot["observation_mode"] == "ui"
    assert compact_snapshot["active_option"]["family"] == "manipulate_target"
    assert compact_snapshot["metadata"]["ui_context"]["page_title"] == "Checkout"
    assert compact_snapshot["memory_cues"][0]["source_scope"] == "semantic"
    assert compact_directive["option"]["target_signature"] == "submit"
    assert compact_directive["planner_notes"] == [
        "Prefer the primary call-to-action after required fields."
    ]
