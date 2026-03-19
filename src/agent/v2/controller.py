from __future__ import annotations

from dataclasses import dataclass, field

from src.agent.v2.executor import DeterministicExecutor, ExecutionStep
from src.agent.v2.memory import RuntimeMemoryManager
from src.agent.v2.planner import SparsePlanner
from src.agent.v2.types import AdapterEvent, OptionContract, PlannerDirective
from src.agent.v2.world_model import WorldModel


@dataclass(frozen=True)
class ControllerStep:
    planner_directive: PlannerDirective
    execution_step: ExecutionStep


@dataclass
class V2Controller:
    planner: SparsePlanner = field(default_factory=SparsePlanner)
    executor: DeterministicExecutor = field(default_factory=DeterministicExecutor)
    memory: RuntimeMemoryManager = field(default_factory=RuntimeMemoryManager)
    failed_actions: set[str] = field(default_factory=set)
    executed_actions: list[str] = field(default_factory=list)
    current_option: OptionContract | None = None

    def observe(self, world_model: WorldModel, event: AdapterEvent) -> None:
        world_model.apply_event(event)
        if event.action_result.action_text:
            self.executed_actions.append(event.action_result.action_text)
        if event.action_result.failure_reason and event.action_result.action_text:
            self.failed_actions.add(event.action_result.action_text)
        self.memory.record_event(
            world_model.to_snapshot(),
            event,
            current_option=self.current_option,
        )

    def step(self, world_model: WorldModel) -> ControllerStep:
        world_model.set_memory_cues(
            self.memory.retrieve_cues(
                world_model.to_snapshot(),
                preferred_option=self.current_option,
            )
        )
        snapshot = world_model.to_snapshot()
        repair_pending = bool(snapshot.metadata.get("repair_pending", False))
        escalation_required = bool(snapshot.metadata.get("escalation_required", False))
        if repair_pending:
            self.current_option = None
            repair_option = OptionContract(
                family="recover_from_failure",
                objective=world_model.task_text,
                reasoning_budget="none",
                metadata={
                    "repair_operator": (
                        snapshot.repair_directive.operator
                        if snapshot.repair_directive is not None
                        else None
                    )
                },
            )
            world_model.set_active_option(repair_option)
            if snapshot.repair_directive is not None:
                execution_step = self.executor.select_repair_step(
                    snapshot,
                    repair_option,
                    snapshot.repair_directive,
                    failed_actions=self.failed_actions,
                    executed_actions=self.executed_actions,
                )
                if execution_step.action_label is not None:
                    self.current_option = repair_option
                    return ControllerStep(
                        planner_directive=PlannerDirective(
                            option=repair_option,
                            reasoning_tier="none",
                            continue_current_option=False,
                            planner_notes=[snapshot.repair_directive.rationale],
                            stop_reason="local_repair_selected",
                        ),
                        execution_step=execution_step,
                    )
            escalation_required = True
        if escalation_required:
            self.current_option = None
            repair_option = OptionContract(
                family="recover_from_failure",
                objective=world_model.task_text,
                reasoning_budget="full",
                metadata={
                    "repair_operator": (
                        snapshot.repair_directive.operator
                        if snapshot.repair_directive is not None
                        else None
                    )
                },
            )
            world_model.set_active_option(repair_option)
            return ControllerStep(
                planner_directive=PlannerDirective(
                    option=repair_option,
                    reasoning_tier="full",
                    continue_current_option=False,
                    planner_notes=[
                        "Bounded local repair did not resolve the contradiction; escalate to semantic recovery."
                    ],
                    stop_reason="semantic_escalation_required",
                ),
                execution_step=ExecutionStep(
                    option=repair_option,
                    operator=None,
                    action_label=None,
                    stop_reason="semantic_escalation_required",
                ),
            )
        directive = self.planner.plan(snapshot, current_option=self.current_option)
        self.current_option = directive.option
        world_model.set_active_option(self.current_option)
        if directive.option is None:
            execution_step = ExecutionStep(
                option=OptionContract(
                    family="recover_from_failure",
                    objective=world_model.task_text,
                    reasoning_budget="full",
                ),
                operator=None,
                action_label=None,
                stop_reason="no_option_selected",
            )
            return ControllerStep(
                planner_directive=directive,
                execution_step=execution_step,
            )
        execution_step = self.executor.select_next_step(
            snapshot,
            directive.option,
            failed_actions=self.failed_actions,
            executed_actions=self.executed_actions,
        )
        return ControllerStep(
            planner_directive=directive,
            execution_step=execution_step,
        )
