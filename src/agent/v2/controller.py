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
        if snapshot.revision_required:
            self.current_option = None
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
