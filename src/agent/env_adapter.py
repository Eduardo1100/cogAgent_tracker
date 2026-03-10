from typing import Protocol, runtime_checkable


@runtime_checkable
class EnvironmentAdapter(Protocol):
    @property
    def observation(self) -> str: ...

    @property
    def admissible_actions(self) -> list[str]: ...

    @property
    def has_won(self) -> bool: ...

    @property
    def task(self) -> str: ...

    @property
    def initial_observation(self) -> str: ...

    def step(self, action: str) -> str: ...

    def infer_task_type(self) -> int | None: ...

    def count_inadmissible_actions(self, log_path: str) -> int: ...


def infer_task_type(task: str) -> int | None:
    """Infer ALFWorld task type (1–6) from the task description string."""
    t = task.lower()
    if "look at" in t:
        return 2
    if "clean" in t:
        return 3
    if "heat" in t:
        return 4
    if "cool" in t:
        return 5
    if "two" in t:
        return 6
    if "put" in t or "place" in t:
        return 1
    return None


class ALFWorldAdapter:
    def __init__(self, env, obs, info):
        self._env = env
        self._obs = obs
        self._info = info

    @property
    def observation(self) -> str:
        return self._obs[0]

    @property
    def admissible_actions(self) -> list[str]:
        return list(self._info["admissible_commands"][0])

    @property
    def has_won(self) -> bool:
        return self._info["won"][0]

    @property
    def task(self) -> str:
        return self._obs[0].split("Your task is to: ")[1]

    @property
    def initial_observation(self) -> str:
        return self._obs[0].split("Your task is to: ")[0].split("\n\n")[1]

    def step(self, action: str) -> str:
        self._obs, _, __, self._info = self._env.step([action])
        return self._obs[0]

    def infer_task_type(self) -> int | None:
        """Infer ALFWorld task type (1–6) from the task description string."""
        return infer_task_type(self.task)

    def count_inadmissible_actions(self, log_path: str) -> int:
        """Count how many times the agent attempted a non-admissible action."""
        try:
            with open(log_path) as f:
                return sum(
                    1 for line in f if "not in the list of admissible actions" in line
                )
        except Exception:
            return 0


class ScienceWorldAdapter:
    def __init__(self, env, obs, info):
        self._env = env
        self._obs = obs        # plain str
        self._info = info
        self._terminated = False

    @property
    def observation(self) -> str:
        return self._obs

    @property
    def admissible_actions(self) -> list[str]:
        return list(self._info.get("validActions", []))

    @property
    def has_won(self) -> bool:
        return self._terminated

    @property
    def task(self) -> str:
        return self._env.getTaskDescription()

    @property
    def initial_observation(self) -> str:
        return self._obs          # no task suffix to strip

    def step(self, action: str) -> str:
        self._obs, _, self._terminated, __, self._info = self._env.step(action)
        return self._obs

    def infer_task_type(self) -> int | None:
        task_names = self._env.getTaskNames()
        curr = getattr(self._env, "taskName", None) or ""
        try:
            return task_names.index(curr)
        except ValueError:
            return None

    def count_inadmissible_actions(self, log_path: str) -> int:
        try:
            with open(log_path) as f:
                return sum(
                    1 for line in f if "not in the list of admissible actions" in line
                )
        except Exception:
            return 0
