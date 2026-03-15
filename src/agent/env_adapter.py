import re
from typing import Protocol, runtime_checkable

try:
    import tales as _tales

    _TALES_ENV2TASK: dict[str, str] = _tales.env2task
except ImportError:
    _TALES_ENV2TASK = {}


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

    def set_observation(self, msg: str) -> None: ...

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
        parts = self._obs[0].split("Your task is to: ")
        return parts[1] if len(parts) > 1 else self._obs[0]

    @property
    def initial_observation(self) -> str:
        return self._obs[0].split("Your task is to: ")[0].split("\n\n")[1]

    def step(self, action: str) -> str:
        self._obs, _, __, self._info = self._env.step([action])
        return self._obs[0]

    def set_observation(self, msg: str) -> None:
        self._obs = [msg]

    def infer_task_type(self) -> int | None:
        """Infer ALFWorld task type (1–6) from the task description string."""
        return infer_task_type(self.task)

    def count_inadmissible_actions(self, log_path: str) -> int:
        """Count how many times the agent attempted a non-admissible action."""
        try:
            with open(log_path) as f:
                return sum(
                    1
                    for line in f
                    if "not in the list of admissible actions" in line
                    or "is not admissible" in line
                )
        except Exception:
            return 0


class ScienceWorldAdapter:
    def __init__(self, env, obs, info, task_name: str = ""):
        self._env = env
        self._obs = obs
        self._info = info
        self._terminated = False
        self._task_name = task_name

    @property
    def observation(self) -> str:
        return self._obs

    @property
    def admissible_actions(self) -> list[str]:
        return list(self._info.get("valid", []))

    @property
    def has_won(self) -> bool:
        return self._info.get("has_won", self._info.get("hasWon", False))

    @property
    def task(self) -> str:
        # CamelCase: getTaskDescription() -> snake_case: get_task_description()
        return self._env.get_task_description()

    @property
    def initial_observation(self) -> str:
        return self._obs

    def step(self, action: str) -> str:
        result = self._env.step(action)
        self._obs, reward, self._terminated, self._info = result[:4]
        return self._obs

    def set_observation(self, msg: str) -> None:
        self._obs = msg

    def infer_task_type(self) -> int | None:
        try:
            return self._env.get_task_names().index(self._task_name)
        except ValueError:
            return None

    def count_inadmissible_actions(self, log_path: str) -> int:
        try:
            with open(log_path) as f:
                return sum(
                    1
                    for line in f
                    if "not in the list of admissible actions" in line
                    or "is not admissible" in line
                )
        except Exception:
            return 0


_INFOCOM_PREAMBLE_RE = re.compile(
    r"^.*?(?:(?:copyright|trademark).*?infocom.*?\n|revision\s+\d+.*?\n)+",
    re.IGNORECASE | re.DOTALL,
)


def _jericho_task_description(env_name: str) -> str:
    """'JerichoEnvZork1' → 'Zork 1 (explore the world, collect treasures)'."""
    name = env_name.removeprefix("JerichoEnv")
    name = re.sub(r"([A-Za-z])(\d)", r"\1 \2", name)  # "Zork1" → "Zork 1"
    return f"{name} (explore the world, collect treasures)"


def _strip_infocom_preamble(obs: str) -> str:
    """Remove Infocom copyright/revision header from the initial observation.

    Jericho games prepend a copyright block before the first room description.
    Stripping it keeps salient-entity extraction focused on room objects.
    """
    stripped = _INFOCOM_PREAMBLE_RE.sub("", obs).lstrip("\n")
    return stripped if stripped else obs


class TalesAdapter:
    def __init__(self, env, obs: str, info: dict, env_name: str = ""):
        self._env = env
        self._env_name = env_name
        self._family = _TALES_ENV2TASK.get(env_name, "unknown")
        self._info = info
        self._terminated = False
        self._obs = self._normalize_obs(obs)

    def _normalize_obs(self, obs: str) -> str:
        """Strip env-family-specific noise from an observation string."""
        if self._family in ("jericho", "textworld"):
            obs = _strip_infocom_preamble(obs)
        elif self._family == "textworld_express":
            task_desc = self._info.get("taskDescription", "")
            if task_desc and obs.startswith(task_desc):
                obs = obs[len(task_desc) :].lstrip("\n")
        elif self._family == "scienceworld":
            task_desc = self._info.get("taskDesc", "")
            if task_desc and obs.startswith(task_desc):
                obs = obs[len(task_desc) :].lstrip("\n")
        # alfworld: no stripping needed (obs is already clean)
        return obs

    @property
    def observation(self) -> str:
        return self._obs

    @property
    def initial_observation(self) -> str:
        return self._obs

    @property
    def admissible_actions(self) -> list[str]:
        cmds = self._info.get("admissible_commands") or self._info.get("valid") or []
        seen: set[str] = set()
        result = []
        for c in cmds:
            c = c.strip()
            if c and c not in seen:
                seen.add(c)
                result.append(c)
        return result

    @property
    def has_won(self) -> bool:
        return bool(
            self._info.get("won")
            or self._info.get("has_won")
            or self._info.get("hasWon")
        )

    @property
    def task(self) -> str:
        if self._family == "textworld_express":
            return self._info.get("taskDescription") or self._env_name
        if self._family == "scienceworld":
            return self._info.get("taskDesc") or self._env_name
        if self._family == "alfworld":
            parts = self._obs.split("Your task is to: ")
            if len(parts) > 1:
                return parts[1].split("\n")[0].strip()
        if self._family == "jericho":
            return _jericho_task_description(self._env_name)
        if self._family == "textworld":
            for line in self._obs.splitlines():
                if line.strip():
                    return line.strip()
        return self._info.get("objective") or self._info.get("task") or self._env_name

    def step(self, action: str) -> str:
        result = self._env.step(action)
        # TALES envs use old-gym 4-tuple (obs, reward, done, info);
        # handle both that and gymnasium's 5-tuple (obs, reward, terminated, truncated, info).
        if len(result) == 4:
            self._obs, _, self._terminated, self._info = result
        else:
            self._obs, _, self._terminated, _truncated, self._info = result[:5]
        return self._obs

    def set_observation(self, msg: str) -> None:
        self._obs = msg

    def infer_task_type(self) -> int | None:
        return None

    def count_inadmissible_actions(self, log_path: str) -> int:
        try:
            with open(log_path) as f:
                return sum(
                    1
                    for line in f
                    if "not in the list of admissible actions" in line
                    or "is not admissible" in line
                )
        except Exception:
            return 0
