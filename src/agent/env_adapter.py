import json
import re
from typing import Any, Protocol, runtime_checkable

from src.agent.v2.types import (
    ActionResult,
    AdapterCapabilities,
    AdapterEvent,
    FrontierDelta,
    OperatorCandidate,
    SpatialContext,
    UIContext,
    UIElementRecord,
)

try:
    from nethack import ACTIONS as _NLE_ACTIONS

    _HAS_NLE = True
except ImportError:
    _NLE_ACTIONS = ()
    _HAS_NLE = False


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


@runtime_checkable
class V2EnvironmentAdapter(Protocol):
    """Normalized adapter seam for the v2 runtime.

    The v2 controller consumes typed adapter events rather than relying on raw
    text prompts. Concrete adapters may still expose legacy methods while the
    migration is in progress.
    """

    @property
    def observation(self) -> str: ...

    @property
    def task(self) -> str: ...

    def get_v2_capabilities(self) -> AdapterCapabilities: ...

    def build_v2_event(
        self,
        *,
        step_index: int,
        action_text: str | None = None,
        action_executed: bool | None = None,
        reward_delta: float | None = None,
        status_delta: dict[str, float | int | str | bool] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> AdapterEvent: ...


_TALES_ENV2TASK_CACHE: dict[str, str] | None = None


def _get_tales_env2task() -> dict[str, str]:
    global _TALES_ENV2TASK_CACHE
    if _TALES_ENV2TASK_CACHE is not None:
        return _TALES_ENV2TASK_CACHE
    try:
        import tales as _tales

        _TALES_ENV2TASK_CACHE = dict(_tales.env2task)
    except ImportError:
        _TALES_ENV2TASK_CACHE = {}
    return _TALES_ENV2TASK_CACHE


def _infer_operator_family(action_label: str) -> str:
    normalized = action_label.strip().lower()
    if normalized.startswith("move ") or normalized in {"go up", "go down"}:
        return "relocation"
    if normalized.startswith(("click ", "tap ", "press ")):
        return "ui_click"
    if normalized.startswith(("select ", "choose ")):
        return "ui_select"
    if normalized.startswith(("type ", "fill ", "enter ")):
        return "ui_type"
    if normalized.startswith(("scroll ", "page ", "hover ", "focus ", "go back")):
        return "ui_navigation"
    if normalized.startswith(("submit ", "confirm ", "send ")):
        return "ui_submit"
    if normalized.startswith(("go to ", "open url ", "visit ")):
        return "navigation"
    if normalized in {"search", "look", "inventory", "wait", "more"}:
        return "inspect"
    if normalized.startswith(("open", "close", "kick", "loot")):
        return "interaction"
    if normalized.startswith(
        (
            "pick up",
            "drop",
            "wear",
            "take off",
            "wield",
            "put on",
            "swap",
        )
    ):
        return "inventory"
    if normalized.startswith(
        ("eat", "quaff", "read", "zap", "throw", "apply", "fire", "cast")
    ):
        return "tool_application"
    if normalized.startswith(("pray", "ride", "rub", "tip", "turn", "teleport")):
        return "device_control"
    return "environment_action"


def _build_operator_candidates(
    adapter_name: str, admissible_actions: list[str]
) -> list[OperatorCandidate]:
    return [
        OperatorCandidate(
            operator_id=f"{adapter_name}:action:{idx}",
            family=_infer_operator_family(action),
            action_label=action,
        )
        for idx, action in enumerate(admissible_actions)
    ]


def build_text_only_adapter_event(
    *,
    adapter_name: str,
    task_text: str,
    observation: str,
    admissible_actions: list[str],
    step_index: int,
    action_text: str | None = None,
    action_executed: bool | None = None,
    reward_delta: float | None = None,
    status_delta: dict[str, float | int | str | bool] | None = None,
    metadata: dict[str, Any] | None = None,
) -> AdapterEvent:
    """Build a minimal normalized event for legacy text or grid adapters.

    This keeps Phase A compatibility simple while making room for richer UI
    adapters such as WebArena later.
    """

    operator_candidates = _build_operator_candidates(adapter_name, admissible_actions)
    event_metadata = {"adapter_name": adapter_name}
    if metadata:
        event_metadata.update(metadata)
    return AdapterEvent(
        step_index=step_index,
        task_text=task_text,
        raw_observation=observation,
        normalized_observation=observation,
        action_result=ActionResult(
            action_text=action_text,
            action_executed=action_executed,
        ),
        operator_candidates=operator_candidates,
        reward_delta=reward_delta,
        status_delta=status_delta or {},
        metadata=event_metadata,
    )


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

    def get_v2_capabilities(self) -> AdapterCapabilities:
        return AdapterCapabilities(
            adapter_name="alfworld",
            observation_mode="text",
            supports_operator_candidates=True,
        )

    def build_v2_event(
        self,
        *,
        step_index: int,
        action_text: str | None = None,
        action_executed: bool | None = None,
        reward_delta: float | None = None,
        status_delta: dict[str, float | int | str | bool] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> AdapterEvent:
        return build_text_only_adapter_event(
            adapter_name="alfworld",
            task_text=self.task,
            observation=self.observation,
            admissible_actions=self.admissible_actions,
            step_index=step_index,
            action_text=action_text,
            action_executed=action_executed,
            reward_delta=reward_delta,
            status_delta=status_delta,
            metadata=metadata,
        )


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

    def get_v2_capabilities(self) -> AdapterCapabilities:
        return AdapterCapabilities(
            adapter_name="scienceworld",
            observation_mode="text",
            supports_reward_delta=True,
            supports_operator_candidates=True,
        )

    def build_v2_event(
        self,
        *,
        step_index: int,
        action_text: str | None = None,
        action_executed: bool | None = None,
        reward_delta: float | None = None,
        status_delta: dict[str, float | int | str | bool] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> AdapterEvent:
        return build_text_only_adapter_event(
            adapter_name="scienceworld",
            task_text=self.task,
            observation=self.observation,
            admissible_actions=self.admissible_actions,
            step_index=step_index,
            action_text=action_text,
            action_executed=action_executed,
            reward_delta=reward_delta,
            status_delta=status_delta,
            metadata=metadata,
        )


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
        self._family = _get_tales_env2task().get(env_name, "unknown")
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

    def get_v2_capabilities(self) -> AdapterCapabilities:
        return AdapterCapabilities(
            adapter_name="tales",
            observation_mode="text",
            supports_operator_candidates=True,
        )

    def build_v2_event(
        self,
        *,
        step_index: int,
        action_text: str | None = None,
        action_executed: bool | None = None,
        reward_delta: float | None = None,
        status_delta: dict[str, float | int | str | bool] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> AdapterEvent:
        return build_text_only_adapter_event(
            adapter_name=f"tales:{self._family}",
            task_text=self.task,
            observation=self.observation,
            admissible_actions=self.admissible_actions,
            step_index=step_index,
            action_text=action_text,
            action_executed=action_executed,
            reward_delta=reward_delta,
            status_delta=status_delta,
            metadata=metadata,
        )


def _coerce_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        parts = [_coerce_text(item).strip() for item in value]
        return "\n".join(part for part in parts if part)
    if isinstance(value, dict):
        try:
            return json.dumps(value, sort_keys=True, default=str)
        except TypeError:
            return str(value)
    return str(value)


def _extract_webarena_task_text(
    observation: Any, info: dict[str, Any], task_config_path: str | None = None
) -> str:
    candidates = [
        info.get("task"),
        info.get("intent"),
        info.get("goal"),
        info.get("instruction"),
    ]
    if isinstance(observation, dict):
        candidates.extend(
            [
                observation.get("task"),
                observation.get("intent"),
                observation.get("goal"),
                observation.get("instruction"),
                observation.get("utterance"),
            ]
        )
    for candidate in candidates:
        text = _coerce_text(candidate).strip()
        if text:
            return text
    return task_config_path or "Complete the assigned web task."


def _extract_webarena_text_observation(observation: Any, info: dict[str, Any]) -> str:
    if isinstance(observation, str):
        return observation
    if isinstance(observation, dict):
        parts = [
            _coerce_text(observation.get(key) or info.get(key) or "").strip()
            for key in (
                "normalized_observation",
                "observation",
                "text",
                "axtree_txt",
                "accessibility_tree",
                "page_text",
                "utterance",
            )
        ]
        text = "\n".join(part for part in parts if part)
        if text:
            return text
    return _coerce_text(observation or info.get("observation") or "").strip()


def _extract_webarena_status_snapshot(
    observation: Any, info: dict[str, Any]
) -> dict[str, float | int | str | bool]:
    snapshot: dict[str, float | int | str | bool] = {}
    if isinstance(observation, dict):
        for key in (
            "url",
            "page_url",
            "title",
            "page_title",
            "success",
            "done",
            "error",
        ):
            if key in observation and observation[key] not in (None, ""):
                snapshot[key] = observation[key]
    for key in (
        "url",
        "page_url",
        "title",
        "page_title",
        "success",
        "done",
        "error",
        "site",
        "task_id",
    ):
        value = info.get(key)
        if value not in (None, ""):
            snapshot[key] = value
    return snapshot


def _extract_webarena_visible_elements(
    observation: Any, info: dict[str, Any]
) -> list[UIElementRecord]:
    raw_elements: Any = []
    if isinstance(observation, dict):
        raw_elements = (
            observation.get("visible_elements")
            or observation.get("elements")
            or observation.get("dom_elements")
            or []
        )
    if not raw_elements:
        raw_elements = (
            info.get("visible_elements")
            or info.get("elements")
            or info.get("dom_elements")
            or []
        )

    records: list[UIElementRecord] = []
    for idx, item in enumerate(raw_elements):
        if isinstance(item, UIElementRecord):
            records.append(item)
            continue
        if not isinstance(item, dict):
            text = _coerce_text(item).strip()
            if text:
                records.append(
                    UIElementRecord(
                        element_id=f"element-{idx}",
                        role="unknown",
                        text=text,
                    )
                )
            continue
        element_id = _coerce_text(
            item.get("element_id")
            or item.get("id")
            or item.get("backend_id")
            or item.get("bid")
            or f"element-{idx}"
        ).strip()
        role = _coerce_text(item.get("role") or item.get("tag") or "unknown").strip()
        records.append(
            UIElementRecord(
                element_id=element_id or f"element-{idx}",
                role=role or "unknown",
                name=_coerce_text(item.get("name") or item.get("label")).strip()
                or None,
                text=_coerce_text(item.get("text") or item.get("value")).strip()
                or None,
                selector=_coerce_text(item.get("selector") or item.get("xpath")).strip()
                or None,
                interactable=bool(item.get("interactable", True)),
                disabled=bool(item.get("disabled", False)),
                selected=bool(item.get("selected", False)),
                bounds=item.get("bounds") or {},
                attributes={
                    key: value
                    for key, value in item.items()
                    if key
                    not in {
                        "element_id",
                        "id",
                        "backend_id",
                        "bid",
                        "role",
                        "tag",
                        "name",
                        "label",
                        "text",
                        "value",
                        "selector",
                        "xpath",
                        "interactable",
                        "disabled",
                        "selected",
                        "bounds",
                    }
                },
            )
        )
    return records


def _extract_webarena_ui_context(observation: Any, info: dict[str, Any]) -> UIContext:
    visible_elements = _extract_webarena_visible_elements(observation, info)
    page_url = ""
    page_title = ""
    focused_element_id = None
    active_dialog = None
    action_scope = []
    if isinstance(observation, dict):
        page_url = _coerce_text(observation.get("page_url") or observation.get("url"))
        page_title = _coerce_text(
            observation.get("page_title") or observation.get("title")
        )
        focused_element_id = (
            _coerce_text(
                observation.get("focused_element_id") or observation.get("focused")
            ).strip()
            or None
        )
        active_dialog = (
            _coerce_text(
                observation.get("active_dialog") or observation.get("dialog")
            ).strip()
            or None
        )
        action_scope = list(observation.get("action_scope") or [])
    if not page_url:
        page_url = _coerce_text(info.get("page_url") or info.get("url"))
    if not page_title:
        page_title = _coerce_text(info.get("page_title") or info.get("title"))
    if not focused_element_id:
        focused_element_id = (
            _coerce_text(info.get("focused_element_id") or info.get("focused")).strip()
            or None
        )
    if not active_dialog:
        active_dialog = (
            _coerce_text(info.get("active_dialog") or info.get("dialog")).strip()
            or None
        )
    if not action_scope:
        action_scope = list(info.get("action_scope") or [])
    if not action_scope:
        inferred_scope: list[str] = ["click", "scroll", "go_back"]
        if any(
            element.role.lower() in {"textbox", "input", "textarea", "searchbox"}
            for element in visible_elements
        ):
            inferred_scope.append("type")
        action_scope = inferred_scope
    return UIContext(
        page_url=page_url or None,
        page_title=page_title or None,
        focused_element_id=focused_element_id,
        active_dialog=active_dialog,
        visible_elements=visible_elements,
        action_scope=action_scope,
        metadata={
            "site": info.get("site"),
            "task_id": info.get("task_id"),
        },
    )


def _build_webarena_action_surface(
    ui_context: UIContext, info: dict[str, Any]
) -> tuple[list[str], dict[str, str]]:
    aliases: dict[str, str] = {}
    raw_candidates = (
        info.get("valid")
        or info.get("valid_actions")
        or info.get("action_candidates")
        or info.get("available_actions")
        or info.get("admissible_actions")
        or []
    )
    normalized: list[str] = []
    if isinstance(raw_candidates, dict):
        raw_candidates = list(raw_candidates.values())
    for item in raw_candidates:
        if isinstance(item, dict):
            label = _coerce_text(item.get("label") or item.get("action")).strip()
            native = _coerce_text(item.get("action") or item.get("label")).strip()
        else:
            label = native = _coerce_text(item).strip()
        if label and label not in normalized:
            normalized.append(label)
            aliases[label] = native or label
    if normalized:
        return normalized, aliases

    generated: list[str] = []
    for element in ui_context.visible_elements[:12]:
        identifier = element.name or element.text or element.element_id
        if not identifier:
            continue
        role = element.role.lower()
        if role in {"textbox", "input", "textarea", "searchbox"}:
            generated.append(f"type {identifier}")
        elif role in {"combobox", "listbox", "select"}:
            generated.append(f"select {identifier}")
        else:
            generated.append(f"click {identifier}")
    generated.extend(["scroll down", "scroll up", "go back"])
    deduped: list[str] = []
    for action in generated:
        if action not in deduped:
            deduped.append(action)
            aliases[action] = action
    return deduped, aliases


def _format_webarena_observation(
    raw_text: str, ui_context: UIContext, task_text: str
) -> str:
    lines: list[str] = []
    task_line = task_text.strip()
    if task_line:
        lines.append(f"[Task] {task_line}")
    if ui_context.page_title or ui_context.page_url:
        lines.append("[Page]")
        if ui_context.page_title:
            lines.append(f"  title: {ui_context.page_title}")
        if ui_context.page_url:
            lines.append(f"  url: {ui_context.page_url}")
        if ui_context.focused_element_id:
            lines.append(f"  focused: {ui_context.focused_element_id}")
        if ui_context.active_dialog:
            lines.append(f"  dialog: {ui_context.active_dialog}")
    if ui_context.visible_elements:
        lines.append("[Visible Elements]")
        for element in ui_context.visible_elements[:12]:
            descriptor = element.name or element.text or element.element_id
            lines.append(
                f"  - id={element.element_id} role={element.role} label={descriptor}"
            )
    text = raw_text.strip()
    if text:
        lines.append("[Observation]")
        lines.append(text)
    return "\n".join(lines).strip()


def _extract_androidworld_task_text(
    goal: str | None, template: str | None, task_name: str | None
) -> str:
    for candidate in (goal, template, task_name):
        text = _coerce_text(candidate).strip()
        if text:
            return text
    return "Complete the assigned Android task."


def _extract_androidworld_ui_context(env: Any, state: Any) -> UIContext:
    raw_elements = list(getattr(state, "ui_elements", []) or [])
    visible_elements: list[UIElementRecord] = []
    for idx, element in enumerate(raw_elements):
        bbox = getattr(element, "bbox", None)
        bounds = {}
        if bbox is not None:
            bounds = {
                "x_min": getattr(bbox, "x_min", None),
                "x_max": getattr(bbox, "x_max", None),
                "y_min": getattr(bbox, "y_min", None),
                "y_max": getattr(bbox, "y_max", None),
            }

        label = (
            _coerce_text(getattr(element, "text", None)).strip()
            or _coerce_text(getattr(element, "content_description", None)).strip()
            or _coerce_text(getattr(element, "hint_text", None)).strip()
        )
        role = (
            _coerce_text(getattr(element, "class_name", None)).strip()
            or "android.widget.View"
        )
        visible_elements.append(
            UIElementRecord(
                element_id=f"ui-{idx}",
                role=role,
                name=label or None,
                text=(_coerce_text(getattr(element, "text", None)).strip() or None),
                interactable=bool(
                    getattr(element, "is_clickable", False)
                    or getattr(element, "is_editable", False)
                    or getattr(element, "is_long_clickable", False)
                    or getattr(element, "is_scrollable", False)
                ),
                disabled=not bool(getattr(element, "is_enabled", True)),
                selected=bool(getattr(element, "is_selected", False)),
                bounds={k: v for k, v in bounds.items() if v is not None},
                attributes={
                    "index": idx,
                    "editable": bool(getattr(element, "is_editable", False)),
                    "clickable": bool(getattr(element, "is_clickable", False)),
                    "long_clickable": bool(
                        getattr(element, "is_long_clickable", False)
                    ),
                    "scrollable": bool(getattr(element, "is_scrollable", False)),
                    "checkable": bool(getattr(element, "is_checkable", False)),
                    "checked": bool(getattr(element, "is_checked", False)),
                    "focused": bool(getattr(element, "is_focused", False)),
                    "visible": bool(getattr(element, "is_visible", True)),
                    "resource_id": _coerce_text(
                        getattr(element, "resource_id", None)
                    ).strip()
                    or None,
                    "package_name": _coerce_text(
                        getattr(element, "package_name", None)
                    ).strip()
                    or None,
                },
            )
        )

    action_scope = ["tap", "long_press", "scroll", "navigate_back", "navigate_home"]
    if any(
        bool(record.attributes.get("editable"))
        for record in visible_elements
        if record.attributes
    ):
        action_scope.append("type")

    page_title = _coerce_text(getattr(env, "foreground_activity_name", "")).strip()
    metadata = {
        "orientation": getattr(env, "orientation", None),
        "screen_size": getattr(env, "logical_screen_size", None),
    }
    return UIContext(
        page_url=None,
        page_title=page_title or None,
        visible_elements=visible_elements,
        action_scope=action_scope,
        metadata=metadata,
    )


def _format_androidworld_element(
    element: UIElementRecord, *, max_label_chars: int = 80
) -> str:
    label = element.name or element.text or ""
    label = label.replace("\n", " ").strip()
    if len(label) > max_label_chars:
        label = f"{label[: max_label_chars - 3]}..."
    attrs = element.attributes or {}
    flags: list[str] = []
    if attrs.get("editable"):
        flags.append("editable")
    if attrs.get("clickable"):
        flags.append("clickable")
    if attrs.get("long_clickable"):
        flags.append("long")
    if attrs.get("scrollable"):
        flags.append("scrollable")
    if attrs.get("checked"):
        flags.append("checked")
    if element.selected:
        flags.append("selected")
    if element.disabled:
        flags.append("disabled")
    flags_text = f" flags={','.join(flags)}" if flags else ""
    role = element.role.rsplit(".", 1)[-1]
    return (
        f"[{attrs.get('index', '?')}] role={role} label={label or '<none>'}{flags_text}"
    )


def _format_androidworld_observation(
    ui_context: UIContext,
    *,
    task_text: str,
    task_name: str | None,
    template: str | None,
) -> str:
    lines: list[str] = []
    if task_name:
        lines.append(f"[Task Name] {task_name}")
    lines.append(f"[Task] {task_text}")
    if template:
        lines.append(f"[Template] {_coerce_text(template).strip()}")
    if ui_context.page_title or ui_context.metadata:
        lines.append("[Screen]")
        if ui_context.page_title:
            lines.append(f"  activity: {ui_context.page_title}")
        screen_size = (ui_context.metadata or {}).get("screen_size")
        if screen_size:
            lines.append(f"  logical_size: {screen_size}")
        orientation = (ui_context.metadata or {}).get("orientation")
        if orientation is not None:
            lines.append(f"  orientation: {orientation}")
    if ui_context.visible_elements:
        lines.append("[Visible Elements]")
        for element in ui_context.visible_elements[:18]:
            lines.append(f"  - {_format_androidworld_element(element)}")
    else:
        lines.append("[Visible Elements]")
        lines.append("  - <none>")
    lines.append("[Action Syntax]")
    lines.append("  - tap [index]")
    lines.append("  - long press [index]")
    lines.append('  - type "text" into [index]')
    lines.append("  - scroll up|down|left|right")
    lines.append("  - scroll up|down|left|right [index]")
    lines.append("  - navigate back")
    lines.append("  - navigate home")
    lines.append("  - keyboard enter")
    lines.append("  - wait")
    return "\n".join(lines).strip()


_ANDROIDWORLD_INDEX_RE = re.compile(r"\[(\d+)\]")
_ANDROIDWORLD_TYPE_RE = re.compile(
    r'^(?:type|input_text)\s+"(?P<text>.*)"\s+into\s+\[(?P<index>\d+)\]\s*$',
    re.IGNORECASE,
)
_ANDROIDWORLD_SCROLL_RE = re.compile(
    r"^(?:scroll|swipe)\s+"
    r"(?P<direction>up|down|left|right)"
    r"(?:\s+\[(?P<index>\d+)\])?\s*$",
    re.IGNORECASE,
)


def _extract_androidworld_index(action: str) -> int:
    match = _ANDROIDWORLD_INDEX_RE.search(action)
    if not match:
        raise ValueError(f"Expected an indexed AndroidWorld action, got: {action!r}")
    return int(match.group(1))


class AndroidWorldAdapter:
    def __init__(
        self,
        env: Any,
        state: Any,
        *,
        task_name: str,
        goal: str,
        template: str | None = None,
        score_provider: Any | None = None,
    ):
        self._env = env
        self._state = state
        self._task_name = task_name
        self._goal = goal
        self._template = template
        self._score_provider = score_provider
        self._current_score = self._read_score()
        self._reward_delta = 0.0
        self._last_action_text: str | None = None
        self._last_action_executed: bool | None = None
        self._terminated = False
        self._ui_context = _extract_androidworld_ui_context(self._env, self._state)
        self._task_text = _extract_androidworld_task_text(
            self._goal, self._template, self._task_name
        )
        self._observation_text = _format_androidworld_observation(
            self._ui_context,
            task_text=self._task_text,
            task_name=self._task_name,
            template=self._template,
        )
        self._initial_observation = self._observation_text

    def _read_score(self) -> float:
        if self._score_provider is None:
            return 0.0
        try:
            return float(self._score_provider() or 0.0)
        except Exception:
            return 0.0

    def _refresh_cached_state(self) -> None:
        self._ui_context = _extract_androidworld_ui_context(self._env, self._state)
        self._observation_text = _format_androidworld_observation(
            self._ui_context,
            task_text=self._task_text,
            task_name=self._task_name,
            template=self._template,
        )

    @property
    def observation(self) -> str:
        return self._observation_text

    @property
    def admissible_actions(self) -> list[str]:
        actions: list[str] = []
        for element in self._ui_context.visible_elements[:18]:
            idx = int((element.attributes or {}).get("index", 0))
            actions.append(f"tap [{idx}]")
            if (element.attributes or {}).get("long_clickable"):
                actions.append(f"long press [{idx}]")
            if (element.attributes or {}).get("editable"):
                actions.append(f'type "..." into [{idx}]')
            if (element.attributes or {}).get("scrollable"):
                actions.extend(
                    [
                        f"scroll down [{idx}]",
                        f"scroll up [{idx}]",
                    ]
                )
        actions.extend(
            [
                "scroll down",
                "scroll up",
                "navigate back",
                "navigate home",
                "keyboard enter",
                "wait",
            ]
        )
        deduped: list[str] = []
        for action in actions:
            if action not in deduped:
                deduped.append(action)
        return deduped

    @property
    def has_won(self) -> bool:
        return self._current_score >= 1.0

    @property
    def task(self) -> str:
        return self._task_text

    @property
    def initial_observation(self) -> str:
        return self._initial_observation

    def _parse_action(self, action: str) -> Any:
        normalized = action.strip()
        lowered = normalized.lower()
        from android_world.env import json_action as aw_json_action

        type_match = _ANDROIDWORLD_TYPE_RE.match(normalized)
        if type_match:
            return aw_json_action.JSONAction(
                action_type=aw_json_action.INPUT_TEXT,
                index=int(type_match.group("index")),
                text=type_match.group("text"),
            )
        scroll_match = _ANDROIDWORLD_SCROLL_RE.match(normalized)
        if scroll_match:
            index = scroll_match.group("index")
            return aw_json_action.JSONAction(
                action_type=aw_json_action.SCROLL,
                direction=scroll_match.group("direction").lower(),
                index=int(index) if index is not None else None,
            )
        if lowered.startswith(("tap ", "click ")):
            return aw_json_action.JSONAction(
                action_type=aw_json_action.CLICK,
                index=_extract_androidworld_index(normalized),
            )
        if lowered.startswith("long press "):
            return aw_json_action.JSONAction(
                action_type=aw_json_action.LONG_PRESS,
                index=_extract_androidworld_index(normalized),
            )
        if lowered == "navigate back":
            return aw_json_action.JSONAction(action_type=aw_json_action.NAVIGATE_BACK)
        if lowered == "navigate home":
            return aw_json_action.JSONAction(action_type=aw_json_action.NAVIGATE_HOME)
        if lowered == "keyboard enter":
            return aw_json_action.JSONAction(action_type=aw_json_action.KEYBOARD_ENTER)
        if lowered == "wait":
            return aw_json_action.JSONAction(action_type=aw_json_action.WAIT)
        raise ValueError(f"Unsupported AndroidWorld action: {action!r}")

    def step(self, action: str) -> str:
        parsed_action = self._parse_action(action)
        self._env.execute_action(parsed_action)
        self._state = self._env.get_state(wait_to_stabilize=True)
        next_score = self._read_score()
        self._reward_delta = next_score - self._current_score
        self._current_score = next_score
        self._last_action_text = action
        self._last_action_executed = True
        self._terminated = self.has_won
        self._refresh_cached_state()
        return self._observation_text

    def set_observation(self, msg: str) -> None:
        self._observation_text = msg

    def infer_task_type(self) -> int | None:
        return None

    def count_inadmissible_actions(self, log_path: str) -> int:
        try:
            with open(log_path) as f:
                return sum(
                    1
                    for line in f
                    if "unsupported androidworld action" in line.lower()
                    or "invalid element index" in line.lower()
                )
        except Exception:
            return 0

    def get_v2_capabilities(self) -> AdapterCapabilities:
        return AdapterCapabilities(
            adapter_name="androidworld",
            observation_mode="ui",
            supports_ui_context=True,
            supports_status_delta=True,
            supports_reward_delta=True,
            supports_operator_candidates=True,
        )

    def build_v2_event(
        self,
        *,
        step_index: int,
        action_text: str | None = None,
        action_executed: bool | None = None,
        reward_delta: float | None = None,
        status_delta: dict[str, float | int | str | bool] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> AdapterEvent:
        event_metadata = {
            "adapter_name": "androidworld",
            "task_name": self._task_name,
            "template": self._template,
            "score": self._current_score,
            "terminated": self._terminated,
        }
        if metadata:
            event_metadata.update(metadata)
        effective_status_delta = status_delta or {
            "score": self._current_score,
            "activity": self._ui_context.page_title or "",
        }
        return AdapterEvent(
            step_index=step_index,
            task_text=self.task,
            raw_observation=self.observation,
            normalized_observation=self.observation,
            action_result=ActionResult(
                action_text=action_text or self._last_action_text,
                action_executed=(
                    action_executed
                    if action_executed is not None
                    else self._last_action_executed
                ),
                operator_family=(
                    _infer_operator_family(action_text or self._last_action_text or "")
                    if (action_text or self._last_action_text)
                    else None
                ),
            ),
            operator_candidates=_build_operator_candidates(
                "androidworld", self.admissible_actions
            ),
            reward_delta=(
                reward_delta if reward_delta is not None else self._reward_delta
            ),
            status_delta=effective_status_delta,
            novelty_signals=[
                element.element_id for element in self._ui_context.visible_elements[:4]
            ],
            ui_context=self._ui_context,
            metadata=event_metadata,
        )


class WebArenaAdapter:
    def __init__(
        self,
        env,
        obs: Any,
        info: dict[str, Any] | None,
        *,
        task_config_path: str | None = None,
    ):
        self._env = env
        self._obs = obs
        self._info = info or {}
        self._terminated = False
        self._truncated = False
        self._reward_delta = 0.0
        self._task_config_path = task_config_path
        self._task_text = _extract_webarena_task_text(
            self._obs, self._info, task_config_path
        )
        self._status_snapshot = _extract_webarena_status_snapshot(self._obs, self._info)
        self._ui_context = _extract_webarena_ui_context(self._obs, self._info)
        self._admissible_actions, self._action_aliases = _build_webarena_action_surface(
            self._ui_context, self._info
        )
        self._observation_text = _format_webarena_observation(
            _extract_webarena_text_observation(self._obs, self._info),
            self._ui_context,
            self._task_text,
        )
        self._initial_observation = self._observation_text
        self._last_action_text: str | None = None
        self._last_action_executed: bool | None = None

    @property
    def observation(self) -> str:
        return self._observation_text

    @property
    def admissible_actions(self) -> list[str]:
        return list(self._admissible_actions)

    @property
    def has_won(self) -> bool:
        return bool(
            self._info.get("success")
            or self._info.get("won")
            or self._info.get("has_won")
            or self._status_snapshot.get("success")
        )

    @property
    def task(self) -> str:
        return self._task_text

    @property
    def initial_observation(self) -> str:
        return self._initial_observation

    def _refresh_cached_state(self) -> None:
        self._task_text = _extract_webarena_task_text(
            self._obs, self._info, self._task_config_path
        )
        self._status_snapshot = _extract_webarena_status_snapshot(self._obs, self._info)
        self._ui_context = _extract_webarena_ui_context(self._obs, self._info)
        self._admissible_actions, self._action_aliases = _build_webarena_action_surface(
            self._ui_context, self._info
        )
        self._observation_text = _format_webarena_observation(
            _extract_webarena_text_observation(self._obs, self._info),
            self._ui_context,
            self._task_text,
        )

    def step(self, action: str) -> str:
        native_action = self._action_aliases.get(action, action)
        result = self._env.step(native_action)
        reward = 0.0
        if len(result) == 5:
            self._obs, reward, self._terminated, self._truncated, self._info = result
        elif len(result) == 4:
            self._obs, reward, self._terminated, self._info = result
            self._truncated = False
        else:
            raise ValueError(
                "Unsupported WebArena step() result. Expected 4 or 5 values."
            )
        self._reward_delta = float(reward or 0.0)
        self._last_action_text = action
        self._last_action_executed = True
        self._refresh_cached_state()
        return self._observation_text

    def set_observation(self, msg: str) -> None:
        self._observation_text = msg

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
                    or "invalid action" in line.lower()
                )
        except Exception:
            return 0

    def get_v2_capabilities(self) -> AdapterCapabilities:
        return AdapterCapabilities(
            adapter_name="webarena",
            observation_mode="ui",
            supports_ui_context=True,
            supports_status_delta=True,
            supports_reward_delta=True,
            supports_operator_candidates=True,
        )

    def build_v2_event(
        self,
        *,
        step_index: int,
        action_text: str | None = None,
        action_executed: bool | None = None,
        reward_delta: float | None = None,
        status_delta: dict[str, float | int | str | bool] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> AdapterEvent:
        effective_status_delta = status_delta or self._status_snapshot
        event_metadata = {
            "adapter_name": "webarena",
            "task_config_path": self._task_config_path,
            "status_snapshot": self._status_snapshot,
            "terminated": self._terminated,
            "truncated": self._truncated,
        }
        if metadata:
            event_metadata.update(metadata)
        return AdapterEvent(
            step_index=step_index,
            task_text=self.task,
            raw_observation=_coerce_text(self._obs),
            normalized_observation=self.observation,
            action_result=ActionResult(
                action_text=action_text or self._last_action_text,
                action_executed=(
                    action_executed
                    if action_executed is not None
                    else self._last_action_executed
                ),
                operator_family=(
                    _infer_operator_family(action_text or self._last_action_text or "")
                    if (action_text or self._last_action_text)
                    else None
                ),
            ),
            operator_candidates=_build_operator_candidates(
                "webarena", self.admissible_actions
            ),
            reward_delta=reward_delta
            if reward_delta is not None
            else self._reward_delta,
            status_delta=effective_status_delta,
            novelty_signals=[
                element.element_id for element in self._ui_context.visible_elements[:4]
            ],
            ui_context=self._ui_context,
            metadata=event_metadata,
        )


# ── NetHack action label helpers ──────────────────────────────────────────────

# Human-readable labels for NLE Command enums.
# Keys are the enum member *name* (e.g. "NORTH"); values are what the agent sees.
_NETHACK_LABEL_OVERRIDES: dict[str, str] = {
    # Cardinal/diagonal movement.
    # NLE enum *names* for the vi-key direction actions are single uppercase
    # letters ("N","S","E","W") and two-letter pairs ("NE","NW","SE","SW").
    # The longer "NORTH"/"SOUTH"/... names do NOT exist in the NLE enum space;
    # they were wrong keys that caused cardinal directions to fall through to
    # the lowercase fallback ("n","e","s","w") instead of "move north" etc.
    "N": "move north",
    "S": "move south",
    "E": "move east",
    "W": "move west",
    "NE": "move northeast",
    "NW": "move northwest",
    "SE": "move southeast",
    "SW": "move southwest",
    # Single-letter enum variants (vi-key style or letter shortcuts).
    # Map to the same "move X" labels so they receive "relocation" family
    # classification and are readable in the LLM's admissible list.
    "n": "move north",
    "s": "move south",
    "e": "move east",
    "w": "move west",
    "y": "move northwest",
    "u": "move northeast",
    "b": "move southwest",
    # Note: vi-key "n" typically means southeast in classic NetHack, but NLE
    # exposes it as a northward shortcut in some builds.  The override unifies
    # whichever cardinal "n" resolves to so the family classifier sees "move ".
    # Duplicate detection in _build_action_labels will skip any later entry
    # that collides with an already-registered label.
    "UP": "go up",
    "DOWN": "go down",
    "WAIT": "wait",
    "MORE": "more",
    "APPLY": "apply",
    "CAST": "cast",
    "CLOSE": "close door",
    "DROP": "drop",
    "DROPTYPE": "drop item type",
    "EAT": "eat",
    "ESC": "cancel",
    "FIRE": "fire",
    "INVENTORY": "inventory",
    "KICK": "kick",
    "LOOK": "look",
    "LOOT": "loot",
    "OPEN": "open door",
    "PAY": "pay",
    "PICKUP": "pick up",
    "PRAY": "pray",
    "PUTON": "put on",
    "QUAFF": "quaff",
    "READ": "read",
    "REMOVE": "take off",
    "RIDE": "ride",
    "RUB": "rub",
    "SEARCH": "search",
    "SWAP": "swap weapons",
    "TAKEOFF": "take off armor",
    "TELEPORT": "teleport",
    "THROW": "throw",
    "TIP": "tip",
    "TURN": "turn undead",
    "TWOWEAPON": "two weapon combat",
    "WEAR": "wear",
    "WIELD": "wield",
    "ZAP": "zap",
}


def _build_action_labels(env) -> tuple[list[str], dict[str, int]]:
    """Build index-aligned label list and label->index mapping for an NLE env."""
    action_space = env.unwrapped.actions if hasattr(env, "unwrapped") else env.actions
    labels: list[str] = []
    label_to_idx: dict[str, int] = {}
    for idx, action in enumerate(action_space):
        name = action.name if hasattr(action, "name") else str(action)
        label = _NETHACK_LABEL_OVERRIDES.get(name, name.lower().replace("_", " "))
        # Skip duplicate labels — two NLE enum values can map to the same
        # human-readable string (e.g. two "move southeast" entries). The first
        # one is sufficient; the duplicate would only confuse the scorer.
        if label in label_to_idx:
            continue
        labels.append(label)
        label_to_idx[label] = idx
    return labels, label_to_idx


# ── NetHackAdapter ────────────────────────────────────────────────────────────

_VARIANT_TASKS: dict[str, str] = {
    "NetHackScore-v0": (
        "Maximize your score by exploring the dungeon, fighting monsters, "
        "and collecting treasure."
    ),
    "NetHackChallenge-v0": (
        "Descend the dungeon, retrieve the Amulet of Yendor, and ascend to win."
    ),
    "NetHackStaircase-v0": (
        "Find and descend the staircase to reach the next dungeon level."
    ),
}

_ASCENSION_RE = re.compile(r"\b(ascended|escaped the Planes)\b", re.IGNORECASE)

# ── NetHack glyph classification for surroundings summary ─────────────────────

_NH_GLYPH: dict[str, str] = {
    # Walls & rock
    "|": "wall",
    "-": "wall",
    " ": "rock",
    # Floor & corridors
    ".": "floor",
    "#": "corridor",
    # Doors
    "+": "closed door",
    "'": "open door",
    # Stairs
    "<": "upstairs",
    ">": "downstairs",
    # Water / lava / air
    "}": "water/pool",
    "{": "fountain",
    "\\": "throne",
    "^": "trap",
    "_": "altar",
    # Items (broad categories)
    ")": "weapon",
    "[": "armor",
    "!": "potion",
    "?": "scroll",
    "/": "wand",
    "=": "ring",
    '"': "amulet",
    "$": "gold",
    "*": "gem/rock",
    "%": "corpse/food",
    "(": "tool",
    "`": "statue/boulder",
}

# Characters that are walkable (floor-like)
_NH_WALKABLE = frozenset(".#<>}{\\^_+'\"")

_DIRECTION_LABELS = {
    (-1, 0): "north",
    (1, 0): "south",
    (0, -1): "west",
    (0, 1): "east",
    (-1, -1): "northwest",
    (-1, 1): "northeast",
    (1, -1): "southwest",
    (1, 1): "southeast",
}

_STATUS_LINE_RE = re.compile(r"\b([A-Za-z$]+):([^\s]+)")


def _parse_surroundings(tty_chars) -> str | None:
    """Parse the 24x80 TTY grid and return a concise spatial summary.

    Returns None if the player position cannot be found.
    """
    # Find @ position (skip row 0 which is often the message line)
    player_row, player_col = None, None
    for r in range(24):
        for c in range(80):
            if chr(tty_chars[r][c]) == "@":
                player_row, player_col = r, c
                break
        if player_row is not None:
            break

    if player_row is None:
        return None

    parts: list[str] = []
    walkable_dirs: list[str] = []
    blocked_dirs: list[str] = []

    for (dr, dc), direction in _DIRECTION_LABELS.items():
        nr, nc = player_row + dr, player_col + dc
        if 0 <= nr < 24 and 0 <= nc < 80:
            ch = chr(tty_chars[nr][nc])
            label = _NH_GLYPH.get(ch)
            if label is None:
                # Uppercase = monster, lowercase = monster/pet, other = unknown
                if ch.isalpha():
                    label = f"creature '{ch}'"
                elif ch == ":":
                    label = "creature ':'"
                elif ch == ";":
                    label = "creature ';'"
                elif ch == "&":
                    label = "creature '&'"
                elif ch == "@":
                    label = "creature '@'"
                else:
                    label = f"'{ch}'"

            if ch in _NH_WALKABLE or ch.isalpha():
                walkable_dirs.append(direction)

            parts.append(f"  {direction}: {label}")

            if ch in ("|", "-", " "):
                blocked_dirs.append(direction)
        else:
            blocked_dirs.append(direction)

    lines = ["[Surroundings]"]
    lines.extend(parts)
    if walkable_dirs:
        lines.append(f"  passable: {', '.join(walkable_dirs)}")
    if blocked_dirs:
        lines.append(f"  blocked: {', '.join(blocked_dirs)}")

    return "\n".join(lines)


def _extract_surroundings_map(observation: str) -> dict[str, str]:
    mapping: dict[str, str] = {}
    for line in observation.splitlines():
        stripped = line.strip()
        if not stripped or stripped == "[Surroundings]":
            continue
        if stripped.startswith("passable:") or stripped.startswith("blocked:"):
            continue
        if ":" not in stripped:
            continue
        direction, label = stripped.split(":", 1)
        direction = direction.strip()
        if direction in _DIRECTION_LABELS.values():
            mapping[direction] = label.strip()
    return mapping


def _extract_passable_directions(observation: str) -> list[str]:
    for line in observation.splitlines():
        stripped = line.strip()
        if stripped.startswith("passable:"):
            values = stripped.removeprefix("passable:").strip()
            return [item.strip() for item in values.split(",") if item.strip()]
    return []


def _extract_blocked_directions(observation: str) -> list[str]:
    for line in observation.splitlines():
        stripped = line.strip()
        if stripped.startswith("blocked:"):
            values = stripped.removeprefix("blocked:").strip()
            return [item.strip() for item in values.split(",") if item.strip()]
    return []


def _extract_nethack_status(screen: str) -> dict[str, int | str]:
    status: dict[str, int | str] = {}
    for line in screen.splitlines():
        if "Dlvl:" not in line and "HP:" not in line and " S:" not in line:
            continue
        for key, value in _STATUS_LINE_RE.findall(line):
            if value.isdigit():
                status[key] = int(value)
            else:
                status[key] = value
    return status


def _compute_status_delta(
    previous: dict[str, int | str], current: dict[str, int | str]
) -> dict[str, int | str]:
    delta: dict[str, int | str] = {}
    for key, current_value in current.items():
        previous_value = previous.get(key)
        if previous_value == current_value:
            continue
        if isinstance(previous_value, int) and isinstance(current_value, int):
            delta[key] = current_value - previous_value
        else:
            delta[key] = current_value
    return delta


# Patterns for interactive prompts that the agent cannot answer through the
# normal action space.  Auto-confirming prevents the cognitive loop from
# burning its entire action budget on an unanswerable prompt.
_YN_PROMPT_RE = re.compile(r"\[y[naq]+\]")
_MORE_PROMPT_RE = re.compile(r"--More--")


class NetHackAdapter:
    """EnvironmentAdapter for the NetHack Learning Environment (NLE)."""

    def __init__(
        self,
        env,
        obs: dict,
        info: dict,
        variant: str = "NetHackScore-v0",
        render: bool = False,
    ):
        self._env = env
        self._obs = obs
        self._info = info
        self._variant = variant
        self._terminated = False
        self._cumulative_reward = 0.0
        self._obs_override: str | None = None
        self._render = render

        self._action_labels, self._label_to_idx = _build_action_labels(env)
        self._yn_confirm_idx = self._find_char_action_idx(env, ord("y"))
        self._more_dismiss_idx = self._find_char_action_idx(env, ord(" "))
        self._initial_obs_str = self._render_tty(obs)
        self._status_snapshot = _extract_nethack_status(self._initial_obs_str)
        self._passable_directions = _extract_passable_directions(
            self.initial_observation
        )
        self._blocked_directions = _extract_blocked_directions(self.initial_observation)
        self._last_transition: dict[str, Any] = {
            "action_text": None,
            "action_executed": None,
            "reward_delta": 0.0,
            "previous_status": self._status_snapshot,
            "current_status": self._status_snapshot,
            "previous_passable": self._passable_directions,
            "current_passable": self._passable_directions,
            "previous_blocked": self._blocked_directions,
            "current_blocked": self._blocked_directions,
        }
        if self._render:
            self._print_tty(self._initial_obs_str)

    # ── prompt helpers ──────────────────────────────────────────────────────

    @staticmethod
    def _find_char_action_idx(env, char_ord: int) -> int | None:
        """Return the action-space index whose enum value equals *char_ord*, or None."""
        action_space = (
            env.unwrapped.actions if hasattr(env, "unwrapped") else env.actions
        )
        for idx, action in enumerate(action_space):
            val = getattr(action, "value", None)
            if val is not None and int(val) == char_ord:
                return idx
        return None

    def _auto_dismiss_prompts(self) -> None:
        """Auto-confirm [ynq] prompts and dismiss --More-- messages.

        NetHack uses interactive prompts (e.g. "eat it? [ynq]") that the agent
        cannot answer through the normal action shortlist.  When the agent
        deliberately chose an action (eat, quaff, …), the confirmation is a
        formality — auto-sending 'y' preserves intent and prevents the
        cognitive loop from stalling.  --More-- messages are similarly
        auto-dismissed so the agent always sees the final game state.
        """
        for _ in range(5):  # bounded loop to prevent infinite cycling
            if self._terminated:
                break
            screen = self._render_tty(self._obs)
            if self._yn_confirm_idx is not None and _YN_PROMPT_RE.search(screen):
                send_idx = self._yn_confirm_idx
            elif self._more_dismiss_idx is not None and _MORE_PROMPT_RE.search(screen):
                send_idx = self._more_dismiss_idx
            else:
                break
            obs, reward, terminated, truncated, info = self._env.step(send_idx)
            self._obs = obs
            self._info = info
            self._terminated = terminated or truncated
            self._cumulative_reward += float(reward)

    # ── rendering ─────────────────────────────────────────────────────────

    @staticmethod
    def _render_tty(obs: dict) -> str:
        """Decode the 24x80 TTY character grid into a human-readable string."""
        tty = obs["tty_chars"]  # shape (24, 80), dtype int
        lines: list[str] = []
        for row in tty:
            line = "".join(chr(int(cell)) for cell in row).rstrip()
            lines.append(line)
        # strip trailing empty lines
        while lines and not lines[-1]:
            lines.pop()
        return "\n".join(lines)

    @staticmethod
    def _print_tty(screen: str) -> None:
        """Print the TTY screen to stdout with a clear-screen escape."""
        print("\033[2J\033[H" + screen, flush=True)

    # ── protocol properties ───────────────────────────────────────────────

    @property
    def observation(self) -> str:
        if self._obs_override is not None:
            return self._obs_override
        screen = self._render_tty(self._obs)
        surroundings = _parse_surroundings(self._obs["tty_chars"])
        if surroundings:
            return screen + "\n" + surroundings
        return screen

    @property
    def initial_observation(self) -> str:
        surroundings = _parse_surroundings(self._obs["tty_chars"])
        if surroundings:
            return self._initial_obs_str + "\n" + surroundings
        return self._initial_obs_str

    @property
    def admissible_actions(self) -> list[str]:
        return list(self._action_labels)

    @property
    def has_won(self) -> bool:
        msg = self._obs.get("message", None)
        if msg is not None:
            try:
                msg_str = bytes(msg).decode("latin-1").strip()
            except Exception:
                msg_str = ""
            if _ASCENSION_RE.search(msg_str):
                return True
        return False

    @property
    def task(self) -> str:
        return _VARIANT_TASKS.get(self._variant, _VARIANT_TASKS["NetHackScore-v0"])

    @property
    def cumulative_reward(self) -> float:
        return self._cumulative_reward

    # ── actions ───────────────────────────────────────────────────────────

    def step(self, action: str) -> str:
        self._obs_override = None
        action_key = action.lower().strip()
        previous_status = _extract_nethack_status(self._render_tty(self._obs))
        previous_passable = _extract_passable_directions(self.observation)
        previous_blocked = _extract_blocked_directions(self.observation)
        idx = self._label_to_idx.get(action_key)
        if idx is None:
            # Fuzzy-match via sentence-transformer similarity
            from src.agent.helpers import get_best_candidate

            best = get_best_candidate(action_key, self._action_labels)
            idx = self._label_to_idx.get(best, 0)

        obs, reward, terminated, truncated, info = self._env.step(idx)
        self._obs = obs
        self._info = info
        self._terminated = terminated or truncated
        self._cumulative_reward += float(reward)
        self._auto_dismiss_prompts()
        current_screen = self._render_tty(self._obs)
        self._status_snapshot = _extract_nethack_status(current_screen)
        self._passable_directions = _extract_passable_directions(self.observation)
        self._blocked_directions = _extract_blocked_directions(self.observation)
        self._last_transition = {
            "action_text": action_key,
            "action_executed": True,
            "reward_delta": float(reward),
            "previous_status": previous_status,
            "current_status": self._status_snapshot,
            "previous_passable": previous_passable,
            "current_passable": self._passable_directions,
            "previous_blocked": previous_blocked,
            "current_blocked": self._blocked_directions,
        }
        result = self.observation
        if self._render:
            self._print_tty(result)
        return result

    def set_observation(self, msg: str) -> None:
        self._obs_override = msg

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

    def get_v2_capabilities(self) -> AdapterCapabilities:
        return AdapterCapabilities(
            adapter_name="nethack",
            observation_mode="grid",
            supports_spatial_context=True,
            supports_reward_delta=True,
            supports_status_delta=True,
            supports_operator_candidates=True,
        )

    def build_v2_event(
        self,
        *,
        step_index: int,
        action_text: str | None = None,
        action_executed: bool | None = None,
        reward_delta: float | None = None,
        status_delta: dict[str, float | int | str | bool] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> AdapterEvent:
        current_observation = self.observation
        current_status = _extract_nethack_status(self._render_tty(self._obs))
        transition = self._last_transition
        effective_action = (
            action_text if action_text is not None else transition.get("action_text")
        )
        effective_executed = (
            action_executed
            if action_executed is not None
            else transition.get("action_executed")
        )
        effective_reward = (
            reward_delta if reward_delta is not None else transition.get("reward_delta")
        )
        computed_status_delta = _compute_status_delta(
            transition.get("previous_status", {}),
            current_status,
        )
        current_passable = _extract_passable_directions(current_observation)
        current_blocked = _extract_blocked_directions(current_observation)
        current_surroundings = _extract_surroundings_map(current_observation)
        opened = [
            direction
            for direction in current_passable
            if direction not in transition.get("previous_passable", [])
        ]
        closed = [
            direction
            for direction in transition.get("previous_passable", [])
            if direction not in current_passable
        ]
        blocked = [
            direction
            for direction in current_blocked
            if direction not in transition.get("previous_blocked", [])
        ]
        spatial_context = SpatialContext(
            topology="grid",
            current_region=(
                f"Dlvl:{current_status['Dlvl']}" if "Dlvl" in current_status else None
            ),
            current_node_id=(
                f"Dlvl:{current_status['Dlvl']}:T:{current_status['T']}"
                if "Dlvl" in current_status and "T" in current_status
                else None
            ),
            visible_nodes=list(current_surroundings.keys()),
            frontier_nodes=current_passable,
            passable_directions=current_passable,
            blocked_directions=current_blocked,
            local_map_summary=current_observation,
            metadata={"adjacent_tiles": current_surroundings},
        )
        entity_updates = [
            {
                "entity_id": f"adjacent:{direction}",
                "direction": direction,
                "label": label,
                "entity_type": "adjacent_tile",
            }
            for direction, label in current_surroundings.items()
        ]
        event_metadata = {
            "adapter_name": "nethack",
            "variant": self._variant,
            "cumulative_reward": self._cumulative_reward,
            "status_snapshot": current_status,
        }
        if metadata:
            event_metadata.update(metadata)
        return AdapterEvent(
            step_index=step_index,
            task_text=self.task,
            raw_observation=current_observation,
            normalized_observation=current_observation,
            action_result=ActionResult(
                action_text=effective_action,
                action_executed=effective_executed,
                operator_family=(
                    _infer_operator_family(effective_action)
                    if effective_action
                    else None
                ),
            ),
            operator_candidates=_build_operator_candidates(
                "nethack", self.admissible_actions
            ),
            entity_updates=entity_updates,
            reward_delta=effective_reward,
            status_delta=status_delta or computed_status_delta,
            frontier_delta=FrontierDelta(
                opened=opened,
                closed=closed,
                blocked=blocked,
            ),
            novelty_signals=opened + blocked,
            spatial_context=spatial_context,
            metadata=event_metadata,
        )
