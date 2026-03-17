import re
from typing import Protocol, runtime_checkable

try:
    import tales as _tales

    _TALES_ENV2TASK: dict[str, str] = _tales.env2task
except ImportError:
    _TALES_ENV2TASK = {}

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
    "|": "wall", "-": "wall", " ": "rock",
    # Floor & corridors
    ".": "floor", "#": "corridor",
    # Doors
    "+": "closed door", "'": "open door",
    # Stairs
    "<": "upstairs", ">": "downstairs",
    # Water / lava / air
    "}": "water/pool", "{": "fountain", "\\": "throne",
    "^": "trap", "_": "altar",
    # Items (broad categories)
    ")": "weapon", "[": "armor", "!": "potion", "?": "scroll",
    "/": "wand", "=": "ring", '"': "amulet", "$": "gold",
    "*": "gem/rock", "%": "corpse/food", "(": "tool", "`": "statue/boulder",
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
        if self._render:
            self._print_tty(self._initial_obs_str)

    # ── prompt helpers ──────────────────────────────────────────────────────

    @staticmethod
    def _find_char_action_idx(env, char_ord: int) -> int | None:
        """Return the action-space index whose enum value equals *char_ord*, or None."""
        action_space = env.unwrapped.actions if hasattr(env, "unwrapped") else env.actions
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
            line = bytes(row).decode("latin-1").rstrip()
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
