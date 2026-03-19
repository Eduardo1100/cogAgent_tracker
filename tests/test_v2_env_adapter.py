from types import SimpleNamespace

import numpy as np

from src.agent.env_adapter import (
    NetHackAdapter,
    _infer_operator_family,
)


def _build_nethack_obs(
    room_lines: list[str],
    *,
    player_header: str,
    status_line: str,
    top_row: int = 3,
    left_col: int = 60,
) -> dict:
    grid = np.full((24, 80), ord(" "), dtype=np.int32)
    for i, line in enumerate(room_lines):
        for j, ch in enumerate(line):
            grid[top_row + i][left_col + j] = ord(ch)
    for j, ch in enumerate(player_header):
        grid[18][j] = ord(ch)
    for j, ch in enumerate(status_line):
        grid[19][j] = ord(ch)
    return {"tty_chars": grid, "message": b""}


class _FakeNetHackEnv:
    def __init__(self, next_obs: dict):
        self.actions = [
            SimpleNamespace(name="N", value=1),
            SimpleNamespace(name="E", value=2),
            SimpleNamespace(name="SEARCH", value=3),
            SimpleNamespace(name="UP", value=4),
            SimpleNamespace(name="DOWN", value=5),
        ]
        self._next_obs = next_obs

    def step(self, idx: int):
        return self._next_obs, 4.0, False, False, {}


def test_infer_operator_family_stays_generic():
    assert _infer_operator_family("move north") == "relocation"
    assert _infer_operator_family("go down") == "relocation"
    assert _infer_operator_family("search") == "inspect"
    assert _infer_operator_family("apply") == "tool_application"
    assert _infer_operator_family("open door") == "interaction"


def test_nethack_adapter_builds_normalized_v2_event():
    initial_obs = _build_nethack_obs(
        [
            "-----",
            "|...|",
            "....|",
            "|.@<|",
            "|.f.|",
            "|...|",
            "---.-",
        ],
        player_header=(
            "Agent the Candidate            St:14 Dx:15 Co:11 In:8 Wi:18 Ch:10 Neutral S:0"
        ),
        status_line="Dlvl:1 $:0 HP:14(14) Pw:4(4) AC:4 Xp:1/0 T:1",
    )
    next_obs = _build_nethack_obs(
        [
            "-----",
            "|...|",
            "....|",
            "|..@|",
            "|.f.|",
            "|...|",
            "---.-",
        ],
        player_header=(
            "Agent the Candidate            St:14 Dx:15 Co:11 In:8 Wi:18 Ch:10 Neutral S:4"
        ),
        status_line="Dlvl:1 $:0 HP:14(14) Pw:4(4) AC:4 Xp:1/0 T:2",
    )
    env = _FakeNetHackEnv(next_obs)
    adapter = NetHackAdapter(env, initial_obs, {}, render=False)

    capabilities = adapter.get_v2_capabilities()
    assert capabilities.adapter_name == "nethack"
    assert capabilities.observation_mode == "grid"
    assert capabilities.supports_spatial_context is True
    assert capabilities.supports_status_delta is True

    adapter.step("move east")
    event = adapter.build_v2_event(step_index=1)
    compact = event.to_compact_dict()

    assert event.action_result.action_text == "move east"
    assert event.action_result.operator_family == "relocation"
    assert event.reward_delta == 4.0
    assert event.status_delta["T"] == 1
    assert event.status_delta["S"] == 4
    assert event.spatial_context is not None
    assert event.spatial_context.topology == "grid"
    assert event.spatial_context.current_region == "Dlvl:1"
    assert "north" in event.spatial_context.passable_directions
    assert "east" in compact["frontier_delta"]["closed"]
    assert compact["metadata"]["variant"] == "NetHackScore-v0"
    assert compact["metadata"]["status_snapshot"]["Dlvl"] == 1
    assert any(
        candidate["family"] == "relocation"
        for candidate in compact["operator_candidates"]
    )
