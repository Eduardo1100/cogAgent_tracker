"""Tests for _parse_surroundings in NetHackAdapter."""

import numpy as np

from src.agent.env_adapter import _parse_surroundings


def _make_tty(lines: list[str], player_row: int = 5, player_col: int = 40) -> np.ndarray:
    """Build a 24x80 TTY character grid from human-readable lines.

    *lines* are placed starting at row ``player_row - len(lines)//2`` so the
    player glyph (``@``) ends up near the given row/col.  Unspecified cells
    default to space (0x20).
    """
    grid = np.full((24, 80), ord(" "), dtype=np.int32)
    start_row = player_row - len(lines) // 2
    for i, line in enumerate(lines):
        r = start_row + i
        if 0 <= r < 24:
            for j, ch in enumerate(line):
                c = player_col - line.index("@") + j if "@" in line else j
                if 0 <= c < 80:
                    grid[r][c] = ord(ch)
    return grid


def _simple_room_tty() -> np.ndarray:
    """Reproduce the starting room from experiment 207."""
    room = [
        "-----",
        "|...|",
        "....|",
        "|.@<|",
        "|.f.|",
        "|...|",
        "---.-",
    ]
    grid = np.full((24, 80), ord(" "), dtype=np.int32)
    top_row, left_col = 3, 60
    for i, line in enumerate(room):
        for j, ch in enumerate(line):
            grid[top_row + i][left_col + j] = ord(ch)
    return grid


class TestParseSurroundings:
    def test_returns_none_when_no_player(self):
        grid = np.full((24, 80), ord(" "), dtype=np.int32)
        assert _parse_surroundings(grid) is None

    def test_starting_room_identifies_passable_directions(self):
        grid = _simple_room_tty()
        result = _parse_surroundings(grid)
        assert result is not None
        assert "[Surroundings]" in result
        # Player @ is at row 6, col 62.  South is 'f' (kitten), east is '<'
        assert "north: floor" in result
        assert "south: creature 'f'" in result
        assert "east: upstairs" in result
        assert "west: floor" in result
        assert "passable:" in result

    def test_wall_directions_are_blocked(self):
        # Put player next to a wall
        grid = np.full((24, 80), ord(" "), dtype=np.int32)
        grid[5][40] = ord("@")
        grid[5][41] = ord("|")
        grid[5][39] = ord(".")
        result = _parse_surroundings(grid)
        assert result is not None
        assert "blocked:" in result
        assert "east: wall" in result

    def test_creature_detection(self):
        grid = _simple_room_tty()
        result = _parse_surroundings(grid)
        assert result is not None
        # The 'f' (kitten) is directly south of the player
        assert "south: creature 'f'" in result

    def test_all_eight_directions_present(self):
        grid = _simple_room_tty()
        result = _parse_surroundings(grid)
        assert result is not None
        for direction in [
            "north", "south", "east", "west",
            "northeast", "northwest", "southeast", "southwest",
        ]:
            assert direction in result
