import itertools
import subprocess

import pytest

from src.config import androidworld_validation


def test_wait_for_androidworld_device_ready_succeeds_after_boot(monkeypatch):
    commands = []
    outputs = iter(["", "device\n", "0\n", "device\n", "1\n"])

    def _fake_check_output(cmd, **kwargs):
        commands.append(cmd)
        return next(outputs)

    monkeypatch.setattr(subprocess, "check_output", _fake_check_output)
    monkeypatch.setattr(androidworld_validation.time, "sleep", lambda _: None)

    androidworld_validation.wait_for_androidworld_device_ready(
        adb_path="/usr/bin/adb",
        console_port=5554,
        timeout_sec=5.0,
    )

    assert commands[0][-1] == "wait-for-device"
    assert ["get-state"] == commands[1][-1:]
    assert commands[2][-3:] == ["shell", "getprop", "sys.boot_completed"]


def test_wait_for_androidworld_device_ready_raises_for_offline_device(monkeypatch):
    clock = itertools.count()

    def _fake_check_output(cmd, **kwargs):
        if cmd[-1] == "wait-for-device":
            return ""
        return "offline\n"

    monkeypatch.setattr(subprocess, "check_output", _fake_check_output)
    monkeypatch.setattr(androidworld_validation.time, "sleep", lambda _: None)
    monkeypatch.setattr(
        androidworld_validation.time,
        "monotonic",
        lambda: float(next(clock)),
    )

    with pytest.raises(RuntimeError, match="not fully ready"):
        androidworld_validation.wait_for_androidworld_device_ready(
            adb_path="/usr/bin/adb",
            console_port=5554,
            timeout_sec=3.0,
        )
