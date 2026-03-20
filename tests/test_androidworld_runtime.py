import subprocess

from src.config import androidworld_runtime


def test_androidworld_install_timeout_from_env(monkeypatch):
    monkeypatch.setenv("ANDROIDWORLD_ADB_INSTALL_TIMEOUT", "900")
    assert androidworld_runtime.androidworld_install_timeout_from_env() == 900.0


def test_androidworld_install_timeout_from_env_ignores_bad_values(monkeypatch):
    monkeypatch.setenv("ANDROIDWORLD_ADB_INSTALL_TIMEOUT", "not-a-number")
    assert androidworld_runtime.androidworld_install_timeout_from_env() == 600.0


def test_adb_package_is_installed_detects_package(monkeypatch):
    monkeypatch.setattr(
        subprocess,
        "check_output",
        lambda *args, **kwargs: "package:/data/app/com.example/base.apk\n",
    )
    assert androidworld_runtime._adb_package_is_installed(
        adb_path="/usr/bin/adb",
        adb_server_port=5037,
        device_name="emulator-5554",
        package_name="com.example",
    )


def test_adb_package_is_installed_handles_errors(monkeypatch):
    def _raise(*args, **kwargs):
        raise subprocess.TimeoutExpired(cmd=["adb"], timeout=5)

    monkeypatch.setattr(subprocess, "check_output", _raise)
    assert not androidworld_runtime._adb_package_is_installed(
        adb_path="/usr/bin/adb",
        adb_server_port=5037,
        device_name="emulator-5554",
        package_name="com.example",
    )
