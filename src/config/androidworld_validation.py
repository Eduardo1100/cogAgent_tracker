import os
import shutil
import subprocess
import time
from pathlib import Path

_DEFAULT_ADB_PATHS = [
    Path("~/Library/Android/sdk/platform-tools/adb").expanduser(),
    Path("~/Android/Sdk/platform-tools/adb").expanduser(),
]
_ADB_SERVER_PORT = 5037
_READY_TIMEOUT_SEC = 180.0
_READY_POLL_INTERVAL_SEC = 2.0


def resolve_androidworld_adb_path(requested_path: str | None = None) -> str:
    candidates: list[Path] = []
    if requested_path:
        candidates.append(Path(requested_path).expanduser())
    env_path = os.getenv("ANDROIDWORLD_ADB_PATH")
    if env_path:
        candidates.append(Path(env_path).expanduser())
    which_adb = shutil.which("adb")
    if which_adb:
        candidates.append(Path(which_adb).expanduser())
    candidates.extend(_DEFAULT_ADB_PATHS)

    for candidate in candidates:
        if candidate.is_file():
            return str(candidate)
    checked = "\n".join(f"  - {candidate}" for candidate in candidates)
    raise RuntimeError(
        "Could not locate adb for AndroidWorld.\n"
        "Set ANDROIDWORLD_ADB_PATH or pass --androidworld-adb-path.\n"
        f"Checked:\n{checked}"
    )


def _run_emulator_adb_command(
    adb_path: str,
    console_port: int,
    *args: str,
    timeout_sec: float,
) -> str:
    return subprocess.check_output(
        [
            adb_path,
            "-P",
            str(_ADB_SERVER_PORT),
            "-s",
            f"emulator-{console_port}",
            *args,
        ],
        text=True,
        stderr=subprocess.STDOUT,
        timeout=timeout_sec,
    )


def wait_for_androidworld_device_ready(
    *,
    adb_path: str,
    console_port: int = 5554,
    timeout_sec: float = _READY_TIMEOUT_SEC,
) -> None:
    expected_device = f"emulator-{console_port}"
    deadline = time.monotonic() + timeout_sec
    last_status = "device readiness checks did not complete"

    try:
        _run_emulator_adb_command(
            adb_path,
            console_port,
            "wait-for-device",
            timeout_sec=max(1.0, timeout_sec),
        )
    except Exception as exc:
        raise RuntimeError(
            "AndroidWorld emulator is visible to adb but never became available.\n"
            f"Expected device: {expected_device}\n"
            f"adb error: {exc}"
        ) from exc

    while time.monotonic() < deadline:
        remaining = max(1.0, deadline - time.monotonic())
        try:
            state = _run_emulator_adb_command(
                adb_path,
                console_port,
                "get-state",
                timeout_sec=min(10.0, remaining),
            ).strip()
            if state != "device":
                last_status = f"adb get-state returned {state!r}"
                time.sleep(_READY_POLL_INTERVAL_SEC)
                continue

            boot_completed = _run_emulator_adb_command(
                adb_path,
                console_port,
                "shell",
                "getprop",
                "sys.boot_completed",
                timeout_sec=min(10.0, remaining),
            ).strip()
            if boot_completed == "1":
                return
            last_status = f"sys.boot_completed returned {boot_completed!r}"
        except subprocess.CalledProcessError as exc:
            details = (exc.output or "").strip() or str(exc)
            last_status = f"adb command failed: {details}"
        except subprocess.TimeoutExpired as exc:
            last_status = f"adb command timed out after {exc.timeout}s"

        time.sleep(_READY_POLL_INTERVAL_SEC)

    raise RuntimeError(
        "AndroidWorld emulator is visible to adb but not fully ready.\n"
        f"Expected device: {expected_device}\n"
        f"Last readiness check: {last_status}\n"
        "Wait for the emulator to finish booting and confirm `adb devices` shows "
        "`device`, not `offline`."
    )


def validate_androidworld_runtime(
    *,
    adb_path: str | None = None,
    console_port: int = 5554,
) -> str:
    try:
        __import__("android_world")
    except ImportError as exc:
        raise RuntimeError(
            "AndroidWorld is not installed in the current Python environment. "
            "Run `bash scripts/bootstrap_androidworld.sh` first."
        ) from exc

    resolved_adb_path = resolve_androidworld_adb_path(adb_path)
    try:
        output = subprocess.check_output(
            [resolved_adb_path, "devices"],
            text=True,
            stderr=subprocess.STDOUT,
        )
    except Exception as exc:
        raise RuntimeError(
            f"Failed to query adb devices via {resolved_adb_path}."
        ) from exc

    expected_device = f"emulator-{console_port}"
    if expected_device not in output:
        raise RuntimeError(
            "AndroidWorld emulator is not visible to adb.\n"
            f"Expected device: {expected_device}\n"
            "Current `adb devices` output:\n"
            f"{output.strip()}"
        )
    wait_for_androidworld_device_ready(
        adb_path=resolved_adb_path,
        console_port=console_port,
    )
    return resolved_adb_path
