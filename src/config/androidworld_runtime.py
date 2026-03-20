import os
import subprocess
import tempfile
from pathlib import Path
from typing import Any

from absl import logging

_A11Y_FORWARDER_PACKAGE = "com.google.androidenv.accessibilityforwarder"
_RUNTIME_PATCHED = False
_PATCHED_INSTALL_TIMEOUT_SEC = 600.0


def _get_env_adb_config(env: Any) -> tuple[str, int, str]:
    adb_config = env._coordinator._simulator._config.adb_controller  # noqa: SLF001
    emulator_config = env._coordinator._simulator._config.emulator_launcher  # noqa: SLF001
    adb_path = adb_config.adb_path
    adb_server_port = int(adb_config.adb_server_port)
    device_name = (
        adb_config.device_name or f"emulator-{emulator_config.emulator_console_port}"
    )
    return adb_path, adb_server_port, device_name


def _adb_package_is_installed(
    *,
    adb_path: str,
    adb_server_port: int,
    device_name: str,
    package_name: str,
    timeout_sec: float = 20.0,
) -> bool:
    try:
        output = subprocess.check_output(
            [
                adb_path,
                "-P",
                str(adb_server_port),
                "-s",
                device_name,
                "shell",
                "pm",
                "path",
                package_name,
            ],
            stderr=subprocess.STDOUT,
            text=True,
            timeout=timeout_sec,
        )
    except Exception:
        return False
    return output.strip().startswith("package:")


def _install_forwarder_via_adb(
    *,
    apk_bytes: bytes,
    adb_path: str,
    adb_server_port: int,
    device_name: str,
    timeout_sec: float,
) -> None:
    temp_path: str | None = None
    try:
        with tempfile.NamedTemporaryFile(suffix=".apk", delete=False) as handle:
            handle.write(apk_bytes)
            temp_path = handle.name
        subprocess.check_output(
            [
                adb_path,
                "-P",
                str(adb_server_port),
                "-s",
                device_name,
                "install",
                "-r",
                "-t",
                "-g",
                temp_path,
            ],
            stderr=subprocess.STDOUT,
            text=True,
            timeout=timeout_sec,
        )
    finally:
        if temp_path:
            Path(temp_path).unlink(missing_ok=True)


def prepare_androidworld_runtime(*, install_timeout_sec: float = 600.0) -> None:
    global _RUNTIME_PATCHED, _PATCHED_INSTALL_TIMEOUT_SEC
    _PATCHED_INSTALL_TIMEOUT_SEC = install_timeout_sec
    if _RUNTIME_PATCHED:
        return

    from android_env.components import config_classes
    from android_env.wrappers import a11y_grpc_wrapper
    from android_world.env import android_world_controller

    original_get_controller = android_world_controller.get_controller
    original_install_forwarder = (
        a11y_grpc_wrapper.A11yGrpcWrapper._install_a11y_forwarding_apk
    )

    def patched_install_a11y_forwarding_apk(self) -> None:
        adb_path, adb_server_port, device_name = _get_env_adb_config(self._env)
        if _adb_package_is_installed(
            adb_path=adb_path,
            adb_server_port=adb_server_port,
            device_name=device_name,
            package_name=_A11Y_FORWARDER_PACKAGE,
        ):
            logging.info(
                "Accessibility forwarder already installed on %s; skipping reinstall.",
                device_name,
            )
            return
        logging.info(
            "Installing accessibility forwarder on %s with extended timeout %.1fs.",
            device_name,
            _PATCHED_INSTALL_TIMEOUT_SEC,
        )
        apk_bytes = a11y_grpc_wrapper._get_accessibility_forwarder_apk()
        _install_forwarder_via_adb(
            apk_bytes=apk_bytes,
            adb_path=adb_path,
            adb_server_port=adb_server_port,
            device_name=device_name,
            timeout_sec=_PATCHED_INSTALL_TIMEOUT_SEC,
        )

    def patched_get_controller(
        console_port: int = 5554,
        adb_path: str = android_world_controller.DEFAULT_ADB_PATH,
        grpc_port: int = 8554,
    ):
        config = config_classes.AndroidEnvConfig(
            task=config_classes.FilesystemTaskConfig(
                path=android_world_controller._write_default_task_proto()  # noqa: SLF001
            ),
            simulator=config_classes.EmulatorConfig(
                emulator_launcher=config_classes.EmulatorLauncherConfig(
                    emulator_console_port=console_port,
                    adb_port=console_port + 1,
                    grpc_port=grpc_port,
                ),
                adb_controller=config_classes.AdbControllerConfig(
                    adb_path=adb_path,
                    default_timeout=max(120.0, _PATCHED_INSTALL_TIMEOUT_SEC),
                    device_name=f"emulator-{console_port}",
                ),
            ),
        )
        android_env_instance = android_world_controller.loader.load(config)
        logging.info("Setting up AndroidWorldController.")
        return android_world_controller.AndroidWorldController(android_env_instance)

    a11y_grpc_wrapper.A11yGrpcWrapper._install_a11y_forwarding_apk = (
        patched_install_a11y_forwarding_apk
    )
    android_world_controller.get_controller = patched_get_controller

    # Keep references on module globals for debugging if needed.
    prepare_androidworld_runtime._original_get_controller = original_get_controller  # type: ignore[attr-defined]
    prepare_androidworld_runtime._original_install_forwarder = (
        original_install_forwarder  # type: ignore[attr-defined]
    )
    _RUNTIME_PATCHED = True


def androidworld_install_timeout_from_env(default: float = 600.0) -> float:
    raw = os.getenv("ANDROIDWORLD_ADB_INSTALL_TIMEOUT")
    if not raw:
        return default
    try:
        return max(120.0, float(raw))
    except ValueError:
        return default
