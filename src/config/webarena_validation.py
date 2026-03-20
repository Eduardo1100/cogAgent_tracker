import os
import socket
from pathlib import Path
from urllib.parse import urlparse

from src.config.env_validation import require_env_vars

_PW_EXTRA_HEADERS_PLACEHOLDER = "/absolute/path/to/extra_headers.json"
_WEBARENA_ENV_VARS = [
    "WA_SHOPPING",
    "WA_SHOPPING_ADMIN",
    "WA_REDDIT",
    "WA_GITLAB",
    "WA_WIKIPEDIA",
    "WA_MAP",
    "WA_HOMEPAGE",
]


def running_inside_container() -> bool:
    return Path("/.dockerenv").exists()


def normalize_pw_extra_headers_env() -> None:
    extra_headers = (os.getenv("PW_EXTRA_HEADERS") or "").strip()
    if extra_headers == _PW_EXTRA_HEADERS_PLACEHOLDER:
        os.environ.pop("PW_EXTRA_HEADERS", None)


def validate_webarena_instance_urls() -> None:
    normalize_pw_extra_headers_env()
    require_env_vars(_WEBARENA_ENV_VARS, context="WebArena evaluation")

    inside_container = running_inside_container()
    localhost_bindings: list[str] = []
    unreachable_bindings: list[str] = []

    for env_var in _WEBARENA_ENV_VARS:
        raw_url = os.environ[env_var].strip()
        parsed = urlparse(raw_url)
        hostname = (parsed.hostname or "").strip().lower()
        port = parsed.port

        if not hostname or not parsed.scheme:
            raise RuntimeError(
                f"{env_var} must be a full URL like http://host:port, got {raw_url!r}."
            )

        if inside_container and hostname in {"localhost", "127.0.0.1", "::1"}:
            localhost_bindings.append(f"{env_var}={raw_url}")
            continue

        if port is None:
            port = 443 if parsed.scheme == "https" else 80

        try:
            with socket.create_connection((hostname, port), timeout=2.0):
                pass
        except OSError as exc:
            unreachable_bindings.append(f"{env_var}={raw_url} ({exc})")

    if localhost_bindings:
        bindings = "\n".join(f"  - {binding}" for binding in localhost_bindings)
        raise RuntimeError(
            "WebArena URLs are pointing at localhost from inside the app container.\n"
            "Inside Docker, localhost refers to the container itself, not your host.\n"
            "Use host.docker.internal for host-run WebArena services, or a Docker "
            "service hostname if the sites run in containers.\n"
            f"{bindings}"
        )

    if unreachable_bindings:
        bindings = "\n".join(f"  - {binding}" for binding in unreachable_bindings)
        raise RuntimeError(
            "WebArena instance URLs are configured but not reachable from the app "
            "container.\n"
            f"{bindings}"
        )
