import os


def require_env_vars(variable_names: list[str], *, context: str) -> None:
    missing = [name for name in variable_names if not os.getenv(name)]
    if not missing:
        return

    missing_list = ", ".join(sorted(missing))
    raise RuntimeError(f"Missing required environment variables for {context}: {missing_list}")
