import argparse

from src.config.androidworld_validation import validate_androidworld_runtime


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--adb-path", default=None)
    parser.add_argument("--console-port", type=int, default=5554)
    args = parser.parse_args()

    resolved_adb_path = validate_androidworld_runtime(
        adb_path=args.adb_path,
        console_port=args.console_port,
    )
    print(
        "AndroidWorld runtime looks reachable."
        f" adb={resolved_adb_path} console_port={args.console_port}"
    )


if __name__ == "__main__":
    main()
