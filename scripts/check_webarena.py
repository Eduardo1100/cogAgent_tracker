from src.config.webarena_validation import validate_webarena_instance_urls


def main() -> None:
    validate_webarena_instance_urls()
    print("WebArena instance URLs are reachable.")


if __name__ == "__main__":
    main()
