#!/usr/bin/env bash
set -euo pipefail

ACTION="${1:-up}"
WEBARENA_REPO_URL="${WEBARENA_REPO_URL:-https://github.com/web-arena-x/webarena.git}"
WEBARENA_REPO_REV="${WEBARENA_REPO_REV:-dce04686a56253aefba7b18a4fa0937cf1dc987b}"
WEBARENA_REPO_DIR="${WEBARENA_REPO_DIR:-.cache/webarena-src}"
WEBARENA_COMPOSE_FILE="${WEBARENA_COMPOSE_FILE:-}"

ensure_repo_checkout() {
  mkdir -p "$(dirname "$WEBARENA_REPO_DIR")"
  if [ ! -d "$WEBARENA_REPO_DIR/.git" ]; then
    echo "Cloning WebArena source into $WEBARENA_REPO_DIR..."
    git clone "$WEBARENA_REPO_URL" "$WEBARENA_REPO_DIR"
  fi

  git -C "$WEBARENA_REPO_DIR" fetch --depth 1 origin "$WEBARENA_REPO_REV"
  git -C "$WEBARENA_REPO_DIR" checkout --detach "$WEBARENA_REPO_REV"
}

find_compose_file() {
  if [ -n "$WEBARENA_COMPOSE_FILE" ]; then
    if [ -f "$WEBARENA_COMPOSE_FILE" ]; then
      printf '%s\n' "$WEBARENA_COMPOSE_FILE"
      return 0
    fi
    echo "WEBARENA_COMPOSE_FILE is set but does not exist: $WEBARENA_COMPOSE_FILE" >&2
    return 1
  fi

  local candidates=(
    "$WEBARENA_REPO_DIR/environment_docker/docker-compose.yml"
    "$WEBARENA_REPO_DIR/environment_docker/docker-compose.yaml"
    "$WEBARENA_REPO_DIR/docker-compose.yml"
    "$WEBARENA_REPO_DIR/docker-compose.yaml"
  )

  local compose_file
  for compose_file in "${candidates[@]}"; do
    if [ -f "$compose_file" ]; then
      printf '%s\n' "$compose_file"
      return 0
    fi
  done

  return 1
}

print_upstream_layout_message() {
  cat >&2 <<EOF
The pinned upstream WebArena checkout does not include a turnkey docker-compose stack.

What the upstream repo actually ships at $WEBARENA_REPO_DIR:
- setup docs in environment_docker/README.md
- the homepage app in environment_docker/webarena-homepage/
- instructions for loading or starting the individual site containers/images

To use this helper, do one of the following:
1. Set WEBARENA_COMPOSE_FILE to your own WebArena site-stack compose file, then rerun:
   WEBARENA_COMPOSE_FILE=/abs/path/to/docker-compose.yml make webarena-up
2. Provision the upstream sites manually by following:
   $WEBARENA_REPO_DIR/environment_docker/README.md
   Then run:
   make webarena-check

The upstream docs expect services for these ports:
- 7770 shopping
- 7780 shopping_admin
- 9999 reddit/forum
- 8023 gitlab
- 8888 wikipedia
- 3000 map
- 4399 homepage
EOF
}

resolve_compose_file() {
  if COMPOSE_FILE="$(find_compose_file)"; then
    printf '%s\n' "$COMPOSE_FILE"
    return 0
  fi

  print_upstream_layout_message
  return 1
}

ensure_repo_checkout
COMPOSE_FILE="$(resolve_compose_file)" || exit 1

case "$ACTION" in
  up)
    docker compose -f "$COMPOSE_FILE" up -d
    ;;
  down)
    docker compose -f "$COMPOSE_FILE" down
    ;;
  ps|status)
    docker compose -f "$COMPOSE_FILE" ps
    ;;
  logs)
    docker compose -f "$COMPOSE_FILE" logs --tail=200
    ;;
  *)
    echo "Usage: $0 {up|down|ps|status|logs}" >&2
    exit 1
    ;;
esac
