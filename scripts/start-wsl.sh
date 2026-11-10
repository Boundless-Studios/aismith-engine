#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
ENV_FILE=${AISMITH_ENV_FILE:-$ROOT/.env}
RUNTIME_DIR_FILE=${AISMITH_RUNTIME_DIR_FILE:-$ROOT/.aismith-runtime-dir}
INSTALL=false

if [[ ${1:-} == "--install" ]]; then
  INSTALL=true
elif [[ $# -gt 0 ]]; then
  echo "Usage: bash scripts/start-wsl.sh [--install]" >&2
  exit 2
fi

if ! grep -qi microsoft /proc/version 2>/dev/null; then
  echo "AISmith's WSL launcher must run inside WSL2." >&2
  exit 1
fi

load_env() {
  [[ -f "$ENV_FILE" ]] || return 0
  while IFS='=' read -r key value; do
    key=${key//[[:space:]]/}
    [[ -n "$key" && ${key:0:1} != "#" ]] || continue
    value=${value%$'\r'}
    value=${value#\"}
    value=${value%\"}
    case "$key" in
      AISMITH_API_HOST|AISMITH_API_PORT|AISMITH_UI_HOST|AISMITH_UI_PORT|AISMITH_DATA_DIR|AISMITH_MODELS_DIR|AISMITH_CORS_ORIGINS|BLENDER_EXE|TRELLIS_GGUF_MODEL_DIR|TRELLIS_REFINER_DIR|KIMODO_DIR|KIMODO_EXECUTABLE|KIMODO_MODEL|KIMODO_TEXT_ENCODER_DEVICE|PYTORCH_INDEX_URL)
        printf -v "$key" '%s' "$value"
        export "${key?}"
        ;;
    esac
  done < "$ENV_FILE"
}

load_env

# Loopback by default: the API has no authentication. WSL2 forwards Windows'
# localhost to loopback listeners, so Windows still reaches it.
AISMITH_API_HOST=${AISMITH_API_HOST:-127.0.0.1}
AISMITH_API_PORT=${AISMITH_API_PORT:-8000}
AISMITH_UI_HOST=${AISMITH_UI_HOST:-127.0.0.1}
AISMITH_UI_PORT=${AISMITH_UI_PORT:-5173}
AISMITH_DATA_DIR=${AISMITH_DATA_DIR:-$ROOT/studio-data}
[[ "$AISMITH_DATA_DIR" == /* ]] || AISMITH_DATA_DIR="$ROOT/$AISMITH_DATA_DIR"
export AISMITH_DATA_DIR

if [[ -z ${BLENDER_EXE:-} ]]; then
  BLENDER_EXE=$(command -v blender || true)
  if [[ -z "$BLENDER_EXE" ]]; then
    BLENDER_EXE=$(find "$HOME/tools" -maxdepth 2 -type f -path '*/blender-*-linux-x64/blender' 2>/dev/null | sort -V | tail -1)
  fi
  [[ -z "$BLENDER_EXE" ]] || export BLENDER_EXE
fi

if [[ -d /usr/lib/wsl/lib ]]; then
  PATH="/usr/lib/wsl/lib:$PATH"
  export PATH
fi

if $INSTALL; then
  [[ -x "$ROOT/.venv/bin/python" ]] || python3 -m venv "$ROOT/.venv"
  "$ROOT/.venv/bin/python" -m pip install -r "$ROOT/requirements.txt"
  npm --prefix "$ROOT/frontend" ci
fi

PYTHON="$ROOT/.venv/bin/python"
[[ -x "$PYTHON" ]] || { echo "Missing .venv; rerun with --install." >&2; exit 1; }
[[ -d "$ROOT/frontend/node_modules" ]] || { echo "Missing frontend dependencies; rerun with --install." >&2; exit 1; }
"$PYTHON" -m pip show fastapi uvicorn >/dev/null
command -v npm >/dev/null

if [[ -f "$RUNTIME_DIR_FILE" ]]; then
  PREVIOUS_DATA_DIR=$(<"$RUNTIME_DIR_FILE")
  if [[ -n "$PREVIOUS_DATA_DIR" && "$PREVIOUS_DATA_DIR" != "$AISMITH_DATA_DIR" ]]; then
    AISMITH_DATA_DIR="$PREVIOUS_DATA_DIR" bash "$ROOT/scripts/stop-wsl.sh" --quiet
  fi
fi
mkdir -p "$AISMITH_DATA_DIR/runtime" "$AISMITH_DATA_DIR/logs"
bash "$ROOT/scripts/stop-wsl.sh" --quiet
printf '%s\n' "$AISMITH_DATA_DIR" > "$RUNTIME_DIR_FILE"

port_is_busy() {
  ss -H -ltn | awk '{print $4}' | grep -Eq "(^|:)$1$"
}

if port_is_busy "$AISMITH_API_PORT"; then
  echo "API port $AISMITH_API_PORT is already in use by another process." >&2
  exit 1
fi
if port_is_busy "$AISMITH_UI_PORT"; then
  echo "UI port $AISMITH_UI_PORT is already in use by another process." >&2
  exit 1
fi

WSL_IP=$(hostname -I | awk '{print $1}')
DEFAULT_ORIGINS="http://localhost:$AISMITH_UI_PORT,http://127.0.0.1:$AISMITH_UI_PORT"
if [[ -n "$WSL_IP" ]]; then
  DEFAULT_ORIGINS="$DEFAULT_ORIGINS,http://$WSL_IP:$AISMITH_UI_PORT"
fi
AISMITH_CORS_ORIGINS=${AISMITH_CORS_ORIGINS:+$AISMITH_CORS_ORIGINS,}$DEFAULT_ORIGINS
export AISMITH_CORS_ORIGINS

# The address to print: Windows reaches a loopback bind through localhost.
case "$AISMITH_UI_HOST" in
  127.*|localhost|::1) UI_ADDRESS=localhost ;;
  *) UI_ADDRESS=${WSL_IP:-127.0.0.1} ;;
esac

cd "$ROOT"
nohup "$PYTHON" -m uvicorn backend.app.main:app --host "$AISMITH_API_HOST" --port "$AISMITH_API_PORT" \
  </dev/null >"$AISMITH_DATA_DIR/logs/api.log" 2>&1 &
API_PID=$!
printf '%s\n' "$API_PID" > "$AISMITH_DATA_DIR/runtime/api.pid"

VITE_API_PORT="$AISMITH_API_PORT" nohup "$ROOT/frontend/node_modules/.bin/vite" "$ROOT/frontend" --host "$AISMITH_UI_HOST" --port "$AISMITH_UI_PORT" \
  </dev/null >"$AISMITH_DATA_DIR/logs/ui.log" 2>&1 &
UI_PID=$!
printf '%s\n' "$UI_PID" > "$AISMITH_DATA_DIR/runtime/ui.pid"

cleanup_failed_start() {
  bash "$ROOT/scripts/stop-wsl.sh" --quiet
  echo "AISmith failed to start. See $AISMITH_DATA_DIR/logs/." >&2
  exit 1
}

for _ in {1..30}; do
  if curl -fsS "http://127.0.0.1:$AISMITH_API_PORT/api/health" >/dev/null 2>&1 && \
     curl -fsS "http://127.0.0.1:$AISMITH_UI_PORT" >/dev/null 2>&1; then
    echo "AISmith API: http://127.0.0.1:$AISMITH_API_PORT"
    echo "AISmith UI:  http://$UI_ADDRESS:$AISMITH_UI_PORT"
    echo "State:       $AISMITH_DATA_DIR"
    exit 0
  fi
  kill -0 "$API_PID" 2>/dev/null || cleanup_failed_start
  kill -0 "$UI_PID" 2>/dev/null || cleanup_failed_start
  sleep 1
done

cleanup_failed_start
