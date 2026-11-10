#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
ENV_FILE=${AISMITH_ENV_FILE:-$ROOT/.env}
RUNTIME_DIR_FILE=${AISMITH_RUNTIME_DIR_FILE:-$ROOT/.aismith-runtime-dir}
QUIET=false
PRINT_DATA_DIR=false
case ${1:-} in
  --quiet) QUIET=true ;;
  --print-data-dir) PRINT_DATA_DIR=true ;;
  "") ;;
  *) echo "Usage: bash scripts/stop-wsl.sh [--quiet|--print-data-dir]" >&2; exit 2 ;;
esac

DATA_DIR_CONFIGURED=${AISMITH_DATA_DIR:+true}
if [[ -z "$DATA_DIR_CONFIGURED" && -f "$ENV_FILE" ]]; then
  while IFS='=' read -r key value; do
    key=${key//[[:space:]]/}
    [[ "$key" == "AISMITH_DATA_DIR" ]] || continue
    value=${value%$'\r'}
    value=${value#\"}
    AISMITH_DATA_DIR=${value%\"}
    DATA_DIR_CONFIGURED=true
  done < "$ENV_FILE"
fi
AISMITH_DATA_DIR=${AISMITH_DATA_DIR:-$ROOT/studio-data}
if [[ -z "$DATA_DIR_CONFIGURED" && -f "$RUNTIME_DIR_FILE" ]]; then
  AISMITH_DATA_DIR=$(<"$RUNTIME_DIR_FILE")
fi
[[ "$AISMITH_DATA_DIR" == /* ]] || AISMITH_DATA_DIR="$ROOT/$AISMITH_DATA_DIR"

if $PRINT_DATA_DIR; then
  printf '%s\n' "$AISMITH_DATA_DIR"
  exit 0
fi

stop_owned_process() {
  local name=$1
  local expected=$2
  local pid_file="$AISMITH_DATA_DIR/runtime/$name.pid"
  [[ -f "$pid_file" ]] || return 0
  local pid
  pid=$(<"$pid_file")
  if [[ "$pid" =~ ^[0-9]+$ && -r "/proc/$pid/cmdline" ]]; then
    local command
    command=$(tr '\0' ' ' < "/proc/$pid/cmdline")
    if [[ "$command" == *"$expected"* ]]; then
      kill "$pid"
      $QUIET || echo "Stopped AISmith $name (PID $pid)."
      for _ in {1..50}; do
        kill -0 "$pid" 2>/dev/null || break
        sleep 0.1
      done
    else
      $QUIET || echo "Ignored stale $name PID $pid; it is not an AISmith process."
    fi
  fi
  rm -f "$pid_file"
}

stop_owned_process api "backend.app.main:app"
stop_owned_process ui "/vite"
