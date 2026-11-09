#!/usr/bin/env bash
set -euo pipefail

# Evidently must never phone home from the dashboard, and the read-only MLflow
# server does not run background Huey job workers.
export EVIDENTLY_DISABLE_TELEMETRY="${EVIDENTLY_DISABLE_TELEMETRY:-1}"
export MLFLOW_DISABLE_AGENT_HINT="${MLFLOW_DISABLE_AGENT_HINT:-1}"
export MLFLOW_SERVER_ENABLE_JOB_EXECUTION="${MLFLOW_SERVER_ENABLE_JOB_EXECUTION:-false}"

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$HERE/../.." && pwd)"
DB="${PRAZO_MLFLOW_DB:-$HERE/mlflow/mlflow.db}"
ARTIFACTS="$(dirname "$DB")/artifacts"
DASHBOARD_PORT="${PRAZO_PORT:-8000}"
MLFLOW_PORT="${PRAZO_MLFLOW_PORT:-5050}"
UPSTREAM="http://127.0.0.1:$MLFLOW_PORT"

find_python() {
  if [ -n "${PRAZO_PYTHON:-}" ]; then
    printf '%s\n' "$PRAZO_PYTHON"
    return
  fi
  local candidate
  for candidate in "$ROOT/it5006-proj/bin/python" "$ROOT/../it5006-proj/bin/python"; do
    if [ -x "$candidate" ]; then
      printf '%s\n' "$candidate"
      return
    fi
  done
  local common main
  common="$(git -C "$ROOT" rev-parse --git-common-dir 2>/dev/null || true)"
  if [ -n "$common" ]; then
    main="$(cd "$(dirname "$common")" && pwd)"
    candidate="$main/it5006-proj/bin/python"
    if [ -x "$candidate" ]; then
      printf '%s\n' "$candidate"
      return
    fi
  fi
  command -v python3
}

export PYTHONPATH="$ROOT/src${PYTHONPATH:+:$PYTHONPATH}"

PYTHON="$(find_python)"
if [ ! -f "$DB" ]; then
  echo "The MLflow store is missing at $DB" >&2
  echo "Build it from the repository root with: $PYTHON -m phase3.registry" >&2
  exit 1
fi
if ! "$PYTHON" -c 'import mlflow' >/dev/null 2>&1; then
  echo "MLflow is not installed for $PYTHON" >&2
  echo "Set PRAZO_PYTHON to the project environment's Python, for example:" >&2
  echo "  PRAZO_PYTHON=/path/to/it5006-proj/bin/python $0" >&2
  exit 1
fi

MLFLOW_PID=""
cleanup() {
  if [ -n "$MLFLOW_PID" ] && kill -0 "$MLFLOW_PID" 2>/dev/null; then
    kill "$MLFLOW_PID" 2>/dev/null || true
    wait "$MLFLOW_PID" 2>/dev/null || true
  fi
}
trap cleanup EXIT INT TERM

(
  cd "$(dirname "$DB")"
  exec "$PYTHON" -m mlflow server \
    --backend-store-uri "sqlite:///$DB" \
    --serve-artifacts \
    --artifacts-destination "$ARTIFACTS" \
    --static-prefix /mlflow \
    --host 127.0.0.1 \
    --port "$MLFLOW_PORT"
) &
MLFLOW_PID=$!

echo "Waiting for MLflow at $UPSTREAM/mlflow/ ..."
HEALTHY=""
for _ in $(seq 1 60); do
  if ! kill -0 "$MLFLOW_PID" 2>/dev/null; then
    break
  fi
  if "$PYTHON" -c "import urllib.request; urllib.request.urlopen('$UPSTREAM/mlflow/health', timeout=1)" >/dev/null 2>&1; then
    HEALTHY=1
    break
  fi
  sleep 1
done
if [ -z "$HEALTHY" ]; then
  echo "MLflow did not become healthy at $UPSTREAM/mlflow/health." >&2
  echo "Check the MLflow output above for the reason." >&2
  exit 1
fi

echo "MLflow UI: $UPSTREAM/mlflow/"
echo "Dashboard: http://127.0.0.1:$DASHBOARD_PORT/"
PRAZO_MLFLOW_DB="$DB" PRAZO_MLFLOW_UI="$UPSTREAM" \
  "$PYTHON" "$HERE/server.py" --host 127.0.0.1 --port "$DASHBOARD_PORT"
