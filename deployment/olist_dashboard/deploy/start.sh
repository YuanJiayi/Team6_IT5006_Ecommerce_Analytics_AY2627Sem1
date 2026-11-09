#!/bin/sh
# Azure startup for the complete dashboard. Mirrors run_local.sh: an MLflow
# server with artifact serving in front of the shipped store, then the
# dashboard. The app folder is on network storage, which makes SQLite reads
# slow enough to hit Ask's 2 s query limit, so the database is served from a
# local-disk copy. Fails loudly if the store is missing or MLflow never
# becomes healthy.
set -eu
HERE="$(cd "$(dirname "$0")" && pwd)"
APP_DIR="$(cd "$HERE/.." && pwd)"
ROOT="$(cd "$APP_DIR/../.." && pwd)"

DB="$APP_DIR/mlflow/mlflow.db"
if [ ! -f "$DB" ]; then
  echo "The MLflow store is missing at $DB" >&2
  exit 1
fi
if ! python3 -c 'import mlflow' >/dev/null 2>&1; then
  echo "MLflow is not installed for python3; the deploy build failed to install requirements.txt" >&2
  exit 1
fi

cp "$APP_DIR/data/analytics.sqlite" /tmp/analytics.sqlite.partial
mv /tmp/analytics.sqlite.partial /tmp/analytics.sqlite
export VAMOS_ANALYTICS_DB=/tmp/analytics.sqlite

rm -rf /tmp/mlflow.partial /tmp/mlflow
cp -R "$APP_DIR/mlflow" /tmp/mlflow.partial
mv /tmp/mlflow.partial /tmp/mlflow
DB="/tmp/mlflow/mlflow.db"

# Evidently must never phone home from the dashboard, and the read-only MLflow
# server does not run background Huey job workers.
export EVIDENTLY_DISABLE_TELEMETRY=1
export MLFLOW_DISABLE_AGENT_HINT=1
export MLFLOW_SERVER_ENABLE_JOB_EXECUTION=false
export PYTHONPATH="$ROOT/src${PYTHONPATH:+:$PYTHONPATH}"

DASHBOARD_PORT="${PORT:-8000}"
MLFLOW_PORT="${PRAZO_MLFLOW_PORT:-5050}"
UPSTREAM="http://127.0.0.1:$MLFLOW_PORT"

MLFLOW_PID=""
cleanup() {
  if [ -n "$MLFLOW_PID" ] && kill -0 "$MLFLOW_PID" 2>/dev/null; then
    kill "$MLFLOW_PID" 2>/dev/null || true
    wait "$MLFLOW_PID" 2>/dev/null || true
  fi
}
trap cleanup EXIT INT TERM

(
  cd /tmp/mlflow
  exec python3 -m mlflow server \
    --backend-store-uri "sqlite:///$DB" \
    --serve-artifacts \
    --artifacts-destination "/tmp/mlflow/artifacts" \
    --static-prefix /mlflow \
    --host 127.0.0.1 \
    --port "$MLFLOW_PORT" \
    --workers 1
) &
MLFLOW_PID=$!

# Network storage makes the first cold start slow; allow two minutes.
echo "Waiting for MLflow at $UPSTREAM/mlflow/ ..."
HEALTHY=""
for _ in $(seq 1 120); do
  if ! kill -0 "$MLFLOW_PID" 2>/dev/null; then
    break
  fi
  if python3 -c "import urllib.request; urllib.request.urlopen('$UPSTREAM/mlflow/health', timeout=1)" >/dev/null 2>&1; then
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

PRAZO_MLFLOW_DB="$DB" PRAZO_MLFLOW_UI="$UPSTREAM" \
  exec python3 "$APP_DIR/server.py" --host 0.0.0.0 --port "$DASHBOARD_PORT"
