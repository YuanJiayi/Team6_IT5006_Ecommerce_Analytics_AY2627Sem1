#!/usr/bin/env bash
# Host the dashboard on Azure App Service (Linux, built-in Python, no container).
#   bash deployment/olist_dashboard/deploy/azure.sh create   # one-off: resource group, plan, web app
#   bash deployment/olist_dashboard/deploy/azure.sh deploy   # stage, upload settings and code (rerun after edits)
#   bash deployment/olist_dashboard/deploy/azure.sh size B1  # change machine size (F1 free, B1 always on); link unchanged
#   bash deployment/olist_dashboard/deploy/azure.sh delete   # remove everything; billing stops
# Needs `az login`. The app link is tied to this Azure account (TenantReuse), so
# delete + create gives the same link back.
set -euo pipefail

DEPLOY_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
APP_DIR="$(cd "$DEPLOY_DIR/.." && pwd)"
BUILD_DIR="$DEPLOY_DIR/build"

GROUP=olist-ops-rg
PLAN=olist-ops-plan
APP=olist-ops
LOCATION=eastasia   # Southeast Asia has no free (F1) capacity on this subscription
RUNTIME="PYTHON:3.13"

# A just-created app can take a few seconds to be readable; retry, then fail loudly.
app_host() {
  local host attempt
  for attempt in 1 2 3 4 5 6; do
    host="$(az webapp show -g "$GROUP" -n "$APP" --query defaultHostName -o tsv 2>/dev/null || true)"
    [[ -n "$host" ]] && { echo "$host"; return 0; }
    sleep 10
  done
  echo "Could not read the link of web app $APP in $GROUP (does it exist? run: $0 create)" >&2
  return 1
}

create() {
  az group create -n "$GROUP" -l "$LOCATION" -o none
  az appservice plan create -g "$GROUP" -n "$PLAN" -l "$LOCATION" --is-linux --sku F1 -o none
  az webapp create -g "$GROUP" -p "$PLAN" -n "$APP" --runtime "$RUNTIME" \
    --domain-name-scope TenantReuse -o none
  local host
  host="$(app_host)"
  echo "Created https://$host/ . Now run: $0 deploy"
}

deploy() {
  local host settings zip_path
  host="$(app_host)"
  bash "$DEPLOY_DIR/stage.sh"

  # Provider settings come from deployment/olist_dashboard/.env; write them to a private temp
  # file so the key never appears in a command line or on screen.
  settings="$(mktemp)"
  zip_path="$(mktemp -d)/app.zip"
  trap 'rm -f "$settings" "$zip_path"; rmdir "$(dirname "$zip_path")" 2>/dev/null || true' RETURN
  python3 - "$APP_DIR" "$host" "$settings" <<'PY'
import json, sys
from pathlib import Path
app_dir, host, out = sys.argv[1:]
sys.path.insert(0, app_dir)
from ask_service import parse_dotenv_text
env = Path(app_dir, ".env")
if not env.is_file():
    sys.exit("Missing {}: it must hold the VAMOS_LLM_* provider settings".format(env))
values = parse_dotenv_text(env.read_text(encoding="utf-8"))
keys = ("VAMOS_LLM_BASE_URL", "VAMOS_LLM_API_KEY", "VAMOS_LLM_MODEL")
missing = [k for k in keys if not values.get(k)]
if missing:
    sys.exit("Missing in {}: {}".format(env, ", ".join(missing)))
settings = {k: values[k] for k in keys}
settings["VAMOS_ALLOWED_HOSTS"] = host
# The App Service build installs the staged requirements.txt (Oryx) before
# start.sh runs; without it the MLflow server and pandas are missing.
settings["SCM_DO_BUILD_DURING_DEPLOYMENT"] = "true"
Path(out).write_text(json.dumps(settings), encoding="utf-8")
PY
  az webapp config appsettings set -g "$GROUP" -n "$APP" --settings "@$settings" -o none
  az webapp config set -g "$GROUP" -n "$APP" \
    --startup-file "sh /home/site/wwwroot/deployment/olist_dashboard/deploy/start.sh" -o none

  (cd "$BUILD_DIR" && zip -qr "$zip_path" .)
  az webapp deploy -g "$GROUP" -n "$APP" --src-path "$zip_path" --type zip -o none
  rm -rf "$BUILD_DIR"
  echo "Deployed https://$host/"
}

size() {
  [[ "${1:-}" =~ ^(F1|B1|B2)$ ]] || { echo "usage: $0 size F1|B1|B2" >&2; exit 2; }
  local host
  az appservice plan update -g "$GROUP" -n "$PLAN" --sku "$1" -o none
  # A paid plan alone does not keep the app warm; Always On must be set on the app (F1 cannot have it).
  local always_on=true
  [[ "$1" == F1 ]] && always_on=false
  az webapp config set -g "$GROUP" -n "$APP" --always-on "$always_on" -o none
  host="$(app_host)"
  echo "Plan is now $1 (link unchanged: https://$host/)"
}

delete() {
  az group delete -n "$GROUP" --yes
  echo "Deleted $GROUP; nothing is billed any more."
}

case "${1:-}" in
  create) create ;;
  deploy) deploy ;;
  size) size "${2:-}" ;;
  delete) delete ;;
  *) echo "usage: $0 create|deploy|size F1|B1|B2|delete" >&2; exit 2 ;;
esac
