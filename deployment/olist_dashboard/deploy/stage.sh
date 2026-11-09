#!/usr/bin/env bash
# Stage the complete dashboard for Azure App Service.
#
# The staged layout mirrors the repository so every repo-root-relative path in
# the code keeps working once the zip lands in the app folder Azure runs it from:
#
#   requirements.txt                          (read by the App Service build)
#   src/{olist,phase3}                        (Actions/Models/Monitor code)
#   data/raw/{orders,customers}.csv           (queue, monitor and lookup loaders)
#   data/processed/{phase2_order_table.csv, phase2_feature_spec.json,
#                   phase2_open_orders.csv}   (order table, feature spec, open orders)
#   deployment/olist_dashboard/               (the app: server, static files,
#                                              data/, mlflow/, deploy/start.sh)
#
# Anything not needed at runtime (tests, verify scripts, notebooks, experiments,
# __pycache__, .env) is left out. PRAZO_STAGE_DIR overrides the build folder.
set -euo pipefail

DEPLOY_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
APP_DIR="$(cd "$DEPLOY_DIR/.." && pwd)"
REPO_ROOT="$(cd "$APP_DIR/../.." && pwd)"
BUILD_DIR="${PRAZO_STAGE_DIR:-$DEPLOY_DIR/build}"
APP_DEST="$BUILD_DIR/deployment/olist_dashboard"
SOURCE_DIR="$REPO_ROOT/data/raw"

# All nine CSVs feed the analytics database that stage.sh builds.
CSVS=(
  olist_orders_dataset.csv
  olist_order_items_dataset.csv
  olist_customers_dataset.csv
  olist_sellers_dataset.csv
  olist_order_payments_dataset.csv
  olist_order_reviews_dataset.csv
  olist_products_dataset.csv
  olist_geolocation_dataset.csv
  product_category_name_translation.csv
)

# The raw files the Phase 3 loaders read at serving time (queue, monitor cohort,
# promise policy and order lookup): purchase events plus customer states.
RAW_CSVS=(olist_orders_dataset.csv olist_customers_dataset.csv)

# The prepared Phase 2 data the Actions, Models and Monitor tabs load.
PROCESSED=(phase2_order_table.csv phase2_feature_spec.json phase2_open_orders.csv)

rm -rf "$BUILD_DIR"
mkdir -p "$APP_DEST/deploy" "$APP_DEST/data"

# --- the app itself -----------------------------------------------------------------
cp "$APP_DIR/index.html" "$APP_DEST/index.html"
cp "$DEPLOY_DIR/start.sh" "$APP_DEST/deploy/start.sh"

for file in "$APP_DIR"/*.py; do
  name="$(basename "$file")"
  case "$name" in
    verify_*.py|test_*.py) continue ;;
  esac
  cp "$file" "$APP_DEST/$name"
done

for file in "$APP_DIR"/*.js; do
  name="$(basename "$file")"
  case "$name" in
    verify_*.js) continue ;;
  esac
  cp "$file" "$APP_DEST/$name"
done

for file in overview-data.js overview.json ask-seed-chats.json; do
  cp "$APP_DIR/data/$file" "$APP_DEST/data/$file"
done
cp -R "$APP_DIR/data/phase2" "$APP_DEST/data/phase2"

# The MLflow store (portable mlflow-artifacts:/ locations; served over HTTP).
cp -R "$APP_DIR/mlflow" "$APP_DEST/mlflow"

# --- code and data at their repository locations --------------------------------------
mkdir -p "$BUILD_DIR/src" "$BUILD_DIR/data/raw" "$BUILD_DIR/data/processed"
cp -R "$REPO_ROOT/src/olist" "$REPO_ROOT/src/phase3" "$BUILD_DIR/src/"
find "$BUILD_DIR/src" -type d -name __pycache__ -prune -exec rm -rf {} +

for csv in "${RAW_CSVS[@]}"; do
  if [[ ! -f "$SOURCE_DIR/$csv" ]]; then
    echo "Missing required CSV: $SOURCE_DIR/$csv" >&2
    exit 1
  fi
  cp "$SOURCE_DIR/$csv" "$BUILD_DIR/data/raw/$csv"
done
for file in "${PROCESSED[@]}"; do
  if [[ ! -f "$REPO_ROOT/data/processed/$file" ]]; then
    echo "Missing required processed file: $REPO_ROOT/data/processed/$file" >&2
    exit 1
  fi
  cp "$REPO_ROOT/data/processed/$file" "$BUILD_DIR/data/processed/$file"
done

# --- requirements at the zip root, where the App Service build reads them -----------
cp "$APP_DIR/requirements.txt" "$BUILD_DIR/requirements.txt"

# --- the analytics database, built fresh from the raw CSVs ---------------------------
for csv in "${CSVS[@]}"; do
  if [[ ! -f "$SOURCE_DIR/$csv" ]]; then
    echo "Missing required CSV: $SOURCE_DIR/$csv" >&2
    exit 1
  fi
done
python3 "$APP_DIR/analytics_db.py" --source "$SOURCE_DIR" \
  --output "$APP_DEST/data/analytics.sqlite"

# --- refuse to ship the provider API key ----------------------------------------------
# Ask reads the key from the repo-root .env or the app's .env; check both.
for env_file in "$REPO_ROOT/.env" "$APP_DIR/.env"; do
  [[ -f "$env_file" ]] || continue
  key_value="$(sed -nE 's/^[[:space:]]*(export[[:space:]]+)?VAMOS_LLM_API_KEY[[:space:]]*=[[:space:]]*//p' "$env_file" | tail -n 1)"
  key_value="${key_value%$'\r'}"
  key_value="${key_value%"${key_value##*[![:space:]]}"}"   # trailing whitespace
  if [[ ${#key_value} -ge 2 ]]; then
    first="${key_value:0:1}"
    last="${key_value: -1}"
    if [[ "$first" == '"' && "$last" == '"' ]] || [[ "$first" == "'" && "$last" == "'" ]]; then
      key_value="${key_value:1:${#key_value}-2}"
    fi
  fi
  if [[ -n "$key_value" ]] && grep -rqF -- "$key_value" "$BUILD_DIR"; then
    echo "Refusing to stage: the provider API key from $env_file is present in the build folder." >&2
    exit 1
  fi
done

echo "Staged build context at $BUILD_DIR"
