# Running the Olist dashboard

The app is static HTML/JavaScript served by a Python server. The Actions and
Models tabs need the packages in `requirements.txt` and the committed MLflow
store; Ask uses a local SQLite database built from the raw Olist CSVs and a
configured OpenAI-compatible chat completions provider.

## 1. Build the local analytics database

Put the raw Olist CSVs in `data/raw/`. Required files:

```text
olist_orders_dataset.csv
olist_order_items_dataset.csv
olist_customers_dataset.csv
olist_sellers_dataset.csv
olist_order_payments_dataset.csv
olist_order_reviews_dataset.csv
olist_products_dataset.csv
olist_geolocation_dataset.csv
product_category_name_translation.csv
```

From the repository root:

```sh
python3 deployment/olist_dashboard/analytics_db.py
```

This writes `deployment/olist_dashboard/data/analytics.sqlite`. That file is generated, ignored
by Git and never served by the app. A rebuild is atomic and idempotent: the new
database is written to a temporary file, fully loaded and indexed, then replaces
the old file. If a source directory or CSV is missing, the command fails with the
exact path and leaves the previous database intact. The `smartcommerce`
consolidated CSV is not read.

Optional explicit paths:

```sh
python3 deployment/olist_dashboard/analytics_db.py --source data/raw \
  --output deployment/olist_dashboard/data/analytics.sqlite
```

The builder creates nine private `_base_*` tables, indexes for the common joins
and event dates, and a private `_vamos_meta` table with row counts, relationships
and observed date bounds (format version 2). The public table names Ask sees are
not stored in the file: the server creates them as temporary views for each
request and applies the selected cutoff there. Ask cannot read the `_base_*`
tables, `_vamos_meta` or any SQLite internal table.

## 2. Configure the Ask provider

Ask uses a DeepSeek-compatible/OpenAI-compatible chat completions endpoint. Set:

```sh
export VAMOS_LLM_BASE_URL="http://127.0.0.1:11434/v1"
export VAMOS_LLM_API_KEY="your-local-key"
export VAMOS_LLM_MODEL="your-model-name"
```

Alternatively, put the same three keys in an uncommitted `.env` in the repository
root or in `deployment/olist_dashboard/`. The environment
wins over the file. Never commit that file.

The server uses the key only for the server-to-provider request. It never sends
the key to the browser, serves `.env`, or writes it to logs or API responses. The
server sends the key only to the configured provider and does not follow provider
HTTP redirects.

If the three keys are absent, the Overview still runs, greetings still work, and
data questions return a clear `provider_not_configured` error. If the database is
missing, Ask returns `analytics_not_ready` with the build command.

## 3. Start the server

From the repository root:

```sh
python3 deployment/olist_dashboard/server.py
```

Then open <http://127.0.0.1:8000/>. The default bind address is localhost. To use
another local port:

```sh
python3 deployment/olist_dashboard/server.py --host 127.0.0.1 --port 8080
```

The server serves only the app files and `POST /api/ask`. It does not serve
`.env`, Python source, dotfiles or the generated SQLite database.

## 4. Models tab

The Models tab reads the three registered models from the committed MLflow store
at `deployment/olist_dashboard/mlflow/`. Rebuild it from the repository root:

```sh
python -m phase3.registry
```

A rebuild deletes and recreates the store through a local MLflow server, which
records portable `mlflow-artifacts:/` locations instead of machine paths, so the
store can be committed. To serve the Models tab and its read-only MLflow UI link
together, start both servers with the project environment's Python:

```sh
PRAZO_PYTHON=/path/to/it5006-proj/bin/python bash deployment/olist_dashboard/run_local.sh
```

The script starts MLflow with artifact serving at <http://127.0.0.1:5050/mlflow/>,
waits for it to be healthy, and then starts the dashboard at
<http://127.0.0.1:8000/>. `PRAZO_MLFLOW_DB` can point at a different store, and
`PRAZO_PORT` / `PRAZO_MLFLOW_PORT` change the ports. If the store or MLflow is
missing, the Models tab shows the command to run instead of inventing numbers.

## 5. Use Ask

Ask accepts natural-language analytical questions, for example:

- `Hi`
- `Which product categories had the highest item price in 2018?`
- `Compare on-time delivery rate by customer state in 2018`
- `Average review score by payment type`
- `Monthly order count in 2017`
- `Which sellers had the most late deliveries last year?`

A greeting returns a normal conversational reply. An ambiguous question gets a
clarification. A data question follows this path:

1. The model receives the schema, relationships, metric definitions, the selected
   `Data as of` exclusive cutoff and up to six validated earlier messages from the
   same page, so follow-ups such as `What about 2017?` can be interpreted.
2. The model proposes one SQLite `SELECT` or `WITH` statement. A malformed plan or
   rejected SQL gets one corrective retry before the request fails.
3. The server opens the database read-only, creates per-request temporary views
   over the private `_base_*` tables with the cutoff applied, validates and runs
   the SQL, then sends the bounded rows back.
4. The model writes a concise answer from those rows and proposes a chart. If the
   answer or title contains a number or date that is not in the rows or the
   selected cutoff, the server rejects it and retries the synthesis once with a
   corrective instruction. The only exception is a duration such as `30` in
   `last 30 days` or `30-day window`: it is allowed solely when the executable
   SQL itself contains a matching date modifier such as
   `date(:as_of, '-30 days')`, and only in a window phrase, never as a count or
   money claim. Matches inside comments or string literals are ignored. Question
   text and other SQL literals are not trusted, and derived window dates are
   still rejected.
5. The UI shows the answer, the exact SQL, a table and a chart when useful.

The cutoff is enforced by the data layer, not by trusting the model's `WHERE`
clause. The public `orders` view contains only orders purchased before the cutoff;
`order_items`, `payments` and `reviews` contain only rows tied to those visible
orders. Approval, carrier and customer-delivery timestamps on or after the cutoff
are `NULL`. Reviews appear only when `review_answer_timestamp` is non-null and
strictly before the cutoff, because `review_score` is not knowable until the
review is answered; a review created before the cutoff but answered later is
absent. A query that merely mentions `:as_of` in a comment cannot bypass this.
Direct reads of `_base_*` tables are denied by the authorizer and by a lexical
guard that also sees quoted names and CTEs that shadow a public view. A
constant-only query that never reads an analytics table is rejected, and a
numeric constant-output projection such as `999999 AS orders` is rejected even
when the query has a `FROM` clause.

Item price is `SUM(order_items.price)`, excludes freight and is not revenue. An
observed association is not causation. The `customers`, `sellers`, `products`,
`geolocation` and translation tables are catalog/reference data and are not
date-filtered. `order_status` is masked to what is historically knowable: the
public view reports `delivered` only when the customer-delivery event is visible
before the cutoff, and `NULL` otherwise. There is no status-event timestamp, so
shipped, canceled and other earlier states cannot be reconstructed; `NULL` is
unknown, not a cancellation, and final status is never presented as the status at
the cutoff. `order_estimated_delivery_date` is a calendar promise date, so on-time
means `date(order_delivered_customer_date) <= date(order_estimated_delivery_date)`
and late means the delivery calendar date is after the promise; a delivery on the
promised date is on time.

The SQL guard is intentionally narrow: one statement, no writes, no `ATTACH`, no
`PRAGMA`, no private or unknown tables, no risky string/encoding functions, a
15-second deadline, at most 200 rows, 30 columns and 200 KB of output. Raw
identifier columns and identifier-shaped output values are blocked even when
aliased or concatenated; address columns and review text are not in the public
views. Constant numeric projections are rejected, but this lexical guard cannot
formally prove that every SQL expression derives from source rows; the response
provenance says so rather than certifying all numbers. This is a
defense-in-depth filter, not a formal guarantee against every possible derived
inference. The server re-validates any SQL the browser sends.

Line charts use real date/month positions when the x column is temporal, so
irregular intervals are spaced by elapsed time; categorical charts keep even
spacing. Ask tables keep meaningful precision for small values instead of
rounding them to zero.


Ask charts are explanatory only; they are not pinned to Overview. Overview pins
keep their existing metric, state, window and visualization definitions and are
unaffected.

## 6. Actions API

The Actions tab reads its daily queue from the local server. The live prediction
endpoint scores one order with the same served champions and as-of rules:

```text
GET /api/actions/predict?date=YYYY-MM-DD&order_id=<32 hex characters>
```

An open parcel that was handed to the carrier before 09:00 and not yet delivered
gets its late-risk percentile, predicted delivery date, spare time, days left,
where it would rank that morning and whether it is already on the list. Orders
that were not bought, not handed over, already delivered or unknown keep the
status message instead of a prediction. The page calls this endpoint for the
Predict an order box.

Example response for an order on the 22 August 2018 list:

```json
{
  "ok": true,
  "date": "2018-08-22",
  "order_id": "d3310038bbf991ce02501a51022f5283",
  "found": true,
  "state": "open",
  "on_list": true,
  "customer_update": false,
  "rank": 1,
  "open_orders": 1363,
  "late_risk_percentile": 0.9944562161009608,
  "spare_time": -6.009415904797717,
  "spare_time_percentile": 0.9952291475099615,
  "predicted_delivery_date": "2018-08-31",
  "promised_date": "2018-08-23",
  "days_left": 1,
  "tier": "act",
  "why": [
    {
      "input": "handover_days",
      "label": "Days from purchase to handover",
      "contribution": 2.0957267033369384
    }
  ],
  "model_versions": {
    "checkout_eta": 1,
    "handover_eta": 1,
    "late_risk": 1
  },
  "message": "This order is on today's list at rank 1.",
  "latency_ms": 1.6
}
```

Every call appends one JSON line to `deployment/olist_dashboard/data/prediction_log.jsonl`
(git-ignored): timestamp, as-of date, order ID, model versions, late-risk score
and percentile, predicted delivery, spare time, rank and latency. The log keeps
the last 10,000 lines.

## 7. Run checks

```sh
bash deployment/olist_dashboard/verify_morning.sh
```

This runs:

```sh
python3 deployment/olist_dashboard/verify_morning.py
python3 deployment/olist_dashboard/verify_analytics.py
python3 deployment/olist_dashboard/verify_ask.py
node deployment/olist_dashboard/verify_morning.js
git diff --check
```

The analytics tests build a tiny fixture database from real CSV files. They check
deterministic rebuilds, real aggregate results, view-enforced cutoff semantics,
future delivery invisibility, review scores before and after the answer boundary,
direct and CTE-shadowed base-table denial, cutoff-comment bypass attempts,
constant-only queries, constant-output projections, identifier
alias/concatenation bypasses, blocked personal columns, result bounds and schema
metadata. The Ask tests use a scripted model and a local mock HTTP provider; no
real provider call or credential is used. They check greetings, clarification,
refusal, invalid model JSON, plan and synthesis retries, ungrounded numbers and
dates, duration claims in and out of window context, bounded follow-up history,
chart shape, HTTP Host/Origin rules, SQL auditability and key non-exposure.

A full data rebuild needs the raw CSVs in `data/raw`. If that folder is absent,
the overview artifact checks report the source rebuild as skipped; the app and
the fixture-based Ask checks still run.

## 8. Deploying to Azure

The same server runs on Azure App Service (Linux, built-in Python, free F1 plan,
region East Asia). There is no login: anyone with the link can use Ask, so
delete the app when it is no longer needed. From the repository root, after
`az login`:

```sh
bash deployment/olist_dashboard/deploy/azure.sh create   # once: resource group, plan, web app
bash deployment/olist_dashboard/deploy/azure.sh deploy   # after every edit
bash deployment/olist_dashboard/deploy/azure.sh size B1  # demo day: always on, own CPU (~US$0.02/h)
bash deployment/olist_dashboard/deploy/azure.sh delete   # remove everything; billing stops
```

`deploy` runs `deploy/stage.sh`, which stages the complete runtime package: the
app files, a database freshly built from `data/raw`, `src/olist` and `src/phase3`,
the Phase 2 data under `data/processed/` and `data/phase2/`, the raw CSVs the
Phase 3 loaders read, and the `mlflow/` model store. The package is about 107 MB
zipped. It refuses to stage if the provider API key from
`deployment/olist_dashboard/.env` appears in any staged file. The provider
settings are sent as private app settings and `VAMOS_ALLOWED_HOSTS` is set to
the app's address, which switches on public mode: the server then refuses to
start without the database and provider settings.

Azure installs the staged `requirements.txt` during deployment
(`SCM_DO_BUILD_DURING_DEPLOYMENT=true`; the runtime is Python 3.13). The pinned
versions come from a Python 3.14 environment; before the first real deploy,
check they also have Python 3.13 wheels.

`deploy/start.sh` starts a local MLflow server for the shipped store (as
`run_local.sh` does), waits for it to become healthy, then starts the dashboard
on Azure's port, with `src/` on `PYTHONPATH` and `EVIDENTLY_DISABLE_TELEMETRY=1`.
It fails loudly if either process cannot start. The analytics database and
MLflow store are copied to local disk at startup, because the app folder is on
network storage.

The link (`olist-ops-e5abckh8h0hfahdp.eastasia-01.azurewebsites.net`) is tied
to this Azure account, so `delete` followed by `create` gives the same link.
