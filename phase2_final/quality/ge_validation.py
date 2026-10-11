"""Great Expectations (GX Core 1.x) validation gate for the Phase 2 order table.

Validates the prepared modelling table *before* any model is fitted:
  1. Cohort contract      - row count, unique ids, non-null targets, split labels
  2. Leakage guard        - exact column set = feature spec + ids/targets/split;
                            any post-purchase column (delivery, approval, handover,
                            review, status) makes the check fail
  3. Plausibility         - ranges and category sets for every feature
  4. Drift gate           - the same feature-only suite is run on each purchase
                            month; monthly means must stay inside the envelope
                            seen in the training period

GX checks schema and values. It cannot prove as-of timing of historical route or
seller features; that stays with the future-truncation test in the modelling code.

Usage (from repo root):
    python milestone2/ge_validation.py
    python milestone2/ge_validation.py --table data/phase2_order_table.csv \
        --spec data/phase2_feature_spec.json --out results/ge_validation.json
Seeds are not needed (no randomness); GX Core version is recorded in the output.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import great_expectations as gx
import great_expectations.expectations as gxe
import pandas as pd

ID_AND_TARGET_COLS = [
    "order_id", "order_purchase_timestamp", "is_late", "delivery_days", "split",
]
# Bookkeeping columns allowed in the table but never used as model inputs.
# Schema v1 (main): time_block. Schema v2 (josh): the rest. Allowed only if present AND
# the spec does not list them as features; outcome_available_at is a post-purchase
# timestamp used solely for fitting eligibility (checked below).
BOOKKEEPING_COLS = ["time_block", "outcome_available_at", "furthest_seller_id",
                    "purchase_month", "random_split", "cv_fold", "random_cv_fold"]
FORBIDDEN_SUBSTRINGS = ["delivered", "approved", "handover", "carrier_date",
                        "review", "order_status", "actual"]
BR_STATES = ["AC", "AL", "AM", "AP", "BA", "CE", "DF", "ES", "GO", "MA", "MG",
             "MS", "MT", "PA", "PB", "PE", "PI", "PR", "RJ", "RN", "RO", "RR",
             "RS", "SC", "SE", "SP", "TO"]
DAYS = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
MONTHS = ["January", "February", "March", "April", "May", "June", "July",
          "August", "September", "October", "November", "December"]
PAYMENT_TYPES = ["credit_card", "boleto", "debit_card", "voucher", "unknown"]
# Columns whose monthly mean must stay inside the training-period envelope.
DRIFT_COLS = ["promised_days", "distance_km", "same_state", "total_price", "total_freight"]
MIN_MONTH_ORDERS = 1000  # months with fewer orders do not define the envelope


def build_contract_suite(df: pd.DataFrame, spec: dict, extra_cols: list[str]) -> gx.ExpectationSuite:
    features = spec["numeric_features"] + spec["categorical_features"]
    allowed = (set(features) | set(ID_AND_TARGET_COLS) | set(extra_cols)
               | {c for c in BOOKKEEPING_COLS if c in df.columns})
    suite = gx.ExpectationSuite(name="phase2_order_table_contract")
    exps = [
        # --- cohort contract ---
        gxe.ExpectTableRowCountToEqual(value=len(df)),
        gxe.ExpectColumnValuesToBeUnique(column="order_id"),
        gxe.ExpectColumnValuesToNotBeNull(column="order_id"),
        gxe.ExpectColumnValuesToNotBeNull(column="order_purchase_timestamp"),
        gxe.ExpectColumnValuesToNotBeNull(column="is_late"),
        gxe.ExpectColumnValuesToNotBeNull(column="delivery_days"),
        gxe.ExpectColumnValuesToBeInSet(column="is_late", value_set=[0, 1]),
        gxe.ExpectColumnValuesToBeInSet(column="split", value_set=sorted(set(df["split"].dropna().unique()) & {"train", "test", "unavailable_at_cutoff"})),
        # --- leakage guard: no column outside the frozen contract ---
        gxe.ExpectTableColumnsToMatchSet(column_set=sorted(allowed), exact_match=True),
        # --- target plausibility ---
        gxe.ExpectColumnValuesToBeBetween(column="delivery_days", min_value=0, max_value=210),
        # --- feature plausibility ---
        gxe.ExpectColumnValuesToBeBetween(column="promised_days", min_value=0, strict_min=True, max_value=210),
        gxe.ExpectColumnValuesToBeBetween(column="distance_km", min_value=0, max_value=4500),
        gxe.ExpectColumnValuesToBeInSet(column="same_state", value_set=[0, 1]),
        gxe.ExpectColumnValuesToBeBetween(column="purchase_hour", min_value=0, max_value=23),
        gxe.ExpectColumnValuesToBeBetween(column="n_items", min_value=1),
        gxe.ExpectColumnValuesToBeBetween(column="n_products", min_value=1),
        gxe.ExpectColumnValuesToBeBetween(column="n_sellers", min_value=1),
        gxe.ExpectColumnValuesToBeBetween(column="total_price", min_value=0, strict_min=True),
        gxe.ExpectColumnValuesToBeBetween(column="total_freight", min_value=0),
        gxe.ExpectColumnValuesToBeBetween(column="total_weight_g", min_value=0),
        gxe.ExpectColumnValuesToBeBetween(column="total_volume_cm3", min_value=0),
        gxe.ExpectColumnValuesToBeBetween(column="max_installments", min_value=0, max_value=24),
        gxe.ExpectColumnValuesToBeInSet(column="customer_state", value_set=BR_STATES),
        gxe.ExpectColumnValuesToBeInSet(column="seller_state", value_set=BR_STATES),
        gxe.ExpectColumnValuesToBeInSet(column="payment_type", value_set=PAYMENT_TYPES),
        gxe.ExpectColumnValuesToBeInSet(column="purchase_dayofweek", value_set=DAYS),
        gxe.ExpectColumnValuesToNotBeNull(column="product_category", mostly=0.98),
        # --- documented missingness (median-filled inside pipelines) ---
        gxe.ExpectColumnValuesToNotBeNull(column="distance_km", mostly=0.99),
        gxe.ExpectColumnValuesToNotBeNull(column="total_weight_g", mostly=0.999),
    ]
    if "purchase_month" in df.columns:
        exps.append(gxe.ExpectColumnValuesToBeInSet(column="purchase_month", value_set=MONTHS))
    # Schema v2 history features (as-of route / seller statistics)
    if "promise_slack" in df.columns:
        lo, hi = spec.get("history_features", {}).get("seller_ship_days_valid_range", [0, 60])
        exps += [
            gxe.ExpectColumnValuesToBeBetween(column="route_typical_days", min_value=0, strict_min=True, max_value=210),
            gxe.ExpectColumnValuesToBeBetween(column="seller_ship_days", min_value=lo, max_value=hi),
            gxe.ExpectColumnValuesToBeBetween(column="promise_slack", min_value=-210, max_value=210),
            gxe.ExpectColumnValuesToNotBeNull(column="route_typical_days", mostly=0.99),
            gxe.ExpectColumnValuesToNotBeNull(column="seller_ship_days", mostly=0.99),
        ]
    for e in exps:
        suite.add_expectation(e)
    return suite


def run_suite(context, name: str, suite: gx.ExpectationSuite, df: pd.DataFrame):
    """Register a one-off dataframe asset/batch/validation and run it."""
    src = context.data_sources.add_or_update_pandas(name=f"src_{name}")
    asset = src.add_dataframe_asset(name=f"asset_{name}")
    bd = asset.add_batch_definition_whole_dataframe(f"batch_{name}")
    vd = context.validation_definitions.add_or_update(
        gx.ValidationDefinition(name=f"vd_{name}", data=bd, suite=suite))
    return vd.run(batch_parameters={"dataframe": df})


def summarise(result) -> list[dict]:
    rows = []
    for r in result.results:
        cfg = r.expectation_config
        rows.append({
            "expectation": cfg.type,
            "column": cfg.kwargs.get("column", "table-level"),
            "success": bool(r.success),
            "unexpected_count": r.result.get("unexpected_count"),
            "observed_value": r.result.get("observed_value"),
        })
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--table", default="data/phase2_order_table.csv")
    ap.add_argument("--spec", default="data/phase2_feature_spec.json")
    ap.add_argument("--out", default="results/ge_validation.json")
    ap.add_argument("--extra-columns", nargs="*", default=[],
                    help="additional approved column names (e.g. schema-v2 as-of features)")
    args = ap.parse_args()

    head = pd.read_csv(args.table, nrows=0).columns
    date_cols = [c for c in ["order_purchase_timestamp", "outcome_available_at"] if c in head]
    df = pd.read_csv(args.table, parse_dates=date_cols)
    spec = json.loads(Path(args.spec).read_text())
    context = gx.get_context(mode="ephemeral")

    # 1-3: contract, leakage guard, plausibility
    suite = context.suites.add(build_contract_suite(df, spec, args.extra_columns))
    contract = run_suite(context, "contract", suite, df)
    contract_rows = summarise(contract)

    # Substring leakage screen (GX has no substring column matcher) - reported separately
    leak_hits = [c for c in df.columns if any(s in c.lower() for s in FORBIDDEN_SUBSTRINGS)]

    # Split/timestamp consistency (plain pandas, reported as custom checks)
    cutoff = pd.Timestamp(spec.get("test_cutoff") or spec["primary_cutoff"])
    tr, te = df[df.split == "train"], df[df.split == "test"]
    un = df[df.split == "unavailable_at_cutoff"]
    custom = {
        "forbidden_substring_columns": leak_hits,
        "train_all_before_cutoff": bool((tr.order_purchase_timestamp < cutoff).all()),
        "test_all_on_or_after_cutoff": bool((te.order_purchase_timestamp >= cutoff).all()) if len(te) else None,
        "n_train": int(len(tr)), "n_test": int(len(te)), "n_unavailable_at_cutoff": int(len(un)),
        "late_rate_train": round(float(tr.is_late.mean()), 4),
        "late_rate_test": round(float(te.is_late.mean()), 4),
    }
    if "outcome_available_at" in df.columns:
        # Maturity rule: a fitting order's outcome must be recorded by the cutoff; orders whose
        # outcome arrives later are held out of fitting. outcome_available_at is never a feature.
        custom["train_outcomes_known_by_cutoff"] = bool((tr.outcome_available_at <= cutoff).all())
        custom["unavailable_outcomes_after_cutoff"] = bool((un.outcome_available_at > cutoff).all()) if len(un) else None
        custom["outcome_after_purchase"] = bool((df.outcome_available_at >= df.order_purchase_timestamp).all())
        custom["outcome_available_at_in_features"] = "outcome_available_at" in (
            spec["numeric_features"] + spec["categorical_features"])
    # The split rule in data_prep puts the boundary-second order on the test side (>= cutoff).

    # 4: drift gate - monthly batches, envelope from training months with enough orders
    df["ym"] = df.order_purchase_timestamp.dt.to_period("M").astype(str)
    drift_cols = DRIFT_COLS + [c for c in ["route_typical_days", "seller_ship_days"] if c in df.columns]
    train_monthly = (df[df.split == "train"].groupby("ym")
                       .agg(n=("order_id", "size"), **{c: (c, "mean") for c in drift_cols}))
    envelope = train_monthly[train_monthly.n >= MIN_MONTH_ORDERS][drift_cols].agg(["min", "max"])
    drift_suite = context.suites.add(gx.ExpectationSuite(
        name="phase2_monthly_feature_drift",
        expectations=[gxe.ExpectColumnMeanToBeBetween(
            column=c, min_value=float(envelope.loc["min", c]), max_value=float(envelope.loc["max", c]))
            for c in drift_cols]))
    drift_rows = []
    skipped = []
    for (ym, sp), g in df[df.split.isin(["train", "test"])].groupby(["ym", "split"]):
        if len(g) < MIN_MONTH_ORDERS:  # sparse months (e.g. 2016, 2018-05 test remainder)
            skipped.append({"month": ym, "split": sp, "n_orders": int(len(g))})
            continue
        res = run_suite(context, f"drift_{ym.replace('-', '_')}_{sp}", drift_suite, g.drop(columns="ym"))
        for row in summarise(res):
            drift_rows.append({"month": ym, "split": sp, "n_orders": int(len(g)), **row})

    out = {
        "gx_core_version": gx.__version__,
        "table": args.table, "rows": int(len(df)),
        "contract_success": bool(contract.success),
        "contract_expectations": contract_rows,
        "n_contract_passed": int(sum(r["success"] for r in contract_rows)),
        "n_contract_total": len(contract_rows),
        "custom_checks": custom,
        "drift_envelope_training_months": {c: [round(float(envelope.loc["min", c]), 3),
                                              round(float(envelope.loc["max", c]), 3)] for c in drift_cols},
        "drift_envelope_min_month_orders": MIN_MONTH_ORDERS,
        "drift_months_skipped_too_few_orders": skipped,
        "drift_results": drift_rows,
    }
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2, default=str))

    print(f"GX Core {gx.__version__}: contract {out['n_contract_passed']}/{out['n_contract_total']} passed")
    for r in contract_rows:
        if not r["success"]:
            print("  FAIL", r["expectation"], r["column"], r["unexpected_count"], r["observed_value"])
    print("custom:", custom)
    fails = [d for d in drift_rows if not d["success"]]
    print(f"drift: {len(fails)} failed monthly checks")
    for d in fails:
        print("  ", d["month"], d["split"].upper(), d["column"], round(d["observed_value"], 2))
    print("wrote", out_path)


if __name__ == "__main__":
    main()
