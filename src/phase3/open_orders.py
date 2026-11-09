"""Scoring inputs for orders that left the seller but are not in the delivered modelling population.

The Phase 2 order table keeps only orders with status `delivered` and both delivery timestamps. An
order that was handed to the carrier and later cancelled, or that stayed `shipped`, therefore has no
model inputs, even though it was open on the carrier's morning list. This module builds those rows with
the same feature code as `notebooks/data_prep.ipynb` and the same handover steps as `load_cohort`.

Open rows are always query rows: route, seller, ZIP and carrier-leg history comes from the delivered
order table only. The output is `data/processed/phase2_open_orders.csv`; `load_open_queue_table` returns
the rows with the columns the existing Phase 2 handover classifier and delivery-time model need.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from olist.classification import feature_columns
from olist.eta import stage_spec
from olist.handover_classifier import add_relative_state_backlog, add_remaining_slack, variant_spec
from olist.paths import PROCESSED_DIR, REPO_ROOT

from phase3.data_prep import (
    MODEL_COLUMNS, SPLIT_COLUMNS, add_history_extension, add_history_features, add_purchase_features,
    add_stage_features, assemble_order_table, build_item_features, build_payment_features,
    clean_geolocation, load_raw, prepare_products, zip_centroids,
)

ROOT = REPO_ROOT
OUTPUT_PATH = PROCESSED_DIR / "phase2_open_orders.csv"
DELIVERED_PATH = PROCESSED_DIR / "phase2_order_table.csv"
EXTRA_COLUMNS = [
    "order_status", "order_approved_at", "order_delivered_carrier_date", "order_estimated_delivery_date",
    "promise_date", "handover_time", "handover_valid", "handover_days", "approval_days", "route",
    "transit_days", "route_transit_90d", "route_transit_90d_at_handover", "remaining_slack",
    "past_promise_at_handover", "slack_shortfall", "zip3_transit_90d_at_handover",
    "seller_handover_90d_at_handover", "state_backlog_recent_vs_8wk",
]


def select_open_orders(orders_raw, model_order_ids):
    """Orders with a valid carrier handover that are not in the delivered modelling table.

    A handover is valid when the carrier timestamp is present and not before the purchase. A recorded
    delivery must also come after the handover. A missing delivery date is allowed: that is the
    never-delivered case this table exists for.
    """
    carrier = orders_raw["order_delivered_carrier_date"]
    delivered = orders_raw["order_delivered_customer_date"]
    valid = (carrier.notna() & carrier.ge(orders_raw["order_purchase_timestamp"])
             & (delivered.isna() | carrier.lt(delivered)))
    return orders_raw.loc[~orders_raw["order_id"].isin(model_order_ids) & valid].copy()


def _customer_zip(orders_raw, customers):
    return orders_raw[["order_id", "customer_id"]].merge(
        customers[["customer_id", "customer_zip_code_prefix"]], on="customer_id", how="left",
        validate="many_to_one")


def _delivered_history(delivered, orders_raw, customers):
    """The delivered order table plus the stage and ZIP columns the as-of history needs."""
    history = add_stage_features(delivered, orders_raw)
    history = history.merge(_customer_zip(orders_raw, customers), on="order_id", how="left",
                            validate="one_to_one")
    return history


def build_open_orders(root=ROOT):
    """Build the scoring-only open-orders table and return it."""
    root = Path(root)
    raw = load_raw(root / "data" / "raw")
    delivered = pd.read_csv(root / "data" / "processed" / DELIVERED_PATH.name,
                            parse_dates=["order_purchase_timestamp", "outcome_available_at"])
    open_orders = select_open_orders(raw["orders"], delivered["order_id"])

    products = prepare_products(raw["products"], raw["translation"])
    centroids = zip_centroids(clean_geolocation(raw["geolocation"]))
    _, item_features = build_item_features(
        open_orders, raw["items"], products, raw["sellers"], raw["customers"], centroids)
    payment_features = build_payment_features(raw["payments"])

    open_orders = add_purchase_features(open_orders)
    open_orders["is_late"] = np.nan
    open_orders["delivery_days"] = np.nan
    assembled = assemble_order_table(open_orders, item_features, payment_features)
    # Stage features will re-merge these from the raw orders; keeping them would duplicate the columns.
    assembled = assembled.drop(columns=["order_approved_at", "order_delivered_carrier_date"])

    history = _delivered_history(delivered, raw["orders"], raw["customers"])
    table = add_history_features(assembled, history=history)[MODEL_COLUMNS]
    table = add_stage_features(table, raw["orders"], history)
    table = add_remaining_slack(table)
    table = add_history_extension(table, _customer_zip(raw["orders"], raw["customers"]), history)
    table = add_relative_state_backlog(table, raw["orders"], raw["customers"])
    # Carrier-to-customer days is a post-handover outcome. Open rows are never a history source, so it
    # stays empty even for the few rows whose eventual delivery was later recorded.
    table["transit_days"] = np.nan

    promises = raw["orders"][["order_id", "order_estimated_delivery_date", "order_status"]]
    table = table.merge(promises, on="order_id", how="left", validate="one_to_one")
    table["promise_date"] = table["order_estimated_delivery_date"].dt.normalize()
    for column in SPLIT_COLUMNS:
        table[column] = pd.NA
    table = table[MODEL_COLUMNS + SPLIT_COLUMNS + EXTRA_COLUMNS]
    return table


def write_open_orders(root=ROOT, path=None):
    """Build the table and write it to `path` (default: data/processed/phase2_open_orders.csv)."""
    path = Path(path) if path is not None else Path(root) / "data" / "processed" / OUTPUT_PATH.name
    table = build_open_orders(root)
    table.to_csv(path, index=False)
    return table, path


def load_open_queue_table(root=ROOT, rebuild=False):
    """Return the open rows with the columns `phase3.queue.load_queue_table` supplies.

    The saved table already went through the same stage and handover steps as `load_cohort`; this
    function restores the queue-facing columns and fails loudly if a model input is missing. Pass
    `rebuild=True` to regenerate the features from the raw CSVs instead of reading the saved file.
    """
    if rebuild:
        table = build_open_orders(root)
    else:
        path = Path(root) / "data" / "processed" / OUTPUT_PATH.name
        if not path.is_file():
            raise FileNotFoundError(
                f"Open-orders table not found: {path}. Run python -m phase3.build_open_orders first.")
        table = pd.read_csv(path, parse_dates=[
            "order_purchase_timestamp", "outcome_available_at", "order_approved_at",
            "order_delivered_carrier_date", "order_estimated_delivery_date", "promise_date",
            "handover_time"])

    spec = json.loads((Path(root) / "data" / "processed" / "phase2_feature_spec.json").read_text())
    required = set(sum(feature_columns(variant_spec(spec)), [])
                   + sum(feature_columns(stage_spec(spec, "handover")), []))
    missing = sorted(required - set(table.columns))
    if missing:
        raise ValueError(f"Open-orders table is missing model inputs: {missing}")

    table["purchase_month"] = table["order_purchase_timestamp"].dt.to_period("M").astype(str)
    table["handover_day"] = table["handover_time"].dt.normalize()
    table["late"] = pd.NA
    return table


def snapshot_counts(orders_raw, as_of="2018-08-22 09:00"):
    """Audit check: orders with the carrier at the cutoff and not delivered by it, by final status."""
    cutoff = pd.Timestamp(as_of)
    carrier = orders_raw["order_delivered_carrier_date"]
    delivered = orders_raw["order_delivered_customer_date"]
    open_at = carrier.notna() & carrier.lt(cutoff) & (delivered.isna() | delivered.ge(cutoff))
    rows = orders_raw.loc[open_at]
    return {
        "open": int(len(rows)),
        "by_status": rows["order_status"].value_counts().to_dict(),
        "never_delivered": int(delivered.loc[open_at].isna().sum()),
        "never_delivered_by_status": rows.loc[delivered.loc[open_at].isna(), "order_status"]
        .value_counts().to_dict(),
    }
