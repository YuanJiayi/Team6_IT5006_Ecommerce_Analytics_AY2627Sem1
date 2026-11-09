"""Checks for the scoring-only table of orders handed over but absent from the delivered population."""

import json
import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from olist.classification import feature_columns  # noqa: E402
from olist.eta import load_stage_table, stage_spec  # noqa: E402
from olist.features import as_of_shrunk_mean  # noqa: E402
from olist.handover_classifier import (  # noqa: E402
    add_history_extension as original_add_history_extension, load_cohort, variant_spec,
)
from olist.paths import RAW_DIR  # noqa: E402
from phase3.data_prep import (  # noqa: E402
    ROUTE_PRIOR_STRENGTH, add_history_extension, add_stage_features,
)
from phase3.open_orders import (  # noqa: E402
    build_open_orders, load_open_queue_table, select_open_orders, snapshot_counts,
)

DATA = ROOT / "data" / "processed"
RAW = RAW_DIR
TIMESTAMPS = ["order_purchase_timestamp", "outcome_available_at"]


class OpenOrdersTableTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.table = pd.read_csv(DATA / "phase2_open_orders.csv", parse_dates=TIMESTAMPS)
        cls.delivered = pd.read_csv(DATA / "phase2_order_table.csv", usecols=["order_id"])
        cls.orders = pd.read_csv(
            RAW / "olist_orders_dataset.csv",
            parse_dates=["order_purchase_timestamp", "order_approved_at",
                         "order_delivered_carrier_date", "order_delivered_customer_date",
                         "order_estimated_delivery_date"])

    def test_population_is_valid_handovers_missing_from_delivered_table(self):
        selected = select_open_orders(self.orders, self.delivered["order_id"])
        self.assertEqual(set(self.table["order_id"]), set(selected["order_id"]))
        self.assertEqual(len(self.table), 1188)
        self.assertEqual(self.table["order_id"].is_unique, True)
        # Every row left the seller; none is in the delivered modelling table.
        self.assertFalse(set(self.table["order_id"]) & set(self.delivered["order_id"]))
        carrier = selected["order_delivered_carrier_date"]
        self.assertTrue(carrier.notna().all())
        self.assertTrue(carrier.ge(selected["order_purchase_timestamp"]).all())

    def test_status_breakdown_and_audit_snapshot(self):
        self.assertEqual(self.table["order_status"].value_counts().to_dict(),
                         {"shipped": 1106, "canceled": 75, "delivered": 7})
        check = snapshot_counts(self.orders)
        self.assertEqual(check["open"], 2755)
        self.assertEqual(check["by_status"], {"delivered": 1584, "shipped": 1102, "canceled": 69})
        self.assertEqual(check["never_delivered_by_status"], {"shipped": 1102, "canceled": 69, "delivered": 7})

    def test_labels_and_evaluation_columns_are_empty(self):
        empty = ["is_late", "delivery_days", "split", "random_split", "cv_fold", "random_cv_fold"]
        self.assertTrue(self.table[empty].isna().all().all())

    def test_raw_delivery_date_kept_when_it_exists(self):
        kept = self.table[self.table["outcome_available_at"].notna()]
        self.assertEqual(kept["order_status"].tolist(), ["canceled"] * 6)

    def test_history_features_use_delivered_rows_only(self):
        delivered = pd.read_csv(DATA / "phase2_order_table.csv", parse_dates=TIMESTAMPS)
        route = self.table["seller_state"] + ">" + self.table["customer_state"]
        history_route = delivered["seller_state"] + ">" + delivered["customer_state"]
        expected = as_of_shrunk_mean(
            history_route, delivered["delivery_days"], delivered["outcome_available_at"],
            route, self.table["order_purchase_timestamp"], ROUTE_PRIOR_STRENGTH)
        np.testing.assert_allclose(self.table["route_typical_days"], expected, equal_nan=True)

    def test_stage_steps_match_phase2_eta_on_delivered_rows(self):
        delivered = pd.read_csv(DATA / "phase2_order_table.csv", parse_dates=TIMESTAMPS)
        cohort, _, _ = load_stage_table(ROOT)
        ours = add_stage_features(delivered, self.orders).set_index("order_id")
        cohort = cohort.set_index("order_id")
        for column in ["handover_days", "approval_days", "route_transit_90d",
                       "route_transit_90d_at_handover"]:
            np.testing.assert_allclose(ours.loc[cohort.index, column], cohort[column], equal_nan=True)

    def test_history_extension_matches_original_on_delivered_rows(self):
        cohort, _, _ = load_cohort(ROOT)
        zips = (self.orders[["order_id", "customer_id"]]
                .merge(pd.read_csv(RAW / "olist_customers_dataset.csv",
                                   usecols=["customer_id", "customer_zip_code_prefix"]),
                       on="customer_id", validate="many_to_one"))
        history = cohort.merge(zips, on="order_id", how="left", validate="one_to_one")
        ours = add_history_extension(cohort, zips, history)
        original = original_add_history_extension(cohort, zips)
        for column in ["zip3_transit_90d_at_handover", "seller_handover_90d_at_handover"]:
            np.testing.assert_allclose(ours[column], original[column], equal_nan=True)

    def test_saved_table_rebuilds_from_raw(self):
        rebuilt = build_open_orders(ROOT)
        self.assertEqual(list(rebuilt.columns), list(self.table.columns))
        self.assertEqual(set(rebuilt["order_id"]), set(self.table["order_id"]))
        numeric = rebuilt.select_dtypes("number").columns
        ours = rebuilt.set_index("order_id")[numeric].sort_index()
        saved = self.table.set_index("order_id")[numeric].sort_index()
        np.testing.assert_allclose(ours.to_numpy(dtype=float), saved.to_numpy(dtype=float), equal_nan=True)

    def test_loader_exposes_the_model_inputs(self):
        table = load_open_queue_table(ROOT)
        spec = json.loads((DATA / "phase2_feature_spec.json").read_text())
        required = set(sum(feature_columns(variant_spec(spec)), [])
                       + sum(feature_columns(stage_spec(spec, "handover")), []))
        self.assertLessEqual(required, set(table.columns))
        self.assertTrue(table["purchase_month"].str.match(r"^\d{4}-\d{2}$").all())
        self.assertTrue(table["handover_day"].notna().all())
        self.assertTrue(table["late"].isna().all())

    def test_delivered_table_is_unchanged(self):
        import hashlib
        digest = hashlib.sha256((DATA / "phase2_order_table.csv").read_bytes()).hexdigest()
        self.assertEqual(digest, "90a25126d4bc9c3686bd1d8f356d5091adf5c86f5cadeebf8d07cf5847c90a43")


if __name__ == "__main__":
    unittest.main()
