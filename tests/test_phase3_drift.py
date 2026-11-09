"""Tests for phase3.drift: fast unit tests on small frames plus one end-to-end run.

Run from the repository root:
    python -m pytest tests/test_phase3_drift.py
"""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT / "src"))

from olist.paths import DASHBOARD_PHASE2, PROCESSED_DIR  # noqa: E402
from phase3 import drift  # noqa: E402


class PrimitiveTests(unittest.TestCase):
    def test_ks_distance_basics(self):
        rng = np.random.default_rng(0)
        a, b = rng.normal(size=5000), rng.normal(size=5000)
        self.assertEqual(drift.ks_distance(a, a), 0.0)
        self.assertEqual(drift.ks_distance([1.0, 2.0, 3.0], [5.0, 6.0, 7.0]), 1.0)
        self.assertLess(drift.ks_distance(a, b), 0.05)
        self.assertIsNone(drift.ks_distance(a, pd.Series([np.nan])))
        self.assertIsNone(drift.ks_distance(pd.Series(dtype=float), a))

    def test_total_variation(self):
        ref = pd.Series(["a"] * 5 + ["b"] * 5)
        self.assertAlmostEqual(drift.total_variation(ref, ref)[0], 0.0)
        disjoint = pd.Series(["c"] * 10)
        self.assertAlmostEqual(drift.total_variation(ref, disjoint)[0], 1.0)
        shifted = pd.Series(["a"] * 8 + ["b"] * 2)
        tv, shifts = drift.total_variation(ref, shifted)
        self.assertAlmostEqual(tv, 0.3)
        self.assertEqual(shifts["a"], 0.3)

    def test_psi_categorical_hand_computed(self):
        ref = pd.Series(["a"] * 50 + ["b"] * 50)
        cur = pd.Series(["a"] * 80 + ["b"] * 20)
        expected = 0.3 * np.log(0.8 / 0.5) + (-0.3) * np.log(0.2 / 0.5)
        self.assertAlmostEqual(drift.population_stability_index(ref, cur, categorical=True),
                               float(expected), places=12)

    def test_psi_numeric_hand_computed(self):
        ref = pd.Series([1.0, 2.0, 3.0, 4.0])
        cur = pd.Series([1.0, 1.0, 1.0, 4.0])
        expected = 0.25 * np.log(0.75 / 0.5) + (-0.25) * np.log(0.25 / 0.5)
        self.assertAlmostEqual(drift.population_stability_index(ref, cur, bins=2),
                               float(expected), places=12)

    def test_psi_is_zero_for_the_same_sample_and_none_for_empty(self):
        values = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0])
        self.assertAlmostEqual(drift.population_stability_index(values, values), 0.0, places=12)
        self.assertIsNone(drift.population_stability_index(values, pd.Series([np.nan])))

    def test_ks_flag_team_thresholds(self):
        self.assertEqual(drift.ks_flag(0.05), "none")
        self.assertEqual(drift.ks_flag(0.1), "watch")
        self.assertEqual(drift.ks_flag(0.19), "watch")
        self.assertEqual(drift.ks_flag(0.2), "act")
        self.assertIsNone(drift.ks_flag(None))

    def test_numeric_profile_missing_and_medians(self):
        ref = pd.Series([1.0, 2.0, np.nan])
        win = pd.Series([3.0, np.nan, np.nan, np.nan])
        profile = drift.numeric_profile(ref, win)
        self.assertEqual(profile["median_reference"], 1.5)
        self.assertEqual(profile["median_window"], 3.0)
        self.assertAlmostEqual(profile["missing_reference"], 1 / 3, places=3)
        self.assertAlmostEqual(profile["missing_window"], 0.75)
        self.assertEqual(profile["ks"], 1.0)
        self.assertEqual(profile["flag"], "act")

    def test_month_age_and_status(self):
        self.assertAlmostEqual(drift.month_age_days(pd.Timestamp("2018-10-17"), "2018-08"), 46.0, places=6)
        self.assertEqual(drift.maturity_status(1.0, 47.0), drift.STILL_ARRIVING)
        self.assertEqual(drift.maturity_status(0.5, 90.0), drift.STILL_ARRIVING)
        self.assertEqual(drift.maturity_status(0.95, 90.0), drift.MATURE)

    def test_late_labels_count_open_past_promise_and_drop_unknown(self):
        frame = pd.DataFrame({
            "order_purchase_timestamp": pd.to_datetime(["2018-01-01"] * 4),
            "promised_days": [10.0, 10.0, 1.0, 30.0],
            "outcome_available_at": pd.to_datetime(
                ["2018-01-05", "2018-01-08", "2018-03-01", "2018-03-01"]),
            "delivery_days": [4.0, 7.0, np.nan, np.nan],
            "is_late": [0, 1, np.nan, np.nan],
        })
        known, labels = drift.late_labels(frame, pd.Timestamp("2018-01-15 09:00"))
        self.assertEqual(known.tolist(), [True, True, True, False])
        self.assertEqual(labels.tolist(), [0, 1, 1, 1])

    def test_reliability_bins(self):
        y = np.array([0, 1, 0, 1, 1])
        p = np.array([0.005, 0.03, 0.06, 0.15, 0.6])
        bins = drift.reliability_bins(y, p, drift.RELIABILITY_BINS_PCT)
        self.assertEqual([b["orders"] for b in bins], [1, 0, 1, 1, 1, 0, 1])
        self.assertEqual(bins[0]["late"], 0)
        self.assertEqual(bins[4]["late"], 1)
        self.assertAlmostEqual(bins[6]["mean_predicted_pct"], 60.0)
        self.assertEqual(bins[5]["mean_predicted_pct"], None)

    def test_match_rule(self):
        self.assertEqual(drift._match(0.2959, 0.296), "yes")
        self.assertEqual(drift._match(5.8094, 5.8085), "close")
        self.assertEqual(drift._match(0.4344, 0.489), "no")
        self.assertEqual(drift._match(None, 1.0), "no")


class MixWithinTests(unittest.TestCase):
    @staticmethod
    def frame(route, days):
        return pd.DataFrame({"route": route, "delivery_days": days})

    def test_additivity_and_pure_mix_shift(self):
        ref = self.frame(["A"] * 8 + ["B"] * 2, [10.0] * 8 + [20.0] * 2)
        # Same route means, different mix: the whole drop must be mix, none within-route.
        later = self.frame(["A"] * 2 + ["B"] * 8, [10.0] * 2 + [20.0] * 8)
        mix, within = drift._mix_within(ref, later)
        drop = ref["delivery_days"].mean() - later["delivery_days"].mean()
        self.assertAlmostEqual(mix + within, drop, places=9)
        self.assertAlmostEqual(within, 0.0, places=9)
        self.assertAlmostEqual(mix, drop, places=9)

    def test_pure_within_route_shift(self):
        ref = self.frame(["A"] * 5 + ["B"] * 5, [10.0] * 5 + [20.0] * 5)
        later = self.frame(["A"] * 5 + ["B"] * 5, [8.0] * 5 + [18.0] * 5)
        mix, within = drift._mix_within(ref, later)
        self.assertAlmostEqual(mix, 0.0, places=9)
        self.assertAlmostEqual(within, 2.0, places=9)

    def test_route_in_one_period_only_is_dropped(self):
        ref = self.frame(["A"] * 5 + ["B"] * 5, [10.0] * 5 + [20.0] * 5)
        later = self.frame(["A"] * 5 + ["C"] * 5, [8.0] * 5 + [100.0] * 5)
        mix, within = drift._mix_within(ref, later)
        self.assertAlmostEqual(mix + within, 2.0, places=9)


class SyntheticRootTestCase(unittest.TestCase):
    """Builds a tiny Phase 2-style data root (fast, no real data) for the end-to-end run."""

    @classmethod
    def setUpClass(cls):
        cls.root = Path(tempfile.mkdtemp(prefix="drift_e2e_")) / "Phase2"
        (cls.root / "data" / "processed").mkdir(parents=True)
        (cls.root / "data" / "raw").mkdir(parents=True)
        rng = np.random.default_rng(42)
        orders, customers, rows = [], [], []
        customer_id = 0

        def batch(start, n, split, fold, late_share):
            nonlocal customer_id
            for i in range(n):
                purchase = pd.Timestamp(start) + pd.Timedelta(hours=rng.integers(0, 24 * 20))
                promised = float(rng.integers(4, 12))
                delivered = purchase + pd.Timedelta(days=float(rng.integers(3, 12)))
                late = rng.random() < late_share
                if late:
                    delivered = purchase + pd.Timedelta(days=float(rng.integers(12, 20)))
                carrier = purchase + pd.Timedelta(days=float(rng.integers(1, 4)))
                approved = purchase + pd.Timedelta(hours=float(rng.integers(1, 48)))
                customer_id += 1
                cid = f"c{customer_id:05d}"
                seller = f"s{rng.integers(0, 3)}"
                estimated = (purchase + pd.Timedelta(days=promised)).normalize()
                customers.append({"customer_id": cid, "customer_zip_code_prefix": str(rng.integers(10000, 99999)),
                                  "customer_state": ["SP", "RJ"][customer_id % 2]})
                orders.append({"order_id": f"o{len(rows):06d}", "customer_id": cid,
                               "order_purchase_timestamp": purchase,
                               "order_delivered_customer_date": delivered,
                               "order_estimated_delivery_date": estimated,
                               "order_approved_at": approved, "order_delivered_carrier_date": carrier})
                row = {
                    "order_id": f"o{len(rows):06d}", "order_purchase_timestamp": purchase,
                    "outcome_available_at": delivered, "purchase_month": str(purchase.to_period("M")),
                    "furthest_seller_id": seller, "promised_days": promised,
                    "route_typical_days": float(rng.integers(4, 12)), "seller_ship_days": float(rng.integers(1, 6)),
                    "distance_km": float(rng.integers(100, 2000)), "same_state": int(customer_id % 2),
                    "purchase_hour": purchase.hour, "n_items": 1, "n_products": 1, "n_sellers": 1,
                    "total_price": float(rng.integers(50, 500)), "total_freight": float(rng.integers(5, 50)),
                    "total_weight_g": float(rng.integers(100, 5000)), "total_volume_cm3": float(rng.integers(500, 9000)),
                    "max_installments": 1, "customer_state": ["SP", "RJ"][customer_id % 2],
                    "seller_state": seller.replace("s", "") and ["SP", "RJ", "MG"][int(seller[1:]) % 3],
                    "product_category": ["bed", "sport"][customer_id % 2],
                    "payment_type": "credit_card", "purchase_dayofweek": purchase.dayofweek,
                    "is_late": int(late), "delivery_days": (delivered - purchase).total_seconds() / 86400.0,
                    "split": split, "random_split": "train", "cv_fold": fold, "random_cv_fold": 0}
                row["promise_slack"] = row["promised_days"] - row["route_typical_days"]
                rows.append(row)

        batch("2017-10-01", 40, "train", -1, 0.15)   # seed history before the first validation fold
        batch("2017-11-01", 30, "train", 0, 0.15)
        batch("2017-12-01", 30, "train", 1, 0.12)
        batch("2018-01-01", 30, "train", 2, 0.10)
        batch("2018-02-01", 30, "train", 3, 0.08)
        batch("2018-03-01", 30, "train", 4, 0.06)
        batch("2018-06-01", 40, "test", -1, 0.10)
        batch("2018-07-01", 40, "test", -1, 0.20)

        pd.DataFrame(rows).to_csv(cls.root / "data" / "processed" / "phase2_order_table.csv", index=False)
        pd.DataFrame(orders).to_csv(cls.root / "data" / "raw" / "olist_orders_dataset.csv", index=False)
        pd.DataFrame(customers).drop_duplicates("customer_id").to_csv(
            cls.root / "data" / "raw" / "olist_customers_dataset.csv", index=False)
        spec = json.loads((PROCESSED_DIR / "phase2_feature_spec.json").read_text())
        (cls.root / "data" / "processed" / "phase2_feature_spec.json").write_text(json.dumps(spec))
        for relative in ("eta/selection.json", "handover/selection.json"):
            source = DASHBOARD_PHASE2 / relative
            target = cls.root / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(source.read_text())
        cls.spec = spec
        cls.out_dir = Path(tempfile.mkdtemp(prefix="drift_e2e_out_"))

    @classmethod
    def tearDownClass(cls):
        import shutil
        shutil.rmtree(cls.root.parent, ignore_errors=True)
        shutil.rmtree(cls.out_dir, ignore_errors=True)


class SelectedModelTests(unittest.TestCase):
    def test_model_parameters_come_from_selection_json(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            eta = root / "eta"
            handover = root / "handover"
            eta.mkdir(parents=True)
            handover.mkdir(parents=True)
            (eta / "selection.json").write_text(json.dumps({"selected_model": {
                "checkout": {"candidate": "ridge_7", "family": "ridge", "parameters": {"alpha": 7.0}},
                "handover": {"candidate": "ridge_9", "family": "ridge", "parameters": {"alpha": 9.0}}}}))
            (handover / "selection.json").write_text(json.dumps({"variants": {"primary": {"selected_model": {
                "candidate": "logistic_c0.5", "family": "linear",
                "parameters": {"C": 0.5, "class_weight": None}}}}}))
            selected = drift.selected_models(root)
            self.assertEqual(selected["checkout"]["parameters"], {"alpha": 7.0})
            self.assertEqual(selected["handover"]["candidate"], "ridge_9")
            self.assertEqual(selected["late_warning"]["parameters"]["C"], 0.5)
            with self.assertRaises(FileNotFoundError):
                drift.selected_models(root / "missing")
            (eta / "selection.json").write_text(json.dumps({"selected_model": {
                "checkout": {"candidate": "ridge_7", "family": "ridge", "parameters": {"alpha": 7.0}}}}))
            with self.assertRaises(KeyError):
                drift.selected_models(root)


class EndToEndTests(SyntheticRootTestCase):
    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.report, cls.ledger = drift.run_report(root=cls.root, results_root=cls.root,
                                                  out_dir=cls.out_dir, as_of="2018-10-01")

    def test_report_written_and_structured(self):
        self.assertTrue((self.out_dir / "drift_report.json").exists())
        self.assertTrue((self.out_dir / "ledger.csv").exists())
        loaded = json.loads((self.out_dir / "drift_report.json").read_text())
        self.assertEqual(loaded["as_of"], self.report["as_of"])
        for section in ("x_drift", "y_drift", "x_to_y", "route_decomposition", "input_counts"):
            self.assertIn(section, loaded)
        for stage in ("checkout", "handover", "late_warning"):
            self.assertIn(stage, loaded["x_to_y"])
            self.assertIn(stage, loaded["x_drift"])

    def test_windows_are_test_months(self):
        windows = list(self.report["x_drift"]["checkout"]["windows"])
        self.assertEqual(windows, ["test_overall", "2018-06", "2018-07"])
        months = {m["month"]: m for m in self.report["y_drift"]["monthly"]}
        self.assertEqual(self.report["x_drift"]["checkout"]["windows"]["2018-07"]["n"], months["2018-07"]["orders"])

    def test_monthly_outcomes_and_maturity(self):
        months = {m["month"]: m for m in self.report["y_drift"]["monthly"]}
        self.assertEqual(months["2018-07"]["maturity_status"], drift.MATURE)
        self.assertEqual(months["2018-06"]["maturity_status"], drift.MATURE)
        self.assertEqual(months["2018-07"]["orders"], 38)  # 2 of 40 lack a valid handover in the cohort
        self.assertLess(months["2018-06"]["late_rate_matured"], months["2018-07"]["late_rate_matured"])

    def test_decomposition_additive(self):
        dec = self.report["route_decomposition"]
        self.assertAlmostEqual(dec["route_mix_effect"] + dec["within_route_effect"], dec["mean_drop"], places=2)
        self.assertGreater(dec["state_routes_used"], 0)

    def test_model_performance_present(self):
        for stage in ("checkout", "handover"):
            block = self.report["x_to_y"][stage]
            self.assertIn("train_in_sample_mae", block)
            self.assertIn("validation_mean_mae", block)
            self.assertEqual(set(block["monthly"]), {"2018-06", "2018-07"})
        cls = self.report["x_to_y"]["late_warning"]
        self.assertIn("ap_pooled", cls["later_overall"])
        self.assertEqual(set(cls["monthly"]), {"2018-06", "2018-07"})
        self.assertTrue(all("reliability_bins" in m for m in cls["monthly"].values()))

    def test_ledger_rows_complete(self):
        frame = pd.DataFrame(self.ledger)
        self.assertTrue({"claim", "stated_value", "reproduced_value", "match"} <= set(frame.columns))
        self.assertTrue((frame["match"].isin(["yes", "close", "no"])).all())
        # On the synthetic root the stated real-data numbers cannot match; every row must still be present.
        self.assertGreater(len(frame), 40)


if __name__ == "__main__":
    unittest.main()


@unittest.skipUnless((PROCESSED_DIR / "phase2_order_table.csv").is_file(),
                     "the Phase 2 data is not available")
class RealDataThresholdTests(unittest.TestCase):
    """The deployed drift bar and the handover-time filter, on the real cohort."""

    @classmethod
    def setUpClass(cls):
        root = ROOT
        cls.cohort, cls.spec, _ = drift.load_cohort(root)
        cls.train = cls.cohort[cls.cohort["split"].eq("train")].reset_index(drop=True)

    def test_bar_is_the_highest_validation_month(self):
        features = [{"feature": "route_transit_90d_at_handover", "type": "numeric"},
                    {"feature": "state_backlog_recent_vs_8wk", "type": "numeric"}]
        table = drift.input_threshold_table(self.train, self.spec, features)
        self.assertEqual(table["readings"], "monthly")
        self.assertEqual(table["n_readings"], 11)
        for feature in ("route_transit_90d_at_handover", "state_backlog_recent_vs_8wk"):
            entry = table["inputs"][feature]
            self.assertEqual(len(entry["validation"]), 11)
            self.assertEqual(entry["threshold"], max(entry["validation"]))
            self.assertNotIn("false_alarms", entry)

    def test_handover_inputs_use_only_orders_handed_over_by_the_cutoff(self):
        cutoff = pd.Timestamp("2018-08-22") + pd.Timedelta(hours=9)
        visible = self.cohort[self.cohort["order_purchase_timestamp"].lt(cutoff)]
        month = pd.Series(drift.month_label(visible["order_purchase_timestamp"]), index=visible.index)
        mask = visible["split"].eq("test") & month.eq("2018-08")
        window = {name: mask for name in ("2018-08",)}
        scores = drift.x_drift(visible, self.spec, window, cutoff=cutoff)
        august = visible.loc[mask.to_numpy()]
        known = august.loc[august["handover_time"].le(cutoff)]
        self.assertEqual(len(known), 5093)
        for feature in ("route_transit_90d_at_handover", "state_backlog_recent_vs_8wk"):
            self.assertAlmostEqual(
                scores["late_warning"]["windows"]["2018-08"][feature]["ks"],
                drift.ks_distance(self.train[feature], known[feature]), places=4)
            self.assertNotAlmostEqual(
                drift.ks_distance(self.train[feature], august[feature]),
                drift.ks_distance(self.train[feature], known[feature]), places=3)
