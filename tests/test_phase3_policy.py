"""Tests for the Phase 3 promise policy (phase3.policy).

Run from the repository root:
    python -m pytest tests/test_phase3_policy.py
"""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT / "src"))

from phase3 import policy  # noqa: E402


def make_table(rows):
    """Small order table with the columns the policy module reads."""
    frame = pd.DataFrame(rows)
    frame["order_purchase_timestamp"] = pd.to_datetime(frame["order_purchase_timestamp"])
    frame["outcome_available_at"] = pd.to_datetime(frame["outcome_available_at"])
    frame["purchase_month"] = frame["order_purchase_timestamp"].dt.to_period("M")
    frame["region"] = policy.region_of(frame["customer_state"])
    return frame


def order_row(state, purchase, outcome, delivery_days, promised_days=20.5, is_late=False, split="test"):
    return {"customer_state": state, "order_purchase_timestamp": purchase,
            "outcome_available_at": outcome, "delivery_days": delivery_days,
            "promised_days": promised_days, "is_late": is_late, "split": split}


class ConstantModel:
    """Stand-in ETA model: always predicts the same days."""

    def __init__(self, value):
        self.value = value

    def predict(self, rows):
        return np.full(len(rows), float(self.value))


EMPTY_SPEC = {"numeric_features": [], "categorical_features": []}


class RegionMappingTest(unittest.TestCase):
    def test_six_regions_with_sp_separate(self):
        states = pd.Series(["SP", "AM", "BA", "DF", "RJ", "RS"])
        self.assertEqual(policy.region_of(states).tolist(), ["SP", "N", "NE", "CO", "SE", "S"])

    def test_unknown_state_raises(self):
        with self.assertRaises(ValueError):
            policy.region_of(pd.Series(["XX"]))

    def test_all_27_states_mapped(self):
        self.assertEqual(len(policy.REGION_OF), 27)


class BufferTest(unittest.TestCase):
    def rows_for_month(self, month, day=10):
        rows = []
        for i in range(150):
            rows.append(order_row("SP", f"{month}-{day} 12:00:00", f"{month}-20 12:00:00",
                                  10.0 + (i % 20)))
        for i in range(10):
            rows.append(order_row("AM", f"{month}-{day} 12:00:00", f"{month}-20 12:00:00",
                                  40.0 + i))
        return rows

    def test_quantile_and_small_region_fallback(self):
        table = make_table(self.rows_for_month("2018-05") + self.rows_for_month("2018-07"))
        month = pd.Period("2018-07")
        scored, buffer = policy.buffers_for(
            table, month, pd.Series(5.0, index=table.index), pd.Series(0.0, index=table.index),
            target=0.1, regional=True)
        pool = table[table["purchase_month"].eq(month - 2)]
        # SP has 150 calibration orders, so it takes its own 90th residual percentile.
        expected_sp = float(pool.loc[pool["region"].eq("SP"), "delivery_days"].quantile(0.9))
        self.assertAlmostEqual(float(buffer[table["region"].eq("SP")].iloc[0]), expected_sp)
        # N (state AM) has 10 < 100 calibration orders, so it takes the pooled global quantile.
        expected_global = float((pool["delivery_days"] - 0.0).quantile(0.9))
        self.assertAlmostEqual(float(buffer[table["region"].eq("N")].iloc[0]), expected_global)

    def test_calibration_pool_excludes_outcomes_after_month_start(self):
        rows = [order_row("SP", "2018-05-10 12:00:00", "2018-05-20 12:00:00", 10.0),
                order_row("SP", "2018-05-11 12:00:00", "2018-07-05 12:00:00", 10.0)]
        table = make_table(rows)
        pool = policy.calibration_for(table, pd.Period("2018-07"))
        self.assertEqual(len(pool), 1)

    def test_empty_pool_raises(self):
        rows = [order_row("SP", "2018-06-10 12:00:00", "2018-06-20 12:00:00", 10.0)]
        table = make_table(rows)
        with self.assertRaises(ValueError):
            policy.calibration_for(table, pd.Period("2018-07"))


class PromiseTableTest(unittest.TestCase):
    def rows_for_month(self, month, n=110):
        return [order_row("SP" if i % 2 else "AM", f"{month}-10 12:00:00", f"{month}-20 12:00:00",
                          10.0 + (i % 30), is_late=bool(i % 3 == 0)) for i in range(n)]

    def test_columns_means_and_ceil(self):
        table = make_table(self.rows_for_month("2018-05") + self.rows_for_month("2018-03"))
        with mock.patch.object(policy, "MIN_CALIBRATION_ORDERS", 10):
            frame = policy.promise_table("2018-05-15", target=0.1, model=ConstantModel(5.0),
                                         table=table, spec=EMPTY_SPEC, regional=True)
        self.assertEqual(set(frame.columns), {"region", "buffer_days", "olist_mean_promise_days",
                                              "our_mean_promise_days", "olist_late_rate",
                                              "our_expected_late_rate", "n_orders"})
        scored = table[table["purchase_month"].eq(pd.Period("2018-05"))]
        sp = frame[frame["region"].eq("SP")].iloc[0]
        self.assertEqual(int(sp["n_orders"]), int(scored["region"].eq("SP").sum()))
        # Olist promise: calendar-day ceiling of 20.5 -> 21.
        self.assertAlmostEqual(float(sp["olist_mean_promise_days"]), 21.0)
        # Constant ETA 5 plus the constant calibration residual quantile; promise = ceil(eta + buffer).
        scored_sp = scored[scored["region"].eq("SP")]
        expected_promise = float(np.ceil(5.0 + (scored_sp["delivery_days"] - 5.0).quantile(0.9)))
        self.assertAlmostEqual(float(sp["our_mean_promise_days"]), expected_promise)
        self.assertAlmostEqual(float(sp["our_expected_late_rate"]),
                               float(policy.calendar_late(scored_sp, expected_promise).mean()))
        self.assertAlmostEqual(float(sp["olist_late_rate"]), float(scored_sp["is_late"].mean()))

    def test_uniform_buffer_is_global(self):
        table = make_table(self.rows_for_month("2018-05") + self.rows_for_month("2018-03"))
        with mock.patch.object(policy, "MIN_CALIBRATION_ORDERS", 10):
            uniform = policy.promise_table("2018-05-15", target=0.1, model=ConstantModel(5.0),
                                           table=table, spec=EMPTY_SPEC, regional=False)
            regional = policy.promise_table("2018-05-15", target=0.1, model=ConstantModel(5.0),
                                            table=table, spec=EMPTY_SPEC, regional=True)
        pool = table[table["purchase_month"].eq(pd.Period("2018-05") - 2)]
        expected_global = float((pool["delivery_days"] - 5.0).quantile(0.9))
        self.assertTrue((uniform["buffer_days"] - expected_global).abs().lt(1e-9).all())
        # Uniform promises are constant, so they cannot exceed the largest regional promise.
        self.assertLessEqual(float(uniform["our_mean_promise_days"].max()),
                             float(regional["our_mean_promise_days"].max()) + 1e-9)


class RouteCentreTest(unittest.TestCase):
    def test_missing_route_values_take_the_fitting_median(self):
        table = make_table([order_row("SP", "2018-05-10 12:00:00", "2018-05-20 12:00:00", 10.0)
                            for _ in range(4)])
        table["route_typical_days"] = [10.0, np.nan, 20.0, np.nan]
        centre = policy.route_centre(table, table.iloc[[0, 2]])
        self.assertEqual(centre.tolist(), [10.0, 15.0, 20.0, 15.0])

    def test_empty_or_unusable_fitting_rows_raise(self):
        table = make_table([order_row("SP", "2018-05-10 12:00:00", "2018-05-20 12:00:00", 10.0)])
        table["route_typical_days"] = [np.nan]
        with self.assertRaises(ValueError):
            policy.route_centre(table, table.iloc[:0])
        with self.assertRaises(ValueError):
            policy.route_centre(table, table)


class RouteBenchmarkOutputTest(unittest.TestCase):
    """The saved route-average comparison must match the policy functions and saved ETA tables."""

    @classmethod
    def setUpClass(cls):
        cls.frame = pd.read_csv(policy.RESULT_DIR / "route_benchmark.csv")
        cls.table, cls._ = policy.load_order_table()

    def test_four_centres_with_finite_values_and_preset_targets(self):
        frame = self.frame
        self.assertEqual(list(frame["centre"]),
                         ["ETA run A", "ETA run B", "Route average", "Olist reference"])
        self.assertEqual(set(frame["test_target"]), {"3.5pct"})
        self.assertEqual(set(frame["validation_target"]), {"5pct"})
        numeric = ["test_mean_promise_days", "test_late_rate",
                   "validation_mean_promise_days", "validation_late_rate"]
        self.assertTrue(np.isfinite(frame[numeric].to_numpy()).all())
        self.assertEqual(frame["test_n_orders"].tolist(), [19363] * 4)
        self.assertEqual(frame["validation_n_orders"].tolist(), [33428] * 4)

    def test_route_row_reproduces_the_route_policy(self):
        row = self.frame.set_index("centre").loc["Route average"]
        test = policy.route_test_period(self.table, 0.035, True)
        test = test[test["purchase_month"].eq("total")].iloc[0]
        self.assertAlmostEqual(float(row["test_mean_promise_days"]),
                               float(test["our_mean_promise_days"]), delta=1e-12)
        self.assertAlmostEqual(float(row["test_late_rate"]), float(test["our_late_rate"]), delta=1e-12)
        validation = policy.validation_monthly_policy(
            self.table, policy.walk_forward_route_centres(self.table, policy.VALIDATION_WF_MONTHS),
            0.05, True)
        validation = validation[validation["purchase_month"].eq("total")].iloc[0]
        self.assertAlmostEqual(float(row["validation_mean_promise_days"]),
                               float(validation["our_mean_promise_days"]), delta=1e-12)
        self.assertAlmostEqual(float(row["validation_late_rate"]),
                               float(validation["our_late_rate"]), delta=1e-12)

    def test_eta_and_olist_rows_match_the_saved_tables(self):
        monthly = pd.read_csv(policy.RESULT_DIR / "test_monthly_policy.csv")
        validation = pd.read_csv(policy.RESULT_DIR / "validation_monthly_policy.csv")
        reference = pd.read_csv(policy.RESULT_DIR / "olist_reference.csv")
        rows = self.frame.set_index("centre")
        for run_name, label in (("A", "ETA run A"), ("B", "ETA run B")):
            test_row = monthly[(monthly["scheme"].eq("frozen_phase2")) & (monthly["run"].eq(run_name))
                               & (monthly["target"].eq("3.5pct")) & (monthly["shape"].eq("regional"))
                               & (monthly["purchase_month"].eq("total"))].iloc[0]
            validation_row = validation[(validation["run"].eq(run_name))
                                        & (validation["target"].eq("5pct"))
                                        & (validation["purchase_month"].eq("total"))].iloc[0]
            self.assertAlmostEqual(float(rows.loc[label, "test_mean_promise_days"]),
                                   float(test_row["our_mean_promise_days"]), delta=1e-12)
            self.assertAlmostEqual(float(rows.loc[label, "test_late_rate"]),
                                   float(test_row["our_late_rate"]), delta=1e-12)
            self.assertAlmostEqual(float(rows.loc[label, "validation_mean_promise_days"]),
                                   float(validation_row["our_mean_promise_days"]), delta=1e-12)
            self.assertAlmostEqual(float(rows.loc[label, "validation_late_rate"]),
                                   float(validation_row["our_late_rate"]), delta=1e-12)
        olist = rows.loc["Olist reference"]
        self.assertAlmostEqual(float(olist["test_mean_promise_days"]),
                               float(reference.loc[0, "olist_mean_promise_days"]), delta=1e-12)
        self.assertAlmostEqual(float(olist["test_late_rate"]),
                               float(reference.loc[0, "olist_late_rate"]), delta=1e-12)
        self.assertAlmostEqual(float(olist["validation_mean_promise_days"]),
                               float(reference.loc[2, "olist_mean_promise_days"]), delta=1e-12)
        self.assertAlmostEqual(float(olist["validation_late_rate"]),
                               float(reference.loc[2, "olist_late_rate"]), delta=1e-12)


class RunSpecTest(unittest.TestCase):
    def test_run_b_drops_both_promise_inputs(self):
        spec = {"numeric_features": ["promised_days", "promise_slack", "distance_km"],
                "categorical_features": ["customer_state"]}
        changed = policy.run_spec(spec, "B")
        self.assertEqual(changed["numeric_features"], ["distance_km"])
        self.assertEqual(spec["numeric_features"], ["promised_days", "promise_slack", "distance_km"])

    def test_unknown_run_raises(self):
        with self.assertRaises(ValueError):
            policy.run_spec({}, "C")


class SelectionTest(unittest.TestCase):
    def write_selection(self, root, payload):
        path = Path(root) / "eta" / "selection.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload))
        return path

    def test_run_a_hyperparameters_come_from_selection_json(self):
        with tempfile.TemporaryDirectory() as root:
            self.write_selection(root, {"selected_model": {"checkout": {
                "candidate": "ridge_7.5", "family": "ridge", "parameters": {"alpha": 7.5}}}})
            candidate, family, params = policy.selected_checkout_model(root)
            self.assertEqual((candidate, family, params), ("ridge_7.5", "ridge", {"alpha": 7.5}))
            table, spec = policy.load_order_table()
            with mock.patch.object(policy, "selected_checkout_model",
                                   return_value=(candidate, family, params)):
                model, _ = policy.fit_eta_model("A", policy.run_spec(spec, "A"),
                                                table.iloc[:500], table.iloc[500:510])
            self.assertAlmostEqual(float(model.named_steps["model"].alpha), 7.5)

    def test_missing_selection_file_raises(self):
        with tempfile.TemporaryDirectory() as root, self.assertRaises(FileNotFoundError):
            policy.selected_checkout_model(root)

    def test_missing_selection_key_raises(self):
        with tempfile.TemporaryDirectory() as root:
            self.write_selection(root, {"selected_model": {"handover": {
                "candidate": "ridge_1000", "family": "ridge", "parameters": {"alpha": 1000.0}}}})
            with self.assertRaises(KeyError):
                policy.selected_checkout_model(root)


class RealDataContractTest(unittest.TestCase):
    def test_runs_map_to_selected_hyperparameters_and_contract(self):
        from olist.classification import feature_columns
        table, spec = policy.load_order_table()
        _, _, selected_params = policy.selected_checkout_model()
        for run_name, params in (("A", selected_params), ("B", {"alpha": 1000.0})):
            run_spec = policy.run_spec(spec, run_name)
            model, _ = policy.fit_eta_model(run_name, run_spec, table.iloc[:500], table.iloc[500:510])
            self.assertAlmostEqual(float(model.named_steps["model"].alpha), params["alpha"])
        run_b_columns = sum(feature_columns(policy.run_spec(spec, "B")), [])
        self.assertEqual(len(run_b_columns), 18)
        self.assertNotIn("promised_days", run_b_columns)
        self.assertNotIn("promise_slack", run_b_columns)
        run_a_columns = sum(feature_columns(policy.run_spec(spec, "A")), [])
        self.assertIn("promised_days", run_a_columns)
        self.assertIn("promise_slack", run_a_columns)


class EndToEndTest(unittest.TestCase):
    """One small end-to-end check on the real table's late-period rows with a constant ETA model."""

    @classmethod
    def setUpClass(cls):
        table, _ = policy.load_order_table()
        cls.table = table[table["order_purchase_timestamp"].ge(pd.Timestamp("2018-04-01"))].copy()
        cls.table["purchase_month"] = cls.table["order_purchase_timestamp"].dt.to_period("M")
        cls.table = cls.table.reset_index(drop=True)

    def test_score_test_period_shape_and_totals(self):
        months = [pd.Period("2018-06"), pd.Period("2018-07")]
        frame = policy.score_test_period("A", ConstantModel(10.0), EMPTY_SPEC, self.table,
                                         target=0.05, regional=True, months=months)
        self.assertEqual(list(frame["purchase_month"]), ["2018-06", "2018-07", "total"])
        self.assertTrue(np.isfinite(frame[["our_mean_promise_days", "our_late_rate"]].to_numpy()).all())
        total = frame.iloc[-1]
        self.assertEqual(int(total["n_orders"]),
                         int(self.table[self.table["split"].eq("test")
                                        & self.table["purchase_month"].isin(months)].shape[0]))
        # Promises are whole days above the constant ETA.
        self.assertGreaterEqual(float(frame.iloc[:-1]["our_mean_promise_days"].min()), 10.0)


if __name__ == "__main__":
    unittest.main()
