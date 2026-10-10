"""Buffer, out-of-fold and level-selection checks for the promise engine, on synthetic data."""

import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from phase2_promise import (BASELINES, CANDIDATES, GROUPS, LEVELS, SMOKE_GROUPS, adaptive_promises,  # noqa: E402
                            buffer_promises, fixed_level, forecast, interpolate, matched_scores, matched_target)


class BufferTests(unittest.TestCase):
    def setUp(self):
        # Calibration candidates for target day 100: purchase days 10..54 qualify (window [10, 55)).
        self.purchase = np.array([5, 10, 20, 54, 55, 60, 100, 100])
        self.delivery = np.array([15, 40, 99, 100, 70, 80, 110, 110])  # row 3 delivered on D: excluded
        self.need = np.array([100, 4, 10, 100, 100, 100, 7, 7])
        self.mu = np.full(8, 5.0)

    def test_only_delivered_before_day_and_purchased_in_window(self):
        target = np.array([6, 7])
        got = buffer_promises(self.mu, self.need, self.purchase, self.delivery, target, levels=np.array([0.9]))
        # qualifying rows: 1 and 2 -> r = (4-5)/5, (10-5)/5 -> q90 = -0.2 + 0.9*1.2 = 0.88 -> ceil(5 + 4.4) = 10
        np.testing.assert_array_equal(got, [[10], [10]])

    def test_changing_excluded_rows_changes_nothing(self):
        target = np.array([6])
        base = buffer_promises(self.mu, self.need, self.purchase, self.delivery, target)
        need = self.need.copy()
        need[[0, 3, 4, 5]] = 1  # too early, delivered on D, too recent, too recent
        np.testing.assert_array_equal(base, buffer_promises(self.mu, need, self.purchase, self.delivery, target))

    def test_integer_and_at_least_one(self):
        mu = np.full(8, 0.2)
        need = np.zeros(8)
        got = buffer_promises(mu, need, self.purchase, self.delivery, np.array([6]))
        self.assertTrue((got >= 1).all())
        np.testing.assert_array_equal(got, np.round(got))

    def test_empty_calibration_is_loud(self):
        with self.assertRaises(ValueError):
            buffer_promises(self.mu, self.need, np.array([100, 100]), np.array([110, 110]), np.array([0]))

    def test_missing_forecast_in_calibration_is_loud(self):
        mu = self.mu.copy()
        mu[1] = np.nan
        with self.assertRaises(ValueError):
            buffer_promises(mu, self.need, self.purchase, self.delivery, np.array([6]))


class OutOfFoldTests(unittest.TestCase):
    def test_fitting_rows_never_predicted_by_a_model_fitted_on_them(self):
        n = 90
        rng = np.random.default_rng(0)
        full = pd.DataFrame({
            "order_purchase_timestamp": pd.date_range("2018-01-01", periods=n, freq="D"),
            "route_typical_days": rng.normal(10, 1, n), "delivery_days": rng.normal(10, 1, n)})
        spec = {"random_state": 0, "numeric_features": ["route_typical_days"], "categorical_features": [],
                "linear_log_candidates": []}
        # Memorising model: unconstrained tree on a unique key. An out-of-fold prediction cannot equal the row's own target.
        full["route_typical_days"] = np.arange(n, dtype=float)
        fit, target = np.arange(60), np.arange(60, 90)
        mu = forecast(full, fit, target, np.arange(60), "tree", {"max_depth": None, "min_samples_leaf": 1}, spec)
        self.assertFalse(np.isnan(mu[:90]).any())
        self.assertFalse(np.isclose(mu[fit], full["delivery_days"].to_numpy()[fit]).any())


class SelectionTests(unittest.TestCase):
    def test_level_uses_given_validation_rates_only(self):
        rates = np.linspace(0.90, 0.99, len(LEVELS))
        idx = fixed_level(rates)
        self.assertGreaterEqual(rates[idx], 0.95)
        self.assertLess(rates[idx - 1], 0.95)
        with self.assertRaises(ValueError):
            fixed_level(np.full(len(LEVELS), 0.9))

    def test_interpolation(self):
        self.assertAlmostEqual(interpolate(np.array([0.9, 1.0]), np.array([10.0, 20.0]), 0.95), 15.0)
        self.assertTrue(np.isnan(interpolate(np.array([0.9, 0.93]), np.array([10.0, 12.0]), 0.95)))

    def test_groups_are_well_formed(self):
        names = [n for n, _, _ in CANDIDATES]
        self.assertEqual(len(names), len(set(names)))
        for groups in (GROUPS, SMOKE_GROUPS):
            for _, baseline, configs in groups:
                self.assertTrue(set(configs) <= set(names) - BASELINES)
                if baseline is not None:
                    self.assertEqual(configs[0], baseline)
        self.assertEqual([g for g, _, _ in GROUPS], ["linear", "tree", "forest", "boost"])


class MatchedTargetTests(unittest.TestCase):
    def test_interpolates_first_crossing(self):
        targets = np.array([0.90, 0.91, 0.92, 0.93])
        self.assertAlmostEqual(matched_target(np.array([0.93, 0.94, 0.96, 0.97]), targets), 0.915)
        self.assertAlmostEqual(matched_target(np.array([0.93, 0.95, 0.96, 0.97]), targets), 0.91)

    def test_unbracketed_is_nan(self):
        targets = np.array([0.90, 0.91])
        self.assertTrue(np.isnan(matched_target(np.array([0.93, 0.94]), targets)))  # never reaches 95%
        self.assertTrue(np.isnan(matched_target(np.array([0.96, 0.97]), targets)))  # already above at the lowest target

    def test_matched_scores_picks_shorter_gamma(self):
        targets = np.array([0.90, 0.91])
        rows = []
        for gamma, promise in [(0.0, (30.0, 40.0)), (0.05, (20.0, 24.0))]:
            for w in range(2):
                for t, on, p in zip(targets, (0.94, 0.96), promise):
                    rows.append({"candidate": "a", "gamma": gamma, "target": t, "window": w, "on_time": on,
                                 "mean_promise": p + w})
        got = matched_scores(pd.DataFrame(rows), targets)["a"]
        self.assertEqual(got["gamma"], 0.05)
        self.assertAlmostEqual(got["target"], 0.905)
        np.testing.assert_allclose(got["per_window"], [22.0, 23.0])


class AdaptiveTests(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(1)
        self.day = np.repeat(np.arange(300), 4)
        self.need = rng.integers(2, 20, len(self.day)).astype(float)
        self.delivery = self.day + self.need.astype(int)
        self.mu = np.full(len(self.day), 8.0)

    def run_(self, gamma, need=None, delivery=None):
        return adaptive_promises(self.mu, self.need if need is None else need, self.day,
                                 self.delivery if delivery is None else delivery, 150, 250, gamma)

    def test_gamma_zero_keeps_nominal_level(self):
        _, trace = self.run_(0.0)
        np.testing.assert_allclose(trace["level"], 0.95)

    def test_update_rule_and_clipping(self):
        _, trace = self.run_(0.01)
        alpha = 0.05
        for _, row in trace.iterrows():
            self.assertAlmostEqual(row["level"], min(max(1 - alpha, 0.5), 0.995))
            if row["n_informing"]:
                alpha += 0.01 * (0.05 - row["e"])
        _, wild = self.run_(5.0)
        self.assertTrue(wild["level"].between(0.5, 0.995).all())
        self.assertEqual(wild["level"].min(), 0.5)  # saturates low and high with this gamma
        self.assertEqual(wild["level"].max(), 0.995)

    def test_no_look_ahead(self):
        promise, trace = self.run_(0.02)
        d0 = 200
        date = self.day + promise
        # Orders delivered on or after d0 whose promise date is d0 or later are unknowable through day d0 - 1.
        future = (self.delivery >= d0) & (date >= d0)
        need, delivery = self.need.copy(), self.delivery.copy()
        need[future] += 7
        delivery[future] += 7
        _, changed = self.run_(0.02, need, delivery)
        np.testing.assert_allclose(trace.loc[trace["day"] <= d0, "level"], changed.loc[changed["day"] <= d0, "level"])
        # sanity: the perturbation does matter later
        self.assertFalse(np.allclose(trace["level"], changed["level"]))


if __name__ == "__main__":
    unittest.main()
