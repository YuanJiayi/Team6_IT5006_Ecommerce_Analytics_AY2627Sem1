"""Checks for the regression leakage assertion and selection rule on synthetic data."""

import unittest

import numpy as np
import pandas as pd

from phase2_regression import SIMPLICITY, assert_fit_precedes, select_candidate


def scores(**overrides):
    base = {name: 5.0 for name in SIMPLICITY}
    base.update({"mean_baseline": 9.0, "route_baseline": 8.0})
    base.update(overrides)
    return base


class RegressionChecksTest(unittest.TestCase):
    def frame(self, available_offset):
        purchase = pd.date_range("2018-01-01", periods=6, freq="D")
        return pd.DataFrame({"order_purchase_timestamp": purchase,
                             "outcome_available_at": purchase + pd.Timedelta(days=available_offset)})

    def test_fit_outcomes_must_precede_window_start(self):
        fit, valid = np.array([0, 1]), np.array([4, 5])
        assert_fit_precedes(self.frame(1), fit, valid)
        with self.assertRaises(AssertionError):
            assert_fit_precedes(self.frame(3), fit, valid)  # outcome known on day 5 == window start
        with self.assertRaises(AssertionError):
            assert_fit_precedes(self.frame(1), np.array([0, 4]), valid)

    def test_lowest_mae_wins_when_clear(self):
        name, _ = select_candidate(scores(forest_leaf20=4.0))
        self.assertEqual(name, "forest_leaf20")

    def test_simpler_model_wins_within_margin(self):
        name, _ = select_candidate(scores(forest_leaf20=4.0, tree_depth12=4.04, ridge_10=4.049))
        self.assertEqual(name, "ridge_10")
        name, _ = select_candidate(scores(forest_leaf20=4.0, tree_depth12=4.06))  # beyond the margin
        self.assertEqual(name, "forest_leaf20")

    def test_baselines_are_not_selectable(self):
        name, _ = select_candidate(scores(mean_baseline=1.0, route_baseline=1.0))
        self.assertIn(name, SIMPLICITY)


if __name__ == "__main__":
    unittest.main()
