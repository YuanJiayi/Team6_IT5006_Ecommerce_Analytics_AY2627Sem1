"""Timing, formula and selection checks for the handover late classifier, on synthetic data."""

import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from phase2_eta import add_stage_features  # noqa: E402
from phase2_handover_classifier import (K_GRID, add_history_extension, add_remaining_slack, choose_k,  # noqa: E402
                                        select_candidate, top_flags, weighted_pr_auc, weighted_recall_top)
from test_phase2_eta import synthetic  # noqa: E402


class HandoverInputTests(unittest.TestCase):
    def setUp(self):
        table, orders = synthetic()
        self.staged = add_stage_features(table, orders)

    def test_approval_after_handover_is_unknown(self):
        staged = self.staged.set_index("order_id")
        self.assertTrue(np.isnan(staged.loc["b", "approval_days"]))
        self.assertFalse(staged["approval_days"].drop("b").isna().any())

    def test_transit_history_excludes_deliveries_after_handover(self):
        table, orders = synthetic()
        orders.loc[orders["order_id"].eq("e"), "order_delivered_carrier_date"] = pd.Timestamp("2018-01-26")
        staged = add_stage_features(table, orders).set_index("order_id")
        # c is delivered 25 Jan, before e's 26 Jan handover, so it joins the history (4, 6, 20, 2 -> shrunk mean).
        self.assertGreater(staged.loc["e", "route_transit_90d_at_handover"], staged.loc["e", "route_transit_90d"])

    def test_remaining_slack_formula_and_unknown_transit(self):
        frame = pd.DataFrame({"promised_days": [20.0, 20.0], "handover_days": [3.0, 3.0],
                              "route_transit_90d_at_handover": [10.0, np.nan]})
        slack = add_remaining_slack(frame)["remaining_slack"]
        self.assertAlmostEqual(slack[0], 7.0)
        self.assertTrue(np.isnan(slack[1]))

    def test_extension_history_is_strictly_before_handover(self):
        staged = self.staged
        staged["furthest_seller_id"] = "s1"
        zips = pd.DataFrame({"order_id": list("abcde"), "customer_zip_code_prefix": 12345})
        out = add_history_extension(staged, zips).set_index("order_id")
        # Seller handover speed: nothing earlier than a's handover (2 Jan); b (handover 4 Jan) sees only a (1 day).
        self.assertTrue(np.isnan(out.loc["a", "seller_handover_90d_at_handover"]))
        self.assertAlmostEqual(out.loc["b", "seller_handover_90d_at_handover"], 1.0)
        # ZIP transit: nothing delivered before c's handover (5 Jan); a is delivered 6 Jan so it cannot inform c.
        self.assertTrue(np.isnan(out.loc["c", "zip3_transit_90d_at_handover"]))
        # d (handover 14 Jan): a (4 days) and b (6 days) delivered, c (25 Jan) not.
        self.assertAlmostEqual(out.loc["d", "zip3_transit_90d_at_handover"], 5.0)


class OperatingPointTests(unittest.TestCase):
    def test_k_is_chosen_from_validation_windows_only(self):
        # Perfect ranking, 10% late: acting on exactly 10% maximises 5*caught - acted.
        y = np.r_[np.ones(10), np.zeros(90)]
        scores = np.r_[np.linspace(1, 0.9, 10), np.linspace(0.5, 0, 90)]
        k, curve = choose_k([(y, scores), (y, scores)])
        self.assertAlmostEqual(k, 0.10)
        self.assertEqual(len(curve), len(K_GRID))

    def test_higher_benefit_acts_on_more(self):
        rng = np.random.default_rng(0)
        y = (rng.random(2000) < 0.1).astype(int)
        scores = y * 0.5 + rng.random(2000)
        self.assertGreaterEqual(choose_k([(y, scores)], 20)[0], choose_k([(y, scores)], 2)[0])

    def test_top_flags_counts(self):
        self.assertEqual(top_flags(np.arange(100.0), 0.1).sum(), 10)
        self.assertTrue(top_flags(np.arange(100.0), 0.1)[-10:].all())


class SelectionAndBootstrapTests(unittest.TestCase):
    def test_simpler_within_margin_wins(self):
        chosen, best, _ = select_candidate({"logistic": 0.30, "logistic_balanced": 0.29, "tree_depth6": 0.20,
                                            "forest_leaf20": 0.303})
        self.assertEqual((chosen, best), ("logistic", "forest_leaf20"))
        chosen, _, _ = select_candidate({"logistic": 0.30, "logistic_balanced": 0.29, "tree_depth6": 0.20,
                                         "forest_leaf20": 0.40})
        self.assertEqual(chosen, "forest_leaf20")

    def test_weighted_metrics_match_sklearn_at_unit_weights(self):
        rng = np.random.default_rng(1)
        y = (rng.random(500) < 0.2).astype(int)
        s = rng.random(500) + y
        self.assertAlmostEqual(weighted_pr_auc(y, s, np.ones(500)), average_precision_score(y, s), places=6)
        order = np.argsort(-s)
        self.assertAlmostEqual(weighted_recall_top(y, s, np.ones(500), 0.1), y[order[:50]].sum() / y.sum())


if __name__ == "__main__":
    unittest.main()
