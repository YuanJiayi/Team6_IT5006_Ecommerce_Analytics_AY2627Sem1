"""Point-in-time and selection checks for the two-stage delivery-time estimate, on synthetic data."""

import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from phase2_eta import SIMPLICITY, STAGE_FEATURES, add_stage_features, select_candidate  # noqa: E402


def synthetic():
    """Five same-route orders. Transit (carrier to customer) days are 4, 6, 20, 2 and 8."""
    table = pd.DataFrame({
        "order_id": list("abcde"),
        "order_purchase_timestamp": pd.to_datetime(["2018-01-01", "2018-01-02", "2018-01-03", "2018-01-12",
                                                    "2018-01-20"]),
        "outcome_available_at": pd.to_datetime(["2018-01-06", "2018-01-10", "2018-01-25", "2018-01-16",
                                                "2018-01-30"]),
        "seller_state": "SP", "customer_state": "RJ",
    })
    orders = pd.DataFrame({
        "order_id": list("abcde"),
        "order_approved_at": pd.to_datetime(["2018-01-01 12:00", "2018-01-05 00:00", "2018-01-03 06:00",
                                             "2018-01-12 01:00", "2018-01-20 02:00"]),
        "order_delivered_carrier_date": pd.to_datetime(["2018-01-02", "2018-01-04", "2018-01-05", "2018-01-14",
                                                        "2018-01-22"]),
    })
    return table, orders


class StageFeatureTests(unittest.TestCase):
    def setUp(self):
        self.staged = add_stage_features(*synthetic()).set_index("order_id")

    def test_elapsed_days_and_unknown_approval(self):
        np.testing.assert_allclose(self.staged["handover_days"], [1, 2, 2, 2, 2])
        self.assertAlmostEqual(self.staged.loc["a", "approval_days"], 0.5)
        # Order b was approved after the carrier handover, so approval is not known at that point.
        self.assertTrue(np.isnan(self.staged.loc["b", "approval_days"]))

    def test_history_uses_only_deliveries_already_recorded(self):
        staged = self.staged
        # Nothing was delivered before the first three purchases or handovers.
        self.assertTrue(staged.loc[list("abc"), "route_transit_90d"].isna().all())
        self.assertTrue(staged.loc[list("abc"), "route_transit_90d_at_handover"].isna().all())
        # Order d (purchased 12 Jan, handed over 14 Jan): a and b are delivered; c (25 Jan) is not.
        self.assertAlmostEqual(staged.loc["d", "route_transit_90d"], 5.0)
        self.assertAlmostEqual(staged.loc["d", "route_transit_90d_at_handover"], 5.0)
        # Order e (purchased 20 Jan): a, b and d are delivered; c is still in transit, its own outcome is excluded.
        self.assertAlmostEqual(staged.loc["e", "route_transit_90d"], 4.0)
        self.assertAlmostEqual(staged.loc["e", "route_transit_90d_at_handover"], 4.0)

    def test_history_at_handover_can_be_fresher_than_at_purchase(self):
        table, orders = synthetic()
        orders.loc[orders["order_id"].eq("e"), "order_delivered_carrier_date"] = pd.Timestamp("2018-01-26")
        staged = add_stage_features(table, orders).set_index("order_id")
        self.assertAlmostEqual(staged.loc["e", "route_transit_90d"], 4.0)             # a, b, d
        self.assertAlmostEqual(staged.loc["e", "route_transit_90d_at_handover"], 8.0)  # a, b, c, d

    def test_invalid_handover_is_flagged(self):
        table, orders = synthetic()
        orders.loc[0, "order_delivered_carrier_date"] = pd.Timestamp("2017-12-31")  # before purchase
        orders.loc[1, "order_delivered_carrier_date"] = pd.Timestamp("2018-01-11")  # after delivery
        orders.loc[2, "order_delivered_carrier_date"] = pd.NaT
        staged = add_stage_features(table, orders).set_index("order_id")
        self.assertEqual(staged["handover_valid"].tolist(), [False, False, False, True, True])
        self.assertTrue(staged.loc[list("abc"), "handover_days"].isna().all())

    def test_checkout_stage_adds_nothing_and_no_stage_uses_the_outcome(self):
        self.assertEqual(STAGE_FEATURES["checkout"], [])
        used = {feature for features in STAGE_FEATURES.values() for feature in features}
        self.assertFalse(used & {"transit_days", "delivery_days", "outcome_available_at", "is_late"})


class SelectionTests(unittest.TestCase):
    def scores(self, **overrides):
        base = {name: 5.0 for name in SIMPLICITY}
        base.update({"mean_baseline": 1.0, "elapsed_plus_route_baseline": 1.0, "voting_family_winners": 1.0})
        base.update(overrides)
        return base

    def test_simpler_model_wins_within_margin(self):
        chosen, lowest, _ = select_candidate(self.scores(boost_slow=4.0, ridge_10=4.049))
        self.assertEqual((chosen, lowest), ("ridge_10", "boost_slow"))

    def test_lowest_mae_wins_beyond_margin(self):
        chosen, _, _ = select_candidate(self.scores(boost_slow=4.0, ridge_10=4.06))
        self.assertEqual(chosen, "boost_slow")

    def test_baselines_and_ensemble_are_not_selectable(self):
        chosen, _, _ = select_candidate(self.scores())
        self.assertIn(chosen, SIMPLICITY)


if __name__ == "__main__":
    unittest.main()
