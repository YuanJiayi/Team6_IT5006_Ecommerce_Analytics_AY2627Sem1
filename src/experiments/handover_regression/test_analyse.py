"""Semantic checks for prediction-time eligibility and scenario evaluation."""
import unittest

import numpy as np
import pandas as pd

from run import decision_rows, paired_comparison, queue_metrics, rank_queue


class HandoverTests(unittest.TestCase):
    def frame(self):
        # Review on Jan 3, 09:00. Promise Jan 3 means through that calendar day.
        f = pd.DataFrame({
            "order_id": ["a", "b", "c", "d"], "period": ["test"] * 4,
            "order_purchase_timestamp": pd.to_datetime(["2018-01-01"] * 4),
            "handover_time": pd.to_datetime(["2018-01-02 12:00"] * 4),
            "order_estimated_delivery_date": pd.to_datetime(["2018-01-03", "2018-01-03", "2018-01-02", "2018-01-03"]),
            "outcome_available_at": pd.to_datetime(["2018-01-04 12:00", "2018-01-03 08:00", "2018-01-05 00:00", "2018-01-03 23:59"]),
            "frozen_ridge": [3.5, 3, 4, 2.9], "selected": [3.5, 3, 4, 2.9],
            "route_rule": [4, 3, 4, 2.5]})
        return f

    def test_calendar_date_and_observable_status(self):
        f = decision_rows(self.frame())
        self.assertEqual(f.eligible.tolist(), [True, False, False, True])
        self.assertLess(f.iloc[3].actual_gap, 0)  # delivery late on promise day is on time
        self.assertEqual(f.iloc[0].review_time, pd.Timestamp("2018-01-03 09:00"))

    def test_ranking_does_not_use_outcomes(self):
        f = decision_rows(self.frame())
        changed = f.copy()
        changed["actual_gap"] = [100, -100, 50, 200]
        changed["outcome_available_at"] = pd.Timestamp("2020-01-01")
        for strategy in ["nearest_deadline", "route_slack", "largest_predicted_delay", "recoverable_ridge"]:
            self.assertEqual(rank_queue(f, strategy, 1).order_id.tolist(),
                             rank_queue(changed, strategy, 1).order_id.tolist())

    def test_equal_capacity_and_hypothetical_recovery(self):
        _, m = queue_metrics(self.frame())
        m = m[m.share.eq(.1) & m.assumed_days_saved.eq(1)]
        self.assertTrue(m.reviewed.eq(1).all())
        self.assertTrue(m.eligible.eq(2).all())
        r = m[m.strategy.eq("largest_predicted_delay")].iloc[0]
        self.assertEqual(r.hypothetical_prevented, 1)
        random = m[m.strategy.eq("random_expected")].iloc[0]
        self.assertEqual(random.hypothetical_prevented, .5)
        self.assertTrue((m.hypothetical_prevented <= m.hindsight_oracle_prevented).all())

    def test_paired_se_uses_differences(self):
        p = paired_comparison([2, 4, 6, 8, 10], [1.9, 3.9, 5.9, 7.9, 9.9])
        self.assertAlmostEqual(p["mean_gain_days"], .1)
        self.assertLess(p["paired_se_days"], 1e-12)
        self.assertFalse(p["plain_within_one_se"])


if __name__ == "__main__":
    unittest.main()
