"""Unit tests for the as-of history helper used to build the route and seller features."""

import unittest

import numpy as np
import pandas as pd

from phase2_features import as_of_counts_and_sums, as_of_shrunk_mean


class AsOfHistoryTest(unittest.TestCase):
    def test_only_strictly_earlier_records_are_counted(self):
        history_time = pd.to_datetime(["2018-01-01", "2018-01-05", "2018-01-10"])
        count, total = as_of_counts_and_sums(["a", "a", "a"], [10.0, 20.0, 30.0], history_time,
                                             ["a", "a", "a", "a"],
                                             pd.to_datetime(["2017-12-31", "2018-01-05", "2018-01-06", "2018-02-01"]))
        self.assertEqual(count.tolist(), [0, 1, 2, 3])  # a record dated exactly at the query time is excluded
        self.assertEqual(total.tolist(), [0, 10, 30, 60])

    def test_an_order_never_sees_its_own_outcome(self):
        purchase = pd.to_datetime(["2018-01-01", "2018-01-03"])
        outcome = pd.to_datetime(["2018-01-04", "2018-01-06"])  # each outcome is after its own purchase
        result = as_of_shrunk_mean(["r", "r"], [5.0, 9.0], outcome, ["r", "r"], purchase, strength=1)
        self.assertTrue(np.isnan(result).all())  # no outcome is known before either purchase

    def test_shrinkage_and_unseen_keys(self):
        history_time = pd.to_datetime(["2018-01-01", "2018-01-02", "2018-01-03"])
        query_time = pd.to_datetime(["2018-02-01"] * 2)
        result = as_of_shrunk_mean(["a", "a", "b"], [10.0, 20.0, 40.0], history_time, ["a", "c"], query_time, strength=2)
        overall = (10 + 20 + 40) / 3
        self.assertAlmostEqual(result[0], (30 + overall * 2) / (2 + 2))
        self.assertAlmostEqual(result[1], overall)  # a key with no history falls back to the overall mean

    def test_later_records_do_not_change_earlier_queries(self):
        base_time = pd.to_datetime(["2018-01-01", "2018-01-02"])
        query = pd.to_datetime(["2018-01-10"])
        before = as_of_shrunk_mean(["a", "a"], [10.0, 30.0], base_time, ["a"], query, strength=3)
        extended_time = pd.to_datetime(["2018-01-01", "2018-01-02", "2018-03-01"])
        after = as_of_shrunk_mean(["a", "a", "a"], [10.0, 30.0, 999.0], extended_time, ["a"], query, strength=3)
        np.testing.assert_allclose(before, after)

    def test_missing_values_are_ignored(self):
        history_time = pd.to_datetime(["2018-01-01", "2018-01-02"])
        count, total = as_of_counts_and_sums(["a", "a"], [np.nan, 4.0], history_time, ["a"], pd.to_datetime(["2018-02-01"]))
        self.assertEqual((count[0], total[0]), (1, 4.0))


if __name__ == "__main__":
    unittest.main()
