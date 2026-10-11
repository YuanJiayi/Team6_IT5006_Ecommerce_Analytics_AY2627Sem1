"""Point-in-time checks for the exploratory recent-history calculation."""

import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "experiments"))
from phase2_recent_history_audit import asof_mean  # noqa: E402


class RecentHistoryTests(unittest.TestCase):
    def test_excludes_current_and_old_outcomes(self):
        # The A order delivered at the query instant is unavailable, while the
        # older A order is outside the five-day lookback. The B order supplies
        # the point-in-time global fallback.
        actual = asof_mean(
            ["A", "A", "B"], [10.0, 20.0, 100.0],
            pd.to_datetime(["2020-01-01", "2020-01-10", "2020-01-09"]),
            ["A", "A", "C"],
            pd.to_datetime(["2020-01-10", "2020-01-11", "2020-01-11"]),
            days=5, strength=0,
        )
        np.testing.assert_allclose(actual, [100.0, 20.0, 60.0])


    def test_uses_only_observed_outcomes(self):
        actual = asof_mean(
            ["A", "A"], [5.0, 50.0],
            pd.to_datetime(["2020-01-02", "2020-02-01"]),
            ["A"], pd.to_datetime(["2020-01-20"]), days=30, strength=20,
        )
        np.testing.assert_allclose(actual, [5.0])


if __name__ == "__main__":
    unittest.main()
