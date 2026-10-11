"""Verify the saved Phase 2 target against raw delivery calendar dates."""

import unittest
from pathlib import Path

import pandas as pd


DATA_DIR = Path(__file__).resolve().parents[1] / "data"


class Phase2LatenessTargetTest(unittest.TestCase):
    def test_saved_target_counts_same_date_delivery_as_on_time(self):
        table = pd.read_csv(DATA_DIR / "phase2_order_table.csv", usecols=["order_id", "is_late"])
        orders = pd.read_csv(
            DATA_DIR / "olist_orders_dataset.csv",
            usecols=[
                "order_id",
                "order_delivered_customer_date",
                "order_estimated_delivery_date",
            ],
            parse_dates=["order_delivered_customer_date", "order_estimated_delivery_date"],
        )
        joined = table.merge(orders, on="order_id", validate="one_to_one")
        self.assertEqual(len(joined), len(table))

        delivered_date = joined["order_delivered_customer_date"].dt.date
        estimated_date = joined["order_estimated_delivery_date"].dt.date
        expected = delivered_date.gt(estimated_date).astype(int)

        self.assertTrue(joined["is_late"].eq(expected).all())
        self.assertTrue(joined.loc[delivered_date.eq(estimated_date), "is_late"].eq(0).all())


if __name__ == "__main__":
    unittest.main()
