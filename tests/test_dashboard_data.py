import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import pandas as pd

from dashboard_data import (
    build_review_score_correlations,
    eligible_deliveries,
    latest_reviews,
    reviewed_deliveries,
)


class DeliveryAndReviewRulesTest(unittest.TestCase):
    def test_checked_in_prepared_data_uses_calendar_dates(self):
        path = Path(__file__).resolve().parents[1] / "data/smartcommerce_consolidated.csv"
        items = pd.read_csv(
            path,
            usecols=[
                "order_status",
                "order_delivered_customer_date",
                "order_estimated_delivery_date",
                "is_on_time",
            ],
            parse_dates=[
                "order_delivered_customer_date",
                "order_estimated_delivery_date",
            ],
        )
        delivered = items.loc[
            items["order_status"].eq("delivered")
            & items["order_delivered_customer_date"].notna()
            & items["order_estimated_delivery_date"].notna()
        ]
        expected = delivered["order_delivered_customer_date"].dt.normalize().le(
            delivered["order_estimated_delivery_date"].dt.normalize()
        )
        self.assertTrue(delivered["is_on_time"].eq(expected).all())

    def test_delivery_on_estimated_calendar_date_is_on_time(self):
        orders = pd.DataFrame(
            {
                "order_id": ["same-day", "next-day"],
                "order_status": ["delivered", "delivered"],
                "order_purchase_timestamp": pd.to_datetime(
                    ["2018-01-01 10:00", "2018-01-01 10:00"]
                ),
                "order_estimated_delivery_date": pd.to_datetime(
                    ["2018-01-05", "2018-01-05"]
                ),
                "order_delivered_customer_date": pd.to_datetime(
                    ["2018-01-05 18:00", "2018-01-06 09:00"]
                ),
                # The stored Phase 1 CSV uses the old timestamp comparison.
                "is_on_time": [False, False],
            }
        )
        eligible = eligible_deliveries(orders)
        self.assertEqual(eligible["is_on_time"].tolist(), [True, False])

        reviews = pd.DataFrame(
            {
                "order_id": ["same-day", "next-day"],
                "review_id": ["a", "b"],
                "review_score": [5, 1],
                "review_creation_date": pd.to_datetime(
                    ["2018-01-07", "2018-01-07"]
                ),
                "review_answer_timestamp": pd.to_datetime(
                    ["2018-01-08", "2018-01-08"]
                ),
            }
        )
        reviewed = reviewed_deliveries(orders, reviews)
        self.assertEqual(reviewed["late_delivery"].tolist(), [False, True])
        self.assertEqual(reviewed["days_late"].tolist(), [0, 1])

    def test_latest_review_uses_answer_time_for_creation_date_ties(self):
        reviews = pd.DataFrame(
            {
                "order_id": ["order-1", "order-1"],
                "review_id": ["z", "a"],
                "review_score": [1, 5],
                "review_creation_date": pd.to_datetime(
                    ["2018-01-07", "2018-01-07"]
                ),
                "review_answer_timestamp": pd.to_datetime(
                    ["2018-01-08", "2018-01-09"]
                ),
            }
        )
        self.assertEqual(latest_reviews(reviews)["review_score"].tolist(), [5])

    def test_review_correlations_use_latest_score_not_mean_of_duplicates(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)

            def save(name, rows):
                path = root / name
                pd.DataFrame(rows).to_csv(path, index=False)
                return path

            items = save(
                "items.csv",
                {
                    "order_id": ["o1", "o2", "o3"],
                    "order_item_id": [1, 1, 1],
                    "customer_id": ["c1", "c2", "c3"],
                    "seller_id": ["s1", "s1", "s1"],
                    "price": [10, 20, 30],
                    "freight_value": [1, 1, 1],
                    "order_delivered_customer_date": ["2018-01-04"] * 3,
                    "order_estimated_delivery_date": ["2018-01-05"] * 3,
                    "delivery_days": [3, 3, 3],
                },
            )
            customers = save(
                "customers.csv",
                {
                    "customer_id": ["c1", "c2", "c3"],
                    "customer_city": ["Rio", "Rio", "Rio"],
                    "customer_state": ["RJ"] * 3,
                },
            )
            sellers = save(
                "sellers.csv",
                {"seller_id": ["s1"], "seller_city": ["Sao Paulo"], "seller_state": ["SP"]},
            )
            geo = save(
                "geo.csv",
                {
                    "geolocation_city": ["Rio", "Sao Paulo"],
                    "geolocation_state": ["RJ", "SP"],
                    "geolocation_lat": [-22.9, -23.5],
                    "geolocation_lng": [-43.2, -46.6],
                },
            )
            reviews = save(
                "reviews.csv",
                {
                    "order_id": ["o1", "o1", "o2", "o3"],
                    "review_id": ["older", "newer", "r2", "r3"],
                    "review_score": [5, 1, 5, 2],
                    "review_creation_date": ["2018-01-07"] * 4,
                    "review_answer_timestamp": [
                        "2018-01-08", "2018-01-09", "2018-01-08", "2018-01-08"
                    ],
                },
            )
            correlations = build_review_score_correlations(
                items, customers, sellers, geo, reviews
            )
            actual = correlations.loc[
                correlations["feature"].eq("Total item price"), "correlation"
            ].iloc[0]
            expected = pd.Series([10, 20, 30]).corr(pd.Series([1, 5, 2]))
            self.assertAlmostEqual(actual, expected)


if __name__ == "__main__":
    unittest.main()
