"""Verify the saved modelling contract against raw orders and order items."""

import json
import unittest
from pathlib import Path

import numpy as np
import pandas as pd


DATA_DIR = Path(__file__).resolve().parents[1] / "data"


class Phase2FeatureContractTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.spec = json.loads((DATA_DIR / "phase2_feature_spec.json").read_text())
        cls.table = pd.read_csv(
            DATA_DIR / "phase2_order_table.csv",
            parse_dates=["order_purchase_timestamp", "outcome_available_at"],
        )

    def test_population_schema_and_missing_values(self):
        spec, table = self.spec, self.table
        features = spec["numeric_features"] + spec["categorical_features"]
        self.assertEqual(spec["schema_version"], 1)
        self.assertEqual(len(features), 18)
        self.assertEqual(len(features), len(set(features)))
        self.assertEqual(len(table), table["order_id"].nunique())
        self.assertTrue(set(features).isdisjoint({
            "order_id", "order_status", "order_approved_at", "order_delivered_carrier_date",
            "order_delivered_customer_date", "outcome_available_at", "is_late",
            "delivery_days", "split", "cv_fold", "random_split", "random_cv_fold",
        }))
        self.assertTrue(set(spec["payment_sensitivity_features"]).issubset(features))
        self.assertEqual(set(table[features].columns[table[features].isna().any()]), {
            "distance_km", "total_weight_g", "total_volume_cm3", "max_installments",
        })
        self.assertTrue(np.isfinite(table[spec["numeric_features"]].stack().to_numpy()).all())
        self.assertTrue(table[spec["categorical_features"]].notna().all().all())
        self.assertTrue(table["delivery_days"].gt(0).all())

        orders = pd.read_csv(
            DATA_DIR / "olist_orders_dataset.csv",
            usecols=["order_id", "order_status", "order_delivered_customer_date",
                     "order_estimated_delivery_date"],
        )
        eligible = orders.loc[
            orders["order_status"].eq("delivered")
            & orders["order_delivered_customer_date"].notna()
            & orders["order_estimated_delivery_date"].notna(),
            "order_id",
        ]
        self.assertEqual(set(table["order_id"]), set(eligible))

    def test_purchase_and_basket_features_match_raw_data(self):
        table = self.table.set_index("order_id")
        purchased = self.table["order_purchase_timestamp"]
        self.assertTrue(self.table["purchase_hour"].eq(purchased.dt.hour).all())
        self.assertTrue(self.table["purchase_dayofweek"].eq(purchased.dt.day_name()).all())
        self.assertTrue(self.table["purchase_month"].eq(purchased.dt.month_name()).all())

        items = pd.read_csv(
            DATA_DIR / "olist_order_items_dataset.csv",
            usecols=["order_id", "order_item_id", "product_id", "seller_id", "price", "freight_value"],
        )
        basket = items.loc[items["order_id"].isin(table.index)].groupby("order_id").agg(
            n_items=("order_item_id", "size"),
            n_products=("product_id", "nunique"),
            n_sellers=("seller_id", "nunique"),
            total_price=("price", "sum"),
            total_freight=("freight_value", "sum"),
        )
        self.assertEqual(set(basket.index), set(table.index))
        for feature in basket:
            np.testing.assert_allclose(table.loc[basket.index, feature], basket[feature])


if __name__ == "__main__":
    unittest.main()
