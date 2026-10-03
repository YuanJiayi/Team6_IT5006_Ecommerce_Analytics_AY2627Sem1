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
        self.assertEqual(spec["schema_version"], 2)
        self.assertEqual(len(features), 20)
        self.assertNotIn("purchase_month", features)
        self.assertIn("purchase_month", spec["retired_features"])
        self.assertEqual(len(features), len(set(features)))
        self.assertEqual(len(table), table["order_id"].nunique())
        self.assertTrue(set(features).isdisjoint({
            "order_id", "order_status", "order_approved_at", "order_delivered_carrier_date",
            "order_delivered_customer_date", "outcome_available_at", "is_late",
            "delivery_days", "split", "cv_fold", "random_split", "random_cv_fold", "furthest_seller_id",
        }))
        self.assertTrue(set(spec["payment_sensitivity_features"]).issubset(features))
        self.assertEqual(set(table[features].columns[table[features].isna().any()]), {
            "distance_km", "total_weight_g", "total_volume_cm3", "max_installments",
            "route_typical_days", "promise_slack", "seller_ship_days",
        })
        self.assertTrue(np.isfinite(table[spec["numeric_features"]].stack().dropna().to_numpy()).all())
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
        self.assertTrue(self.table["purchase_month"].eq(purchased.dt.month_name()).all())  # reference column only

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


    def test_route_history_feature_uses_only_orders_delivered_before_purchase(self):
        """Recompute route_typical_days by brute force for a sample of orders."""
        table = self.table
        params = self.spec["history_features"]
        strength = params["route_prior_strength"]
        route = table["seller_state"] + ">" + table["customer_state"]
        sample = table.sample(150, random_state=0).index
        for i in sample:
            known = table["outcome_available_at"] < table.at[i, "order_purchase_timestamp"]
            overall = table.loc[known, "delivery_days"].mean() if known.any() else np.nan
            same = known & route.eq(route[i])
            expected = overall if not same.any() else (
                (table.loc[same, "delivery_days"].sum() + overall * strength) / (same.sum() + strength))
            actual = table.at[i, "route_typical_days"]
            if np.isnan(expected):
                self.assertTrue(np.isnan(actual))
            else:
                self.assertAlmostEqual(actual, expected, places=8)
            if not np.isnan(expected):
                self.assertAlmostEqual(table.at[i, "promise_slack"], table.at[i, "promised_days"] - expected, places=8)

    def test_seller_ship_days_uses_only_handovers_recorded_before_purchase(self):
        """Recompute seller_ship_days by brute force from the raw carrier timestamps for a sample of orders."""
        table = self.table
        params = self.spec["history_features"]
        low, high = params["seller_ship_days_valid_range"]
        strength = params["seller_prior_strength"]
        raw = pd.read_csv(DATA_DIR / "olist_orders_dataset.csv", usecols=["order_id", "order_delivered_carrier_date"],
                          parse_dates=["order_delivered_carrier_date"]).set_index("order_id")
        carrier = table["order_id"].map(raw["order_delivered_carrier_date"])
        ship = (carrier - table["order_purchase_timestamp"]).dt.total_seconds() / 86_400
        usable = ship.between(low, high)

        items = pd.read_csv(DATA_DIR / "olist_order_items_dataset.csv", usecols=["order_id", "seller_id"])
        sellers_of_order = set(zip(items["order_id"], items["seller_id"]))
        self.assertTrue(all(pair in sellers_of_order for pair in zip(table["order_id"], table["furthest_seller_id"])))

        for i in table.sample(150, random_state=1).index:
            known = usable & carrier.lt(table.at[i, "order_purchase_timestamp"])
            overall = ship[known].mean() if known.any() else np.nan
            same = known & table["furthest_seller_id"].eq(table.at[i, "furthest_seller_id"])
            expected = overall if not same.any() else (ship[same].sum() + overall * strength) / (same.sum() + strength)
            actual = table.at[i, "seller_ship_days"]
            if np.isnan(expected):
                self.assertTrue(np.isnan(actual))
            else:
                self.assertAlmostEqual(actual, expected, places=8)

    def test_history_features_are_missing_only_for_the_earliest_orders(self):
        table = self.table.sort_values("order_purchase_timestamp")
        for feature in self.spec["history_features"]["features"]:
            missing = table.loc[table[feature].isna(), "order_purchase_timestamp"]
            self.assertLess(len(missing), 400)
            self.assertLess(missing.max(), pd.Timestamp("2017-02-01"))


if __name__ == "__main__":
    unittest.main()
