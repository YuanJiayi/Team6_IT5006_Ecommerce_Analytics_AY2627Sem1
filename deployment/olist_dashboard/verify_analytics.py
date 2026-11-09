#!/usr/bin/env python3
"""Focused checks for the local analytics database and read-only SQL guard.

The fixture is a tiny, real CSV set written to a temporary directory, so these
tests do not need the full raw data and never touch the generated database.
"""

from __future__ import annotations

import csv
import json
import sys
import tempfile
import unittest
from datetime import date
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import analytics_db  # noqa: E402
import analytics_sql  # noqa: E402


AS_OF = date(2018, 3, 1)

HEADERS = {
    "olist_orders_dataset.csv": [
        "order_id",
        "customer_id",
        "order_status",
        "order_purchase_timestamp",
        "order_approved_at",
        "order_delivered_carrier_date",
        "order_delivered_customer_date",
        "order_estimated_delivery_date",
    ],
    "olist_order_items_dataset.csv": [
        "order_id",
        "order_item_id",
        "product_id",
        "seller_id",
        "shipping_limit_date",
        "price",
        "freight_value",
    ],
    "olist_customers_dataset.csv": [
        "customer_id",
        "customer_unique_id",
        "customer_zip_code_prefix",
        "customer_city",
        "customer_state",
    ],
    "olist_sellers_dataset.csv": [
        "seller_id",
        "seller_zip_code_prefix",
        "seller_city",
        "seller_state",
    ],
    "olist_order_payments_dataset.csv": [
        "order_id",
        "payment_sequential",
        "payment_type",
        "payment_installments",
        "payment_value",
    ],
    "olist_order_reviews_dataset.csv": [
        "review_id",
        "order_id",
        "review_score",
        "review_comment_title",
        "review_comment_message",
        "review_creation_date",
        "review_answer_timestamp",
    ],
    "olist_products_dataset.csv": [
        "product_id",
        "product_category_name",
        "product_name_lenght",
        "product_description_lenght",
        "product_photos_qty",
        "product_weight_g",
        "product_length_cm",
        "product_height_cm",
        "product_width_cm",
    ],
    "olist_geolocation_dataset.csv": [
        "geolocation_zip_code_prefix",
        "geolocation_lat",
        "geolocation_lng",
        "geolocation_city",
        "geolocation_state",
    ],
    "product_category_name_translation.csv": [
        "product_category_name",
        "product_category_name_english",
    ],
}

ROWS = {
    "olist_orders_dataset.csv": [
        ["a1a1a1a1a1a1a1a1a1a1a1a1a1a1a1a1", "b1b1b1b1b1b1b1b1b1b1b1b1b1b1b1b1", "delivered", "2018-01-01 10:00:00", "2018-01-01 11:00:00",
         "2018-01-02 10:00:00", "2018-01-05 10:00:00", "2018-01-04 00:00:00"],
        ["a2a2a2a2a2a2a2a2a2a2a2a2a2a2a2a2", "b2b2b2b2b2b2b2b2b2b2b2b2b2b2b2b2", "delivered", "2018-02-01 10:00:00", "2018-02-01 11:00:00",
         "2018-02-02 10:00:00", "2018-02-03 10:00:00", "2018-02-03 00:00:00"],
        ["a3a3a3a3a3a3a3a3a3a3a3a3a3a3a3a3", "b1b1b1b1b1b1b1b1b1b1b1b1b1b1b1b1", "canceled", "2018-02-10 10:00:00", None, None, None,
         "2018-02-20 00:00:00"],
    ],
    "olist_order_items_dataset.csv": [
        ["a1a1a1a1a1a1a1a1a1a1a1a1a1a1a1a1", "1", "c1c1c1c1c1c1c1c1c1c1c1c1c1c1c1c1", "d1d1d1d1d1d1d1d1d1d1d1d1d1d1d1d1", "2018-01-03 10:00:00", "100.0", "10.0"],
        ["a2a2a2a2a2a2a2a2a2a2a2a2a2a2a2a2", "1", "c1c1c1c1c1c1c1c1c1c1c1c1c1c1c1c1", "d1d1d1d1d1d1d1d1d1d1d1d1d1d1d1d1", "2018-02-02 10:00:00", "50.0", "5.0"],
        ["a2a2a2a2a2a2a2a2a2a2a2a2a2a2a2a2", "2", "c2c2c2c2c2c2c2c2c2c2c2c2c2c2c2c2", "d1d1d1d1d1d1d1d1d1d1d1d1d1d1d1d1", "2018-02-02 10:00:00", "25.0", "2.5"],
    ],
    "olist_customers_dataset.csv": [
        ["b1b1b1b1b1b1b1b1b1b1b1b1b1b1b1b1", "f1f1f1f1f1f1f1f1f1f1f1f1f1f1f1f1", "01000", "sao paulo", "SP"],
        ["b2b2b2b2b2b2b2b2b2b2b2b2b2b2b2b2", "f2f2f2f2f2f2f2f2f2f2f2f2f2f2f2f2", "02000", "rio", "RJ"],
    ],
    "olist_sellers_dataset.csv": [["d1d1d1d1d1d1d1d1d1d1d1d1d1d1d1d1", "01000", "sao paulo", "SP"]],
    "olist_order_payments_dataset.csv": [
        ["a1a1a1a1a1a1a1a1a1a1a1a1a1a1a1a1", "1", "credit_card", "1", "110.0"],
        ["a2a2a2a2a2a2a2a2a2a2a2a2a2a2a2a2", "1", "boleto", "1", "80.0"],
    ],
    "olist_order_reviews_dataset.csv": [
        ["e1e1e1e1e1e1e1e1e1e1e1e1e1e1e1e1", "a1a1a1a1a1a1a1a1a1a1a1a1a1a1a1a1", "5", "", "", "2018-01-06", "2018-01-07"],
        ["e2e2e2e2e2e2e2e2e2e2e2e2e2e2e2e2", "a2a2a2a2a2a2a2a2a2a2a2a2a2a2a2a2", "3", "", "", "2018-02-04", "2018-02-05"],
    ],
    "olist_products_dataset.csv": [
        ["c1c1c1c1c1c1c1c1c1c1c1c1c1c1c1c1", "informatica", "10", "100", "1", "500", "10", "10", "10"],
        ["c2c2c2c2c2c2c2c2c2c2c2c2c2c2c2c2", "moveis", "20", "200", "2", "1000", "20", "20", "20"],
    ],
    "olist_geolocation_dataset.csv": [
        ["01000", "-23.5", "-46.6", "sao paulo", "SP"],
        ["02000", "-22.9", "-43.2", "rio", "RJ"],
    ],
    "product_category_name_translation.csv": [
        ["informatica", "computers"],
        ["moveis", "furniture"],
    ],
}


def write_fixture(source_dir):
    source_dir = Path(source_dir)
    source_dir.mkdir(parents=True, exist_ok=True)
    for name, rows in ROWS.items():
        with (source_dir / name).open("w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(HEADERS[name])
            writer.writerows(rows)


def build_fixture(root):
    root = Path(root)
    source = root / "data"
    write_fixture(source)
    path = root / "analytics.sqlite"
    analytics_db.build_database(source, path, quiet=True)
    return path


class FixtureBase(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temporary = tempfile.TemporaryDirectory()
        cls.root = Path(cls.temporary.name)
        cls.path = build_fixture(cls.root)
        cls.store = analytics_sql.AnalyticsStore(cls.path)

    @classmethod
    def tearDownClass(cls):
        cls.temporary.cleanup()


class DatabaseTests(FixtureBase):
    def test_build_is_deterministic_and_metadata_is_present(self):
        second = build_fixture(self.root / "second")
        metadata = analytics_sql.load_metadata(self.path)
        second_metadata = analytics_sql.load_metadata(second)
        self.assertEqual(metadata["row_counts"], second_metadata["row_counts"])
        self.assertEqual(metadata["event_bounds"], second_metadata["event_bounds"])
        self.assertEqual(metadata["format_version"], 2)
        self.assertEqual(
            metadata["row_counts"],
            {
                "orders": 3,
                "order_items": 3,
                "customers": 2,
                "sellers": 1,
                "payments": 2,
                "reviews": 2,
                "products": 2,
                "geolocation": 2,
                "product_category_translation": 2,
            },
        )
        self.assertNotIn("smartcommerce", json.dumps(metadata).lower())

    def test_real_aggregate_results_and_cutoff(self):
        by_state = self.store.execute(
            "SELECT c.customer_state AS state, COUNT(DISTINCT o.order_id) AS orders "
            "FROM orders o JOIN customers c ON c.customer_id = o.customer_id "
            "WHERE o.order_purchase_timestamp < :as_of "
            "GROUP BY c.customer_state ORDER BY orders DESC",
            AS_OF,
        )
        self.assertEqual(
            by_state["rows"],
            [{"state": "SP", "orders": 2}, {"state": "RJ", "orders": 1}],
        )

        before = self.store.execute(
            "SELECT c.customer_state AS state, COUNT(DISTINCT o.order_id) AS orders "
            "FROM orders o JOIN customers c ON c.customer_id = o.customer_id "
            "WHERE o.order_purchase_timestamp < :as_of "
            "GROUP BY c.customer_state ORDER BY orders DESC",
            date(2018, 2, 1),
        )
        self.assertEqual(before["rows"], [{"state": "SP", "orders": 1}])

        item_price = self.store.execute(
            "SELECT ROUND(SUM(oi.price), 2) AS item_price "
            "FROM order_items oi JOIN orders o ON o.order_id = oi.order_id "
            "WHERE o.order_purchase_timestamp < :as_of",
            AS_OF,
        )
        self.assertEqual(item_price["rows"], [{"item_price": 175.0}])

        delivery = self.store.execute(
            "SELECT SUM(CASE WHEN date(o.order_delivered_customer_date) <= "
            "date(o.order_estimated_delivery_date) THEN 1 ELSE 0 END) AS on_time, "
            "COUNT(*) AS delivered FROM orders o "
            "WHERE o.order_delivered_customer_date IS NOT NULL "
            "AND o.order_delivered_customer_date < :as_of",
            AS_OF,
        )
        self.assertEqual(delivery["rows"], [{"on_time": 1, "delivered": 2}])

        monthly = self.store.execute(
            "SELECT strftime('%Y-%m', o.order_purchase_timestamp) AS month, "
            "COUNT(DISTINCT o.order_id) AS orders FROM orders o "
            "WHERE o.order_purchase_timestamp < :as_of "
            "GROUP BY month ORDER BY month",
            AS_OF,
        )
        self.assertEqual(
            monthly["rows"],
            [
                {"month": "2018-01", "orders": 1},
                {"month": "2018-02", "orders": 2},
            ],
        )

        categories = self.store.execute(
            "SELECT t.product_category_name_english AS category, "
            "ROUND(SUM(oi.price), 2) AS item_price FROM order_items oi "
            "JOIN orders o ON o.order_id = oi.order_id "
            "JOIN products p ON p.product_id = oi.product_id "
            "LEFT JOIN product_category_translation t "
            "ON t.product_category_name = p.product_category_name "
            "WHERE o.order_purchase_timestamp < :as_of "
            "GROUP BY category ORDER BY item_price DESC",
            AS_OF,
        )
        self.assertEqual(
            categories["rows"],
            [{"category": "computers", "item_price": 150.0},
             {"category": "furniture", "item_price": 25.0}],
        )

        reviews = self.store.execute(
            "SELECT ROUND(AVG(r.review_score), 2) AS average_score "
            "FROM reviews r JOIN orders o ON o.order_id = r.order_id "
            "WHERE o.order_purchase_timestamp < :as_of",
            AS_OF,
        )
        self.assertEqual(reviews["rows"], [{"average_score": 4.0}])

    def test_on_time_uses_calendar_dates_for_the_promise(self):
        same_day = self.store.execute(
            "SELECT COUNT(*) AS on_time FROM orders "
            "WHERE date(order_delivered_customer_date) = "
            "date(order_estimated_delivery_date) "
            "AND date(:as_of) IS NOT NULL",
            AS_OF,
        )
        self.assertEqual(same_day["rows"], [{"on_time": 1}])

        late = self.store.execute(
            "SELECT COUNT(*) AS late FROM orders "
            "WHERE date(order_delivered_customer_date) > "
            "date(order_estimated_delivery_date) "
            "AND date(:as_of) IS NOT NULL",
            AS_OF,
        )
        self.assertEqual(late["rows"], [{"late": 1}])

    def test_as_of_views_filter_cohorts_and_mask_future_events(self):
        later = date(2018, 2, 2)
        orders = self.store.execute(
            "SELECT COUNT(*) AS orders FROM orders WHERE date(:as_of) IS NOT NULL",
            later,
        )
        self.assertEqual(orders["rows"], [{"orders": 2}])
        delivered = self.store.execute(
            "SELECT COUNT(*) AS delivered FROM orders "
            "WHERE order_delivered_customer_date IS NOT NULL "
            "AND date(:as_of) IS NOT NULL",
            later,
        )
        self.assertEqual(delivered["rows"], [{"delivered": 1}])
        reviews = self.store.execute(
            "SELECT COUNT(*) AS reviews FROM reviews WHERE date(:as_of) IS NOT NULL",
            later,
        )
        self.assertEqual(reviews["rows"], [{"reviews": 1}])
        items = self.store.execute(
            "SELECT COUNT(*) AS items FROM order_items WHERE date(:as_of) IS NOT NULL",
            later,
        )
        self.assertEqual(items["rows"], [{"items": 3}])
        payments = self.store.execute(
            "SELECT COUNT(*) AS payments FROM payments WHERE date(:as_of) IS NOT NULL",
            later,
        )
        self.assertEqual(payments["rows"], [{"payments": 2}])

        earlier = date(2018, 2, 1)
        for sql, column, expected in (
            ("SELECT COUNT(*) AS orders FROM orders WHERE date(:as_of) IS NOT NULL", "orders", 1),
            ("SELECT COUNT(*) AS items FROM order_items WHERE date(:as_of) IS NOT NULL", "items", 1),
            ("SELECT COUNT(*) AS payments FROM payments WHERE date(:as_of) IS NOT NULL", "payments", 1),
            ("SELECT COUNT(*) AS reviews FROM reviews WHERE date(:as_of) IS NOT NULL", "reviews", 1),
        ):
            result = self.store.execute(sql, earlier)
            self.assertEqual(result["rows"], [{column: expected}], sql)

        pending_answer = self.store.execute(
            "SELECT COUNT(*) AS answered FROM reviews "
            "WHERE review_answer_timestamp IS NOT NULL "
            "AND date(:as_of) IS NOT NULL",
            date(2018, 1, 7),
        )
        self.assertEqual(pending_answer["rows"], [{"answered": 0}])
        settled_answer = self.store.execute(
            "SELECT COUNT(*) AS answered FROM reviews "
            "WHERE review_answer_timestamp IS NOT NULL "
            "AND date(:as_of) IS NOT NULL",
            date(2018, 1, 8),
        )
        self.assertEqual(settled_answer["rows"], [{"answered": 1}])

    def test_review_score_is_hidden_until_the_answer_is_before_the_cutoff(self):
        pending = self.store.execute(
            "SELECT COUNT(*) AS reviews, MAX(review_score) AS score "
            "FROM reviews WHERE date(:as_of) IS NOT NULL",
            date(2018, 1, 7),
        )
        self.assertEqual(pending["rows"], [{"reviews": 0, "score": None}])
        answered = self.store.execute(
            "SELECT COUNT(*) AS reviews, MAX(review_score) AS score "
            "FROM reviews WHERE date(:as_of) IS NOT NULL",
            date(2018, 1, 8),
        )
        self.assertEqual(answered["rows"], [{"reviews": 1, "score": 5}])

    def test_order_status_is_masked_to_historically_visible_delivery(self):
        later = date(2018, 2, 2)
        result = self.store.execute(
            "SELECT order_status FROM orders WHERE date(:as_of) IS NOT NULL",
            later,
        )
        statuses = [row["order_status"] for row in result["rows"]]
        self.assertEqual(statuses.count("delivered"), 1)
        self.assertEqual(statuses.count(None), 1)
        self.assertNotIn("canceled", statuses)

        final = self.store.execute(
            "SELECT order_status FROM orders WHERE date(:as_of) IS NOT NULL",
            AS_OF,
        )
        final_statuses = [row["order_status"] for row in final["rows"]]
        self.assertEqual(final_statuses.count("delivered"), 2)
        self.assertEqual(final_statuses.count(None), 1)
        self.assertNotIn("canceled", final_statuses)

    def test_schema_context_has_joins_metrics_and_cutoff(self):
        context = self.store.schema_context(AS_OF)
        self.assertIn("orders.customer_id = customers.customer_id", context)
        self.assertIn("order_items.order_id = orders.order_id", context)
        self.assertIn("item price is not revenue", context.lower())
        self.assertIn(":as_of", context)
        self.assertIn("2018-03-01", context)
        self.assertIn("[blocked]", context)
        self.assertIn("_base_*", context)
        self.assertIn("cannot be reconstructed", context.lower())
        self.assertIn("date(order_delivered_customer_date) <=", context)
        self.assertIn("on the promised date is on time", context)
        self.assertIn("'delivered' only when the customer-delivery event", context)
        self.assertIn("do not treat null as a cancellation", context.lower())
        self.assertIn("Identifier-shaped", context)
        self.assertIn("review_answer_timestamp", context)
        self.assertIn("constant-output projections", context)
        self.assertNotIn("smartcommerce", context)

    def test_missing_source_is_a_clear_error(self):
        with self.assertRaises(analytics_db.BuildError) as context:
            analytics_db.build_database(self.root / "absent", self.root / "x.sqlite")
        self.assertIn("Source directory not found", str(context.exception))


class SqlGuardTests(FixtureBase):
    def assert_blocked(self, sql, codes=("sql_rejected", "sql_error", "privacy_blocked", "invalid_result")):
        with self.assertRaises(analytics_sql.QueryError, msg=sql) as context:
            self.store.execute(sql, AS_OF)
        self.assertIn(context.exception.code, codes, sql)
        self.assertFalse(self.store.path.with_name("analytics.sqlite-journal").exists())

    def test_product_view_adds_plain_english_display_name(self):
        result = self.store.execute(
            "SELECT product_category_display AS category FROM products "
            "WHERE :as_of IS NOT NULL ORDER BY category",
            AS_OF,
        )
        self.assertEqual(
            [row["category"] for row in result["rows"]],
            ["Computers", "Furniture"],
        )
        schema = self.store.schema_context(AS_OF)
        self.assertIn("product_category_display", schema)
        self.assertIn("Bed bath table", schema)

    def test_window_start_bind_parameter_runs_and_is_required(self):
        sql = (
            "SELECT COUNT(DISTINCT o.order_id) AS orders FROM orders o "
            "WHERE o.order_purchase_timestamp >= "
            "COALESCE(:window_start, date(:as_of, '-30 days')) "
            "AND o.order_purchase_timestamp < :as_of"
        )
        default = self.store.execute(sql, AS_OF, {"window_start": None})
        self.assertEqual(default["rows"][0]["orders"], 2)
        broad = self.store.execute(sql, AS_OF, {"window_start": "2018-01-01"})
        self.assertEqual(broad["rows"][0]["orders"], 3)
        with self.assertRaises(analytics_sql.QueryError) as context:
            self.store.execute(sql, AS_OF)
        self.assertEqual(context.exception.code, "sql_rejected")
        self.assertIn("bind", context.exception.message.lower())

    def test_writes_escape_hatches_and_internal_tables_are_blocked(self):
        attempts = [
            "DELETE FROM orders WHERE order_id < :as_of",
            "UPDATE orders SET order_status = 'x' WHERE order_id < :as_of",
            "INSERT INTO orders (order_id) VALUES ('x')",
            "DROP TABLE orders",
            "CREATE TABLE injected (x TEXT)",
            "ATTACH DATABASE '/tmp/injected.sqlite' AS injected",
            "PRAGMA table_info(orders)",
            "SELECT * FROM sqlite_master WHERE name < :as_of",
            "SELECT * FROM _vamos_meta WHERE key < :as_of",
            "SELECT load_extension('anything') WHERE 'x' < :as_of",
            "SELECT * FROM orders WHERE order_id < :as_of; DROP TABLE orders",
        ]
        for sql in attempts:
            self.assert_blocked(sql)
        count = self.store.execute(
            "SELECT COUNT(*) AS n FROM orders WHERE date(:as_of) IS NOT NULL", AS_OF
        )
        self.assertEqual(count["rows"], [{"n": 3}])

    def test_personal_columns_and_identifier_output_are_blocked(self):
        attempts = [
            "SELECT customer_unique_id FROM customers WHERE customer_id < :as_of",
            "SELECT customer_zip_code_prefix FROM customers WHERE customer_id < :as_of",
            "SELECT geolocation_lat FROM geolocation WHERE geolocation_state < :as_of",
            "SELECT review_comment_message FROM reviews WHERE order_id < :as_of",
            "SELECT customer_id FROM orders WHERE order_id < :as_of",
            "SELECT order_id FROM orders WHERE order_id < :as_of",
            "SELECT review_id FROM reviews WHERE order_id < :as_of",
            "SELECT * FROM orders WHERE order_id < :as_of",
        ]
        for sql in attempts:
            self.assert_blocked(sql)

    def test_cutoff_multiple_statements_and_duplicate_columns(self):
        self.assert_blocked("SELECT COUNT(*) AS n FROM orders")
        self.assert_blocked(
            "SELECT customer_state AS state FROM customers "
            "WHERE customer_id < :as_of; SELECT 1"
        )
        self.assert_blocked(
            "SELECT customer_state AS state, customer_state AS state "
            "FROM customers WHERE customer_id < :as_of GROUP BY state"
        )

    def test_cutoff_is_enforced_by_views_not_by_sql_text(self):
        cutoff = date(2018, 2, 1)
        bypass = self.store.execute(
            "SELECT COUNT(*) AS orders FROM orders WHERE date(:as_of) IS NOT NULL",
            cutoff,
        )
        self.assertEqual(bypass["rows"], [{"orders": 1}])
        self.assert_blocked("SELECT COUNT(*) AS orders FROM orders /* :as_of */")
        self.assert_blocked(
            "SELECT COUNT(*) AS orders FROM orders "
            "WHERE 'x :as_of' = 'x :as_of'"
        )
        self.assert_blocked("SELECT COUNT(*) AS orders FROM orders -- :as_of\n")

    def test_private_base_and_unknown_tables_are_blocked(self):
        attempts = [
            "SELECT COUNT(*) AS n FROM _base_orders WHERE date(:as_of) IS NOT NULL",
            "SELECT COUNT(*) AS n FROM main._base_orders WHERE date(:as_of) IS NOT NULL",
            'SELECT COUNT(*) AS n FROM "main"."_base_orders" '
            "WHERE date(:as_of) IS NOT NULL",
            "WITH orders AS (SELECT COUNT(*) AS n FROM main._base_orders) "
            "SELECT n FROM orders WHERE :as_of IS NOT NULL",
            "WITH orders AS (SELECT COUNT(*) AS n FROM \"_base_orders\") "
            "SELECT n FROM orders WHERE :as_of IS NOT NULL",
            "WITH orders AS (SELECT COUNT(*) AS n FROM [_base_orders]) "
            "SELECT n FROM orders WHERE :as_of IS NOT NULL",
            "SELECT COUNT(*) AS n FROM _vamos_meta WHERE date(:as_of) IS NOT NULL",
            "SELECT COUNT(*) AS n FROM pragma_table_info('orders') "
            "WHERE date(:as_of) IS NOT NULL",
            "SELECT COUNT(*) AS n FROM dbstat WHERE date(:as_of) IS NOT NULL",
        ]
        for sql in attempts:
            self.assert_blocked(sql)
        public_cte = self.store.execute(
            "WITH recent AS (SELECT COUNT(*) AS n FROM orders "
            "WHERE date(:as_of) IS NOT NULL) "
            "SELECT n FROM recent WHERE :as_of IS NOT NULL",
            AS_OF,
        )
        self.assertEqual(public_cte["rows"], [{"n": 3}])

    def test_cte_shadowing_private_base_probe_is_blocked_at_reported_cutoff(self):
        cutoff = date(2018, 2, 1)
        visible = self.store.execute(
            "SELECT COUNT(*) AS n FROM orders WHERE :as_of IS NOT NULL", cutoff
        )
        self.assertEqual(visible["rows"], [{"n": 1}])
        probe = (
            "WITH orders AS (SELECT COUNT(*) AS n FROM main._base_orders) "
            "SELECT n FROM orders WHERE :as_of IS NOT NULL"
        )
        with self.assertRaises(analytics_sql.QueryError) as context:
            self.store.execute(probe, cutoff)
        self.assertEqual(context.exception.code, "sql_rejected")

    def test_identifier_values_are_blocked_even_aliased_or_concatenated(self):
        attempts = [
            "SELECT order_id AS label FROM orders WHERE date(:as_of) IS NOT NULL",
            "SELECT 'id:' || order_id AS label FROM orders "
            "WHERE date(:as_of) IS NOT NULL",
            "SELECT customer_id AS customer FROM orders "
            "WHERE date(:as_of) IS NOT NULL",
            "SELECT upper(order_id) AS label FROM orders "
            "WHERE date(:as_of) IS NOT NULL",
            "SELECT group_concat(order_id, ',') AS ids FROM orders "
            "WHERE date(:as_of) IS NOT NULL",
            "SELECT substr(order_id, 1, 8) AS short_id FROM orders "
            "WHERE date(:as_of) IS NOT NULL",
            "SELECT replace(order_id, 'a', 'b') AS label FROM orders "
            "WHERE date(:as_of) IS NOT NULL",
        ]
        for sql in attempts:
            self.assert_blocked(sql)

    def test_constant_only_query_is_rejected(self):
        with self.assertRaises(analytics_sql.QueryError) as context:
            self.store.execute(
                "SELECT 999999 AS orders WHERE :as_of IS NOT NULL", AS_OF
            )
        self.assertIn(context.exception.code, {"invalid_query", "sql_rejected"})

    def test_constant_output_projections_are_rejected(self):
        attempts = [
            "SELECT 999999 AS orders FROM orders "
            "WHERE :as_of IS NOT NULL LIMIT 1",
            "SELECT (1000000 - 1) AS orders FROM orders "
            "WHERE :as_of IS NOT NULL",
            "SELECT '999999' AS orders FROM orders WHERE :as_of IS NOT NULL",
            "SELECT length('abcdef') AS orders FROM orders "
            "WHERE :as_of IS NOT NULL",
        ]
        for sql in attempts:
            with self.assertRaises(analytics_sql.QueryError, msg=sql) as context:
                self.store.execute(sql, AS_OF)
            self.assertEqual(context.exception.code, "sql_rejected", sql)
        preserved = self.store.execute(
            "SELECT COUNT(*) AS orders, "
            "ROUND(SUM(oi.price) / COUNT(*), 2) AS avg_item_price "
            "FROM order_items oi WHERE date(:as_of) IS NOT NULL",
            AS_OF,
        )
        self.assertEqual(
            preserved["rows"], [{"orders": 3, "avg_item_price": 58.33}]
        )

    def test_row_column_and_output_bounds(self):
        one_row = analytics_sql.AnalyticsStore(self.path, max_rows=1)
        with self.assertRaises(analytics_sql.QueryError) as context:
            one_row.execute(
                "SELECT customer_state AS state FROM customers "
                "WHERE date(:as_of) IS NOT NULL GROUP BY customer_state",
                AS_OF,
            )
        self.assertEqual(context.exception.code, "row_limit")

        two_columns = analytics_sql.AnalyticsStore(self.path, max_columns=2)
        with self.assertRaises(analytics_sql.QueryError) as context:
            two_columns.execute(
                "SELECT customer_state AS a, customer_state AS b, "
                "customer_state AS c FROM customers "
                "WHERE date(:as_of) IS NOT NULL GROUP BY customer_state",
                AS_OF,
            )
        self.assertEqual(context.exception.code, "column_limit")


if __name__ == "__main__":
    unittest.main(verbosity=2)
