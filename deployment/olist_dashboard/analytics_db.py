#!/usr/bin/env python3
"""Deterministic CSV -> SQLite builder for the private Olist base tables.

The generated database is a local, disposable read model for Ask. It is never
committed and contains no credentials. Rebuilds are atomic: the new database is
written to a temporary file and only replaces the old one after every source
file has been loaded and indexed.

Run from the repository root:
    python3 deployment/olist_dashboard/analytics_db.py
    python3 deployment/olist_dashboard/analytics_db.py --source data/raw \
        --output deployment/olist_dashboard/data/analytics.sqlite
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sqlite3
import sys
from pathlib import Path

FORMAT_VERSION = 2
BASE_PREFIX = "_base_"
ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SOURCE = ROOT / "data" / "raw"
DEFAULT_OUTPUT = Path(__file__).resolve().parent / "data" / "analytics.sqlite"
BATCH_SIZE = 5000


class BuildError(Exception):
    """A source file is missing or could not be loaded."""


TABLES = {
    "orders": {
        "source": "olist_orders_dataset.csv",
        "description": "One row per order purchased before the cutoff with purchase, handover and delivery event timestamps. The delivery promise is a calendar date. The public order_status is 'delivered' only when the customer-delivery event is visible before the cutoff; otherwise it is NULL because there is no status-event timestamp and final status is not historically knowable.",
        "columns": [
            ("order_id", "TEXT"),
            ("customer_id", "TEXT"),
            ("order_status", "TEXT"),
            ("order_purchase_timestamp", "TEXT"),
            ("order_approved_at", "TEXT"),
            ("order_delivered_carrier_date", "TEXT"),
            ("order_delivered_customer_date", "TEXT"),
            ("order_estimated_delivery_date", "TEXT"),
        ],
        "primary_key": ["order_id"],
        "indexes": [
            ("idx_orders_customer", ["customer_id"]),
            ("idx_orders_purchase", ["order_purchase_timestamp"]),
            ("idx_orders_delivered", ["order_delivered_customer_date"]),
            ("idx_orders_estimated", ["order_estimated_delivery_date"]),
            ("idx_orders_status", ["order_status"]),
        ],
    },
    "order_items": {
        "source": "olist_order_items_dataset.csv",
        "description": "One row per item line. price is the item price; freight_value is separate. Their sum is not revenue.",
        "columns": [
            ("order_id", "TEXT"),
            ("order_item_id", "INTEGER"),
            ("product_id", "TEXT"),
            ("seller_id", "TEXT"),
            ("shipping_limit_date", "TEXT"),
            ("price", "REAL"),
            ("freight_value", "REAL"),
        ],
        "primary_key": ["order_id", "order_item_id"],
        "indexes": [
            ("idx_items_order", ["order_id"]),
            ("idx_items_product", ["product_id"]),
            ("idx_items_seller", ["seller_id"]),
        ],
    },
    "customers": {
        "source": "olist_customers_dataset.csv",
        "description": "Order-scoped customer key plus buyer state. Personal unique id, zip and city are blocked for privacy.",
        "columns": [
            ("customer_id", "TEXT"),
            ("customer_unique_id", "TEXT"),
            ("customer_zip_code_prefix", "TEXT"),
            ("customer_city", "TEXT"),
            ("customer_state", "TEXT"),
        ],
        "primary_key": ["customer_id"],
        "indexes": [
            ("idx_customers_state", ["customer_state"]),
            ("idx_customers_unique", ["customer_unique_id"]),
        ],
    },
    "sellers": {
        "source": "olist_sellers_dataset.csv",
        "description": "Seller key and seller state. Zip and city are blocked for privacy.",
        "columns": [
            ("seller_id", "TEXT"),
            ("seller_zip_code_prefix", "TEXT"),
            ("seller_city", "TEXT"),
            ("seller_state", "TEXT"),
        ],
        "primary_key": ["seller_id"],
        "indexes": [("idx_sellers_state", ["seller_state"])],
    },
    "payments": {
        "source": "olist_order_payments_dataset.csv",
        "description": "Payment rows per order. payment_value is the payment amount, not item price.",
        "columns": [
            ("order_id", "TEXT"),
            ("payment_sequential", "INTEGER"),
            ("payment_type", "TEXT"),
            ("payment_installments", "INTEGER"),
            ("payment_value", "REAL"),
        ],
        "primary_key": ["order_id", "payment_sequential"],
        "indexes": [
            ("idx_payments_order", ["order_id"]),
            ("idx_payments_type", ["payment_type"]),
        ],
    },
    "reviews": {
        "source": "olist_order_reviews_dataset.csv",
        "description": "Review rows answered before the cutoff. review_score is not knowable until the review is answered, so reviews created but not yet answered are absent. Free-text comments are blocked for privacy.",
        "columns": [
            ("review_id", "TEXT"),
            ("order_id", "TEXT"),
            ("review_score", "INTEGER"),
            ("review_comment_title", "TEXT"),
            ("review_comment_message", "TEXT"),
            ("review_creation_date", "TEXT"),
            ("review_answer_timestamp", "TEXT"),
        ],
        "primary_key": [],
        "indexes": [
            ("idx_reviews_order", ["order_id"]),
            ("idx_reviews_score", ["review_score"]),
            ("idx_reviews_creation", ["review_creation_date"]),
        ],
    },
    "products": {
        "source": "olist_products_dataset.csv",
        "description": "Product attributes. The CSV spells the length columns product_name_lenght and product_description_lenght.",
        "columns": [
            ("product_id", "TEXT"),
            ("product_category_name", "TEXT"),
            ("product_name_lenght", "INTEGER"),
            ("product_description_lenght", "INTEGER"),
            ("product_photos_qty", "INTEGER"),
            ("product_weight_g", "INTEGER"),
            ("product_length_cm", "INTEGER"),
            ("product_height_cm", "INTEGER"),
            ("product_width_cm", "INTEGER"),
        ],
        "primary_key": ["product_id"],
        "indexes": [("idx_products_category", ["product_category_name"])],
    },
    "geolocation": {
        "source": "olist_geolocation_dataset.csv",
        "description": "Zip-prefix geolocation rows. Coordinates, city and zip prefix are blocked for privacy; state is available.",
        "columns": [
            ("geolocation_zip_code_prefix", "TEXT"),
            ("geolocation_lat", "REAL"),
            ("geolocation_lng", "REAL"),
            ("geolocation_city", "TEXT"),
            ("geolocation_state", "TEXT"),
        ],
        "primary_key": [],
        "indexes": [
            ("idx_geo_zip", ["geolocation_zip_code_prefix"]),
            ("idx_geo_state", ["geolocation_state"]),
        ],
    },
    "product_category_translation": {
        "source": "product_category_name_translation.csv",
        "description": "Portuguese-to-English product category names.",
        "columns": [
            ("product_category_name", "TEXT"),
            ("product_category_name_english", "TEXT"),
        ],
        "primary_key": ["product_category_name"],
        "indexes": [("idx_translation_english", ["product_category_name_english"])],
    },
}

RELATIONSHIPS = [
    ("orders.customer_id", "customers.customer_id", "many orders per customer key"),
    ("order_items.order_id", "orders.order_id", "many item lines per order"),
    ("order_items.product_id", "products.product_id", "many item lines per product"),
    ("order_items.seller_id", "sellers.seller_id", "many item lines per seller"),
    ("payments.order_id", "orders.order_id", "many payment rows per order"),
    ("reviews.order_id", "orders.order_id", "review rows per order"),
    ("products.product_category_name", "product_category_translation.product_category_name", "category name lookup"),
    (
        "customers.customer_zip_code_prefix",
        "geolocation.geolocation_zip_code_prefix",
        "non-unique zip-prefix lookup (privacy-blocked in Ask queries)",
    ),
    (
        "sellers.seller_zip_code_prefix",
        "geolocation.geolocation_zip_code_prefix",
        "non-unique zip-prefix lookup (privacy-blocked in Ask queries)",
    ),
]

# These columns are never readable by Ask, even in a WHERE or JOIN clause. The
# SQL executor also refuses to return identifier-like columns.
BLOCKED_COLUMNS = {
    "customers": {"customer_unique_id", "customer_zip_code_prefix", "customer_city"},
    "sellers": {"seller_zip_code_prefix", "seller_city"},
    "geolocation": {
        "geolocation_zip_code_prefix",
        "geolocation_lat",
        "geolocation_lng",
        "geolocation_city",
    },
    "reviews": {"review_comment_title", "review_comment_message"},
}

OUTPUT_BLOCKED_COLUMNS = {
    "customer_id",
    "customer_unique_id",
    "seller_id",
    "order_id",
    "review_id",
}

INTEGER_COLUMNS = {
    "order_items": {"order_item_id"},
    "payments": {"payment_sequential", "payment_installments"},
    "reviews": {"review_score"},
    "products": {
        "product_name_lenght",
        "product_description_lenght",
        "product_photos_qty",
        "product_weight_g",
        "product_length_cm",
        "product_height_cm",
        "product_width_cm",
    },
}
REAL_COLUMNS = {
    "order_items": {"price", "freight_value"},
    "payments": {"payment_value"},
    "geolocation": {"geolocation_lat", "geolocation_lng"},
}
DATE_COLUMNS = {
    "orders": {
        "order_purchase_timestamp",
        "order_approved_at",
        "order_delivered_carrier_date",
        "order_delivered_customer_date",
        "order_estimated_delivery_date",
    },
    "order_items": {"shipping_limit_date"},
    "reviews": {"review_creation_date", "review_answer_timestamp"},
}


def _coerce(table, column, value):
    value = (value or "").strip()
    if value == "":
        return None
    if column in INTEGER_COLUMNS.get(table, set()):
        return int(value)
    if column in REAL_COLUMNS.get(table, set()):
        return float(value)
    return value


def _quote(identifier):
    return '"' + identifier.replace('"', '""') + '"'


def _create_table(connection, table, spec):
    columns = ",\n  ".join(
        "{} {}".format(_quote(name), kind) for name, kind in spec["columns"]
    )
    primary = spec.get("primary_key") or []
    if primary:
        columns += ",\n  PRIMARY KEY ({})".format(
            ", ".join(_quote(name) for name in primary)
        )
    connection.execute("CREATE TABLE {} (\n  {}\n)".format(_quote(table), columns))


def _load_table(connection, table, physical_table, spec, source_dir):
    path = source_dir / spec["source"]
    if not path.is_file():
        raise BuildError(
            "Missing source CSV for table {!r}: {}. "
            "Put the raw Olist CSVs in {} and rebuild.".format(
                table, path, source_dir
            )
        )
    column_names = [name for name, _ in spec["columns"]]
    placeholders = ", ".join("?" for _ in column_names)
    insert = "INSERT INTO {} ({}) VALUES ({})".format(
        _quote(physical_table),
        ", ".join(_quote(name) for name in column_names),
        placeholders,
    )
    count = 0
    try:
        with path.open(newline="", encoding="utf-8-sig") as handle:
            reader = csv.DictReader(handle)
            missing = set(column_names) - set(reader.fieldnames or [])
            if missing:
                raise BuildError(
                    "{} is missing required columns: {}.".format(
                        path.name, ", ".join(sorted(missing))
                    )
                )
            batch = []
            for row in reader:
                batch.append(
                    tuple(_coerce(table, name, row.get(name)) for name in column_names)
                )
                if len(batch) >= BATCH_SIZE:
                    connection.executemany(insert, batch)
                    count += len(batch)
                    batch = []
            if batch:
                connection.executemany(insert, batch)
                count += len(batch)
    except (OSError, csv.Error, ValueError) as error:
        raise BuildError("Could not load {}: {}.".format(path, error)) from None
    return count


def _event_bounds(connection):
    bounds = {}
    queries = {
        "purchase_timestamp": (
            "SELECT MIN(order_purchase_timestamp), MAX(order_purchase_timestamp) FROM {}orders".format(BASE_PREFIX)
        ),
        "delivered_customer_date": (
            "SELECT MIN(order_delivered_customer_date), MAX(order_delivered_customer_date) FROM {}orders".format(BASE_PREFIX)
        ),
        "estimated_delivery_date": (
            "SELECT MIN(order_estimated_delivery_date), MAX(order_estimated_delivery_date) FROM {}orders".format(BASE_PREFIX)
        ),
    }
    for name, sql in queries.items():
        row = connection.execute(sql).fetchone()
        bounds[name] = {
            "min": (row[0] or "")[:10] or None,
            "max": (row[1] or "")[:10] or None,
        }
    return bounds


def build_database(source_dir=None, output_path=None, quiet=False):
    """Build the SQLite read model atomically and return its path."""
    source_dir = Path(source_dir) if source_dir is not None else DEFAULT_SOURCE
    output_path = Path(output_path) if output_path is not None else DEFAULT_OUTPUT
    if not source_dir.is_dir():
        raise BuildError(
            "Source directory not found: {}. Pass --source or put the Olist CSVs "
            "in data/raw.".format(source_dir)
        )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_name(output_path.name + ".tmp")
    if temporary.exists():
        temporary.unlink()
    connection = sqlite3.connect(str(temporary))
    try:
        connection.execute("PRAGMA journal_mode=OFF")
        connection.execute("PRAGMA synchronous=OFF")
        connection.execute("BEGIN")
        row_counts = {}
        for table, spec in TABLES.items():
            base_table = BASE_PREFIX + table
            _create_table(connection, base_table, spec)
            row_counts[table] = _load_table(
                connection, table, base_table, spec, source_dir
            )
            for index_name, columns in spec.get("indexes", []):
                connection.execute(
                    "CREATE INDEX {} ON {} ({})".format(
                        _quote(index_name),
                        _quote(base_table),
                        ", ".join(_quote(column) for column in columns),
                    )
                )
        metadata = {
            "format_version": FORMAT_VERSION,
            "source": "Olist public CSVs (data/raw)",
            "base_table_prefix": BASE_PREFIX,
            "public_tables": list(TABLES),
            "tables": {
                table: {
                    "source": spec["source"],
                    "description": spec["description"],
                    "columns": [name for name, _ in spec["columns"]],
                }
                for table, spec in TABLES.items()
            },
            "relationships": [
                {"from": left, "to": right, "cardinality": cardinality}
                for left, right, cardinality in RELATIONSHIPS
            ],
            "row_counts": row_counts,
            "event_bounds": _event_bounds(connection),
        }
        connection.execute(
            "CREATE TABLE _vamos_meta (key TEXT PRIMARY KEY, value TEXT NOT NULL)"
        )
        connection.executemany(
            "INSERT INTO _vamos_meta (key, value) VALUES (?, ?)",
            [
                ("format_version", str(FORMAT_VERSION)),
                ("metadata", json.dumps(metadata, sort_keys=True, separators=(",", ":"))),
            ],
        )
        connection.commit()
    except BaseException:
        connection.rollback()
        connection.close()
        if temporary.exists():
            temporary.unlink()
        raise
    connection.close()
    os.replace(str(temporary), str(output_path))
    if not quiet:
        print("Built {}".format(output_path))
        for table, count in row_counts.items():
            print("  {:28s} {:>8,} rows".format(table, count))
    return output_path


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Build the local read-only Olist analytics SQLite database."
    )
    parser.add_argument(
        "--source",
        default=str(DEFAULT_SOURCE),
        help="Directory containing the raw Olist CSVs (default: data/raw).",
    )
    parser.add_argument(
        "--output",
        default=str(DEFAULT_OUTPUT),
        help="SQLite file to write (default: deployment/olist_dashboard/data/analytics.sqlite).",
    )
    parser.add_argument("--quiet", action="store_true", help="Only print errors.")
    args = parser.parse_args(argv)
    try:
        build_database(args.source, args.output, quiet=args.quiet)
    except BuildError as error:
        print("Analytics database build failed: {}".format(error), file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
