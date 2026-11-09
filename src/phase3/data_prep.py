"""Shared data-preparation code for the Phase 2 order table and the scoring-only open-orders table.

`notebooks/data_prep.ipynb` builds the delivered modelling population from these functions, and
`phase3.build_open_orders` builds the rows for parcels that left the seller but never entered that population.
Keeping both on the same functions is what makes the open rows scoreable by the Phase 2 models.

The history rule is unchanged: as-of values use only delivered orders whose outcome was recorded before the
query time. Open orders are query rows only; they never enter a history source.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from olist.features import as_of_shrunk_mean, asof_mean

TIMESTAMPS = [
    "order_purchase_timestamp", "order_approved_at", "order_delivered_carrier_date",
    "order_delivered_customer_date", "order_estimated_delivery_date",
]
BRAZIL_LAT = (-34.0, 5.5)
BRAZIL_LNG = (-74.0, -34.5)
SECONDS_PER_DAY = 86_400
ROUTE_PRIOR_STRENGTH = 20
SELLER_PRIOR_STRENGTH = 10
SHIP_DAYS_RANGE = (0, 60)  # drops negative and implausible handover times
TRANSIT_WINDOW_DAYS = 90

NUMERIC_FEATURES = [
    "promised_days", "promise_slack", "route_typical_days", "seller_ship_days",
    "distance_km", "same_state", "purchase_hour",
    "n_items", "n_products", "n_sellers", "total_price", "total_freight",
    "total_weight_g", "total_volume_cm3", "max_installments",
]
CATEGORICAL_FEATURES = [
    "customer_state", "seller_state", "product_category", "payment_type",
    "purchase_dayofweek",
]
HISTORY_FEATURES = ["promise_slack", "route_typical_days", "seller_ship_days"]
FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES
TARGETS = ["is_late", "delivery_days"]
MODEL_COLUMNS = ["order_id", "order_purchase_timestamp", "outcome_available_at", "purchase_month",
                 "furthest_seller_id", *FEATURES, *TARGETS]
SPLIT_COLUMNS = ["split", "random_split", "cv_fold", "random_cv_fold"]


def days_between(later, earlier):
    return (later - earlier).dt.total_seconds() / SECONDS_PER_DAY


def load_raw(data_dir):
    """Read the raw Olist tables with the timestamp columns parsed."""
    data_dir = Path(data_dir)
    return {
        "orders": pd.read_csv(data_dir / "olist_orders_dataset.csv", parse_dates=TIMESTAMPS),
        "items": pd.read_csv(data_dir / "olist_order_items_dataset.csv"),
        "products": pd.read_csv(data_dir / "olist_products_dataset.csv"),
        "translation": pd.read_csv(data_dir / "product_category_name_translation.csv"),
        "sellers": pd.read_csv(data_dir / "olist_sellers_dataset.csv"),
        "customers": pd.read_csv(data_dir / "olist_customers_dataset.csv"),
        "payments": pd.read_csv(data_dir / "olist_order_payments_dataset.csv"),
        "geolocation": pd.read_csv(data_dir / "olist_geolocation_dataset.csv"),
    }


def delivered_population(orders_raw):
    """Delivered orders with both a delivery and an estimated delivery timestamp (the modelling population)."""
    is_delivered = orders_raw["order_status"].eq("delivered")
    has_dates = (orders_raw["order_delivered_customer_date"].notna()
                 & orders_raw["order_estimated_delivery_date"].notna())
    return orders_raw.loc[is_delivered & has_dates].copy()


def prepare_products(products, translation):
    """Add the English category name and the packed volume used by the order-level features."""
    out = products.merge(translation, on="product_category_name", how="left")
    out["product_category"] = out["product_category_name_english"].fillna("unknown")
    out["volume_cm3"] = out["product_length_cm"] * out["product_height_cm"] * out["product_width_cm"]
    return out


def filter_geolocation(geo):
    """Drop geolocation points outside Brazil (a known Olist data error)."""
    inside_brazil = (geo["geolocation_lat"].between(*BRAZIL_LAT)
                     & geo["geolocation_lng"].between(*BRAZIL_LNG))
    return geo.loc[inside_brazil]


def clean_geolocation(geo):
    """Drop points outside Brazil, then exact duplicate rows."""
    return filter_geolocation(geo).drop_duplicates()


def zip_centroids(geo):
    """One median latitude/longitude centre per customer/seller ZIP prefix."""
    return geo.groupby("geolocation_zip_code_prefix").agg(
        lat=("geolocation_lat", "median"), lng=("geolocation_lng", "median"))


def haversine_km(lat1, lng1, lat2, lng2):
    lat1, lng1, lat2, lng2 = map(np.radians, [lat1, lng1, lat2, lng2])
    inner = (np.sin((lat2 - lat1) / 2) ** 2
             + np.cos(lat1) * np.cos(lat2) * np.sin((lng2 - lng1) / 2) ** 2)
    return 2 * 6371.0 * np.arcsin(np.sqrt(inner))


def build_item_features(orders, items, products, sellers, customers, centroids):
    """Aggregate item rows to one row per order, including the furthest seller and largest category."""
    order_customers = orders[["order_id", "customer_id"]].merge(
        customers[["customer_id", "customer_zip_code_prefix", "customer_state"]], on="customer_id")
    order_items = (
        items.merge(order_customers, on="order_id")  # keeps only the orders passed in
        .merge(sellers[["seller_id", "seller_zip_code_prefix", "seller_state"]], on="seller_id", how="left")
    )
    customer_loc = centroids.reindex(order_items["customer_zip_code_prefix"]).to_numpy()
    seller_loc = centroids.reindex(order_items["seller_zip_code_prefix"]).to_numpy()
    order_items["distance_km"] = haversine_km(customer_loc[:, 0], customer_loc[:, 1],
                                              seller_loc[:, 0], seller_loc[:, 1])
    order_items["same_state"] = order_items["seller_state"].eq(order_items["customer_state"])
    order_items = order_items.merge(
        products[["product_id", "product_category", "product_weight_g", "volume_cm3"]],
        on="product_id", how="left")

    grouped = order_items.groupby("order_id")
    item_features = grouped.agg(
        n_items=("order_item_id", "size"),
        n_products=("product_id", "nunique"),
        n_sellers=("seller_id", "nunique"),
        total_price=("price", "sum"),
        total_freight=("freight_value", "sum"),
        distance_km=("distance_km", "max"),
        same_state=("same_state", "all"),
        customer_state=("customer_state", "first"),
    )
    item_features["total_weight_g"] = grouped["product_weight_g"].sum(min_count=1)
    item_features["total_volume_cm3"] = grouped["volume_cm3"].sum(min_count=1)
    furthest_seller = (
        order_items.sort_values(["distance_km", "order_item_id"], ascending=[False, True], na_position="last")
        .drop_duplicates("order_id").set_index("order_id")[["seller_state", "seller_id"]]
        .rename(columns={"seller_id": "furthest_seller_id"})
    )
    largest_item_category = (
        order_items.sort_values(["volume_cm3", "order_item_id"], ascending=[False, True], na_position="last")
        .drop_duplicates("order_id").set_index("order_id")["product_category"]
    )
    item_features = item_features.join(furthest_seller).join(largest_item_category)
    return order_items, item_features


def build_payment_features(payments):
    """Primary payment type (lowest-numbered row) and highest instalment count per order."""
    return pd.DataFrame({
        "payment_type": (payments.sort_values("payment_sequential").drop_duplicates("order_id")
                         .set_index("order_id")["payment_type"]),
        "max_installments": payments.groupby("order_id")["payment_installments"].max(),
    })


def add_purchase_features(orders):
    """Purchase-time features available for every order, delivered or not."""
    out = orders.copy()
    purchased = out["order_purchase_timestamp"]
    out["promised_days"] = days_between(out["order_estimated_delivery_date"], purchased)
    out["purchase_hour"] = purchased.dt.hour
    out["purchase_dayofweek"] = purchased.dt.day_name()
    out["purchase_month"] = purchased.dt.month_name()
    return out


def add_order_targets(orders):
    """Targets on top of the purchase-time features. Only meaningful for orders with a recorded delivery."""
    out = add_purchase_features(orders)
    purchased = out["order_purchase_timestamp"]
    out["is_late"] = (out["order_delivered_customer_date"].dt.normalize()
                      > out["order_estimated_delivery_date"].dt.normalize()).astype(int)
    out["delivery_days"] = days_between(out["order_delivered_customer_date"], purchased)
    return out


def assemble_order_table(orders, item_features, payment_features):
    """Join the order-level, item and payment attributes into the one-row-per-order frame."""
    order_table = (
        orders.set_index("order_id")
        .join(item_features)
        .join(payment_features)
        .reset_index()
    )
    order_table["payment_type"] = order_table["payment_type"].fillna("unknown")
    # Nullable so a scored order with no item rows keeps a missing value for the Pipeline imputer.
    order_table["same_state"] = order_table["same_state"].astype("Int64")
    order_table = order_table.rename(columns={"order_delivered_customer_date": "outcome_available_at"})
    assert order_table["order_id"].is_unique
    return order_table


def add_history_features(order_table, history=None):
    """Route and seller as-of history features.

    `history` defaults to `order_table` (the notebook's delivered table). Open rows pass the delivered table
    explicitly, so an order can never see its own outcome or another open row's handover.
    """
    history = order_table if history is None else history
    route = order_table["seller_state"] + ">" + order_table["customer_state"]
    history_route = history["seller_state"] + ">" + history["customer_state"]
    order_table = order_table.copy()
    order_table["route_typical_days"] = as_of_shrunk_mean(
        history_route, history["delivery_days"], history["outcome_available_at"],
        route, order_table["order_purchase_timestamp"], ROUTE_PRIOR_STRENGTH)
    order_table["promise_slack"] = order_table["promised_days"] - order_table["route_typical_days"]

    history_ship_days = days_between(history["order_delivered_carrier_date"],
                                     history["order_purchase_timestamp"])
    history_usable = history_ship_days.between(*SHIP_DAYS_RANGE)
    order_table["seller_ship_days"] = as_of_shrunk_mean(
        history.loc[history_usable, "furthest_seller_id"], history_ship_days[history_usable],
        history.loc[history_usable, "order_delivered_carrier_date"],
        order_table["furthest_seller_id"], order_table["order_purchase_timestamp"], SELLER_PRIOR_STRENGTH)
    return order_table


def add_stage_features(query_table, orders, history=None):
    """Handover-stage inputs for query rows, with `history` as the only carrier-leg history source.

    This mirrors `olist.eta.add_stage_features` exactly: with `history=None` the query frame is its own
    source, as in the original. Open rows pass the delivered cohort explicitly, so they never add their own
    transit to history.
    """
    raw = orders[["order_id", "order_approved_at", "order_delivered_carrier_date"]]
    out = query_table.merge(raw, on="order_id", how="left", validate="one_to_one")
    purchase, delivered = out["order_purchase_timestamp"], out["outcome_available_at"]
    handover = out["order_delivered_carrier_date"]
    out["handover_time"] = handover
    out["route"] = out["seller_state"].astype(str) + "->" + out["customer_state"].astype(str)
    # For open rows there is no delivery yet; a missing delivery date is not a reason to reject the handover.
    out["handover_valid"] = (handover.notna() & handover.ge(purchase)
                             & (delivered.isna() | handover.lt(delivered)))

    out["handover_days"] = days_between(handover, purchase).where(out["handover_valid"])
    approval = days_between(out["order_approved_at"], purchase)
    known = out["order_approved_at"].le(handover) & approval.ge(0)
    out["approval_days"] = approval.where(known)

    transit = days_between(delivered, handover).where(out["handover_valid"])
    out["transit_days"] = transit
    source = out if history is None else history
    history_route = source["seller_state"].astype(str) + "->" + source["customer_state"].astype(str)
    out["route_transit_90d"] = asof_mean(
        history_route, source["transit_days"], source["outcome_available_at"], out["route"], purchase,
        TRANSIT_WINDOW_DAYS, ROUTE_PRIOR_STRENGTH)
    at_handover = asof_mean(
        history_route, source["transit_days"], source["outcome_available_at"], out["route"],
        handover.where(out["handover_valid"], purchase), TRANSIT_WINDOW_DAYS, ROUTE_PRIOR_STRENGTH)
    out["route_transit_90d_at_handover"] = np.where(out["handover_valid"], at_handover, np.nan)
    return out


def add_history_extension(query_table, customer_zip, history):
    """ZIP-3 transit and seller-handover history for query rows, sourced only from delivered `history`."""
    out = query_table.merge(customer_zip, on="order_id", how="left", validate="one_to_one")
    if out["customer_zip_code_prefix"].isna().any():
        raise ValueError("Every order needs a customer ZIP prefix")
    zip3 = out["customer_zip_code_prefix"].astype(int).astype(str).str.zfill(5).str[:3]
    handover = out["handover_time"].where(out["handover_valid"])

    history_zip3 = history["customer_zip_code_prefix"].astype(int).astype(str).str.zfill(5).str[:3]
    out["zip3_transit_90d_at_handover"] = asof_mean(
        history_zip3, history["transit_days"], history["outcome_available_at"], zip3, handover,
        TRANSIT_WINDOW_DAYS, 20)

    history_speed = history["handover_days"].where(history["handover_days"].between(0, 60))
    out["seller_handover_90d_at_handover"] = asof_mean(
        history["furthest_seller_id"], history_speed, history["handover_time"],
        out["furthest_seller_id"], handover, TRANSIT_WINDOW_DAYS, SELLER_PRIOR_STRENGTH)
    return out.drop(columns="customer_zip_code_prefix")
