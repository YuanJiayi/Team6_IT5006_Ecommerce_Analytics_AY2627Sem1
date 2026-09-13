"""Data loading and aggregation helpers for the Olist dashboard."""

from pathlib import Path
import re
import unicodedata

import numpy as np
import pandas as pd
import streamlit as st


PERIOD_ALIASES = {"Day": "D", "Month": "M", "Year": "Y"}
DATE_FREQUENCIES = {"Day": "D", "Month": "MS", "Year": "YS"}
TREND_START = pd.Timestamp("2017-01-01")
TREND_END = pd.Timestamp("2018-09-01")


@st.cache_data
def load_data(path: Path) -> pd.DataFrame:
    return pd.read_csv(
        path,
        parse_dates=[
            "order_purchase_timestamp",
            "order_delivered_customer_date",
            "order_estimated_delivery_date",
        ],
    )


@st.cache_data
def load_customer_ids(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, usecols=["customer_id", "customer_unique_id"])


@st.cache_data
def load_reviews(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, parse_dates=["review_creation_date"])


def latest_reviews(reviews: pd.DataFrame) -> pd.DataFrame:
    return reviews.sort_values(["order_id", "review_creation_date", "review_id"]).drop_duplicates(
        "order_id", keep="last"
    )


def _latest_rating_reviews(reviews: pd.DataFrame) -> pd.DataFrame:
    """Match the ratings notebook's deterministic duplicate-review tie-breaker."""
    ratings = reviews.copy()
    ratings["review_answer_timestamp"] = pd.to_datetime(
        ratings["review_answer_timestamp"]
    )
    return ratings.sort_values(
        ["order_id", "review_creation_date", "review_answer_timestamp", "review_id"]
    ).drop_duplicates("order_id", keep="last")


def add_period(data: pd.DataFrame, date_column: str, granularity: str) -> pd.DataFrame:
    return data.assign(
        period=data[date_column].dt.to_period(PERIOD_ALIASES[granularity]).dt.to_timestamp()
    )


def trim_trend_window(data: pd.DataFrame, date_column: str) -> pd.DataFrame:
    return data.loc[
        data[date_column].ge(TREND_START) & data[date_column].lt(TREND_END)
    ].copy()


def complete_periods(time_series: pd.DataFrame, granularity: str) -> pd.DataFrame:
    periods = pd.date_range(
        start=time_series["period"].min(), end=time_series["period"].max(), freq=DATE_FREQUENCIES[granularity]
    )
    return time_series.set_index("period").reindex(periods, fill_value=0).rename_axis("period").reset_index()


def build_growth_series(data: pd.DataFrame, granularity: str) -> pd.DataFrame:
    data = trim_trend_window(data, "order_purchase_timestamp")
    series = add_period(data, "order_purchase_timestamp", granularity).groupby("period", as_index=False).agg(
        orders=("order_id", "nunique"), product_revenue=("price", "sum"), items_sold=("order_item_id", "size")
    )
    return complete_periods(series, granularity)


def build_commercial_series(data: pd.DataFrame, granularity: str) -> pd.DataFrame:
    data = trim_trend_window(data, "order_purchase_timestamp")
    series = add_period(data, "order_purchase_timestamp", granularity).groupby("period", as_index=False).agg(
        orders=("order_id", "nunique"), product_revenue=("price", "sum"), items_sold=("order_item_id", "size")
    )
    series["average_order_value"] = series["product_revenue"] / series["orders"]
    series["items_per_order"] = series["items_sold"] / series["orders"]
    return series


def build_review_series(order_data: pd.DataFrame, reviews: pd.DataFrame, granularity: str) -> pd.DataFrame:
    review_data = order_data[["order_id", "order_purchase_timestamp"]].merge(
        latest_reviews(reviews)[["order_id", "review_score"]], on="order_id", how="inner", validate="one_to_one"
    )
    review_data = trim_trend_window(review_data, "order_purchase_timestamp").assign(
        low_rating=lambda data: data["review_score"].le(2), five_star=lambda data: data["review_score"].eq(5)
    )
    series = add_period(review_data, "order_purchase_timestamp", granularity).groupby("period", as_index=False).agg(
        average_review_score=("review_score", "mean"), low_rating_rate=("low_rating", "mean"), five_star_rate=("five_star", "mean")
    )
    series["satisfaction_proxy"] = (series["five_star_rate"] - series["low_rating_rate"]) * 100
    series["low_rating_rate"] *= 100
    return series


def eligible_deliveries(order_data: pd.DataFrame) -> pd.DataFrame:
    return order_data.loc[
        order_data["order_status"].eq("delivered")
        & order_data["order_delivered_customer_date"].notna()
        & order_data["order_estimated_delivery_date"].notna()
    ].copy()


def build_delivery_series(order_data: pd.DataFrame, granularity: str) -> pd.DataFrame:
    delivered = trim_trend_window(eligible_deliveries(order_data), "order_purchase_timestamp")
    delivered["delivery_days_exact"] = (
        delivered["order_delivered_customer_date"] - delivered["order_purchase_timestamp"]
    ).dt.total_seconds() / 86_400
    delivered["late_delivery"] = delivered["order_delivered_customer_date"] > delivered["order_estimated_delivery_date"]
    series = add_period(delivered, "order_purchase_timestamp", granularity).groupby("period", as_index=False).agg(
        median_delivery_days=("delivery_days_exact", "median"), late_delivery_rate=("late_delivery", "mean")
    )
    series["late_delivery_rate"] *= 100
    return series


def reviewed_deliveries(order_data: pd.DataFrame, reviews: pd.DataFrame, trim: bool = False) -> pd.DataFrame:
    data = eligible_deliveries(order_data).merge(
        latest_reviews(reviews)[["order_id", "review_score"]], on="order_id", how="inner", validate="one_to_one"
    )
    if trim:
        data = trim_trend_window(data, "order_purchase_timestamp")
    return data.assign(
        late_delivery=lambda frame: frame["order_delivered_customer_date"] > frame["order_estimated_delivery_date"],
        low_rating=lambda frame: frame["review_score"].le(2),
        days_late=lambda frame: (frame["order_delivered_customer_date"] - frame["order_estimated_delivery_date"]).dt.total_seconds() / 86_400,
    )


def build_delivery_experience_series(order_data: pd.DataFrame, reviews: pd.DataFrame) -> pd.DataFrame:
    data = reviewed_deliveries(order_data, reviews, trim=True)
    series = add_period(data, "order_purchase_timestamp", "Month").groupby("period", as_index=False).agg(
        late_delivery_rate=("late_delivery", "mean"), low_rating_rate=("low_rating", "mean")
    )
    return series.assign(**{"Late-delivery rate (%)": series["late_delivery_rate"] * 100, "Low-rating rate (%)": series["low_rating_rate"] * 100})


def build_delivery_review_analysis(order_data: pd.DataFrame, reviews: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, float]:
    data = reviewed_deliveries(order_data, reviews)
    comparison = data.groupby("late_delivery", as_index=False)["low_rating"].mean().assign(
        delivery_timing=lambda frame: frame["late_delivery"].map({False: "On time or early", True: "Late"}),
        low_rating_rate=lambda frame: frame["low_rating"] * 100,
    )
    risk_ratio = comparison.loc[comparison["late_delivery"], "low_rating_rate"].iloc[0] / comparison.loc[~comparison["late_delivery"], "low_rating_rate"].iloc[0]
    data["lateness_band"] = pd.cut(data["days_late"], [-float("inf"), 0, 3, 7, float("inf")], labels=["On time or early", "1–3 days late", "4–7 days late", "8+ days late"])
    severity = data.groupby("lateness_band", observed=True, as_index=False)["low_rating"].mean().assign(low_rating_rate=lambda frame: frame["low_rating"] * 100)
    return comparison, severity, risk_ratio


def _rating_rate_summary(data: pd.DataFrame, group: str) -> pd.DataFrame:
    """Summarise a binary low-rating outcome with 95% Wilson intervals."""
    summary = (
        data.groupby(group, observed=True)["low_rating"]
        .agg(orders="size", low_ratings="sum")
        .reset_index()
    )
    n = summary["orders"].astype(float)
    proportion = summary["low_ratings"] / n
    z = 1.96
    centre = (proportion + z**2 / (2 * n)) / (1 + z**2 / n)
    half_width = (
        z
        * np.sqrt(proportion * (1 - proportion) / n + z**2 / (4 * n**2))
        / (1 + z**2 / n)
    )
    summary["low_rating_rate"] = proportion * 100
    summary["lower_rate"] = (centre - half_width) * 100
    summary["upper_rate"] = (centre + half_width) * 100
    return summary


@st.cache_data
def build_rating_complexity_summary(
    items_path: Path, reviews_path: Path
) -> pd.DataFrame:
    """Compare low-rating rates for single- and multi-item/seller orders."""
    items = pd.read_csv(
        items_path, usecols=["order_id", "order_item_id", "seller_id"]
    )
    order_complexity = items.groupby("order_id", as_index=False).agg(
        item_count=("order_item_id", "size"),
        seller_count=("seller_id", "nunique"),
    )
    reviewed = order_complexity.merge(
        _latest_rating_reviews(load_reviews(reviews_path))[["order_id", "review_score"]],
        on="order_id",
        how="inner",
        validate="one_to_one",
    ).assign(low_rating=lambda frame: frame["review_score"].le(2))

    item_groups = reviewed.assign(
        group=np.where(reviewed["item_count"].gt(1), "Multiple items", "Single item")
    )
    item_summary = _rating_rate_summary(item_groups, "group").assign(
        dimension="Number of items"
    )
    seller_groups = reviewed.assign(
        group=np.where(
            reviewed["seller_count"].gt(1), "Multiple sellers", "Single seller"
        )
    )
    seller_summary = _rating_rate_summary(seller_groups, "group").assign(
        dimension="Number of sellers"
    )
    summary = pd.concat([item_summary, seller_summary], ignore_index=True)
    summary["group_order"] = summary["group"].map(
        {
            "Single item": 1,
            "Multiple items": 2,
            "Single seller": 1,
            "Multiple sellers": 2,
        }
    )
    summary["order_share"] = summary["orders"] / len(reviewed) * 100
    summary["low_rating_capture"] = (
        summary["low_ratings"] / reviewed["low_rating"].sum() * 100
    )
    return summary


@st.cache_data
def build_rating_delivery_timing_summary(
    orders_path: Path, items_path: Path, reviews_path: Path
) -> pd.DataFrame:
    """Summarise delivery timing within the item-bearing ratings population."""
    orders = pd.read_csv(
        orders_path,
        usecols=[
            "order_id",
            "order_purchase_timestamp",
            "order_delivered_customer_date",
            "order_estimated_delivery_date",
        ],
        parse_dates=[
            "order_purchase_timestamp",
            "order_delivered_customer_date",
            "order_estimated_delivery_date",
        ],
    )
    item_order_ids = pd.read_csv(items_path, usecols=["order_id"]).drop_duplicates()
    reviewed = orders.merge(
        item_order_ids,
        on="order_id",
        how="inner",
        validate="one_to_one",
    ).merge(
        _latest_rating_reviews(load_reviews(reviews_path))[["order_id", "review_score"]],
        on="order_id",
        how="inner",
        validate="one_to_one",
    ).dropna(
        subset=["order_delivered_customer_date", "order_estimated_delivery_date"]
    )
    reviewed["delivery_days"] = (
        reviewed["order_delivered_customer_date"]
        - reviewed["order_purchase_timestamp"]
    ).dt.total_seconds() / 86_400
    reviewed = reviewed.loc[reviewed["delivery_days"].ge(0)].copy()
    reviewed["days_vs_estimate"] = (
        reviewed["order_delivered_customer_date"]
        - reviewed["order_estimated_delivery_date"]
    ).dt.total_seconds() / 86_400
    timing_order = [
        "More than 7 days early",
        "0–7 days early",
        "1–3 days late",
        "4–7 days late",
        "Over 7 days late",
    ]
    days = reviewed["days_vs_estimate"]
    reviewed["delivery_timing"] = pd.Categorical(
        np.select(
            [days.lt(-7), days.le(0), days.le(3), days.le(7)],
            timing_order[:-1],
            default=timing_order[-1],
        ),
        categories=timing_order,
        ordered=True,
    )
    summary = _rating_rate_summary(reviewed.assign(
        low_rating=reviewed["review_score"].le(2)
    ), "delivery_timing")
    summary["timing_order"] = summary["delivery_timing"].map(
        {label: index for index, label in enumerate(timing_order, start=1)}
    ).astype(int)
    return summary


@st.cache_data
def build_review_score_correlations(
    consolidated_path: Path,
    customers_path: Path,
    sellers_path: Path,
    geo_path: Path,
    reviews_path: Path,
) -> pd.DataFrame:
    """Match the route-distance notebook's Pearson feature screening."""
    line_items = pd.read_csv(
        consolidated_path,
        usecols=[
            "order_id",
            "order_item_id",
            "customer_id",
            "seller_id",
            "price",
            "freight_value",
            "order_delivered_customer_date",
            "order_estimated_delivery_date",
            "delivery_days",
        ],
        parse_dates=[
            "order_delivered_customer_date",
            "order_estimated_delivery_date",
        ],
    )
    orders = line_items.groupby("order_id", as_index=False).agg(
        customer_id=("customer_id", "first"),
        seller_id=("seller_id", "first"),
        total_item_price=("price", "sum"),
        total_freight_value=("freight_value", "sum"),
        item_count=("order_item_id", "size"),
        order_delivered_customer_date=("order_delivered_customer_date", "first"),
        order_estimated_delivery_date=("order_estimated_delivery_date", "first"),
        delivery_days=("delivery_days", "first"),
    )
    customers = pd.read_csv(
        customers_path, usecols=["customer_id", "customer_city", "customer_state"]
    )
    sellers = pd.read_csv(
        sellers_path, usecols=["seller_id", "seller_city", "seller_state"]
    )
    orders = orders.merge(customers, on="customer_id", how="left").merge(
        sellers, on="seller_id", how="left"
    )
    reviews = pd.read_csv(reviews_path, usecols=["order_id", "review_score"])
    reviews = reviews.dropna(subset=["review_score"]).groupby(
        "order_id", as_index=False
    ).agg(review_score=("review_score", "mean"))
    geo = pd.read_csv(
        geo_path,
        usecols=["geolocation_city", "geolocation_state", "geolocation_lat", "geolocation_lng"],
    )

    def city_key(city: str, state: str) -> str:
        value = unicodedata.normalize("NFKD", f"{city} {state}".casefold())
        value = "".join(character for character in value if not unicodedata.combining(character))
        return re.sub(r"[^a-z0-9]+", " ", value).strip()

    geo["city_key"] = [
        city_key(city, state)
        for city, state in zip(geo["geolocation_city"], geo["geolocation_state"])
    ]
    coordinates = geo.groupby("city_key", as_index=False).agg(
        latitude=("geolocation_lat", "mean"), longitude=("geolocation_lng", "mean")
    )
    orders["buyer_city_key"] = [
        city_key(city, state)
        for city, state in zip(orders["customer_city"], orders["customer_state"])
    ]
    orders["seller_city_key"] = [
        city_key(city, state)
        for city, state in zip(orders["seller_city"], orders["seller_state"])
    ]
    orders = orders.merge(
        coordinates.rename(
            columns={"city_key": "buyer_city_key", "latitude": "buyer_latitude", "longitude": "buyer_longitude"}
        ),
        on="buyer_city_key",
        how="inner",
    ).merge(
        coordinates.rename(
            columns={"city_key": "seller_city_key", "latitude": "seller_latitude", "longitude": "seller_longitude"}
        ),
        on="seller_city_key",
        how="inner",
    )
    orders["route_distance_km"] = _haversine_km(
        orders["seller_latitude"],
        orders["seller_longitude"],
        orders["buyer_latitude"],
        orders["buyer_longitude"],
    )
    scored = orders.merge(reviews, on="order_id", how="inner").dropna(
        subset=["route_distance_km", "review_score"]
    )
    delivery_dates_available = scored[
        ["order_delivered_customer_date", "order_estimated_delivery_date"]
    ].notna().all(axis=1)
    scored["late_delivery"] = pd.Series(pd.NA, index=scored.index, dtype="boolean")
    scored.loc[delivery_dates_available, "late_delivery"] = (
        scored.loc[delivery_dates_available, "order_delivered_customer_date"]
        > scored.loc[delivery_dates_available, "order_estimated_delivery_date"]
    )
    feature_labels = {
        "route_distance_km": "Route distance",
        "delivery_days": "Delivery days",
        "late_delivery": "Late delivery",
        "total_item_price": "Total item price",
        "total_freight_value": "Total freight value",
        "item_count": "Items per order",
    }
    rows = []
    for feature, label in feature_labels.items():
        pairs = scored[[feature, "review_score"]].dropna().astype(float)
        rows.append(
            {
                "feature": label,
                "correlation": pairs[feature].corr(pairs["review_score"]),
                "orders": len(pairs),
            }
        )
    return pd.DataFrame(rows).sort_values("correlation").reset_index(drop=True)


# ---------------------------------------------------------------------------
# Delivery correlations tab
# ---------------------------------------------------------------------------

LEAD_STAGES = ["processing_time", "handling_time", "shipping_time"]
LEAD_STAGE_LABELS = {
    "processing_time": "Processing (purchase → approval)",
    "handling_time": "Handling (approval → carrier)",
    "shipping_time": "Shipping (carrier → customer)",
}


@st.cache_data
def load_order_stage_timestamps(path: Path) -> pd.DataFrame:
    """Approval / carrier timestamps live only in the raw orders file, not the
    consolidated line-item CSV, so they are pulled straight from here."""
    return pd.read_csv(
        path,
        usecols=[
            "order_id",
            "order_purchase_timestamp",
            "order_approved_at",
            "order_delivered_carrier_date",
            "order_delivered_customer_date",
        ],
        parse_dates=[
            "order_purchase_timestamp",
            "order_approved_at",
            "order_delivered_carrier_date",
            "order_delivered_customer_date",
        ],
    )


def _order_review_scores(reviews_path: Path) -> pd.DataFrame:
    return latest_reviews(load_reviews(reviews_path))[["order_id", "review_score"]]


@st.cache_data
def build_leadtime_decomposition(
    orders_path: Path, reviews_path: Path
) -> tuple[pd.DataFrame, pd.DataFrame, float, float]:
    stages = load_order_stage_timestamps(orders_path).dropna(
        subset=[
            "order_purchase_timestamp",
            "order_approved_at",
            "order_delivered_carrier_date",
            "order_delivered_customer_date",
        ]
    )
    stages["processing_time"] = (
        stages["order_approved_at"] - stages["order_purchase_timestamp"]
    ).dt.total_seconds() / 86_400
    stages["handling_time"] = (
        stages["order_delivered_carrier_date"] - stages["order_approved_at"]
    ).dt.total_seconds() / 86_400
    stages["shipping_time"] = (
        stages["order_delivered_customer_date"] - stages["order_delivered_carrier_date"]
    ).dt.total_seconds() / 86_400

    valid = (stages[LEAD_STAGES] >= 0).all(axis=1)
    excluded_share = float((~valid).mean() * 100)
    clean = stages.loc[valid].copy()
    clean["total_time"] = clean[LEAD_STAGES].sum(axis=1)
    total_mean = float(clean["total_time"].mean())

    summary = pd.DataFrame(
        {
            "stage": [LEAD_STAGE_LABELS[key] for key in LEAD_STAGES],
            "stage_key": LEAD_STAGES,
            "mean_days": [clean[key].mean() for key in LEAD_STAGES],
            "median_days": [clean[key].median() for key in LEAD_STAGES],
        }
    )
    summary["share_pct"] = summary["mean_days"] / total_mean * 100

    scored = clean.merge(_order_review_scores(reviews_path), on="order_id", how="inner")
    scored["shipping_bucket"] = pd.cut(
        scored["shipping_time"],
        bins=[0, 3, 7, 14, 21, 30, float("inf")],
        labels=["0–3", "3–7", "7–14", "14–21", "21–30", "30+"],
        include_lowest=True,
    )
    shipping_review = (
        scored.groupby("shipping_bucket", observed=True, as_index=False)
        .agg(mean_review_score=("review_score", "mean"), orders=("order_id", "size"))
    )
    return summary, shipping_review, excluded_share, total_mean


@st.cache_data
def load_zip_centroids(path: Path) -> pd.DataFrame:
    geo = pd.read_csv(
        path,
        usecols=["geolocation_zip_code_prefix", "geolocation_lat", "geolocation_lng"],
    )
    return geo.groupby("geolocation_zip_code_prefix", as_index=False).agg(
        lat=("geolocation_lat", "mean"), lng=("geolocation_lng", "mean")
    )


def _haversine_km(lat1, lng1, lat2, lng2):
    radius = 6371.0
    lat1, lng1, lat2, lng2 = map(np.radians, [lat1, lng1, lat2, lng2])
    dlat = lat2 - lat1
    dlng = lng2 - lng1
    inner = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlng / 2) ** 2
    return 2 * radius * np.arcsin(np.sqrt(inner))


@st.cache_data
def build_distance_table(
    consolidated_path: Path,
    customers_path: Path,
    sellers_path: Path,
    geo_path: Path,
    reviews_path: Path,
) -> tuple[pd.DataFrame, float]:
    orders = eligible_deliveries(load_data(consolidated_path).drop_duplicates("order_id"))[
        [
            "order_id",
            "customer_id",
            "seller_id",
            "order_delivered_customer_date",
            "order_estimated_delivery_date",
            "delivery_days",
        ]
    ]
    customers = pd.read_csv(
        customers_path, usecols=["customer_id", "customer_zip_code_prefix"]
    )
    sellers = pd.read_csv(sellers_path, usecols=["seller_id", "seller_zip_code_prefix"])
    centroids = load_zip_centroids(geo_path)

    merged = orders.merge(customers, on="customer_id", how="left").merge(
        sellers, on="seller_id", how="left"
    )
    total = len(merged)
    merged = merged.merge(
        centroids.rename(
            columns={
                "geolocation_zip_code_prefix": "customer_zip_code_prefix",
                "lat": "cust_lat",
                "lng": "cust_lng",
            }
        ),
        on="customer_zip_code_prefix",
        how="left",
    ).merge(
        centroids.rename(
            columns={
                "geolocation_zip_code_prefix": "seller_zip_code_prefix",
                "lat": "sell_lat",
                "lng": "sell_lng",
            }
        ),
        on="seller_zip_code_prefix",
        how="left",
    )
    located = merged.dropna(subset=["cust_lat", "sell_lat"]).copy()
    dropped_share = float((1 - len(located) / total) * 100) if total else 0.0
    located["distance_km"] = _haversine_km(
        located["cust_lat"], located["cust_lng"], located["sell_lat"], located["sell_lng"]
    )
    located["is_late"] = (
        located["order_delivered_customer_date"]
        > located["order_estimated_delivery_date"]
    )
    located = located.merge(_order_review_scores(reviews_path), on="order_id", how="left")
    return (
        located[["order_id", "distance_km", "delivery_days", "is_late", "review_score"]],
        dropped_share,
    )


def build_distance_buckets(distance_table: pd.DataFrame) -> pd.DataFrame:
    data = distance_table.dropna(subset=["distance_km", "delivery_days"]).copy()
    data["distance_bucket"] = pd.qcut(data["distance_km"], 6, duplicates="drop")
    grouped = (
        data.groupby("distance_bucket", observed=True)
        .agg(
            mean_delivery_days=("delivery_days", "mean"),
            mean_review_score=("review_score", "mean"),
            late_rate=("is_late", "mean"),
            orders=("order_id", "size"),
        )
        .reset_index()
    )
    grouped["late_rate"] *= 100
    grouped["distance_label"] = grouped["distance_bucket"].apply(
        lambda interval: f"{max(interval.left, 0):,.0f}–{interval.right:,.0f}"
    )
    return grouped


def _delivery_review_cut(
    consolidated_path: Path, reviews_path: Path, group_columns: list[str]
) -> pd.DataFrame:
    items = eligible_deliveries(load_data(consolidated_path))
    keep = list(dict.fromkeys([*group_columns, "order_id", "delivery_days", "is_on_time"]))
    pairs = items[keep].dropna(subset=group_columns).drop_duplicates(
        [*group_columns, "order_id"]
    )
    pairs = pairs.merge(_order_review_scores(reviews_path), on="order_id", how="left")
    pairs["is_late"] = ~pairs["is_on_time"].astype(bool)
    grouped = (
        pairs.groupby(group_columns)
        .agg(
            orders=("order_id", "nunique"),
            mean_delivery_days=("delivery_days", "mean"),
            mean_review_score=("review_score", "mean"),
            late_rate=("is_late", "mean"),
        )
        .reset_index()
    )
    grouped["late_rate"] *= 100
    return grouped


@st.cache_data
def build_category_cuts(
    consolidated_path: Path, reviews_path: Path, min_orders: int = 100
) -> pd.DataFrame:
    grouped = _delivery_review_cut(
        consolidated_path, reviews_path, ["product_category_name_english"]
    )
    return (
        grouped.loc[grouped["orders"] >= min_orders]
        .sort_values("mean_delivery_days", ascending=False)
        .reset_index(drop=True)
    )


@st.cache_data
def build_seller_cuts(
    consolidated_path: Path, reviews_path: Path, min_orders: int = 20
) -> pd.DataFrame:
    grouped = _delivery_review_cut(consolidated_path, reviews_path, ["seller_id"])
    return (
        grouped.loc[grouped["orders"] >= min_orders]
        .sort_values("orders", ascending=False)
        .reset_index(drop=True)
    )


@st.cache_data
def load_primary_payment_types(path: Path) -> pd.DataFrame:
    """An order can have more than one payment row (e.g. a voucher topped up
    with a card); `payment_sequential == 1` is Olist's primary payment record —
    same convention used in the Phase 1 EDA notebook for this exact reason."""
    payments = pd.read_csv(path, usecols=["order_id", "payment_sequential", "payment_type"])
    return payments.loc[payments["payment_sequential"].eq(1), ["order_id", "payment_type"]]


@st.cache_data
def build_payment_cuts(
    consolidated_path: Path, payments_path: Path, reviews_path: Path, min_orders: int = 50
) -> pd.DataFrame:
    orders = eligible_deliveries(load_data(consolidated_path)).drop_duplicates("order_id")
    orders = orders.merge(load_primary_payment_types(payments_path), on="order_id", how="inner")
    orders = orders.merge(_order_review_scores(reviews_path), on="order_id", how="left")
    orders["is_late"] = (
        orders["order_delivered_customer_date"] > orders["order_estimated_delivery_date"]
    )
    grouped = orders.groupby("payment_type", as_index=False).agg(
        orders=("order_id", "size"),
        mean_delivery_days=("delivery_days", "mean"),
        mean_review_score=("review_score", "mean"),
        late_rate=("is_late", "mean"),
    )
    grouped["late_rate"] *= 100
    return (
        grouped.loc[grouped["orders"] >= min_orders]
        .sort_values("orders", ascending=False)
        .reset_index(drop=True)
    )


@st.cache_data
def build_payment_stage_breakdown(orders_path: Path, payments_path: Path) -> pd.DataFrame:
    """Same processing/handling/shipping stages as `build_leadtime_decomposition`,
    grouped by payment type instead of collapsed to one overall summary."""
    stages = load_order_stage_timestamps(orders_path).dropna(
        subset=[
            "order_purchase_timestamp",
            "order_approved_at",
            "order_delivered_carrier_date",
            "order_delivered_customer_date",
        ]
    )
    stages["processing_time"] = (
        stages["order_approved_at"] - stages["order_purchase_timestamp"]
    ).dt.total_seconds() / 86_400
    stages["handling_time"] = (
        stages["order_delivered_carrier_date"] - stages["order_approved_at"]
    ).dt.total_seconds() / 86_400
    stages["shipping_time"] = (
        stages["order_delivered_customer_date"] - stages["order_delivered_carrier_date"]
    ).dt.total_seconds() / 86_400

    valid = (stages[LEAD_STAGES] >= 0).all(axis=1)
    clean = stages.loc[valid].merge(
        load_primary_payment_types(payments_path), on="order_id", how="inner"
    )

    grouped = clean.groupby("payment_type", as_index=False).agg(
        **{stage: (stage, "mean") for stage in LEAD_STAGES},
        orders=("order_id", "size"),
    )
    long = grouped.melt(
        id_vars=["payment_type", "orders"],
        value_vars=LEAD_STAGES,
        var_name="stage_key",
        value_name="days",
    )
    long["stage"] = long["stage_key"].map(LEAD_STAGE_LABELS)
    long["stage_order"] = long["stage_key"].map({stage: i for i, stage in enumerate(LEAD_STAGES)})
    return long


def build_retention_series(order_data: pd.DataFrame, granularity: str) -> tuple[pd.DataFrame, float]:
    customers = order_data.sort_values(["customer_unique_id", "order_purchase_timestamp", "order_id"]).copy()
    repeat_rate = customers.groupby("customer_unique_id")["order_id"].nunique().gt(1).mean()
    customers = add_period(customers, "order_purchase_timestamp", granularity)
    customers["first_period"] = customers.groupby("customer_unique_id")["period"].transform("min")
    customer_periods = customers.drop_duplicates(["customer_unique_id", "period"]).copy()
    customer_periods["returning_customer"] = customer_periods["period"].gt(customer_periods["first_period"])
    customer_periods = trim_trend_window(customer_periods, "order_purchase_timestamp")
    series = (
        customer_periods.groupby(["period", "returning_customer"])["customer_unique_id"]
        .nunique()
        .unstack(fill_value=0)
        .reindex(columns=[False, True], fill_value=0)
        .rename(columns={False: "new_customers", True: "returning_customers"})
        .reset_index()
    )
    series = complete_periods(series, granularity)
    series["returning_customer_share"] = series["returning_customers"] / (series["new_customers"] + series["returning_customers"]).replace(0, pd.NA) * 100
    return series, repeat_rate


# ---------------------------------------------------------------------------
# Operational Capacity tab
# ---------------------------------------------------------------------------

DAY_OF_WEEK_ORDER = [
    "Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday",
]


def add_week_period(data: pd.DataFrame, date_column: str) -> pd.DataFrame:
    return data.assign(period=data[date_column].dt.to_period("W").dt.start_time)


@st.cache_data
def build_capacity_series(order_data: pd.DataFrame) -> pd.DataFrame:
    """Weekly order volume vs. mean/median delivery time, to spot capacity strain."""
    delivered = trim_trend_window(eligible_deliveries(order_data), "order_purchase_timestamp")
    weekly = (
        add_week_period(delivered, "order_purchase_timestamp")
        .groupby("period", as_index=False)
        .agg(
            orders=("order_id", "nunique"),
            median_delivery_days=("delivery_days", "median"),
            mean_delivery_days=("delivery_days", "mean"),
        )
        .sort_values("period")
    )
    return weekly


@st.cache_data
def build_hour_dow_heatmap(order_data: pd.DataFrame) -> pd.DataFrame:
    """Mean delivery time by purchase day-of-week and hour-of-day."""
    delivered = eligible_deliveries(order_data).assign(
        purchase_hour=lambda frame: frame["order_purchase_timestamp"].dt.hour
    )
    grouped = delivered.groupby(["day_of_week", "purchase_hour"], as_index=False).agg(
        mean_delivery_days=("delivery_days", "mean"), orders=("order_id", "size")
    )
    return grouped


@st.cache_data
def build_freight_ratio_buckets(line_items: pd.DataFrame, buckets: int = 6) -> pd.DataFrame:
    """Freight-to-price ratio (order-level) vs. late-delivery rate."""
    order_economics = line_items.groupby("order_id", as_index=False).agg(
        price_sum=("price", "sum"), freight_sum=("freight_value", "sum")
    )
    order_status_cols = line_items.drop_duplicates("order_id")[
        [
            "order_id",
            "order_status",
            "order_delivered_customer_date",
            "order_estimated_delivery_date",
            "is_on_time",
        ]
    ]
    merged = order_economics.merge(order_status_cols, on="order_id", how="left")
    eligible = eligible_deliveries(merged)
    eligible = eligible.loc[eligible["price_sum"] > 0].copy()
    eligible["freight_ratio"] = eligible["freight_sum"] / eligible["price_sum"]
    eligible["late"] = ~eligible["is_on_time"].astype(bool)
    eligible["ratio_bucket"] = pd.qcut(eligible["freight_ratio"], buckets, duplicates="drop")

    grouped = (
        eligible.groupby("ratio_bucket", observed=True)
        .agg(
            mean_ratio=("freight_ratio", "mean"),
            late_rate=("late", "mean"),
            orders=("order_id", "size"),
        )
        .reset_index()
    )
    grouped["late_rate"] *= 100
    grouped["ratio_label"] = grouped["ratio_bucket"].apply(
        lambda interval: f"{max(interval.left, 0):.2f}-{interval.right:.2f}"
    )
    return grouped


@st.cache_data
def build_backlog_series(order_data: pd.DataFrame) -> pd.DataFrame:
    """Cumulative gap between orders placed and orders delivered, as a backlog proxy."""
    delivered = trim_trend_window(eligible_deliveries(order_data), "order_purchase_timestamp")
    placed = add_week_period(delivered, "order_purchase_timestamp").groupby("period").size()
    placed.name = "placed"
    completed = add_week_period(delivered, "order_delivered_customer_date").groupby("period").size()
    completed.name = "completed"

    combined = pd.concat([placed, completed], axis=1).fillna(0)
    full_index = pd.date_range(combined.index.min(), combined.index.max(), freq="W-MON")
    combined = combined.reindex(full_index, fill_value=0)
    combined.index.name = "period"
    combined = combined.reset_index()
    combined["backlog"] = (combined["placed"] - combined["completed"]).cumsum()
    return combined


@st.cache_data
def build_promise_buffer_series(order_data: pd.DataFrame) -> pd.DataFrame:
    """Weekly mean promised (estimated) vs. actual delivery days, to see whether
    Olist's own delivery-date promise reacts to capacity strain/backlog."""
    delivered = trim_trend_window(eligible_deliveries(order_data), "order_purchase_timestamp")
    delivered = delivered.assign(
        promised_days=lambda frame: (
            frame["order_estimated_delivery_date"] - frame["order_purchase_timestamp"]
        ).dt.total_seconds() / 86_400,
        actual_days=lambda frame: (
            frame["order_delivered_customer_date"] - frame["order_purchase_timestamp"]
        ).dt.total_seconds() / 86_400,
        late=lambda frame: frame["order_delivered_customer_date"] > frame["order_estimated_delivery_date"],
    )
    weekly = add_week_period(delivered, "order_purchase_timestamp").groupby(
        "period", as_index=False
    ).agg(
        orders=("order_id", "size"),
        mean_promised_days=("promised_days", "mean"),
        mean_actual_days=("actual_days", "mean"),
        late_rate=("late", "mean"),
    )
    weekly["buffer_days"] = weekly["mean_promised_days"] - weekly["mean_actual_days"]
    weekly["late_rate"] *= 100
    return weekly


def _clean_stage_durations(orders_path: Path) -> pd.DataFrame:
    """Per-order stage durations with missing-timestamp and negative-duration rows dropped.
    Reuses load_order_stage_timestamps() defined above for the Decomposition chart."""
    stages = load_order_stage_timestamps(orders_path).dropna(
        subset=[
            "order_purchase_timestamp",
            "order_approved_at",
            "order_delivered_carrier_date",
            "order_delivered_customer_date",
        ]
    )
    stages["processing_time"] = (
        stages["order_approved_at"] - stages["order_purchase_timestamp"]
    ).dt.total_seconds() / 86_400
    stages["handling_time"] = (
        stages["order_delivered_carrier_date"] - stages["order_approved_at"]
    ).dt.total_seconds() / 86_400
    stages["shipping_time"] = (
        stages["order_delivered_customer_date"] - stages["order_delivered_carrier_date"]
    ).dt.total_seconds() / 86_400

    valid = (stages[LEAD_STAGES] >= 0).all(axis=1)
    return stages.loc[valid].copy()


@st.cache_data
def build_stage_duration_series(orders_path: Path) -> pd.DataFrame:
    """Weekly mean/median duration of each fulfilment stage, to see which stage degrades under load."""
    clean = trim_trend_window(_clean_stage_durations(orders_path), "order_purchase_timestamp")

    grouped = add_week_period(clean, "order_purchase_timestamp").groupby("period", as_index=False)
    mean_weekly = grouped[LEAD_STAGES].mean().melt(
        id_vars="period", value_vars=LEAD_STAGES, var_name="stage_key", value_name="mean_days"
    )
    median_weekly = grouped[LEAD_STAGES].median().melt(
        id_vars="period", value_vars=LEAD_STAGES, var_name="stage_key", value_name="median_days"
    )
    long = mean_weekly.merge(median_weekly, on=["period", "stage_key"])
    long["stage"] = long["stage_key"].map(LEAD_STAGE_LABELS)
    return long


@st.cache_data
def build_stage_duration_by_weekday(orders_path: Path) -> pd.DataFrame:
    """Mean duration of each fulfilment stage by the day-of-week the order was purchased."""
    clean = _clean_stage_durations(orders_path)
    clean["day_of_week"] = clean["order_purchase_timestamp"].dt.day_name()

    by_weekday = clean.groupby("day_of_week", as_index=False)[LEAD_STAGES].mean()
    long = by_weekday.melt(
        id_vars="day_of_week", value_vars=LEAD_STAGES, var_name="stage_key", value_name="mean_days"
    )
    long["stage"] = long["stage_key"].map(LEAD_STAGE_LABELS)
    return long


# ---------------------------------------------------------------------------
# Problem Candidate 1 tab (delivery lead-time prediction)
# ---------------------------------------------------------------------------


@st.cache_data
def build_weekday_delivery_summary(order_data: pd.DataFrame) -> pd.DataFrame:
    """Mean delivery days and late-rate by the day of week the order was purchased."""
    delivered = eligible_deliveries(order_data).drop_duplicates("order_id")
    grouped = delivered.groupby("day_of_week", as_index=False).agg(
        mean_delivery_days=("delivery_days", "mean"),
        late_rate=("is_on_time", lambda values: (~values.astype(bool)).mean()),
        orders=("order_id", "size"),
    )
    grouped["late_rate"] *= 100
    grouped["day_order"] = grouped["day_of_week"].map(
        {day: index for index, day in enumerate(DAY_OF_WEEK_ORDER)}
    )
    return grouped.sort_values("day_order")


@st.cache_data
def build_complexity_delivery_summary(line_items: pd.DataFrame) -> pd.DataFrame:
    """Mean delivery days by order complexity (item count, seller count)."""
    orders = eligible_deliveries(line_items).drop_duplicates("order_id")[
        ["order_id", "delivery_days"]
    ]
    item_counts = line_items.groupby("order_id").size().rename("item_count")
    seller_counts = line_items.groupby("order_id")["seller_id"].nunique().rename("seller_count")
    merged = orders.merge(item_counts, on="order_id").merge(seller_counts, on="order_id")

    groups = {
        "Single item": merged["item_count"].eq(1),
        "Multiple items": merged["item_count"].gt(1),
        "Single seller": merged["item_count"].gt(1) & merged["seller_count"].eq(1),
        "Multiple sellers": merged["item_count"].gt(1) & merged["seller_count"].gt(1),
    }
    rows = [
        {"group": label, "mean_delivery_days": merged.loc[mask, "delivery_days"].mean(), "orders": int(mask.sum())}
        for label, mask in groups.items()
    ]
    return pd.DataFrame(rows)


@st.cache_data
def build_weight_buckets(line_items: pd.DataFrame, buckets: int = 6) -> pd.DataFrame:
    """Total order weight (summed across items) vs. delivery days and late-rate."""
    orders = eligible_deliveries(line_items).drop_duplicates("order_id")[
        ["order_id", "delivery_days", "is_on_time"]
    ]
    order_weight = line_items.groupby("order_id", as_index=False)["product_weight_g"].sum()
    merged = orders.merge(order_weight, on="order_id", how="left").dropna(subset=["product_weight_g"])
    merged = merged.loc[merged["product_weight_g"] > 0].copy()
    merged["late"] = ~merged["is_on_time"].astype(bool)
    merged["weight_bucket"] = pd.qcut(merged["product_weight_g"], buckets, duplicates="drop")

    grouped = (
        merged.groupby("weight_bucket", observed=True)
        .agg(
            mean_weight_g=("product_weight_g", "mean"),
            mean_delivery_days=("delivery_days", "mean"),
            late_rate=("late", "mean"),
            orders=("order_id", "size"),
        )
        .reset_index()
    )
    grouped["late_rate"] *= 100
    grouped["weight_label"] = grouped["weight_bucket"].apply(
        lambda interval: f"{max(interval.left, 0):,.0f}–{interval.right:,.0f}"
    )
    return grouped
