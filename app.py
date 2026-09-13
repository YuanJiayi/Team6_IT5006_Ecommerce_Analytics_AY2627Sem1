"""Streamlit dashboard for the Olist exploratory analysis."""

from pathlib import Path

import altair as alt
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components

from dashboard_data import (
    build_backlog_series,
    build_capacity_series,
    build_category_cuts,
    build_commercial_series,
    build_complexity_delivery_summary,
    build_delivery_experience_series,
    build_delivery_review_analysis,
    build_distance_buckets,
    build_distance_table,
    build_freight_ratio_buckets,
    build_growth_series,
    build_hour_dow_heatmap,
    build_leadtime_decomposition,
    build_payment_cuts,
    build_payment_stage_breakdown,
    build_promise_buffer_series,
    build_rating_complexity_summary,
    build_rating_delivery_timing_summary,
    build_review_score_correlations,
    build_seller_cuts,
    build_stage_duration_by_weekday,
    build_stage_duration_series,
    build_weekday_delivery_summary,
    build_weight_buckets,
    DAY_OF_WEEK_ORDER,
    eligible_deliveries,
    latest_reviews as select_latest_reviews,
    load_customer_ids,
    load_data,
    load_reviews,
)

DATA_PATH = Path("data/smartcommerce_consolidated.csv")
CUSTOMERS_PATH = Path("data/olist_customers_dataset.csv")
REVIEWS_PATH = Path("data/olist_order_reviews_dataset.csv")
ORDERS_PATH = Path("data/olist_orders_dataset.csv")
SELLERS_PATH = Path("data/olist_sellers_dataset.csv")
GEOLOCATION_PATH = Path("data/olist_geolocation_dataset.csv")
PAYMENTS_PATH = Path("data/olist_order_payments_dataset.csv")
ORDER_ITEMS_PATH = Path("data/olist_order_items_dataset.csv")
MAP_PATH = Path("brazil_order_density_map.html")
BLUE = "#003D7C"
ORANGE = "#EF7C00"


def figure_heading(number: int, title: str) -> None:
    """Give every chart a stable reference number."""
    st.markdown(f"### Figure {number}. {title}")


def line_chart(
    data: pd.DataFrame,
    column: str,
    y_title: str,
    tooltip_format: str,
    color: str,
    show_points: bool,
) -> alt.Chart:
    """Build a consistently labelled zero-baseline time-series chart."""
    return (
        alt.Chart(data)
        .mark_line(point=show_points, color=color, strokeWidth=2.5)
        .encode(
            x=alt.X("period:T", title="Period"),
            y=alt.Y(f"{column}:Q", title=y_title, scale=alt.Scale(zero=True)),
            tooltip=[
                alt.Tooltip("period:T", title="Period"),
                alt.Tooltip(f"{column}:Q", title=y_title, format=tooltip_format),
            ],
        )
        .properties(height=340)
    )

st.set_page_config(page_title="IT5006 Olist Dashboard", layout="wide")
st.title("Olist E-Commerce Dashboard")

overview_tab, delivery_correlations_tab, capacity_tab, map_tab, problem1_tab, ratings_tab = st.tabs(
    ["Overview", "Delivery correlations", "Operational Capacity", "Mapped orders and sellers", "Candidate 1", "Candidate 2"]
)

with overview_tab:
    required_paths = [DATA_PATH, CUSTOMERS_PATH, REVIEWS_PATH]
    if not all(path.exists() for path in required_paths):
        st.error("Required dashboard data files could not be found in the `data` folder.")
        st.stop()

    line_items = load_data(DATA_PATH)
    customer_ids = load_customer_ids(CUSTOMERS_PATH)
    reviews = load_reviews(REVIEWS_PATH)
    order_data = (
        line_items.drop_duplicates("order_id")
        .merge(customer_ids, on="customer_id", how="left", validate="one_to_one")
    )

    order_count = line_items["order_id"].nunique()
    items_sold = len(line_items)
    product_revenue = line_items["price"].sum()
    customer_count = order_data["customer_unique_id"].nunique()
    seller_count = line_items["seller_id"].nunique()
    purchase_time = line_items["order_purchase_timestamp"]

    category_summary = (
        line_items.assign(
            product_category_name_english=line_items[
                "product_category_name_english"
            ].fillna("Unknown")
        )
        .groupby("product_category_name_english", as_index=False)
        .agg(items_sold=("order_item_id", "size"), product_sales=("price", "sum"))
        .sort_values("items_sold", ascending=False)
    )
    state_summary = (
        line_items.groupby("customer_state", as_index=False)
        .agg(orders=("order_id", "nunique"))
        .sort_values("orders", ascending=False)
    )
    state_summary["order_share"] = state_summary["orders"] / order_count * 100
    sao_paulo_share = state_summary.loc[
        state_summary["customer_state"].eq("SP"), "order_share"
    ].iloc[0]

    seller_summary = (
        line_items.groupby("seller_id", as_index=False)
        .agg(orders=("order_id", "nunique"), product_sales=("price", "sum"))
        .sort_values("product_sales", ascending=False)
    )
    core_order_ids = order_data[["order_id"]]
    order_reviews = (
        select_latest_reviews(reviews)[["order_id", "review_score"]]
        .merge(core_order_ids, on="order_id", how="inner", validate="one_to_one")
    )
    average_score = order_reviews["review_score"].mean()
    five_star_rate = order_reviews["review_score"].eq(5).mean()
    low_rating_rate = order_reviews["review_score"].le(2).mean()

    eligible_delivery_orders = eligible_deliveries(order_data)
    st.caption(
        f"Purchase-date coverage: {purchase_time.min():%d %b %Y}–"
        f"{purchase_time.max():%d %b %Y}. Each transaction row represents one order item; "
        "product sales exclude freight."
    )
    headline_columns = st.columns(5)
    headline_columns[0].metric("Product sales", f"R$ {product_revenue:,.0f}")
    headline_columns[1].metric("Orders", f"{order_count:,}")
    headline_columns[2].metric("Items sold", f"{items_sold:,}")
    headline_columns[3].metric("Unique customers", f"{customer_count:,}")
    headline_columns[4].metric("Sellers", f"{seller_count:,}")

    st.divider()
    st.header("Growth and commercial performance")
    st.caption(
        "Figures 1–2 show January 2017–August 2018, excluding the launch period and "
        "incomplete boundary month."
    )
    granularity = st.radio(
        "Time granularity for Figures 1–2",
        ["Day", "Month"],
        index=1,
        horizontal=True,
        key="overview_granularity",
    )

    figure_heading(1, "Order and product-sales growth over time")
    growth_series = build_growth_series(line_items, granularity)
    orders_growth_chart = line_chart(
        growth_series, "orders", "Orders", ",.0f", BLUE, granularity != "Day"
    )
    sales_growth_chart = line_chart(
        growth_series,
        "product_revenue",
        "Product sales (R$)",
        ",.0f",
        ORANGE,
        granularity != "Day",
    )
    orders_growth_column, sales_growth_column = st.columns(2)
    with orders_growth_column:
        st.markdown("#### Total orders")
        st.altair_chart(orders_growth_chart, use_container_width=True)
    with sales_growth_column:
        st.markdown("#### Product sales")
        st.altair_chart(sales_growth_chart, use_container_width=True)
    figure_heading(2, "Commercial performance over time")
    commercial_series = build_commercial_series(line_items, granularity)
    commercial_aov_column, commercial_items_column = st.columns(2)
    with commercial_aov_column:
        st.markdown("#### Average order value")
        st.altair_chart(
            line_chart(
                commercial_series,
                "average_order_value",
                "Average order value (R$)",
                ",.2f",
                BLUE,
                granularity != "Day",
            ),
            use_container_width=True,
        )
    with commercial_items_column:
        st.markdown("#### Items per order")
        st.altair_chart(
            line_chart(
                commercial_series,
                "items_per_order",
                "Items per order",
                ".2f",
                ORANGE,
                granularity != "Day",
            ),
            use_container_width=True,
        )
    st.caption("Average order value is product sales per order.")

    st.divider()
    st.header("Marketplace composition")
    geography_column, category_column = st.columns(2)
    with geography_column:
        st.metric("São Paulo share of orders", f"{sao_paulo_share:.1f}%")
        figure_heading(3, "Top 10 customer states by orders")
        top_states = state_summary.head(10)
        geography_chart = (
            alt.Chart(top_states)
            .mark_bar(color=BLUE)
            .encode(
                y=alt.Y("customer_state:N", title="Customer state", sort="-x"),
                x=alt.X("orders:Q", title="Orders", scale=alt.Scale(zero=True)),
                tooltip=[
                    alt.Tooltip("customer_state:N", title="State"),
                    alt.Tooltip("orders:Q", title="Orders", format=","),
                    alt.Tooltip("order_share:Q", title="Share of all orders (%)", format=".1f"),
                ],
            )
            .properties(height=350)
        )
        st.altair_chart(geography_chart, use_container_width=True)

    with category_column:
        figure_heading(4, "Top 10 product categories")
        category_metric = st.radio(
            "Category metric",
            ["Items sold", "Product sales"],
            index=0,
            horizontal=True,
            key="category_metric",
        )
        category_options = {
            "Items sold": ("items_sold", "Items sold", items_sold, ",.0f"),
            "Product sales": (
                "product_sales",
                "Product sales (R$)",
                product_revenue,
                ",.0f",
            ),
        }
        category_value, category_axis, category_total, category_format = category_options[
            category_metric
        ]
        top_categories = category_summary.nlargest(10, category_value).copy()
        top_categories["metric_share"] = (
            top_categories[category_value] / category_total * 100
        )
        st.metric(
            f"Top 10 category share of {category_metric.lower()}",
            f"{top_categories[category_value].sum() / category_total:.1%}",
        )
        category_chart = (
            alt.Chart(top_categories)
            .mark_bar(color=ORANGE)
            .encode(
                y=alt.Y(
                    "product_category_name_english:N",
                    title="Product category",
                    sort="-x",
                ),
                x=alt.X(
                    f"{category_value}:Q",
                    title=category_axis,
                    scale=alt.Scale(zero=True),
                ),
                tooltip=[
                    alt.Tooltip("product_category_name_english:N", title="Category"),
                    alt.Tooltip(
                        f"{category_value}:Q",
                        title=category_axis,
                        format=category_format,
                    ),
                    alt.Tooltip(
                        "metric_share:Q",
                        title=f"Share of all {category_metric.lower()} (%)",
                        format=".1f",
                    ),
                ],
            )
            .properties(height=350)
        )
        st.altair_chart(category_chart, use_container_width=True)
        st.caption(
            "Missing or untranslated categories are retained as Unknown."
        )

    st.divider()
    st.header("Seller and customer structure")
    figure_heading(5, "Cumulative product sales by seller share")
    seller_pareto = seller_summary.reset_index(drop=True).copy()
    seller_pareto["seller_rank"] = seller_pareto.index + 1
    seller_pareto["seller_share"] = (
        seller_pareto["seller_rank"] / seller_count * 100
    )
    seller_pareto["cumulative_sales_share"] = (
        seller_pareto["product_sales"].cumsum() / product_revenue * 100
    )
    seller_pareto = pd.concat(
        [
            pd.DataFrame(
                {
                    "seller_id": ["Start"],
                    "orders": [0],
                    "product_sales": [0.0],
                    "seller_rank": [0],
                    "seller_share": [0.0],
                    "cumulative_sales_share": [0.0],
                }
            ),
            seller_pareto,
        ],
        ignore_index=True,
    )
    top_decile_count = (seller_count + 9) // 10
    top_decile = seller_pareto.loc[
        seller_pareto["seller_rank"].eq(top_decile_count)
    ].copy()
    top_decile_share = top_decile["cumulative_sales_share"].iloc[0]
    pareto_line = (
        alt.Chart(seller_pareto)
        .mark_line(color=BLUE, strokeWidth=3)
        .encode(
            x=alt.X(
                "seller_share:Q",
                title="Cumulative share of sellers (%)",
                scale=alt.Scale(domain=[0, 100]),
            ),
            y=alt.Y(
                "cumulative_sales_share:Q",
                title="Cumulative share of product sales (%)",
                scale=alt.Scale(domain=[0, 100]),
            ),
            tooltip=[
                alt.Tooltip("seller_rank:Q", title="Sellers included", format=","),
                alt.Tooltip(
                    "seller_share:Q", title="Cumulative seller share (%)", format=".1f"
                ),
                alt.Tooltip(
                    "cumulative_sales_share:Q",
                    title="Cumulative product-sales share (%)",
                    format=".1f",
                ),
            ],
        )
    )
    equality_line = (
        alt.Chart(
            pd.DataFrame(
                {
                    "seller_share": [0, 100],
                    "cumulative_sales_share": [0, 100],
                }
            )
        )
        .mark_line(color="#8C8C8C", strokeDash=[6, 5])
        .encode(x="seller_share:Q", y="cumulative_sales_share:Q")
    )
    top_decile_rule = (
        alt.Chart(top_decile)
        .mark_rule(color=ORANGE, strokeDash=[4, 4])
        .encode(x="seller_share:Q")
    )
    top_decile_point = (
        alt.Chart(top_decile)
        .mark_point(color=ORANGE, filled=True, size=110)
        .encode(x="seller_share:Q", y="cumulative_sales_share:Q")
    )
    st.altair_chart(
        (equality_line + pareto_line + top_decile_rule + top_decile_point).properties(
            height=380
        ),
        use_container_width=True,
    )
    st.caption(
        f"Sellers are ranked from highest to lowest product sales. The dashed diagonal "
        f"shows equal distribution; the top 10% ({top_decile_count:,} sellers) generated "
        f"{top_decile_share:.1f}% of product sales."
    )

    figure_heading(6, "Customer purchase-frequency distribution")
    customer_frequency = (
        order_data.groupby("customer_unique_id")["order_id"]
        .nunique()
        .rename("Orders")
        .reset_index()
    )
    customer_frequency["Purchase-frequency band"] = pd.cut(
        customer_frequency["Orders"],
        bins=[0, 1, 2, 3, float("inf")],
        labels=["1 order", "2 orders", "3 orders", "4+ orders"],
    )
    customer_frequency_summary = (
        customer_frequency.groupby("Purchase-frequency band", observed=False)
        .size()
        .rename("Customers")
        .reset_index()
    )
    customer_frequency_summary["Customer share"] = (
        customer_frequency_summary["Customers"] / customer_count * 100
    )
    customer_frequency_summary["Share label"] = customer_frequency_summary[
        "Customer share"
    ].map(lambda value: f"{value:.2f}%")
    frequency_bars = (
        alt.Chart(customer_frequency_summary)
        .mark_bar(color=ORANGE)
        .encode(
            x=alt.X(
                "Purchase-frequency band:N",
                title="Orders per customer",
                sort=["1 order", "2 orders", "3 orders", "4+ orders"],
            ),
            y=alt.Y(
                "Customer share:Q",
                title="Customers (%)",
                scale=alt.Scale(domain=[0, 100]),
            ),
            tooltip=[
                alt.Tooltip("Purchase-frequency band:N", title="Orders per customer"),
                alt.Tooltip("Customers:Q", title="Customers", format=","),
                alt.Tooltip(
                    "Customer share:Q", title="Share of customers (%)", format=".2f"
                ),
            ],
        )
        .properties(height=330)
    )
    frequency_labels = frequency_bars.mark_text(dy=-9, color="#31333F").encode(
        text="Share label:N"
    )
    st.altair_chart(frequency_bars + frequency_labels, use_container_width=True)
    st.caption("Purchase frequency covers the full dataset period.")

    st.divider()
    st.header("Customer reviews")
    experience_columns = st.columns(3)
    experience_columns[0].metric("Average review score", f"{average_score:.2f} / 5")
    experience_columns[1].metric("Five-star reviews", f"{five_star_rate:.1%}")
    experience_columns[2].metric("Low ratings", f"{low_rating_rate:.1%}")
    figure_heading(7, "Review-score distribution")
    review_distribution = order_reviews.groupby("review_score", as_index=False).agg(
        reviews=("order_id", "size")
    )
    review_distribution["review_share"] = (
        review_distribution["reviews"] / len(order_reviews) * 100
    )
    review_chart = (
        alt.Chart(review_distribution)
        .mark_bar(color=BLUE)
        .encode(
            x=alt.X("review_score:O", title="Review score", sort=[1, 2, 3, 4, 5]),
            y=alt.Y("reviews:Q", title="Reviewed orders", scale=alt.Scale(zero=True)),
            tooltip=[
                alt.Tooltip("review_score:O", title="Review score"),
                alt.Tooltip("reviews:Q", title="Reviewed orders", format=","),
                alt.Tooltip("review_share:Q", title="Share of reviews (%)", format=".1f"),
            ],
        )
        .properties(height=330)
    )
    st.altair_chart(review_chart, use_container_width=True)
    st.caption("One latest review per consolidated order; low rating means 1–2 stars.")

    st.divider()
    st.header("Delivery performance and customer ratings")
    figure_heading(8, "Late-delivery and low-rating rates over time")
    delivery_experience = build_delivery_experience_series(order_data, reviews)
    delivery_experience_long = delivery_experience.melt(
        id_vars="period",
        value_vars=["Late-delivery rate (%)", "Low-rating rate (%)"],
        var_name="Metric",
        value_name="Rate",
    )
    delivery_trend = (
        alt.Chart(delivery_experience_long)
        .mark_line(point=True, strokeWidth=2.5)
        .encode(
            x=alt.X("period:T", title="Month"),
            y=alt.Y("Rate:Q", title="Rate (%)", scale=alt.Scale(zero=True)),
            color=alt.Color(
                "Metric:N",
                title=None,
                scale=alt.Scale(
                    domain=["Late-delivery rate (%)", "Low-rating rate (%)"],
                    range=[BLUE, ORANGE],
                ),
            ),
            tooltip=[
                alt.Tooltip("period:T", title="Month", format="%b %Y"),
                alt.Tooltip("Metric:N", title="Metric"),
                alt.Tooltip("Rate:Q", title="Rate", format=".1f"),
            ],
        )
        .properties(height=350)
    )
    st.altair_chart(delivery_trend, use_container_width=True)
    st.caption(
        "Late-delivery and low-rating rates increased together during the main peaks. "
        "Results cover January 2017–August 2018 and orders with the required delivery and review data."
    )

    figure_heading(9, "Low-rating rate by delivery timing")
    comparison, _, risk_ratio = build_delivery_review_analysis(order_data, reviews)
    reviewed_delivery_counts = (
        eligible_delivery_orders[[
            "order_id",
            "order_delivered_customer_date",
            "order_estimated_delivery_date",
        ]]
        .merge(order_reviews, on="order_id", how="inner", validate="one_to_one")
        .assign(
            late_delivery=lambda frame: frame["order_delivered_customer_date"]
            > frame["order_estimated_delivery_date"]
        )
        .groupby("late_delivery", as_index=False)
        .agg(orders=("order_id", "size"))
    )
    comparison = comparison.merge(
        reviewed_delivery_counts, on="late_delivery", how="left", validate="one_to_one"
    )
    comparison["low_rating_label"] = comparison.apply(
        lambda row: f"{row['low_rating_rate']:.1f}% (n={row['orders']:,.0f})", axis=1
    )
    comparison_chart = (
        alt.Chart(comparison)
        .mark_bar()
        .encode(
            x=alt.X(
                "delivery_timing:N",
                title=None,
                sort=["On time or early", "Late"],
            ),
            y=alt.Y(
                "low_rating_rate:Q",
                title="Low-rating rate (%)",
                scale=alt.Scale(zero=True),
            ),
            color=alt.Color(
                "delivery_timing:N",
                title=None,
                scale=alt.Scale(
                    domain=["On time or early", "Late"],
                    range=[BLUE, ORANGE],
                ),
                legend=None,
            ),
            tooltip=[
                alt.Tooltip("delivery_timing:N", title="Delivery timing"),
                alt.Tooltip("low_rating_rate:Q", title="Low-rating rate", format=".1f"),
                alt.Tooltip("orders:Q", title="Reviewed orders", format=","),
            ],
        )
        .properties(height=330)
    )
    comparison_labels = comparison_chart.mark_text(dy=-10, color="#31333F").encode(
        text="low_rating_label:N"
    )
    st.altair_chart(comparison_chart + comparison_labels, use_container_width=True)
    st.metric("Late-order low-rating risk ratio", f"{risk_ratio:.2f}×")
    st.caption(
        "Delivered orders with an estimated date and latest review. The comparison shows "
        "an association and does not establish causation."
    )


def boxed_label(text: str) -> None:
    """Render a filled label chip that captions the chart directly below it."""
    st.markdown(
        "<span style='display:inline-block;background:#003D7C;color:#FFFFFF;"
        "font-weight:600;font-size:0.85rem;letter-spacing:0.03em;"
        "text-transform:uppercase;padding:3px 12px;border-radius:4px;"
        f"margin-bottom:10px;'>{text}</span>",
        unsafe_allow_html=True,
    )


with delivery_correlations_tab:
    corr_required = [
        DATA_PATH,
        ORDERS_PATH,
        CUSTOMERS_PATH,
        SELLERS_PATH,
        GEOLOCATION_PATH,
        REVIEWS_PATH,
        PAYMENTS_PATH,
    ]
    if not all(path.exists() for path in corr_required):
        st.error("Required data files for this tab could not be found in the `data` folder.")
        st.stop()

    st.caption(
        "Order-level view. `order_approved_at` / `order_delivered_carrier_date` are "
        "read from `olist_orders_dataset.csv` (absent from the consolidated file) and "
        "merged on `order_id`."
    )

    (
        stage_summary,
        shipping_review,
        excluded_share,
        total_lead_mean,
    ) = build_leadtime_decomposition(ORDERS_PATH, REVIEWS_PATH)
    distance_table, dropped_share = build_distance_table(
        DATA_PATH, CUSTOMERS_PATH, SELLERS_PATH, GEOLOCATION_PATH, REVIEWS_PATH
    )
    distance_buckets = build_distance_buckets(distance_table)
    category_cuts = build_category_cuts(DATA_PATH, REVIEWS_PATH, min_orders=100)
    top_categories = category_cuts.head(13)
    category_order = top_categories["product_category_name_english"].tolist()
    seller_cuts = build_seller_cuts(DATA_PATH, REVIEWS_PATH, min_orders=20)
    volume_threshold = float(seller_cuts["orders"].quantile(0.90))
    seller_cuts = seller_cuts.assign(
        volume_band=lambda frame: frame["orders"]
        .ge(volume_threshold)
        .map({True: "Top-decile volume", False: "Other sellers"})
    )
    payment_cuts = build_payment_cuts(DATA_PATH, PAYMENTS_PATH, REVIEWS_PATH, min_orders=50)
    payment_order = payment_cuts["payment_type"].tolist()
    payment_stages = build_payment_stage_breakdown(ORDERS_PATH, PAYMENTS_PATH)

    # payment_cuts is already sorted by order count descending, so the first row
    # is the most-used method — the natural baseline to compare the slowest against.
    slowest_payment = payment_cuts.loc[payment_cuts["mean_delivery_days"].idxmax()]
    common_payment = payment_cuts.iloc[0]
    payment_stage_pivot = payment_stages.pivot(index="payment_type", columns="stage_key", values="days")
    processing_gap = (
        payment_stage_pivot.loc[slowest_payment["payment_type"], "processing_time"]
        - payment_stage_pivot.loc[common_payment["payment_type"], "processing_time"]
    )
    handling_gap = (
        payment_stage_pivot.loc[slowest_payment["payment_type"], "handling_time"]
        - payment_stage_pivot.loc[common_payment["payment_type"], "handling_time"]
    )
    shipping_gap = (
        payment_stage_pivot.loc[slowest_payment["payment_type"], "shipping_time"]
        - payment_stage_pivot.loc[common_payment["payment_type"], "shipping_time"]
    )

    # Processing is a tiny slice, so it gets the high-contrast orange to pop;
    # handling takes the dark blue, shipping the lighter blue.
    STAGE_COLORS = ["#EF7C00", "#003D7C", "#7FA9D0"]

    row1_left, row1_right = st.columns(2)

    with row1_left:
        with st.container(border=True):
            boxed_label("Decomposition")
            stat_choice = st.radio(
                "Value used for the breakdown",
                ["Mean", "Median"],
                horizontal=True,
                key="leadtime_stat",
            )
            value_column = {"Mean": "mean_days", "Median": "median_days"}[stat_choice]
            stage_order = stage_summary["stage"].tolist()
            stage_share = stage_summary[["stage", value_column]].rename(
                columns={value_column: "days"}
            )
            stage_share["share_pct"] = stage_share["days"] / stage_share["days"].sum() * 100
            stage_share["slice_label"] = stage_share.apply(
                lambda row: f"{row['days']:.1f}d ({row['share_pct']:.0f}%)", axis=1
            )

            stage_base = alt.Chart(stage_share).encode(
                theta=alt.Theta("days:Q", stack=True),
                color=alt.Color(
                    "stage:N",
                    title=None,
                    sort=stage_order,
                    scale=alt.Scale(domain=stage_order, range=STAGE_COLORS),
                    legend=alt.Legend(orient="bottom", columns=1),
                ),
                order=alt.Order("stage:N", sort="ascending"),
                tooltip=[
                    alt.Tooltip("stage:N", title="Stage"),
                    alt.Tooltip("days:Q", title=f"{stat_choice} days", format=".2f"),
                    alt.Tooltip("share_pct:Q", title="Share of shipping time", format=".1f"),
                ],
            )
            stage_pie = stage_base.mark_arc(
                outerRadius=88, stroke="#FFFFFF", strokeWidth=2
            )
            stage_pie_labels = stage_base.mark_text(
                radius=112, fontSize=12, fontWeight="bold"
            ).encode(text="slice_label:N", color=alt.value("#31333F"))
            st.altair_chart(
                (stage_pie + stage_pie_labels).properties(height=340),
                use_container_width=True,
            )
            total_selected = stage_share["days"].sum()
            st.caption(
                f"Slices show each stage's {stat_choice.lower()} duration as a share of "
                f"the three stages combined ({total_selected:.1f} days; mean total "
                f"shipping time {total_lead_mean:.1f} days). Shipping (carrier → "
                "customer) is the dominant stage. Toggle switches every slice between "
                f"the mean and the median stage duration. Excluded {excluded_share:.1f}% "
                "of orders with a negative stage duration (inconsistent timestamps) and "
                "any order missing one of the four timestamps."
            )

    with row1_right:
        with st.container(border=True):
            boxed_label("Mean review score")
            shipping_review_display = shipping_review.assign(
                shipping_bucket=shipping_review["shipping_bucket"].astype(str)
            )
            shipping_line = alt.Chart(shipping_review_display).mark_line(
                point=True, color="#EF7C00"
            ).encode(
                x=alt.X(
                    "shipping_bucket:N",
                    title="Shipping time (days, carrier → customer)",
                    sort=["0–3", "3–7", "7–14", "14–21", "21–30", "30+"],
                ),
                y=alt.Y(
                    "mean_review_score:Q",
                    title="Mean review score",
                    scale=alt.Scale(domain=[1, 5]),
                ),
                tooltip=[
                    alt.Tooltip("shipping_bucket:N", title="Shipping days"),
                    alt.Tooltip(
                        "mean_review_score:Q", title="Mean review score", format=".2f"
                    ),
                    alt.Tooltip("orders:Q", title="Orders", format=","),
                ],
            )
            st.altair_chart(
                shipping_line.properties(height=340), use_container_width=True
            )
            st.caption(
                f"Delivered orders with a review; negative-duration rows already excluded "
                f"({excluded_share:.1f}%). Buckets are days, right-open."
            )

    with st.container(border=True):
        boxed_label("Geographic distance")
        dist_left, dist_right = st.columns(2)
        with dist_left:
            st.markdown("###### Mean delivery days by customer–seller distance")
            dist_days = alt.Chart(distance_buckets).mark_bar(color="#003D7C").encode(
                x=alt.X(
                    "distance_label:N",
                    title="Distance (km)",
                    sort=distance_buckets["distance_label"].tolist(),
                ),
                y=alt.Y("mean_delivery_days:Q", title="Mean delivery days"),
                tooltip=[
                    alt.Tooltip("distance_label:N", title="Distance (km)"),
                    alt.Tooltip(
                        "mean_delivery_days:Q", title="Mean delivery days", format=".1f"
                    ),
                    alt.Tooltip("orders:Q", title="Orders", format=","),
                ],
            )
            st.altair_chart(dist_days.properties(height=300), use_container_width=True)
        with dist_right:
            st.markdown("###### Mean review score & late-rate by distance")
            base = alt.Chart(distance_buckets).encode(
                x=alt.X(
                    "distance_label:N",
                    title="Distance (km)",
                    sort=distance_buckets["distance_label"].tolist(),
                )
            )
            score_line = base.mark_line(point=True, color="#EF7C00").encode(
                y=alt.Y(
                    "mean_review_score:Q",
                    title="Mean review score",
                    scale=alt.Scale(domain=[1, 5]),
                ),
                tooltip=[
                    alt.Tooltip("distance_label:N", title="Distance (km)"),
                    alt.Tooltip(
                        "mean_review_score:Q", title="Mean review score", format=".2f"
                    ),
                    alt.Tooltip("late_rate:Q", title="Late rate (%)", format=".1f"),
                ],
            )
            late_line = base.mark_line(
                point=True, color="#003D7C", strokeDash=[4, 3]
            ).encode(y=alt.Y("late_rate:Q", title="Late rate (%)"))
            st.altair_chart(
                alt.layer(score_line, late_line)
                .resolve_scale(y="independent")
                .properties(height=300),
                use_container_width=True,
            )
        st.caption(
            "Distance is a haversine km between customer and seller zip-prefix "
            f"centroids (mean lat/lng per prefix from the geolocation table), not exact "
            f"addresses. Dropped {dropped_share:.1f}% of delivered orders whose customer "
            "or seller zip prefix is absent from the geolocation table. Buckets are "
            "sextiles of distance. Orange = review score (left axis), dashed blue = "
            "late rate (right axis)."
        )

    with st.container(border=True):
        boxed_label("Category")
        cat_left, cat_right = st.columns(2)
        with cat_left:
            st.markdown("###### Mean delivery days")
            cat_days = alt.Chart(top_categories).mark_bar(color="#003D7C").encode(
                y=alt.Y(
                    "product_category_name_english:N",
                    title=None,
                    sort=category_order,
                ),
                x=alt.X("mean_delivery_days:Q", title="Mean delivery days"),
                tooltip=[
                    alt.Tooltip(
                        "product_category_name_english:N", title="Category"
                    ),
                    alt.Tooltip(
                        "mean_delivery_days:Q", title="Mean delivery days", format=".1f"
                    ),
                    alt.Tooltip("orders:Q", title="Orders", format=","),
                ],
            )
            st.altair_chart(cat_days.properties(height=380), use_container_width=True)
        with cat_right:
            st.markdown("###### Mean review score")
            cat_score = alt.Chart(top_categories).mark_bar(color="#EF7C00").encode(
                y=alt.Y(
                    "product_category_name_english:N",
                    title=None,
                    sort=category_order,
                    axis=alt.Axis(labels=False),
                ),
                x=alt.X(
                    "mean_review_score:Q",
                    title="Mean review score",
                    scale=alt.Scale(domain=[1, 5]),
                ),
                tooltip=[
                    alt.Tooltip(
                        "product_category_name_english:N", title="Category"
                    ),
                    alt.Tooltip(
                        "mean_review_score:Q", title="Mean review score", format=".2f"
                    ),
                    alt.Tooltip("late_rate:Q", title="Late rate (%)", format=".1f"),
                ],
            )
            st.altair_chart(cat_score.properties(height=380), use_container_width=True)
        st.caption(
            "Categories with ≥100 delivered orders, top 13 by mean delivery time. Same "
            "row order in both charts. Late rate = share delivered after the estimated "
            "date (`is_on_time` from the consolidated file). One row per (order, category)."
        )

    with st.container(border=True):
        boxed_label("Seller volume size")
        seller_scatter = alt.Chart(seller_cuts).mark_circle(opacity=0.55).encode(
            x=alt.X(
                "orders:Q", title="Delivered orders", scale=alt.Scale(type="log")
            ),
            y=alt.Y("late_rate:Q", title="Late rate (%)"),
            size=alt.Size("orders:Q", title="Orders", legend=None),
            color=alt.Color(
                "volume_band:N",
                title=None,
                scale=alt.Scale(
                    domain=["Top-decile volume", "Other sellers"],
                    range=["#EF7C00", "#003D7C"],
                ),
            ),
            tooltip=[
                alt.Tooltip("seller_id:N", title="Seller"),
                alt.Tooltip("orders:Q", title="Orders", format=","),
                alt.Tooltip("late_rate:Q", title="Late rate (%)", format=".1f"),
                alt.Tooltip(
                    "mean_delivery_days:Q", title="Mean delivery days", format=".1f"
                ),
                alt.Tooltip(
                    "mean_review_score:Q", title="Mean review score", format=".2f"
                ),
            ],
        )
        st.altair_chart(
            seller_scatter.properties(height=340), use_container_width=True
        )
        st.caption(
            f"Sellers with ≥20 delivered orders. Orange = top-decile volume "
            f"(≥{volume_threshold:.0f} orders); x-axis log-scaled. Marker size also "
            "encodes order volume. Late rate from `is_on_time` in the consolidated "
            "file. One row per (order, seller)."
        )

    with st.container(border=True):
        boxed_label("Payment method")
        pay_left, pay_right = st.columns(2)
        with pay_left:
            st.markdown("###### Mean delivery days by payment method")
            pay_days = alt.Chart(payment_cuts).mark_bar(color="#003D7C").encode(
                x=alt.X("payment_type:N", title="Payment method", sort=payment_order),
                y=alt.Y("mean_delivery_days:Q", title="Mean delivery days"),
                tooltip=[
                    alt.Tooltip("payment_type:N", title="Payment type"),
                    alt.Tooltip("mean_delivery_days:Q", title="Mean delivery days", format=".1f"),
                    alt.Tooltip("late_rate:Q", title="Late rate (%)", format=".1f"),
                    alt.Tooltip("orders:Q", title="Orders", format=","),
                ],
            )
            st.altair_chart(pay_days.properties(height=300), use_container_width=True)
        with pay_right:
            st.markdown("###### Mean review score & late-rate by payment method")
            pay_base = alt.Chart(payment_cuts).encode(
                x=alt.X("payment_type:N", title="Payment method", sort=payment_order)
            )
            pay_score_line = pay_base.mark_line(point=True, color="#EF7C00").encode(
                y=alt.Y("mean_review_score:Q", title="Mean review score", scale=alt.Scale(domain=[1, 5])),
                tooltip=[
                    alt.Tooltip("payment_type:N", title="Payment type"),
                    alt.Tooltip("mean_review_score:Q", title="Mean review score", format=".2f"),
                    alt.Tooltip("late_rate:Q", title="Late rate (%)", format=".1f"),
                ],
            )
            pay_late_line = pay_base.mark_line(point=True, color="#003D7C", strokeDash=[4, 3]).encode(
                y=alt.Y("late_rate:Q", title="Late rate (%)"),
            )
            st.altair_chart(
                alt.layer(pay_score_line, pay_late_line).resolve_scale(y="independent").properties(height=300),
                use_container_width=True,
            )
        st.caption(
            "Payment types with ≥50 matched delivered orders. Primary payment "
            "(`payment_sequential == 1`) per order, from "
            "`olist_order_payments_dataset.csv`. Orange = review score (left axis), "
            "dashed blue = late rate (right axis)."
        )

    with st.container(border=True):
        boxed_label("Payment method — lead-time breakdown")
        stage_bar = alt.Chart(payment_stages).mark_bar().encode(
            y=alt.Y("payment_type:N", title=None, sort=payment_order),
            x=alt.X("days:Q", title="Mean days", stack="zero"),
            color=alt.Color(
                "stage:N",
                title=None,
                sort=stage_order,
                scale=alt.Scale(domain=stage_order, range=STAGE_COLORS),
                legend=alt.Legend(orient="bottom", columns=3),
            ),
            order=alt.Order("stage_order:Q"),
            tooltip=[
                alt.Tooltip("payment_type:N", title="Payment type"),
                alt.Tooltip("stage:N", title="Stage"),
                alt.Tooltip("days:Q", title="Mean delivery days", format=".2f"),
                alt.Tooltip("orders:Q", title="Orders", format=","),
            ],
        )
        st.altair_chart(stage_bar.properties(height=260), use_container_width=True)
        st.caption(
            "Same processing/handling/shipping stages slicing as the "
            "Decomposition chart above, grouped by payment type instead of collapsed "
            "into one overall summary. Same exclusions as Decomposition (negative "
            "duration or missing timestamps)."
        )
with capacity_tab:
    capacity_required = [DATA_PATH, ORDERS_PATH]
    if not all(path.exists() for path in capacity_required):
        st.error("Required data files for this tab could not be found in the `data` folder.")
        st.stop()

    st.caption(
        "Trend charts show January 2017 to August 2018, excluding launch and partial "
        "boundary periods. `order_approved_at` / `order_delivered_carrier_date` are read "
        "from `olist_orders_dataset.csv` (absent from the consolidated file) and merged "
        "on `order_id`."
    )

    capacity_line_items = load_data(DATA_PATH)
    capacity_orders = capacity_line_items.drop_duplicates("order_id")

    st.subheader("Capacity strain: weekly order volume vs. delivery time")
    capacity_stat_choice = st.radio(
        "Value used for delivery time",
        ["Median", "Mean"],
        horizontal=True,
        key="capacity_stat",
    )
    capacity_value_column = {"Median": "median_delivery_days", "Mean": "mean_delivery_days"}[
        capacity_stat_choice
    ]
    capacity_series = build_capacity_series(capacity_orders)
    capacity_base = alt.Chart(capacity_series).encode(x=alt.X("period:T", title=None))
    volume_bars = capacity_base.mark_bar(color="#7FA9D0").encode(
        y=alt.Y("orders:Q", title="Orders placed"),
        tooltip=[
            alt.Tooltip("period:T", title="Week of"),
            alt.Tooltip("orders:Q", title="Orders", format=","),
            alt.Tooltip(f"{capacity_value_column}:Q", title=f"{capacity_stat_choice} delivery days", format=".1f"),
        ],
    )
    lead_time_line = capacity_base.mark_line(color="#EF7C00", point=True).encode(
        y=alt.Y(f"{capacity_value_column}:Q", title=f"{capacity_stat_choice} delivery days")
    )
    st.altair_chart(
        alt.layer(volume_bars, lead_time_line).resolve_scale(y="independent").properties(height=320),
        use_container_width=True,
    )
    st.caption(
        f"Weekly order volume (bars, left axis) vs. {capacity_stat_choice.lower()} delivery "
        "time (orange line, right axis). If lead time rises alongside volume spikes, delay "
        "is partly a capacity/throughput problem, not just a per-order attribute."
    )

    st.subheader("Does weekly volume actually predict delivery time?")
    volume_corr = capacity_series["orders"].corr(capacity_series[capacity_value_column])
    volume_scatter = alt.Chart(capacity_series).mark_circle(color=BLUE, size=90, opacity=0.75).encode(
        x=alt.X("orders:Q", title="Weekly order volume"),
        y=alt.Y(f"{capacity_value_column}:Q", title=f"{capacity_stat_choice} delivery days (that week)"),
        tooltip=[
            alt.Tooltip("period:T", title="Week of"),
            alt.Tooltip("orders:Q", title="Orders", format=","),
            alt.Tooltip(f"{capacity_value_column}:Q", title=f"{capacity_stat_choice} delivery days", format=".1f"),
        ],
    )
    volume_trend = volume_scatter.transform_regression("orders", capacity_value_column).mark_line(
        color="#EF7C00", strokeDash=[5, 3], strokeWidth=2.5
    )
    st.altair_chart(
        (volume_scatter + volume_trend).properties(height=340),
        use_container_width=True,
    )
    st.caption(
        f"Each point is one week (n={len(capacity_series)}). Pearson r = {volume_corr:.2f} between "
        "weekly order volume and that week's delivery time, a weak relationship. Points spread across "
        "almost the full range of delivery times at nearly every volume level, so a busy week alone does "
        "not reliably predict a slow week; delay is better explained by other factors such as accumulated "
        "backlog (see below) than by concurrent volume."
    )

    st.subheader("Purchase timing: day-of-week × hour heatmap")
    heatmap_metric_choice = st.radio(
        "Colour by",
        ["Mean delivery days", "Order volume"],
        horizontal=True,
        key="heatmap_metric",
    )
    heatmap_metric_column = {"Mean delivery days": "mean_delivery_days", "Order volume": "orders"}[
        heatmap_metric_choice
    ]
    heatmap_data = build_hour_dow_heatmap(capacity_orders)
    heatmap_chart = alt.Chart(heatmap_data).mark_rect().encode(
        x=alt.X("purchase_hour:O", title="Purchase hour"),
        y=alt.Y("day_of_week:N", title=None, sort=DAY_OF_WEEK_ORDER),
        color=alt.Color(
            f"{heatmap_metric_column}:Q",
            title=heatmap_metric_choice,
            scale=alt.Scale(scheme="blues"),
        ),
        tooltip=[
            alt.Tooltip("day_of_week:N", title="Day"),
            alt.Tooltip("purchase_hour:O", title="Hour"),
            alt.Tooltip("mean_delivery_days:Q", title="Mean delivery days", format=".1f"),
            alt.Tooltip("orders:Q", title="Orders", format=","),
        ],
    )
    st.altair_chart(heatmap_chart.properties(height=320), use_container_width=True)
    st.caption(
        "Colour toggle switches between mean delivery days and order volume by the "
        "day-of-week and hour of the purchase timestamp. Purchase timing is known at "
        "order time, making it a zero-cost candidate feature."
    )

    st.subheader("Which stage slows down for weekend purchases?")
    weekday_stage = build_stage_duration_by_weekday(ORDERS_PATH)
    stage_key_order = ["processing_time", "handling_time", "shipping_time"]
    stage_key_colors = dict(zip(stage_key_order, ["#EF7C00", "#003D7C", "#7FA9D0"]))
    stage_key_labels = dict(zip(stage_key_order, weekday_stage.drop_duplicates("stage_key").set_index("stage_key")["stage"]))
    weekday_cols = st.columns(3)
    for col, stage_key in zip(weekday_cols, stage_key_order):
        stage_data = weekday_stage.loc[weekday_stage["stage_key"] == stage_key]
        with col:
            st.markdown(f"###### {stage_key_labels[stage_key]}")
            stage_bar = alt.Chart(stage_data).mark_bar(color=stage_key_colors[stage_key]).encode(
                x=alt.X("day_of_week:N", title=None, sort=DAY_OF_WEEK_ORDER, axis=alt.Axis(labelAngle=-45)),
                y=alt.Y("mean_days:Q", title="Mean days"),
                tooltip=[
                    alt.Tooltip("day_of_week:N", title="Purchase day"),
                    alt.Tooltip("mean_days:Q", title="Mean days", format=".2f"),
                ],
            )
            st.altair_chart(stage_bar.properties(height=260), use_container_width=True)
    st.caption(
        "Mean duration of each fulfilment stage by the day-of-week the order was "
        "purchased (each stage has its own y-axis, since shipping is ~20x longer than "
        "processing). Tests whether the day-of-week effect seen above sits specifically "
        "in one stage (e.g. weekend purchases queuing before approval) rather than "
        "being spread evenly across the pipeline. Negative-duration and incomplete-"
        "timestamp rows excluded, same as the stage-duration chart below."
    )

    st.subheader("Freight cost efficiency: freight-to-price ratio vs. late-rate")
    freight_buckets = build_freight_ratio_buckets(capacity_line_items)
    freight_base = alt.Chart(freight_buckets).encode(
        x=alt.X("ratio_label:N", title="Freight ÷ price", sort=freight_buckets["ratio_label"].tolist())
    )
    freight_bars = freight_base.mark_bar(color="#7FA9D0").encode(
        y=alt.Y("late_rate:Q", title="Late-delivery rate (%)"),
        tooltip=[
            alt.Tooltip("ratio_label:N", title="Freight ÷ price"),
            alt.Tooltip("late_rate:Q", title="Late-delivery rate (%)", format=".1f"),
            alt.Tooltip("orders:Q", title="Orders", format=","),
        ],
    )
    st.altair_chart(freight_bars.properties(height=300), use_container_width=True)
    st.caption(
        "Orders bucketed into sextiles of (order-level freight value ÷ price). Late-rate "
        "stays roughly flat across buckets (about 7.8-8.4%), so despite bundling "
        "distance, weight and carrier pricing into one number, this ratio alone isn't a "
        "strong standalone predictor of lateness."
    )

    st.subheader("Backlog: orders placed vs. delivered over time")
    backlog_series = build_backlog_series(capacity_orders)
    backlog_chart = alt.Chart(backlog_series).mark_area(
        color="#EF7C00", opacity=0.6, line={"color": "#EF7C00"}
    ).encode(
        x=alt.X("period:T", title=None),
        y=alt.Y("backlog:Q", title="Cumulative backlog (orders)"),
        tooltip=[
            alt.Tooltip("period:T", title="Week of"),
            alt.Tooltip("placed:Q", title="Orders placed", format=","),
            alt.Tooltip("completed:Q", title="Orders delivered", format=","),
            alt.Tooltip("backlog:Q", title="Cumulative backlog", format=","),
        ],
    )
    st.altair_chart(backlog_chart.properties(height=300), use_container_width=True)
    st.caption(
        "Cumulative (orders placed − orders delivered) by week. Caveat: the final few "
        "weeks are inflated by right-censoring, since recently placed orders haven't had "
        "time to be delivered yet as of the data snapshot. That's not the same as a real "
        "operational backlog."
    )

    st.subheader("Does Olist's own delivery promise react to capacity strain?")
    promise_secondary_choice = st.radio(
        "Secondary axis",
        ["Late-delivery rate", "Backlog"],
        horizontal=True,
        key="promise_secondary",
    )
    promise_series = build_promise_buffer_series(capacity_orders)
    promise_series = promise_series.merge(
        backlog_series[["period", "backlog"]], on="period", how="left"
    )
    promise_long = promise_series.melt(
        id_vars=["period", "orders", "buffer_days", "late_rate", "backlog"],
        value_vars=["mean_promised_days", "mean_actual_days"],
        var_name="series",
        value_name="days",
    )
    promise_long["series"] = promise_long["series"].map(
        {"mean_promised_days": "Promised (estimated delivery date)", "mean_actual_days": "Actual"}
    )
    promise_base = alt.Chart(promise_long).encode(x=alt.X("period:T", title=None))
    promise_lines = promise_base.mark_line(point=True).encode(
        y=alt.Y("days:Q", title="Mean days from purchase"),
        color=alt.Color(
            "series:N",
            title=None,
            scale=alt.Scale(
                domain=["Promised (estimated delivery date)", "Actual"],
                range=["#7FA9D0", "#EF7C00"],
            ),
            legend=alt.Legend(orient="bottom"),
        ),
        tooltip=[
            alt.Tooltip("period:T", title="Week of"),
            alt.Tooltip("series:N", title="Series"),
            alt.Tooltip("days:Q", title="Mean days", format=".1f"),
        ],
    )
    promise_secondary_column = {
        "Late-delivery rate": "late_rate",
        "Backlog": "backlog",
    }[promise_secondary_choice]
    promise_secondary_title = {
        "Late-delivery rate": "Late-delivery rate (%)",
        "Backlog": "Cumulative backlog (orders)",
    }[promise_secondary_choice]
    secondary_line = alt.Chart(promise_series).mark_line(
        point=True, color="#003D7C", strokeDash=[4, 3]
    ).encode(
        x=alt.X("period:T", title=None),
        y=alt.Y(f"{promise_secondary_column}:Q", title=promise_secondary_title),
        tooltip=[
            alt.Tooltip("period:T", title="Week of"),
            alt.Tooltip(f"{promise_secondary_column}:Q", title=promise_secondary_title, format=".1f"),
        ],
    )
    st.altair_chart(
        alt.layer(promise_lines, secondary_line).resolve_scale(y="independent").properties(height=340),
        use_container_width=True,
    )
    st.caption(
        "Mean promised (order_estimated_delivery_date minus purchase timestamp) vs. "
        "actual delivery days (left axis), plus the toggled secondary metric (dashed "
        "dark-blue line, right axis), by week. The promised window tends to widen with "
        "a lag after volume spikes (it barely moves during the Nov 2017 peak week itself, "
        "but keeps climbing for weeks afterward). It correlates more with a several-week "
        "trailing average of actual delivery performance (r≈0.45 at 8-12 weeks) than "
        "with current-week backlog (r=-0.26), suggesting a slow-reacting historical "
        "baseline rather than a live capacity signal. Watch whether late-rate falls as "
        "the promise widens: since the late/on-time label is defined relative to this "
        "promise, a wider promise can lower the late-rate even if actual delivery isn't "
        "getting faster."
    )

    st.subheader("Where does the delay accumulate? Stage duration over time")
    stage_series_stat_choice = st.radio(
        "Value used for stage duration",
        ["Mean", "Median"],
        horizontal=True,
        key="stage_series_stat",
    )
    stage_series_value_column = {"Mean": "mean_days", "Median": "median_days"}[stage_series_stat_choice]
    stage_series = build_stage_duration_series(ORDERS_PATH)
    stage_order = list(dict.fromkeys(stage_series.sort_values("period")["stage"]))
    stage_area = alt.Chart(stage_series).mark_area().encode(
        x=alt.X("period:T", title=None),
        y=alt.Y(
            f"{stage_series_value_column}:Q",
            title=f"{stage_series_stat_choice} stage duration (days)",
            stack="zero",
        ),
        color=alt.Color(
            "stage:N",
            title=None,
            sort=stage_order,
            scale=alt.Scale(range=["#EF7C00", "#003D7C", "#7FA9D0"]),
            legend=alt.Legend(orient="bottom", columns=1),
        ),
        tooltip=[
            alt.Tooltip("period:T", title="Week of"),
            alt.Tooltip("stage:N", title="Stage"),
            alt.Tooltip(f"{stage_series_value_column}:Q", title=f"{stage_series_stat_choice} days", format=".2f"),
        ],
    )
    st.altair_chart(stage_area.properties(height=340), use_container_width=True)
    st.caption(
        f"Weekly {stage_series_stat_choice.lower()} duration of each fulfilment stage, "
        "stacked. Shows whether backlog during high-volume periods concentrates in a "
        "specific stage (e.g. warehouse hand-off vs. carrier transit) rather than "
        "spreading evenly. Negative-duration and incomplete-timestamp rows excluded."
    )

with ratings_tab:
    ratings_required = [ORDER_ITEMS_PATH, ORDERS_PATH, REVIEWS_PATH]
    if not all(path.exists() for path in ratings_required):
        st.error("Required data files for this tab could not be found in the `data` folder.")
        st.stop()

    st.header("Low-rating candidate exploration")
    st.caption(
        "A low rating is a latest review score of 1–2. Analysis includes reviewed orders "
        "with at least one recorded item. Relationships shown are exploratory associations "
        "and do not establish causation."
    )

    complexity_summary = build_rating_complexity_summary(
        ORDER_ITEMS_PATH, REVIEWS_PATH
    )
    timing_summary = build_rating_delivery_timing_summary(
        ORDERS_PATH, ORDER_ITEMS_PATH, REVIEWS_PATH
    )
    score_correlations = build_review_score_correlations(
        DATA_PATH, CUSTOMERS_PATH, SELLERS_PATH, GEOLOCATION_PATH, REVIEWS_PATH
    )

    figure_heading(10, "Low-rating rate by delivery timing")
    timing_order = timing_summary.sort_values("timing_order")[
        "delivery_timing"
    ].astype(str).tolist()
    timing_base = alt.Chart(timing_summary).encode(
        x=alt.X(
            "delivery_timing:N",
            title="Performance against estimated delivery date",
            sort=timing_order,
            axis=alt.Axis(labelAngle=-20),
        ),
        tooltip=[
            alt.Tooltip("delivery_timing:N", title="Delivery timing"),
            alt.Tooltip(
                "low_rating_rate:Q", title="Low-rating rate (%)", format=".1f"
            ),
            alt.Tooltip("orders:Q", title="Reviewed orders", format=","),
        ],
    )
    timing_intervals = timing_base.mark_rule(
        color="#31333F", strokeWidth=2
    ).encode(
        y=alt.Y(
            "lower_rate:Q",
            title="Low-rating rate (%)",
            scale=alt.Scale(domain=[0, 85]),
        ),
        y2="upper_rate:Q",
    )
    timing_line = timing_base.mark_line(
        color=ORANGE, point=alt.OverlayMarkDef(filled=True, size=120), strokeWidth=2.5
    ).encode(
        y=alt.Y(
            "low_rating_rate:Q",
            title="Low-rating rate (%)",
            scale=alt.Scale(domain=[0, 85]),
        )
    )
    st.altair_chart(
        (timing_intervals + timing_line).properties(height=360),
        use_container_width=True,
    )
    st.caption(
        "Low-rating rates rise as deliveries become later. Vertical lines are 95% Wilson "
        "confidence intervals."
    )

    st.divider()
    figure_columns = st.columns(2)
    complexity_order = [
        "Single item",
        "Multiple items",
        "Single seller",
        "Multiple sellers",
    ]
    complexity_chart_data = complexity_summary.assign(
        display_group=lambda frame: frame["group"].str.split().str[0]
    )
    complexity_base = alt.Chart(complexity_chart_data).encode(
        x=alt.X(
            "display_group:N",
            title=None,
            sort=["Single", "Multiple"],
            axis=alt.Axis(labelAngle=0, labelLimit=110),
        ),
        tooltip=[
            alt.Tooltip("group:N", title="Order group"),
            alt.Tooltip(
                "low_rating_rate:Q", title="Low-rating rate (%)", format=".1f"
            ),
            alt.Tooltip("orders:Q", title="Reviewed orders", format=","),
            alt.Tooltip(
                "order_share:Q", title="Share of eligible orders (%)", format=".2f"
            ),
            alt.Tooltip(
                "low_rating_capture:Q",
                title="Share of low ratings captured (%)",
                format=".1f",
            ),
        ],
    ).properties(height=360, width=250)
    complexity_intervals = complexity_base.mark_rule(
        color="#31333F", strokeWidth=2
    ).encode(
        y=alt.Y(
            "lower_rate:Q",
            title="Low-rating rate (%)",
            scale=alt.Scale(domain=[0, 55]),
        ),
        y2="upper_rate:Q",
    )
    complexity_lower_caps = complexity_base.mark_tick(
        color="#31333F", thickness=2, width=14
    ).encode(
        y=alt.Y(
            "lower_rate:Q",
            title="Low-rating rate (%)",
            scale=alt.Scale(domain=[0, 55]),
        )
    )
    complexity_upper_caps = complexity_base.mark_tick(
        color="#31333F", thickness=2, width=14
    ).encode(
        y=alt.Y(
            "upper_rate:Q",
            title="Low-rating rate (%)",
            scale=alt.Scale(domain=[0, 55]),
        )
    )
    complexity_points = complexity_base.mark_point(
        color=BLUE, filled=True, size=70
    ).encode(
        y=alt.Y(
            "low_rating_rate:Q",
            title="Low-rating rate (%)",
            scale=alt.Scale(domain=[0, 55]),
        )
    )
    complexity_chart = alt.layer(
        complexity_intervals,
        complexity_lower_caps,
        complexity_upper_caps,
        complexity_points,
    ).facet(
        column=alt.Column(
            "dimension:N",
            title=None,
            sort=["Number of items", "Number of sellers"],
            header=alt.Header(labelFontSize=14, labelFontWeight="bold"),
        ),
        spacing=45,
    ).resolve_scale(x="independent")

    correlation_base = alt.Chart(score_correlations).encode(
        y=alt.Y("feature:N", title=None, sort=score_correlations["feature"].tolist()),
        x=alt.X(
            "correlation:Q",
            title="Pearson correlation with review score",
            scale=alt.Scale(domain=[-0.45, 0]),
        ),
        tooltip=[
            alt.Tooltip("feature:N", title="Feature"),
            alt.Tooltip("correlation:Q", title="Pearson correlation", format=".3f"),
            alt.Tooltip("orders:Q", title="Reviewed orders", format=","),
        ],
    )
    correlation_bars = correlation_base.mark_bar(color=ORANGE)
    correlation_zero = alt.Chart(pd.DataFrame({"zero": [0]})).mark_rule(
        color="#31333F", strokeWidth=1
    ).encode(x="zero:Q")
    with figure_columns[0]:
        figure_heading(11, "Low-rating rate by order complexity")
        st.altair_chart(complexity_chart, use_container_width=True)
        st.markdown(
            "<div style='min-height:48px'>"
            "<small>Multi-item and multi-seller orders have higher low-rating rates. "
            "Lines show 95% Wilson confidence intervals.</small></div>",
            unsafe_allow_html=True,
        )
    with figure_columns[1]:
        figure_heading(12, "Feature correlations with review score")
        st.altair_chart(
            (correlation_bars + correlation_zero).properties(height=430),
            use_container_width=True,
        )
        st.markdown(
            "<div style='min-height:48px'>"
            "<small>Correlations use reviewed orders with item records. Bars show "
            "negative associations, not causal effects.</small></div>",
            unsafe_allow_html=True,
        )


with problem1_tab:
    st.header("Candidate Problem 1: Delivery lead-time prediction")
    st.caption(
        "This page presents the exploratory evidence supporting Problem Statement 1: "
        "predicting delivery lead time as a regression on delivery days, and "
        "predicting on-time/late status as a classification against Olist's "
        "estimated delivery date. Each chart below examines a candidate predictor's "
        "relationship with delivery outcomes."
    )

    weekday_summary = build_weekday_delivery_summary(capacity_orders)
    complexity_delivery = build_complexity_delivery_summary(line_items)
    weight_buckets = build_weight_buckets(capacity_line_items)
    p1_stage_colors = ["#EF7C00", "#003D7C", "#7FA9D0"]

    st.subheader("Delivery time and late-rate by customer–seller distance")
    dist_base = alt.Chart(distance_buckets).encode(
        x=alt.X(
            "distance_label:N",
            title="Distance (km, sextiles)",
            sort=distance_buckets["distance_label"].tolist(),
        )
    )
    dist_bars = dist_base.mark_bar(color="#7FA9D0").encode(
        y=alt.Y("mean_delivery_days:Q", title="Mean delivery days"),
        tooltip=[
            alt.Tooltip("distance_label:N", title="Distance (km)"),
            alt.Tooltip("mean_delivery_days:Q", title="Mean delivery days", format=".1f"),
            alt.Tooltip("orders:Q", title="Orders", format=","),
        ],
    )
    dist_late_line = dist_base.mark_line(color="#EF7C00", point=True).encode(
        y=alt.Y("late_rate:Q", title="Late rate (%)"),
        tooltip=[alt.Tooltip("late_rate:Q", title="Late rate (%)", format=".1f")],
    )
    st.altair_chart(
        alt.layer(dist_bars, dist_late_line).resolve_scale(y="independent").properties(height=320),
        use_container_width=True,
    )
    st.caption(
        "Bars indicate mean delivery days (left axis); the line indicates the "
        "late-delivery rate against Olist's estimate (right axis), both by distance "
        "sextile. Mean delivery time increases substantially with distance, more "
        "than tripling between the nearest and farthest sextiles, while the "
        "late-rate increases only modestly over the same range. This divergence "
        "suggests that Olist's estimated delivery date already partially accounts "
        "for distance, making distance a stronger candidate predictor for the "
        "regression target than for the classification target."
    )

    st.subheader("Purchase timing: day-of-week × hour heatmap")
    p1_heatmap_metric_choice = st.radio(
        "Colour by",
        ["Mean delivery days", "Order volume"],
        horizontal=True,
        key="p1_heatmap_metric",
    )
    p1_heatmap_metric_column = {"Mean delivery days": "mean_delivery_days", "Order volume": "orders"}[
        p1_heatmap_metric_choice
    ]
    p1_heatmap_data = build_hour_dow_heatmap(capacity_orders)
    p1_heatmap_chart = alt.Chart(p1_heatmap_data).mark_rect().encode(
        x=alt.X("purchase_hour:O", title="Purchase hour"),
        y=alt.Y("day_of_week:N", title=None, sort=DAY_OF_WEEK_ORDER),
        color=alt.Color(
            f"{p1_heatmap_metric_column}:Q",
            title=p1_heatmap_metric_choice,
            scale=alt.Scale(scheme="blues"),
        ),
        tooltip=[
            alt.Tooltip("day_of_week:N", title="Day"),
            alt.Tooltip("purchase_hour:O", title="Hour"),
            alt.Tooltip("mean_delivery_days:Q", title="Mean delivery days", format=".1f"),
            alt.Tooltip("orders:Q", title="Orders", format=","),
        ],
    )
    st.altair_chart(p1_heatmap_chart.properties(height=320), use_container_width=True)
    st.caption(
        "Purchase day and hour are observed at order placement and therefore carry "
        "no lookahead risk as model inputs. The heatmap indicates that mean "
        "delivery time is not uniformly distributed across the purchase window, "
        "supporting purchase timing as a viable, zero-cost predictor for the "
        "lead-time model."
    )

    st.subheader("Mean delivery time by day of week purchased")
    weekday_chart = alt.Chart(weekday_summary).mark_bar(color="#003D7C").encode(
        x=alt.X("day_of_week:N", title=None, sort=DAY_OF_WEEK_ORDER),
        y=alt.Y("mean_delivery_days:Q", title="Mean delivery days"),
        tooltip=[
            alt.Tooltip("day_of_week:N", title="Purchase day"),
            alt.Tooltip("mean_delivery_days:Q", title="Mean delivery days", format=".2f"),
            alt.Tooltip("late_rate:Q", title="Late rate (%)", format=".1f"),
            alt.Tooltip("orders:Q", title="Orders", format=","),
        ],
    ).properties(height=300)
    st.altair_chart(weekday_chart, use_container_width=True)
    st.caption(
        "Mean delivery time varies by purchase day, ranging from 11.47 days for "
        "Sunday purchases to 13.12 days for Friday purchases, a difference of "
        "approximately 14%. This indicates that purchase day of week carries "
        "predictive signal independent of distance or payment method."
    )

    st.subheader("Which stage slows down for weekend purchases?")
    p1_weekday_stage = build_stage_duration_by_weekday(ORDERS_PATH)
    p1_stage_key_order = ["processing_time", "handling_time", "shipping_time"]
    p1_stage_key_colors = dict(zip(p1_stage_key_order, p1_stage_colors))
    p1_stage_key_labels = dict(
        zip(p1_stage_key_order, p1_weekday_stage.drop_duplicates("stage_key").set_index("stage_key")["stage"])
    )
    p1_weekday_cols = st.columns(3)
    for col, stage_key in zip(p1_weekday_cols, p1_stage_key_order):
        stage_data = p1_weekday_stage.loc[p1_weekday_stage["stage_key"] == stage_key]
        with col:
            st.markdown(f"###### {p1_stage_key_labels[stage_key]}")
            stage_bar = alt.Chart(stage_data).mark_bar(color=p1_stage_key_colors[stage_key]).encode(
                x=alt.X("day_of_week:N", title=None, sort=DAY_OF_WEEK_ORDER, axis=alt.Axis(labelAngle=-45)),
                y=alt.Y("mean_days:Q", title="Mean days"),
                tooltip=[
                    alt.Tooltip("day_of_week:N", title="Purchase day"),
                    alt.Tooltip("mean_days:Q", title="Mean days", format=".2f"),
                ],
            )
            st.altair_chart(stage_bar.properties(height=240), use_container_width=True)
    st.caption(
        "Decomposing total lead time into its three constituent stages localises "
        "the day-of-week effect observed above to the order-handling stage "
        "(approval to carrier pickup); the shipping stage remains approximately "
        "constant regardless of purchase day. This indicates that the day-of-week "
        "effect originates in Olist's internal order-handling process rather than "
        "in carrier transit, a distinction relevant to feature attribution."
    )

    st.subheader("Delivery time by payment method")
    p1_pay_days = alt.Chart(payment_cuts).mark_bar(color="#003D7C").encode(
        x=alt.X("payment_type:N", title=None, sort=payment_order),
        y=alt.Y("mean_delivery_days:Q", title="Mean delivery days"),
        tooltip=[
            alt.Tooltip("payment_type:N", title="Payment type"),
            alt.Tooltip("mean_delivery_days:Q", title="Mean delivery days", format=".2f"),
            alt.Tooltip("late_rate:Q", title="Late rate (%)", format=".1f"),
            alt.Tooltip("orders:Q", title="Orders", format=","),
        ],
    )
    st.altair_chart(p1_pay_days.properties(height=300), use_container_width=True)
    st.caption(
        "Mean delivery time differs by payment method: Boleto orders average 13.04 "
        "days versus 11.88 for credit card. This is consistent with Boleto's "
        "manual bank-confirmation requirement, which delays order approval prior "
        "to fulfilment."
    )

    st.subheader("Lead-time breakdown by payment method")
    p1_stage_bar = alt.Chart(payment_stages).mark_bar().encode(
        y=alt.Y("payment_type:N", title=None, sort=payment_order),
        x=alt.X("days:Q", title="Mean days", stack="zero"),
        color=alt.Color(
            "stage:N",
            title=None,
            sort=stage_order,
            scale=alt.Scale(domain=stage_order, range=STAGE_COLORS),
            legend=alt.Legend(orient="bottom", columns=3),
        ),
        order=alt.Order("stage_order:Q"),
        tooltip=[
            alt.Tooltip("payment_type:N", title="Payment type"),
            alt.Tooltip("stage:N", title="Stage"),
            alt.Tooltip("days:Q", title="Mean delivery days", format=".2f"),
            alt.Tooltip("orders:Q", title="Orders", format=","),
        ],
    )
    st.altair_chart(p1_stage_bar.properties(height=300), use_container_width=True)
    st.caption(
        "The same stage decomposition, grouped by payment method, confirms that "
        "Boleto's additional lead time is concentrated in the processing stage "
        "(purchase to approval), consistent with the manual confirmation "
        "requirement, rather than being distributed across handling or shipping."
    )

    st.subheader("Delivery time by order complexity")
    complexity_order = ["Single item", "Multiple items", "Single seller", "Multiple sellers"]
    complexity_delivery_chart = alt.Chart(complexity_delivery).mark_bar(color="#003D7C").encode(
        x=alt.X("group:N", title=None, sort=complexity_order),
        y=alt.Y("mean_delivery_days:Q", title="Mean delivery days"),
        tooltip=[
            alt.Tooltip("group:N", title="Order group"),
            alt.Tooltip("mean_delivery_days:Q", title="Mean delivery days", format=".2f"),
            alt.Tooltip("orders:Q", title="Orders", format=","),
        ],
    ).properties(height=300)
    st.altair_chart(complexity_delivery_chart, use_container_width=True)
    st.caption(
        "Order complexity shows a counter-intuitive relationship with delivery "
        "time: multi-seller orders arrive faster on average than single-seller "
        "orders (8.66 versus 12.14 days), and multi-item orders arrive modestly "
        "faster than single-item orders (11.36 versus 12.17 days). The "
        "multi-seller group is comparatively small (n = 1,275; order counts are "
        "available on hover), so this estimate should be treated with some "
        "caution pending further investigation of the underlying mechanism."
    )

    st.subheader("Delivery time by order weight")
    p1_weight_base = alt.Chart(weight_buckets).encode(
        x=alt.X(
            "weight_label:N",
            title="Total order weight (g, sextiles)",
            sort=weight_buckets["weight_label"].tolist(),
        )
    )
    p1_weight_bars = p1_weight_base.mark_bar(color="#7FA9D0").encode(
        y=alt.Y("mean_delivery_days:Q", title="Mean delivery days"),
        tooltip=[
            alt.Tooltip("weight_label:N", title="Weight (g)"),
            alt.Tooltip("mean_delivery_days:Q", title="Mean delivery days", format=".2f"),
            alt.Tooltip("orders:Q", title="Orders", format=","),
        ],
    )
    p1_weight_line = p1_weight_base.mark_line(color="#EF7C00", point=True).encode(
        y=alt.Y("late_rate:Q", title="Late rate (%)"),
        tooltip=[alt.Tooltip("late_rate:Q", title="Late rate (%)", format=".1f")],
    )
    st.altair_chart(
        alt.layer(p1_weight_bars, p1_weight_line).resolve_scale(y="independent").properties(height=320),
        use_container_width=True,
    )
    st.caption(
        "Total order weight shows only a weak association with delivery time and "
        "late-rate relative to distance; both series are comparatively flat across "
        "weight sextiles. This suggests that, on its own, weight is a weaker "
        "candidate predictor for the lead-time model than distance, purchase "
        "timing, or payment method."
    )


with map_tab:
    st.header("Brazil order and seller map")
    st.caption(
        "Interactive Folium map generated using zipcodes from geolocation dataset"
        " and publically available .geojson values for Brazil"
        ". Use the layer control in the top-right corner to switch between order density, seller density, "
        "seller-buyer routes, and state order density."
    )
    if not MAP_PATH.exists():
        st.error("Map not found. Run geomapper.py from the project folder first.")
    else:
        components.html(
            MAP_PATH.read_text(encoding="utf-8"),
            height=760,
            scrolling=False,
        )
