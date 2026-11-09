#!/usr/bin/env python3
"""Build the compact historical overview artifact from the checked-in Olist CSVs.

Every event is filtered with a strict cutoff of 00:00 on the selected date, so
the latest complete calendar day is the day before. The artifact contains daily
aggregates for Overview and Ask, and a compact order-event table for the
historical inspection queue.

Definitions:
  * Open order: purchased before the cutoff and with no customer delivery
    event before the cutoff. Final order_status is not used.
  * Overdue: open with a recorded promise date before the selected date.
  * Due soon: open with a recorded promise date from the selected date through
    three calendar days later.
  * Handed over: a carrier handover event is before the cutoff.
  * Deliveries completed: orders with a customer delivery timestamp in the
    window. Final order_status is not used.
  * On-time: the customer delivery calendar date is on or before
    order_estimated_delivery_date. Late is the complement.
  * Late days: calendar days from the promise date to the customer delivery
    date, over late deliveries only.
  * Review score: visible only when both the review creation date and the review
    answer date are strictly before the selected cutoff. A delivery's review is
    counted under the delivery date, not the review date.
  * Overdue, not yet delivered: purchased before the cutoff, promised in the
    last 30 complete days, and with no customer delivery event before the
    cutoff. Cancellations are undated, so they stay open. The denominator
    counts every order with a promise in the window.
  * Item price: sum of item price, freight excluded. It is not revenue.

No network calls and no fabricated values.
"""

from __future__ import annotations

import csv
import json
from collections import defaultdict
from datetime import date, datetime, timedelta
from decimal import Decimal
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SOURCE_DIR = ROOT / "data" / "raw"
OUT_DIR = Path(__file__).resolve().parent / "data"

ORDERS = SOURCE_DIR / "olist_orders_dataset.csv"
ITEMS = SOURCE_DIR / "olist_order_items_dataset.csv"
CUSTOMERS = SOURCE_DIR / "olist_customers_dataset.csv"
REVIEWS = SOURCE_DIR / "olist_order_reviews_dataset.csv"

AS_OF_MIN = date(2018, 5, 1)
AS_OF_MAX = date(2018, 10, 18)
DEFAULT_AS_OF = date(2018, 8, 22)
LAST_FULL_DAY = AS_OF_MAX - timedelta(days=1)

# A rate or average is only shown when its base is at least this many orders.
# The same value is written to the artifact metadata so the Overview UI applies
# exactly the same rule without a second copy of the number.
MIN_SAMPLE = 30

QUEUE_DELIVERY_BUFFER = AS_OF_MIN - timedelta(days=1)

STATE_NAMES = {
    "AC": "Acre",
    "AL": "Alagoas",
    "AM": "Amazonas",
    "AP": "Amapá",
    "BA": "Bahia",
    "CE": "Ceará",
    "DF": "Distrito Federal",
    "ES": "Espírito Santo",
    "GO": "Goiás",
    "MA": "Maranhão",
    "MG": "Minas Gerais",
    "MS": "Mato Grosso do Sul",
    "MT": "Mato Grosso",
    "PA": "Pará",
    "PB": "Paraíba",
    "PE": "Pernambuco",
    "PI": "Piauí",
    "PR": "Paraná",
    "RJ": "Rio de Janeiro",
    "RN": "Rio Grande do Norte",
    "RO": "Rondônia",
    "RR": "Roraima",
    "RS": "Rio Grande do Sul",
    "SC": "Santa Catarina",
    "SE": "Sergipe",
    "SP": "São Paulo",
    "TO": "Tocantins",
}

QUEUE_FIELDS = [
    "order_id",
    "customer_state",
    "purchase_day",
    "handover_day",
    "delivered_day",
    "promise_day",
    "item_price_cents",
]


def parse_day(value: str) -> date | None:
    value = (value or "").strip()
    if not value:
        return None
    return datetime.strptime(value[:10], "%Y-%m-%d").date()


def price_cents(value: str) -> int:
    return int((Decimal(value) * 100).to_integral_value())


def day_range(start: date, end: date):
    current = start
    while current <= end:
        yield current
        current += timedelta(days=1)


def month_start(value: date) -> date:
    return value.replace(day=1)


def derive_series_starts(coverage):
    """Start each series at the first month present in the source data.

    No calendar date is hard-coded: the purchase series begins with the
    earliest purchase month, and the delivery and state series begin with the
    earliest delivery month. That way every window the date picker allows is
    backed by real data rather than leading zero months.
    """
    purchase_first = coverage["order_purchase_timestamp"][0]
    delivery_first = coverage["order_delivered_customer_date"][0]
    if purchase_first is None or delivery_first is None:
        raise ValueError("Source data has no purchase or delivery timestamps")
    return {
        "purchases": month_start(purchase_first),
        "deliveries": month_start(delivery_first),
        "states": month_start(delivery_first),
    }


def read_orders():
    orders = {}
    coverage = {}
    for column in (
        "order_purchase_timestamp",
        "order_delivered_customer_date",
        "order_estimated_delivery_date",
    ):
        coverage[column] = [None, None]

    with ORDERS.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            order_id = row["order_id"].strip()
            purchase_day = parse_day(row["order_purchase_timestamp"])
            handover_day = parse_day(row["order_delivered_carrier_date"])
            delivered_day = parse_day(row["order_delivered_customer_date"])
            promised_day = parse_day(row["order_estimated_delivery_date"])
            orders[order_id] = {
                "customer_id": row["customer_id"].strip(),
                "status": row["order_status"].strip(),
                "purchase_day": purchase_day,
                "handover_day": handover_day,
                "delivered_day": delivered_day,
                "promised_day": promised_day,
            }
            for column, value in (
                ("order_purchase_timestamp", purchase_day),
                ("order_delivered_customer_date", delivered_day),
                ("order_estimated_delivery_date", promised_day),
            ):
                if value is None:
                    continue
                low, high = coverage[column]
                coverage[column] = [
                    value if low is None or value < low else low,
                    value if high is None or value > high else high,
                ]
    return orders, coverage


def read_customers():
    states = {}
    with CUSTOMERS.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            states[row["customer_id"].strip()] = row["customer_state"].strip()
    return states


def read_item_totals():
    totals = defaultdict(int)
    with ITEMS.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            totals[row["order_id"].strip()] += price_cents(row["price"])
    return totals


def read_reviews():
    """Map each order to its review scores and the date each review is knowable.

    A review is knowable only after both its creation date and its answer date
    have passed, matching the Ask analytics views' exclusive cutoff rule.
    """
    reviews = defaultdict(list)
    with REVIEWS.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            score = (row["review_score"] or "").strip()
            creation_day = parse_day(row["review_creation_date"])
            answer_day = parse_day(row["review_answer_timestamp"])
            if not score or creation_day is None or answer_day is None:
                continue
            visible_day = max(creation_day, answer_day)
            reviews[row["order_id"].strip()].append((visible_day, int(score)))
    return reviews


def aggregate(orders, customer_states, item_totals, reviews, starts):
    purchase_start = starts["purchases"]
    delivery_start = starts["deliveries"]
    state_start = starts["states"]
    purchase_orders = defaultdict(int)
    sales_cents = defaultdict(int)
    delivery_total = defaultdict(int)
    delivery_on_time = defaultdict(int)
    delivery_late_days = defaultdict(int)
    review_rollup = defaultdict(lambda: [0, 0])
    state_total = defaultdict(lambda: defaultdict(int))
    state_late = defaultdict(lambda: defaultdict(int))
    state_purchase_orders = defaultdict(lambda: defaultdict(int))
    state_purchase_cents = defaultdict(lambda: defaultdict(int))

    for order_id, order in orders.items():
        purchase_day = order["purchase_day"]
        if purchase_day is not None and purchase_start <= purchase_day <= LAST_FULL_DAY:
            purchase_orders[purchase_day] += 1
            cents = item_totals.get(order_id, 0)
            sales_cents[purchase_day] += cents
            purchase_state = customer_states.get(order["customer_id"])
            if purchase_state:
                state_purchase_orders[purchase_state][purchase_day] += 1
                state_purchase_cents[purchase_state][purchase_day] += cents

        delivered_day = order["delivered_day"]
        if delivered_day is None:
            continue
        if not (delivery_start <= delivered_day <= LAST_FULL_DAY):
            continue

        promised_day = order["promised_day"]
        is_late = promised_day is not None and delivered_day > promised_day
        delivery_total[delivered_day] += 1
        if not is_late:
            delivery_on_time[delivered_day] += 1
        elif promised_day is not None:
            delivery_late_days[delivered_day] += (delivered_day - promised_day).days

        for visible_day, score in reviews.get(order_id, ()):
            rollup = review_rollup[(delivered_day, visible_day, 1 if is_late else 0)]
            rollup[0] += 1
            rollup[1] += score

        if delivered_day >= state_start:
            state = customer_states.get(order["customer_id"])
            if state:
                state_total[state][delivered_day] += 1
                if is_late:
                    state_late[state][delivered_day] += 1

    return (
        purchase_orders,
        sales_cents,
        delivery_total,
        delivery_on_time,
        delivery_late_days,
        review_rollup,
        state_total,
        state_late,
        state_purchase_orders,
        state_purchase_cents,
    )


def build_order_rows(orders, customer_states, item_totals):
    rows = []
    for order_id in sorted(orders):
        order = orders[order_id]
        purchase_day = order["purchase_day"]
        if purchase_day is None or purchase_day >= AS_OF_MAX:
            continue
        delivered_day = order["delivered_day"]
        if delivered_day is not None and delivered_day < QUEUE_DELIVERY_BUFFER:
            continue
        rows.append(
            [
                order_id,
                customer_states.get(order["customer_id"], ""),
                purchase_day.isoformat(),
                order["handover_day"].isoformat() if order["handover_day"] else None,
                delivered_day.isoformat() if delivered_day else None,
                order["promised_day"].isoformat() if order["promised_day"] else None,
                item_totals.get(order_id, 0),
            ]
        )
    return rows


def build_overdue_not_delivered(orders):
    """Per as-of day: opened promises that are still undelivered and the due base.

    Each row is [as_of_day, count, previous_count, due_total, carrier_count].
    A promise falls in the window when it is on or before the prior day and
    within the last 30 days ending there. The count keeps only orders purchased
    before the cutoff with no customer delivery event before it. The carrier
    count keeps those with a carrier handover event before the cutoff; the rest
    are not yet shipped. The previous count is this same measure as of 30 days
    earlier: cutoff 30 days before, promise window the 30 days before that.
    """
    promise_days = [
        order["promised_day"] for order in orders.values() if order["promised_day"] is not None
    ]
    if not promise_days:
        return []
    start = min(promise_days) + timedelta(days=1)
    end = AS_OF_MAX
    if start > end:
        return []
    size = (end - start).days + 1
    due_diff = [0] * (size + 1)
    count_diff = [0] * (size + 1)
    carrier_diff = [0] * (size + 1)

    def add_range(diff, low, high):
        if low > high or high < start or low > end:
            return
        low = max(low, start)
        high = min(high, end)
        diff[(low - start).days] += 1
        diff[(high - start).days + 1] -= 1

    for order in orders.values():
        promised = order["promised_day"]
        if promised is None:
            continue
        add_range(due_diff, promised + timedelta(days=1), promised + timedelta(days=30))

        purchase = order["purchase_day"]
        if purchase is None:
            continue
        delivered = order["delivered_day"]

        current_low = max(promised + timedelta(days=1), purchase + timedelta(days=1))
        current_high = promised + timedelta(days=30)
        if delivered is not None:
            current_high = min(current_high, delivered)
        add_range(count_diff, current_low, current_high)

        handover = order["handover_day"]
        if handover is not None:
            add_range(carrier_diff, max(current_low, handover + timedelta(days=1)), current_high)

    rows = []
    counts = []
    due_total = 0
    count = 0
    carrier = 0
    for index in range(size):
        due_total += due_diff[index]
        count += count_diff[index]
        carrier += carrier_diff[index]
        counts.append(count)
        previous = counts[index - 30] if index >= 30 else 0
        rows.append(
            [
                (start + timedelta(days=index)).isoformat(),
                count,
                previous,
                due_total,
                carrier,
            ]
        )
    return rows


def build_payload(
    coverage,
    starts,
    purchase_orders,
    sales_cents,
    delivery_total,
    delivery_on_time,
    delivery_late_days,
    review_rollup,
    state_total,
    state_late,
    state_purchase_orders,
    state_purchase_cents,
    order_rows,
    overdue_not_delivered,
):
    purchase_daily = []
    for day in day_range(starts["purchases"], LAST_FULL_DAY):
        purchase_daily.append(
            [day.isoformat(), purchase_orders.get(day, 0), sales_cents.get(day, 0)]
        )

    delivery_daily = []
    for day in day_range(starts["deliveries"], LAST_FULL_DAY):
        total = delivery_total.get(day, 0)
        if total == 0:
            continue
        delivery_daily.append(
            [
                day.isoformat(),
                total,
                delivery_on_time.get(day, 0),
                delivery_late_days.get(day, 0),
            ]
        )

    review_daily = []
    for (delivered_day, visible_day, is_late) in sorted(review_rollup):
        count, score_sum = review_rollup[(delivered_day, visible_day, is_late)]
        review_daily.append(
            [delivered_day.isoformat(), visible_day.isoformat(), is_late, count, score_sum]
        )

    state_purchases = {}
    for state in sorted(state_purchase_orders):
        rows = []
        for day in day_range(starts["purchases"], LAST_FULL_DAY):
            total = state_purchase_orders[state].get(day, 0)
            if total == 0:
                continue
            rows.append([day.isoformat(), total, state_purchase_cents[state].get(day, 0)])
        if rows:
            state_purchases[state] = rows

    state_daily = {}
    for state in sorted(state_total):
        rows = []
        for day in day_range(starts["states"], LAST_FULL_DAY):
            total = state_total[state].get(day, 0)
            if total == 0:
                continue
            rows.append([day.isoformat(), total, state_late[state].get(day, 0)])
        if rows:
            state_daily[state] = rows

    return {
        "schema_version": 4,
        "meta": {
            "replay_only": True,
            "as_of_min": AS_OF_MIN.isoformat(),
            "as_of_max": AS_OF_MAX.isoformat(),
            "default_as_of": DEFAULT_AS_OF.isoformat(),
            "min_sample": MIN_SAMPLE,
            "cutoff": {
                "exclusive": True,
                "meaning": "An event is visible only when its date is strictly before the selected replay date.",
                "latest_full_day": "Selected replay date minus one calendar day.",
                "clock": "The selected date is used only as an exclusive upper bound.",
                "label": "Prior end of day",
            },
            "as_of_max_reason": "18 Oct 2018 is the day after the last order event in the source (17 Oct 2018).",
            "series_start": {
                "purchases": starts["purchases"].isoformat(),
                "deliveries": starts["deliveries"].isoformat(),
                "states": starts["states"].isoformat(),
                "last_full_day": LAST_FULL_DAY.isoformat(),
            },
            "source_coverage": {
                "purchase_min": coverage["order_purchase_timestamp"][0].isoformat(),
                "purchase_max": coverage["order_purchase_timestamp"][1].isoformat(),
                "delivery_min": coverage["order_delivered_customer_date"][0].isoformat(),
                "delivery_max": coverage["order_delivered_customer_date"][1].isoformat(),
                "promise_min": coverage["order_estimated_delivery_date"][0].isoformat(),
                "promise_max": coverage["order_estimated_delivery_date"][1].isoformat(),
            },
            "definitions": {
                "orders_placed": "Distinct orders whose purchase date is in the window. No final-status filter.",
                "product_sales": "Sum of item price in R$ for items whose order purchase date is in the window. Freight is excluded and this is not revenue.",
                "deliveries_completed": "Orders with a customer delivery timestamp in the window. Final order_status is not used.",
                "on_time_share": "Share of those deliveries whose customer delivery calendar date is on or before order_estimated_delivery_date.",
                "late_share": "Share of those deliveries whose customer delivery calendar date is after order_estimated_delivery_date. Late share is 100% minus on-time share.",
                "avg_days_late": "Mean calendar days from order_estimated_delivery_date to the customer delivery date, over late deliveries only.",
                "review_on_time_vs_late": "Mean review score across visible reviews attached to on-time versus late deliveries in the window. A review is visible only when both review_creation_date and review_answer_timestamp are strictly before the replay cutoff.",
                "hotspots": "Buyer states ranked by late deliveries among customer delivery events in the 90 calendar days ending the prior day. The rate uses all deliveries to that state in the window.",
                "promise": "order_estimated_delivery_date is the recorded promise and is compared at calendar-day granularity.",
                "open_order": "Purchased before the cutoff and with no customer delivery event before the cutoff.",
                "overdue": "Open with a recorded promise date strictly before the selected replay date.",
                "overdue_not_delivered": "Orders purchased before the cutoff with a promised date in the last 30 complete days and no customer delivery event before the cutoff. Cancellations are undated, so they stay open. The split counts orders with a carrier handover event before the cutoff as with carrier and the rest as not yet shipped. The denominator counts every order with a promised date in the window.",
                "due_soon": "Open with a recorded promise date from the selected date through three calendar days later.",
                "handover_age": "Calendar days from the carrier handover date to the selected date.",
                "queue_order": "Overdue before due soon. Within each group, handed-over orders first, then oldest promise first, then order id.",
            },
            "queue": {
                "fields": QUEUE_FIELDS,
                "status_at_cutoff": "An order is either handed over with no delivery recorded yet, or has no handover recorded by cutoff. Cancellation history is not in the source.",
                "final_status_used": False,
            },
            "limitations": [
                "Cancellation history is unavailable. An open order with no handover event may already have been cancelled or may be a stale source record.",
                "Some records marked delivered in the source have no customer delivery timestamp. They stay open in the as-of view because only events are used.",
                "Later delivery events, reviews and final status are never used to rank an order at an earlier cutoff.",
                "Item price excludes freight and is not revenue.",
                "No customer names, addresses or other personal details are included.",
            ],
            "state_names": STATE_NAMES,
            "note": "Historical replay of the 2016-2018 Olist dataset. Not live or production data.",
        },
        "daily_purchases": purchase_daily,
        "daily_deliveries": delivery_daily,
        "daily_reviews": review_daily,
        "daily_overdue_not_delivered": overdue_not_delivered,
        "state_deliveries": state_daily,
        "state_purchases": state_purchases,
        "orders": order_rows,
    }


def summarize_as_of(payload, as_of: date):
    previous_day = as_of - timedelta(days=1)
    context_start = previous_day - timedelta(days=29)
    hotspot_start = previous_day - timedelta(days=89)

    overdue = 0
    due_soon = 0
    open_orders = 0
    overdue_handover = 0
    overdue_without_handover = 0
    overdue_by_state = defaultdict(int)
    soon_handover = 0
    due_end = as_of + timedelta(days=3)

    for row in payload["orders"]:
        _, state, purchase_day, handover_day, delivered_day, promise_day, _ = row
        if purchase_day >= as_of.isoformat():
            continue
        if delivered_day is not None and delivered_day < as_of.isoformat():
            continue
        open_orders += 1
        handed_over = handover_day is not None and handover_day < as_of.isoformat()
        if promise_day is not None and promise_day < as_of.isoformat():
            overdue += 1
            overdue_by_state[state] += 1
            if handed_over:
                overdue_handover += 1
            else:
                overdue_without_handover += 1
        elif promise_day is not None and as_of.isoformat() <= promise_day <= due_end.isoformat():
            due_soon += 1
            if handed_over:
                soon_handover += 1

    orders_placed = 0
    sales = 0
    for day_text, count, cents in payload["daily_purchases"]:
        if context_start.isoformat() <= day_text <= previous_day.isoformat():
            orders_placed += count
            sales += cents

    delivered = 0
    on_time = 0
    late_days = 0
    for day_text, count, on_time_count, late_day_total in payload["daily_deliveries"]:
        if context_start.isoformat() <= day_text <= previous_day.isoformat():
            delivered += count
            on_time += on_time_count
            late_days += late_day_total

    on_time_reviews = 0
    on_time_review_score = 0
    late_reviews = 0
    late_review_score = 0
    for delivered_text, visible_text, is_late, count, score_sum in payload["daily_reviews"]:
        if not (context_start.isoformat() <= delivered_text <= previous_day.isoformat()):
            continue
        if visible_text >= as_of.isoformat():
            continue
        if is_late:
            late_reviews += count
            late_review_score += score_sum
        else:
            on_time_reviews += count
            on_time_review_score += score_sum

    late = delivered - on_time

    overdue_due = [0, 0, 0]
    for day_text, count, previous_count, due_total, _ in payload["daily_overdue_not_delivered"]:
        if day_text == as_of.isoformat():
            overdue_due = [count, previous_count, due_total]
            break

    states = []
    for state, rows in payload["state_deliveries"].items():
        total = 0
        state_late = 0
        for day_text, count, late_count in rows:
            if hotspot_start.isoformat() <= day_text <= previous_day.isoformat():
                total += count
                state_late += late_count
        if total:
            states.append((state_late, total, state))
    states.sort(reverse=True)

    ranked_states = sorted(overdue_by_state.items(), key=lambda item: (-item[1], item[0]))
    top_state = ranked_states[0] if ranked_states else ("", 0)

    return {
        "as_of": as_of.isoformat(),
        "latest_full_day": previous_day.isoformat(),
        "context_window": [context_start.isoformat(), previous_day.isoformat()],
        "hotspot_window": [hotspot_start.isoformat(), previous_day.isoformat()],
        "open_orders": open_orders,
        "overdue": overdue,
        "overdue_handover": overdue_handover,
        "overdue_without_handover": overdue_without_handover,
        "due_soon": due_soon,
        "due_soon_handover": soon_handover,
        "top_overdue_state": list(top_state),
        "orders_placed": orders_placed,
        "product_sales_brl": round(sales / 100, 2),
        "overdue_not_delivered": overdue_due[0],
        "overdue_not_delivered_previous": overdue_due[1],
        "overdue_not_delivered_due": overdue_due[2],
        "deliveries_completed": delivered,
        "on_time": on_time,
        "on_time_share": round(on_time / delivered, 4) if delivered >= MIN_SAMPLE else None,
        "late": late,
        "late_share": round(late / delivered, 4) if delivered >= MIN_SAMPLE else None,
        "late_days_sum": late_days,
        "avg_days_late": round(late_days / late, 4) if late >= MIN_SAMPLE else None,
        "on_time_reviews": on_time_reviews,
        "on_time_review_avg": (
            round(on_time_review_score / on_time_reviews, 4)
            if on_time_reviews >= MIN_SAMPLE
            else None
        ),
        "late_reviews": late_reviews,
        "late_review_avg": (
            round(late_review_score / late_reviews, 4)
            if late_reviews >= MIN_SAMPLE
            else None
        ),
        "top_states": states[:8],
    }


def main():
    orders, coverage = read_orders()
    customer_states = read_customers()
    item_totals = read_item_totals()
    reviews = read_reviews()
    starts = derive_series_starts(coverage)
    (
        purchase_orders,
        sales_cents,
        delivery_total,
        delivery_on_time,
        delivery_late_days,
        review_rollup,
        state_total,
        state_late,
        state_purchase_orders,
        state_purchase_cents,
    ) = aggregate(orders, customer_states, item_totals, reviews, starts)
    order_rows = build_order_rows(orders, customer_states, item_totals)
    overdue_not_delivered = build_overdue_not_delivered(orders)

    payload = build_payload(
        coverage,
        starts,
        purchase_orders,
        sales_cents,
        delivery_total,
        delivery_on_time,
        delivery_late_days,
        review_rollup,
        state_total,
        state_late,
        state_purchase_orders,
        state_purchase_cents,
        order_rows,
        overdue_not_delivered,
    )

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    json_path = OUT_DIR / "overview.json"
    js_path = OUT_DIR / "overview-data.js"
    compact = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
    json_path.write_text(compact + "\n", encoding="utf-8")
    js_path.write_text(
        "/* Generated by build_overview_data.py from data/raw/*.csv. */\n"
        "window.OVERVIEW_DATA=" + compact + ";\n",
        encoding="utf-8",
    )

    snapshot = summarize_as_of(payload, date.fromisoformat(payload["meta"]["default_as_of"]))
    print("Overview data built")
    print(f"  source rows: orders={len(orders):,} customers={len(customer_states):,}")
    print(
        "  daily rows: purchases={:,} deliveries={:,} reviews={:,} state-deliveries={:,} state-purchases={:,} queue-orders={:,}".format(
            len(payload["daily_purchases"]),
            len(payload["daily_deliveries"]),
            len(payload["daily_reviews"]),
            sum(len(rows) for rows in payload["state_deliveries"].values()),
            sum(len(rows) for rows in payload["state_purchases"].values()),
            len(payload["orders"]),
        )
    )
    print(f"  as-of range: {payload['meta']['as_of_min']} .. {payload['meta']['as_of_max']}")
    print(f"  default as-of: {snapshot['as_of']} (events through {snapshot['latest_full_day']})")
    print(f"  context cohort: {snapshot['context_window'][0]} .. {snapshot['context_window'][1]}")
    print(f"  open orders: {snapshot['open_orders']:,}")
    print(
        "  overdue: {:,} (handover {:,}, no handover {:,})  due soon: {:,}".format(
            snapshot["overdue"],
            snapshot["overdue_handover"],
            snapshot["overdue_without_handover"],
            snapshot["due_soon"],
        )
    )
    print(
        "  top overdue state: {} ({:,})".format(
            snapshot["top_overdue_state"][0], snapshot["top_overdue_state"][1]
        )
    )
    print(
        "  overdue not yet delivered: {:,} (prior {:,}) of {:,} due in window".format(
            snapshot["overdue_not_delivered"],
            snapshot["overdue_not_delivered_previous"],
            snapshot["overdue_not_delivered_due"],
        )
    )
    print(f"  orders placed in context: {snapshot['orders_placed']:,}")
    print(f"  item price in context: R$ {snapshot['product_sales_brl']:,.2f}")
    print(
        "  deliveries completed: {:,}  on time: {:,}  on-time share: {}  late: {:,} ({})  avg days late: {}".format(
            snapshot["deliveries_completed"],
            snapshot["on_time"],
            "n/a" if snapshot["on_time_share"] is None else f"{snapshot['on_time_share']:.1%}",
            snapshot["late"],
            "n/a" if snapshot["late_share"] is None else f"{snapshot['late_share']:.1%}",
            "n/a" if snapshot["avg_days_late"] is None else f"{snapshot['avg_days_late']:.1f}",
        )
    )
    print(
        "  review score (on time vs late): {} vs {}  (visible reviews: {:,} vs {:,})".format(
            "n/a" if snapshot["on_time_review_avg"] is None else f"{snapshot['on_time_review_avg']:.1f}",
            "n/a" if snapshot["late_review_avg"] is None else f"{snapshot['late_review_avg']:.1f}",
            snapshot["on_time_reviews"],
            snapshot["late_reviews"],
        )
    )
    print(
        "  hotspot window: {} .. {}  top states (late / delivered): {}".format(
            snapshot["hotspot_window"][0],
            snapshot["hotspot_window"][1],
            ", ".join(
                f"{state} {late}/{total}" for late, total, state in snapshot["top_states"]
            ),
        )
    )
    print(f"  files: {json_path.relative_to(ROOT)} ({json_path.stat().st_size:,} bytes)")
    print(f"         {js_path.relative_to(ROOT)} ({js_path.stat().st_size:,} bytes)")


if __name__ == "__main__":
    main()
