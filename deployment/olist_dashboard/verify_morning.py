#!/usr/bin/env python3
"""Focused checks for the historical overview artifact, queue, Ask contract and charts."""

from __future__ import annotations

import csv
import hashlib
import json
import re
import shutil
import subprocess
import sys
from collections import defaultdict
from datetime import date, datetime, timedelta
from decimal import Decimal
from pathlib import Path


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
DATA = HERE / "data" / "overview.json"
JS_DATA = HERE / "data" / "overview-data.js"
BUILDER = HERE / "build_overview_data.py"
SOURCE = ROOT / "data" / "raw"
REVIEWS = SOURCE / "olist_order_reviews_dataset.csv"

if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))
import build_overview_data as builder  # noqa: E402

CHECKS = 0


def check(condition, message):
    global CHECKS
    CHECKS += 1
    if not condition:
        raise AssertionError(message)


def parse_day(value):
    value = (value or "").strip()
    if not value:
        return None
    return datetime.strptime(value[:10], "%Y-%m-%d").date()


def price_cents(value):
    return int((Decimal(value) * 100).to_integral_value())


def read_source():
    orders = []
    with (SOURCE / "olist_orders_dataset.csv").open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            orders.append(
                {
                    "order_id": row["order_id"].strip(),
                    "customer_id": row["customer_id"].strip(),
                    "status": row["order_status"].strip(),
                    "purchase": parse_day(row["order_purchase_timestamp"]),
                    "handover": parse_day(row["order_delivered_carrier_date"]),
                    "delivered": parse_day(row["order_delivered_customer_date"]),
                    "promise": parse_day(row["order_estimated_delivery_date"]),
                }
            )
    states = {}
    with (SOURCE / "olist_customers_dataset.csv").open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            states[row["customer_id"].strip()] = row["customer_state"].strip()
    item_totals = defaultdict(int)
    with (SOURCE / "olist_order_items_dataset.csv").open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            item_totals[row["order_id"].strip()] += price_cents(row["price"])
    reviews = defaultdict(list)
    with REVIEWS.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            score = (row["review_score"] or "").strip()
            creation = parse_day(row["review_creation_date"])
            answer = parse_day(row["review_answer_timestamp"])
            if not score or creation is None or answer is None:
                continue
            reviews[row["order_id"].strip()].append((max(creation, answer), int(score)))
    return orders, states, item_totals, reviews


def expected_counts(orders, states, as_of):
    due_end = as_of + timedelta(days=3)
    result = {
        "open": 0,
        "overdue": 0,
        "due_soon": 0,
        "overdue_handover": 0,
        "overdue_no_handover": 0,
        "recent_overdue": 0,
        "recent_overdue_handover": 0,
        "older_overdue": 0,
        "older_overdue_handover": 0,
        "due_soon_handover": 0,
        "overdue_by_state": defaultdict(int),
    }
    for order in orders:
        purchase = order["purchase"]
        delivered = order["delivered"]
        if purchase is None or purchase >= as_of:
            continue
        if delivered is not None and delivered < as_of:
            continue
        result["open"] += 1
        state = states.get(order["customer_id"], "??")
        promise = order["promise"]
        handover = order["handover"]
        handed_over = handover is not None and handover < as_of
        if promise is not None and promise < as_of:
            result["overdue"] += 1
            result["overdue_by_state"][state] += 1
            recent = (as_of - promise).days <= 7
            if recent:
                result["recent_overdue"] += 1
            else:
                result["older_overdue"] += 1
            if handed_over:
                result["overdue_handover"] += 1
                if recent:
                    result["recent_overdue_handover"] += 1
                else:
                    result["older_overdue_handover"] += 1
            else:
                result["overdue_no_handover"] += 1
        elif promise is not None and as_of <= promise <= due_end:
            result["due_soon"] += 1
            if handed_over:
                result["due_soon_handover"] += 1
    return result


def payload_counts(payload, as_of):
    due_end = as_of + timedelta(days=3)
    result = {
        "open": 0,
        "overdue": 0,
        "due_soon": 0,
        "overdue_handover": 0,
        "overdue_no_handover": 0,
        "recent_overdue": 0,
        "recent_overdue_handover": 0,
        "older_overdue": 0,
        "older_overdue_handover": 0,
        "due_soon_handover": 0,
        "overdue_by_state": defaultdict(int),
    }
    for row in payload["orders"]:
        _, state, purchase_text, handover_text, delivered_text, promise_text, _ = row
        purchase = date.fromisoformat(purchase_text)
        delivered = date.fromisoformat(delivered_text) if delivered_text else None
        promise = date.fromisoformat(promise_text) if promise_text else None
        handover = date.fromisoformat(handover_text) if handover_text else None
        if purchase >= as_of:
            continue
        if delivered is not None and delivered < as_of:
            continue
        result["open"] += 1
        handed_over = handover is not None and handover < as_of
        if promise is not None and promise < as_of:
            result["overdue"] += 1
            result["overdue_by_state"][state] += 1
            recent = (as_of - promise).days <= 7
            if recent:
                result["recent_overdue"] += 1
            else:
                result["older_overdue"] += 1
            if handed_over:
                result["overdue_handover"] += 1
                if recent:
                    result["recent_overdue_handover"] += 1
                else:
                    result["older_overdue_handover"] += 1
            else:
                result["overdue_no_handover"] += 1
        elif promise is not None and as_of <= promise <= due_end:
            result["due_soon"] += 1
            if handed_over:
                result["due_soon_handover"] += 1
    return result


def expected_overdue_not_delivered(orders, as_of):
    def measure(cutoff):
        end = cutoff - timedelta(days=1)
        start = end - timedelta(days=29)
        count = 0
        due_total = 0
        carrier = 0
        for order in orders:
            promise = order["promise"]
            if promise is None:
                continue
            purchased = order["purchase"]
            delivered = order["delivered"]
            if not (start <= promise <= end):
                continue
            due_total += 1
            if purchased is not None and purchased < cutoff and (
                delivered is None or delivered >= cutoff
            ):
                count += 1
                handover = order["handover"]
                if handover is not None and handover < cutoff:
                    carrier += 1
        return count, due_total, carrier

    count, due_total, carrier = measure(as_of)
    previous, _, _ = measure(as_of - timedelta(days=30))
    return count, previous, due_total, carrier


def sum_daily(rows, start, end, value_index):
    total = 0
    for row in rows:
        day = date.fromisoformat(row[0])
        if start <= day <= end:
            total += row[value_index]
    return total


def expected_outcomes(orders, reviews, as_of):
    end = as_of - timedelta(days=1)
    start = end - timedelta(days=29)
    result = {
        "delivered": 0,
        "on_time": 0,
        "late": 0,
        "late_days_sum": 0,
        "on_time_reviews": 0,
        "on_time_review_score": 0,
        "late_reviews": 0,
        "late_review_score": 0,
    }
    for order in orders:
        delivered = order["delivered"]
        if delivered is None or not (start <= delivered <= end):
            continue
        promise = order["promise"]
        is_late = promise is not None and delivered > promise
        result["delivered"] += 1
        if is_late:
            result["late"] += 1
            result["late_days_sum"] += (delivered - promise).days
        else:
            result["on_time"] += 1
        for visible_day, score in reviews.get(order["order_id"], ()):
            if visible_day >= as_of:
                continue
            if is_late:
                result["late_reviews"] += 1
                result["late_review_score"] += score
            else:
                result["on_time_reviews"] += 1
                result["on_time_review_score"] += score
    return result


def payload_outcomes(payload, as_of):
    end = as_of - timedelta(days=1)
    start = end - timedelta(days=29)
    result = {
        "delivered": 0,
        "on_time": 0,
        "late": 0,
        "late_days_sum": 0,
        "on_time_reviews": 0,
        "on_time_review_score": 0,
        "late_reviews": 0,
        "late_review_score": 0,
    }
    for row in payload["daily_deliveries"]:
        day = date.fromisoformat(row[0])
        if not (start <= day <= end):
            continue
        result["delivered"] += row[1]
        result["on_time"] += row[2]
        result["late_days_sum"] += row[3]
    result["late"] = result["delivered"] - result["on_time"]
    for row in payload["daily_reviews"]:
        delivered = date.fromisoformat(row[0])
        visible = date.fromisoformat(row[1])
        if not (start <= delivered <= end) or visible >= as_of:
            continue
        if row[2]:
            result["late_reviews"] += row[3]
            result["late_review_score"] += row[4]
        else:
            result["on_time_reviews"] += row[3]
            result["on_time_review_score"] += row[4]
    return result


def check_outcomes(expected, actual, as_of):
    for key in (
        "delivered",
        "on_time",
        "late",
        "late_days_sum",
        "on_time_reviews",
        "on_time_review_score",
        "late_reviews",
        "late_review_score",
    ):
        check(expected[key] == actual[key], f"{as_of} {key}: expected {expected[key]}, artifact {actual[key]}")
    check(
        expected["on_time_reviews"] == actual["on_time_reviews"]
        and expected["on_time_review_score"] == actual["on_time_review_score"],
        f"{as_of} on-time review aggregates must use only reviews visible at the as-of date",
    )
    check(
        expected["late_reviews"] == actual["late_reviews"]
        and expected["late_review_score"] == actual["late_review_score"],
        f"{as_of} late review aggregates must use only reviews visible at the as-of date",
    )
    check(
        actual["late"] + actual["on_time"] == actual["delivered"],
        f"{as_of} late and on-time deliveries must add up to all deliveries",
    )
    if actual["delivered"]:
        late_share = actual["late"] / actual["delivered"]
        on_time_share = actual["on_time"] / actual["delivered"]
        check(
            abs(late_share + on_time_share - 1) < 1e-12,
            f"{as_of} late share and on-time share must total 100%",
        )
    if actual["late"]:
        check(
            actual["late_days_sum"] / actual["late"] > 0,
            f"{as_of} average days late must be positive when late deliveries exist",
        )


def check_deterministic():
    if not SOURCE.is_dir():
        print("  note: raw CSVs are not present; deterministic data rebuild skipped")
        return
    before = {
        "json": hashlib.sha256(DATA.read_bytes()).hexdigest(),
        "js": hashlib.sha256(JS_DATA.read_bytes()).hexdigest(),
    }
    subprocess.run(
        [sys.executable, str(BUILDER)],
        cwd=str(ROOT),
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
    )
    after = {
        "json": hashlib.sha256(DATA.read_bytes()).hexdigest(),
        "js": hashlib.sha256(JS_DATA.read_bytes()).hexdigest(),
    }
    check(before == after, "builder output is not deterministic")


def check_dom_references():
    html = (HERE / "index.html").read_text(encoding="utf-8")
    script = (HERE / "overview.js").read_text(encoding="utf-8")
    ids = set(re.findall(r'\bid="([^"]+)"', html))
    refs = set(re.findall(r'\$\("([^"]+)"\)', script))
    missing = sorted(refs - ids)
    check(not missing, f"overview.js references missing ids: {missing}")


def check_js_syntax():
    node = shutil.which("node")
    if not node:
        print("  note: node not found; JavaScript syntax check skipped")
        return
    for name in ("morning-core.js", "ask-chats.js", "overview.js", "data/overview-data.js"):
        subprocess.run(
            [node, "--check", str(HERE / name)],
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
        )
        check(True, f"{name} syntax")


def check_queue_count_scope():
    actions = (HERE / "actions.js").read_text(encoding="utf-8")
    overview = (HERE / "overview.js").read_text(encoding="utf-8")
    check(
        '"/api/actions/queue?date="' in actions
        and '"/api/actions/promises?date="' in actions,
        "the Actions queue and promise lists must load for the selected date",
    )
    check(
        "cache[date] = entry;" in actions,
        "the Actions queue must cache each date's response",
    )
    check(
        "window.PrazoActions.setDate(state.asOf);" in overview,
        "changing the Overview date must update Actions",
    )


def check_redesign_contract():
    html = (HERE / "index.html").read_text(encoding="utf-8")
    script = (HERE / "overview.js").read_text(encoding="utf-8")
    for label in ("Overview", "Actions", "Models", "Ask"):
        check(
            re.search(r'<button type="button" data-view="[^"]+"[^>]*>' + label + "</button>", html) is not None,
            f"top navigation must include {label}",
        )
    for view in ("view-overview", "view-actions", "view-drift", "view-ask"):
        check(f'id="{view}"' in html, f"missing separate view {view}")
    check(
        'id="view-ask" aria-labelledby="ask-title" hidden' in html,
        "Ask must be its own view, not a panel below Overview",
    )
    check("<strong>Olist</strong>" in html and "Vamos" not in html, "visible brand must be Olist Operations")
    check(
        'id="ask-form"' in html and 'id="ask-input"' in html and 'id="ask-submit"' in html,
        "Ask must accept a question",
    )
    check(
        'id="ask-thread"' in html and 'id="ask-status"' in html,
        "Ask must have answer and status containers",
    )
    check(
        'id="ask-chat-list"' in html and 'id="ask-new-chat"' in html
        and 'id="ask-storage-note"' in html,
        "Ask must have a responsive saved-chat sidebar",
    )
    check(
        'id="ask-examples"' not in html and "data-ask-example" not in html
        and "data-ask-example" not in script,
        "clickable example chips must be removed",
    )
    check(
        'id="ask-fold-template"' in html and "@keyframes avatar-think" in html,
        "Ask must include the accepted animated Fold avatar",
    )
    check(
        'src="ask-chats.js"' in html and "olistOps.askChats.v1" in (HERE / "ask-chats.js").read_text(encoding="utf-8"),
        "Ask chat history must use the local bounded store",
    )
    check("ask-provenance" not in html and "ask-provenance" not in script,
        "Ask must not repeat date badges beside every message")
    check(
        "Ask is not available yet" not in html and "Coming tomorrow" not in html,
        "the unavailable Ask placeholder must be gone",
    )
    check(
        "askForm.addEventListener" in script and 'fetch("/api/ask"' in script,
        "Ask form must call the local API",
    )
    check(
        "handleAsk(" not in script
        and "Core.interpretAsk" not in script
        and "Core.interpret(" not in script,
        "legacy Ask and queue parsers must not be reachable from the UI",
    )
    check("Data as of" in html, "header must use Data as of wording")
    date_input = re.search(r'<input\b[^>]*\bid="asof-date"[^>]*>', html)
    check(date_input is not None, "header must include an as-of date picker")
    check('type="date"' in date_input.group(0), "as-of control must be an input type=date")
    check(
        re.search(r'<label[^>]*for="asof-date"[^>]*>\s*Data as of', html) is not None,
        "date picker must keep the visible Data as of label",
    )
    check(
        'dateInput.min = META.as_of_min;' in script
        and 'dateInput.max = META.as_of_max;' in script
        and 'dateInput.value = state.asOf;' in script,
        "date picker must use the artifact date range and stay synchronized",
    )
    check(
        'dateInput.addEventListener("change"' in script
        and 'setDate(dateInput.value);' in script,
        "date picker changes must call setDate",
    )
    check("Morning replay" not in html and "morning replay" not in html, "morning replay wording must be gone")
    overview_html = html.split('id="view-overview"', 1)[1].split('id="view-actions"', 1)[0]
    check("09:00" not in overview_html and "09:00" not in script,
          "09:00 clock wording must be gone from the Overview (Actions keeps its 09:00 review)")
    check('id="metrics"' in html, "Overview must have key metrics")
    check('id="metrics-secondary"' in html, "Overview must keep demand context below the headline cards")
    check('id="delivery-chart"' in html and 'id="price-chart"' in html, "Overview must have two main charts")
    check('id="hotspot-list"' in html, "Overview must have compact hotspots")
    check('id="actions-root"' in html and 'id="view-actions"' in html, "the worklist must live in Actions")
    check("localStorage" in script and "olistOps.pins.v2" in script, "pins must persist in localStorage")
    engine = (HERE / "ask_service.py").read_text(encoding="utf-8")
    sql_engine = (HERE / "analytics_sql.py").read_text(encoding="utf-8")
    db_builder = (HERE / "analytics_db.py").read_text(encoding="utf-8")
    server = (HERE / "server.py").read_text(encoding="utf-8")
    evidence_script = (HERE / "ask-evidence.js").read_text(encoding="utf-8")
    check((HERE / "server.py").is_file() and (HERE / "ask_service.py").is_file(),
          "local Ask server and service must exist")
    check((HERE / "analytics_sql.py").is_file() and (HERE / "analytics_db.py").is_file(),
          "local analytics SQL guard and CSV builder must exist")
    check("PRAGMA query_only=ON" in sql_engine and "set_authorizer" in sql_engine,
          "Ask SQL must be read-only and deny-by-default")
    check(
        ":as_of" in sql_engine
        and "CREATE TEMP VIEW" in sql_engine
        and "BASE_PREFIX" in sql_engine
        and "identifier-shaped" in sql_engine,
        "Ask SQL must bind :as_of and enforce the cutoff with temp views over private base tables",
    )
    check("analytics.sqlite" in engine and "deployment/olist_dashboard/data/analytics.sqlite" in engine,
          "Ask must name the local analytics database")
    check("textContent = raw" in script, "Ask question input must be inserted as text, not HTML")
    check("item price is not revenue" in sql_engine.lower()
          or "is not revenue" in engine.lower(),
          "item price must never be presented as revenue")
    check("model.complete(" in engine, "Ask must use the configured model client")
    check("_NoRedirectHandler" in engine and "build_opener" in engine,
          "provider redirects must be disabled so the key cannot be forwarded")
    check(
        "TRUSTED_HOSTS" in server
        and "is_trusted_host" in server
        and "is_trusted_origin" in server
        and "is_json_content_type" in server,
        "Ask must validate Host, Origin and JSON content type",
    )
    check("Access-Control-Allow-Origin" not in server, "Ask must not add permissive CORS headers")
    check("VAMOS_LLM_API_KEY" in engine and "VAMOS_LLM_API_KEY" in server,
          "provider configuration must use VAMOS_LLM_* keys")
    check("Authorization" in engine and "Bearer " in engine,
          "the provider client must keep the key server-side")
    check("drawAskChart" in script and "Executed SQL" in evidence_script,
          "Ask UI must render arbitrary charts and expose the SQL")
    check('src="ask-evidence.js"' in html
          and "answer.sql" in evidence_script
          and 'className = "ask-evidence"' in evidence_script,
          "Ask UI must keep SQL and a concise Evidence record")
    check("plainSource" in evidence_script and "plainMeasure" in evidence_script
          and "Data available" in evidence_script and "Rows" in evidence_script,
          "Ask Evidence must show data availability, rows, source and measure in plain language")
    check("data date" in html.lower()
          and "SQL that ran" in html
          and "shows the period" not in html.lower(),
          "Ask help must describe the evidence: the data date and the SQL that ran")
    check("innerHTML" not in evidence_script,
          "Ask Evidence must be built with DOM nodes, not unsafe innerHTML")
    chart_start = script.index("function drawAskChart(")
    chart_end = script.index("function renderAskAnswer(", chart_start)
    chart_body = script[chart_start:chart_end]
    check("createElementNS" in script and "innerHTML" not in chart_body,
          "Ask provider content must be built with DOM nodes, not unsafe innerHTML")
    check("read-only" in html.lower(), "Ask UI must describe the read-only query path")
    check("Local intent model" not in html, "Ask must not imply a local-only model")
    running = (HERE / "README.md").read_text(encoding="utf-8")
    check("analytics.sqlite" in running and "analytics_db.py" in running,
          "running docs must explain the local analytics database")


def main():
    payload = json.loads(DATA.read_text(encoding="utf-8"))
    check(payload["schema_version"] == 4, "expected schema version 4")
    check("state_purchases" in payload, "artifact must include state purchase aggregates")
    check("daily_reviews" in payload, "artifact must include review aggregates")
    check(
        all(len(row) == 5 for row in payload["daily_reviews"]),
        "review aggregate rows must be [delivery day, visible day, late flag, count, score sum]",
    )
    check(payload["meta"]["queue"]["final_status_used"] is False, "final status must not be used")
    check(len(payload["meta"]["queue"]["fields"]) == 7, "queue record shape changed")
    check(payload["meta"]["min_sample"] == 30, "artifact must carry the shared minimum sample")
    last_full_day = payload["meta"]["series_start"]["last_full_day"]
    check(
        payload["daily_purchases"][-1][0] == last_full_day,
        "purchase series must end at the latest complete day",
    )
    check(
        payload["daily_deliveries"][-1][0] == last_full_day,
        "delivery series must end at the latest complete day",
    )

    if not SOURCE.is_dir():
        check_dom_references()
        check_deterministic()
        check_js_syntax()
        check_queue_count_scope()
        check_redesign_contract()
        print(
            f"PASS artifact checks ({CHECKS} assertions); "
            f"SKIPPED source rebuild checks because {SOURCE} is unavailable"
        )
        return

    orders, states, item_totals, reviews = read_source()
    purchase_min = min(order["purchase"] for order in orders if order["purchase"] is not None)
    delivery_min = min(order["delivered"] for order in orders if order["delivered"] is not None)
    check(
        payload["meta"]["series_start"]["purchases"] == purchase_min.replace(day=1).isoformat(),
        "purchase series must start at the earliest source month",
    )
    check(
        payload["meta"]["series_start"]["deliveries"] == delivery_min.replace(day=1).isoformat(),
        "delivery series must start at the earliest source month",
    )
    check(
        payload["daily_purchases"][0][0] == payload["meta"]["series_start"]["purchases"],
        "purchase rows must begin at the derived series start",
    )
    check(
        payload["daily_deliveries"][0][0][:7] == payload["meta"]["series_start"]["deliveries"][:7],
        "delivery rows must begin at the derived series start",
    )

    for window_start, window_end, expected_delivered, expected_late in (
        (date(2017, 9, 1), date(2018, 8, 21), 72777, 5702),
        (date(2017, 5, 1), date(2018, 4, 30), 62187, 4806),
    ):
        raw_delivered = sum(
            1
            for order in orders
            if order["delivered"] is not None and window_start <= order["delivered"] <= window_end
        )
        raw_late = sum(
            1
            for order in orders
            if order["delivered"] is not None
            and window_start <= order["delivered"] <= window_end
            and order["promise"] is not None
            and order["delivered"] > order["promise"]
        )
        actual_delivered = sum_daily(payload["daily_deliveries"], window_start, window_end, 1)
        actual_on_time = sum_daily(payload["daily_deliveries"], window_start, window_end, 2)
        actual_late = actual_delivered - actual_on_time
        check(
            actual_delivered == raw_delivered == expected_delivered,
            f"{window_start}..{window_end} deliveries: raw {raw_delivered}, "
            f"artifact {actual_delivered}, expected {expected_delivered}",
        )
        check(
            actual_late == raw_late == expected_late,
            f"{window_start}..{window_end} late deliveries: raw {raw_late}, "
            f"artifact {actual_late}, expected {expected_late}",
        )

    small = builder.summarize_as_of(payload, date(2018, 10, 18))
    check(
        small["deliveries_completed"] > 0,
        "18 Oct 2018 still has completed deliveries in its 30-day window",
    )
    check(small["late_share"] is not None, "18 Oct 2018 late share must be computed when deliveries > 0")
    check(small["on_time_share"] is not None, "18 Oct 2018 on-time share must be computed when deliveries > 0")
    check(small["avg_days_late"] is not None, "18 Oct 2018 average days late must be computed when late > 0")
    check(small["on_time_review_avg"] is None, "18 Oct 2018 on-time review average is None when on_time_reviews == 0")
    check(small["late_review_avg"] is not None, "18 Oct 2018 late review average must be computed when late_reviews > 0")
    normal = builder.summarize_as_of(payload, date(2018, 8, 22))
    check(normal["late_share"] is not None, "22 Aug 2018 late share must still show")
    check(normal["avg_days_late"] is not None, "22 Aug 2018 average days late must still show")
    overdue_series = {
        row[0]: row for row in payload["daily_overdue_not_delivered"]
    }
    check(
        all(len(row) == 5 for row in overdue_series.values()),
        "overdue aggregate rows must be [as-of day, count, previous count, due total, carrier count]",
    )
    for as_of in (date(2018, 8, 22), date(2018, 6, 1), date(2017, 12, 1)):
        expected = expected_overdue_not_delivered(orders, as_of)
        actual = overdue_series.get(as_of.isoformat())
        check(actual is not None, f"{as_of} overdue aggregate must exist")
        check(
            tuple(actual[1:]) == expected,
            f"{as_of} overdue aggregate: expected {expected}, artifact {tuple(actual[1:])}",
        )
    pinned = overdue_series["2018-08-22"]
    check(pinned[4] == 167, f"2018-08-22 with carrier: expected 167, artifact {pinned[4]}")
    check(
        pinned[1] - pinned[4] == 134,
        f"2018-08-22 not yet shipped: expected 134, artifact {pinned[1] - pinned[4]}",
    )
    dates = [date(2018, 5, 1), date(2018, 6, 15), date(2018, 8, 22)]

    for as_of in dates:
        expected = expected_counts(orders, states, as_of)
        actual = payload_counts(payload, as_of)
        for key in (
            "open",
            "overdue",
            "due_soon",
            "overdue_handover",
            "overdue_no_handover",
            "recent_overdue",
            "recent_overdue_handover",
            "older_overdue",
            "older_overdue_handover",
            "due_soon_handover",
        ):
            check(
                expected[key] == actual[key],
                f"{as_of} {key}: expected {expected[key]}, artifact {actual[key]}",
            )
        top_expected = sorted(expected["overdue_by_state"].items(), key=lambda item: (-item[1], item[0]))
        top_actual = sorted(actual["overdue_by_state"].items(), key=lambda item: (-item[1], item[0]))
        check(top_expected[:5] == top_actual[:5], f"{as_of} top overdue states differ")

    for as_of in dates:
        previous_day = as_of - timedelta(days=1)
        current = payload_counts(payload, as_of)
        previous = payload_counts(payload, previous_day)
        purchases = 0
        for row in payload["daily_purchases"]:
            if row[0] == previous_day.isoformat():
                purchases = row[1]
        deliveries = 0
        for row in payload["daily_deliveries"]:
            if row[0] == previous_day.isoformat():
                deliveries = row[1]
        check(
            current["open"] == previous["open"] + purchases - deliveries,
            f"{as_of} open orders do not reconcile with the new day's purchases and deliveries",
        )

    for as_of in dates:
        end = as_of - timedelta(days=1)
        start = end - timedelta(days=29)
        expected_orders = sum(
            1
            for order in orders
            if order["purchase"] is not None and start <= order["purchase"] <= end
        )
        actual_orders = sum_daily(payload["daily_purchases"], start, end, 1)
        check(actual_orders == expected_orders, f"{as_of} orders placed aggregate differs")
        expected_cents = sum(
            item_totals.get(order["order_id"], 0)
            for order in orders
            if order["purchase"] is not None and start <= order["purchase"] <= end
        )
        actual_cents = sum_daily(payload["daily_purchases"], start, end, 2)
        check(actual_cents == expected_cents, f"{as_of} item price aggregate differs")
        expected_deliveries = sum(
            1
            for order in orders
            if order["delivered"] is not None and start <= order["delivered"] <= end
        )
        actual_deliveries = sum_daily(payload["daily_deliveries"], start, end, 1)
        check(actual_deliveries == expected_deliveries, f"{as_of} deliveries aggregate differs")
        expected_on_time = sum(
            1
            for order in orders
            if order["delivered"] is not None
            and start <= order["delivered"] <= end
            and order["promise"] is not None
            and order["delivered"] <= order["promise"]
        )
        actual_on_time = sum_daily(payload["daily_deliveries"], start, end, 2)
        check(actual_on_time == expected_on_time, f"{as_of} on-time aggregate differs")

    for as_of in dates:
        end = as_of - timedelta(days=1)
        start = end - timedelta(days=29)
        future_reviews = sum(
            row[3]
            for row in payload["daily_reviews"]
            if start <= date.fromisoformat(row[0]) <= end
            and date.fromisoformat(row[1]) >= as_of
        )
        check(
            future_reviews > 0,
            f"{as_of} review check must include reviews that are not visible yet",
        )
        check_outcomes(
            expected_outcomes(orders, reviews, as_of),
            payload_outcomes(payload, as_of),
            as_of,
        )

    state_sample_dates = [
        (date(2018, 8, 22), 30, "SP"),
        (date(2018, 8, 22), 14, "RJ"),
        (date(2018, 6, 15), 60, "MG"),
    ]
    for as_of, days, state in state_sample_dates:
        end = as_of - timedelta(days=1)
        start = end - timedelta(days=days - 1)
        expected_state_orders = sum(
            1
            for order in orders
            if states.get(order["customer_id"]) == state
            and order["purchase"] is not None
            and start <= order["purchase"] <= end
        )
        expected_state_cents = sum(
            item_totals.get(order["order_id"], 0)
            for order in orders
            if states.get(order["customer_id"]) == state
            and order["purchase"] is not None
            and start <= order["purchase"] <= end
        )
        actual_state_orders = sum_daily(payload["state_purchases"].get(state, []), start, end, 1)
        actual_state_cents = sum_daily(payload["state_purchases"].get(state, []), start, end, 2)
        check(
            actual_state_orders == expected_state_orders,
            f"{as_of} {state} {days}-day state orders placed differ",
        )
        check(
            actual_state_cents == expected_state_cents,
            f"{as_of} {state} {days}-day state item price differs",
        )

    item_lookup = {row[0]: row[6] for row in payload["orders"]}
    checked_prices = 0
    for order in orders:
        if order["order_id"] not in item_lookup:
            continue
        if order["purchase"] is None or order["purchase"] < date(2018, 5, 1):
            continue
        check(
            item_lookup[order["order_id"]] == item_totals.get(order["order_id"], 0),
            f"{order['order_id']} item price differs",
        )
        checked_prices += 1
        if checked_prices >= 500:
            break

    synthetic = {
        "canceled": {"promised_day": date(2018, 8, 10), "purchase_day": date(2018, 8, 1),
                     "delivered_day": None, "handover_day": None, "status": "canceled"},
        "open": {"promised_day": date(2018, 8, 10), "purchase_day": date(2018, 8, 1),
                 "delivered_day": None, "handover_day": None, "status": "shipped"},
    }
    synthetic_rows = builder.build_overdue_not_delivered(synthetic)
    synthetic_aug = next(row for row in synthetic_rows if row[0] == "2018-08-22")
    check(synthetic_aug[1] == 2,
          "canceled orders must stay open in overdue, not yet delivered")
    check(synthetic_aug[3] == 2,
          "canceled orders must stay in the overdue due denominator")

    check_dom_references()
    check_deterministic()
    check_js_syntax()
    check_queue_count_scope()
    check_redesign_contract()
    print(f"PASS data checks ({CHECKS} assertions)")


if __name__ == "__main__":
    main()
