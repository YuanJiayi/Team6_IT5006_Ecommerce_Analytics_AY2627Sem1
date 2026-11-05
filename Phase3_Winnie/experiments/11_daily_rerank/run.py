"""Simulate re-ranking all open orders each morning against ranking only the previous day's handovers.

Run from the repository root:
    python Phase3_Winnie/experiments/11_daily_rerank/run.py

Declared in Phase3_Winnie/EVALUATION_CRITERIA.md. Follows Phase2/reports/phase2_complete_daily_replay.py for periods, the
9am review and the promised date, and stops if it does not reproduce that replay's saved counts. Writes CSVs next to this file.
"""

from __future__ import annotations

import importlib.util
import itertools
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

from task4_common import PHASE2, PROMISE, TOP_SHARE, LABEL, load_data, saved_model_scores, shared, top_flags, validation_rows  # noqa: E402

_spec = importlib.util.spec_from_file_location("experiment_04", HERE.parent / "04_mix_ratios" / "run.py")
exp04 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(exp04)

VERSIONS = ["A_rule", "A_mix", "queue_fixed_rule", "queue_fixed_mix", "B_rule", "B_mix"]
PAIRS = [("B_mix", "A_mix"), ("B_rule", "A_rule"), ("B_mix", "B_rule"), ("A_mix", "A_rule"), ("B_mix", "queue_fixed_mix"),
         ("B_rule", "queue_fixed_rule"), ("queue_fixed_mix", "A_mix"), ("queue_fixed_rule", "A_rule")]
REPLAY_SUMMARY = PHASE2 / "results/phase2/final/complete_daily_replay_summary.csv"
LATER = "later_opened"
LEAD_DAYS = [1, 3, 5]  # added after the first run, because the updated rule reviewed late orders with fewer days left


def add_promised_day(table):
    promises = pd.read_csv(PHASE2 / "data/olist_orders_dataset.csv", usecols=["order_id", "order_estimated_delivery_date"],
                           parse_dates=["order_estimated_delivery_date"])
    out = table.merge(promises, on="order_id", how="left", validate="one_to_one")
    out["promised_day"] = out["order_estimated_delivery_date"].dt.normalize()
    if not out["outcome_available_at"].dt.normalize().gt(out["promised_day"]).eq(out[LABEL].astype(bool)).all():
        raise ValueError("Original promised date does not reproduce the saved late label")
    return out


def span_orders(table, start, end):
    orders = table[table["handover_time"].ge(start) & table["handover_time"].lt(end + pd.Timedelta(days=1))]
    return orders.sort_values(["handover_time", "order_id"], kind="stable").reset_index(drop=True)


def days_between(later, earlier):
    return (later - earlier) / np.timedelta64(1, "D")


def choose(score, pool, count):
    index = np.flatnonzero(pool)
    return index[np.argsort(-score[index], kind="stable")[:count]]


def simulate(period, orders, start, end, model_scores, rule_scores, model_ref, rule_ref):
    """One row per morning and version; also returns the replay counts for the plain top-10% model list."""
    handover = orders["handover_time"].to_numpy(dtype="datetime64[ns]")
    delivered = orders["outcome_available_at"].to_numpy(dtype="datetime64[ns]")
    purchase = orders["order_purchase_timestamp"].to_numpy(dtype="datetime64[ns]")
    promised_day = orders["promised_day"].to_numpy(dtype="datetime64[ns]")
    promised_days = orders[PROMISE].to_numpy(dtype=float)
    handover_day = orders["handover_time"].dt.normalize().to_numpy(dtype="datetime64[ns]")
    y = orders[LABEL].to_numpy().astype(bool)
    model_pct = shared.percentile(model_ref, model_scores)
    mix_fixed = 0.5 * model_pct + 0.5 * shared.percentile(rule_ref, rule_scores)
    reviewed = {name: np.zeros(len(orders), dtype=bool) for name in VERSIONS}
    rows, replay = [], {"flagged": 0, "caught": 0, "before": 0}
    for day in pd.date_range(start, end):
        batch = handover_day == day.to_datetime64()
        count = int(top_flags(np.zeros(int(batch.sum())), TOP_SHARE).sum()) if batch.any() else 0
        morning = (day + pd.Timedelta(days=1)).to_datetime64()
        review = (day + pd.Timedelta(days=1, hours=9)).to_datetime64()
        still_open = delivered > review
        before_date = promised_day >= morning
        open_batch = batch & still_open & before_date
        queue = (handover < morning) & still_open & before_date
        if batch.any():
            index = np.flatnonzero(batch)
            selected = index[top_flags(model_scores[index], TOP_SHARE)]
            replay["flagged"] += len(selected)
            replay["caught"] += int(y[selected].sum())
            replay["before"] += int((y[selected] & still_open[selected] & before_date[selected]).sum())
        days_left = promised_days - days_between(review, purchase)
        updated_rule = -np.minimum(-rule_scores, days_left)
        updated_mix = 0.5 * model_pct + 0.5 * shared.percentile(rule_ref, updated_rule)
        lists = {"A_rule": (rule_scores, open_batch), "A_mix": (mix_fixed, open_batch), "queue_fixed_rule": (rule_scores, queue),
                 "queue_fixed_mix": (mix_fixed, queue), "B_rule": (updated_rule, queue), "B_mix": (updated_mix, queue)}
        for name, (score, pool) in lists.items():
            picked = choose(score, pool & ~reviewed[name], count)
            reviewed[name][picked] = True
            late = y[picked]
            rows.append({"period": period, "morning": str(day.date() + pd.Timedelta(days=1)), "version": name, "capacity": count,
                         "reviewed": len(picked), "caught_before_date": int(late.sum()),
                         "days_left_total": float(days_left[picked][late].sum()), "queue_size": int(queue.sum()),
                         **{f"caught_{days}plus_days": int((days_left[picked][late] >= days).sum()) for days in LEAD_DAYS}})
    return pd.DataFrame(rows), replay


def summarise(frame, label):
    rows = []
    for name in VERSIONS:
        part = frame[frame["version"].eq(name)]
        caught = int(part["caught_before_date"].sum())
        rows.append({"scope": label, "version": name, "capacity": int(part["capacity"].sum()), "reviewed": int(part["reviewed"].sum()), "caught_before_date": caught,
                     "precision": caught / max(int(part["reviewed"].sum()), 1), "mean_days_left": part["days_left_total"].sum() / max(caught, 1),
                     "median_queue": float(part["queue_size"].median()),
                     **{f"caught_{days}plus_days": int(part[f"caught_{days}plus_days"].sum()) for days in LEAD_DAYS}})
    return rows


def morning_intervals(test_frame):
    rng = np.random.default_rng(shared.SEED)
    size = test_frame["morning"].nunique()
    picks = [rng.integers(0, size, size) for _ in range(shared.BOOTSTRAPS)]
    rows = []
    for metric in ["caught_before_date", *[f"caught_{days}plus_days" for days in LEAD_DAYS]]:
        table = test_frame.pivot(index="morning", columns="version", values=metric)
        for a, b in PAIRS:
            gap = (table[a] - table[b]).to_numpy(dtype=float)
            draws = np.array([gap[p].sum() for p in picks])
            rows.append({"metric": metric, "a": a, "b": b, "a_minus_b": gap.sum(), "ci_low": np.percentile(draws, 2.5), "ci_high": np.percentile(draws, 97.5)})
    return pd.DataFrame(rows)


def check_replay(name, replay, saved):
    row = saved.loc[name]
    got = (replay["flagged"], replay["caught"], replay["before"])
    expected = (int(row["flagged"]), int(row["caught"]), int(row["caught_open_and_promise_not_passed_next_09"]))
    print(f"{name}: flagged, caught, before date {got} (replay {expected})", flush=True)
    if got != expected:
        raise AssertionError(f"{name} does not reproduce the report's replay")


def run():
    data = load_data(add_promised_day)
    table, train, spec = data["table"], data["train"], data["spec"]
    saved = pd.read_csv(REPLAY_SUMMARY).set_index("period")
    frames = []
    for number, (fit_rows, rows) in enumerate(validation_rows(train, data["folds"])):
        start = rows["order_purchase_timestamp"].min().normalize() + pd.Timedelta(days=1)
        end = rows["order_purchase_timestamp"].max().normalize()
        orders = span_orders(table, start, end)
        model_fit, model_scores = exp04.phase2_scores(data, fit_rows, orders)
        frame, replay = simulate(f"W{number + 1}", orders, start, end, model_scores, shared.rule_scores(fit_rows, orders),
                                 model_fit, shared.rule_scores(fit_rows, fit_rows))
        check_replay(f"validation_{number}", replay, saved)
        frames.append(frame)
    start = pd.Timestamp(spec["primary_cutoff"])
    end = data["test"]["order_purchase_timestamp"].max().normalize()
    orders = span_orders(table, start, end)
    test_frame, replay = simulate("test", orders, start, end, saved_model_scores(data, orders), shared.rule_scores(train, orders),
                                  saved_model_scores(data, train), shared.rule_scores(train, train))
    check_replay(LATER, replay, saved)

    validation = pd.concat(frames, ignore_index=True)
    rows = summarise(validation, "validation")
    for period, part in validation.groupby("period"):
        rows += summarise(part, period)
    rows += summarise(test_frame, "test")
    for month, part in test_frame.groupby(test_frame["morning"].str[:7]):
        rows += summarise(part, month)
    summary = pd.DataFrame(rows)
    intervals = morning_intervals(test_frame)

    pd.concat([validation, test_frame], ignore_index=True).to_csv(HERE / "mornings.csv", index=False)
    summary.to_csv(HERE / "summary.csv", index=False)
    intervals.to_csv(HERE / "test_intervals.csv", index=False)

    print(summary.round(3).to_string(index=False))
    print(intervals.round(2).to_string(index=False))
    test_part = summary[summary["scope"].eq("test")].set_index("version")["caught_before_date"]
    valid_part = summary[summary["scope"].eq("validation")].set_index("version")["caught_before_date"]
    lead = intervals[intervals["metric"].eq("caught_before_date") & intervals["a"].eq("B_mix") & intervals["b"].eq("A_mix")].iloc[0]
    recommended = bool(test_part["B_mix"] > test_part["A_mix"] and lead["ci_low"] > 0 and valid_part["B_mix"] > valid_part["A_mix"])
    print(f"Declared decision, option B recommended: {recommended}", flush=True)


if __name__ == "__main__":
    run()
