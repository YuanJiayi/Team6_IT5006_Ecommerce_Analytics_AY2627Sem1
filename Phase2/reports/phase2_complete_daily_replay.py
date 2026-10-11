"""Complete-calendar-day replay of the *already selected* handover classifier.

Run from the Phase2 folder:
    python reports/phase2_complete_daily_replay.py

This evaluation uses an illustrative top-10% review workload. It does not change
the selected model or its features. Each validation fit uses only outcomes available before its
purchase-window start; the later-period fit uses the mature pre-cutoff train.
For a non-overlapping calendar span associated with each validation window,
score *every* eligible carrier handover on each day, including orders outside
the purchase-defined evaluation cohort. The later span runs from the fixed
test cutoff through the last test-purchase calendar day. At day's end, rank
that day's full eligible batch and flag ceil(share of it) for next-day review.

The replay was devised after the later period was inspected. Its later-period
numbers are descriptive, not an independent, prospectively specified test.
The source table contains delivered orders only, so the replay cannot recover
undelivered orders that a live operations queue would have contained.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from phase2_eta import chronological_folds  # noqa: E402
from phase2_handover_classifier import (  # noqa: E402
    action_metrics, fit_scores, load_cohort, top_flags, variant_spec,
)

OUT = ROOT / "results/phase2/final"
SUMMARY = OUT / "complete_daily_replay_summary.csv"
DAILY = OUT / "complete_daily_replay_days.csv"


def score_period(name: str, fit: pd.DataFrame, cohort: pd.DataFrame,
                 start: pd.Timestamp, end: pd.Timestamp, features: dict,
                 log_features: list[str], share: float) -> tuple[dict, pd.DataFrame]:
    """Score all eligible handovers from start through end, both at midnight."""
    assert start == start.normalize() and end == end.normalize() and start <= end
    orders = cohort.loc[cohort["handover_time"].ge(start)
                        & cohort["handover_time"].lt(end + pd.Timedelta(days=1))].copy()
    orders.sort_values(["handover_time", "order_id"], kind="stable", inplace=True)
    orders.reset_index(drop=True, inplace=True)
    if orders.empty or not orders["order_id"].is_unique:
        raise ValueError(f"Empty or duplicate scoring cohort: {name}")
    if not fit["outcome_available_at"].lt(start).all():
        raise ValueError(f"Training outcome available after scoring starts: {name}")
    if set(fit["order_id"]).intersection(orders["order_id"]):
        raise ValueError(f"Training rows reappear in scoring cohort: {name}")
    if not orders["handover_time"].lt(orders["outcome_available_at"]).all():
        raise ValueError(f"Outcome precedes handover: {name}")

    scores = fit_scores("linear", {"C": 0.01, "class_weight": None},
                        features, fit, orders, log_features)
    labels = orders["is_late"].to_numpy().astype(bool)
    days = orders["handover_time"].dt.normalize().to_numpy()
    flags = np.zeros(len(orders), dtype=bool)
    rows = []
    for day in pd.date_range(start, end):
        indices = np.flatnonzero(days == day.to_datetime64())
        if not len(indices):
            rows.append({"period": name, "handover_day": day.date(), "orders": 0,
                         "late": 0, "flagged": 0, "caught": 0,
                         "flagged_delivered_by_next_09": 0, "caught_still_open_next_09": 0,
                         "flagged_promise_passed_by_next_09": 0,
                         "caught_promise_passed_by_next_09": 0,
                         "caught_open_and_promise_not_passed_next_09": 0})
            continue
        selected = top_flags(scores[indices], share)
        flags[indices[selected]] = True
        review_time = day + pd.Timedelta(days=1, hours=9)
        delivered = orders.iloc[indices]["outcome_available_at"].le(review_time).to_numpy()
        # Olist promises a calendar date. On that date the promise has not yet
        # been missed, even if the review happens at 09:00.
        promise_passed = orders.iloc[indices]["promised_day"].lt(review_time.normalize()).to_numpy()
        rows.append({
            "period": name, "handover_day": day.date(), "orders": len(indices),
            "late": int(labels[indices].sum()), "flagged": int(selected.sum()),
            "caught": int(np.sum(labels[indices] & selected)),
            "flagged_delivered_by_next_09": int(np.sum(selected & delivered)),
            "caught_still_open_next_09": int(np.sum(selected & labels[indices] & ~delivered)),
            "flagged_promise_passed_by_next_09": int(np.sum(selected & promise_passed)),
            "caught_promise_passed_by_next_09": int(np.sum(selected & labels[indices] & promise_passed)),
            "caught_open_and_promise_not_passed_next_09":
                int(np.sum(selected & labels[indices] & ~delivered & ~promise_passed)),
        })
    daily = pd.DataFrame(rows)
    populated = daily.loc[daily["orders"].gt(0)]
    assert int(daily["orders"].sum()) == len(orders)
    assert int(daily["flagged"].sum()) == int(flags.sum())
    assert int(daily["caught"].sum()) == int(np.sum(flags & labels))
    summary = {
        "period": name, "first_day": start.date(), "last_day": end.date(),
        "calendar_days": len(daily), "days_with_handovers": len(populated),
        "orders": len(orders), "late": int(labels.sum()), "late_rate": float(labels.mean()),
        "flagged": int(flags.sum()), "caught": int(np.sum(flags & labels)),
        "action_share": float(flags.mean()),
        "pr_auc": float(average_precision_score(labels, scores)),
        "roc_auc": float(roc_auc_score(labels, scores)),
        "median_daily_orders": float(populated["orders"].median()),
        "p90_daily_orders": float(populated["orders"].quantile(.9)),
        "max_daily_orders": int(populated["orders"].max()),
        "median_daily_flagged": float(populated["flagged"].median()),
        "p90_daily_flagged": float(populated["flagged"].quantile(.9)),
        "max_daily_flagged": int(populated["flagged"].max()),
        "flagged_delivered_by_next_09": int(daily["flagged_delivered_by_next_09"].sum()),
        "caught_still_open_next_09": int(daily["caught_still_open_next_09"].sum()),
        "flagged_promise_passed_by_next_09": int(daily["flagged_promise_passed_by_next_09"].sum()),
        "caught_promise_passed_by_next_09": int(daily["caught_promise_passed_by_next_09"].sum()),
        "caught_open_and_promise_not_passed_next_09":
            int(daily["caught_open_and_promise_not_passed_next_09"].sum()),
        **action_metrics(labels, flags),
    }
    return summary, daily


def main() -> None:
    selection = json.loads((ROOT / "results/phase2/handover/selection.json").read_text())
    primary = selection["variants"]["primary"]
    if primary["selected"] != "logistic_c0.01":
        raise ValueError("Saved selected model changed; update this replay deliberately")
    share = 0.10  # Report scenario; the saved model-selection share remains unchanged.
    cohort, spec, _ = load_cohort(ROOT)
    promises = pd.read_csv(ROOT / "data/olist_orders_dataset.csv",
                           usecols=["order_id", "order_estimated_delivery_date"],
                           parse_dates=["order_estimated_delivery_date"])
    if not promises["order_id"].is_unique:
        raise ValueError("Expected one original Olist promise per order")
    cohort = cohort.merge(promises, on="order_id", how="left", validate="one_to_one")
    cohort["promised_day"] = cohort["order_estimated_delivery_date"].dt.normalize()
    if cohort["promised_day"].isna().any():
        raise ValueError("Every replayed order needs an original promised date")
    observed_late = cohort["outcome_available_at"].dt.normalize().gt(cohort["promised_day"])
    if not observed_late.eq(cohort["is_late"].astype(bool)).all():
        raise ValueError("Original promised date does not reproduce the saved late label")
    train = cohort.loc[cohort[spec["primary_split"]].eq("train")].reset_index(drop=True)
    test = cohort.loc[cohort[spec["primary_split"]].eq("test")].reset_index(drop=True)
    folds = chronological_folds(train, spec)
    features = variant_spec(spec)
    log_features = spec["linear_log_candidates"]

    summaries, daily_rows = [], []
    for window, (fit_idx, valid_idx) in enumerate(folds):
        valid = train.iloc[valid_idx]
        # Begin at the first *complete* day after the fold's training cutoff;
        # the final purchase day belongs to this non-overlapping calendar span.
        start = valid["order_purchase_timestamp"].min().normalize() + pd.Timedelta(days=1)
        end = valid["order_purchase_timestamp"].max().normalize()
        result, days = score_period(f"validation_{window}", train.iloc[fit_idx],
                                    cohort, start, end, features, log_features, share)
        summaries.append(result)
        daily_rows.append(days)
    start = pd.Timestamp(spec["primary_cutoff"])
    end = test["order_purchase_timestamp"].max().normalize()
    result, days = score_period("later_opened", train, cohort, start, end,
                                features, log_features, share)
    summaries.append(result)
    daily_rows.append(days)

    OUT.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(summaries).to_csv(SUMMARY, index=False)
    pd.concat(daily_rows, ignore_index=True).to_csv(DAILY, index=False)
    print(SUMMARY)
    print(DAILY)
    print(pd.DataFrame(summaries).to_string(index=False))


if __name__ == "__main__":
    main()
