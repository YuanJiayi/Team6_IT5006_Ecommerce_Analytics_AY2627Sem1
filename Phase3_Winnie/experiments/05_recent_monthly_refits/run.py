"""Refit the late-warning model each month on recent orders only.

Run from the repository root:
    python Phase3_Winnie/experiments/05_recent_monthly_refits/run.py

Each scored order uses the latest refit before its handover. A refit uses orders purchased and handed over before the
refit date with delivery outcomes already known, as in Phase2/experiments/monthly_retraining/run.py, limited to the last
90 or 180 days. The window with the higher mean validation PR-AUC goes to the gates. Writes CSVs next to this file.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

from task4_common import (  # noqa: E402
    BASELINE_CONFIG, LABEL, TOP_SHARE, evaluate, exp03, load_data, monthly_flags, saved_model_scores, validation_rows,
)

WINDOWS = {"monthly_90d": 90, "monthly_180d": 180}
ALL_HISTORY = "monthly_all_history"
TEST_START = pd.Timestamp("2018-05-26")
PHASE2_HANDOVER_MONTH = {ALL_HISTORY: 379, "model": 370, "slack_rule": 394}


def refit_dates(start, last_handover):
    months = pd.date_range(start.normalize() + pd.offsets.MonthBegin(1), last_handover, freq="MS")
    return [start, *months]


def monthly_scores(data, rows, start, days):
    """Scores for `rows`, using the latest refit before each order's handover. `days` of None keeps all history."""
    table = data["table"]
    dates = refit_dates(start, rows["handover_time"].max())
    result = np.full(len(rows), np.nan)
    for begin, end in zip(dates, [*dates[1:], pd.Timestamp.max]):
        take = rows["handover_time"].ge(begin) & rows["handover_time"].lt(end)
        if not take.any():
            continue
        known = (table["order_purchase_timestamp"].lt(begin) & table["handover_time"].lt(begin)
                 & table["outcome_available_at"].lt(begin))
        if days is not None:
            known &= table["order_purchase_timestamp"].ge(begin - pd.Timedelta(days=days))
        pipeline, columns = exp03.fit_config(BASELINE_CONFIG, table[known], begin, data["spec"], data["spec_base"], data["params"])
        result[np.flatnonzero(take.to_numpy())] = pipeline.predict_proba(rows.loc[take, columns])[:, 1]
    if np.isnan(result).any():
        raise AssertionError(f"{np.isnan(result).sum()} orders have no refit before their handover")
    return result


def run():
    data = load_data()
    windows = validation_rows(data["train"], data["folds"])
    valid_scores = {}
    for name, days in WINDOWS.items():
        valid_scores[name] = [monthly_scores(data, rows, rows["handover_time"].min(), days) for _, rows in windows]
    means = {}
    for name, per_window in valid_scores.items():
        means[name] = np.mean([shared_pr(rows[LABEL], s) for (_, rows), s in zip(windows, per_window)])
    chosen = max(means, key=means.get)
    print(f"Validation means {means}; candidate fixed before test scoring: {chosen}", flush=True)

    test = data["test"]
    test_scores = {chosen: monthly_scores(data, test, TEST_START, WINDOWS[chosen]),
                   ALL_HISTORY: monthly_scores(data, test, TEST_START, None)}
    handover_months = test["handover_time"].dt.to_period("M").astype(str).to_numpy()
    y = test[LABEL].to_numpy().astype(bool)
    replay = {**test_scores, "model": saved_model_scores(data, test), "slack_rule": exp03.shared.rule_scores(data["train"], test)}
    counts = {name: int((y & monthly_flags(s, handover_months, TOP_SHARE)).sum()) for name, s in replay.items()}
    print(f"Late caught by handover month: {counts}", flush=True)
    for name, expected in PHASE2_HANDOVER_MONTH.items():
        if counts[name] != expected:
            raise AssertionError(f"{name} does not reproduce the Phase 2 replay: {counts[name]} vs {expected}")
    pd.Series(counts, name="caught_handover_month").rename_axis("ranking").to_csv(HERE / "test_handover_month.csv")
    evaluate(HERE, data, valid_scores, test_scores, {chosen: "recent_monthly"})


def shared_pr(y, scores):
    return exp03.shared.average_precision_score(y, scores)


if __name__ == "__main__":
    run()
