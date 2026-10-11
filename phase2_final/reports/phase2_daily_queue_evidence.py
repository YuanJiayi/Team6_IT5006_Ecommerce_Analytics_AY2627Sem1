"""Fixed-k handover-day batch sensitivity for the selected late-risk classifier.

Run from the phase2_final folder:
    python reports/phase2_daily_queue_evidence.py

The saved validation choice (unweighted logistic C=0.01, k=14%) is held fixed.
Each chronological validation model is fitted only on mature, earlier orders;
the opened later test is scored after one full pre-test training fit. This is
not another model-selection run.

The batch policy ranks orders by actual carrier-handover calendar day. It could
be executed after that day's handovers are known, with review the next day.
Its exact top-k membership is not available immediately at each handover.
Also, both validation and test contain only orders from purchase-defined
evaluation cohorts, so they omit other orders handed over on the same day.
The results are therefore a timing sensitivity, not proof of a complete live
daily queue. The comparison row reproduces the report's retrospective
validation-window policy or purchase-month test policy as appropriate.
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
    action_metrics, fit_scores, load_cohort, monthly_flags, top_flags, variant_spec,
)

OUTPUT = ROOT / "results/phase2/final/daily_queue_sensitivity.csv"


def handover_day_flags(scores: np.ndarray, handover_time: pd.Series, share: float) -> np.ndarray:
    """Top share in each observed handover day; ties follow saved stable ranking."""
    days = handover_time.dt.normalize().to_numpy()
    flags = np.zeros(len(scores), dtype=bool)
    for day in np.unique(days):
        indices = np.flatnonzero(days == day)
        flags[indices[top_flags(scores[indices], share)]] = True
    return flags


def score_rows(period: str, orders: pd.DataFrame, scores: np.ndarray, share: float) -> list[dict]:
    labels = orders["is_late"].to_numpy().astype(bool)
    days = orders["handover_time"].dt.normalize()
    daily = handover_day_flags(scores, orders["handover_time"], share)
    if period.startswith("validation_"):
        comparison = top_flags(scores, share)
        comparison_name = "original_validation_window"
    else:
        months = orders["order_purchase_timestamp"].dt.to_period("M").astype(str).to_numpy()
        comparison = monthly_flags(scores, months, share)
        comparison_name = "original_test_purchase_month"

    day_sizes = days.value_counts()
    common = {
        "period": period,
        "orders": len(orders),
        "late": int(labels.sum()),
        "handover_days": len(day_sizes),
        "first_handover_day": str(days.min().date()),
        "last_handover_day": str(days.max().date()),
        "median_orders_per_handover_day_in_cohort": float(day_sizes.median()),
        "days_with_fewer_than_8_cohort_orders": int((day_sizes < 8).sum()),
        "pr_auc": float(average_precision_score(labels, scores)),
        "roc_auc": float(roc_auc_score(labels, scores)),
    }
    next_day_start = days + pd.Timedelta(days=1)
    delivered_before_next_day = orders["outcome_available_at"].le(next_day_start).to_numpy()
    output = []
    for policy, flags in (("handover_day_batch", daily), (comparison_name, comparison)):
        output.append({
            **common,
            "policy": policy,
            "flagged": int(flags.sum()),
            "caught": int(np.sum(flags & labels)),
            "actual_action_share": float(flags.mean()),
            "flagged_delivered_before_next_day": int(np.sum(flags & delivered_before_next_day)),
            **action_metrics(labels, flags),
        })
    return output


def main() -> None:
    selected = json.loads((ROOT / "results/phase2/handover/selection.json").read_text())
    primary = selected["variants"]["primary"]
    assert primary["selected"] == "logistic_c0.01"
    share = float(primary["k"])
    assert share == 0.14

    table, spec, _ = load_cohort(ROOT)
    train = table.loc[table[spec["primary_split"]].eq("train")].reset_index(drop=True)
    test = table.loc[table[spec["primary_split"]].eq("test")].reset_index(drop=True)
    folds = chronological_folds(train, spec)
    features = variant_spec(spec)
    log_features = spec["linear_log_candidates"]
    params = {"C": 0.01, "class_weight": None}

    rows = []
    for window, (fit, valid) in enumerate(folds):
        validation = train.iloc[valid].reset_index(drop=True)
        scores = fit_scores("linear", params, features, train.iloc[fit], validation, log_features)
        rows.extend(score_rows(f"validation_{window}", validation, scores, share))
    test_scores = fit_scores("linear", params, features, train, test, log_features)
    rows.extend(score_rows("test_opened_posthoc", test, test_scores, share))

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(OUTPUT, index=False)
    print(OUTPUT)


if __name__ == "__main__":
    main()
