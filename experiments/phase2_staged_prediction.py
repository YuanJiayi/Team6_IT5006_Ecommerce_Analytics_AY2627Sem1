"""Explore whether new information at carrier handover improves late-risk ranking.

Run from the repository root with:
    it5006-proj/bin/python experiments/phase2_staged_prediction.py

This is a training-period validation experiment. It does not read test rows from
the prepared table, fit on test outcomes, or alter the frozen final evaluation.
All three models use the same orders, temporal folds, and logistic settings; only
the information available at each prediction stage changes.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score
from threadpoolctl import threadpool_limits


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from phase2_classification import (  # noqa: E402
    feature_columns, feature_stages, load_primary_training, make_pipeline, temporal_folds,
)


def attach_handover_facts(train: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Join only training orders to their original event timestamps."""
    dates = [
        "order_approved_at", "order_delivered_carrier_date",
        "order_estimated_delivery_date",
    ]
    raw = pd.read_csv(
        ROOT / "data/olist_orders_dataset.csv",
        usecols=["order_id", *dates], parse_dates=dates,
    )
    frame = train.merge(raw, on="order_id", how="left", validate="one_to_one")
    purchase = frame["order_purchase_timestamp"]
    approval = frame["order_approved_at"]
    handover = frame["order_delivered_carrier_date"]
    promise = frame["order_estimated_delivery_date"]

    # Handover is the second prediction moment. An approval recorded after it
    # cannot be used then. If handover was after the promised calendar day,
    # lateness is already certain, so that simple rule is reported separately.
    valid_handover = handover.notna() & (handover >= purchase)
    already_late = valid_handover & (handover.dt.normalize() > promise.dt.normalize())
    eligible = valid_handover & ~already_late
    approval_known = approval.notna() & (approval >= purchase) & (approval <= handover)

    frame["approval_lag_days"] = ((approval - purchase).dt.total_seconds() / 86400).where(approval_known)
    frame["handover_lag_days"] = (handover - purchase).dt.total_seconds() / 86400
    frame["days_left_at_handover"] = (promise.dt.normalize() - handover.dt.normalize()).dt.days
    frame["handover_eligible"] = eligible

    counts = {
        "primary_training_orders": len(frame),
        "eligible_before_or_on_promised_day": int(eligible.sum()),
        "handover_after_promised_day_obvious_late": int(already_late.sum()),
        "missing_or_pre_purchase_handover": int((~valid_handover).sum()),
        "approval_not_known_at_handover_among_eligible": int((eligible & ~approval_known).sum()),
    }
    assert frame["order_id"].is_unique
    assert frame.loc[already_late, "is_late"].eq(1).all()
    return frame, counts


def top_tenth(y: np.ndarray, probability: np.ndarray) -> dict:
    k = int(np.ceil(len(y) * 0.10))
    chosen = np.argsort(-probability, kind="stable")[:k]
    found = int(y[chosen].sum())
    return {
        "reviewed": k, "late_found": found,
        "precision_at_10pct": found / k,
        "recall_at_10pct": found / int(y.sum()),
    }


def main() -> None:
    train, spec = load_primary_training(ROOT)
    folds = temporal_folds(train, spec)
    frame, cohort = attach_handover_facts(train)
    base = feature_stages(spec)["seller"]
    log_features = spec["linear_log_candidates"]
    stages = {
        "checkout": [],
        "after_approval": ["approval_lag_days"],
        "at_handover": ["approval_lag_days", "handover_lag_days", "days_left_at_handover"],
    }

    rows = []
    for fold_number, (fit, valid) in enumerate(folds):
        fit = fit[frame.iloc[fit]["handover_eligible"].to_numpy()]
        valid = valid[frame.iloc[valid]["handover_eligible"].to_numpy()]
        # The same subset in all three stages isolates the value of new facts.
        assert len(fit) and len(valid)
        y_fit = frame.iloc[fit][spec["classification_target"]]
        y_valid = frame.iloc[valid][spec["classification_target"]].to_numpy()
        assert y_fit.nunique() == 2 and len(np.unique(y_valid)) == 2
        for name, extra in stages.items():
            feature_set = {
                "numeric": [*base["numeric"], *extra],
                "categorical": base["categorical"],
            }
            numeric, categorical = feature_columns(spec, feature_set=feature_set)
            pipeline = make_pipeline(
                "linear", {"C": 0.1, "class_weight": None}, spec,
                feature_set=feature_set, log_features=log_features,
            )
            with threadpool_limits(limits=1):
                pipeline.fit(frame.iloc[fit][numeric + categorical], y_fit)
                probability = pipeline.predict_proba(frame.iloc[valid][numeric + categorical])[:, 1]
            row = {
                "stage": name, "fold": fold_number, "fit_orders": len(fit),
                "validation_orders": len(valid), "late_orders": int(y_valid.sum()),
                "late_rate": float(y_valid.mean()),
                "average_precision": float(average_precision_score(y_valid, probability)),
                "roc_auc": float(roc_auc_score(y_valid, probability)),
                **top_tenth(y_valid, probability),
            }
            rows.append(row)

    results = pd.DataFrame(rows)
    summary = results.groupby("stage", sort=False).agg(
        mean_ap=("average_precision", "mean"),
        mean_roc_auc=("roc_auc", "mean"),
        reviewed=("reviewed", "sum"),
        late_found=("late_found", "sum"),
        late_total=("late_orders", "sum"),
    )
    summary["pooled_precision_at_10pct"] = summary["late_found"] / summary["reviewed"]
    summary["pooled_recall_at_10pct"] = summary["late_found"] / summary["late_total"]
    summary["mean_ap_gain_vs_checkout"] = summary["mean_ap"] - summary.loc["checkout", "mean_ap"]
    print("Cohort:", cohort)
    print("\nMean of five temporal-window scores; top 10% selected within each window:")
    print(summary.round(4).to_string())
    print("\nPer-window average precision:")
    print(results.pivot(index="fold", columns="stage", values="average_precision").round(4).to_string())

    output = ROOT / "experiments/phase2_staged_prediction.json"
    output.write_text(json.dumps({
        "scope": "exploratory primary-training temporal folds only; same eligible orders at each stage",
        "prediction_stage": "at carrier handover, before customer delivery; handover after promised date separated",
        "cohort": cohort,
        "extra_features_by_stage": stages,
        "rows": results.to_dict(orient="records"),
        "summary": summary.reset_index().to_dict(orient="records"),
    }, indent=2) + "\n")
    print(f"\nSaved {output}")


if __name__ == "__main__":
    main()
