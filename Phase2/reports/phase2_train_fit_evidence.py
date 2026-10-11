"""Apparent training scores for the saved Phase 2 selected models.

Run from the Phase2 folder with ``python reports/phase2_train_fit_evidence.py``.
The script loads only rows assigned to the pre-test training split, fits each
selected Pipeline on those rows, and scores those same rows. These are
descriptive in-sample scores, not independent validation results. Model
selection and the held-out test results are not recomputed here.
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

from phase2_classification import feature_columns, load_primary_training, make_pipeline  # noqa: E402
from phase2_eta import add_stage_features, make_stage_pipeline, stage_spec  # noqa: E402
from phase2_handover_classifier import (  # noqa: E402
    action_metrics, add_history_extension, add_remaining_slack, load_cohort, monthly_flags, variant_spec,
)
from phase2_regression import regression_metrics  # noqa: E402

OUTPUT = ROOT / "results/phase2/final/train_apparent_metrics.csv"


def main() -> None:
    train, spec = load_primary_training(ROOT)
    cutoff = pd.Timestamp(spec["primary_cutoff"])
    assert train["order_purchase_timestamp"].lt(cutoff).all()
    assert train["outcome_available_at"].lt(cutoff).all()

    orders = pd.read_csv(
        ROOT / "data/olist_orders_dataset.csv",
        usecols=["order_id", "customer_id", "order_approved_at", "order_delivered_carrier_date"],
        parse_dates=["order_approved_at", "order_delivered_carrier_date"],
    )
    staged = add_stage_features(train, orders)
    staged = staged.loc[staged["handover_valid"]].reset_index(drop=True)
    assert len(staged) == 75039

    eta_selection = json.loads((ROOT / "results/phase2/eta/selection.json").read_text())
    classifier_selection = json.loads((ROOT / "results/phase2/handover/selection.json").read_text())
    assert eta_selection["checkout_selected_candidate"] == "ridge_100"
    assert eta_selection["selected_candidate"] == "ridge_100"
    assert classifier_selection["variants"]["primary"]["selected"] == "logistic_c0.01"
    share = float(classifier_selection["variants"]["primary"]["k"])

    rows = []
    for stage in ("checkout", "handover"):
        stage_features = stage_spec(spec, stage)
        columns = sum(feature_columns(stage_features), [])
        model = make_stage_pipeline("ridge", {"alpha": 100.0}, stage_features)
        with threadpool_limits(limits=1):
            model.fit(staged[columns], staged["delivery_days"])
            predicted = model.predict(staged[columns])
        rows.append({
            "task": "regression", "stage": stage, "model": "ridge_100",
            "scope": "apparent_in_sample", "orders": len(staged),
            **regression_metrics(staged["delivery_days"], predicted),
        })

    customers = pd.read_csv(
        ROOT / "data/olist_customers_dataset.csv",
        usecols=["customer_id", "customer_zip_code_prefix"],
    )
    zips = orders.merge(customers, on="customer_id", validate="many_to_one")[
        ["order_id", "customer_zip_code_prefix"]
    ]
    classified = add_history_extension(add_remaining_slack(staged), zips)
    backlog, _, _ = load_cohort(ROOT)
    classified = classified.merge(backlog[["order_id", "state_backlog_recent_vs_8wk"]],
                                  on="order_id", how="left", validate="one_to_one")
    classification_features = variant_spec(spec)
    numeric, categorical = feature_columns(classification_features)
    model = make_pipeline(
        "linear", {"C": 0.01, "class_weight": None}, classification_features,
        log_features=spec["linear_log_candidates"],
    )
    with threadpool_limits(limits=1):
        model.fit(classified[numeric + categorical], classified["is_late"])
        scores = model.predict_proba(classified[numeric + categorical])[:, 1]
    labels = classified["is_late"].to_numpy().astype(bool)
    months = classified["order_purchase_timestamp"].dt.to_period("M").astype(str).to_numpy()
    flags = monthly_flags(scores, months, share)
    rows.append({
        "task": "classification", "stage": "handover", "model": "logistic_c0.01",
        "scope": "apparent_in_sample", "orders": len(classified),
        "late_orders": int(labels.sum()), "late_rate": float(labels.mean()),
        "pr_auc": float(average_precision_score(labels, scores)),
        "roc_auc": float(roc_auc_score(labels, scores)),
        "action_share": share, "flagged": int(flags.sum()), "caught": int(np.sum(labels & flags)),
        **action_metrics(labels, flags),
    })
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(OUTPUT, index=False)
    print(OUTPUT)


if __name__ == "__main__":
    main()
