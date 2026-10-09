"""Compare model fitting horizons on the original chronological validation windows.

Run from the repository root:
    it5006-proj/bin/python experiments/phase2_training_window_audit.py

This changes only which already-eligible orders fit each model. Validation
orders, features, model settings, and outcome-availability rules are unchanged.
Historical route and seller features remain point-in-time values prepared for
each order; their source history is not truncated with the model fitting set.
The later test period is not scored or used for a training-window choice.
"""

from pathlib import Path
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, mean_absolute_error, roc_auc_score
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from phase2_classification import (
    feature_columns,
    load_primary_training,
    make_pipeline,
    temporal_folds,
)
from phase2_regression import make_regression_pipeline


OUT = ROOT / "experiments/phase2_training_window_audit"


def variants(fit, validation_start):
    purchase = fit.order_purchase_timestamp
    return {
        "all_eligible": fit,
        "start_2017_01": fit.loc[purchase.ge(pd.Timestamp("2017-01-01"))],
        "start_2017_03": fit.loc[purchase.ge(pd.Timestamp("2017-03-01"))],
        "trailing_365d": fit.loc[purchase.ge(validation_start - pd.Timedelta(days=365))],
        "trailing_180d": fit.loc[purchase.ge(validation_start - pd.Timedelta(days=180))],
    }


def main():
    train, spec = load_primary_training(ROOT)
    folds = temporal_folds(train, spec)
    columns = sum(feature_columns(spec), [])
    rows = []
    for window, (fit_idx, valid_idx) in enumerate(folds, start=1):
        valid = train.iloc[valid_idx]
        start = valid.order_purchase_timestamp.min()
        fit_all = train.iloc[fit_idx]
        for name, fit in variants(fit_all, start).items():
            assert len(fit) > 0 and fit.is_late.nunique() == 2
            assert fit.order_purchase_timestamp.lt(start).all()
            assert fit.outcome_available_at.lt(start).all()
            classifier = make_pipeline(
                "linear", {"C": 0.1, "class_weight": None}, spec,
                log_features=spec["linear_log_candidates"],
            )
            regressor = make_regression_pipeline("ridge", {"alpha": 100}, spec)
            with threadpool_limits(limits=1):
                classifier.fit(fit[columns], fit.is_late)
                risk = classifier.predict_proba(valid[columns])[:, 1]
                regressor.fit(fit[columns], fit.delivery_days)
                days = regressor.predict(valid[columns])
            rows.append({
                "window": window,
                "variant": name,
                "fit_orders": len(fit),
                "validation_orders": len(valid),
                "validation_late_rate": float(valid.is_late.mean()),
                "pr_auc": float(average_precision_score(valid.is_late, risk)),
                "mae_days": float(mean_absolute_error(valid.delivery_days, days)),
            })
    OUT.mkdir(exist_ok=True)
    detail = pd.DataFrame(rows)
    detail.to_csv(OUT / "validation.csv", index=False)
    summary = detail.groupby("variant", sort=False).agg(
        mean_fit_orders=("fit_orders", "mean"),
        mean_pr_auc=("pr_auc", "mean"),
        mean_mae_days=("mae_days", "mean"),
    ).reset_index()
    baseline = detail.loc[detail.variant.eq("all_eligible")].set_index("window")
    summary["pr_auc_windows_better_than_all"] = [
        int((detail.loc[detail.variant.eq(name)].set_index("window").pr_auc
             > baseline.pr_auc).sum()) for name in summary.variant
    ]
    summary["mae_windows_better_than_all"] = [
        int((detail.loc[detail.variant.eq(name)].set_index("window").mae_days
             < baseline.mae_days).sum()) for name in summary.variant
    ]
    summary.to_csv(OUT / "summary.csv", index=False)
    print(summary.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    # Compare the validation-favoured horizon with the original fit on the
    # already-open later period, including the operational approval cohort.
    # This diagnoses transfer; it does not create a fresh selection holdout.
    table = pd.read_csv(
        ROOT / "data/phase2_order_table.csv",
        parse_dates=["order_purchase_timestamp", "outcome_available_at"],
    )
    test = table.loc[table.split.eq("test")].copy()
    raw = pd.read_csv(
        ROOT / "data/olist_orders_dataset.csv",
        usecols=["order_id", "order_approved_at", "order_delivered_customer_date"],
        parse_dates=["order_approved_at", "order_delivered_customer_date"],
    )
    test = test.merge(raw, on="order_id", validate="one_to_one")
    approved = (test.order_approved_at.notna()
                & test.order_approved_at.ge(test.order_purchase_timestamp)
                & test.order_approved_at.le(test.order_delivered_customer_date))
    comparisons = []
    for name, fit in (
        ("all_eligible", train),
        ("trailing_180d", train.loc[train.order_purchase_timestamp.ge(
            pd.Timestamp(spec["primary_cutoff"]) - pd.Timedelta(days=180))]),
    ):
        classifier = make_pipeline(
            "linear", {"C": 0.1, "class_weight": None}, spec,
            log_features=spec["linear_log_candidates"],
        )
        regressor = make_regression_pipeline("ridge", {"alpha": 100}, spec)
        with threadpool_limits(limits=1):
            classifier.fit(fit[columns], fit.is_late)
            regressor.fit(fit[columns], fit.delivery_days)
            for population, score in (
                ("all_later", test), ("approved_later", test.loc[approved]),
            ):
                risk = classifier.predict_proba(score[columns])[:, 1]
                days = regressor.predict(score[columns])
                top = np.argsort(-risk, kind="stable")[:int(np.ceil(.1 * len(score)))]
                comparisons.append({
                    "variant": name,
                    "population": population,
                    "fit_orders": len(fit),
                    "orders": len(score),
                    "pr_auc": float(average_precision_score(score.is_late, risk)),
                    "roc_auc": float(roc_auc_score(score.is_late, risk)),
                    "top10_precision": float(score.is_late.iloc[top].mean()),
                    "mae_days": float(mean_absolute_error(score.delivery_days, days)),
                    "bias_days": float(np.mean(days - score.delivery_days)),
                })
    later = pd.DataFrame(comparisons)
    later.to_csv(OUT / "later_comparison.csv", index=False)
    print("\nLater comparison:\n", later.to_string(index=False, float_format=lambda x: f"{x:.4f}"))


if __name__ == "__main__":
    main()
