"""Exploratory XGBoost comparison on the frozen Phase 2 temporal folds.

This script reads the prepared table and saved reference metrics. It never fits
or scores the reserved later-period test orders. Run from the repository root:

    it5006-proj/bin/python experiments/phase2_xgboost_validation.py
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn
import xgboost
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from phase2_classification import load_primary_training, make_pipeline, temporal_folds  # noqa: E402


# Declared before examining any XGBoost scores. All use the same final 20 inputs,
# fold-fitted preprocessing, natural class frequencies, and seed as the references.
SETTINGS = {
    "xgb_shallow": dict(n_estimators=300, max_depth=2, learning_rate=0.05,
                        min_child_weight=20, subsample=0.8, colsample_bytree=0.8,
                        reg_lambda=10),
    "xgb_moderate": dict(n_estimators=300, max_depth=3, learning_rate=0.05,
                         min_child_weight=10, subsample=0.8, colsample_bytree=0.8,
                         reg_lambda=5),
    "xgb_deeper": dict(n_estimators=400, max_depth=4, learning_rate=0.03,
                       min_child_weight=10, subsample=0.8, colsample_bytree=0.8,
                       reg_lambda=5),
}


def main() -> None:
    train, spec = load_primary_training(ROOT)
    folds = temporal_folds(train, spec)
    columns = spec["numeric_features"] + spec["categorical_features"]
    X = train[columns]
    y = train[spec["classification_target"]]
    rows = []
    started = time.monotonic()
    for name, settings in SETTINGS.items():
        for fold_number, (fit, valid) in enumerate(folds):
            prepare = make_pipeline("forest", {"max_depth": 12, "min_samples_leaf": 20},
                                    spec).named_steps["prepare"]
            model = XGBClassifier(
                objective="binary:logistic", eval_metric="logloss", tree_method="hist",
                random_state=spec["random_state"], n_jobs=4, **settings,
            )
            pipeline = Pipeline([("prepare", prepare), ("model", model)])
            pipeline.fit(X.iloc[fit], y.iloc[fit])
            probability = pipeline.predict_proba(X.iloc[valid])[:, 1]
            row = {
                "candidate": name, "fold": fold_number, "fit_orders": len(fit),
                "validation_orders": len(valid),
                "validation_late_rate": float(y.iloc[valid].mean()),
                "average_precision": float(average_precision_score(y.iloc[valid], probability)),
                "roc_auc": float(roc_auc_score(y.iloc[valid], probability)),
            }
            rows.append(row)
            print(f"{name} fold {fold_number}: AP={row['average_precision']:.4f}, "
                  f"ROC-AUC={row['roc_auc']:.4f}", flush=True)

    frame = pd.DataFrame(rows)
    reference = pd.read_csv(ROOT / "results/phase2/classification/cv_metrics.csv")
    reference = reference[(reference["stage"] == "final_grid")
                          & reference["candidate"].isin(["logistic_c01", "forest_depth12"])]
    comparison = pd.concat([reference[frame.columns], frame], ignore_index=True)
    summary = (comparison.groupby("candidate", sort=False)
               .agg(mean_ap=("average_precision", "mean"),
                    mean_roc_auc=("roc_auc", "mean"),
                    min_ap=("average_precision", "min"),
                    max_ap=("average_precision", "max"))
               .sort_values("mean_ap", ascending=False))
    print("\nFive-window comparison (mean of per-window scores):")
    print(summary.round(4).to_string(), flush=True)
    logistic = comparison[comparison["candidate"] == "logistic_c01"].sort_values("fold")
    for name in SETTINGS:
        candidate = frame[frame["candidate"] == name].sort_values("fold")
        delta = (candidate["average_precision"].to_numpy()
                 - logistic["average_precision"].to_numpy())
        print(f"{name} minus logistic AP by fold: {np.round(delta, 4).tolist()}; "
              f"wins {int((delta > 0).sum())}/5", flush=True)

    output = ROOT / "experiments/phase2_xgboost_validation.json"
    payload = {
        "scope": "exploratory training-period temporal validation only; no test rows",
        "settings": SETTINGS,
        "rows": rows,
        "summary": summary.reset_index().to_dict(orient="records"),
        "versions": {"python": sys.version.split()[0], "sklearn": sklearn.__version__,
                     "xgboost": xgboost.__version__},
        "runtime_seconds": round(time.monotonic() - started, 2),
    }
    output.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"Saved {output} ({payload['runtime_seconds']:.1f}s)", flush=True)


if __name__ == "__main__":
    main()
