"""Check balanced class weights at the extended C grid, and the Phase 2 classifier's in-sample scores.

Run from the repository root:
    python src/experiments/ranking_strategy/19_class_weights_insample/run.py

Declared in src/experiments/ranking_strategy/EVALUATION_CRITERIA.md. Adds no candidates and does not use the gates. Stops unless the
unweighted fits reproduce experiment 03 and C = 0.01 reproduces Phase 2. Writes CSVs to data/outputs/experiments/ranking_strategy/19_class_weights_insample/.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score
from threadpoolctl import threadpool_limits

HERE = Path(__file__).resolve().parent
EXPERIMENTS = HERE.parent
sys.path.insert(0, str(HERE.parents[2]))

from olist.paths import EXPERIMENT_OUTPUTS  # noqa: E402

OUT = EXPERIMENT_OUTPUTS / "ranking_strategy" / HERE.name
OUT.mkdir(parents=True, exist_ok=True)
OUTPUTS = EXPERIMENT_OUTPUTS / "ranking_strategy"
sys.path.insert(0, str(EXPERIMENTS))

from task4_common import LABEL, exp03, feature_columns, load_data, make_pipeline, saved_model_scores, validation_rows  # noqa: E402

WEIGHTS = {"unweighted": None, "balanced": "balanced"}
EXPECTED = {("unweighted", 0.01): 0.392, ("balanced", 0.01): 0.385, ("unweighted", 0.0001): 0.375}
TOLERANCE = 5e-4


def window_pr_auc(data, c, weight):
    columns = sum(feature_columns(data["spec_base"]), [])
    values = []
    for fit_rows, rows in validation_rows(data["train"], data["folds"]):
        pipeline = make_pipeline("linear", {**data["params"], "C": c, "class_weight": weight}, data["spec_base"],
                                 log_features=data["spec"]["linear_log_candidates"])
        with threadpool_limits(limits=1):
            pipeline.fit(fit_rows[columns], fit_rows[LABEL])
        values.append(average_precision_score(rows[LABEL], pipeline.predict_proba(rows[columns])[:, 1]))
    return np.array(values)


def run():
    data = load_data()
    per_window, rows = {}, []
    for c in exp03.C_GRID:
        for label, weight in WEIGHTS.items():
            per_window[(label, c)] = window_pr_auc(data, c, weight)
            print(f"C = {c}, {label}: {per_window[(label, c)].mean():.4f}", flush=True)
        gaps = per_window[("balanced", c)] - per_window[("unweighted", c)]
        rows.append({"C": c, "unweighted": per_window[("unweighted", c)].mean(), "balanced": per_window[("balanced", c)].mean(),
                     "balanced_minus_unweighted": gaps.mean(), "paired_se": gaps.std(ddof=1) / np.sqrt(len(gaps)),
                     "windows_balanced_ahead": int((gaps > 0).sum())})
    table = pd.DataFrame(rows)
    for key, expected in EXPECTED.items():
        if abs(per_window[key].mean() - expected) > TOLERANCE:
            raise AssertionError(f"{key} does not reproduce the earlier result: {per_window[key].mean():.4f} against {expected}")
    print("Unweighted fits and C = 0.01 reproduce experiment 03 and Phase 2", flush=True)
    table.to_csv(OUT / "class_weights.csv", index=False)

    train = data["train"]
    score = saved_model_scores(data, train)
    insample = pd.DataFrame([{"rows": len(train), "late": int(train[LABEL].sum()),
                              "pr_auc": average_precision_score(train[LABEL], score), "roc_auc": roc_auc_score(train[LABEL], score)}])
    insample.to_csv(OUT / "insample_scores.csv", index=False)

    pd.set_option("display.width", 200)
    print(table.round(4).to_string(index=False))
    print(insample.round(4).to_string(index=False))


if __name__ == "__main__":
    run()
