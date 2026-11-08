"""Refit and save the three final Phase 2 models presented in the report.

Run from the repository root:
    python src/experiments/ranking_strategy/rebuild_phase2_final_models.py

Nothing in the Phase 2 code or outputs is changed: the script imports the olist package and reads the Phase 2
data and saved results, and writes only to data/outputs/experiments/ranking_strategy/models/.

Each model is refitted once on every primary training order (outcomes known before the 26 May 2018 cutoff),
exactly as the report's test scoring does, then checked against the saved test results before it is written:
- checkout delivery-time estimate: ridge (alpha=100), checkout inputs
- handover delivery-time estimate: ridge (alpha=100), checkout inputs plus handover progress
- late-warning classifier: regularised logistic regression (C=0.01), handover inputs, slack shape and backlog

Writes the three phase2_final_*.joblib files and phase2_final_models_manifest.json to data/outputs/experiments/ranking_strategy/models/.
Load a model with joblib.load and score a frame that holds the manifest's input columns, built as in load_cohort.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.metrics import average_precision_score, roc_auc_score
from threadpoolctl import threadpool_limits

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))

from olist.paths import EXPERIMENT_OUTPUTS, REPO_ROOT  # noqa: E402

from olist.classification import feature_columns, make_pipeline  # noqa: E402
from olist.eta import CANDIDATES as ETA_CANDIDATES, PHASE2_REFERENCE, TARGET, make_stage_pipeline, stage_spec  # noqa: E402
from olist.handover_classifier import CANDIDATES as CLS_CANDIDATES, LABEL, load_cohort, variant_spec  # noqa: E402
from olist.regression import regression_metrics  # noqa: E402

ROOT = REPO_ROOT
MODEL_DIR = EXPERIMENT_OUTPUTS / "ranking_strategy" / "models"
FILE_NAMES = {"checkout": "phase2_final_checkout_delivery_days_ridge",
              "handover": "phase2_final_handover_delivery_days_ridge",
              "late_warning": "phase2_final_late_warning_logistic"}
MANIFEST = "phase2_final_models_manifest.json"
# Refits differ from the saved runs only by floating-point noise (about 1e-5); the report rounds to 0.01.
TOLERANCE = 1e-3


def lookup(candidates, name):
    return next((family, params) for n, family, params in candidates if n == name)


def fit_regression(name, spec_s, train):
    family, params = lookup(ETA_CANDIDATES, name)
    pipeline = make_stage_pipeline(family, params, spec_s)
    with threadpool_limits(limits=1):
        pipeline.fit(train[sum(feature_columns(spec_s), [])], train[TARGET])
    return pipeline


def fit_classifier(name, spec_v, train, log_features):
    family, params = lookup(CLS_CANDIDATES, name)
    pipeline = make_pipeline(family, params, spec_v, log_features=log_features)
    with threadpool_limits(limits=1):
        pipeline.fit(train[sum(feature_columns(spec_v), [])], train[LABEL])
    return pipeline


def check(label, got, expected):
    gap = abs(got - expected)
    print(f"{label}: {got:.6f} (report {expected:.6f})", flush=True)
    if gap > TOLERANCE:
        raise AssertionError(f"{label} differs from the saved result by {gap:.2e}")


def run(root=ROOT, out_dir=MODEL_DIR):
    root, out_dir = Path(root), Path(out_dir)
    table, spec, _ = load_cohort(root)
    train = table[table[spec["primary_split"]].eq("train")].reset_index(drop=True)
    test = table[table[spec["primary_split"]].eq("test")].reset_index(drop=True)
    cls_selection = json.loads((root / "data/outputs/phase2/handover/selection.json").read_text())
    saved_eta = pd.read_csv(root / "data/outputs/phase2/eta/test_metrics.csv")
    saved_scores = pd.read_csv(root / "data/outputs/phase2/handover/test_scores.csv")

    specs = {"checkout": stage_spec(spec, "checkout"), "handover": stage_spec(spec, "handover"),
             "late_warning": variant_spec(spec)}
    # The Phase 2 regression selection (ridge alpha=100); olist.eta now selects alpha=1000 for Phase 3.
    chosen = {"checkout": PHASE2_REFERENCE,
              "handover": PHASE2_REFERENCE,
              "late_warning": cls_selection["variants"]["primary"]["selected"]}
    log_features = spec["linear_log_candidates"]
    models = {stage: fit_regression(chosen[stage], specs[stage], train) for stage in ("checkout", "handover")}
    models["late_warning"] = fit_classifier(chosen["late_warning"], specs["late_warning"], train, log_features)

    # The refits must reproduce the report's test results before anything is saved.
    for stage in ("checkout", "handover"):
        predicted = models[stage].predict(test[sum(feature_columns(specs[stage]), [])])
        metrics = regression_metrics(test[TARGET], predicted)
        row = saved_eta.query("stage == @stage and candidate == @chosen[@stage] and scope == 'overall'").iloc[0]
        for metric in ("mae", "rmse", "r2", "mean_signed_error"):
            check(f"{stage} test {metric}", metrics[metric], row[metric])
    scores = models["late_warning"].predict_proba(test[sum(feature_columns(specs["late_warning"]), [])])[:, 1]
    merged = saved_scores.merge(pd.DataFrame({"order_id": test["order_id"], "refit": scores}), on="order_id",
                                validate="one_to_one")
    if len(merged) != len(test):
        raise AssertionError("Refit test orders do not match the saved late-warning scores")
    check("late-warning max score difference", float(np.abs(merged["refit"] - merged["score"]).max()), 0.0)
    y = test[LABEL].to_numpy()
    print(f"late-warning test PR-AUC {average_precision_score(y, scores):.3f}, "
          f"ROC-AUC {roc_auc_score(y, scores):.3f}", flush=True)

    out_dir.mkdir(parents=True, exist_ok=True)
    manifest = {"fitted_on": f"{len(train)} primary training orders (outcomes recorded before the test cutoff)",
                "verified_against": "data/outputs/phase2/eta/test_metrics.csv and data/outputs/phase2/handover/test_scores.csv",
                "scikit_learn": sklearn.__version__, "random_state": spec["random_state"], "models": {}}
    for stage, model in models.items():
        numeric, categorical = feature_columns(specs[stage])
        path = out_dir / f"{FILE_NAMES[stage]}.joblib"
        joblib.dump(model, path)
        manifest["models"][stage] = {
            "file": path.name, "candidate": chosen[stage],
            "output": "predicted purchase-to-delivery days" if stage != "late_warning"
            else "probability of arriving after the promised calendar date (predict_proba[:, 1])",
            "numeric_inputs": numeric, "categorical_inputs": categorical}
    (out_dir / MANIFEST).write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"Saved {', '.join(models)} to {out_dir}", flush=True)
    return models, manifest


if __name__ == "__main__":
    run()
