"""Save the selected Phase 2 models as deployable pipelines, then verify the saved files.

Run from the repository root:
    python -m olist.export_models   (with src/ on PYTHONPATH or after `pip install -e .`)

Step 7, save: each selected model is refitted on all primary training orders, exactly as for the reported test
scores, and saved with joblib as one sklearn Pipeline (preprocessing + estimator), so a caller passes raw input
columns and needs no separate encoder. model_metadata.json records what a caller needs to call each pipeline and the
test results it should reproduce; feature_stats.json records training distributions for drift checks.

Step 8, verify: every file is reloaded as a fresh object. Its test predictions must equal those of the model that
was saved, exactly, and its test metrics must match the saved Phase 2 outputs at report precision. (The saved ETA
test metrics predate the current environment and differ from a refit in the fourth to sixth decimal; main reproduces
the same refit values, so the gap is not from the export.)

The pipelines call functions from olist (for example the purchase-hour encoding), so loading them needs this
package importable and the same scikit-learn version that saved them.
"""

from __future__ import annotations

import hashlib
import json
import platform
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.metrics import average_precision_score, roc_auc_score
from threadpoolctl import threadpool_limits

from olist.classification import feature_columns, make_pipeline
from olist.eta import CANDIDATES as ETA_CANDIDATES, TARGET, make_stage_pipeline, stage_spec
from olist.handover_classifier import CANDIDATES as CLASSIFIER_CANDIDATES, LABEL, load_cohort, monthly_flags, variant_spec
from olist.paths import MODELS_DIR, PHASE2_OUTPUTS, PROCESSED_DIR, REPO_ROOT
from olist.regression import regression_metrics

LINEAR_FAMILIES = ("linear", "ridge")
REPORT_TOLERANCE = 1e-3  # report tables round to 2-3 decimals


def input_columns(spec):
    numeric, categorical = feature_columns(spec)
    return numeric + categorical


def category_classes(pipeline, spec):
    """Categories learned by the fitted one-hot encoder; anything else is ignored at prediction time."""
    encoder = pipeline.named_steps["prepare"].named_transformers_["category"]
    _, categorical = feature_columns(spec)
    return {column: [str(c) for c in classes] for column, classes in zip(categorical, encoder.categories_)}


def fit(pipeline, family, rows, columns, target):
    with threadpool_limits(limits=1 if family in LINEAR_FAMILIES else None):
        return pipeline.fit(rows[columns], rows[target])


def json_ready(value):
    """Plain JSON values: NaN becomes null, numpy and pandas scalars become Python values."""
    if isinstance(value, dict):
        return {str(k): json_ready(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_ready(v) for v in value]
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if isinstance(value, (np.integer, np.bool_)):
        return value.item()
    if isinstance(value, (float, np.floating)):
        return None if np.isnan(value) else float(value)
    return value


def build_models(table, spec):
    """Fit the selected ETA (checkout, handover) and late-risk models on all primary training orders."""
    train = table[table[spec["primary_split"]].eq("train")].reset_index(drop=True)
    eta_selection = json.loads((PHASE2_OUTPUTS / "eta/selection.json").read_text())
    classifier_selection = json.loads((PHASE2_OUTPUTS / "handover/selection.json").read_text())["variants"]["primary"]
    eta_lookup = {name: (family, params) for name, family, params in ETA_CANDIDATES}
    classifier_lookup = {name: (family, params) for name, family, params in CLASSIFIER_CANDIDATES}

    models = {}
    for stage, candidate in (("checkout", eta_selection["checkout_selected_candidate"]),
                             ("handover", eta_selection["selected_candidate"])):
        family, params = eta_lookup[candidate]
        spec_s = stage_spec(spec, stage)
        columns = input_columns(spec_s)
        pipeline = fit(make_stage_pipeline(family, params, spec_s), family, train, columns, TARGET)
        models[f"eta_{stage}"] = {
            "pipeline": pipeline, "spec": spec_s, "columns": columns, "candidate": candidate, "family": family,
            "params": params, "task": "regression", "target": TARGET, "stage": stage,
            "output": "predicted days from purchase to customer delivery (pipeline.predict)",
            "selection_reason": eta_selection["reason"][stage],
        }

    candidate = classifier_selection["selected"]
    family, params = classifier_lookup[candidate]
    spec_v = variant_spec(spec)
    columns = input_columns(spec_v)
    pipeline = make_pipeline(family, params, spec_v, log_features=spec["linear_log_candidates"])
    models["late_risk_handover"] = {
        "pipeline": fit(pipeline, family, train, columns, LABEL), "spec": spec_v, "columns": columns,
        "candidate": candidate, "family": family, "params": params, "task": "classification", "target": LABEL,
        "stage": "handover",
        "output": "probability the order is delivered after its promised date (pipeline.predict_proba[:, 1])",
        "selection_reason": classifier_selection["reason"],
        "operating_point": {
            "review_share": classifier_selection["k"],
            "rule": "within each purchase month, flag the highest-scoring share of orders for review",
        },
    }
    return models, train


def model_metadata(name, model, train, test, metrics):
    numeric, categorical = feature_columns(model["spec"])
    target = train[model["target"]]
    info = {key: model[key] for key in ("task", "stage", "target", "candidate", "family", "params", "output",
                                        "selection_reason")}
    info.update({
        "file": f"{name}.pkl",
        "prediction_point": (model["spec"]["prediction_point"] if model["stage"] == "checkout"
                             else "when the carrier records receiving the parcel from the seller"),
        "feature_columns": model["columns"], "numeric_columns": numeric, "categorical_columns": categorical,
        "category_classes": category_classes(model["pipeline"], model["spec"]),
        "pipeline_steps": list(model["pipeline"].named_steps),
        "training_orders": len(train), "test_orders": len(test),
        "training_purchase_window": [train["order_purchase_timestamp"].min(), train["order_purchase_timestamp"].max()],
        "test_purchase_window": [test["order_purchase_timestamp"].min(), test["order_purchase_timestamp"].max()],
        "target_mean_train": float(target.mean()), "target_mean_test": float(test[model["target"]].mean()),
        "test_metrics": metrics,
        "example_input": test.loc[0, model["columns"]].to_dict(),
    })
    if "operating_point" in model:
        info["operating_point"] = model["operating_point"]
    return info


def feature_statistics(train, models):
    """Training distribution of every numeric input, before preprocessing, for drift checks."""
    numeric = list(dict.fromkeys(c for m in models.values() for c in feature_columns(m["spec"])[0]))
    stats = {}
    for column in numeric:
        described = train[column].describe().to_dict()
        described["missing_share"] = float(train[column].isna().mean())
        stats[column] = described
    return stats


# ------------------------------------------------------------------------------------- verification

def saved_eta_metrics(stage, candidate):
    saved = pd.read_csv(PHASE2_OUTPUTS / "eta/test_metrics.csv")
    row = saved[saved["stage"].eq(stage) & saved["candidate"].eq(candidate) & saved["scope"].eq("overall")]
    return row.iloc[0][["mae", "rmse", "r2", "mean_signed_error"]].astype(float).to_dict()


def verify(models, test, metadata):
    """Reload every saved file; check it predicts exactly as the fitted model and matches the saved Phase 2 results."""
    checks = []
    for name, model in models.items():
        loaded = joblib.load(MODELS_DIR / f"{name}.pkl")
        columns = metadata[name]["feature_columns"]
        X = test[columns]
        if model["task"] == "regression":
            predicted = loaded.predict(X)
            assert np.array_equal(predicted, model["pipeline"].predict(X)), f"{name}: reloaded predictions differ"
            got = regression_metrics(test[TARGET], predicted)
            expected = saved_eta_metrics(model["stage"], model["candidate"])
            gap = max(abs(got[key] - value) for key, value in expected.items())
            assert gap <= REPORT_TOLERANCE, f"{name}: test metrics differ from the saved results by {gap}"
            checks.append(f"{name}: reload exact; test MAE {got['mae']:.4f} days "
                          f"(saved {expected['mae']:.4f}; largest metric gap {gap:.1e})")
        else:
            scores = loaded.predict_proba(X)[:, 1]
            assert np.array_equal(scores, model["pipeline"].predict_proba(X)[:, 1]), f"{name}: reloaded scores differ"
            saved = pd.read_csv(PHASE2_OUTPUTS / "handover/test_scores.csv")
            aligned = test[["order_id"]].assign(score=scores).merge(saved, on="order_id", validate="one_to_one",
                                                                     suffixes=("", "_saved"))
            assert len(aligned) == len(test), f"{name}: saved scores cover {len(aligned)} of {len(test)} orders"
            score_gap = float(np.max(np.abs(aligned["score"] - aligned["score_saved"])))
            saved_row = pd.read_csv(PHASE2_OUTPUTS / "handover/test_metrics.csv").query(
                "variant == 'primary' and ranking == 'model'").iloc[0]
            pr_auc = average_precision_score(test[LABEL], scores)
            gap = max(score_gap, abs(pr_auc - saved_row["pr_auc"]),
                      abs(roc_auc_score(test[LABEL], scores) - saved_row["roc_auc"]))
            assert gap <= REPORT_TOLERANCE, f"{name}: scores or metrics differ from the saved results by {gap}"
            checks.append(f"{name}: reload exact; PR-AUC {pr_auc:.4f} (saved {saved_row['pr_auc']:.4f}); "
                          f"largest per-order score gap vs saved {score_gap:.1e}")
    return checks


def run():
    table, spec, coverage = load_cohort(REPO_ROOT)
    test = table[table[spec["primary_split"]].eq("test")].reset_index(drop=True)
    models, train = build_models(table, spec)

    # Step 7: save pipelines and metadata.
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    metadata = {}
    for name, model in models.items():
        joblib.dump(model["pipeline"], MODELS_DIR / f"{name}.pkl")
        columns = model["columns"]
        if model["task"] == "regression":
            metrics = regression_metrics(test[TARGET], model["pipeline"].predict(test[columns]))
        else:
            scores = model["pipeline"].predict_proba(test[columns])[:, 1]
            months = test["order_purchase_timestamp"].dt.to_period("M").astype(str).to_numpy()
            flags = monthly_flags(scores, months, model["operating_point"]["review_share"])
            y = test[LABEL].to_numpy().astype(bool)
            metrics = {"pr_auc": float(average_precision_score(y, scores)), "roc_auc": float(roc_auc_score(y, scores)),
                       "late_orders": int(y.sum()), "flagged": int(flags.sum()), "late_caught": int((y & flags).sum()),
                       "recall_at_operating_point": float((y & flags).sum() / y.sum())}
        metadata[name] = model_metadata(name, model, train, test, metrics)

    inputs = {name: hashlib.sha256((PROCESSED_DIR / name).read_bytes()).hexdigest()
              for name in ("phase2_order_table.csv", "phase2_feature_spec.json")}
    document = {
        "models": metadata,
        "cohort": coverage | {"train_orders": len(train), "test_orders": len(test)},
        "split": {"column": spec["primary_split"], "cutoff": spec["primary_cutoff"]},
        "environment": {"python": platform.python_version(), "sklearn": sklearn.__version__,
                        "pandas": pd.__version__, "numpy": np.__version__, "joblib": joblib.__version__},
        "requires": "olist package importable (pip install -e . from the repository root) and the same sklearn version",
        "input_sha256": inputs,
        "export_script": "src/olist/export_models.py",
    }
    (MODELS_DIR / "model_metadata.json").write_text(json.dumps(json_ready(document), indent=2) + "\n")
    stats = feature_statistics(train, models)
    (MODELS_DIR / "feature_stats.json").write_text(json.dumps(json_ready(stats), indent=2) + "\n")

    # Step 8: reload every file as a fresh start and check it reproduces the saved test results.
    checks = verify(models, test, metadata)
    print("Saved to", MODELS_DIR.relative_to(REPO_ROOT))
    for path in sorted(MODELS_DIR.iterdir()):
        print(f"  {path.name:<28} {path.stat().st_size / 1024:8.1f} KB")
    print("Verification:")
    for line in checks:
        print("  [OK]", line)


if __name__ == "__main__":
    run()
