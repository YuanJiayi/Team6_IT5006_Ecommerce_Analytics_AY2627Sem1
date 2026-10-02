"""Reproducible Phase 2 classification validation; no holdout evaluation or final fit.

Run from the repository: python phase2_classification.py
"""

from __future__ import annotations

import hashlib
import json
import platform
import time
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score, confusion_matrix, f1_score, matthews_corrcoef,
    precision_score, recall_score, roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier
from threadpoolctl import threadpool_limits


ROOT = Path(__file__).resolve().parent
RESULT_DIR = ROOT / "results" / "phase2" / "classification"
MODEL_FAMILIES = {"dummy": "baseline", "linear": "linear", "tree": "tree_based", "forest": "tree_based"}

# Fixed before looking at classification validation scores. Order breaks exact ties.
CANDIDATES = [
    ("prior_baseline", "dummy", {}),
    ("logistic_baseline", "linear", {"C": 1.0, "class_weight": None}),
    ("logistic_c01", "linear", {"C": 0.1, "class_weight": None}),
    ("logistic_c10", "linear", {"C": 10.0, "class_weight": None}),
    ("logistic_balanced", "linear", {"C": 1.0, "class_weight": "balanced"}),
    ("tree_baseline", "tree", {"max_depth": None, "min_samples_leaf": 1}),
    ("tree_depth6", "tree", {"max_depth": 6, "min_samples_leaf": 50}),
    ("tree_depth12", "tree", {"max_depth": 12, "min_samples_leaf": 20}),
    ("tree_balanced", "tree", {"max_depth": 12, "min_samples_leaf": 20, "class_weight": "balanced"}),
    ("forest_baseline", "forest", {"max_depth": None, "min_samples_leaf": 1}),
    ("forest_depth12", "forest", {"max_depth": 12, "min_samples_leaf": 20}),
    ("forest_leaf20", "forest", {"max_depth": None, "min_samples_leaf": 20}),
    ("forest_balanced", "forest", {"max_depth": None, "min_samples_leaf": 20, "class_weight": "balanced_subsample"}),
]


def load_primary_training(root=ROOT):
    root = Path(root)
    spec = json.loads((root / "data/phase2_feature_spec.json").read_text())
    table = pd.read_csv(
        root / "data/phase2_order_table.csv",
        parse_dates=["order_purchase_timestamp", "outcome_available_at"],
    )
    # Select the agreed training population before any statistics or fitting.
    train = table.loc[table[spec["primary_split"]].eq("train")].reset_index(drop=True)
    if not train["order_id"].is_unique:
        raise ValueError("Primary training must have unique order IDs")
    cutoff = pd.Timestamp(spec["primary_cutoff"])
    if not (train["order_purchase_timestamp"].lt(cutoff).all()
            and train["outcome_available_at"].lt(cutoff).all()):
        raise ValueError("Primary training contains an order unavailable at the cutoff")
    features = spec["numeric_features"] + spec["categorical_features"]
    forbidden = {spec["classification_target"], spec["regression_target"], "order_id",
                 "outcome_available_at", "order_purchase_timestamp", "split", "cv_fold",
                 "random_split", "random_cv_fold"}
    if len(features) != len(set(features)) or forbidden.intersection(features):
        raise ValueError("Invalid predictor contract")
    return train, spec


def temporal_folds(train, spec):
    """Return positional indices compatible with sklearn's explicit CV interface."""
    end = pd.Timestamp(spec["primary_cutoff"]) - pd.Timedelta(days=spec["validation_maturity_days"])
    folds = []
    previous_end = None
    for fold in range(spec["n_folds"]):
        valid = np.flatnonzero(train[spec["primary_fold"]].eq(fold).to_numpy())
        if not len(valid):
            raise ValueError(f"Empty validation fold {fold}")
        starts = train.iloc[valid]["order_purchase_timestamp"]
        start = starts.min()
        if starts.max() >= end or (previous_end is not None and previous_end >= start):
            raise ValueError("Validation folds overlap in time or violate the maturity gap")
        fit = np.flatnonzero((train["order_purchase_timestamp"].lt(start)
                              & train["outcome_available_at"].lt(start)).to_numpy())
        if not len(fit) or np.intersect1d(fit, valid).size:
            raise ValueError("Invalid training fold")
        if train.iloc[fit][spec["classification_target"]].nunique() != 2:
            raise ValueError("Training fold needs both classes")
        if train.iloc[valid][spec["classification_target"]].nunique() != 2:
            raise ValueError("Validation fold needs both classes for ROC-AUC")
        folds.append((fit, valid))
        previous_end = starts.max()
    return folds


def hour_components(values):
    hours = np.asarray(values, dtype=float).reshape(-1)
    angle = hours * 2 * np.pi / 24
    return np.column_stack([np.sin(angle), np.cos(angle)])


def feature_columns(spec, include_payment=True):
    omitted = set() if include_payment else set(spec["payment_sensitivity_features"])
    return ([c for c in spec["numeric_features"] if c not in omitted],
            [c for c in spec["categorical_features"] if c not in omitted])


def make_pipeline(family, params, spec, include_payment=True):
    numeric, categorical = feature_columns(spec, include_payment)
    seed = spec["random_state"]
    branches = []
    if family == "linear":
        numeric = [c for c in numeric if c != "purchase_hour"]
        branches.append(("hour", FunctionTransformer(hour_components), ["purchase_hour"]))
    numeric_steps = [("impute", SimpleImputer(strategy="median", keep_empty_features=True))]
    if family == "linear":
        numeric_steps.append(("scale", StandardScaler()))
    branches.extend([
        ("numeric", Pipeline(numeric_steps), numeric),
        ("category", OneHotEncoder(handle_unknown="ignore"), categorical),
    ])
    preparation = ColumnTransformer(branches, remainder="drop")
    if family == "dummy":
        model = DummyClassifier(strategy="prior")
    elif family == "linear":
        model = LogisticRegression(solver="lbfgs", max_iter=3000, random_state=seed, **params)
    elif family == "tree":
        model = DecisionTreeClassifier(random_state=seed, **params)
    elif family == "forest":
        model = RandomForestClassifier(n_estimators=200, max_features="sqrt", n_jobs=4,
                                       random_state=seed, **params)
    else:
        raise ValueError(f"Unknown model family: {family}")
    return Pipeline([("prepare", preparation), ("model", model)])


def metrics(y, probability, threshold=0.5):
    predicted = np.asarray(probability) >= threshold
    tn, fp, fn, tp = confusion_matrix(y, predicted, labels=[0, 1]).ravel()
    return {
        "average_precision": float(average_precision_score(y, probability)),
        "roc_auc": float(roc_auc_score(y, probability)),
        "precision": float(precision_score(y, predicted, zero_division=0)),
        "recall": float(recall_score(y, predicted, zero_division=0)),
        "f1": float(f1_score(y, predicted, zero_division=0)),
        "mcc": float(matthews_corrcoef(y, predicted)),
        "true_negative": int(tn), "false_positive": int(fp),
        "false_negative": int(fn), "true_positive": int(tp),
    }


def mcc_threshold(y, probability):
    """Maximise MCC over distinct score boundaries; ties favour fewer alerts."""
    y, probability = np.asarray(y, dtype=int), np.asarray(probability, dtype=float)
    if len(y) != len(probability) or len(np.unique(y)) != 2:
        raise ValueError("Threshold selection needs aligned predictions and both classes")
    if not np.isfinite(probability).all() or not np.isin(y, [0, 1]).all():
        raise ValueError("Invalid labels or probabilities")
    order = np.argsort(-probability, kind="stable")
    scores, labels = probability[order], y[order]
    last = np.r_[np.flatnonzero(scores[:-1] != scores[1:]), len(scores) - 1]
    tp = np.cumsum(labels, dtype=float)[last]
    fp = last + 1 - tp
    fn = labels.sum() - tp
    tn = len(labels) - labels.sum() - fp
    denominator = np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    mcc = np.divide(tp * tn - fp * fn, denominator, out=np.zeros_like(tp), where=denominator != 0)
    # Include the always-negative option, including when all probabilities equal 1.
    choices = np.r_[np.nextafter(scores[0], np.inf), scores[last]]
    return float(choices[np.argmax(np.r_[0.0, mcc])])


def evaluate_candidate(train, spec, folds, name, family, params, include_payment=True):
    rows, predictions = [], []
    numeric, categorical = feature_columns(spec, include_payment)
    X, y = train[numeric + categorical], train[spec["classification_target"]]
    started = time.monotonic()
    for fold, (fit, valid) in enumerate(folds):
        pipeline = make_pipeline(family, params, spec, include_payment)
        with threadpool_limits(limits=1):
            pipeline.fit(X.iloc[fit], y.iloc[fit])
            probability = pipeline.predict_proba(X.iloc[valid])[:, 1]
        rows.append({"candidate": name, "algorithm": family, "model_family": MODEL_FAMILIES[family],
                     "include_payment": include_payment,
                     "fold": fold, "fit_orders": len(fit), "validation_orders": len(valid),
                     "validation_late_rate": float(y.iloc[valid].mean()),
                     **metrics(y.iloc[valid], probability)})
        predictions.append(pd.DataFrame({
            "order_id": train.iloc[valid]["order_id"].to_numpy(),
            "fold": fold, "is_late": y.iloc[valid].to_numpy(), "probability": probability,
        }))
    elapsed = time.monotonic() - started
    print(f"{name}: mean validation AP={np.mean([r['average_precision'] for r in rows]):.4f}; {elapsed:.1f}s", flush=True)
    return rows, pd.concat(predictions, ignore_index=True), elapsed


def run_classification(root=ROOT):
    root = Path(root)
    output = root / "results/phase2/classification"
    output.mkdir(parents=True, exist_ok=True)
    train, spec = load_primary_training(root)
    folds = temporal_folds(train, spec)
    all_rows, predictions, settings, elapsed_times = [], {}, {}, {}
    for name, family, params in CANDIDATES:
        rows, oof, elapsed = evaluate_candidate(train, spec, folds, name, family, params)
        all_rows.extend(rows)
        predictions[name] = oof
        settings[name] = {"algorithm": family, "model_family": MODEL_FAMILIES[family],
                          "parameters": params, "include_payment": True}
        elapsed_times[name] = elapsed

    def summarize():
        return pd.DataFrame(all_rows).groupby("candidate", sort=False).agg(
            mean_average_precision=("average_precision", "mean"),
            std_average_precision=("average_precision", "std"),
            mean_roc_auc=("roc_auc", "mean"),
            mean_precision_at_05=("precision", "mean"),
            mean_recall_at_05=("recall", "mean"),
            mean_f1_at_05=("f1", "mean"),
            mean_mcc_at_05=("mcc", "mean"),
        ).sort_values("mean_average_precision", ascending=False, kind="stable")

    initial_summary = summarize()
    # A paired sensitivity analysis of each tuned model: retain parameters, drop payments.
    for family in ("linear", "tree", "forest"):
        winner = next(n for n in initial_summary.index if settings[n]["algorithm"] == family)
        name = winner + "_no_payment"
        params = settings[winner]["parameters"]
        rows, oof, elapsed = evaluate_candidate(train, spec, folds, name, family, params, False)
        all_rows.extend(rows)
        predictions[name] = oof
        settings[name] = {"algorithm": family, "model_family": MODEL_FAMILIES[family],
                          "parameters": params, "include_payment": False,
                          "paired_with": winner}
        elapsed_times[name] = elapsed

    summary = summarize()
    selected = summary.index[0]
    oof = predictions[selected]
    threshold = mcc_threshold(oof["is_late"], oof["probability"])
    threshold_rows = []
    for label, value in [("default_0.5", 0.5), ("validation_selected_mcc", threshold)]:
        threshold_rows.append({"rule": label, "threshold": value,
                               **metrics(oof["is_late"], oof["probability"], value)})
    family_winners = {
        family: next(n for n in summary.index if settings[n]["model_family"] == family)
        for family in ("linear", "tree_based")
    }
    fold_dates = []
    for fold, (fit, valid) in enumerate(folds):
        dates = train.iloc[valid]["order_purchase_timestamp"]
        fold_dates.append({"fold": fold, "fit_orders": len(fit), "validation_orders": len(valid),
                           "validation_start": str(dates.min()), "validation_end": str(dates.max()),
                           "latest_training_outcome": str(train.iloc[fit]["outcome_available_at"].max()),
                           "late_rate": float(train.iloc[valid][spec["classification_target"]].mean())})

    pd.DataFrame(all_rows).to_csv(output / "cv_metrics.csv", index=False)
    summary.assign(seconds=pd.Series(elapsed_times)).to_csv(output / "cv_summary.csv")
    oof.to_csv(output / "selected_validation_predictions.csv", index=False)
    pd.DataFrame(threshold_rows).to_csv(output / "threshold_metrics.csv", index=False)
    pd.DataFrame(fold_dates).to_csv(output / "folds.csv", index=False)
    selection = {
        "status": "validation_only_pending_review", "selection_metric": "mean fold average precision",
        "pr_auc_definition": "sklearn average_precision_score (not trapezoidal PR area)",
        "selected_candidate": selected, "selected_threshold": threshold,
        "threshold_rule": "maximise pooled validation MCC; exact ties favour fewer alerts",
        "threshold_caveat": "selected and measured on the same validation predictions; not an unbiased final score",
        "family_winners": family_winners, "candidates": settings,
        "fixed_settings": {
            "logistic": {"solver": "lbfgs", "max_iter": 3000},
            "forest": {"n_estimators": 200, "max_features": "sqrt", "n_jobs": 4},
            "numeric_imputation": "training-fold median", "categorical_encoding": "one-hot; ignore unknown",
            "linear_numeric_scaling": "StandardScaler", "linear_hour": "sine/cosine; period 24",
        },
        "random_state": spec["random_state"], "train_orders": len(train), "validation_orders": len(oof),
        "primary_test_evaluated": False, "final_model_fitted": False,
        "environment": {"python": platform.python_version(), "sklearn": sklearn.__version__,
                        "pandas": pd.__version__, "numpy": np.__version__},
        "input_sha256": {name: hashlib.sha256((root / "data" / name).read_bytes()).hexdigest()
                         for name in ("phase2_order_table.csv", "phase2_feature_spec.json")},
        "implementation_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    }
    (output / "selection.json").write_text(json.dumps(selection, indent=2) + "\n")
    print(f"Selected {selected}; validation MCC threshold {threshold:.6f}. No final model or holdout scores.", flush=True)
    return summary, pd.DataFrame(threshold_rows), selection


if __name__ == "__main__":
    run_classification()
