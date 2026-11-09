"""Rebuild the local MLflow store and register the three Phase 2 selected models.

Run from the repository root:
    python -m phase3.registry [--store DIR] [--results DIR]

The store is a SQLite tracking database plus a local artifact directory. It is
deleted and rebuilt on every run through a local MLflow server, so it records
portable ``mlflow-artifacts:/`` locations instead of machine paths. Every
candidate run refits the Phase 2 model
on each chronological validation window while its run is open, checks the
rebuilt scores against the saved Phase 2 outputs, and logs the real fit times.
The final run refits the selected candidate on the Phase 2 training rows and
checks its test metrics against the saved ones.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import sqlite3
import tempfile
import time
from pathlib import Path

os.environ["MLFLOW_DISABLE_AGENT_HINT"] = "1"

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from mlflow.models import infer_signature
from mlflow.tracking import MlflowClient
from sklearn.inspection import partial_dependence
from sklearn.metrics import average_precision_score, check_scoring, roc_auc_score
from sklearn.utils import check_random_state
from threadpoolctl import threadpool_limits

from olist.classification import feature_columns, make_pipeline
from olist.eta import CANDIDATES as ETA_CANDIDATES
from olist.eta import SIMPLICITY as ETA_SIMPLICITY
from olist.eta import (TARGET, chronological_folds, fit_predict, load_stage_table, make_stage_pipeline,
                       stage_spec)
from olist.eta import select_candidate as select_eta_candidate
from olist.handover_classifier import CANDIDATES as HANDOVER_CANDIDATES
from olist.handover_classifier import (LABEL, TOP_SHARE, action_metrics, baseline_scores,
                                       fit_scores, monthly_flags, ranking_metrics, variant_spec)
from olist.handover_classifier import select_candidate as select_handover_candidate
from olist.paths import DASHBOARD_PHASE2, MLFLOW_STORE, PHASE3_OUTPUTS, REPO_ROOT
from olist.regression import regression_metrics

from phase3 import drift as drift_module
from phase3 import queue as queue_module
from phase3.explain import input_label
from phase3.mlflow_server import mlflow_server

ROOT = REPO_ROOT

# B's classifier grid plus the extra logistic C values our Phase 2 selection used.
CLASSIFIER_CANDIDATES = list(HANDOVER_CANDIDATES) + [
    (f"logistic_c{c:g}{'_bal' if weight else ''}", "linear", {"C": c, "class_weight": weight})
    for c in (0.001, 0.0001, 0.0003, 0.003) for weight in (None, "balanced")
]

DEFAULT_STORE = MLFLOW_STORE
DEFAULT_RESULTS = PHASE3_OUTPUTS / "registry"
ETA_RESULTS = DASHBOARD_PHASE2 / "eta"
HANDOVER_RESULTS = DASHBOARD_PHASE2 / "handover"
FINAL_RESULTS = DASHBOARD_PHASE2 / "final"
ETA_SELECTION = ETA_RESULTS / "selection.json"
CLASSIFIER_SELECTION = HANDOVER_RESULTS / "selection.json"
LATE_RISK_FIRST_REFIT = pd.Timestamp("2018-05-01")
LATE_RISK_LAST_REFIT = pd.Timestamp("2018-10-31")

MODELS = ("checkout_eta", "handover_eta", "late_risk")
REGRESSION_FIELDS = ("mae", "rmse", "r2", "mean_signed_error")
REGRESSION_LADDER_FIELDS = ("mae", "rmse", "r2")
CLASSIFIER_FIELDS = ("precision", "recall", "f1", "mcc", "roc_auc", "pr_auc")
CLASSIFIER_LADDER_FIELDS = ("pr_auc", "roc_auc", "mcc", "precision", "recall", "f1")
LADDER_FAMILIES = ("Linear", "Tree")
METRIC_TOLERANCE = 1e-6  # random-forest refits differ from saved scores by ~1e-8 (float noise); real changes are far larger
BOOTSTRAP_RESAMPLES = 2000
BOOTSTRAP_SEED = 42
NOTE_TAG = "mlflow.note.content"
ACTION_FIELDS = ("precision", "recall", "f1", "mcc")
TEST_GAP_METHOD = (
    "pooled test gap; 95% interval = 2.5th-97.5th percentile of 2,000 paired bootstrap "
    "resamples of the 15 purchase weeks (seed 42); a gap counts as real only when the "
    "interval excludes zero; otherwise level. Model selection on validation still uses "
    "the one-SE rule."
)
SKOPS_TRUSTED_TYPES = ("numpy.dtype", "olist.classification.hour_components",
                       "olist.classification.hour_names")
SLACK_RULE_DEFINITION = (
    "rank by promised_days - handover_days - route_transit_90d_at_handover "
    "(median route transit from the fitting rows fills missing values); review the top 10% per purchase month"
)
DEADLINE_RULE_DEFINITION = (
    "rank by promised_days - handover_days (fewest days left before the promise first); "
    "review the top 10% per purchase month"
)
BLEND_RULE_DEFINITION = (
    "50/50 mean of the model's risk percentile and the remaining-slack percentile, "
    "both scored against the fitting orders' distributions (Winnie's Task 4 method)"
)
RANDOM_RULE_DEFINITION = "a random order; PR-AUC equals the late rate"
MEAN_BASELINE_DEFINITION = "predict the training mean of delivery_days for every order"
CHECKOUT_ROUTE_DEFINITION = (
    "route_typical_days: average purchase-to-delivery days on this seller-state \u2192 customer-state route, "
    "from all orders delivered before this purchase, shrunk toward the overall average (20 pseudo-orders); "
    "the fitting rows' route median fills missing values"
)
ETA_BENCHMARKS = {
    "checkout": {
        "name": "route_typical_days",
        "label": "Route's average so far",
        "definition": CHECKOUT_ROUTE_DEFINITION,
        "feature": "route_typical_days",
    },
    "handover": {
        "name": "elapsed_plus_route_baseline",
        "label": "Days so far + route time",
        "definition": ("handover_days + route_transit_90d_at_handover "
                       "(median route transit from the fitting rows fills missing values)"),
    },
}
# Plain wording for every reference rule. The page reads these from MLflow so the
# short and hover text live with the numbers instead of in the front end.
REFERENCE_WORDING = {
    "route_typical_days": {
        "label": "Route's average so far",
        "short": "Operations rule: the route's average delivery time.",
        "hover": ("Average purchase-to-delivery days on this route (seller's state \u2192 customer's state), "
                  "from all earlier delivered orders."),
        "role": "headline",
    },
    "elapsed_plus_route_baseline": {
        "label": "Days so far + route time",
        "short": "Operations rule: days so far + the route's average carrier time.",
        "hover": ("Days to reach the carrier, plus the route's average carrier-to-customer days "
                  "over the last 90 days."),
        "role": "headline",
    },
    "mean_baseline": {
        "label": "Predict the average for everyone",
        "short": "Predict the average for everyone",
        "hover": ("Predict the training-average delivery time for every order and ignore the route "
                  "and order details."),
        "role": "floor",
    },
    "deadline_rule": {
        "label": "Closest to deadline",
        "short": "Operations rule: parcels with the fewest days left go first.",
        "hover": "Promised days minus days already elapsed; fewest first.",
        "role": "context",
    },
    "slack_rule": {
        "label": "Remaining slack",
        "short": "Operations rule: parcels with the least remaining slack go first.",
        "hover": ("Remaining slack at handover = days left before the promised date at handover, "
                  "minus the route's usual carrier time (last 90 days)."),
        "role": "headline",
    },
    "blend_rule": {
        "label": "Blend, monthly refits",
        "short": "50/50 blend of the model and remaining slack, refitted monthly (what the Actions list uses).",
        "hover": ("Half the model's risk percentile plus half the remaining-slack percentile; "
                  "this is the order the Actions list uses."),
        "role": "context",
    },
    "random_rule": {
        "label": "Random order",
        "short": "Random order",
        "hover": "Shuffle the parcels; a random order scores PR-AUC equal to the late rate.",
        "role": "floor",
    },
}


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def reset_store(store):
    store = Path(store)
    if store.exists():
        try:
            shutil.rmtree(store)
        except OSError as exc:
            raise RuntimeError(f"Could not delete the MLflow store at {store}: {exc}") from exc
    (store / "artifacts").mkdir(parents=True)
    return store


def configure_mlflow(tracking_uri):
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_registry_uri(tracking_uri)
    return MlflowClient(tracking_uri=tracking_uri, registry_uri=tracking_uri)


def experiment_id(client, name):
    experiment = client.get_experiment_by_name(name)
    if experiment is not None:
        return experiment.experiment_id
    # Without this tag MLflow 3 opens the experiment in its GenAI evaluation view.
    return client.create_experiment(name,
                                    tags={"mlflow.experimentKind": "custom_model_development"})


# An absolute POSIX path, but not the ":/" inside a URI such as mlflow-artifacts:/0.
_ABSOLUTE_PATH = re.compile(r"(?<![\w:/+.-])/(?!/)")
_LOCATION_COLUMNS = (
    ("experiments", "artifact_location"),
    ("runs", "artifact_uri"),
    ("model_versions", "storage_location"),
    ("logged_models", "artifact_location"),
)


def _location_problem(value):
    if value is None:
        return None
    value = str(value)
    if value.startswith("file:"):
        return "is a file:// location"
    if _ABSOLUTE_PATH.search(value):
        return "contains an absolute path"
    return None


def check_store_location(store):
    """Fail if the store records any machine-specific location.

    MLflow only writes portable ``mlflow-artifacts:/...`` URIs when artifacts go
    through its HTTP server. A ``file:`` URI or an absolute path means the store
    would break as soon as the repository is cloned elsewhere.
    """
    store = Path(store)
    database = store / "mlflow.db"
    if not database.is_file():
        raise RuntimeError(f"The MLflow store has no database at {database}")
    problems = []
    connection = sqlite3.connect(database)
    try:
        tables = {row[0] for row in connection.execute(
            "SELECT name FROM sqlite_master WHERE type = 'table'")}
        for table, column in _LOCATION_COLUMNS:
            if table not in tables:
                continue
            for row in connection.execute(f"SELECT {column} FROM {table}"):
                problem = _location_problem(row[0])
                if problem:
                    problems.append(f"{table}.{column} {problem}: {row[0]}")
                    break
    finally:
        connection.close()
    artifacts = store / "artifacts"
    if artifacts.is_dir():
        for path in sorted(artifacts.rglob("MLmodel")):
            problem = _location_problem(path.read_text(encoding="utf-8"))
            if problem:
                problems.append(f"{path.relative_to(store)} {problem}")
    if problems:
        raise RuntimeError(
            "The MLflow store at {} records machine-specific locations:\n  {}".format(
                store, "\n  ".join(problems)))


def _param_value(value):
    if value is None:
        return "None"
    if isinstance(value, (dict, list, tuple)):
        return json.dumps(value, sort_keys=True)
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    return value


def _log_params(params):
    mlflow.log_params({key: _param_value(value) for key, value in params.items()})


def _top_share_suffix(share):
    """Name suffix that puts a threshold-dependent metric's operating point in its name."""
    return f"_top{float(share) * 100:g}pct"


def _display_field(field, share):
    if field in ACTION_FIELDS:
        return f"{field}{_top_share_suffix(share)}"
    return field


def _plain_number(value):
    if isinstance(value, (int, float, np.integer, np.floating)):
        return f"{float(value):g}"
    return str(value)


def _candidate_note(definition, windows):
    """One plain sentence describing a candidate run for MLflow's run page."""
    family = definition.get("family")
    extras = []
    if definition.get("class_weight") not in (None, "", "None"):
        extras.append(f"class weights = {definition['class_weight']}")
    if family == "mean":
        model = "training-mean baseline (it ignores every order input)"
    elif family == "ridge":
        model = f"ridge regression, alpha = {_plain_number(definition.get('alpha'))}"
    elif family == "linear" and "C" in definition:
        penalty = definition.get("C")
        if isinstance(penalty, (int, float, np.integer, np.floating)) and np.isinf(float(penalty)):
            model = "logistic regression, no penalty"
        else:
            model = f"logistic regression, C = {_plain_number(penalty)}"
    elif family == "linear":
        model = "linear regression"
    elif family == "tree":
        depth = definition.get("max_depth")
        leaf = definition.get("min_samples_leaf")
        model = "decision tree"
        if depth not in (None, "", "None"):
            model += f", max depth = {depth}"
        if leaf not in (None, ""):
            model += f", min samples per leaf = {leaf}"
    elif family == "forest":
        model = f"random forest, {definition.get('n_estimators', 'many')} trees"
        if definition.get("max_features") not in (None, ""):
            model += f", max features = {definition['max_features']}"
        leaf = definition.get("min_samples_leaf")
        if leaf not in (None, ""):
            model += f", min samples per leaf = {leaf}"
    elif family == "boost":
        model = "gradient boosting"
        if definition.get("learning_rate") not in (None, ""):
            model += f", learning rate = {_plain_number(definition['learning_rate'])}"
        if definition.get("n_estimators") not in (None, ""):
            model += f", {definition['n_estimators']} trees"
        if definition.get("max_iter") not in (None, ""):
            model += f", {definition['max_iter']} boosting rounds"
        if definition.get("min_samples_leaf") not in (None, ""):
            model += f", min samples per leaf = {definition['min_samples_leaf']}"
    else:
        model = str(family or "model")
    if extras:
        model += " (" + ", ".join(extras) + ")"
    return f"Candidate: {model}, scored on {windows} time-ordered validation windows."


def _eta_definition(family, params, spec):
    definition = {"family": family, **params}
    if family in ("linear", "ridge"):
        definition.update({
            "numeric_imputer": "median", "numeric_scaler": "standard",
            "categorical_encoder": "one_hot", "purchase_hour": "sin_cos",
            "log_features": ",".join(spec["linear_log_candidates"]),
        })
    if family in ("ridge", "tree", "forest", "boost"):
        definition.setdefault("random_state", spec["random_state"])
    if family == "forest":
        definition.setdefault("n_estimators", 200)
        definition.setdefault("max_features", "sqrt")
        definition.setdefault("n_jobs", 1)
    if family == "boost":
        definition.update({"numeric_imputer": "median", "categorical_encoder": "ordinal"})
    if family == "mean":
        definition["strategy"] = "mean"
    return definition


def _classifier_definition(family, params, spec):
    definition = {"family": family, **params}
    if family == "linear":
        definition.update({
            "solver": "lbfgs", "max_iter": 3000, "numeric_imputer": "median",
            "numeric_scaler": "standard", "categorical_encoder": "one_hot",
            "purchase_hour": "sin_cos", "log_features": ",".join(spec["linear_log_candidates"]),
        })
    if family in ("linear", "tree", "forest", "boost"):
        definition.setdefault("random_state", spec["random_state"])
    if family == "forest":
        definition.setdefault("n_estimators", 200)
        definition.setdefault("max_features", "sqrt")
        definition.setdefault("n_jobs", 1)
    if family == "boost":
        definition.update({"early_stopping": False, "numeric_imputer": "median",
                           "categorical_encoder": "ordinal"})
    return definition


def _candidate_lookup(definitions):
    return {name: (family, params) for name, family, params in definitions}


def _stage2_map(rows):
    return {row["candidate"]: row for row in rows}


def _eta_ladders(selection, stage, stage2):
    linear = ["linear_baseline"]
    selected = selection["selected_model"][stage]
    if selected["family"] != "ridge":
        raise ValueError("The ETA linear ladder expects the selected model to be a ridge")
    linear.append(selected["candidate"])
    boosts = ["boost_baseline", "boost_slow"]
    boost = min(boosts, key=lambda name: stage2[name]["mean"])
    tree = ["tree_baseline", "tree_depth6", "forest_leaf20", boost]
    return {"Linear": linear, "Tree": tree}


def _classifier_ladders(selection):
    primary = selection["variants"]["primary"]
    selected = primary["selected_model"]
    if selected["family"] != "linear":
        raise ValueError("The late-risk linear ladder expects the selected model to be a logistic")
    linear = ["logistic_unregularized", selected["candidate"]]
    tuned = primary["tuned"]
    tree = ["tree_plain", tuned["tree"], tuned["forest"], tuned["boost"]]
    return {"Linear": linear, "Tree": tree}


def _ladder_index(ladders):
    return {candidate: (family, rung)
            for family, candidates in ladders.items()
            for rung, candidate in enumerate(candidates)}


def _metric_subset(metrics, fields):
    return {field: float(metrics[field]) for field in fields if field in metrics}


def _gap_verdict(low, high):
    """Classify a test gap by its 95% interval: real only when the interval excludes zero."""
    if low > 0:
        return "better"
    if high < 0:
        return "worse"
    return "level"


def _average_precision(y, score, pos_label=1):
    """Average precision with the same arithmetic as sklearn, without its per-call validation.

    The bootstrap calls this thousands of times; the inputs are always finite 0/1 labels
    and scores, so the checks in ``average_precision_score`` only add overhead. Precision
    and recall are read at the same distinct-score thresholds, so the value is identical.
    """
    y = np.asarray(y)
    score = np.asarray(score, dtype=float)
    targets = (y == pos_label).astype(np.int32)
    order = np.argsort(-score)
    sorted_score = score[order]
    sorted_targets = targets[order].astype(np.float64)
    threshold_idxs = np.concatenate((np.nonzero(np.diff(sorted_score))[0],
                                     np.asarray([sorted_targets.size - 1])))
    tps = np.cumsum(sorted_targets)[threshold_idxs]
    fps = 1 + threshold_idxs.astype(np.float64) - tps
    ps = tps + fps
    precision = np.where(ps != 0, tps / ps, 0.0)
    recall = tps / tps[-1]
    recall_curve = np.concatenate((recall[::-1], np.asarray([0.0])))
    precision_curve = np.concatenate((precision[::-1], np.asarray([1.0])))
    return float(max(0.0, -np.sum(np.diff(recall_curve) * precision_curve[:-1])))


def _score_week_counts(values, labels, groups, y, n_weeks):
    """Positives and negatives per week at each distinct score, ordered by score."""
    unique = np.unique(values)
    ranks = np.searchsorted(unique, values)
    positives = np.empty((n_weeks, unique.size), dtype=np.float64)
    negatives = np.empty((n_weeks, unique.size), dtype=np.float64)
    late = (y == 1).astype(np.float64)
    for week, rows in enumerate(groups):
        ranks_week = ranks[rows]
        positives[week] = np.bincount(ranks_week, weights=late[rows], minlength=unique.size)
        negatives[week] = np.bincount(ranks_week, weights=1.0 - late[rows], minlength=unique.size)
    return positives, negatives


def _average_precision_draws(multiplicities, positives, negatives, chunk=128):
    """Average precision for every bootstrap draw from per-week score counts.

    A draw is a multiset of whole weeks, so its score counts are the week counts
    weighted by how many times each week was drawn. That avoids sorting the rows
    again for every one of the 2,000 draws; the per-draw sum keeps sklearn's
    arithmetic order so the values are identical.
    """
    total = multiplicities.shape[0]
    values = np.empty(total, dtype=np.float64)
    for start in range(0, total, chunk):
        block = multiplicities[start:start + chunk]
        pos = (block @ positives)[:, ::-1]
        neg = (block @ negatives)[:, ::-1]
        tps = np.cumsum(pos, axis=1)
        fps = np.cumsum(neg, axis=1)
        positive_totals = tps[:, -1]
        row_totals = positive_totals + fps[:, -1]
        if (positive_totals == 0).any() or (positive_totals == row_totals).any():
            raise AssertionError(
                "PR-AUC is undefined in a bootstrap resample: it needs both late and on-time orders")
        for offset in range(len(block)):
            present = (pos[offset] + neg[offset]) > 0
            t = tps[offset][present]
            f = fps[offset][present]
            ps = t + f
            precision = np.where(ps != 0, t / ps, 0.0)
            recall = t / positive_totals[offset]
            precision_curve = np.concatenate((precision[::-1], np.asarray([1.0])))
            recall_curve = np.concatenate((recall[::-1], np.asarray([0.0])))
            values[start + offset] = max(
                0.0, -np.sum(np.diff(recall_curve) * precision_curve[:-1]))
    return values


def paired_week_gap(y, pred_c, pred_r, weeks, metric,
                    resamples=BOOTSTRAP_RESAMPLES, seed=BOOTSTRAP_SEED):
    """Pooled test gap between candidate C and reference R, with a paired week-bootstrap interval.

    Positive gaps mean C is better: MAE uses R_mae - C_mae, PR-AUC uses C_pr - R_pr. The 95%
    interval is the 2.5th to 97.5th percentile of the same pooled gap over resamples of whole
    purchase weeks, using the same rows for C and R. A gap counts as real only when the
    interval excludes zero; otherwise the verdict is level. Model selection on validation
    still uses the one-standard-error rule.
    """
    if metric not in ("mae", "pr_auc"):
        raise ValueError(f"Unsupported metric for a paired test gap: {metric!r}")
    y = np.asarray(y, dtype=float)
    pred_c = np.asarray(pred_c, dtype=float)
    pred_r = np.asarray(pred_r, dtype=float)
    weeks = np.asarray(weeks)
    if not (len(y) == len(pred_c) == len(pred_r) == len(weeks)):
        raise ValueError("y, predictions and weeks must describe the same rows")
    if not len(y):
        raise ValueError("Cannot bootstrap a test gap over no rows")
    labels = pd.factorize(pd.Series(weeks), sort=True)[0]
    if (labels < 0).any():
        raise ValueError("weeks contain missing values")
    n_weeks = int(labels.max()) + 1
    groups = [np.flatnonzero(labels == index) for index in range(n_weeks)]

    rng = np.random.default_rng(seed)
    draws = rng.integers(0, n_weeks, size=(resamples, n_weeks))

    if metric == "mae":
        error_c = np.abs(y - pred_c)
        error_r = np.abs(y - pred_r)
        sums_c = np.array([error_c[group].sum() for group in groups])
        sums_r = np.array([error_r[group].sum() for group in groups])
        counts = np.array([len(group) for group in groups])
        sampled_counts = counts[draws].sum(axis=1)
        gaps = sums_r[draws].sum(axis=1) / sampled_counts - sums_c[draws].sum(axis=1) / sampled_counts
        pooled_gap = float(error_r.mean() - error_c.mean())
    else:
        if np.unique(y).size != 2:
            raise AssertionError("PR-AUC is undefined: the test rows need both late and on-time orders")
        multiplicities = np.zeros((resamples, n_weeks), dtype=np.float64)
        for index, sampled in enumerate(draws):
            multiplicities[index] = np.bincount(sampled, minlength=n_weeks)
        candidate_counts = _score_week_counts(pred_c, labels, groups, y, n_weeks)
        reference_counts = _score_week_counts(pred_r, labels, groups, y, n_weeks)
        gaps = (_average_precision_draws(multiplicities, *candidate_counts)
                - _average_precision_draws(multiplicities, *reference_counts))
        pooled_gap = float(_average_precision(y, pred_c) - _average_precision(y, pred_r))

    low, high = np.percentile(gaps, [2.5, 97.5])
    low, high = float(low), float(high)
    return {"gap": pooled_gap, "low": low, "high": high,
            "verdict": _gap_verdict(low, high)}


ETA_VALIDATION_FIELDS = ("mae", "rmse", "r2")
ETA_VALIDATION_SOURCES = (("mae", "mae"), ("rmse", "rmse"), ("r2", "r2"))
CLASSIFIER_VALIDATION_FIELDS = (("pr_auc", "pr_auc"), ("roc_auc", "roc_auc"),
                                ("precision", "precision_top10"), ("recall", "recall_top10"))


def _eta_ladder_block(ladders, cv, stage, ladder_train, ladder_test, ladder_gaps):
    block = {}
    for family in LADDER_FAMILIES:
        block[family] = []
        for rung, name in enumerate(ladders[family]):
            windows = cv[(cv["stage"] == stage) & (cv["candidate"] == name)].sort_values("window")
            mean = {field: float(windows[field].mean()) for field in ETA_VALIDATION_FIELDS}
            block[family].append({
                "candidate": name, "rung": rung, "train": ladder_train[name],
                "validation": {"mean": mean, "windows": [
                    {"window": int(row["window"]),
                     **{field: float(row[field]) for field in ETA_VALIDATION_FIELDS}}
                    for _, row in windows.iterrows()]},
                "test": ladder_test[name],
                **ladder_gaps[name],
            })
    return block


def _classifier_ladder_block(ladders, validation, ladder_train, ladder_test, ladder_gaps, ladder_mcc):
    block = {}
    for family in LADDER_FAMILIES:
        block[family] = []
        for rung, name in enumerate(ladders[family]):
            windows = validation[(validation["variant"] == "primary") & (validation["candidate"] == name)]
            windows = windows.sort_values("window")
            mcc = {int(row["window"]): float(row["mcc"]) for row in ladder_mcc.get(name, [])}
            mean = {field: float(windows[source].mean()) for field, source in CLASSIFIER_VALIDATION_FIELDS}
            if mcc:
                mean["mcc"] = float(np.mean(list(mcc.values())))
            block[family].append({
                "candidate": name, "rung": rung, "train": ladder_train[name],
                "validation": {"mean": mean, "windows": [
                    {"window": int(row["window"]),
                     **{field: float(row[source]) for field, source in CLASSIFIER_VALIDATION_FIELDS},
                     **({"mcc": mcc[int(row["window"])]} if int(row["window"]) in mcc else {})}
                    for _, row in windows.iterrows()]},
                "test": ladder_test[name],
                **ladder_gaps[name],
            })
    return block


def _classification_metrics(labels, scores, months, share):
    flags = monthly_flags(scores, months, share)
    return {**action_metrics(labels, flags),
            "roc_auc": float(roc_auc_score(labels, scores)),
            "pr_auc": float(average_precision_score(labels, scores))}


def _fit_eta_ladder(family, params, spec_v, train, test, columns):
    pipeline = make_stage_pipeline(family, params, spec_v)
    if family == "forest":
        pipeline.set_params(model__n_jobs=1)
    with threadpool_limits(limits=1 if family in ("linear", "ridge", "forest") else None):
        pipeline.fit(train[columns], train[TARGET])
        train_predicted = pipeline.predict(train[columns])
        test_predicted = pipeline.predict(test[columns])
    return (pipeline, regression_metrics(train[TARGET], train_predicted),
            regression_metrics(test[TARGET], test_predicted), test_predicted)


def _fit_classifier_ladder(family, params, spec_v, log_features, train, test, columns, share,
                           display_share=TOP_SHARE):
    """Fit one rung. Returns metrics at `share` (to check against Phase 2's saved files) and at
    `display_share` (what MLflow logs and the dashboard shows)."""
    pipeline = make_pipeline(family, params, spec_v, log_features=log_features)
    if family == "forest":
        pipeline.set_params(model__n_jobs=1)
    with threadpool_limits(limits=1):
        pipeline.fit(train[columns], train[LABEL])
        train_scores = pipeline.predict_proba(train[columns])[:, 1]
        test_scores = pipeline.predict_proba(test[columns])[:, 1]
    train_months = train["order_purchase_timestamp"].dt.to_period("M").astype(str).to_numpy()
    test_months = test["order_purchase_timestamp"].dt.to_period("M").astype(str).to_numpy()
    y_train, y_test = train[LABEL].to_numpy(), test[LABEL].to_numpy()
    return (pipeline,
            _classification_metrics(y_train, train_scores, train_months, share),
            _classification_metrics(y_test, test_scores, test_months, share),
            _classification_metrics(y_train, train_scores, train_months, display_share),
            _classification_metrics(y_test, test_scores, test_months, display_share),
            test_scores)


def _check_metric(label, rebuilt, saved, tolerance=METRIC_TOLERANCE):
    """Fail loudly when a rebuilt value does not reproduce its saved Phase 2 value."""
    difference = abs(float(rebuilt) - float(saved))
    if difference > tolerance:
        raise AssertionError(
            f"{label}: rebuilt {rebuilt!r} does not match the saved Phase 2 value "
            f"{saved!r} (diff {difference:g})")


def _check_validation_windows(label, rebuilt, saved, fields, tolerance=METRIC_TOLERANCE):
    """Compare every rebuilt window value with its saved row, field by field."""
    for _, row in rebuilt.sort_values("window").iterrows():
        window = int(row["window"])
        expected = saved[saved["window"] == window]
        if len(expected) != 1:
            raise ValueError(f"{label} window {window}: expected one saved Phase 2 row")
        expected = expected.iloc[0]
        for field, source in fields:
            _check_metric(f"{label} window {window} {field}", row[source], expected[source], tolerance)


def _check_validation_means(label, rebuilt, saved, fields, tolerance=METRIC_TOLERANCE):
    """Compare each rebuilt window mean with the mean of the saved rows."""
    for field, source in fields:
        _check_metric(f"{label} validation mean {field}",
                      rebuilt[source].mean(), saved[source].mean(), tolerance)


def _eta_validation_windows(family, params, spec_v, train, folds, columns):
    """Refit one ETA candidate on each time-ordered window, exactly as Phase 2 did."""
    rows = []
    for window, (fit, valid) in enumerate(folds):
        fit_rows, valid_rows = train.iloc[fit], train.iloc[valid]
        pipeline = make_stage_pipeline(family, params, spec_v)
        started = time.perf_counter()
        with threadpool_limits(limits=1 if family in ("linear", "ridge") else None):
            pipeline.fit(fit_rows[columns], fit_rows[TARGET])
        fit_seconds = time.perf_counter() - started
        if family == "forest":
            # Predict in tree order: parallel accumulation reorders floating sums at ~1e-17,
            # which can flip near-tied rankings and change a saved metric beyond the 1e-9 check.
            pipeline.set_params(model__n_jobs=1)
        predicted = pipeline.predict(valid_rows[columns])
        metrics = regression_metrics(valid_rows[TARGET], predicted)
        rows.append({
            "window": window, "fit_orders": int(len(fit)), "validation_orders": int(len(valid)),
            "fit_seconds": float(fit_seconds),
            **{field: float(metrics[field]) for field in ETA_VALIDATION_FIELDS},
        })
    return pd.DataFrame(rows)


def _classifier_validation_windows(family, params, spec_v, log_features, train, folds, columns, share):
    """Refit one late-risk candidate on each time-ordered window, exactly as Phase 2 did.

    Phase 2 saved PR-AUC, ROC-AUC, precision and recall but not MCC, so the window's MCC
    is computed here from the same monthly top-share flags the reference rules use.
    """
    rows = []
    for window, (fit, valid) in enumerate(folds):
        fit_rows, valid_rows = train.iloc[fit], train.iloc[valid]
        pipeline = make_pipeline(family, params, spec_v, log_features=log_features)
        if family == "forest":
            pipeline.set_params(model__n_jobs=-1)
        started = time.perf_counter()
        with threadpool_limits(limits=None if family == "forest" else 1):
            pipeline.fit(fit_rows[columns], fit_rows[LABEL])
        fit_seconds = time.perf_counter() - started
        if family == "forest":
            # Predict in tree order: parallel accumulation reorders floating sums at ~1e-17,
            # which can flip near-tied rankings and change a saved metric beyond the 1e-9 check.
            pipeline.set_params(model__n_jobs=1)
        scores = pipeline.predict_proba(valid_rows[columns])[:, 1]
        labels = valid_rows[LABEL].to_numpy()
        months = valid_rows["order_purchase_timestamp"].dt.to_period("M").astype(str).to_numpy()
        flags = monthly_flags(scores, months, share)
        rows.append({
            "window": window, "fit_orders": int(len(fit)), "validation_orders": int(len(valid)),
            "fit_seconds": float(fit_seconds), **ranking_metrics(labels, scores),
            "mcc": float(action_metrics(labels, flags)["mcc"]),
        })
    return pd.DataFrame(rows)


def _route_benchmark(meta, spec_v, train, test):
    """Checkout benchmark: the route's recent average, median-filled from each fit's own rows.

    route_typical_days is a checkout-time history feature in the Phase 2 contract, so no model is fitted.
    The five validation windows and rows are the Phase 2 chronological folds; missing route values take
    the median of that window's fitting rows. The test split is scored with the Phase 2 training median.
    """
    feature = meta["feature"]
    windows = []
    for window, (fit, valid) in enumerate(chronological_folds(train, spec_v)):
        fitting, validation = train.iloc[fit], train.iloc[valid]
        predicted = validation[feature].fillna(fitting[feature].median())
        metrics = regression_metrics(validation[TARGET], predicted)
        windows.append({"window": window,
                        **{field: float(metrics[field]) for field in ETA_VALIDATION_FIELDS}})
    mean = {field: float(np.mean([row[field] for row in windows])) for field in ETA_VALIDATION_FIELDS}
    predicted = test[feature].fillna(train[feature].median())
    scored = regression_metrics(test[TARGET], predicted)
    return {
        "name": meta["name"],
        "label": meta["label"],
        "definition": meta["definition"],
        "validation": {"mean": mean, "windows": windows},
        "test": {field: float(scored[field]) for field in REGRESSION_FIELDS},
    }, np.asarray(predicted, dtype=float)


def _eta_benchmark(stage, cv, lookup, spec_v, train, test):
    meta = ETA_BENCHMARKS[stage]
    name = meta["name"]
    if "feature" in meta:
        return _route_benchmark(meta, spec_v, train, test)
    windows = cv[(cv["stage"] == stage) & (cv["candidate"] == name)].sort_values("window")
    if not len(windows):
        raise ValueError(f"No saved validation windows for the {stage} benchmark {name}")
    mean = {field: float(windows[field].mean()) for field in ETA_VALIDATION_FIELDS}
    family, params = lookup[name]
    predicted = fit_predict(family, params, spec_v, train, test)
    scored = regression_metrics(test[TARGET], predicted)
    return {
        "name": name,
        "label": meta["label"],
        "definition": meta["definition"],
        "validation": {"mean": mean, "windows": [
            {"window": int(row["window"]), **{field: float(row[field]) for field in ETA_VALIDATION_FIELDS}}
            for _, row in windows.iterrows()]},
        "test": {field: float(scored[field]) for field in REGRESSION_FIELDS},
    }, np.asarray(predicted, dtype=float)


def _reference(key, validation, test, definition):
    """One reference rule with the wording the page shows and its per-split metrics."""
    return {"key": key, **REFERENCE_WORDING[key], "definition": definition,
            "validation": validation, "test": test}


def _mean_benchmark(stage, cv, train, test):
    """Regression floor: predict the training average for everyone."""
    windows = cv[(cv["stage"] == stage) & (cv["candidate"] == "mean_baseline")].sort_values("window")
    if not len(windows):
        raise ValueError(f"No saved mean-baseline windows for {stage}")
    mean = {field: float(windows[field].mean()) for field in ETA_VALIDATION_FIELDS}
    predicted = np.full(len(test), float(train[TARGET].mean()))
    scored = regression_metrics(test[TARGET], predicted)
    return {
        "validation": {"mean": mean, "windows": [
            {"window": int(row["window"]), **{field: float(row[field]) for field in ETA_VALIDATION_FIELDS}}
            for _, row in windows.iterrows()]},
        "test": {field: float(scored[field]) for field in REGRESSION_FIELDS},
    }, predicted


def _classifier_references(train, test, spec, spec_v, champion_scores, family, params, log_features):
    """Score the four late-risk reference rules with the queue's own static scoring.

    Deadline and remaining slack come from ``phase3.queue``. The 50/50 blend's
    test row is replaced with the monthly-refit blend after this function runs,
    which is the report's evaluation method. A random order's PR-AUC is the late
    rate of the split being scored.
    """
    folds = chronological_folds(train, spec)
    keys = ("deadline_rule", "slack_rule", "blend_rule", "random_rule")
    names = {"deadline_rule": "deadline", "slack_rule": "slack", "blend_rule": "blend"}
    windows = {key: [] for key in keys}
    for window, (fit, valid) in enumerate(folds):
        fit_rows, valid_rows = train.iloc[fit], train.iloc[valid]
        scored_rows = valid_rows.copy()
        model_fit = fit_scores(family, params, spec_v, fit_rows, fit_rows, log_features)
        scored_rows["risk"] = fit_scores(family, params, spec_v, fit_rows, valid_rows, log_features)
        context = queue_module.QueueContext(
            table=scored_rows, history=None, spec=spec,
            median_transit=float(fit_rows["route_transit_90d_at_handover"].median()),
            label=f"validation_{window}")
        scores, labels = queue_module._static_scores(context)
        rule_fit = queue_module.winnie_rule_scores(fit_rows, fit_rows)
        rule_valid = queue_module.winnie_rule_scores(fit_rows, valid_rows)
        blend = (0.5 * queue_module.winnie_percentile(
                     model_fit, scored_rows["risk"].to_numpy(dtype=float))
                 + 0.5 * queue_module.winnie_percentile(rule_fit, rule_valid))
        window_scores = {"deadline_rule": scores["deadline"], "slack_rule": scores["slack"],
                         "blend_rule": blend}
        months = valid_rows["order_purchase_timestamp"].dt.to_period("M").astype(str).to_numpy()
        for key in ("deadline_rule", "slack_rule", "blend_rule"):
            windows[key].append({"window": window,
                                 **_classification_metrics(labels, window_scores[key], months, TOP_SHARE)})
        windows["random_rule"].append({"window": window, "pr_auc": float(labels.mean())})

    scored_test = test.copy()
    scored_test["risk"] = champion_scores
    context = queue_module.QueueContext(
        table=scored_test, history=None, spec=spec,
        median_transit=float(train["route_transit_90d_at_handover"].median()), label="test")
    scores, labels = queue_module._static_scores(context)
    train_risk = fit_scores(family, params, spec_v, train, train, log_features)
    rule_fit = queue_module.winnie_rule_scores(train, train)
    rule_test = queue_module.winnie_rule_scores(train, test)
    blend_test = (0.5 * queue_module.winnie_percentile(train_risk, champion_scores)
                  + 0.5 * queue_module.winnie_percentile(rule_fit, rule_test))
    test_values = {"deadline_rule": scores["deadline"], "slack_rule": scores["slack"],
                   "blend_rule": blend_test}
    test_metrics = {key: {"pr_auc": float(average_precision_score(labels, test_values[key]))}
                    for key in ("deadline_rule", "slack_rule", "blend_rule")}
    test_metrics["random_rule"] = {"pr_auc": float(labels.mean())}

    definitions = {
        "deadline_rule": DEADLINE_RULE_DEFINITION,
        "slack_rule": SLACK_RULE_DEFINITION,
        "blend_rule": BLEND_RULE_DEFINITION,
        "random_rule": RANDOM_RULE_DEFINITION,
    }
    references = []
    for key in keys:
        present = [field for field in CLASSIFIER_FIELDS
                   if windows[key] and field in windows[key][0]]
        references.append(_reference(
            key,
            {"mean": {field: float(np.mean([row[field] for row in windows[key]])) for field in present},
             "windows": windows[key]},
            test_metrics[key], definitions[key]))
    test_scores = {key: np.asarray(test_values[key], dtype=float)
                   for key in ("deadline_rule", "slack_rule", "blend_rule")}
    return references, test_scores


def _log_references(references, validation_fields, test_fields):
    """Log each reference rule's wording and metrics under its own key prefix."""
    mlflow.log_param("reference_keys", ",".join(reference["key"] for reference in references))
    has_gap = False
    for reference in references:
        key = reference["key"]
        _log_params({
            f"reference_{key}_label": reference["label"],
            f"reference_{key}_short": reference["short"],
            f"reference_{key}_hover": reference["hover"],
            f"reference_{key}_role": reference["role"],
            f"reference_{key}_definition": reference["definition"],
        })
        for row in reference["validation"]["windows"]:
            mlflow.log_metrics({f"reference_{key}_validation_window_{field}": float(row[field])
                                for field in validation_fields if field in row}, step=int(row["window"]))
        mlflow.log_metrics({
            f"reference_{key}_validation_mean_{field}": float(reference["validation"]["mean"][field])
            for field in validation_fields if field in reference["validation"]["mean"]})
        if test_fields:
            mlflow.log_metrics({f"reference_{key}_test_{_display_field(field, TOP_SHARE)}":
                                float(reference["test"][field])
                                for field in test_fields if field in reference["test"]})
        if "test_gap_vs_headline" in reference:
            mlflow.log_metrics({
                f"reference_{key}_test_gap_vs_headline": float(reference["test_gap_vs_headline"]),
                f"reference_{key}_test_gap_low_vs_headline": float(reference["test_gap_low_vs_headline"]),
                f"reference_{key}_test_gap_high_vs_headline": float(reference["test_gap_high_vs_headline"]),
            })
            mlflow.set_tag(f"reference_{key}_test_gap_verdict", reference["test_gap_verdict"])
            has_gap = True
    if has_gap:
        mlflow.log_param("test_gap_method", TEST_GAP_METHOD)


SHARED_VIEW_STATE_TAG = "mlflow.sharedViewState.ladder"


def set_ladder_view_state(client, experiment, metric):
    """Save the runs-page view the Models tab's Compare link opens.

    It lists every candidate run for the experiment so the page shows the whole
    search, not just the six ladder rungs. MLflow 3 reads this experiment tag when
    the runs page has ``?viewStateShareKey=ladder``; keep the format in step with
    the UI bundle.
    """
    key = f"validation_mean_{metric}"
    state = {
        "searchFilter": "tags.run_kind = 'candidate'",
        "orderByKey": f"metrics.`{key}`",
        "orderByAsc": metric == "mae",
        "selectedColumns": [
            f"metrics.`{key}`",
            f"metrics.`test_{metric}`",
            "metrics.`gap_to_best`",
            "metrics.`se_of_gap`",
            "metrics.`within_one_se`",
        ],
    }
    client.set_experiment_tag(experiment, SHARED_VIEW_STATE_TAG,
                              json.dumps(state, separators=(",", ":"), sort_keys=True))
    return state


def _log_candidate(experiment, name, definition, window_fitter, validation_fields, selection_metric, stage2, selected,
                   ladder=None, train_metrics=None, test_metrics=None, test_reference=False,
                   run_kind="candidate", test_gap=None):
    """Open one candidate run, train its validation windows, then log everything it produced."""
    tags = {"run_kind": run_kind, "candidate": name, "selected": "true" if selected else "false"}
    if ladder:
        family, rung = ladder
        tags.update({"ladder_family": family, "ladder_rung": str(rung)})
    if test_metrics is not None and test_reference:
        tags["test_use"] = "reference_only_selection_on_validation"
    with mlflow.start_run(experiment_id=experiment, run_name=name, nested=True, tags=tags):
        _log_params(definition)
        windows = window_fitter().sort_values("window").reset_index(drop=True)
        mlflow.set_tag(NOTE_TAG, _candidate_note(definition, len(windows)))
        for _, row in windows.iterrows():
            step = int(row["window"])
            mlflow.log_metrics({f"validation_window_{_display_field(field, TOP_SHARE)}": float(row[source])
                                for field, source in validation_fields}, step=step)
            mlflow.log_metrics({
                "fit_seconds": float(row["fit_seconds"]),
                "window_fit_orders": float(row["fit_orders"]),
                "window_validation_orders": float(row["validation_orders"]),
            }, step=step)
            if "mcc" in windows.columns:
                mlflow.log_metric(f"validation_window_{_display_field('mcc', TOP_SHARE)}",
                                  float(row["mcc"]), step=step)
        mlflow.log_metric("fit_seconds_total", float(windows["fit_seconds"].sum()))
        means = {f"validation_mean_{_display_field(field, TOP_SHARE)}": float(windows[source].mean())
                 for field, source in validation_fields}
        if stage2:
            means[f"validation_mean_{selection_metric}"] = float(stage2["mean"])
        if "mcc" in windows.columns:
            means[f"validation_mean_{_display_field('mcc', TOP_SHARE)}"] = float(windows["mcc"].mean())
        mlflow.log_metrics(means)
        if stage2:
            mlflow.log_metrics({
                "gap_to_best": float(stage2["gap_to_best"]),
                "se_of_gap": float(stage2["se_of_gap"]),
                "within_one_se": 1.0 if stage2["within_one_se"] else 0.0,
            })
        if train_metrics is not None:
            mlflow.log_metrics({f"train_{_display_field(key, TOP_SHARE)}": float(value)
                                for key, value in train_metrics.items()})
        if test_metrics is not None:
            mlflow.log_metrics({f"test_{_display_field(key, TOP_SHARE)}": float(value)
                                for key, value in test_metrics.items()})
        if test_gap is not None:
            mlflow.log_metrics({
                "test_gap_vs_headline": float(test_gap["test_gap_vs_headline"]),
                "test_gap_low_vs_headline": float(test_gap["test_gap_low_vs_headline"]),
                "test_gap_high_vs_headline": float(test_gap["test_gap_high_vs_headline"]),
            })
            mlflow.set_tag("test_gap_verdict", test_gap["test_gap_verdict"])
            mlflow.log_param("test_gap_method", TEST_GAP_METHOD)
        return windows


def _check_saved(label, got, expected, fields):
    failures = []
    for field in fields:
        difference = abs(float(got[field]) - float(expected[field]))
        if difference > METRIC_TOLERANCE:
            failures.append(f"{field}: live {got[field]!r}, saved {expected[field]!r} (diff {difference:g})")
    if failures:
        raise AssertionError(f"{label} does not reproduce the saved Phase 2 metrics:\n" + "\n".join(failures))


def _coefficients(pipeline):
    model = pipeline.named_steps["model"]
    if not hasattr(model, "coef_"):
        raise ValueError("The selected model has no coefficients; the registry expects a linear or ridge model")
    names = pipeline.named_steps["prepare"].get_feature_names_out()
    coefficients = np.asarray(model.coef_).reshape(-1)
    if len(names) != len(coefficients):
        raise ValueError("Coefficient count does not match the prepared feature count")
    frame = pd.DataFrame({"feature": names, "coefficient": coefficients})
    frame["abs_coefficient"] = frame["coefficient"].abs()
    frame["rank"] = frame["abs_coefficient"].rank(method="first", ascending=False).astype(int)
    return frame.sort_values(["rank", "feature"]).reset_index(drop=True)


def _input_directions(pipeline, spec):
    """Direction of each raw input from the champion's standardised coefficients.

    A raw input owns one or more prepared columns (purchase hour owns sine and cosine; a
    log-transformed input owns its logged column; a categorical input owns its one-hot
    columns). A sign is shown only when every owned coefficient has the same sign, and
    categorical inputs never get one.
    """
    prepare = pipeline.named_steps["prepare"]
    model = pipeline.named_steps["model"]
    if not hasattr(model, "coef_"):
        raise ValueError("The selected model has no coefficients, so direction signs cannot be read")
    names = np.asarray(prepare.get_feature_names_out())
    coefficients = np.asarray(model.coef_).reshape(-1)
    if len(names) != len(coefficients):
        raise ValueError("Coefficient count does not match the prepared feature count")
    numeric, categorical = feature_columns(spec)
    logged = set(spec.get("linear_log_candidates", ()))
    directions = {}
    for name in numeric + categorical:
        if name in categorical:
            directions[name] = ""
            continue
        if name == "purchase_hour":
            owned = np.isin(names, ("hour__hour_sin", "hour__hour_cos"))
        elif name in logged:
            owned = names == f"logged__{name}"
        else:
            owned = names == f"numeric__{name}"
        values = coefficients[owned]
        if len(values) and np.all(values > 0):
            directions[name] = "+"
        elif len(values) and np.all(values < 0):
            directions[name] = "\u2212"
        else:
            directions[name] = ""
    return directions


LATE_SLACK_INPUTS = (
    "promised_days", "promise_slack", "route_typical_days", "handover_days",
    "route_transit_90d_at_handover", "remaining_slack", "past_promise_at_handover",
    "slack_shortfall",
)


def _importance_groups(columns):
    """Groups of inputs that describe the same thing; every other input stays on its own."""
    columns = list(columns)
    groups = []
    if all(name in columns for name in LATE_SLACK_INPUTS):
        groups.append({
            "name": "Promise, handover time and spare time",
            "key": "promise-handover-spare-time",
            "members": list(LATE_SLACK_INPUTS),
        })
    elif "route_transit_90d_at_handover" in columns:
        groups.append({
            "name": "Promise and route time",
            "key": "promise-and-route-time",
            "members": ["promised_days", "promise_slack", "route_typical_days",
                        "route_transit_90d_at_handover"],
        })
    elif "route_typical_days" in columns:
        groups.append({
            "name": "Promise and usual route time",
            "key": "promise-and-usual-route-time",
            "members": ["promised_days", "promise_slack", "route_typical_days"],
        })
    seen = {member for group in groups for member in group["members"]}
    for name in columns:
        if name not in seen:
            groups.append({"name": input_label(name), "key": name, "members": [name]})
    return groups


def _partial_direction(pipeline, table, columns, numeric, name, kind):
    """Direction of the model's average prediction as one numeric input rises on test data."""
    features = table[columns].copy()
    for column in numeric:
        features[column] = features[column].astype(float)
    result = partial_dependence(
        pipeline, features, [name], grid_resolution=9, percentiles=(0.05, 0.95),
        kind="average", response_method="predict_proba" if kind == "classification" else "auto")
    values = np.asarray(result["average"][0], dtype=float)
    grid = np.asarray(result["grid_values"][0], dtype=float)
    if grid.size < 2 or not np.isfinite(values).all():
        return 0
    slope = float(np.polyfit(grid, values, 1)[0])
    if not np.isfinite(slope) or abs(slope) < 1e-12:
        return 0
    return 1 if slope > 0 else -1


def _direction_text(sign, kind):
    if sign == "+":
        return "higher \u2192 more likely late" if kind == "classification" else "higher \u2192 more days"
    if sign == "\u2212":
        return "higher \u2192 less likely late" if kind == "classification" else "higher \u2192 fewer days"
    return ""


def _single_direction(pipeline, table, columns, numeric, name, directions, kind):
    """A worded direction only when the coefficient sign and test partial dependence agree."""
    if name not in numeric or name == "purchase_hour":
        return "mixed/overlapping"
    sign = directions.get(name, "")
    if sign not in ("+", "\u2212"):
        return "mixed/overlapping"
    partial = _partial_direction(pipeline, table, columns, numeric, name, kind)
    expected = 1 if sign == "+" else -1
    if partial != expected:
        return "mixed/overlapping"
    return _direction_text(sign, kind)


def _fold_predictions(pipelines, fold_rows, frame, kind):
    """Predict each row with the pipeline fitted on that row's validation window."""
    fold_rows = np.asarray(fold_rows)
    out = np.empty(len(frame), dtype=float)
    for fold, pipeline in enumerate(pipelines):
        mask = fold_rows == fold
        if not mask.any():
            continue
        if kind == "classification":
            out[mask] = pipeline.predict_proba(frame.loc[mask])[:, 1]
        else:
            out[mask] = pipeline.predict(frame.loc[mask])
    return out


def _grouped_permutation_importance(pipeline, table, spec, target, scoring, kind,
                                    fold_pipelines=None, fold_rows=None):
    """Permutation importance, shuffling each overlapping group of inputs together.

    With `fold_pipelines`, `table` is the pooled validation data and each row is scored by the
    model fitted on its own chronological validation window's training rows. Without it, the
    supplied pipeline scores the whole table. A group is shuffled with one row order for all its
    members, so the relationships inside the group survive; the bar answers what happens when
    the whole idea is scrambled.
    """
    numeric, categorical = feature_columns(spec)
    columns = numeric + categorical
    features = table[columns]
    labels = table[target]
    scorer = check_scoring(pipeline, scoring=scoring)

    def score(frame):
        if fold_pipelines is None:
            return scorer(pipeline, frame, labels)
        predictions = _fold_predictions(fold_pipelines, fold_rows, frame, kind)
        if kind == "classification":
            return average_precision_score(labels, predictions)
        return -float(np.mean(np.abs(np.asarray(labels) - predictions)))

    baseline = score(features)
    rng = check_random_state(spec["random_state"])
    directions = _input_directions(pipeline, spec)
    rows = []
    for group in _importance_groups(columns):
        values = []
        for _ in range(10):
            shuffled = features.copy()
            order = rng.permutation(len(features))
            for member in group["members"]:
                shuffled[member] = features[member].to_numpy()[order]
            values.append(baseline - score(shuffled))
        members = group["members"]
        direction = "mixed/overlapping"
        if len(members) == 1:
            direction = _single_direction(pipeline, table, columns, numeric, members[0],
                                          directions, kind)
        rows.append({
            "input": group["name"],
            "key": group["key"],
            "members": ",".join(members),
            "importance_mean": float(np.mean(values)),
            "importance_std": float(np.std(values)),
            "direction": direction,
        })
    frame = pd.DataFrame(rows)
    return frame.sort_values("importance_mean", ascending=False).reset_index(drop=True)


def _validation_importance_rows(pipeline, train, spec, spec_v, target, scoring, kind,
                                family, params, log_features=None):
    """Importance on the pooled validation windows, each row scored by its window's fitted model."""
    folds = chronological_folds(train, spec)
    columns = sum(feature_columns(spec_v), [])
    parts, pipelines, fold_rows = [], [], []
    for fold, (fit, valid) in enumerate(folds):
        if kind == "classification":
            member = make_pipeline(family, params, spec_v, log_features=log_features)
        else:
            member = make_stage_pipeline(family, params, spec_v)
            if family == "forest":
                member.set_params(model__n_jobs=1)
        with threadpool_limits(limits=1 if family in ("linear", "ridge", "forest") else None):
            member.fit(train.iloc[fit][columns], train.iloc[fit][target])
        pipelines.append(member)
        parts.append(train.iloc[valid])
        fold_rows.extend([fold] * len(valid))
    pooled = pd.concat(parts).reset_index(drop=True)
    return _grouped_permutation_importance(
        pipeline, pooled, spec_v, target, scoring, kind,
        fold_pipelines=pipelines, fold_rows=np.asarray(fold_rows))


def _inputs(table, spec):
    numeric, categorical = feature_columns(spec)
    return [{"name": name, "dtype": str(table[name].dtype)} for name in numeric + categorical]


def _plot_grid(rows, selected, path, ylabel, title):
    names = [row["candidate"] for row in rows]
    means = np.array([row["mean"] for row in rows], dtype=float)
    errors = np.array([row["se_of_gap"] for row in rows], dtype=float)
    positions = np.arange(len(rows))
    colors = ["#d62728" if name == selected else "#4c72b0" for name in names]
    sizes = [90 if name == selected else 45 for name in names]

    figure, axes = plt.subplots(figsize=(max(7.0, 0.75 * len(rows)), 4.8))
    axes.errorbar(positions, means, yerr=errors, fmt="none", ecolor="#7f7f7f", capsize=4, zorder=1)
    axes.scatter(positions, means, c=colors, s=sizes, zorder=2)
    axes.set_xticks(positions)
    axes.set_xticklabels(names, rotation=45, ha="right")
    axes.set_xlim(-0.5, len(rows) - 0.5)
    axes.set_xlabel("Candidate grid")
    axes.set_ylabel(ylabel)
    axes.grid(axis="y", alpha=0.3)
    axes.set_title(title)
    figure.tight_layout()
    figure.savefig(path, dpi=150)
    plt.close(figure)


def _apparent_row(frame, task, stage, model):
    rows = frame[(frame["task"] == task) & (frame["stage"] == stage) & (frame["model"] == model)]
    if len(rows) != 1:
        raise ValueError(f"Expected one train-apparent row for {task}/{stage}/{model}")
    return rows.iloc[0]


def _log_apparent(row, exclude=()):
    metrics = {}
    for name, value in row.items():
        if name in ("task", "stage", "model", "scope") or name in exclude:
            continue
        if isinstance(value, (int, float, np.integer, np.floating)) and not pd.isna(value):
            metrics[f"train_apparent_{name}"] = float(value)
    mlflow.log_metrics(metrics)


def _register(client, model_name, candidate, train_cutoff):
    versions = client.search_model_versions(f"name = '{model_name}'")
    if not versions:
        raise RuntimeError(f"MLflow did not register {model_name}")
    version = max(versions, key=lambda item: int(item.version))
    tags = {"source": "phase2_selection", "candidate": candidate, "train_cutoff": train_cutoff}
    for key, value in tags.items():
        client.set_model_version_tag(model_name, str(version.version), key, value)
    client.set_registered_model_alias(model_name, "champion", str(version.version))
    return int(version.version)


def _register_monthly(client, model_name, candidate, refit_date, fit_rows):
    """Tag the just-registered monthly version; unlike ``_register`` it keeps the champion alias."""
    versions = client.search_model_versions(f"name = '{model_name}'")
    if not versions:
        raise RuntimeError(f"MLflow did not register the {refit_date:%Y-%m} {model_name} version")
    version = max(versions, key=lambda item: int(item.version))
    refit = refit_date.date().isoformat()
    tags = {"source": "monthly_refit", "candidate": candidate, "train_cutoff": refit,
            "refit_date": refit, "fit_rows": str(int(fit_rows))}
    for key, value in tags.items():
        client.set_model_version_tag(model_name, str(version.version), key, value)
    return int(version.version)


def _monthly_late_risk(table, spec, test, start=LATE_RISK_FIRST_REFIT):
    """Monthly-refit late-risk scores, the blend scores and each month's fitted pipeline.

    Uses ``queue.winnie_monthly_refits`` exactly as the report reproduction does,
    starting at the first of May 2018.  ``model_scores`` is the model alone;
    ``blend_scores`` is half the model's percentile plus half the remaining-slack
    percentile, both against that month's fitting orders (the report's refresh).
    """
    model_scores = np.full(len(test), np.nan)
    blend_scores = np.full(len(test), np.nan)
    refits = []
    for begin, end, known, pipeline, columns in queue_module.winnie_monthly_refits(
            table, spec, start, LATE_RISK_LAST_REFIT):
        take = test["handover_time"].ge(begin) & test["handover_time"].lt(end)
        where = np.flatnonzero(take.to_numpy())
        if len(where):
            part = test.loc[take]
            model_ref = pipeline.predict_proba(known[columns])[:, 1]
            rule_ref = queue_module.winnie_rule_scores(known, known)
            model_values = pipeline.predict_proba(part[columns])[:, 1]
            rule_values = queue_module.winnie_rule_scores(known, part)
            model_scores[where] = model_values
            blend_scores[where] = (0.5 * queue_module.winnie_percentile(model_ref, model_values)
                                   + 0.5 * queue_module.winnie_percentile(rule_ref, rule_values))
        refits.append((begin, known, pipeline, columns))
    if np.isnan(model_scores).any() or np.isnan(blend_scores).any():
        raise AssertionError("A monthly late-risk refit is missing a test handover")
    return model_scores, blend_scores, refits


def _monthly_warning_days(table, spec):
    """Blend and remaining-slack mean warning days in the test daily replay.

    The blend follows ``queue._winnie_refit_daily`` (monthly refits); the slack
    rule follows the report's ``B_rule`` replay.  These are the figures the
    Models tab quotes for the live choice.
    """
    cutoff = pd.Timestamp(spec["primary_cutoff"])
    test = table[table[spec["primary_split"]].eq("test")].reset_index(drop=True)
    end = test["order_purchase_timestamp"].max().normalize()
    orders = table[table["handover_time"].ge(cutoff)
                   & table["handover_time"].lt(end + pd.Timedelta(days=1))]
    orders = orders.sort_values(["handover_time", "order_id"], kind="stable").reset_index(drop=True)
    refit = queue_module._winnie_refit_daily("test", table, spec, orders, cutoff, end)
    blend = pd.DataFrame(queue_module._winnie_summarise(refit, "test", ("B_mix_refits",))).iloc[0]
    train = table[table[spec["primary_split"]].eq("train")].reset_index(drop=True)
    dummy_scores = np.zeros(len(orders), dtype=float)
    _, slack_daily = queue_module.winnie_daily_selection(
        "test", orders, cutoff, end, dummy_scores, queue_module.winnie_rule_scores(train, orders),
        dummy_scores, queue_module.winnie_rule_scores(train, train), versions=("B_rule",))
    slack = pd.DataFrame(queue_module._winnie_summarise(slack_daily, "test", ("B_rule",))).iloc[0]
    return (float(blend["mean_days_left"]), float(slack["mean_days_left"]),
            int(blend["caught_before_date"]), int(blend["caught_3plus_days"]))


def _drift_threshold_payload(model_name, train, spec, spec_v, importance):
    """The monitoring thresholds this model version carries, computed the same way as the tab."""
    numeric, categorical = feature_columns(spec_v)
    expanded = drift_module.expand_importance(importance)
    features = drift_module.important_inputs(expanded, numeric, categorical)
    table = drift_module.input_threshold_table(train, spec, features)
    return {
        "model": model_name,
        "reference": "champion training data",
        "readings": table["readings"],
        "n_readings": table["n_readings"],
        "inputs": table["inputs"],
    }


def _log_final_artifacts(pipeline, table, spec, grid_rows, selected, grid_ylabel, grid_title, importance,
                         drift_thresholds=None):
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        coefficients = _coefficients(pipeline)
        coefficients.to_csv(tmp / "coefficients.csv", index=False)
        importance.to_csv(tmp / "permutation_importance.csv", index=False)
        (tmp / "inputs.json").write_text(json.dumps(_inputs(table, spec), indent=2) + "\n")
        _plot_grid(grid_rows, selected, tmp / "grid.png", grid_ylabel, grid_title)
        artifacts = ["coefficients.csv", "permutation_importance.csv", "inputs.json", "grid.png"]
        if drift_thresholds is not None:
            (tmp / "drift_thresholds.json").write_text(
                json.dumps(drift_thresholds, indent=2, sort_keys=True) + "\n")
            artifacts.append("drift_thresholds.json")
            mlflow.set_tag("drift_thresholds", "drift_thresholds.json")
        for artifact in artifacts:
            mlflow.log_artifact(str(tmp / artifact))


def register_eta_model(client, model_name, stage, selection):
    result_dir = ETA_RESULTS
    cv = pd.read_csv(result_dir / "cv_metrics.csv")
    saved = pd.read_csv(result_dir / "test_metrics.csv")
    apparent_frame = pd.read_csv(FINAL_RESULTS / "train_apparent_metrics.csv")
    lookup = _candidate_lookup(ETA_CANDIDATES)

    table, spec, _ = load_stage_table(ROOT)
    spec_v = stage_spec(spec, stage)
    columns = sum(feature_columns(spec_v), [])
    train = table[table[spec["primary_split"]].eq("train")].reset_index(drop=True)
    test = table[table[spec["primary_split"]].eq("test")].reset_index(drop=True)
    folds = chronological_folds(train, spec)

    selected_model = selection["selected_model"][stage]
    champion = selected_model["candidate"]
    family, params = lookup[champion]
    if family != selected_model["family"] or params != selected_model["parameters"]:
        raise ValueError(f"{model_name}: selected candidate does not match the Phase 2 candidate definition")

    stage2 = _stage2_map(selection["stage2"][stage])
    names = list(ETA_SIMPLICITY)
    ladders = _eta_ladders(selection, stage, stage2)
    ladder_index = _ladder_index(ladders)
    ladder_names = [name for family in LADDER_FAMILIES for name in ladders[family]]
    for name in ladder_names:
        if name not in names:
            names.append(name)

    ladder_train, ladder_test, ladder_pipelines, ladder_predictions = {}, {}, {}, {}
    for name in ladder_names:
        if name not in lookup:
            raise KeyError(f"{model_name}: no Phase 2 candidate definition for {name}")
        candidate_family, candidate_params = lookup[name]
        pipeline_rung, train_metrics, test_metrics, test_predicted = _fit_eta_ladder(
            candidate_family, candidate_params, spec_v, train, test, columns)
        ladder_pipelines[name] = pipeline_rung
        ladder_train[name] = _metric_subset(train_metrics, REGRESSION_LADDER_FIELDS)
        ladder_test[name] = _metric_subset(test_metrics, REGRESSION_LADDER_FIELDS)
        ladder_predictions[name] = np.asarray(test_predicted, dtype=float)
        saved_row = saved[(saved["stage"] == stage) & (saved["candidate"] == name)
                          & (saved["scope"] == "overall")]
        if len(saved_row):
            _check_saved(f"{model_name} {name} test", test_metrics, saved_row.iloc[0], REGRESSION_FIELDS)
    _check_saved(f"{model_name} {champion} train", ladder_train[champion],
                 _apparent_row(apparent_frame, "regression", stage, champion), REGRESSION_LADDER_FIELDS)

    benchmark, benchmark_prediction = _eta_benchmark(stage, cv, lookup, spec_v, train, test)
    mean_floor, _ = _mean_benchmark(stage, cv, train, test)
    references = [
        _reference(benchmark["name"], benchmark["validation"], benchmark["test"], benchmark["definition"]),
        _reference("mean_baseline", mean_floor["validation"], mean_floor["test"], MEAN_BASELINE_DEFINITION),
    ]
    weeks = test["order_purchase_timestamp"].dt.to_period("W")
    test_target = test[TARGET].to_numpy()
    ladder_gaps = {}
    for name in ladder_names:
        gap = paired_week_gap(test_target, ladder_predictions[name], benchmark_prediction, weeks, "mae")
        ladder_gaps[name] = {
            "test_gap_vs_headline": gap["gap"],
            "test_gap_low_vs_headline": gap["low"],
            "test_gap_high_vs_headline": gap["high"],
            "test_gap_verdict": gap["verdict"],
        }

    experiment = experiment_id(client, model_name)
    parent_tags = {
        "run_kind": "selection", "model": model_name, "stage": stage,
        "selection_file": str(ETA_SELECTION.relative_to(ROOT)),
        "selection_sha256": sha256(ETA_SELECTION), "selected_candidate": champion,
        NOTE_TAG: ("Selection run: trains every candidate on the time-ordered validation "
                   "windows and records the model the one-standard-error rule chose."),
    }
    final_metrics = {}
    rebuilt = {}
    with mlflow.start_run(experiment_id=experiment, run_name="selection", tags=parent_tags):
        for name in names:
            if name not in lookup:
                raise KeyError(f"{model_name}: no Phase 2 candidate definition for {name}")
            candidate_family, candidate_params = lookup[name]
            saved_windows = cv[(cv["stage"] == stage) & (cv["candidate"] == name)].sort_values("window")
            if len(saved_windows) != len(folds):
                raise ValueError(f"{model_name} {name}: expected one saved row per validation window")
            windows = _log_candidate(
                experiment, name, _eta_definition(candidate_family, candidate_params, spec_v),
                lambda: _eta_validation_windows(candidate_family, candidate_params, spec_v, train, folds, columns),
                ETA_VALIDATION_SOURCES, "mae", stage2.get(name), name == champion,
                ladder=ladder_index.get(name), train_metrics=ladder_train.get(name),
                test_metrics=ladder_test.get(name), test_reference=name != champion,
                run_kind="candidate" if name in stage2 else "ladder",
                test_gap=ladder_gaps.get(name))
            _check_validation_windows(f"{model_name} {name}", windows, saved_windows, ETA_VALIDATION_SOURCES)
            _check_validation_means(f"{model_name} {name}", windows, saved_windows, ETA_VALIDATION_SOURCES)
            if name in stage2:
                _check_metric(f"{model_name} {name} selection mean",
                              windows["mae"].mean(), stage2[name]["mean"])
            rebuilt[name] = windows

        chosen, _, _, _ = select_eta_candidate(
            {name: rebuilt[name].sort_values("window")["mae"].tolist() for name in ETA_SIMPLICITY})
        if chosen != champion:
            raise AssertionError(
                f"{model_name}: rebuilt validation windows select {chosen}, but Phase 2 saved {champion}")

        pipeline = ladder_pipelines.get(champion)
        if pipeline is None:
            pipeline = make_stage_pipeline(family, params, spec_v)
            if family == "forest":
                pipeline.set_params(model__n_jobs=1)
            with threadpool_limits(limits=1 if family in ("linear", "ridge", "forest") else None):
                pipeline.fit(train[columns], train[TARGET])
        predicted = pipeline.predict(test[columns])

        live = {"overall": regression_metrics(test[TARGET], predicted)}
        month = test["order_purchase_timestamp"].dt.to_period("M").astype(str)
        for value, index in test.groupby(month).groups.items():
            live[value] = regression_metrics(test.loc[index, TARGET], predicted[index])
        expected = saved[(saved["stage"] == stage) & (saved["candidate"] == champion)]
        for scope, values in live.items():
            row = expected[expected["scope"] == scope]
            if len(row) != 1:
                raise ValueError(f"{model_name}: missing saved test row for {scope}")
            _check_saved(f"{model_name} {scope}", values, row.iloc[0], REGRESSION_FIELDS)
            prefix = "test_overall" if scope == "overall" else f"test_month_{scope}"
            final_metrics.update({f"{prefix}_{field}": float(values[field]) for field in REGRESSION_FIELDS})

        importance = _validation_importance_rows(
            pipeline, train, spec, spec_v, TARGET, "neg_mean_absolute_error", "regression",
            family, params)
        drift_thresholds = _drift_threshold_payload(model_name, train, spec, spec_v, importance)

        final_tags = {"run_kind": "final", "model": model_name, "stage": stage,
                      "candidate": champion, "train_cutoff": spec["primary_cutoff"],
                      "selection_pool": ",".join(stage2),
                      NOTE_TAG: (f"The model the dashboard serves: refit on all data up to "
                                 f"{spec['primary_cutoff']}, scored once on test.")}
        with mlflow.start_run(experiment_id=experiment, run_name=f"chosen model \u2013 {champion}",
                              nested=True, tags=final_tags):
            _log_params(_eta_definition(family, params, spec_v))
            mlflow.log_metrics(final_metrics)
            _log_references(references, ETA_VALIDATION_FIELDS, REGRESSION_FIELDS)
            _log_apparent(_apparent_row(apparent_frame, "regression", stage, champion))
            example = test.iloc[:5][columns].copy()
            signature = infer_signature(example, pipeline.predict(example))
            mlflow.sklearn.log_model(pipeline, name="model", signature=signature, input_example=example,
                                     registered_model_name=model_name,
                                     skops_trusted_types=list(SKOPS_TRUSTED_TYPES))
            grid_rows = [stage2[name] for name in names if name in stage2]
            _log_final_artifacts(pipeline, table, spec_v, grid_rows, champion,
                                 "Validation mean MAE (days; lower is better)",
                                 f"{model_name}: {champion} selected on {stage}-stage validation windows",
                                 importance, drift_thresholds)

        version = _register(client, model_name, champion, spec["primary_cutoff"])

    set_ladder_view_state(client, experiment, "mae")
    return {
        "champion_candidate": champion,
        "champion_params": params,
        "test_metrics": {scope: {field: float(values[field]) for field in REGRESSION_FIELDS}
                         for scope, values in live.items()},
        "benchmark": benchmark,
        "references": references,
        "ladders": _eta_ladder_block(ladders, cv, stage, ladder_train, ladder_test, ladder_gaps),
        "registered_model": model_name,
        "registered_version": version,
        "alias": "champion",
        "train_cutoff": spec["primary_cutoff"],
    }


def register_classifier_model(client, model_name, selection):
    result_dir = HANDOVER_RESULTS
    validation = pd.read_csv(result_dir / "validation_windows.csv")
    saved = pd.read_csv(result_dir / "test_metrics.csv")
    apparent_frame = pd.read_csv(FINAL_RESULTS / "train_apparent_metrics.csv")
    lookup = _candidate_lookup(CLASSIFIER_CANDIDATES)

    table, spec, _ = queue_module.load_queue_table(ROOT)
    # The registry is an evaluation record: delivered orders only.  The open
    # parcels in the queue table exist for the Actions lists and would otherwise
    # take review places in the daily replays.
    table = table[table[LABEL].notna()].reset_index(drop=True)
    spec_v = variant_spec(spec)
    numeric, categorical = feature_columns(spec_v)
    columns = numeric + categorical
    train = table[table[spec["primary_split"]].eq("train")].reset_index(drop=True)
    test = table[table[spec["primary_split"]].eq("test")].reset_index(drop=True)
    folds = chronological_folds(train, spec)
    # The blend row and the servable versions use the May-October refits; the
    # model-alone row stays the frozen Phase 2 fit the report compares.
    _, monthly_blend_scores, monthly_refits = _monthly_late_risk(table, spec, test)

    primary = selection["variants"]["primary"]
    selected_model = primary["selected_model"]
    # Phase 2 saved its action metrics at its cost-ratio share (0.14); the report and Task 4 use
    # TOP_SHARE (10%). Checks against the saved files use review_k; train, validation and test are all
    # logged at TOP_SHARE.
    review_k = float(primary["k"])
    champion = selected_model["candidate"]
    family, params = lookup[champion]
    if family != selected_model["family"] or params != selected_model["parameters"]:
        raise ValueError(f"{model_name}: selected candidate does not match the Phase 2 candidate definition")

    available = set(validation[validation["variant"] == "primary"]["candidate"])
    names = [name for group in selection["groups"] for name in group["configs"] if name in available]
    for name in primary["lineup"]:
        if name not in names:
            names.append(name)
    stage2 = _stage2_map(primary["stage2"])
    log_features = spec["linear_log_candidates"]
    ladders = _classifier_ladders(selection)
    ladder_index = _ladder_index(ladders)
    ladder_names = [name for family in LADDER_FAMILIES for name in ladders[family]]

    ladder_train, ladder_test, ladder_pipelines, ladder_scores = {}, {}, {}, {}
    for name in ladder_names:
        if name not in lookup:
            raise KeyError(f"{model_name}: no Phase 2 candidate definition for {name}")
        candidate_family, candidate_params = lookup[name]
        (pipeline_rung, train_metrics, test_metrics, train_shown, test_shown,
         test_scores) = _fit_classifier_ladder(
            candidate_family, candidate_params, spec_v, log_features, train, test, columns, review_k)
        ladder_pipelines[name] = pipeline_rung
        ladder_train[name] = _metric_subset(train_shown, CLASSIFIER_LADDER_FIELDS)
        ladder_test[name] = _metric_subset(test_shown, CLASSIFIER_LADDER_FIELDS)
        if name == champion:
            champion_train_checked = train_metrics
            champion_train_shown = train_shown
        ladder_scores[name] = np.asarray(test_scores, dtype=float)
        if name == champion:
            expected = saved[(saved["variant"] == "primary") & (saved["ranking"] == "model")]
            if len(expected) != 1:
                raise ValueError(f"{model_name}: missing saved test row for the champion")
            _check_saved(f"{model_name} {name} test", test_metrics, expected.iloc[0], CLASSIFIER_FIELDS)
    _check_saved(f"{model_name} {champion} train", champion_train_checked,
                 _apparent_row(apparent_frame, "classification", "handover", champion), CLASSIFIER_FIELDS)

    weeks = test["order_purchase_timestamp"].dt.to_period("W")
    test_labels = test[LABEL].to_numpy()
    references, reference_scores = _classifier_references(
        train, test, spec, spec_v, ladder_scores[champion], family, params, log_features)
    # The blend row is the monthly-refit blend, not the single-fit percentile blend.
    for reference in references:
        if reference["key"] == "blend_rule":
            reference["test"]["pr_auc"] = float(average_precision_score(test_labels, monthly_blend_scores))
    reference_scores["blend_rule"] = monthly_blend_scores
    headline = next(reference for reference in references if reference["role"] == "headline")
    headline_scores = reference_scores[headline["key"]]
    ladder_gaps = {}
    for name in ladder_names:
        # The model-alone rows (including the champion) are the frozen Phase 2 fit;
        # only the blend is scored with monthly refits.
        gap = paired_week_gap(test_labels, ladder_scores[name], headline_scores, weeks, "pr_auc")
        ladder_gaps[name] = {
            "test_gap_vs_headline": gap["gap"],
            "test_gap_low_vs_headline": gap["low"],
            "test_gap_high_vs_headline": gap["high"],
            "test_gap_verdict": gap["verdict"],
        }
    for reference in references:
        if reference["role"] in ("headline", "floor"):
            continue
        gap = paired_week_gap(test_labels, reference_scores[reference["key"]],
                              headline_scores, weeks, "pr_auc")
        reference.update({
            "test_gap_vs_headline": gap["gap"],
            "test_gap_low_vs_headline": gap["low"],
            "test_gap_high_vs_headline": gap["high"],
            "test_gap_verdict": gap["verdict"],
        })
    experiment = experiment_id(client, model_name)
    parent_tags = {
        "run_kind": "selection", "model": model_name,
        "selection_file": str(CLASSIFIER_SELECTION.relative_to(ROOT)),
        "selection_sha256": sha256(CLASSIFIER_SELECTION), "selected_candidate": champion,
        NOTE_TAG: ("Selection run: trains every candidate on the time-ordered validation "
                   "windows and records the model the one-standard-error rule chose."),
    }
    rebuilt = {}
    with mlflow.start_run(experiment_id=experiment, run_name="selection", tags=parent_tags):
        for name in names:
            if name not in lookup:
                raise KeyError(f"{model_name}: no Phase 2 candidate definition for {name}")
            candidate_family, candidate_params = lookup[name]
            saved_windows = validation[(validation["variant"] == "primary")
                                       & (validation["candidate"] == name)].sort_values("window")
            if len(saved_windows) != len(folds):
                raise ValueError(f"{model_name} {name}: expected one saved row per validation window")
            windows = _log_candidate(
                experiment, name, _classifier_definition(candidate_family, candidate_params, spec_v),
                lambda: _classifier_validation_windows(
                    candidate_family, candidate_params, spec_v, log_features, train, folds, columns,
                    TOP_SHARE),
                CLASSIFIER_VALIDATION_FIELDS, "pr_auc",
                stage2.get(name), name == champion, ladder=ladder_index.get(name),
                train_metrics=ladder_train.get(name), test_metrics=ladder_test.get(name),
                test_reference=name != champion, test_gap=ladder_gaps.get(name))
            # RandomForest fits vary by up to ~3e-7 across runs in this environment, so its
            # ladder rungs are checked at that noise floor; every other family stays at 1e-9.
            forest_tolerance = 1e-6 if candidate_family == "forest" else METRIC_TOLERANCE
            _check_validation_windows(
                f"{model_name} {name}", windows, saved_windows, CLASSIFIER_VALIDATION_FIELDS,
                forest_tolerance)
            _check_validation_means(
                f"{model_name} {name}", windows, saved_windows, CLASSIFIER_VALIDATION_FIELDS,
                forest_tolerance)
            if name in stage2:
                _check_metric(f"{model_name} {name} selection mean",
                              windows["pr_auc"].mean(), stage2[name]["mean"])
            rebuilt[name] = windows

        chosen, _, _, _ = select_handover_candidate(
            {name: rebuilt[name].sort_values("window")["pr_auc"].tolist() for name in stage2})
        if chosen != champion:
            raise AssertionError(
                f"{model_name}: rebuilt validation windows select {chosen}, but Phase 2 saved {champion}")

        ladder_mcc = {
            name: [{"window": int(row["window"]), "mcc": float(row["mcc"])}
                   for _, row in rebuilt[name].sort_values("window").iterrows()]
            for name in ladder_names
        }

        pipeline = ladder_pipelines.get(champion)
        if pipeline is None:
            pipeline = make_pipeline(family, params, spec_v, log_features=log_features)
            if family == "forest":
                pipeline.set_params(model__n_jobs=1)
            with threadpool_limits(limits=1):
                pipeline.fit(train[columns], train[LABEL])
        scores = pipeline.predict_proba(test[columns])[:, 1]

        eta_selection = json.loads(ETA_SELECTION.read_text())
        eta_handover = eta_selection["selected_model"]["handover"]["candidate"]
        regression_family, regression_params = next(
            (candidate_family, candidate_params) for name, candidate_family, candidate_params in ETA_CANDIDATES
            if name == eta_handover)
        base = baseline_scores(train, test, stage_spec(spec, "handover"), regression_family, regression_params)
        y = test[LABEL].to_numpy()
        months = test["order_purchase_timestamp"].dt.to_period("M").astype(str).to_numpy()
        rankings = {"model": scores, "slack_rule": base["slack_rule"]}
        live = {}
        for ranking, values in rankings.items():
            checked = _classification_metrics(y, values, months, review_k)
            expected = saved[(saved["variant"] == "primary") & (saved["ranking"] == ranking)]
            if len(expected) != 1:
                raise ValueError(f"{model_name}: missing saved test row for {ranking}")
            _check_saved(f"{model_name} {ranking}", checked, expected.iloc[0], CLASSIFIER_FIELDS)
            live[ranking] = _classification_metrics(y, values, months, TOP_SHARE)

        # live["model"] stays the frozen Phase 2 fit checked above; the report's
        # monthly refits apply to the blend row and the servable versions only.
        blend_live = _classification_metrics(y, monthly_blend_scores, months, TOP_SHARE)
        model_caught = int((y.astype(bool) & monthly_flags(scores, months, TOP_SHARE)).sum())
        blend_caught_top = int((y.astype(bool) & monthly_flags(monthly_blend_scores, months, TOP_SHARE)).sum())
        (blend_warning_days, slack_warning_days,
         blend_caught, blend_caught_3plus) = _monthly_warning_days(table, spec)

        headline["test"].update(live["slack_rule"])
        benchmark = {"name": headline["key"], "label": headline["label"],
                     "short": headline["short"], "hover": headline["hover"],
                     "definition": headline["definition"],
                     "validation": headline["validation"], "test": headline["test"]}
        final_metrics = {f"test_{_display_field(field, TOP_SHARE)}": float(live["model"][field])
                         for field in CLASSIFIER_FIELDS}
        final_metrics.update({
            "test_blend_mean_warning_days": blend_warning_days,
            "test_slack_mean_warning_days": slack_warning_days,
            "test_blend_caught": float(blend_caught),
            "test_blend_caught_3plus": float(blend_caught_3plus),
            "test_model_caught_top10pct": float(model_caught),
            "test_blend_caught_top10pct": float(blend_caught_top),
        })
        importance = _validation_importance_rows(
            pipeline, train, spec, spec_v, LABEL, "average_precision", "classification",
            family, params, log_features=log_features)
        drift_thresholds = _drift_threshold_payload(model_name, train, spec, spec_v, importance)
        final_tags = {"run_kind": "final", "model": model_name, "candidate": champion,
                      "train_cutoff": spec["primary_cutoff"],
                      "selection_pool": ",".join(stage2),
                      NOTE_TAG: (f"Phase 2 model: refit on all data up to "
                                 f"{spec['primary_cutoff']}, scored once on test. The blend row "
                                 "and the servable versions use monthly refits.")}
        with mlflow.start_run(experiment_id=experiment, run_name=f"chosen model \u2013 {champion}",
                              nested=True, tags=final_tags):
            _log_params(_classifier_definition(family, params, spec_v))
            mlflow.log_metrics(final_metrics)
            _log_references(references, CLASSIFIER_FIELDS, CLASSIFIER_FIELDS)
            _log_apparent(
                _apparent_row(apparent_frame, "classification", "handover", champion),
                exclude=("action_share", "flagged", "caught") + ACTION_FIELDS)
            mlflow.log_metrics({
                f"train_apparent_{_display_field(field, TOP_SHARE)}":
                    float(champion_train_shown[field])
                for field in ACTION_FIELDS
            })
            example = test.iloc[:5][columns].copy()
            signature = infer_signature(example, pipeline.predict(example))
            mlflow.sklearn.log_model(pipeline, name="model", signature=signature, input_example=example,
                                     registered_model_name=model_name,
                                     skops_trusted_types=list(SKOPS_TRUSTED_TYPES))
            grid_rows = [stage2[name] for name in names if name in stage2]
            _log_final_artifacts(pipeline, table, spec_v, grid_rows, champion,
                                 "Validation mean PR-AUC (higher is better)",
                                 f"{model_name}: {champion} selected on handover validation windows",
                                 importance, drift_thresholds)

        version = _register(client, model_name, champion, spec["primary_cutoff"])

    # One servable version per month the dashboard offers (May-October 2018), each
    # fitted with the shared monthly-refit rule and tagged with its refit date.
    monthly_versions = []
    for begin, known, monthly_pipeline, monthly_columns in monthly_refits:
        monthly_tags = {
            "run_kind": "monthly_refit", "model": model_name, "candidate": champion,
            "refit_date": begin.date().isoformat(), "fit_rows": str(len(known)),
            "train_cutoff": begin.date().isoformat(),
            NOTE_TAG: (f"Monthly refit live from {begin:%d %b %Y}: trained on the "
                       f"{len(known):,} orders whose purchase, handover and outcome were "
                       "recorded before that date."),
        }
        with mlflow.start_run(experiment_id=experiment,
                              run_name=f"monthly refit \u2013 {begin:%Y-%m}", tags=monthly_tags):
            _log_params(_classifier_definition(family, params, spec_v))
            mlflow.log_metric("fit_rows", float(len(known)))
            example = test.iloc[:5][monthly_columns].copy()
            signature = infer_signature(example, monthly_pipeline.predict(example))
            mlflow.sklearn.log_model(monthly_pipeline, name="model", signature=signature,
                                     input_example=example, registered_model_name=model_name,
                                     skops_trusted_types=list(SKOPS_TRUSTED_TYPES))
        monthly_versions.append({
            "refit_date": begin.date().isoformat(), "fit_rows": int(len(known)),
            "version": _register_monthly(client, model_name, champion, begin, len(known)),
        })

    set_ladder_view_state(client, experiment, "pr_auc")
    return {
        "champion_candidate": champion,
        "champion_params": params,
        "test_metrics": {ranking: {field: float(values[field]) for field in CLASSIFIER_FIELDS}
                         for ranking, values in {**live, "blend_rule": blend_live}.items()},
        "test_daily": {"blend_mean_warning_days": blend_warning_days,
                       "slack_mean_warning_days": slack_warning_days,
                       "blend_caught": blend_caught, "blend_caught_3plus": blend_caught_3plus},
        "test_counts": {"model_caught_top10": model_caught, "blend_caught_top10": blend_caught_top},
        "monthly_refits": monthly_versions,
        "ladders": _classifier_ladder_block(ladders, validation, ladder_train, ladder_test, ladder_gaps,
                                            ladder_mcc),
        "benchmark": benchmark,
        "references": references,
        "registered_model": model_name,
        "registered_version": version,
        "alias": "champion",
        "train_cutoff": spec["primary_cutoff"],
    }


def run(store=DEFAULT_STORE, results=DEFAULT_RESULTS):
    started = time.monotonic()
    store, results = Path(store), Path(results)
    reset_store(store)
    eta_selection = json.loads(ETA_SELECTION.read_text())
    classifier_selection = json.loads(CLASSIFIER_SELECTION.read_text())

    with mlflow_server(store) as tracking_uri:
        client = configure_mlflow(tracking_uri)
        summary = {
            "checkout_eta": register_eta_model(client, "checkout_eta", "checkout", eta_selection),
            "handover_eta": register_eta_model(client, "handover_eta", "handover", eta_selection),
            "late_risk": register_classifier_model(client, "late_risk", classifier_selection),
        }
    check_store_location(store)
    results.mkdir(parents=True, exist_ok=True)
    (results / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(f"Rebuilt {store} and registered {', '.join(MODELS)} in {time.monotonic() - started:.1f}s")
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--store", default=str(DEFAULT_STORE),
                        help="directory for the SQLite tracking store and artifacts")
    parser.add_argument("--results", default=str(DEFAULT_RESULTS),
                        help="directory for the tracked summary.json")
    args = parser.parse_args()
    run(store=args.store, results=args.results)


if __name__ == "__main__":
    main()
