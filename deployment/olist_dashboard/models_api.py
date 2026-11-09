"""Read the local MLflow registry for the Models tab and proxy its read-only UI.

MLflow is imported lazily so the dashboard still starts with the Python standard
library alone. Only /api/models* and /mlflow/* use this module.
"""

from __future__ import annotations

import csv
import datetime
import json
import math
import os
import re
import statistics
import sys
import tempfile
import threading
from http import client as http_client
from pathlib import Path
from urllib.parse import urlencode, urlparse

APP_ROOT = Path(__file__).resolve().parent
REPO_ROOT = APP_ROOT.parents[1]
for _entry in (str(REPO_ROOT / "src"), str(APP_ROOT)):
    if _entry not in sys.path:
        sys.path.insert(0, _entry)
# MLflow 3 starts new browsers in its GenAI view; this forces the model-training view.
ML_VIEW = "machine_learning"
from olist.paths import DASHBOARD_PHASE2, MLFLOW_STORE
from phase3.mlflow_server import STATIC_PREFIX as MLFLOW_STATIC_PREFIX

DEFAULT_DB = MLFLOW_STORE / "mlflow.db"
DEFAULT_UPSTREAM = "http://127.0.0.1:5050"
LATE_TEST_METRICS = DASHBOARD_PHASE2 / "handover" / "test_metrics.csv"
LATE_TEST_MONTHS = DASHBOARD_PHASE2 / "handover" / "test_monthly_counts.csv"

MODELS = ("checkout_eta", "handover_eta", "late_risk")
DESCRIPTIONS = {
    "checkout_eta": (
        "Delivery time at checkout",
        "Estimated days from purchase to customer delivery, shown when the order is placed.",
        "regression",
    ),
    "handover_eta": (
        "Delivery time at carrier handover",
        "Estimated days from purchase to customer delivery, updated once the parcel reaches the carrier.",
        "regression",
    ),
    "late_risk": (
        "Late-delivery risk at handover",
        "Ranks handed-over orders by how likely they are to miss the promised delivery date.",
        "classification",
    ),
}
REG_FIELDS = ("mae", "rmse", "r2", "mean_signed_error")
CLS_FIELDS = ("pr_auc", "mcc", "roc_auc", "precision", "recall")
LADDER_FIELDS = {
    "regression": ("mae", "rmse", "r2"),
    "classification": ("pr_auc", "roc_auc", "mcc", "precision", "recall", "f1"),
}
ACTION_FIELDS = ("precision", "recall", "f1", "mcc")
# Only estimator hyperparameters and preprocessing belong under "Full hyperparameters".
MODEL_SETTING_KEYS = (
    "family", "alpha", "C", "class_weight", "solver", "max_iter", "max_depth",
    "min_samples_leaf", "max_features", "n_estimators", "n_jobs", "learning_rate",
    "early_stopping", "random_state", "numeric_imputer", "numeric_scaler",
    "categorical_encoder", "purchase_hour", "log_features", "strategy",
)
WINDOW_FIELDS = {
    "regression": (("mae", "validation_window_mae"), ("rmse", "validation_window_rmse"),
                   ("r2", "validation_window_r2")),
    "classification": (
        ("pr_auc", "validation_window_pr_auc"),
        ("roc_auc", "validation_window_roc_auc"),
        ("mcc", "validation_window_mcc_top10pct"),
        ("precision", "validation_window_precision_top10pct"),
        ("recall", "validation_window_recall_top10pct"),
    ),
}
BENCHMARK_VALIDATION_FIELDS = {
    "regression": ("mae", "rmse", "r2"),
    "classification": CLS_FIELDS,
}
BENCHMARK_TEST_FIELDS = {"regression": REG_FIELDS, "classification": ("pr_auc",)}
REFERENCE_TEST_FIELDS = {"regression": REG_FIELDS, "classification": CLS_FIELDS}
BENCHMARK_LABELS = {
    "route_typical_days": "Route's average so far",
    "mean_baseline": "Predict the average for everyone",
    "elapsed_plus_route_baseline": "Days so far + route time",
    "deadline_rule": "Closest to deadline",
    "slack_rule": "Remaining slack",
    "blend_rule": "Blend, monthly refits",
    "random_rule": "Random order",
}

_CACHE = {"key": None, "payload": None}
_BASE_CACHE = {}
_IMPORTANCE_CACHE = {}
_COMPUTE_LOCK = threading.RLock()
MONTH_ABBR = ("Jan", "Feb", "Mar", "Apr", "May", "Jun",
              "Jul", "Aug", "Sep", "Oct", "Nov", "Dec")

ALLOWED_POST_PATHS = {
    "/mlflow/ajax-api/2.0/mlflow/experiments/search",
    "/mlflow/ajax-api/2.0/mlflow/experiments/search-datasets",
    "/mlflow/ajax-api/2.0/mlflow/logged-models/search",
    "/mlflow/ajax-api/2.0/mlflow/metrics/get-history-bulk",
    "/mlflow/ajax-api/2.0/mlflow/metrics/get-history-bulk-interval",
    "/mlflow/ajax-api/2.0/mlflow/model-versions/search",
    "/mlflow/ajax-api/2.0/mlflow/registered-models/search",
    "/mlflow/ajax-api/2.0/mlflow/runs/search",
    "/mlflow/ajax-api/2.0/mlflow/traces/search",
    "/mlflow/ajax-api/3.0/mlflow/issues/search",
    "/mlflow/ajax-api/3.0/mlflow/traces/metrics",
    "/mlflow/ajax-api/3.0/mlflow/traces/search",
    "/mlflow/api/2.0/mlflow/experiments/search",
    "/mlflow/api/2.0/mlflow/experiments/search-datasets",
    "/mlflow/api/2.0/mlflow/logged-models/search",
    "/mlflow/api/2.0/mlflow/metrics/get-history-bulk",
    "/mlflow/api/2.0/mlflow/metrics/get-history-bulk-interval",
    "/mlflow/api/2.0/mlflow/model-versions/search",
    "/mlflow/api/2.0/mlflow/registered-models/search",
    "/mlflow/api/2.0/mlflow/runs/search",
    "/mlflow/api/2.0/mlflow/traces/search",
    "/mlflow/api/3.0/mlflow/issues/search",
    "/mlflow/api/3.0/mlflow/traces/metrics",
    "/mlflow/api/3.0/mlflow/traces/search",
    "/mlflow/graphql",
}

HOP_BY_HOP = {
    "connection",
    "keep-alive",
    "proxy-authenticate",
    "proxy-authorization",
    "te",
    "trailers",
    "transfer-encoding",
    "upgrade",
}


class ModelsUnavailable(Exception):
    def __init__(self, message):
        super().__init__(message)
        self.message = message


def _database_path():
    value = os.environ.get("PRAZO_MLFLOW_DB")
    if not value:
        return DEFAULT_DB
    if value.startswith("sqlite:///"):
        return Path(value[len("sqlite:///"):])
    path = Path(value).expanduser()
    if path.is_dir():
        return path / "mlflow.db"
    return path


def _http_tracking_uri(value):
    value = value.rstrip("/")
    if not value.endswith(MLFLOW_STATIC_PREFIX):
        value += MLFLOW_STATIC_PREFIX
    return value


def _tracking_uri():
    ui = os.environ.get("PRAZO_MLFLOW_UI")
    if ui:
        return _http_tracking_uri(ui)
    return "sqlite:///{}".format(_database_path().resolve())


def _client():
    uri = _tracking_uri()
    if uri.startswith("sqlite:///"):
        path = Path(uri[len("sqlite:///"):])
        if not path.is_file():
            raise ModelsUnavailable(
                "The model registry is missing at {}. Build it from the repository "
                "root with: python -m phase3.registry".format(path)
            )
    try:
        from mlflow.tracking import MlflowClient
    except ImportError as error:
        raise ModelsUnavailable(
            "MLflow is not available to this Python interpreter ({}). Start the "
            "dashboard with the project environment and build the store with "
            "python -m phase3.registry.".format(error)
        )
    return MlflowClient(tracking_uri=uri, registry_uri=uri)


def _history(client, run_id, key):
    points = client.get_metric_history(run_id, key)
    return sorted(((point.step, float(point.value)) for point in points), key=lambda item: item[0])


def _threshold_key(field, threshold):
    return "{}_{}".format(field, threshold) if threshold and field in ACTION_FIELDS else field


def _mean_metrics(run, prefix, fields, threshold=None):
    values = {}
    for field in fields:
        key = prefix + _threshold_key(field, threshold)
        if key in run.data.metrics:
            values[field] = float(run.data.metrics[key])
    return values


def _benchmark_payload(references, kind):
    headline = next((reference for reference in references if reference["role"] == "headline"), None)
    if headline is None:
        return None
    label = headline["label"] or BENCHMARK_LABELS.get(headline["key"], headline["key"].replace("_", " "))
    test = {field: value for field, value in headline["test"].items()
            if field in BENCHMARK_TEST_FIELDS[kind]}
    return {
        "name": headline["key"],
        "label": label,
        "short_label": label,
        "short": headline["short"],
        "hover": headline["hover"],
        "definition": headline["definition"],
        "validation": headline["validation"],
        "test": test,
    }


def _references_payload(client, run, kind):
    """Every reference rule the final run logged: wording, role and per-split metrics."""
    keys = [key for key in (run.data.params.get("reference_keys") or "").split(",") if key]
    validation_fields = BENCHMARK_VALIDATION_FIELDS[kind]
    test_fields = REFERENCE_TEST_FIELDS[kind]
    references = []
    for key in keys:
        prefix = "reference_{}_".format(key)
        histories = {
            field: dict(_history(client, run.info.run_id, "{}validation_window_{}".format(prefix, field)))
            for field in validation_fields
        }
        steps = sorted(set().union(*(set(history) for history in histories.values()))) if histories else []
        windows = [
            {"window": step,
             **{field: history[step] for field, history in histories.items() if step in history}}
            for step in steps
        ]
        gap = run.data.metrics.get(prefix + "test_gap_vs_headline")
        gap_low = run.data.metrics.get(prefix + "test_gap_low_vs_headline")
        gap_high = run.data.metrics.get(prefix + "test_gap_high_vs_headline")
        references.append({
            "key": key,
            "label": run.data.params.get(prefix + "label", key.replace("_", " ")),
            "short": run.data.params.get(prefix + "short", ""),
            "hover": run.data.params.get(prefix + "hover", ""),
            "role": run.data.params.get(prefix + "role", ""),
            "definition": run.data.params.get(prefix + "definition", ""),
            "validation": {
                "mean": _mean_metrics(run, prefix + "validation_mean_", validation_fields),
                "windows": windows,
            },
            "test": _mean_metrics(run, prefix + "test_", test_fields, "top10pct"),
            "test_gap_vs_headline": float(gap) if gap is not None else None,
            "test_gap_low_vs_headline": float(gap_low) if gap_low is not None else None,
            "test_gap_high_vs_headline": float(gap_high) if gap_high is not None else None,
            "test_gap_verdict": run.data.tags.get(prefix + "test_gap_verdict"),
            "test_gap_method": run.data.params.get("test_gap_method"),
        })
    return references


def _format_number(value):
    number = float(value)
    if number.is_integer():
        return str(int(number))
    return ("%g" % number)


def _model_settings(params):
    return {key: value for key, value in params.items() if key in MODEL_SETTING_KEYS}


def _param_summary(params):
    family = params.get("family", "model")
    if family == "ridge" and "alpha" in params:
        return "ridge, alpha {}".format(_format_number(params["alpha"]))
    if family == "linear" and "C" in params:
        return "logistic, C {}".format(params["C"])
    if family == "boost" and "learning_rate" in params:
        return "boost, learning rate {}".format(params["learning_rate"])
    return family


def _candidate_type(candidate):
    if candidate == "mean_baseline":
        return "baseline"
    if candidate == "linear_baseline":
        return "linear"
    if candidate.startswith("ridge_"):
        return "ridge"
    if candidate == "logistic_unregularized":
        return "logistic"
    if candidate.startswith("logistic_"):
        return "regularised_logistic"
    if candidate in ("tree_baseline", "tree_plain"):
        return "tree"
    if candidate.startswith("tree_"):
        return "shallow_tree"
    if candidate.startswith("forest_"):
        return "forest"
    if candidate.startswith("boost_"):
        return "boosting"
    return "other"


def _run_counts(children):
    counts = {}
    for run in children:
        if run.data.tags.get("run_kind") not in ("candidate", "ladder"):
            continue
        key = _candidate_type(run.data.tags.get("candidate"))
        counts[key] = counts.get(key, 0) + 1
    return counts


def _comparison_rows(kind):
    if kind == "regression":
        return [
            {"split": "train", "metric": "mae", "label": "Train MAE (days)"},
            {"split": "validation", "metric": "mae", "label": "Validation MAE (days)"},
            {"split": "test", "metric": "mae", "label": "Test MAE (days)"},
            {"split": "test", "metric": "r2", "label": "Test R\u00b2"},
        ]
    return [
        {"split": "train", "metric": "pr_auc", "label": "Train PR-AUC"},
        {"split": "validation", "metric": "pr_auc", "label": "Validation PR-AUC"},
        {"split": "test", "metric": "pr_auc", "label": "Test PR-AUC"},
        {"split": "test", "metric": "mcc", "label": "Test MCC"},
    ]


def _late_rate():
    try:
        with LATE_TEST_METRICS.open(newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                if row.get("variant") == "primary" and row.get("ranking") == "model":
                    return float(row["late_rate"])
    except (OSError, KeyError, TypeError, ValueError):
        return None
    return None


def _late_test_months():
    try:
        with LATE_TEST_MONTHS.open(newline="", encoding="utf-8") as handle:
            return sorted({row["month"] for row in csv.DictReader(handle) if row.get("month")})
    except (OSError, KeyError, TypeError):
        return []


def _ladder_meta(candidate, params):
    family = params.get("family")
    if candidate == "mean_baseline":
        return "Baseline", "Training-mean benchmark; it ignores every order input.", "Baseline"
    if candidate == "linear_baseline":
        return "Linear", "Linear regression on every available input.", "Linear"
    if family == "ridge" and "alpha" in params:
        alpha = _format_number(float(params["alpha"]))
        return ("Ridge",
                "Linear regression with an L2 penalty (alpha {}).".format(alpha),
                "Ridge")
    if family == "linear" and "C" in params:
        try:
            penalty = float(params["C"])
        except (TypeError, ValueError):
            penalty = None
        if penalty is None or math.isinf(penalty):
            return ("Logistic", "Logistic regression without a penalty.", "Logistic")
        c = _format_number(penalty)
        return ("Regularised logistic",
                "Logistic regression with regularisation strength C {}.".format(c),
                "Regularised logistic")
    if family == "tree":
        depth = params.get("max_depth")
        leaf = params.get("min_samples_leaf")
        if depth in (None, "", "None"):
            return "Full-depth tree", "One unconstrained decision tree.", "Full-depth tree"
        return ("Shallow tree",
                "Decision tree limited to depth {} with at least {} rows per leaf.".format(depth, leaf),
                "Shallow tree")
    if family == "forest":
        trees = params.get("n_estimators", "many")
        leaf = params.get("min_samples_leaf")
        return ("Forest",
                "Random forest of {} trees with at least {} rows per leaf.".format(trees, leaf),
                "Forest")
    if family == "boost":
        rate = params.get("learning_rate")
        detail = " with learning rate {}".format(rate) if rate not in (None, "") else ""
        return "Boosting", "Gradient boosting{}.".format(detail), "Boosting"
    return candidate.replace("_", " "), "Candidate from the Phase 2 selection.", candidate


def _mean_and_se(values):
    if not values:
        return None, None
    mean = sum(values) / len(values)
    se = statistics.stdev(values) / math.sqrt(len(values)) if len(values) > 1 else 0.0
    return mean, se


def _ladder_rung(client, run, kind):
    candidate = run.data.tags.get("candidate")
    params = dict(run.data.params)
    label, description, short_label = _ladder_meta(candidate, params)
    fields = LADDER_FIELDS[kind]
    threshold = "top10pct" if kind == "classification" else None
    train = _mean_metrics(run, "train_", fields, threshold)
    test = _mean_metrics(run, "test_", fields, threshold)
    histories, mean, se = {}, {}, {}
    for field, source in WINDOW_FIELDS[kind]:
        history = dict(_history(client, run.info.run_id, source))
        histories[field] = history
        value_mean, value_se = _mean_and_se(list(history.values()))
        if value_mean is not None:
            mean[field] = value_mean
            se[field] = value_se
    steps = sorted(set().union(*(set(history) for history in histories.values())))
    windows = [{"window": step,
                **{field: history[step] for field, history in histories.items() if step in history}}
               for step in steps]
    gap = run.data.metrics.get("test_gap_vs_headline")
    gap_low = run.data.metrics.get("test_gap_low_vs_headline")
    gap_high = run.data.metrics.get("test_gap_high_vs_headline")
    return {
        "candidate": candidate,
        "run_id": run.info.run_id,
        "rung": int(run.data.tags.get("ladder_rung", -1)),
        "type": _candidate_type(candidate),
        "label": label,
        "description": description,
        "short_label": short_label,
        "params": _model_settings(params),
        "train": train,
        "validation": {"mean": mean, "se": se, "windows": windows},
        "test": test,
        "test_gap_vs_headline": float(gap) if gap is not None else None,
        "test_gap_low_vs_headline": float(gap_low) if gap_low is not None else None,
        "test_gap_high_vs_headline": float(gap_high) if gap_high is not None else None,
        "test_gap_verdict": run.data.tags.get("test_gap_verdict"),
        "test_gap_method": run.data.params.get("test_gap_method"),
    }


def _ladders_payload(client, runs, kind):
    families = {}
    for run in runs:
        family = run.data.tags.get("ladder_family")
        families.setdefault(family, []).append(_ladder_rung(client, run, kind))
    return {family: sorted(rows, key=lambda row: row["rung"]) for family, rows in families.items()}


def _champion_line(client, name, run, children):
    versions = client.search_model_versions("name = '{}'".format(name))
    candidates = {item.data.tags.get("candidate") for item in children
                  if item.data.tags.get("run_kind") in ("candidate", "ladder")}
    started = run.info.start_time
    started_at = (datetime.datetime.fromtimestamp(started / 1000.0, tz=datetime.timezone.utc)
                  .isoformat().replace("+00:00", "Z") if started is not None else None)
    version = client.get_model_version_by_alias(name, "champion")
    return {
        "registered_model": name,
        "champion_version": int(version.version),
        "total_versions": len(versions),
        "candidate_count": len(candidates),
        "final_run_started_at": started_at,
    }


def _tuning_key(params):
    family = params.get("family")
    if family == "ridge" and "alpha" in params:
        return "alpha"
    if family == "linear" and "C" in params:
        return "C"
    return None


def _grid(client, candidates, regression, champion, champion_params, selection_pool):
    metric = "mae" if regression else "pr_auc"
    window_key = "validation_window_{}".format(metric)
    mean_key = "validation_mean_{}".format(metric)
    histories = {}
    means = {}
    for run in candidates:
        name = run.data.tags.get("candidate")
        history = _history(client, run.info.run_id, window_key)
        histories[name] = dict(history)
        if mean_key in run.data.metrics:
            means[name] = float(run.data.metrics[mean_key])
        elif history:
            means[name] = statistics.mean(value for _, value in history)
        else:
            means[name] = None
    available = [name for name, value in means.items() if value is not None]
    best_name = (min if regression else max)(available, key=lambda name: means[name])
    best_values = histories.get(best_name, {})
    tuning = _tuning_key(champion_params)

    edges = set()
    if tuning:
        family = champion_params.get("family")
        tuned = []
        for run in candidates:
            params = run.data.params
            if params.get("family") != family or tuning not in params:
                continue
            try:
                value = float(params[tuning])
            except (TypeError, ValueError):
                continue
            if math.isfinite(value) and value > 0:
                tuned.append((value, run.data.tags.get("candidate")))
        if tuned:
            edges.add(min(tuned)[1])
            edges.add(max(tuned)[1])

    rows = []
    for run in candidates:
        name = run.data.tags.get("candidate")
        mean = means.get(name)
        history_values = list(histories.get(name, {}).values())
        if mean is None:
            gap = se = signed_gap = None
            within = False
            validation_se = None
        else:
            signed_gap = mean - means[best_name]
            gap = abs(signed_gap)
            steps = sorted(set(best_values) & set(histories.get(name, {})))
            differences = [histories[name][step] - best_values[step] for step in steps]
            se = statistics.stdev(differences) / math.sqrt(len(differences)) if len(differences) > 1 else 0.0
            within = gap <= se + 1e-12
            validation_se = (statistics.stdev(history_values) / math.sqrt(len(history_values))
                             if len(history_values) > 1 else 0.0)
        params = dict(run.data.params)
        tuning_value = None
        if tuning and params.get(tuning) not in (None, "", "None"):
            try:
                value = float(params[tuning])
            except (TypeError, ValueError):
                value = None
            if value is not None and math.isfinite(value):
                tuning_value = value
        rows.append({
            "name": name,
            "validation_mean": mean,
            "validation_se": validation_se,
            "gap_to_best": gap,
            "gap_signed": signed_gap,
            "se_of_gap": se,
            "within_one_se": bool(within),
            "selected": run.data.tags.get("selected") == "true" or name == champion,
            "selection_pool": name in selection_pool,
            "grid_edge": name in edges,
            "tuning_value": tuning_value,
            "params": _model_settings(params),
        })
    return rows, best_name


def _child_runs(client, run, experiment_id):
    parent = run.data.tags.get("mlflow.parentRunId")
    if not parent:
        return []
    children = client.search_runs(
        [experiment_id],
        filter_string="tags.mlflow.parentRunId = '{}'".format(parent),
        max_results=5000,
    )
    return sorted(children, key=lambda item: (item.info.start_time or 0, item.info.run_id))


def _candidate_runs(client, run, experiment_id):
    return [item for item in _child_runs(client, run, experiment_id)
            if item.data.tags.get("run_kind") == "candidate"]


def _test_months(run):
    pattern = re.compile(r"^test_month_(.+)_(mae|rmse|r2|mean_signed_error)$")
    months = {}
    for key, value in run.data.metrics.items():
        match = pattern.match(key)
        if match:
            months.setdefault(match.group(1), {})[match.group(2)] = float(value)
    return [{"month": month, "metrics": months[month]} for month in sorted(months)]


def _regression_payload(client, name, version, run, candidates, experiment_id):
    champion = run.data.tags.get("candidate")
    champion_run = next((item for item in candidates if item.data.tags.get("candidate") == champion), None)
    windows = []
    mean = {}
    if champion_run is not None:
        histories = {
            field: dict(_history(client, champion_run.info.run_id, source))
            for field, source in WINDOW_FIELDS["regression"]
        }
        steps = sorted(set().union(*(set(history) for history in histories.values())))
        for step in steps:
            windows.append({
                "window": step,
                **{field: history[step] for field, history in histories.items() if step in history},
            })
        for field, history in histories.items():
            if history:
                mean[field] = statistics.mean(history.values())
    params = dict(run.data.params)
    pool = [key for key in (run.data.tags.get("selection_pool") or "").split(",") if key]
    grid, best = _grid(client, candidates, True, champion, params, pool)
    return {
        "train": _mean_metrics(run, "train_apparent_", REG_FIELDS),
        "validation": {"mean": mean, "windows": windows},
        "test": {"overall": _mean_metrics(run, "test_overall_", REG_FIELDS),
                 "months": _test_months(run)},
    }, grid, best


def _classifier_payload(client, name, version, run, candidates, experiment_id, references):
    champion = run.data.tags.get("candidate")
    champion_run = next((item for item in candidates if item.data.tags.get("candidate") == champion), None)
    windows = []
    mean = {}
    if champion_run is not None:
        histories = {
            field: dict(_history(client, champion_run.info.run_id, source))
            for field, source in WINDOW_FIELDS["classification"]
        }
        steps = sorted(set().union(*(set(history) for history in histories.values())))
        for step in steps:
            windows.append({
                "window": step,
                **{field: history[step] for field, history in histories.items() if step in history},
            })
        for field, history in histories.items():
            if history:
                mean[field] = statistics.mean(history.values())
    params = dict(run.data.params)
    pool = [key for key in (run.data.tags.get("selection_pool") or "").split(",") if key]
    grid, best = _grid(client, candidates, False, champion, params, pool)
    headline = next((reference for reference in references if reference["role"] == "headline"), None)
    definition = headline["definition"] if headline else ""
    match = re.search(r"top\s+([0-9]+(?:\.[0-9]+)?)\s*%", definition)
    return {
        "train": _mean_metrics(run, "train_apparent_", CLS_FIELDS, "top10pct"),
        "validation": {"mean": mean, "windows": windows},
        "test": {
            "model": _mean_metrics(run, "test_", CLS_FIELDS, "top10pct"),
            "slack_rule": dict(headline["test"]) if headline else {},
        },
        "review_share_percent": float(match.group(1)) if match else None,
    }, grid, best


def _live_model(client, date):
    """The monthly late-risk version that serves ``date``, with its Models-tab line."""
    from phase3 import serving

    version = serving.late_risk_version_for_date(date, client=client)
    refit = datetime.date.fromisoformat(version.tags["refit_date"])
    label = "Live model: retrained {} {} {}".format(
        refit.day, MONTH_ABBR[refit.month - 1], refit.year)
    return {"version": int(version.version), "refit_date": version.tags["refit_date"], "label": label}


def _model_payload(client, name, date=None):
    version = client.get_model_version_by_alias(name, "champion")
    run = client.get_run(version.run_id)
    experiment_id = run.info.experiment_id
    children = _child_runs(client, run, experiment_id)
    candidates = [item for item in children if item.data.tags.get("run_kind") == "candidate"]
    ladder_runs = [item for item in children if "ladder_family" in item.data.tags]
    title, purpose, kind = DESCRIPTIONS[name]
    references = _references_payload(client, run, kind)
    if kind == "regression":
        metrics, grid, best = _regression_payload(client, name, version, run, candidates, experiment_id)
    else:
        metrics, grid, best = _classifier_payload(client, name, version, run, candidates, experiment_id,
                                                  references)
    review_share = metrics.pop("review_share_percent", None)
    params = dict(run.data.params)
    benchmark = _benchmark_payload(references, kind)
    ladders = _ladders_payload(client, ladder_runs, kind)
    run_counts = _run_counts(children)
    if kind == "regression":
        test_months = [item["month"] for item in metrics["test"]["months"]]
        late_rate = None
    else:
        test_months = _late_test_months()
        late_rate = _late_rate()
    payload = {
        "name": name,
        "title": title,
        "purpose": purpose,
        "kind": kind,
        "version": int(version.version),
        "candidate": run.data.tags.get("candidate"),
        "param_summary": _param_summary(params),
        "params": _model_settings(params),
        "train_cutoff": run.data.tags.get("train_cutoff") or version.tags.get("train_cutoff"),
        "review_share_percent": review_share,
        "primary_metric": "mae" if kind == "regression" else "pr_auc",
        "primary_label": "MAE (days)" if kind == "regression" else "PR-AUC",
        "lower_is_better": kind == "regression",
        "metrics": metrics,
        "grid": grid,
        "best_candidate": best,
        "ladders": ladders,
        "run_counts": run_counts,
        "tracked_runs": sum(run_counts.values()),
        "comparison_rows": _comparison_rows(kind),
        "has_slack_rule": kind == "classification",
        "test_months": test_months,
        "late_rate": late_rate,
        "champion_line": _champion_line(client, name, run, children),
        "benchmark": benchmark,
        "references": references,
        "links": {
            "experiment_id": experiment_id,
            "run_id": version.run_id,
            "experiment_path": "/mlflow/#/experiments/{}/runs?workflowType={}".format(experiment_id, ML_VIEW),
            "run_path": "/mlflow/#/experiments/{}/runs/{}?workflowType={}".format(experiment_id, version.run_id, ML_VIEW),
            "compare_path": "/mlflow/#/experiments/{}/runs?{}".format(
                experiment_id, urlencode({"viewStateShareKey": "ladder", "workflowType": ML_VIEW})),
        },
    }
    if kind == "classification":
        payload["warning_days"] = {
            "blend": (float(run.data.metrics["test_blend_mean_warning_days"])
                      if "test_blend_mean_warning_days" in run.data.metrics else None),
            "slack": (float(run.data.metrics["test_slack_mean_warning_days"])
                      if "test_slack_mean_warning_days" in run.data.metrics else None),
        }
        payload["live_model"] = _live_model(client, date) if date else None
    return payload


def models_response(date=None):
    try:
        uri = _tracking_uri()
        key = (uri, str(date))
        if _CACHE.get("key") == key and _CACHE.get("payload") is not None:
            return 200, _CACHE["payload"]
        with _COMPUTE_LOCK:
            if "key" not in _CACHE:
                _BASE_CACHE.clear()
            if _CACHE.get("key") == key and _CACHE.get("payload") is not None:
                return 200, _CACHE["payload"]
            client = _client()
            base = _BASE_CACHE.get(uri)
            if base is None:
                base = [_model_payload(client, name, None) for name in MODELS]
                _BASE_CACHE[uri] = base
            payload = []
            for item in base:
                if item.get("kind") == "classification":
                    copy_item = dict(item)
                    copy_item["live_model"] = _live_model(client, date) if date else None
                    payload.append(copy_item)
                else:
                    payload.append(item)
            _CACHE.update({"key": key, "payload": payload})
            return 200, payload
    except ModelsUnavailable as error:
        return 503, {"ok": False, "error": {"code": "models_unavailable", "message": error.message}}
    except Exception as error:
        return 503, {
            "ok": False,
            "error": {
                "code": "models_unavailable",
                "message": "The model registry could not be read: {}".format(error),
            },
        }


def importance_response(name):
    if name not in MODELS:
        return 404, {"ok": False, "error": {"code": "not_found", "message": "Unknown model."}}
    try:
        uri = _tracking_uri()
        key = (uri, name)
        if key in _IMPORTANCE_CACHE:
            return 200, _IMPORTANCE_CACHE[key]
        with _COMPUTE_LOCK:
            if key not in _IMPORTANCE_CACHE:
                client = _client()
                version = client.get_model_version_by_alias(name, "champion")
                with tempfile.TemporaryDirectory() as tmp:
                    path = Path(client.download_artifacts(version.run_id, "permutation_importance.csv", tmp))
                    with path.open(newline="", encoding="utf-8") as handle:
                        rows = list(csv.DictReader(handle))
                rows.sort(key=lambda row: float(row["importance_mean"]), reverse=True)
                _IMPORTANCE_CACHE[key] = {
                    "model": name,
                    "total": len(rows),
                    "inputs": [
                        {
                            "input": row["input"],
                            "key": row.get("key") or "",
                            "members": row.get("members") or "",
                            "importance_mean": float(row["importance_mean"]),
                            "importance_std": float(row["importance_std"]),
                            "direction": row.get("direction") or "",
                        }
                        for row in rows[:15]
                    ],
                }
            return 200, _IMPORTANCE_CACHE[key]
    except ModelsUnavailable as error:
        return 503, {"ok": False, "error": {"code": "models_unavailable", "message": error.message}}
    except Exception as error:
        return 503, {
            "ok": False,
            "error": {
                "code": "models_unavailable",
                "message": "The permutation importance could not be read: {}".format(error),
            },
        }


def _send_json(handler, status, payload, close=False):
    body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    handler.send_response(status)
    handler.send_header("Content-Type", "application/json; charset=utf-8")
    handler.send_header("Content-Length", str(len(body)))
    handler.send_header("Cache-Control", "no-store")
    handler.send_header("X-Content-Type-Options", "nosniff")
    if close:
        handler.send_header("Connection", "close")
        handler.close_connection = True
    handler.end_headers()
    handler.wfile.write(body)


def _upstream():
    value = os.environ.get("PRAZO_MLFLOW_UI") or DEFAULT_UPSTREAM
    parsed = urlparse(value)
    if parsed.scheme not in ("http", "https") or not parsed.hostname:
        raise ModelsUnavailable("PRAZO_MLFLOW_UI is not a valid http URL: {}".format(value))
    return parsed


def _proxy_forbidden(handler):
    # The refused request's body may be unread; close the connection so those bytes
    # are never parsed as the next request on a keep-alive connection.
    _send_json(handler, 403, {
        "ok": False,
        "error": {
            "code": "forbidden_proxy_request",
            "message": "This method or path is not allowed through the read-only MLflow proxy.",
        },
    }, close=True)


def proxy(handler):
    method = handler.command
    parsed = urlparse(handler.path)
    if method not in ("GET", "HEAD", "POST"):
        _proxy_forbidden(handler)
        return
    if method == "POST" and parsed.path not in ALLOWED_POST_PATHS:
        _proxy_forbidden(handler)
        return
    try:
        upstream = _upstream()
    except ModelsUnavailable as error:
        _send_json(handler, 502, {"ok": False, "error": {"code": "proxy_unavailable", "message": error.message}})
        return

    body = None
    if method == "POST":
        try:
            length = int(handler.headers.get("Content-Length", "0"))
        except ValueError:
            length = 0
        body = handler.rfile.read(length) if length > 0 else b""
        if parsed.path == "/mlflow/graphql" and re.search(r"\bmutation\b", body.decode("utf-8", "ignore"), re.IGNORECASE):
            _proxy_forbidden(handler)
            return

    connection_class = (
        http_client.HTTPSConnection if upstream.scheme == "https" else http_client.HTTPConnection
    )
    connection = connection_class(upstream.hostname, upstream.port or (443 if upstream.scheme == "https" else 80), timeout=60)
    path = parsed.path
    if parsed.query:
        path += "?" + parsed.query
    headers = {}
    for key, value in handler.headers.items():
        lower = key.lower()
        if lower in HOP_BY_HOP or lower in ("host", "content-length", "origin", "referer"):
            continue
        headers[key] = value
    headers["Host"] = upstream.netloc
    if body is not None:
        headers["Content-Length"] = str(len(body))
    try:
        connection.request(method, path, body=body, headers=headers)
        response = connection.getresponse()
    except OSError as error:
        _send_json(handler, 502, {
            "ok": False,
            "error": {
                "code": "proxy_unavailable",
                "message": "The MLflow UI at {} is not reachable ({}). Start it with deployment/olist_dashboard/run_local.sh.".format(
                    upstream.geturl(), error
                ),
            },
        })
        return

    handler.send_response(response.status)
    for key, value in response.getheaders():
        lower = key.lower()
        if lower in HOP_BY_HOP or lower == "content-length":
            continue
        if lower == "location":
            value = _rewrite_location(value, upstream)
        handler.send_header(key, value)
    length = response.getheader("Content-Length")
    if length is not None:
        handler.send_header("Content-Length", length)
    else:
        handler.send_header("Connection", "close")
        handler.close_connection = True
    handler.end_headers()
    try:
        if method != "HEAD":
            while True:
                chunk = response.read(65536)
                if not chunk:
                    break
                handler.wfile.write(chunk)
                handler.wfile.flush()
    except OSError:
        handler.close_connection = True
    finally:
        response.close()
        connection.close()


def _rewrite_location(value, upstream):
    base = "{}://{}".format(upstream.scheme, upstream.netloc)
    if value.startswith(base):
        return value[len(base):] or "/mlflow/"
    return value
