"""Drift monitoring for the dashboard's Monitoring section.

Builds a per-as-of-date monitor from the Phase 2 selected models and the
point-in-time order cohort in `phase3.drift`. The page answers one question
per model: is it still OK, and what should I do?

- The status cards come from the model checks only: the operations rule against
  the model on mature test months (the same paired week bootstrap as the Models
  tab), mean signed error against its validation range, and the late-risk blend's
  ROC-AUC. PR-AUC is still shown for the blend, but it moves with the late rate,
  so the monthly Watch check uses ROC-AUC. The late-risk score is the dashboard's
  50/50 blend, not the classifier alone, and each test order is scored by the
  monthly refit live at its handover.
- Inputs are an early-warning panel with one level, Unusual. It starts above the
  largest score seen in the 11 validation monthly readings, when the models were
  known to be good. Evidently computes the displayed scores and receives the same
  per-input thresholds for its report. Unusual inputs never change a model's
  status.
- Outcomes use one maturity rule: a month's delivery days and late rate are
  ready once at least 95% of its orders are delivered. Open orders past their
  promise date count as late; open orders not yet past it are unknown and are
  not counted.

Run from the repository root:
    python -m phase3.monitor --as-of 2018-08-22
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import statistics
import tempfile
import time
from pathlib import Path

# Evidently must never phone home.
os.environ.setdefault("EVIDENTLY_DISABLE_TELEMETRY", "1")
os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp
from sklearn.metrics import roc_auc_score
from evidently import DataDefinition, Dataset, Regression, Report
from evidently.core.report import Context
from evidently.legacy.calculations.stattests.registry import StatTest, register_stattest
from evidently.legacy.core import ColumnType
from evidently.metrics import MAE, MeanError, ValueDrift

from olist.paths import DASHBOARD_PHASE2, MLFLOW_STORE, REPO_ROOT
from phase3 import drift, registry, serving

WATCH_KS = drift.WATCH_KS
VALIDATION_SD_MULTIPLIER = 2.0
MIN_MONTH_ORDERS = 500
MIN_LATE_ORDERS = 50
MIN_POOLED_WEEKS = 4
DELIVERY_MATURE_SHARE = 0.95
IMPORTANCE_CUTOFF = drift.INPUT_IMPORTANCE_CUTOFF
PSI_CUTOFFS = (0.1, 0.25)

KS_DISTANCE_NAME = "ks_distance"
TV_DISTANCE_NAME = "tv_distance"
DRIFT_THRESHOLDS_ARTIFACT = "drift_thresholds.json"

INPUT_LABELS = {
    "promised_days": "Promised delivery time",
    "promise_slack": "Promise slack",
    "route_typical_days": "Route's usual delivery time",
    "seller_ship_days": "Seller's usual dispatch time",
    "distance_km": "Distance",
    "same_state": "Same state",
    "purchase_hour": "Purchase hour",
    "n_items": "Number of items",
    "n_products": "Number of products",
    "n_sellers": "Number of sellers",
    "total_price": "Order value",
    "total_freight": "Freight cost",
    "total_weight_g": "Order weight",
    "total_volume_cm3": "Order volume",
    "max_installments": "Installments",
    "customer_state": "Customer state",
    "seller_state": "Seller state",
    "product_category": "Product category",
    "payment_type": "Payment type",
    "purchase_dayofweek": "Day of week",
    "approval_days": "Approval time",
    "handover_days": "Days from purchase to handover",
    "route_transit_90d_at_handover": "Route's usual transit time",
    "remaining_slack": "Remaining slack",
    "past_promise_at_handover": "Already past promise",
    "slack_shortfall": "Shortfall vs usual",
    "state_backlog_recent_vs_8wk": "Regional backlog",
    "zip3_transit_90d_at_handover": "Local route's usual delivery time",
    "seller_handover_90d_at_handover": "Seller's usual dispatch time",
}

MODEL_META = {
    "checkout": {
        "name": "checkout_eta",
        "title": "Delivery time at checkout",
        "short": "Checkout",
        "kind": "regression",
        "rule_label": "Route's average so far",
    },
    "handover": {
        "name": "handover_eta",
        "title": "Delivery time at carrier handover",
        "short": "Handover",
        "kind": "regression",
        "rule_label": "Days so far + route time",
    },
    "late_warning": {
        "name": "late_risk",
        "title": "Late-delivery risk at handover",
        "short": "Late risk",
        "kind": "classification",
        "rule_label": "Remaining slack",
    },
}

OUTCOME_META = {
    "mean_delivery_days": {"label": "Mean delivery days", "unit": "days"},
    "mean_promised_days": {"label": "Mean promised days", "unit": "days"},
    "late_rate": {"label": "Late rate", "unit": "percent"},
}

OUTCOME_MATURITY = {
    "mean_delivery_days": "delivery_mature",
    "mean_promised_days": None,
    "late_rate": "late_mature",
}

NOTE = (
    "Is each model still fit to use? Only the model checks can trigger retraining; "
    "inputs and outcomes explain why."
)

_LEVEL_ORDER = {"none": 0, "watch": 1, "act": 2}
_STATE_CACHE = {}
_MONITOR_CACHE = {}
_REPORT_CACHE = {}
_STAT_TESTS_READY = False


class MonitorUnavailable(Exception):
    """Raised when the monitoring data or its dependencies cannot be prepared."""


# --------------------------------------------------------------------------- Evidently stat tests

def _ks_distance_stat(reference, current, feature_type, threshold):
    """Two-sample KS distance (not the p-value) with drift above the threshold."""
    a = pd.Series(reference).dropna()
    b = pd.Series(current).dropna()
    if len(a) == 0 or len(b) == 0:
        return float("nan"), False
    # The p-value is never used; the asymptotic method skips its exact calculation
    # and leaves the distance itself bit-for-bit unchanged.
    distance = float(ks_2samp(a, b, method="asymp").statistic)
    return distance, distance > threshold


def _tv_distance_stat(reference, current, feature_type, threshold):
    """Total-variation distance with missing values treated as their own category."""
    a = pd.Series(reference).fillna("(missing)").value_counts(normalize=True)
    b = pd.Series(current).fillna("(missing)").value_counts(normalize=True)
    union = a.to_dict() | b.to_dict()
    distance = 0.5 * sum(abs(a.get(key, 0.0) - b.get(key, 0.0)) for key in union)
    return float(distance), distance > threshold


KS_DISTANCE_STAT = StatTest(
    name=KS_DISTANCE_NAME,
    display_name="KS distance",
    allowed_feature_types=[ColumnType.Numerical],
    default_threshold=WATCH_KS,
)
TV_DISTANCE_STAT = StatTest(
    name=TV_DISTANCE_NAME,
    display_name="Total variation distance",
    allowed_feature_types=[ColumnType.Categorical],
    default_threshold=WATCH_KS,
)


def _register_stat_tests():
    global _STAT_TESTS_READY
    if _STAT_TESTS_READY:
        return
    register_stattest(KS_DISTANCE_STAT, _ks_distance_stat)
    register_stattest(TV_DISTANCE_STAT, _tv_distance_stat)
    _STAT_TESTS_READY = True


_register_stat_tests()


# ----------------------------------------------------------------------------------------- helpers

def validate_date(value):
    """Return YYYY-MM-DD or raise ValueError with a clear message."""
    if not isinstance(value, str) or not value.strip():
        raise ValueError("A date is required, for example 2018-08-22.")
    text = value.strip()
    if len(text) != 10 or text[4] != "-" or text[7] != "-":
        raise ValueError("The date must be in YYYY-MM-DD form, for example 2018-08-22.")
    try:
        parsed = pd.Timestamp(text)
    except (TypeError, ValueError) as error:
        raise ValueError("The date must be a valid YYYY-MM-DD date.") from error
    if parsed != parsed.normalize() or parsed.strftime("%Y-%m-%d") != text:
        raise ValueError("The date must be a valid YYYY-MM-DD date.")
    return text


def cutoff_for(date_str):
    """Only data known by 09:00 on the selected date is used."""
    return pd.Timestamp(date_str) + pd.Timedelta(hours=9)


def _round(value, digits=4):
    if value is None:
        return None
    number = float(value)
    if not math.isfinite(number):
        return None
    return round(number, digits)


def _level_max(*levels):
    return max(levels, key=lambda level: _LEVEL_ORDER.get(level, 0))


def _month_name(label):
    return pd.Period(label).strftime("%b %Y")


def _feature_label(feature):
    """Plain input name for the screen; never expose snake_case."""
    if feature in INPUT_LABELS:
        return INPUT_LABELS[feature]
    return feature.replace("_", " ").strip().capitalize()


def _bias_bounds(validation):
    """Watch outside the validation bias range; Act one validation SD beyond it."""
    signed = validation.get("signed") if validation else None
    if not signed:
        return None
    sd = signed["sd"]
    return {
        "min": signed["min"],
        "max": signed["max"],
        "sd": sd,
        "act_min": _round(signed["min"] - sd),
        "act_max": _round(signed["max"] + sd),
    }


def bias_level(value, bounds):
    """Positive means the model predicts too long. Act is a clear move beyond validation."""
    if value is None or bounds is None:
        return "none"
    if value < bounds["act_min"] or value > bounds["act_max"]:
        return "act"
    if value < bounds["min"] or value > bounds["max"]:
        return "watch"
    return "none"


def performance_level(value, validation, higher_is_worse):
    """Watch when the model is more than two validation SDs worse than its validation mean."""
    if value is None or validation is None:
        return "none"
    mean, sd = validation.get("mean"), validation.get("sd")
    if mean is None or sd is None:
        return "none"
    worse = value - mean if higher_is_worse else mean - value
    return "watch" if worse > VALIDATION_SD_MULTIPLIER * sd else "none"


def _rule_level(gap):
    """Act only when the operations rule is clearly better; Watch only for a real model decline."""
    if gap is None:
        return "none"
    return "act" if gap["verdict"] == "worse" else "none"


def _input_level(score, threshold):
    """Unusual only above the highest score seen in the validation periods."""
    if score is None or threshold is None:
        return "normal"
    return "unusual" if score > threshold else "normal"


# --------------------------------------------------------------------------------- monthly outcomes

def month_outcomes(orders, cutoff):
    """Monthly outcomes with one maturity rule for delivery days and the late rate.

    A month is ready once at least 95% of its orders are delivered by the cutoff.
    Delivered orders are late or on time as recorded. An order still open past its
    promise date is certainly late, so it counts as late. An open order not yet
    past its promise date is unknown and is left out of the late rate.
    """
    month = pd.Series(drift.month_label(orders["order_purchase_timestamp"]), index=orders.index)
    promised_at = orders["order_purchase_timestamp"] + pd.to_timedelta(orders["promised_days"], unit="D")
    delivered = orders["delivery_days"].notna() & orders["outcome_available_at"].le(cutoff)
    rows = []
    for label in sorted(month.unique()):
        mask = month.eq(label)
        sub = orders.loc[mask]
        done = delivered.loc[mask]
        count = int(len(sub))
        delivered_count = int(done.sum())
        share = delivered_count / count if count else 0.0
        open_mask = ~done
        past_promise = promised_at.loc[mask].le(cutoff)
        open_past = open_mask & past_promise
        unknown = open_mask & ~past_promise
        ready = bool(count) and share >= DELIVERY_MATURE_SHARE
        late_orders = None
        late_rate = None
        delivered_late = None
        delivered_rate = None
        known_orders = None
        if delivered_count:
            delivered_late = int(sub.loc[done, "is_late"].sum())
            delivered_rate = delivered_late / delivered_count
        if ready and delivered_count:
            known_orders = delivered_count + int(open_past.sum())
            late_orders = delivered_late + int(open_past.sum())
            late_rate = late_orders / known_orders if known_orders else None
        rows.append({
            "month": label,
            "orders": count,
            "matured_orders": delivered_count,
            "share_matured": round(share, 4),
            "open_orders": int(open_mask.sum()),
            "open_past_promise": int(open_past.sum()),
            "unknown_orders": int(unknown.sum()),
            "known_orders": known_orders,
            "mean_delivery_days": _round(sub.loc[done, "delivery_days"].mean()) if delivered_count else None,
            "mean_promised_days": _round(sub["promised_days"].mean()),
            "late_orders": late_orders,
            "late_rate": _round(late_rate),
            "delivered_late_orders": delivered_late,
            "late_rate_delivered_only": _round(delivered_rate),
            "late_mature": ready,
            "delivery_mature": ready,
        })
    return rows


def reference_ranges(visible, cutoff, reference_cutoff=None):
    """Outcome ranges from training months that were mature when the reference period ended.

    A month counts only if at least 95% of the orders placed in it by the reference
    cutoff were delivered by then. Otherwise its outcomes were still arriving, and
    the surviving fast orders would drag the lower bounds down. The values come from
    the training split, so the range describes the reference period itself.
    """
    train = visible[visible["split"].eq("train")]
    training_cutoff = pd.Timestamp(reference_cutoff) if reference_cutoff is not None else cutoff
    as_of = min(pd.Timestamp(cutoff), training_cutoff)
    universe = visible[visible["order_purchase_timestamp"].lt(as_of)]
    mature = {row["month"]: row["delivery_mature"]
              for row in month_outcomes(universe, as_of)}
    months = [row for row in month_outcomes(train, as_of)
              if row["orders"] >= MIN_MONTH_ORDERS and mature.get(row["month"], False)]
    ranges = {}
    for metric in OUTCOME_META:
        values = [row[metric] for row in months if row[metric] is not None]
        if values:
            low, high = min(values), max(values)
            ranges[metric] = {"min": _round(low), "max": _round(high), "width": _round(high - low)}
        else:
            ranges[metric] = None
    return ranges, len(months)


def outcome_level(value, bounds):
    """Outside the reference range is Watch; more than its width outside is Act."""
    if value is None or bounds is None:
        return "none"
    low, high, width = bounds["min"], bounds["max"], bounds["width"]
    if value < low:
        distance = low - value
    elif value > high:
        distance = value - high
    else:
        return "none"
    return "act" if distance > width else "watch"


# ------------------------------------------------------------------------------------ loading state

def _validation_stats(values, signed_values=None):
    values = [float(value) for value in values]
    mean = float(np.mean(values))
    sd = float(statistics.stdev(values)) if len(values) > 1 else 0.0
    stats = {"mean": _round(mean), "sd": _round(sd), "n_windows": len(values),
             "values": [_round(v) for v in values]}
    if signed_values is not None:
        signed = [float(value) for value in signed_values]
        signed_sd = float(statistics.stdev(signed)) if len(signed) > 1 else 0.0
        stats["signed"] = {
            "min": _round(min(signed)),
            "max": _round(max(signed)),
            "sd": _round(signed_sd),
            "values": [_round(v) for v in signed],
        }
    return stats


def _read_validation_from_files(root, selected):
    eta = DASHBOARD_PHASE2 / "eta" / "ladder_windows.csv"
    if not eta.is_file():
        return None
    try:
        eta_frame = pd.read_csv(eta)
        stats = {}
        for stage in ("checkout", "handover"):
            candidate = selected[stage]["candidate"]
            rows = eta_frame[(eta_frame["stage"] == stage) & (eta_frame["model"] == candidate)]
            values = [float(value) for value in rows["mae"].tolist()]
            signed = [float(value) for value in rows["mean_signed_error"].tolist()]
            if not values:
                return None
            stats[stage] = _validation_stats(values, signed)
        return stats
    except (OSError, KeyError, TypeError, ValueError):
        return None


def _validation_from_data(train, spec, contracts, selected):
    """Reproduce drift.validation_metrics per window, so the SD and sign are available."""
    folds = drift.chronological_folds(train, spec)
    stats = {}
    for stage in ("checkout", "handover"):
        contract = contracts[stage]
        columns = sum(drift.feature_columns(contract), [])
        values, signed = [], []
        for fit, valid in folds:
            pipeline = drift.make_stage_pipeline(
                selected[stage]["family"], selected[stage]["parameters"], contract)
            with drift.threadpool_limits(limits=1):
                pipeline.fit(train.iloc[fit][columns], train.iloc[fit][drift.TARGET])
                predicted = pipeline.predict(train.iloc[valid][columns])
            metrics = drift.regression_metrics(train.iloc[valid][drift.TARGET], predicted)
            values.append(metrics["mae"])
            signed.append(metrics["mean_signed_error"])
        stats[stage] = _validation_stats(values, signed)
    return stats


def _contracts_from_spec(spec):
    """The Phase 2 feature contracts the selected models were trained with."""
    return {
        "checkout": drift.stage_spec(spec, "checkout"),
        "handover": drift.stage_spec(spec, "handover"),
        "late_warning": drift.variant_spec(spec),
    }


def _tracking_uri():
    """The MLflow tracking URI: the dashboard's server when PRAZO_MLFLOW_UI is set."""
    ui = os.environ.get("PRAZO_MLFLOW_UI")
    if ui:
        return serving.tracking_uri()
    path = _mlflow_database_path()
    if not path.is_file():
        raise MonitorUnavailable(
            f"The model registry is missing at {path}. Build it from the repository root "
            "with: python -m phase3.registry")
    return "sqlite:///{}".format(path.resolve())


def _load_champion_models():
    """Load the three registered champions by alias; never refit as a fallback."""
    uri = _tracking_uri()
    try:
        champions = serving.load_champions(uri=uri)
    except Exception as error:  # noqa: BLE001 - a missing store or alias must name the rebuild command
        raise MonitorUnavailable(
            f"The champion models could not be loaded from {uri} ({error}). Rebuild the model "
            "store from the repository root with: python -m phase3.registry") from error
    return {
        "checkout": champions.models["checkout_eta"],
        "handover": champions.models["handover_eta"],
        "late_warning": champions.models["late_risk"],
    }


def load_state(root=REPO_ROOT):
    """Load the cohort, champion models and validation numbers once per data root."""
    root = Path(root).resolve()
    key = str(root)
    if key in _STATE_CACHE:
        return _STATE_CACHE[key]
    try:
        cohort, spec, coverage = drift.load_cohort(root)
        selected = drift.selected_models()
        models = _load_champion_models()
        contracts = _contracts_from_spec(spec)
        train = cohort[cohort["split"].eq("train")].reset_index(drop=True)
        validation = _read_validation_from_files(root, selected)
        if validation is None:
            validation = _validation_from_data(train, spec, contracts, selected)
        handover_contract = contracts["handover"]
        handover_columns = sum(drift.feature_columns(handover_contract), [])
        handover_rule = drift.make_stage_pipeline("rule", {}, handover_contract)
        handover_rule.fit(train[handover_columns], train[drift.TARGET])
    except MonitorUnavailable:
        raise
    except FileNotFoundError as error:
        raise MonitorUnavailable(f"The monitoring data is missing: {error}") from error
    except Exception as error:  # noqa: BLE001 - surface a clear 503 instead of a traceback
        raise MonitorUnavailable(f"The monitoring data could not be prepared: {error}") from error
    state = {
        "root": root,
        "cohort": cohort,
        "spec": spec,
        "coverage": coverage,
        "selected": selected,
        "models": models,
        "contracts": contracts,
        "train": train,
        "validation": validation,
        "handover_rule": handover_rule,
        "handover_columns": handover_columns,
        "input_availability": drift.input_availability(spec),
        "importance": {},
    }
    _STATE_CACHE[key] = state
    return state


# --------------------------------------------------------------------------------- Evidently calls

def _as_category(series):
    return series.astype("string").fillna("(missing)").astype(str)


def _mask_unavailable_inputs(current, availability, cutoff):
    """Blank handover-time inputs for orders the carrier did not have by the cutoff.

    The drift statistic drops missing values, so this is the same as scoring the
    feature on the rows that were actually known at the as-of date.
    """
    handover = [feature for feature, when in availability.items()
                if when == "handover" and feature in current.columns]
    if not handover or current.empty:
        return current
    unknown = ~current["handover_time"].le(cutoff).to_numpy()
    if not unknown.any():
        return current
    out = current.copy()
    out.loc[unknown, handover] = np.nan
    return out


def _prepare_frame(frame, numeric, categorical):
    out = frame[numeric + categorical].copy()
    for feature in categorical:
        out[feature] = _as_category(out[feature])
    return out


def _usable_columns(reference, current, numeric, categorical):
    usable_numeric, usable_categorical = [], []
    for feature in numeric:
        if reference[feature].notna().any() and current[feature].notna().any():
            usable_numeric.append(feature)
    for feature in categorical:
        if reference[feature].notna().any() and current[feature].notna().any():
            usable_categorical.append(feature)
    return usable_numeric, usable_categorical


def evidently_value_drift(reference, current, numeric, categorical, thresholds=None,
                          prepared_reference=None):
    """Run Evidently ValueDrift per input; return feature -> score and drift verdict."""
    _register_stat_tests()
    thresholds = {} if thresholds is None else thresholds
    numeric = list(numeric)
    categorical = list(categorical)
    if prepared_reference is None:
        prepared_reference = _prepare_frame(reference, numeric, categorical)
    prepared_current = _prepare_frame(current, numeric, categorical)
    usable_numeric, usable_categorical = _usable_columns(
        prepared_reference, prepared_current, numeric, categorical)
    results = {feature: {"score": None, "drifted": False} for feature in numeric + categorical}
    metric_thresholds = {}
    metrics = []
    for feature in usable_numeric:
        threshold = float(thresholds.get(feature, WATCH_KS))
        metric_thresholds[feature] = threshold
        metrics.append(ValueDrift(column=feature, method=KS_DISTANCE_NAME, threshold=threshold))
    for feature in usable_categorical:
        threshold = float(thresholds.get(feature, WATCH_KS))
        metric_thresholds[feature] = threshold
        metrics.append(ValueDrift(column=feature, method=TV_DISTANCE_NAME, threshold=threshold))
    if not metrics:
        return results
    definition = DataDefinition(
        numerical_columns=usable_numeric,
        categorical_columns=usable_categorical,
    )
    report = Report(metrics=metrics)
    # Calculate each metric without asking Evidently to draw its Plotly widget:
    # the dashboard shows the score itself, so rendering would be discarded work.
    context = Context(report)
    context.init_dataset(
        current_data=Dataset.from_pandas(prepared_current, data_definition=definition),
        reference_data=Dataset.from_pandas(prepared_reference, data_definition=definition),
    )
    for metric in metrics:
        calculation = metric.to_calculation()
        # Evidently's calculation draws its Plotly widgets eagerly. The dashboard
        # shows only the score, so skip that render on this calculation instance.
        calculation._render = lambda *args, **kwargs: None
        result, _ = calculation.call(context)
        feature = metric.column
        if feature in results:
            score = float(result.value)
            results[feature] = {
                "score": score,
                "drifted": score >= metric_thresholds[feature],
            }
    return results


def evidently_drift_scores(reference, current, numeric, categorical, thresholds=None):
    """Run Evidently ValueDrift for each input and return feature -> distance."""
    verdicts = evidently_value_drift(reference, current, numeric, categorical, thresholds)
    return {feature: value["score"] for feature, value in verdicts.items()}


def evidently_regression_metrics(target, predicted):
    """MAE and mean signed error (prediction - target) from Evidently."""
    frame = pd.DataFrame({"target": np.asarray(target, dtype=float),
                          "prediction": np.asarray(predicted, dtype=float)})
    definition = DataDefinition(
        regression=[Regression(target="target", prediction="prediction")])
    report = Report(metrics=[MAE(), MeanError()])
    snapshot = report.run(current_data=Dataset.from_pandas(frame, data_definition=definition))
    values = {}
    for result in snapshot.metric_results.values():
        metric_type = result.metric_value_location.metric.params.get("type", "")
        if metric_type.endswith(":MAE"):
            values["mae"] = float(result.mean.value)
        elif metric_type.endswith(":MeanError"):
            values["mean_error"] = float(result.mean.value)
    return values


# --------------------------------------------------------------------------------------- model inputs

def _saved_importance(stage):
    """Permutation importance per input, from the artifact phase3.registry saved with the model.

    Importance is saved per group of inputs that move together (``members``); every member
    of a group gets the group's importance, because the model relies on them jointly.
    It is read from the champion version's run, so a copied store needs no local paths.
    A missing or malformed artifact is an error: rebuild the store rather than guess.
    """
    name = MODEL_META[stage]["name"]
    client = _mlflow_client()
    try:
        version = client.get_model_version_by_alias(name, serving.ALIAS)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(client.download_artifacts(
                version.run_id, "permutation_importance.csv", tmp))
            frame = pd.read_csv(path)
        importance = drift.expand_importance(frame)
    except (OSError, KeyError, ValueError) as error:
        raise MonitorUnavailable(
            f"The saved input importance for {name} could not be read ({error}). "
            "Rebuild the store with python -m phase3.registry.") from error
    except Exception as error:  # noqa: BLE001 - a missing alias or artifact must name the rebuild command
        raise MonitorUnavailable(
            f"No saved input importance for {name} in the champion version ({error}). "
            "Rebuild the store with python -m phase3.registry.") from error
    return importance


def _model_importance(state, stage):
    """Permutation importance per input for one model, read once from its saved artifact."""
    if stage not in state["importance"]:
        state["importance"][stage] = _saved_importance(stage)
    return state["importance"][stage]


DEFAULT_MLFLOW_DB = MLFLOW_STORE / "mlflow.db"


def _mlflow_database_path():
    path = os.environ.get("PRAZO_MLFLOW_DB")
    if not path:
        return DEFAULT_MLFLOW_DB
    if path.startswith("sqlite:///"):
        return Path(path[len("sqlite:///"):])
    candidate = Path(path).expanduser()
    if candidate.is_dir():
        return candidate / "mlflow.db"
    return candidate


def _mlflow_client():
    uri = _tracking_uri()
    try:
        from mlflow.tracking import MlflowClient
    except ImportError as error:
        raise MonitorUnavailable(
            f"MLflow is not available to this Python interpreter ({error}). Build the store "
            "with: python -m phase3.registry") from error
    return MlflowClient(tracking_uri=uri, registry_uri=uri)


def _thresholds_error(model_name, detail):
    return MonitorUnavailable(
        f"The input-drift thresholds for {model_name} are {detail}. Rebuild the model store "
        "from the repository root with: python -m phase3.registry")


def _load_model_drift_thresholds(client, model_name):
    """Read one champion version's drift_thresholds.json from its MLflow run."""
    try:
        version = client.get_model_version_by_alias(model_name, "champion")
        run = client.get_run(version.run_id)
    except Exception as error:  # noqa: BLE001 - a missing alias or run must name the model
        raise _thresholds_error(
            model_name, f"missing from the champion version ({error})") from error
    with tempfile.TemporaryDirectory() as tmp:
        try:
            local = client.download_artifacts(version.run_id, DRIFT_THRESHOLDS_ARTIFACT, tmp)
        except Exception as error:  # noqa: BLE001 - the artifact may never have been logged
            raise _thresholds_error(
                model_name,
                f"missing from champion version {version.version} (run {run.info.run_id})") from error
        try:
            payload = json.loads(Path(local).read_text())
        except (OSError, ValueError) as error:
            raise _thresholds_error(
                model_name,
                f"not valid JSON in champion version {version.version}") from error
    if not isinstance(payload, dict):
        raise _thresholds_error(model_name, "not a JSON object")
    if payload.get("model") != model_name:
        raise _thresholds_error(
            model_name, f"labelled for {payload.get('model')!r} instead")
    if payload.get("reference") != "champion training data":
        raise _thresholds_error(model_name, "missing its training-data reference")
    if payload.get("readings") not in ("monthly", "window"):
        raise _thresholds_error(model_name, "missing its reading type")
    n_readings = payload.get("n_readings")
    if not isinstance(n_readings, int) or n_readings < 0:
        raise _thresholds_error(model_name, "missing its reading count")
    inputs = payload.get("inputs")
    if not isinstance(inputs, dict) or not inputs:
        raise _thresholds_error(model_name, "missing its input thresholds")
    for feature, entry in inputs.items():
        if not isinstance(entry, dict):
            raise _thresholds_error(model_name, f"malformed for {feature}")
        threshold = entry.get("threshold")
        if threshold is not None and not isinstance(threshold, (int, float)):
            raise _thresholds_error(model_name, f"malformed threshold for {feature}")
        if entry.get("statistic") not in ("ks", "tv"):
            raise _thresholds_error(model_name, f"missing its statistic for {feature}")
        readings = entry.get("readings")
        if not isinstance(readings, int) or readings < 0:
            raise _thresholds_error(model_name, f"missing its readings for {feature}")
        if not isinstance(entry.get("validation"), list):
            raise _thresholds_error(model_name, f"missing its validation scores for {feature}")
    return payload


def _stricter(candidate, current):
    """A lower threshold wins when one input is used by more than one model."""
    left, right = candidate.get("threshold"), current.get("threshold")
    if left is None:
        return False
    if right is None:
        return True
    return float(left) < float(right)


def _champion_thresholds(client=None):
    """Merge the champion versions' logged thresholds; one stricter entry per input."""
    client = client or _mlflow_client()
    merged = {}
    reading_method = None
    n_readings = None
    for stage in ("checkout", "handover", "late_warning"):
        model_name = MODEL_META[stage]["name"]
        payload = _load_model_drift_thresholds(client, model_name)
        if reading_method is None:
            reading_method = payload["readings"]
            n_readings = payload["n_readings"]
        elif (payload["readings"], payload["n_readings"]) != (reading_method, n_readings):
            raise _thresholds_error(
                model_name, "built with different validation readings from the other models")
        for feature, entry in payload["inputs"].items():
            current = merged.get(feature)
            if current is None or _stricter(entry, current):
                merged[feature] = dict(entry)
    return merged, reading_method, n_readings


def _index_digest(index):
    """A stable fingerprint of the rows in one window, used to reuse date-independent work."""
    return hashlib.sha256(np.asarray(index).tobytes()).hexdigest()


def _input_setup(state):
    """Everything the input panel needs that does not depend on the as-of date."""
    cached = state.get("input_setup")
    if cached is not None:
        return cached
    features = _selected_inputs(state)
    loaded, reading_method, n_readings = _champion_thresholds()
    thresholds = {}
    missing = []
    for item in features:
        entry = loaded.get(item["feature"])
        if entry is None:
            missing.append(item)
            continue
        thresholds[item["feature"]] = {
            "threshold": entry["threshold"],
            "validation": entry["validation"],
        }
    if missing:
        names = []
        for item in missing:
            for meta in MODEL_META.values():
                if meta["short"] in item["used_by"] and meta["name"] not in names:
                    names.append(meta["name"])
        raise MonitorUnavailable(
            "The input-drift thresholds are missing inputs for "
            f"{', '.join(names) or 'the models'}: "
            f"{', '.join(item['feature'] for item in missing)}. Rebuild the model store from "
            "the repository root with: python -m phase3.registry")
    model_inputs = {stage: drift.model_input_lists(state["spec"], stage)
                    for stage in ("checkout", "handover", "late_warning")}
    all_numeric = sorted({feature for numeric, _ in model_inputs.values() for feature in numeric})
    all_categorical = sorted({feature for _, categorical in model_inputs.values()
                              for feature in categorical})
    cached = {
        "features": features,
        "thresholds": thresholds,
        "model_inputs": model_inputs,
        "all_numeric": all_numeric,
        "all_categorical": all_categorical,
        "prepared_reference": _prepare_frame(state["train"], all_numeric, all_categorical),
        "validation": _validation_input_check(
            state["train"], state["spec"], features, loaded, reading_method, n_readings),
    }
    state["input_setup"] = cached
    return cached


def _expected_input_drift(visible, state, windows, window_names, cutoff):
    """drift.py's per-window numbers, reused when a complete month is queried again."""
    if not window_names:
        return {}
    cache = state.setdefault("expected_windows", {})
    expected = {stage: {"numeric_features": [], "categorical_features": [], "windows": {}}
                for stage in ("checkout", "handover", "late_warning")}
    for name in window_names:
        current = visible.loc[windows[name].to_numpy()]
        key = (name, _index_digest(current.index), cutoff)
        block = cache.get(key)
        if block is None:
            block = drift.x_drift(visible, state["spec"], {name: windows[name]}, cutoff=cutoff)
            cache[key] = block
        for stage in expected:
            expected[stage]["windows"][name] = block[stage]["windows"][name]
            expected[stage]["numeric_features"] = block[stage]["numeric_features"]
            expected[stage]["categorical_features"] = block[stage]["categorical_features"]
    return expected


def _evidently_window_verdicts(visible, state, windows, window_names, setup, unusual_thresholds,
                               cutoff):
    """Evidently scores per window, reused when a complete month is queried again."""
    cache = state.setdefault("evidently_windows", {})
    verdicts_by_window = {}
    for name in window_names:
        current = visible.loc[windows[name].to_numpy()]
        current = _mask_unavailable_inputs(current, state["input_availability"], cutoff)
        key = (name, _index_digest(current.index), cutoff)
        verdicts = cache.get(key)
        if verdicts is None:
            verdicts = evidently_value_drift(
                state["train"], current, setup["all_numeric"], setup["all_categorical"],
                unusual_thresholds, prepared_reference=setup["prepared_reference"])
            cache[key] = verdicts
        verdicts_by_window[name] = verdicts
    return verdicts_by_window


def _selected_inputs(state):
    """Inputs each model relies on, combined once, from permutation importance at the cutoff."""
    contract_types = {}
    used_by = {}
    for stage in ("checkout", "handover", "late_warning"):
        numeric, categorical = drift.model_input_lists(state["spec"], stage)
        for feature in numeric:
            contract_types[feature] = "numeric"
        for feature in categorical:
            contract_types[feature] = "categorical"
        importance = _model_importance(state, stage)
        for feature, value in importance.items():
            if value >= IMPORTANCE_CUTOFF:
                used_by.setdefault(feature, [])
                if stage not in used_by[feature]:
                    used_by[feature].append(stage)
    features = []
    for feature, stages in used_by.items():
        if feature not in contract_types:
            continue
        features.append({
            "feature": feature,
            "label": _feature_label(feature),
            "type": contract_types[feature],
            "used_by": [MODEL_META[stage]["short"] for stage in stages],
        })
    features.sort(key=lambda item: (-max(
        _model_importance(state, stage).get(item["feature"], 0.0)
        for stage in ("checkout", "handover", "late_warning")), item["feature"]))
    return features


def _validation_input_check(train, spec, features, loaded, reading_method, n_readings):
    """PSI comparison over the validation readings, using the logged thresholds."""
    method, readings = drift.validation_readings(train, spec)
    reading_count = len(readings)
    if (reading_method, n_readings) != (method, reading_count):
        raise MonitorUnavailable(
            "The logged input-drift thresholds use different validation readings from the "
            "current data. Rebuild the model store with: python -m phase3.registry")
    psi = {item["feature"]: [] for item in features}
    for _, fit, positions, _ in readings:
        reference = train.iloc[fit]
        current = train.iloc[positions]
        for item in features:
            feature = item["feature"]
            if item["type"] == "numeric":
                psi[feature].append(drift.population_stability_index(
                    reference[feature], current[feature], categorical=False))
            else:
                psi[feature].append(drift.population_stability_index(
                    reference[feature], current[feature], categorical=True))
    for item in features:
        entry = loaded[item["feature"]]
        if entry["readings"] != reading_count:
            raise MonitorUnavailable(
                f"The logged input-drift thresholds for {item['feature']} use "
                f"{entry['readings']} readings, but the current data has {reading_count}. "
                "Rebuild the model store with: python -m phase3.registry")
    psi_rows = []
    psi_flagged = {cutoff: 0 for cutoff in PSI_CUTOFFS}
    for item in features:
        values = psi[item["feature"]]
        counts = {cutoff: int(sum(value is not None and value >= cutoff for value in values))
                  for cutoff in PSI_CUTOFFS}
        for cutoff in PSI_CUTOFFS:
            psi_flagged[cutoff] += counts[cutoff]
        psi_rows.append({
            "feature": item["feature"],
            "label": item["label"],
            "flagged": {f"{cutoff:g}": counts[cutoff] for cutoff in PSI_CUTOFFS},
            "readings": reading_count,
        })
    psi_readings = reading_count * len(features)
    return {
        "reading": method,
        "reading_count": reading_count,
        "psi": {
            "bins": drift.PSI_BINS,
            "epsilon": drift.PSI_EPSILON,
            "flagged": {f"{cutoff:g}": psi_flagged[cutoff] for cutoff in PSI_CUTOFFS},
            "readings": psi_readings,
            "features": psi_rows,
        },
    }


def _windows(visible, window_names):
    month = drift.month_label(visible["order_purchase_timestamp"])
    test = visible["split"].eq("test")
    return {name: (test & (month == name)) for name in window_names}


def input_monitor(visible, state, window_names, mismatches, cutoff):
    """One combined input table with history-based levels and Evidently scores."""
    setup = _input_setup(state)
    features = setup["features"]
    thresholds = setup["thresholds"]
    model_inputs = setup["model_inputs"]
    windows = _windows(visible, window_names)
    expected = _expected_input_drift(visible, state, windows, window_names, cutoff)
    unusual_thresholds = {feature: value["threshold"] for feature, value in thresholds.items()
                           if value["threshold"] is not None}
    verdicts_by_window = _evidently_window_verdicts(
        visible, state, windows, window_names, setup, unusual_thresholds, cutoff)
    scores_by_window = {}
    drifted_by_window = {}
    for name in window_names:
        verdicts = verdicts_by_window[name]
        checked = {}
        drifted = {}
        for stage, (numeric, categorical) in model_inputs.items():
            for feature in numeric:
                fallback = expected[stage]["windows"][name][feature]["ks"]
                got = verdicts.get(feature, {}).get("score")
                if fallback is None or got is None or abs(fallback - _round(got)) > 1e-9:
                    if fallback is not None:
                        mismatches.append(
                            f"{stage}.{name}.{feature}: Evidently {_round(got)} vs drift.py "
                            f"{fallback}; using drift.py")
                    checked[feature] = fallback
                    drifted[feature] = bool(
                        fallback is not None and feature in unusual_thresholds
                        and fallback > unusual_thresholds[feature])
                else:
                    checked[feature] = got
                    drifted[feature] = bool(verdicts[feature]["drifted"])
            for feature in categorical:
                fallback = expected[stage]["windows"][name][feature]["total_variation"]
                got = verdicts.get(feature, {}).get("score")
                if fallback is None or got is None or abs(fallback - _round(got)) > 1e-9:
                    if fallback is not None:
                        mismatches.append(
                            f"{stage}.{name}.{feature}: Evidently {_round(got)} vs drift.py "
                            f"{fallback}; using drift.py")
                    checked[feature] = fallback
                    drifted[feature] = bool(
                        fallback is not None and feature in unusual_thresholds
                        and fallback > unusual_thresholds[feature])
                else:
                    checked[feature] = got
                    drifted[feature] = bool(verdicts[feature]["drifted"])
        scores_by_window[name] = checked
        drifted_by_window[name] = drifted
    rows = []
    counts = {name: 0 for name in window_names}
    for item in features:
        feature = item["feature"]
        levels = thresholds[feature]
        months = {}
        for name in window_names:
            score = scores_by_window[name].get(feature)
            level = _input_level(score, levels["threshold"])
            months[name] = {"score": _round(score), "level": level,
                            "evidently_drifted": bool(drifted_by_window[name].get(feature, False))}
            if level == "unusual":
                counts[name] += 1
        rows.append({
            "feature": feature,
            "label": item["label"],
            "type": item["type"],
            "used_by": item["used_by"],
            "threshold": levels["threshold"],
            "validation": levels["validation"],
            "months": months,
        })
    summary = {}
    for name in window_names:
        moved = counts[name]
        if not moved:
            text = f"No selected input is outside its normal range in {_month_name(name)}."
        else:
            singular = moved == 1
            noun = "input" if singular else "inputs"
            verb = "is" if singular else "are"
            pronoun = "its" if singular else "their"
            text = (f"{moved} {noun} {verb} outside {pronoun} normal range in "
                    f"{_month_name(name)}. Early warning only.")
        summary[name] = {"unusual": moved, "text": text}
    return {
        "note": "Early warning only: input drift cannot justify retraining on its own.",
        "source": "Scores computed with Evidently.",
        "importance_cutoff": IMPORTANCE_CUTOFF,
        "importance_note": (f"Inputs with permutation importance of at least "
                            f"{IMPORTANCE_CUTOFF:g} for one model, as shown on Models. Grouped inputs "
                            f"share their group's importance; each input appears once."),
        "threshold_note": (
            f"Unusual = above the highest score in the "
            f"{setup['validation']['reading_count']} validation "
            + ("months" if setup["validation"]["reading"] == "monthly" else "windows")
            + ", when the models were known to be good."),
        "recalibration_note": ("Thresholds are learned when each model is trained and stored "
                               "with that model version in MLflow."),
        "validation": setup["validation"],
        "windows": window_names,
        "features": rows,
        "summary": summary,
    }


def outcome_monitor(visible, cutoff, window_names, reference_cutoff=None):
    """Monthly outcomes with the reference-period range and each metric's maturity."""
    by_month = {row["month"]: row for row in month_outcomes(visible, cutoff)}
    ranges, reference_months = reference_ranges(visible, cutoff, reference_cutoff)
    months = []
    for name in window_names:
        row = by_month.get(name)
        if row is None:
            continue
        levels = {}
        for metric in OUTCOME_META:
            rule = OUTCOME_MATURITY[metric]
            ready = rule is None or row[rule]
            levels[metric] = outcome_level(row[metric], ranges.get(metric)) if ready else "none"
        months.append({**row, "levels": levels})
    return {"reference_ranges": ranges, "reference_months": reference_months,
            "reference_min_orders": MIN_MONTH_ORDERS, "months": months}


# ------------------------------------------------------------------------------------- model checks

def _paired_gap(target, model_pred, rule_pred, weeks, metric):
    """The Models tab's paired week bootstrap, reused from phase3.registry."""
    return registry.paired_week_gap(target, model_pred, rule_pred, weeks, metric)


def _gap_digest(target, model_pred, rule_pred, weeks):
    digest = hashlib.sha256()
    for values in (target, model_pred, rule_pred, np.asarray(weeks)):
        array = np.asarray(values)
        digest.update(str(array.dtype).encode())
        digest.update(array.tobytes())
    return digest.hexdigest()


def _cached_paired_gap(state, target, model_pred, rule_pred, weeks, metric):
    """Reuse a bootstrap when a new as-of date leaves the pooled rows unchanged."""
    cache = state.setdefault("paired_gaps", {})
    key = (metric, _gap_digest(target, model_pred, rule_pred, weeks))
    if key not in cache:
        cache[key] = _paired_gap(target, model_pred, rule_pred, weeks, metric)
    return cache[key]


def _rule_payload(gap):
    return {
        "gap": _round(gap["gap"]),
        "low": _round(gap["low"]),
        "high": _round(gap["high"]),
        "verdict": gap["verdict"],
    }


def _regression_reason(month_name, signed, rule_label, rule_level, gap, bias_level_, performance):
    bits = []
    if rule_level == "act":
        bits.append(f"{rule_label} clearly did better over the test months through {month_name} "
                    f"(MAE lower by {abs(gap['gap']):.2f} days)")
    if bias_level_ == "act":
        direction = "too long" if signed > 0 else "too short"
        bits.append(f"predictions ran {abs(signed):.1f} days {direction}")
    elif bias_level_ == "watch":
        direction = "long" if signed > 0 else "short"
        bits.append(f"predictions ran {abs(signed):.1f} days {direction}, outside the validation range")
    if performance == "watch":
        bits.append("the model is more than two validation SDs worse than in validation")
    return "; ".join(bits) if bits else "within the validation range"


def _regression_months(stage, later, later_month, mature_months, state, mismatches):
    contract = state["contracts"][stage]
    columns = sum(drift.feature_columns(contract), [])
    train = state["train"]
    model = state["models"][stage]
    validation = state["validation"][stage]
    bias_bounds = _bias_bounds(validation)
    meta = MODEL_META[stage]
    weeks = later["order_purchase_timestamp"].dt.to_period("W").astype(str)
    months = []
    if not later.empty and mature_months:
        predicted = model.predict(later[columns])
        if stage == "checkout":
            rule_predicted = later["route_typical_days"].fillna(
                train["route_typical_days"].median()).to_numpy(dtype=float)
        else:
            rule_predicted = state["handover_rule"].predict(later[state["handover_columns"]])
        for name in mature_months:
            mask = np.asarray(later_month == name)
            cumulative = np.asarray(later_month <= name)
            orders = int(mask.sum())
            pooled_weeks = int(weeks[cumulative].nunique())
            if orders < MIN_MONTH_ORDERS or pooled_weeks < MIN_POOLED_WEEKS:
                if pooled_weeks < MIN_POOLED_WEEKS:
                    reason = (f"Too few purchase weeks to judge ({pooled_weeks} pooled; "
                              f"at least {MIN_POOLED_WEEKS} needed).")
                else:
                    reason = (f"Too few orders to judge ({orders} orders; "
                              f"at least {MIN_MONTH_ORDERS} needed).")
                months.append({
                    "month": name,
                    "orders": orders,
                    "too_few": True,
                    "model": {"mae": None, "signed_error": None},
                    "rule": {"mae": None, "signed_error": None},
                    "rule_gap": None,
                    "levels": {"rule": "none", "bias": "none", "performance": "none", "overall": "none"},
                    "level": "none",
                    "reason": reason,
                })
                continue
            pooled = later.loc[cumulative, drift.TARGET]
            target = later.loc[mask, drift.TARGET]
            model_metrics = drift.regression_metrics(target, predicted[mask])
            rule_metrics = drift.regression_metrics(target, rule_predicted[mask])
            evidenced = evidently_regression_metrics(target, predicted[mask])
            model_mae = evidenced.get("mae")
            signed = evidenced.get("mean_error")
            if model_mae is None or abs(model_mae - model_metrics["mae"]) > 1e-9:
                mismatches.append(
                    f"{stage}.{name}.mae: Evidently {model_mae} vs drift.py {model_metrics['mae']}; "
                    f"using drift.py")
                model_mae = model_metrics["mae"]
            if signed is None or abs(signed - model_metrics["mean_signed_error"]) > 1e-9:
                mismatches.append(
                    f"{stage}.{name}.signed_error: Evidently {signed} vs drift.py "
                    f"{model_metrics['mean_signed_error']}; using drift.py")
                signed = model_metrics["mean_signed_error"]
            model_mae, signed = _round(model_mae), _round(signed)
            rule_mae, rule_signed = _round(rule_metrics["mae"]), _round(rule_metrics["mean_signed_error"])
            gap = _cached_paired_gap(state, pooled.to_numpy(dtype=float), predicted[cumulative],
                                     rule_predicted[cumulative],
                                     weeks[cumulative].to_numpy(), "mae")
            rule_payload = _rule_payload(gap)
            rule_payload["weeks"] = int(weeks[cumulative].nunique())
            rule_level = _rule_level(gap)
            bias = bias_level(signed, bias_bounds)
            performance = performance_level(model_mae, validation, higher_is_worse=True)
            overall = _level_max(rule_level, bias, performance)
            months.append({
                "month": name,
                "orders": orders,
                "model": {"mae": model_mae, "signed_error": signed},
                "rule": {"mae": rule_mae, "signed_error": rule_signed},
                "rule_gap": rule_payload,
                "levels": {"rule": rule_level, "bias": bias, "performance": performance,
                           "overall": overall},
                "level": overall,
                "reason": _regression_reason(_month_name(name), signed, meta["rule_label"],
                                             rule_level, rule_payload, bias, performance),
            })
    return {
        "name": meta["name"],
        "stage": stage,
        "title": meta["title"],
        "kind": "regression",
        "metric": "mae",
        "rule_label": meta["rule_label"],
        "validation": {"mean": validation["mean"], "sd": validation["sd"],
                       "n_windows": validation["n_windows"]},
        "bias_bounds": bias_bounds,
        "months": months,
    }


def _late_risk_refit_dates(state):
    """The monthly refit periods the registry holds, each live from its first day."""
    cached = state.get("late_risk_refit_dates")
    if cached is not None:
        return cached
    test = state["cohort"][state["cohort"][state["spec"]["primary_split"]].eq("test")]
    if test.empty:
        cached = []
    else:
        queue = serving.queue_module
        first = pd.Timestamp(state["spec"]["primary_cutoff"]).to_period("M").to_timestamp()
        starts = queue._winnie_refit_dates(first, test["handover_time"].max())
        cached = list(zip(starts, [*starts[1:], pd.Timestamp.max]))
    state["late_risk_refit_dates"] = cached
    return cached


def _late_risk_refit(state, begin):
    """One registered monthly refit plus the fitting orders its percentiles use."""
    cached = state.setdefault("late_risk_refits", {})
    if begin in cached:
        return cached[begin]
    queue = serving.queue_module
    table = state["cohort"]
    columns = sum(drift.feature_columns(state["contracts"]["late_warning"]), [])
    known = table[table["order_purchase_timestamp"].lt(begin)
                  & table["handover_time"].lt(begin)
                  & table["outcome_available_at"].lt(begin)].reset_index(drop=True)
    pipeline, version = serving.load_late_risk_for_date(begin, uri=_tracking_uri())
    cached[begin] = {
        "pipeline": pipeline,
        "version": int(version.version),
        "columns": columns,
        "known": known,
        "model_fit": pipeline.predict_proba(known[columns])[:, 1],
        "rule_fit": queue.winnie_rule_scores(known, known),
    }
    return cached[begin]


def _late_risk_scores(state, rows, cutoff):
    """The 50/50 blend the dashboard ranks by, scored by the monthly refit live then.

    A test order is scored with the refit that was live at its handover, against
    that refit's own fitting orders, exactly as the Models tab's monthly-refit
    blend. Orders not handed over by the cutoff are left unscored: the model is a
    handover-time model, so they were not available to it.
    """
    queue = serving.queue_module
    scores = pd.Series(np.nan, index=rows.index, dtype=float)
    handover = rows["handover_time"]
    for begin, end in _late_risk_refit_dates(state):
        if begin >= cutoff:
            break
        take = handover.ge(begin) & handover.lt(end) & handover.le(cutoff)
        if not take.any():
            continue
        refit = _late_risk_refit(state, begin)
        part = rows.loc[take]
        model_values = refit["pipeline"].predict_proba(part[refit["columns"]])[:, 1]
        rule_values = queue.winnie_rule_scores(refit["known"], part)
        scores.loc[take] = (
            0.5 * queue.winnie_percentile(refit["model_fit"], model_values)
            + 0.5 * queue.winnie_percentile(refit["rule_fit"], rule_values))
    return scores.to_numpy(dtype=float)


def _late_risk_validation(state):
    """The blend's validation PR-AUC and ROC-AUC ranges, on the Phase 2 chronological windows.

    ROC-AUC is computed from the same per-window blend as PR-AUC, because the
    score does not move with the late rate. The monthly Watch check uses ROC-AUC.
    """
    cached = state.get("late_risk_validation")
    if cached is not None:
        return cached
    queue = serving.queue_module
    train = state["train"]
    spec = state["spec"]
    contract = state["contracts"]["late_warning"]
    columns = sum(drift.feature_columns(contract), [])
    choice = state["selected"]["late_warning"]
    ap_values = []
    roc_values = []
    for fit, valid in drift.chronological_folds(train, spec):
        pipeline = drift.make_pipeline(
            choice["family"], choice["parameters"], contract,
            log_features=spec["linear_log_candidates"])
        with drift.threadpool_limits(limits=1):
            pipeline.fit(train.iloc[fit][columns], train.iloc[fit][drift.LABEL])
        model_fit = pipeline.predict_proba(train.iloc[fit][columns])[:, 1]
        rule_fit = queue.winnie_rule_scores(train.iloc[fit], train.iloc[fit])
        model_valid = pipeline.predict_proba(train.iloc[valid][columns])[:, 1]
        rule_valid = queue.winnie_rule_scores(train.iloc[fit], train.iloc[valid])
        blend = (0.5 * queue.winnie_percentile(model_fit, model_valid)
                 + 0.5 * queue.winnie_percentile(rule_fit, rule_valid))
        labels = train.iloc[valid][drift.LABEL]
        ap_values.append(drift.average_precision_score(labels, blend))
        roc_values.append(roc_auc_score(labels, blend))
    cached = _validation_stats(ap_values)
    cached["roc_auc"] = _validation_stats(roc_values)
    state["late_risk_validation"] = cached
    return cached


def _classification_reason(month_name, model_ap, model_roc, rule_level, gap, performance,
                           validation):
    bits = []
    if rule_level == "act":
        bits.append(f"the remaining-slack rule clearly did better over the test months "
                    f"through {month_name} (PR-AUC higher by {abs(gap['gap']):.3f})")
    if performance == "watch":
        roc = validation["roc_auc"]
        bits.append(f"blend ROC-AUC {model_roc:.3f} is more than two validation SDs "
                    f"below {roc['mean']:.3f}")
    return "; ".join(bits) if bits else "within the validation range"


def _classification_months(visible, cutoff, mature_months, state, mismatches):
    train = state["train"]
    validation = _late_risk_validation(state)
    meta = MODEL_META["late_warning"]
    months = []
    test = visible[visible["split"].eq("test")]
    if not test.empty and mature_months:
        test_month = drift.month_label(test["order_purchase_timestamp"])
        known, labels = drift.late_labels(test, cutoff)
        scores = _late_risk_scores(state, test, cutoff)
        known = known & np.isfinite(scores)
        median = train["route_transit_90d_at_handover"].median()
        slack = (test["promised_days"] - test["handover_days"]
                 - test["route_transit_90d_at_handover"].fillna(median))
        rule_scores = -slack.to_numpy(dtype=float)
        weeks = test["order_purchase_timestamp"].dt.to_period("W").astype(str)
        for name in mature_months:
            mask = np.asarray(test_month == name) & known
            cumulative = np.asarray(test_month <= name) & known
            orders = int(mask.sum())
            late_orders = int(labels[mask].sum()) if orders else 0
            if orders < MIN_MONTH_ORDERS or late_orders < MIN_LATE_ORDERS:
                if late_orders < MIN_LATE_ORDERS:
                    reason = (f"Not enough late orders to judge ({late_orders} late of "
                              f"{orders} orders; at least {MIN_LATE_ORDERS} late needed).")
                else:
                    reason = (f"Too few known orders to judge ({orders} orders; "
                              f"at least {MIN_MONTH_ORDERS} needed).")
                months.append({
                    "month": name,
                    "orders": orders,
                    "late_orders": late_orders,
                    "too_few": True,
                    "model": {"pr_auc": None, "roc_auc": None},
                    "rule": {"pr_auc": None},
                    "rule_gap": None,
                    "levels": {"rule": "none", "performance": "none", "overall": "none"},
                    "level": "none",
                    "reason": reason,
                })
                continue
            model_ap = float(drift.average_precision_score(labels[mask], scores[mask]))
            model_roc = float(roc_auc_score(labels[mask], scores[mask]))
            rule_ap = float(drift.average_precision_score(labels[mask], rule_scores[mask]))
            gap = _cached_paired_gap(state, labels[cumulative].astype(float), scores[cumulative],
                                     rule_scores[cumulative],
                                     weeks[cumulative].to_numpy(), "pr_auc")
            rule_payload = _rule_payload(gap)
            rule_payload["weeks"] = int(weeks[cumulative].nunique())
            rule_level = _rule_level(gap)
            performance = performance_level(_round(model_roc), validation["roc_auc"],
                                            higher_is_worse=False)
            overall = _level_max(rule_level, performance)
            model_ap, model_roc, rule_ap = _round(model_ap), _round(model_roc), _round(rule_ap)
            months.append({
                "month": name,
                "orders": orders,
                "late_orders": late_orders,
                "model": {"pr_auc": model_ap, "roc_auc": model_roc},
                "rule": {"pr_auc": rule_ap},
                "rule_gap": rule_payload,
                "levels": {"rule": rule_level, "performance": performance, "overall": overall},
                "level": overall,
                "reason": _classification_reason(_month_name(name), model_ap, model_roc,
                                                 rule_level, rule_payload, performance,
                                                 validation),
            })
    return {
        "name": meta["name"],
        "stage": "late_warning",
        "title": meta["title"],
        "kind": "classification",
        "metric": "pr_auc",
        "rule_label": meta["rule_label"],
        "ranking": "50/50 blend of the model and remaining slack",
        "validation": {"mean": validation["mean"], "sd": validation["sd"],
                       "n_windows": validation["n_windows"],
                       "roc_auc": validation["roc_auc"]},
        "months": months,
    }


def _input_context(inputs, stage):
    """How many inputs this model relies on moved this month; never a status input."""
    if not inputs["windows"]:
        return ""
    latest = inputs["windows"][-1]
    short = MODEL_META[stage]["short"]
    count = sum(1 for item in inputs["features"]
                if short in item["used_by"]
                and item["months"].get(latest, {}).get("level") == "unusual")
    if not count:
        return ""
    noun = "input" if count == 1 else "inputs"
    return f"{count} {noun} shifted"


def _month_reason(model, row, month_name):
    """One plain sentence on what one judged month shows for this model."""
    if model["kind"] == "regression":
        signed = row["model"]["signed_error"]
        if row["levels"]["bias"] in ("watch", "act") and signed is not None:
            direction = "too long" if signed > 0 else "too short"
            return f"Predicting about {abs(signed):.0f} days {direction}."
        if row["levels"]["rule"] == "act":
            return f"{model['rule_label']} ahead of the model."
        if row["levels"]["performance"] == "watch":
            return "Error is clearly worse than in validation."
        return f"Holding up in {month_name}."
    if row["levels"]["performance"] == "watch":
        return "Late-risk blend outside its validation range."
    if row["levels"]["rule"] == "act":
        return f"{model['rule_label']} ahead of the blend."
    return f"Late-risk blend within its validation range in {month_name}."


def _status_card(model, inputs):
    """One plain status, verdict from the latest judged month and an action."""
    title = model["title"]
    months = [row for row in model["months"] if not row.get("too_few")]
    if not months:
        return {
            "name": model["name"],
            "title": title,
            "level": "none",
            "month": None,
            "reason": "Not enough mature data to check this model yet.",
            "action": "Wait for more deliveries",
            "input_note": _input_context(inputs, model["stage"]),
        }
    level = _level_max(*(row["levels"]["overall"] for row in months))
    latest = months[-1]
    month_name = _month_name(latest["month"])
    action = "No action"
    if level == "act":
        action = "Retrain on recent orders"
    elif level == "watch":
        action = "Watch next month"
    reason = _month_reason(model, latest, month_name)
    if _LEVEL_ORDER.get(latest["levels"]["overall"], 0) < _LEVEL_ORDER.get(level, 0):
        # The status was set by an earlier month; say so, then how the latest month looks.
        trigger = [row for row in months if row["levels"]["overall"] == level][-1]
        trigger_name = _month_name(trigger["month"])
        latest_part = (f"{month_name} is normal again, but the status stays {level.capitalize()}."
                       if latest["levels"]["overall"] == "none"
                       else f"{reason[0].lower()}{reason[1:]}")
        reason = f"{_month_reason(model, trigger, trigger_name).rstrip('.')} in {trigger_name}; {latest_part}"
    return {
        "name": model["name"],
        "title": title,
        "level": level,
        "month": latest["month"],
        "reason": reason,
        "action": action,
        "input_note": _input_context(inputs, model["stage"]),
    }


def model_monitor(visible, cutoff, outcomes, state, mismatches, inputs):
    """Monthly model checks on mature months, with status cards and short alerts."""
    test = visible[visible["split"].eq("test")]
    known = test["outcome_available_at"].le(cutoff) & test["delivery_days"].notna()
    later = test[known]
    later_month = drift.month_label(later["order_purchase_timestamp"])
    delivery_months = [row["month"] for row in outcomes["months"] if row["delivery_mature"]]
    late_months = [row["month"] for row in outcomes["months"] if row["late_mature"]]
    models = [
        _regression_months("checkout", later, later_month, delivery_months, state, mismatches),
        _regression_months("handover", later, later_month, delivery_months, state, mismatches),
        _classification_months(visible, cutoff, late_months, state, mismatches),
    ]
    cards = [_status_card(model, inputs) for model in models]
    test_start = state["spec"].get("primary_cutoff")
    start_month = _month_name(test_start[:7]) if test_start else None
    return {
        "cohort_note": "Model checks use the test split only, on months whose outcomes are mature.",
        "test_start": test_start,
        "test_start_note": (f"The test period starts on {test_start}, so {start_month} is partial."
                            if test_start else None),
        "min_note": (f"A month needs {MIN_MONTH_ORDERS} known orders ({MIN_LATE_ORDERS} late for "
                     f"late risk) and {MIN_POOLED_WEEKS} pooled purchase weeks for delivery time; "
                     f"otherwise it is too few to judge."),
        "rule_note": ("Model advantage is how much better the model did than the rule over the test "
                      "months to date, with a 95% interval from paired weeks; positive means the model "
                      "did better. For late risk the model is the 50/50 blend with the rule. Months are "
                      "pooled because one month has few purchase weeks; the MAE and PR-AUC columns are "
                      "that month only. The late-risk Watch check uses ROC-AUC because PR-AUC moves "
                      "with the late rate."),
        "bias_note": ("Bias is prediction minus actual: positive means too long. Watch = outside the "
                      "five validation windows; Act = one validation SD beyond."),
        "models": models,
        "status_cards": cards,
    }


# ----------------------------------------------------------------------------------------- alerts

def build_alerts(model_checks):
    """At most one line per model per month, newest first."""
    alerts = []
    for model in model_checks["models"]:
        for month in model["months"]:
            level = month["level"]
            if level not in ("watch", "act") or month.get("too_few"):
                continue
            name = _month_name(month["month"])
            if model["kind"] == "regression":
                parts = []
                if month["levels"]["rule"] == "act":
                    parts.append(f"{model['rule_label']} better over test months "
                                 f"(MAE lower by {abs(month['rule_gap']['gap']):.2f} days)")
                if month["levels"]["bias"] in ("watch", "act"):
                    signed = month["model"]["signed_error"]
                    direction = "too long" if signed > 0 else "too short"
                    parts.append(f"predictions {abs(signed):.1f} days {direction}")
                if month["levels"]["performance"] == "watch":
                    parts.append("error more than two validation SDs worse")
                sentence = f"{model['title']}, {name}: " + "; ".join(parts) + "."
            else:
                if month["levels"]["rule"] == "act":
                    sentence = (f"{model['title']}, {name}: {model['rule_label']} better over test "
                                f"months (PR-AUC higher by "
                                f"{abs(month['rule_gap']['gap']):.3f}).")
                else:
                    sentence = (f"{model['title']}, {name}: {month['reason']}.")
            alerts.append({
                "level": level,
                "kind": "model",
                "month": month["month"],
                "model": model["name"],
                "sentence": sentence,
            })
    alerts.sort(key=lambda item: (item["month"], _LEVEL_ORDER.get(item["level"], 0)), reverse=True)
    return alerts


# ------------------------------------------------------------------------------------- public API

def build_monitor(date, root=REPO_ROOT):
    """Build (or return the cached) monitoring payload for one as-of date."""
    date_str = validate_date(date)
    root = Path(root).resolve()
    key = (str(root), date_str)
    cached = _MONITOR_CACHE.get(key)
    if cached is not None:
        payload = dict(cached)
        payload["cached"] = True
        return payload
    started = time.perf_counter()
    state = load_state(root)
    cutoff = cutoff_for(date_str)
    cohort = state["cohort"]
    visible = cohort[cohort["order_purchase_timestamp"] < cutoff]
    test_visible = visible[visible["split"].eq("test")]
    test_month = drift.month_label(test_visible["order_purchase_timestamp"])
    window_names = sorted(str(value) for value in pd.unique(test_month))
    mismatches = []
    inputs = input_monitor(visible, state, window_names, mismatches, cutoff)
    outcomes = outcome_monitor(visible, cutoff, window_names,
                               state["spec"].get("primary_cutoff"))
    model_checks = model_monitor(visible, cutoff, outcomes, state, mismatches, inputs)
    alerts = build_alerts(model_checks)
    latest = window_names[-1] if window_names else None
    payload = {
        "ok": True,
        "as_of": date_str,
        "cutoff": cutoff.isoformat(),
        "note": NOTE,
        "latest_window": latest,
        "report_url": (f"/api/models/monitor/report?date={date_str}" if latest else None),
        "first_load_ms": None,
        "cached": False,
        "reference": {
            "period": "Phase 2 training split",
            "orders": int(visible["split"].eq("train").sum()),
            "cutoff": state["spec"].get("primary_cutoff"),
        },
        "alerts": alerts,
        "status_cards": model_checks["status_cards"],
        "inputs": inputs,
        "outcomes": outcomes,
        "model_checks": model_checks,
        "mismatches": mismatches,
    }
    payload["first_load_ms"] = int(round((time.perf_counter() - started) * 1000))
    _MONITOR_CACHE[key] = payload
    return payload


def _report_frames(visible, window_names):
    if not window_names:
        raise MonitorUnavailable("No later purchase month is known by this date.")
    latest = window_names[-1]
    windows = _windows(visible, window_names)
    reference = visible.loc[visible["split"].eq("train").to_numpy()]
    current = visible.loc[windows[latest].to_numpy()]
    return latest, reference, current


def build_report_html(date, root=REPO_ROOT):
    """Generate the Evidently HTML input-drift report for the latest month."""
    date_str = validate_date(date)
    root = Path(root).resolve()
    key = (str(root), date_str)
    cached = _REPORT_CACHE.get(key)
    if cached is not None:
        return cached
    payload = build_monitor(date_str, root)
    features = payload["inputs"]["features"]
    if not features:
        raise MonitorUnavailable("No input has enough importance to show for this date.")
    state = load_state(root)
    cutoff = cutoff_for(date_str)
    visible = state["cohort"]
    visible = visible[visible["order_purchase_timestamp"] < cutoff]
    test_visible = visible[visible["split"].eq("test")]
    test_month = drift.month_label(test_visible["order_purchase_timestamp"])
    window_names = sorted(str(value) for value in pd.unique(test_month))
    latest, reference, current = _report_frames(visible, window_names)
    current = _mask_unavailable_inputs(current, state["input_availability"], cutoff)
    numeric = [item["feature"] for item in features if item["type"] == "numeric"]
    categorical = [item["feature"] for item in features if item["type"] == "categorical"]
    thresholds = {item["feature"]: item["threshold"] for item in features
                  if item["threshold"] is not None}
    verdicts = evidently_value_drift(reference, current, numeric, categorical, thresholds)
    prepared_reference = _prepare_frame(reference, numeric, categorical)
    prepared_current = _prepare_frame(current, numeric, categorical)
    usable_numeric, usable_categorical = _usable_columns(
        prepared_reference, prepared_current, numeric, categorical)
    metrics = (
        [ValueDrift(column=feature, method=KS_DISTANCE_NAME,
                    threshold=float(thresholds.get(feature, WATCH_KS)))
         for feature in usable_numeric]
        + [ValueDrift(column=feature, method=TV_DISTANCE_NAME,
                      threshold=float(thresholds.get(feature, WATCH_KS)))
           for feature in usable_categorical]
    )
    if not metrics:
        raise MonitorUnavailable("No input columns have data for the latest month.")
    definition = DataDefinition(
        numerical_columns=usable_numeric,
        categorical_columns=usable_categorical,
    )
    report = Report(metrics=metrics)
    snapshot = report.run(
        current_data=Dataset.from_pandas(prepared_current, data_definition=definition),
        reference_data=Dataset.from_pandas(prepared_reference, data_definition=definition),
    )
    validation = payload["inputs"]["validation"]
    reading_count = validation["reading_count"]
    reading_noun = "months" if validation["reading"] == "monthly" else "windows"
    html = snapshot.get_html_str(as_iframe=False)
    html = html.replace(
        "<body>",
        f"<body><p style=\"font-family:system-ui,sans-serif;padding:12px 20px;"
        f"background:#EAF6F6;color:#0A1F33\">Input drift for the latest purchase month "
        f"known by {date_str} 09:00: {latest}. Reference: the Phase 2 training split. "
        f"Each input is Unusual above its own highest score in the {reading_count} validation "
        f"{reading_noun}. A handover-time input uses only orders handed over by the cutoff. "
        f"KS distance for numeric inputs; total variation for categorical inputs. "
        f"Scores computed with Evidently. Dashboard verdicts agree: "
        f"{sum(value['drifted'] for value in verdicts.values())} of {len(verdicts)} inputs detected.</p>",
        1,
    )
    _REPORT_CACHE[key] = html
    return html


def clear_cache():
    _STATE_CACHE.clear()
    _MONITOR_CACHE.clear()
    _REPORT_CACHE.clear()


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--as-of", required=True, help="monitoring date, YYYY-MM-DD")
    parser.add_argument("--json", action="store_true", help="print the payload as JSON")
    args = parser.parse_args()
    payload = build_monitor(args.as_of)
    if args.json:
        print(json.dumps(payload, indent=1, default=str))
    else:
        print(f"as_of {payload['as_of']} (first load {payload['first_load_ms']} ms)")
        print(f"windows: {', '.join(payload['inputs']['windows']) or 'none'}")
        for card in payload["status_cards"]:
            print(f"[{card['level'].upper()}] {card['title']}: {card['reason']} {card['action']}")
        for alert in payload["alerts"]:
            print(f"[{alert['level'].upper()}] {alert['sentence']}")


if __name__ == "__main__":
    main()
