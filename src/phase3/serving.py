"""Serve the three registered champions and check monthly probability recalibration.

The champions are loaded once from the local MLflow store by alias
(``models:/<name>@champion``) and scored through the same Phase 2 pipelines the
queue refits, so serving and reproduction share one feature contract.  Late-risk
probabilities are recalibrated per scoring month with a Platt-style logistic fit
(intercept and slope on the raw-risk logit) on handovers purchased in month M-2
whose outcome was already recorded when month M started.  That is the same lag
as ``phase3.policy.calibration_for``; unlike the policy pool it keeps every order
whose outcome is known, not only the Phase 2 training split.

The calibration check scores validation months (Nov 2017-Mar 2018) with the
Phase 2 chronological validation fits and test months (May-Aug 2018) with the
registered champion.  It writes ``calibration_check.csv`` under data/outputs/phase3.
and sets ``calibration_ok`` only when every test month's recalibrated mean
predicted late rate is within two percentage points of the realised rate.

Run from the repository root after building the registry:
    python -m phase3.serving
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path

os.environ["MLFLOW_DISABLE_AGENT_HINT"] = "1"

import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from mlflow.tracking import MlflowClient
from sklearn.linear_model import LogisticRegression

from olist.classification import feature_columns
from olist.eta import chronological_folds, stage_spec
from olist.handover_classifier import variant_spec
from olist.paths import MLFLOW_STORE, PHASE3_OUTPUTS, PROCESSED_DIR, REPO_ROOT

from phase3 import queue as queue_module
from phase3.mlflow_server import STATIC_PREFIX as MLFLOW_STATIC_PREFIX

ROOT = REPO_ROOT
DEFAULT_STORE = MLFLOW_STORE
DEFAULT_DB = DEFAULT_STORE / "mlflow.db"
RESULT_DIR = PHASE3_OUTPUTS / "serving"
CALIBRATION_FILE = RESULT_DIR / "calibration_check.csv"
MODELS = ("checkout_eta", "handover_eta", "late_risk")
ALIAS = "champion"
LATE_RISK_REFIT_TAG = "refit_date"
VALIDATION_MONTHS = ("2017-11", "2017-12", "2018-01", "2018-02", "2018-03")
TEST_MONTHS = ("2018-05", "2018-06", "2018-07", "2018-08")
MIN_CALIBRATION_LATE = 50
RELIABILITY_BINS = 10
CALIBRATION_TOLERANCE = 0.02
LOGIT_EPSILON = 1e-12


# ------------------------------------------------------------------------------------- tracking store

def database_path(store=None):
    """Resolve the MLflow SQLite file from an argument, PRAZO_MLFLOW_DB or the default store."""
    value = store or os.environ.get("PRAZO_MLFLOW_DB") or DEFAULT_STORE
    if isinstance(value, str) and value.startswith("sqlite:///"):
        path = Path(value[len("sqlite:///"):])
    else:
        path = Path(value).expanduser()
        if path.is_dir():
            path = path / "mlflow.db"
    return path


def http_tracking_uri(value):
    """Turn an MLflow UI base URL into the tracking URI MlflowClient expects."""
    value = value.rstrip("/")
    if not value.endswith(MLFLOW_STATIC_PREFIX):
        value += MLFLOW_STATIC_PREFIX
    return value


def tracking_uri(store=None):
    """The tracking URI to use: an explicit store, or PRAZO_MLFLOW_UI when set."""
    if isinstance(store, str) and store.startswith(("http://", "https://")):
        return http_tracking_uri(store)
    if store is None:
        ui = os.environ.get("PRAZO_MLFLOW_UI")
        if ui:
            return http_tracking_uri(ui)
    path = database_path(store)
    if not path.is_file():
        raise FileNotFoundError(
            f"The MLflow store is missing at {path}. Build it from the repository root with "
            f"python -m phase3.registry"
        )
    return "sqlite:///{}".format(path.resolve())


# ----------------------------------------------------------------------------------------- champions

@dataclass
class Champions:
    """The three loaded champion pipelines plus the versions and scoring columns they need."""

    models: dict
    versions: dict
    classifier_columns: list
    eta_columns: list

    def queue_models(self, fit_rows):
        """The mapping phase3.queue.build_context expects, with the queue's median transit rule."""
        return {
            "classifier": self.models["late_risk"],
            "eta": self.models["handover_eta"],
            "classifier_columns": list(self.classifier_columns),
            "eta_columns": list(self.eta_columns),
            "median_transit": float(fit_rows["route_transit_90d_at_handover"].median()),
        }


def model_versions(champions):
    return {name: int(version) for name, version in champions.versions.items()}


def load_champions(store=None, uri=None, spec=None):
    """Load checkout ETA, handover ETA and late risk by the champion alias and record their versions."""
    uri = uri or tracking_uri(store)
    mlflow.set_tracking_uri(uri)
    mlflow.set_registry_uri(uri)
    client = MlflowClient(tracking_uri=uri, registry_uri=uri)
    if spec is None:
        spec = json.loads((PROCESSED_DIR / "phase2_feature_spec.json").read_text())

    models, versions = {}, {}
    for name in MODELS:
        try:
            version = client.get_model_version_by_alias(name, ALIAS)
        except Exception as error:  # mlflow raises several concrete types for a missing alias
            raise RuntimeError(f"Champion alias {ALIAS!r} is missing for {name}: {error}") from error
        models[name] = mlflow.sklearn.load_model(f"models:/{name}@{ALIAS}")
        versions[name] = int(version.version)

    classifier_columns = sum(feature_columns(variant_spec(spec)), [])
    eta_columns = sum(feature_columns(stage_spec(spec, "handover")), [])
    return Champions(models=models, versions=versions, classifier_columns=classifier_columns,
                     eta_columns=eta_columns)


def late_risk_version_for_date(date, store=None, client=None):
    """The registered late-risk version whose refit date is the 1st of ``date``'s month.

    The registry stores one version per month it serves.  A date outside those
    months, or a month with no (or several) matching versions, is a loud error:
    silently serving another month's model would misstate what the dashboard does.
    """
    month_start = pd.Timestamp(date).normalize().to_period("M").to_timestamp()
    if client is None:
        uri = tracking_uri(store)
        client = MlflowClient(tracking_uri=uri, registry_uri=uri)
    versions = client.search_model_versions("name = 'late_risk'")
    matching = [version for version in versions
                if version.tags.get(LATE_RISK_REFIT_TAG)
                and pd.Timestamp(version.tags[LATE_RISK_REFIT_TAG]).normalize() == month_start]
    if len(matching) != 1:
        raise RuntimeError(
            "Expected one late-risk model refitted on {} but found {}. Rebuild the "
            "registry with python -m phase3.registry.".format(
                month_start.date(), len(matching)))
    return matching[0]


def load_late_risk_for_date(date, store=None, uri=None, client=None):
    """Load the monthly late-risk pipeline that serves ``date`` and its registry version."""
    uri = uri or tracking_uri(store)
    mlflow.set_tracking_uri(uri)
    mlflow.set_registry_uri(uri)
    if client is None:
        client = MlflowClient(tracking_uri=uri, registry_uri=uri)
    version = late_risk_version_for_date(date, client=client)
    model = mlflow.sklearn.load_model("models:/late_risk/{}".format(int(version.version)))
    return model, version


def monthly_context(champions, late_risk_model, table, spec, fit_rows=None):
    """Score the cohort with the frozen delivery-time champions and a monthly late-risk model."""
    if fit_rows is None:
        fit_rows = table[table[spec["primary_split"]].eq("train")]
    models = champions.queue_models(fit_rows)
    models["classifier"] = late_risk_model
    return queue_module.build_context(fit_rows, table, table, spec, label="monthly", models=models)


def champion_context(champions, table, spec, fit_rows=None):
    """Score the whole cohort once with the champions and wrap it in a queue context."""
    if fit_rows is None:
        fit_rows = table[table[spec["primary_split"]].eq("train")]
    models = champions.queue_models(fit_rows)
    return queue_module.build_context(fit_rows, table, table, spec, label="champion", models=models)


def equivalence_check(champions, table, spec, dates, capacity=None):
    """Compare the champion path with the refit path: risk, ETA and queue order must match."""
    train = table[table[spec["primary_split"]].eq("train")]
    champion_ctx = champion_context(champions, table, spec, fit_rows=train)
    refit_ctx = queue_module.build_context(train, table, table, spec, label="refit")
    rows = []
    for date in dates:
        champion_frame = queue_module.queue(date, capacity=capacity, context=champion_ctx)
        refit_frame = queue_module.queue(date, capacity=capacity, context=refit_ctx)
        merged = champion_frame.merge(refit_frame, on="order_id", suffixes=("_champion", "_refit"))
        if len(merged) != len(champion_frame) or len(merged) != len(refit_frame):
            raise AssertionError(f"{date}: champion and refit queues contain different orders")
        risk_diff = float(np.max(np.abs(merged["risk_champion"] - merged["risk_refit"]))) if len(merged) else 0.0
        eta_diff = float(np.max(np.abs(merged["eta_days_champion"] - merged["eta_days_refit"]))) if len(merged) else 0.0
        champion_order = champion_frame[~champion_frame["past_promise"]].sort_values(
            ["rank", "order_id"], kind="stable")["order_id"].tolist()
        refit_order = refit_frame[~refit_frame["past_promise"]].sort_values(
            ["rank", "order_id"], kind="stable")["order_id"].tolist()
        rows.append({"date": str(pd.Timestamp(date).date()), "orders": int(len(champion_frame)),
                     "max_risk_diff": risk_diff, "max_eta_diff": eta_diff,
                     "same_order": champion_order == refit_order})
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------------------- recalibration

@dataclass(frozen=True)
class Recalibration:
    """A monthly Platt fit on the raw-risk logit; ``fallback`` means raw risk is used."""

    slope: float
    intercept: float
    fallback: bool
    pool_orders: int
    pool_late: int
    reason: str = ""

    def as_dict(self):
        return {"slope": self.slope, "intercept": self.intercept, "fallback": self.fallback,
                "pool_orders": self.pool_orders, "pool_late": self.pool_late, "reason": self.reason}


def _logit(raw):
    values = np.clip(np.asarray(raw, dtype=float), LOGIT_EPSILON, 1.0 - LOGIT_EPSILON)
    return np.log(values / (1.0 - values))


def apply_recalibration(raw, recalibration):
    """Apply a monthly recalibration; the fallback returns the raw risk unchanged."""
    raw = np.asarray(raw, dtype=float)
    if recalibration.fallback:
        return raw.copy()
    z = recalibration.slope * _logit(raw) + recalibration.intercept
    return np.where(z >= 0, 1.0 / (1.0 + np.exp(-z)), np.exp(z) / (1.0 + np.exp(z)))


def fit_recalibration(raw, late):
    """Fit intercept + slope on the raw-risk logit, falling back to raw risk below the late floor."""
    raw = np.asarray(raw, dtype=float)
    late = np.asarray(late, dtype=bool)
    pool_orders, pool_late = int(len(raw)), int(late.sum())
    if pool_orders == 0:
        raise ValueError("The calibration pool is empty")
    if pool_late < MIN_CALIBRATION_LATE:
        return Recalibration(1.0, 0.0, True, pool_orders, pool_late,
                             f"fewer than {MIN_CALIBRATION_LATE} late orders in the pool")
    if len(np.unique(late)) < 2:
        return Recalibration(1.0, 0.0, True, pool_orders, pool_late,
                             "the calibration pool has only one outcome class")
    model = LogisticRegression(C=np.inf, solver="lbfgs", max_iter=1000)
    model.fit(_logit(raw).reshape(-1, 1), late)
    slope = float(np.asarray(model.coef_).reshape(-1)[0])
    intercept = float(np.asarray(model.intercept_).reshape(-1)[0])
    if not np.isfinite(slope) or slope <= 0:
        raise RuntimeError(
            "The monthly recalibration slope is not positive; recalibration would change the ranking "
            f"(slope={slope!r}, pool_orders={pool_orders}, pool_late={pool_late})"
        )
    return Recalibration(slope, intercept, False, pool_orders, pool_late)


def month_periods(table):
    return pd.PeriodIndex(table["purchase_month"], freq="M")


def calibration_pool_index(table, months, month):
    """Rows purchased in month M-2 whose outcome was recorded before month M started."""
    month = pd.Period(month, freq="M")
    previous = month - 2
    start = month.to_timestamp()
    mask = (months == previous) & table["outcome_available_at"].lt(start).to_numpy()
    return np.flatnonzero(mask)


def outcomes(table, selector):
    """Late flags for the selected delivered rows.

    The queue table also holds open parcels with no outcome; selecting first keeps
    them out, and an open parcel inside the selection fails loudly here.
    """
    return table["late"].to_numpy()[selector].astype(bool)


def month_calibration(table, months, raw_all, month):
    """Fit month M's recalibration on the M-2 pool scored by the same model as month M."""
    index = calibration_pool_index(table, months, month)
    if len(index) == 0:
        raise ValueError(f"No calibration orders purchased in {pd.Period(month, freq='M') - 2}")
    late = outcomes(table, index)
    return fit_recalibration(np.asarray(raw_all)[index], late)


def _reliability(predicted, late, bins=RELIABILITY_BINS):
    predicted = np.clip(np.asarray(predicted, dtype=float), 0.0, 1.0)
    late = np.asarray(late, dtype=float)
    edges = np.linspace(0.0, 1.0, bins + 1)
    index = np.clip(np.digitize(predicted, edges[1:-1], right=False), 0, bins - 1)
    rows = []
    for number in range(bins):
        part = index == number
        rows.append({
            "bin": number,
            "bin_lower": float(edges[number]),
            "bin_upper": float(edges[number + 1]),
            "bin_orders": int(part.sum()),
            "bin_mean_predicted": float(predicted[part].mean()) if part.any() else float("nan"),
            "bin_actual_late_rate": float(late[part].mean()) if part.any() else float("nan"),
        })
    return rows


def _ranking_preserved(raw, recalibrated):
    """True when recalibration is an order-preserving (monotone) map on these scores.

    Probabilities very close to one can saturate to the same floating-point
    value after the sigmoid; that creates ties but must never create an
    inversion.  Sorting by the raw score and requiring the recalibrated score
    to be non-decreasing checks exactly that.
    """
    raw = np.asarray(raw, dtype=float)
    recalibrated = np.asarray(recalibrated, dtype=float)
    if not np.isfinite(recalibrated).all():
        return False
    order = np.argsort(raw, kind="stable")
    return bool(np.all(np.diff(recalibrated[order]) >= -1e-12))


def _month_calibration_rows(scope, month, raw, late, recalibration):
    raw = np.asarray(raw, dtype=float)
    late = np.asarray(late, dtype=bool)
    recalibrated = apply_recalibration(raw, recalibration)
    if not _ranking_preserved(raw, recalibrated):
        raise AssertionError(f"{scope} {month}: recalibration changed the ranking")
    rows = []
    for label, predicted in (("raw", raw), ("recalibrated", recalibrated)):
        brier = float(np.mean((predicted - late.astype(float)) ** 2))
        summary = {
            "scope": scope,
            "month": str(month),
            "calibration": label,
            "orders": int(len(raw)),
            "late": int(late.sum()),
            "actual_late_rate": float(late.mean()) if len(late) else float("nan"),
            "mean_predicted": float(predicted.mean()) if len(predicted) else float("nan"),
            "brier": brier,
            "fallback": bool(recalibration.fallback) if label == "recalibrated" else False,
            "pool_orders": int(recalibration.pool_orders) if label == "recalibrated" else 0,
            "pool_late": int(recalibration.pool_late) if label == "recalibrated" else 0,
            "slope": float(recalibration.slope) if label == "recalibrated" else float("nan"),
            "intercept": float(recalibration.intercept) if label == "recalibrated" else float("nan"),
            "reason": recalibration.reason if label == "recalibrated" else "",
            "rank_preserved": True,
        }
        for bin_row in _reliability(predicted, late):
            rows.append({**summary, **bin_row})
    return rows


# --------------------------------------------------------------------------------- calibration check

def _validation_models(table, spec):
    """Phase 2 chronological fits and the purchase timestamp that makes each one live."""
    train = table[table[spec["primary_split"]].eq("train")].reset_index(drop=True)
    folds = chronological_folds(train, spec)
    starts = [train.iloc[valid]["order_purchase_timestamp"].min() for _, valid in folds]
    return train, folds, starts


def _live_validation_fold(starts, month):
    start = pd.Period(month, freq="M").to_timestamp()
    live = [number for number, fold_start in enumerate(starts) if fold_start <= start]
    if not live:
        raise ValueError(f"No Phase 2 validation fit is live for {month}")
    return live[-1]


def build_calibration_check(store=None, root=ROOT, out=CALIBRATION_FILE, champions=None):
    """Build the raw/recalibrated monthly check and return its summary.

    Validation months use the Phase 2 validation fit live at the start of the
    month (the chronological fold whose validation window has started).  Test
    months use the registered champion.  Scored validation rows are the train
    split in that month; test rows are the test split.  Pools always include
    every order whose outcome was known before the month started.
    """
    table, spec, _ = queue_module.load_queue_table(root)
    table = table.reset_index(drop=True)
    months = month_periods(table)
    if champions is None:
        champions = load_champions(store=store, spec=spec)

    champion_scored = champion_context(champions, table, spec).table
    champion_raw = champion_scored["risk"].to_numpy(dtype=float)

    train, folds, starts = _validation_models(table, spec)
    fitted = {}

    def validation_raw(month):
        number = _live_validation_fold(starts, month)
        if number not in fitted:
            fit_rows = train.iloc[folds[number][0]]
            model = queue_module.fit_selected_models(fit_rows, spec)
            fitted[number] = queue_module.score_rows(model, table)["risk"].to_numpy(dtype=float)
        return fitted[number]

    rows = []
    test_summary = []
    test_split = table["split"].eq("test").to_numpy()
    train_split = table["split"].eq("train").to_numpy()
    valid_fold = table[spec["primary_fold"]].isin(range(spec["n_folds"])).to_numpy()

    for month in VALIDATION_MONTHS:
        raw_all = validation_raw(month)
        in_month = np.asarray(months == pd.Period(month, freq="M"), dtype=bool)
        mask = train_split & valid_fold & in_month
        recalibration = month_calibration(table, months, raw_all, month)
        rows.extend(_month_calibration_rows("validation", month, raw_all[mask],
                                            outcomes(table, mask), recalibration))

    for month in TEST_MONTHS:
        in_month = np.asarray(months == pd.Period(month, freq="M"), dtype=bool)
        mask = test_split & in_month
        recalibration = month_calibration(table, months, champion_raw, month)
        month_rows = _month_calibration_rows("test", month, champion_raw[mask],
                                             outcomes(table, mask), recalibration)
        rows.extend(month_rows)
        recalibrated = next(row for row in month_rows if row["calibration"] == "recalibrated" and row["bin"] == 0)
        test_summary.append(recalibrated)

    frame = pd.DataFrame(rows)
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(out, index=False)

    calibration_ok = all(
        abs(row["mean_predicted"] - row["actual_late_rate"]) <= CALIBRATION_TOLERANCE
        for row in test_summary
    )
    return {
        "calibration_ok": bool(calibration_ok),
        "tolerance": CALIBRATION_TOLERANCE,
        "test_months": [
            {"month": row["month"], "actual_late_rate": row["actual_late_rate"],
             "mean_predicted_recalibrated": row["mean_predicted"], "fallback": row["fallback"]}
            for row in test_summary
        ],
        "rows": frame,
        "model_versions": model_versions(champions),
        "path": str(out),
    }


def test_calibration_status(table, raw_all, champions, months=None):
    """The calibration_ok flag and per-test-month means used before serving probabilities."""
    if months is None:
        months = month_periods(table)
    split = table["split"].eq("test").to_numpy()
    rows = []
    for month in TEST_MONTHS:
        mask = split & np.asarray(months == pd.Period(month, freq="M"), dtype=bool)
        recalibration = month_calibration(table, months, raw_all, month)
        predicted = apply_recalibration(np.asarray(raw_all)[mask], recalibration)
        rows.append({
            "month": month,
            "orders": int(mask.sum()),
            "late": int(outcomes(table, mask).sum()),
            "actual_late_rate": float(outcomes(table, mask).mean()),
            "mean_predicted_recalibrated": float(predicted.mean()),
            "mean_predicted_raw": float(np.asarray(raw_all)[mask].mean()),
            "fallback": bool(recalibration.fallback),
        })
    ok = all(abs(row["mean_predicted_recalibrated"] - row["actual_late_rate"]) <= CALIBRATION_TOLERANCE
             for row in rows)
    return {"calibration_ok": bool(ok), "tolerance": CALIBRATION_TOLERANCE,
            "months": rows, "model_versions": model_versions(champions)}


def main():
    import argparse

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--store", default=None, help="MLflow store directory or SQLite file")
    parser.add_argument("--out", default=str(CALIBRATION_FILE), help="calibration check CSV")
    args = parser.parse_args()
    summary = build_calibration_check(store=args.store, out=args.out)
    print(json.dumps({key: value for key, value in summary.items() if key != "rows"}, indent=2))
    print(summary["rows"].query("bin == 0").to_string(index=False))
    return summary


if __name__ == "__main__":
    main()
