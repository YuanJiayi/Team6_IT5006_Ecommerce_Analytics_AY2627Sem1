"""Phase 3 drift signals: rebuild of drift_investigation.pdf.

Run from the repository root:
    python -m phase3.drift [--as-of YYYY-MM-DD]

Compares the Phase 2 training period (the reference) with each later purchase
month on three axes, using only the Phase 2 selected models and the Phase 2
point-in-time order table:
- X drift: per-input two-sample KS distance, medians, missing shares and
  categorical total-variation distance, with team-set Watch/Act flags.
- Y drift: monthly outcome means, promise and late rate, with label maturity:
  an order only counts once its delivery is recorded by `as_of`; months that
  are under 60 days old or less than 90% matured are "outcomes still arriving"
  and never flagged.
- X -> Y drift: monthly MAE and signed error for the two duration models and
  calibration/ranking drift for the late-warning classifier, plus a
  state-route-mix vs within-route decomposition of the mean delivery-time drop.

Writes data/outputs/phase3 (drift_report.json, one block per model per month)
and ledger.csv (stated PDF values vs reproduced values). The retraining-policy
replay is a separate job and is not built here.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp
from sklearn.metrics import average_precision_score
from threadpoolctl import threadpool_limits

from olist.classification import feature_columns, make_pipeline
from olist.eta import TARGET, chronological_folds, make_stage_pipeline, stage_spec
from olist.handover_classifier import (
    LABEL, TOP_SHARE, action_metrics, load_cohort, top_flags, variant_spec, weighted_pr_auc)
from olist.paths import DASHBOARD_PHASE2, PHASE3_OUTPUTS, REPO_ROOT
from olist.regression import regression_metrics

RESULTS = PHASE3_OUTPUTS / "drift"
# The dashboard's own Phase 2 result files; Phase 3 reads its model choice from there.
ETA_SELECTION = Path("eta/selection.json")
CLASSIFIER_SELECTION = Path("handover/selection.json")

# Team-set alert thresholds from the brief (not derived from data).
WATCH_KS, ACT_KS = 0.1, 0.2
# Monitoring input panel: only inputs with at least this grouped permutation
# importance are shown, and a validation reading needs this many orders to count.
INPUT_IMPORTANCE_CUTOFF = 0.02
VALIDATION_MIN_MONTH_ORDERS = 500
# Label maturity guard rails (the 60 days matches Phase 2's validation maturity window).
MATURITY_MIN_SHARE, MATURITY_MIN_AGE_DAYS = 0.9, 60
STILL_ARRIVING = "outcomes_still_arriving"
MATURE = "mature"
# Predicted late-risk reliability bins in percent, from the brief.
RELIABILITY_BINS_PCT = [0.0, 1.0, 2.5, 5.0, 10.0, 20.0, 50.0, 100.0]
BOOTSTRAP_DRAWS, BOOTSTRAP_SEED = 300, 42
# Fixed illustration window from the PDF: purchases 1-25 May labelled as of the Phase 2 cutoff.
MATURITY_EXAMPLE = {"start": "2018-05-01", "end_exclusive": "2018-05-26", "labelled_as_of": "2018-05-26"}
# Derived features kept for the PDF comparison: promise slack over the route's typical transit (an input of an
# earlier contract), and remaining slack with the PDF's deadline (the promised date itself).
PROMISE_SLACK = "promise_slack"  # promised_days - route_typical_days; not an input of the promise-free run B estimate
REMAINING_SLACK_PDF = "remaining_slack_pdf"  # deadline = promised date; the current input adds a 1-day deadline


# ----------------------------------------------------------------------------------- primitives

def ks_distance(ref, win):
    """Two-sample KS distance; None when either side has no observed values."""
    a, b = pd.Series(ref).dropna(), pd.Series(win).dropna()
    if len(a) == 0 or len(b) == 0:
        return None
    # Only the statistic is used; the asymptotic method skips the exact p-value
    # calculation and leaves the statistic bit-for-bit unchanged.
    return float(ks_2samp(a, b, method="asymp").statistic)


def total_variation(ref, win):
    """Total-variation distance between two categorical distributions (union of categories)."""
    a = pd.Series(ref).fillna("(missing)").value_counts(normalize=True)
    b = pd.Series(win).fillna("(missing)").value_counts(normalize=True)
    if a.sum() == 0 or b.sum() == 0:
        return None, {}
    union = a.to_dict() | b.to_dict()
    tv = 0.5 * sum(abs(a.get(k, 0.0) - b.get(k, 0.0)) for k in union)
    shifts = {k: round(float(b.get(k, 0.0) - a.get(k, 0.0)), 4)
              for k in sorted(union, key=lambda k: -abs(a.get(k, 0.0) - b.get(k, 0.0)))[:3]}
    return float(tv), shifts


PSI_BINS = 10
PSI_EPSILON = 0.0001


def population_stability_index(reference, current, categorical=False,
                               bins=PSI_BINS, epsilon=PSI_EPSILON):
    """Population stability index between a reference and a current sample.

    Numeric inputs use `bins` quantile bins from the reference; categorical
    inputs use the observed categories. Empty bins get `epsilon` so the log is
    finite. Returns None when either side has no observed values.
    """
    ref, cur = pd.Series(reference), pd.Series(current)
    if categorical:
        ref = ref.fillna("(missing)")
        cur = cur.fillna("(missing)")
        if ref.empty or cur.empty:
            return None
        ref_share = ref.value_counts(normalize=True)
        cur_share = cur.value_counts(normalize=True)
    else:
        ref = ref.dropna()
        cur = cur.dropna()
        if ref.empty or cur.empty:
            return None
        edges = np.unique(np.quantile(ref.to_numpy(dtype=float), np.linspace(0.0, 1.0, bins + 1)))
        if len(edges) < 2:
            edges = np.array([-np.inf, np.inf])
        else:
            edges = np.concatenate(([-np.inf], edges[1:-1], [np.inf]))
        ref_share = pd.cut(ref, edges, include_lowest=True).value_counts(normalize=True)
        cur_share = pd.cut(cur, edges, include_lowest=True).value_counts(normalize=True)
    keys = ref_share.index.union(cur_share.index)
    ref_share = ref_share.reindex(keys, fill_value=0.0)
    cur_share = cur_share.reindex(keys, fill_value=0.0)
    ref_share = ref_share.where(ref_share.gt(0.0), epsilon)
    cur_share = cur_share.where(cur_share.gt(0.0), epsilon)
    return float(np.sum((cur_share - ref_share) * np.log(cur_share / ref_share)))


def ks_flag(ks):
    """Team-set thresholds: Act at KS >= 0.2, Watch at KS >= 0.1."""
    if ks is None:
        return None
    return "act" if ks >= ACT_KS else ("watch" if ks >= WATCH_KS else "none")


def numeric_profile(ref, win):
    ks = ks_distance(ref, win)
    return {"ks": round(ks, 4) if ks is not None else None,
            "median_reference": _round_or_none(pd.Series(ref).median()),
            "median_window": _round_or_none(pd.Series(win).median()),
            "missing_reference": round(float(pd.Series(ref).isna().mean()), 4),
            "missing_window": round(float(pd.Series(win).isna().mean()), 4),
            "flag": ks_flag(ks)}


def _round_or_none(value, digits=4):
    return None if pd.isna(value) else round(float(value), digits)


def month_label(times):
    return pd.PeriodIndex(pd.Series(times), freq="M").astype(str)


def month_end(label):
    return pd.Period(label).to_timestamp(how="end")


def month_age_days(as_of, label):
    return float((as_of - month_end(label)).total_seconds() / 86400.0)


def maturity_status(share_matured, age_days):
    """Months that are too young or too incomplete have outcomes still arriving."""
    if share_matured < MATURITY_MIN_SHARE or age_days < MATURITY_MIN_AGE_DAYS:
        return STILL_ARRIVING
    return MATURE


def reliability_bins(y, predicted, edges_pct):
    """Observed vs mean predicted late risk per predicted-risk bin (percent)."""
    y, predicted = np.asarray(y, dtype=float), np.asarray(predicted, dtype=float)
    pct = predicted * 100.0
    rows = []
    for lo, hi in zip(edges_pct[:-1], edges_pct[1:]):
        in_bin = (pct >= lo) & (pct < hi if hi < edges_pct[-1] else pct <= hi)
        rows.append({"bin_pct": f"[{lo:g}, {hi:g}]",
                     "orders": int(in_bin.sum()), "late": int(y[in_bin].sum()),
                     "mean_predicted_pct": round(float(pct[in_bin].mean()), 3) if in_bin.any() else None,
                     "observed_late_pct": round(float(y[in_bin].mean()) * 100, 3) if in_bin.any() else None})
    return rows


# --------------------------------------------------------------------------- X drift (inputs)

def model_input_lists(spec, stage):
    """(numeric, categorical) input lists of a selected model, from the Phase 2 contract."""
    if stage == "late_warning":
        s = variant_spec(spec)
    else:
        s = stage_spec(spec, stage)
    return list(s["numeric_features"]), list(s["categorical_features"])


def expand_importance(frame):
    """Raw-input importance from a permutation-importance frame.

    A grouped row answers what happens when a whole idea is scrambled, so every
    member of that group gets the group's importance. Older per-input frames
    (one input per row) still work.
    """
    importance = {}
    for _, row in frame.iterrows():
        value = float(row["importance_mean"])
        members = row.get("members")
        if members is None or (isinstance(members, float) and np.isnan(members)):
            names = [str(row.get("key") or row.get("input") or "")]
        else:
            names = str(members).split(",")
        for name in names:
            name = name.strip()
            if not name:
                continue
            if name not in importance or value > importance[name]:
                importance[name] = value
    return importance


def important_inputs(importance, numeric, categorical, cutoff=INPUT_IMPORTANCE_CUTOFF):
    """Inputs at the importance cutoff, with their drift statistic."""
    types = {}
    for feature in numeric:
        types[feature] = "numeric"
    for feature in categorical:
        types[feature] = "categorical"
    return [{"feature": feature, "type": types[feature], "importance": float(value)}
            for feature, value in importance.items()
            if feature in types and float(value) >= cutoff]


def validation_readings(train, spec):
    """Validation readings: monthly when every window has 2+ months of >=500 orders, else whole windows."""
    folds = chronological_folds(train, spec)
    month = pd.Series(month_label(train["order_purchase_timestamp"]), index=train.index)
    monthly = []
    usable = True
    for fold, (fit, valid) in enumerate(folds):
        labels = sorted(str(value) for value in pd.unique(month.iloc[valid]))
        if len(labels) < 2:
            usable = False
        parts = []
        for label in labels:
            positions = valid[(month.iloc[valid] == label).to_numpy()]
            if len(positions) < VALIDATION_MIN_MONTH_ORDERS:
                usable = False
            parts.append((label, positions))
        monthly.append((fit, parts))
    if usable:
        return "monthly", [(fold, fit, positions, label)
                           for fold, (fit, parts) in enumerate(monthly)
                           for label, positions in parts]
    return "window", [(fold, fit, valid, f"window {fold + 1}")
                      for fold, (fit, valid) in enumerate(folds)]


def input_availability(spec):
    """When each model input is known: at purchase, or only once the carrier has it.

    A feature that the checkout model uses is known at purchase. The handover and
    late-warning models add features that only exist after handover.
    """
    purchase_numeric, purchase_categorical = model_input_lists(spec, "checkout")
    purchase = set(purchase_numeric) | set(purchase_categorical)
    features = set(purchase)
    for stage in ("checkout", "handover", "late_warning"):
        numeric, categorical = model_input_lists(spec, stage)
        features.update(numeric)
        features.update(categorical)
    return {feature: ("purchase" if feature in purchase else "handover")
            for feature in sorted(features)}


def input_threshold_table(train, spec, features):
    """Per-input Unusual thresholds, from the validation readings.

    `features` is a list of {"feature", "type"} entries. A threshold is the
    largest drift seen in the validation readings (the same monthly readings the
    monthly check uses).
    """
    method, readings = validation_readings(train, spec)
    scores = {item["feature"]: [] for item in features}
    for _, fit, positions, _ in readings:
        reference = train.iloc[fit]
        current = train.iloc[positions]
        for item in features:
            feature = item["feature"]
            if item["type"] == "numeric":
                scores[feature].append(ks_distance(reference[feature], current[feature]))
            else:
                scores[feature].append(total_variation(reference[feature], current[feature])[0])
    thresholds = {}
    for item in features:
        feature = item["feature"]
        values = [float(value) for value in scores[feature] if value is not None]
        thresholds[feature] = {
            "threshold": round(max(values), 4) if values else None,
            "validation": [round(float(value), 4) if value is not None else None
                           for value in scores[feature]],
            "statistic": "ks" if item["type"] == "numeric" else "tv",
            "readings": len(readings),
        }
    return {"readings": method, "n_readings": len(readings), "inputs": thresholds}


def x_drift(cohort, spec, windows, cutoff=None):
    """Per-model, per-window input drift: KS/medians/missing for numeric, TV for categorical.

    A feature shared by more than one model is scored once per window and reused: the
    reference and current rows are the same, so recomputing them would only repeat work.
    With a cutoff, a handover-time input only uses orders the carrier already had by
    then, so a window never scores data that did not exist at the as-of date.
    """
    month = month_label(cohort["order_purchase_timestamp"])
    ref_mask = cohort["split"].eq("train").to_numpy()
    availability = input_availability(spec) if cutoff is not None else {}
    handover_known = (cohort["handover_time"].le(cutoff).to_numpy()
                      if cutoff is not None else None)
    numeric_cache, categorical_cache, derived_cache = {}, {}, {}
    frames = {}
    for stage in ("checkout", "handover", "late_warning"):
        numeric, categorical = model_input_lists(spec, stage)
        profiles = {"numeric_features": numeric, "categorical_features": categorical, "windows": {}}
        for name, mask in windows.items():
            win_mask = mask.reindex(cohort.index).fillna(False).astype(bool).to_numpy()
            window = {"n": int(win_mask.sum())}
            for feature in numeric:
                key = (name, feature)
                if key not in numeric_cache:
                    known = availability.get(feature) == "handover"
                    numeric_cache[key] = numeric_profile(
                        cohort.loc[ref_mask & handover_known if known else ref_mask, feature],
                        cohort.loc[win_mask & handover_known if known else win_mask, feature])
                window[feature] = numeric_cache[key]
            for feature in categorical:
                key = (name, feature)
                if key not in categorical_cache:
                    known = availability.get(feature) == "handover"
                    tv, shifts = total_variation(
                        cohort.loc[ref_mask & handover_known if known else ref_mask, feature],
                        cohort.loc[win_mask & handover_known if known else win_mask, feature])
                    categorical_cache[key] = {"total_variation": round(tv, 4),
                                              "largest_share_shifts": shifts}
                window[feature] = categorical_cache[key]
            # Derived features from the PDF era, not inputs of the current selected models.
            if name not in derived_cache:
                slack = cohort["promised_days"] - cohort["route_typical_days"]
                promise_slack = numeric_profile(slack[ref_mask], slack[win_mask])
                promise_slack["note"] = "derived; not an input of the current selected models"
                slack_pdf = (cohort["promised_days"] - cohort["handover_days"]
                             - cohort["route_transit_90d_at_handover"])
                remaining_slack = numeric_profile(slack_pdf[ref_mask], slack_pdf[win_mask])
                remaining_slack["note"] = "derived with the PDF's deadline (promised date)"
                derived_cache[name] = (promise_slack, remaining_slack)
            window[PROMISE_SLACK], window[REMAINING_SLACK_PDF] = derived_cache[name]
            profiles["windows"][name] = window
        frames[stage] = profiles
    return frames


# --------------------------------------------------------------------------- Y drift (labels)

def late_labels(orders, as_of):
    """Known late labels under the monitoring maturity rule.

    Delivered orders carry their recorded late/on-time status. An order still
    open past its promise date is certainly late, so it is labelled late. An open
    order not yet past its promise date is unknown and is excluded by `known`.
    """
    delivered = orders["delivery_days"].notna() & orders["outcome_available_at"].le(as_of)
    promised_at = orders["order_purchase_timestamp"] + pd.to_timedelta(orders["promised_days"], unit="D")
    past = promised_at.le(as_of)
    known = (delivered | past).to_numpy()
    labels = np.where(delivered.to_numpy(), orders["is_late"].fillna(0).astype(int).to_numpy(), 1)
    return known, labels


def month_outcomes(orders, as_of):
    """Outcome summary per purchase month: promise is known at purchase, outcomes only when matured."""
    month = pd.Series(month_label(orders["order_purchase_timestamp"]), index=orders.index)
    matured = orders["outcome_available_at"].le(as_of)
    rows = []
    for label in sorted(month.unique()):
        in_month = month.eq(label)
        sub, done = orders[in_month], orders[in_month & matured]
        share = len(done) / len(sub) if len(sub) else 0.0
        age = month_age_days(as_of, label)
        rows.append({
            "month": label, "orders": int(len(sub)), "matured_orders": int(len(done)),
            "share_matured": round(share, 4), "month_age_days_at_as_of": round(age, 1),
            "maturity_status": maturity_status(share, age),
            "mean_promised_days": _round_or_none(sub["promised_days"].mean()),
            "mean_delivery_days": _round_or_none(done["delivery_days"].mean()),
            "median_delivery_days": _round_or_none(done["delivery_days"].median()),
            "late_rate_matured": round(float(done["is_late"].mean()), 4) if len(done) else None,
        })
    return rows


def maturity_example(cohort):
    """PDF illustration: label maturity bias for 1-25 May purchases as of the Phase 2 cutoff."""
    start, end = pd.Timestamp(MATURITY_EXAMPLE["start"]), pd.Timestamp(MATURITY_EXAMPLE["end_exclusive"])
    labelled_at = pd.Timestamp(MATURITY_EXAMPLE["labelled_as_of"])
    window = cohort[cohort["order_purchase_timestamp"].ge(start) & cohort["order_purchase_timestamp"].lt(end)]
    done = window[window["outcome_available_at"].lt(labelled_at)]
    return {"window": f"{MATURITY_EXAMPLE['start']} to {MATURITY_EXAMPLE['end_exclusive']}",
            "labelled_as_of": MATURITY_EXAMPLE["labelled_as_of"],
            "orders": int(len(window)), "labelled_by_as_of": int(len(done)),
            "share_labelled": round(len(done) / len(window), 4) if len(window) else None,
            "labelled_mean_delivery_days": _round_or_none(done["delivery_days"].mean()),
            "labelled_late_rate": round(float(done["is_late"].mean()), 4) if len(done) else None,
            "full_mean_delivery_days": _round_or_none(window["delivery_days"].mean()),
            "full_late_rate": round(float(window["is_late"].mean()), 4) if len(window) else None,
            "note": "the labelled subset is younger than the full set: labelling early understates delivery days"}


def y_drift(cohort, as_of):
    known = cohort["outcome_available_at"].le(as_of)
    ref = cohort[cohort["split"].eq("train") & known]
    later = cohort[cohort["split"].eq("test") & known]
    return {"monthly": month_outcomes(cohort, as_of),
            "reference_period": {"split": "train", "orders": int(known[cohort["split"].eq("train")].sum()),
                                 "mean_delivery_days": _round_or_none(ref["delivery_days"].mean()),
                                 "late_rate": round(float(ref["is_late"].mean()), 4)},
            "later_period": {"split": "test", "orders": int(known[cohort["split"].eq("test")].sum()),
                             "mean_delivery_days": _round_or_none(later["delivery_days"].mean()),
                             "late_rate": round(float(later["is_late"].mean()), 4)},
            "maturity_example": maturity_example(cohort)}


# ------------------------------------------------------------------- X -> Y drift (models)

def _selected_entry(path, keys):
    """Read one selected model from a Phase 2 selection file, failing loudly when it is absent."""
    if not path.is_file():
        raise FileNotFoundError(f"Phase 2 selection file not found: {path}")
    node = json.loads(path.read_text())
    for key in keys:
        if not isinstance(node, dict) or key not in node:
            raise KeyError(f"Phase 2 selection is missing {'.'.join(keys)} in {path}")
        node = node[key]
    missing = [key for key in ("candidate", "family", "parameters") if key not in node]
    if missing:
        raise KeyError(f"Phase 2 selection entry {'.'.join(keys)} in {path} is missing {', '.join(missing)}")
    return {"candidate": node["candidate"], "family": node["family"], "parameters": dict(node["parameters"])}


def selected_models(results_root=DASHBOARD_PHASE2):
    """The Phase 2 selected model for each stage, read from the saved selection files."""
    results_root = Path(results_root)
    return {
        "checkout": _selected_entry(results_root / ETA_SELECTION, ("selected_model", "checkout")),
        "handover": _selected_entry(results_root / ETA_SELECTION, ("selected_model", "handover")),
        "late_warning": _selected_entry(results_root / CLASSIFIER_SELECTION,
                                        ("variants", "primary", "selected_model")),
    }


def fit_selected_models(cohort, spec, selected):
    """Fit the three Phase 2 selected models once on the full training split."""
    train = cohort[cohort["split"].eq("train")].reset_index(drop=True)
    models, contracts = {}, {}
    for stage in ("checkout", "handover"):
        s = stage_spec(spec, stage)
        columns = sum(feature_columns(s), [])
        pipeline = make_stage_pipeline(selected[stage]["family"], selected[stage]["parameters"], s)
        with threadpool_limits(limits=1):
            pipeline.fit(train[columns], train[TARGET])
        models[stage], contracts[stage] = pipeline, s
    s = variant_spec(spec)
    columns = sum(feature_columns(s), [])
    choice = selected["late_warning"]
    pipeline = make_pipeline(choice["family"], choice["parameters"], s,
                             log_features=spec["linear_log_candidates"])
    with threadpool_limits(limits=1):
        pipeline.fit(train[columns], train[LABEL])
    models["late_warning"], contracts["late_warning"] = pipeline, s
    return models, contracts, train


def validation_metrics(train, spec, contracts, selected):
    """Chronological 5-window validation: mean MAE for the regressions, mean AP for the classifier."""
    folds = chronological_folds(train, spec)
    out = {}
    for stage in ("checkout", "handover"):
        s, columns = contracts[stage], sum(feature_columns(contracts[stage]), [])
        maes = []
        for fit, valid in folds:
            pipeline = make_stage_pipeline(selected[stage]["family"], selected[stage]["parameters"], s)
            with threadpool_limits(limits=1):
                pipeline.fit(train.iloc[fit][columns], train.iloc[fit][TARGET])
                predicted = pipeline.predict(train.iloc[valid][columns])
            maes.append(regression_metrics(train.iloc[valid][TARGET], predicted)["mae"])
        out[stage] = {"validation_mean_mae": round(float(np.mean(maes)), 4), "n_windows": len(folds)}
    s, columns = contracts["late_warning"], sum(feature_columns(contracts["late_warning"]), [])
    choice = selected["late_warning"]
    aps = []
    for fit, valid in folds:
        pipeline = make_pipeline(choice["family"], choice["parameters"], s,
                                 log_features=spec["linear_log_candidates"])
        with threadpool_limits(limits=1):
            pipeline.fit(train.iloc[fit][columns], train.iloc[fit][LABEL])
            scores = pipeline.predict_proba(train.iloc[valid][columns])[:, 1]
        aps.append(average_precision_score(train.iloc[valid][LABEL], scores))
    out["late_warning"] = {"validation_mean_ap": round(float(np.mean(aps)), 4), "n_windows": len(folds)}
    return out


def xy_regressions(cohort, models, contracts, train, as_of, windows):
    """Monthly MAE and signed error (predicted - actual) on orders whose delivery is known by as_of."""
    known = cohort["outcome_available_at"].le(as_of)
    later = cohort[cohort["split"].eq("test") & known]
    month = month_label(later["order_purchase_timestamp"])
    out = {}
    for stage in ("checkout", "handover"):
        columns = sum(feature_columns(contracts[stage]), [])
        with threadpool_limits(limits=1):
            in_sample = regression_metrics(train[TARGET], models[stage].predict(train[columns]))
            predicted = models[stage].predict(later[columns])
        monthly = {}
        for name, mask in windows.items():
            if name == "test_overall":
                continue
            index = mask.reindex(later.index).fillna(False).to_numpy()
            metrics = regression_metrics(later.loc[index, TARGET], predicted[index]) if index.any() else {}
            monthly[name] = {"orders": int(index.sum()), "mae": _round_or_none(metrics.get("mae")),
                             "signed_error": _round_or_none(metrics.get("mean_signed_error")),
                             "r2": _round_or_none(metrics.get("r2"))}
        overall = regression_metrics(later[TARGET], predicted)
        out[stage] = {"n_inputs": len(columns),
                      "train_in_sample_mae": round(in_sample["mae"], 4),
                      "later_overall": {"orders": int(len(later)), "mae": round(overall["mae"], 4),
                                        "signed_error": round(overall["mean_signed_error"], 4),
                                        "r2": round(overall["r2"], 4)},
                      "monthly": monthly}
    return out


def xy_classifier(cohort, model, contract, train, as_of, windows):
    """Classifier drift: predicted vs observed risk per month, reliability bins, AP and top-10% MCC."""
    columns = sum(feature_columns(contract), [])
    known = cohort["outcome_available_at"].le(as_of)
    later = cohort[cohort["split"].eq("test") & known]
    with threadpool_limits(limits=1):
        scores = model.predict_proba(later[columns])[:, 1]
    y = later[LABEL].to_numpy()
    month = month_label(later["order_purchase_timestamp"])
    monthly = {}
    monthly_aps = []
    for name, mask in windows.items():
        if name == "test_overall":
            continue
        index = mask.reindex(later.index).fillna(False).to_numpy()
        if not index.any():
            continue
        y_m, s_m = y[index], scores[index]
        mean_pred = float(np.mean(s_m))
        monthly[name] = {"orders": int(index.sum()), "late": int(y_m.sum()),
                         "mean_predicted_risk": round(mean_pred, 4),
                         "observed_late_rate": round(float(np.mean(y_m)), 4),
                         "expected_late": int(round(mean_pred * len(y_m))),
                         "observed_late": int(y_m.sum()),
                         "reliability_bins": reliability_bins(y_m, s_m, RELIABILITY_BINS_PCT),
                         "ap": round(float(average_precision_score(y_m, s_m)), 4) if y_m.sum() else None,
                         "mcc_top10": round(action_metrics(y_m, top_flags(s_m, TOP_SHARE))["mcc"], 4)}
        monthly_aps.append(average_precision_score(y_m, s_m))
    weights = np.array([1.0 / (month == m).sum() for m in month])
    return {"n_inputs": len(columns),
            "train_late_rate": round(float(train[LABEL].mean()), 4),
            "later_overall": {"orders": int(len(later)), "late": int(y.sum()),
                              "mean_predicted_risk": round(float(np.mean(scores)), 4),
                              "observed_late_rate": round(float(np.mean(y)), 4),
                              "expected_late": int(round(np.mean(scores) * len(y))),
                              "reliability_bins": reliability_bins(y, scores, RELIABILITY_BINS_PCT),
                              "ap_pooled": round(float(average_precision_score(y, scores)), 4),
                              "ap_month_weighted": round(float(weighted_pr_auc(y, scores, weights)), 4),
                              "ap_monthly_mean": round(float(np.mean(monthly_aps)), 4) if monthly_aps else None,
                              "mcc_top10": round(action_metrics(y, top_flags(scores, TOP_SHARE))["mcc"], 4)},
            "monthly": monthly}


# ------------------------------------------------------------------------- route decomposition

ROUTE = "route"


def add_route(frame):
    out = frame.copy()
    out[ROUTE] = out["seller_state"].astype(str) + "->" + out["customer_state"].astype(str)
    return out


def _mix_within(ref, later):
    """Mix effect at reference route means, within-route effect at later shares; they sum to the drop.

    Both periods are standardised to the state-routes observed in both, so the two effects add exactly
    to the mean drop.
    """
    routes = ref[ROUTE].value_counts().index.intersection(later[ROUTE].value_counts().index)
    ref, later = ref[ref[ROUTE].isin(routes)], later[later[ROUTE].isin(routes)]
    shares = pd.DataFrame({
        "ref": ref[ROUTE].value_counts(normalize=True),
        "later": later[ROUTE].value_counts(normalize=True)}).fillna(0.0)
    means = pd.concat([ref.groupby(ROUTE)["delivery_days"].mean().rename("ref"),
                       later.groupby(ROUTE)["delivery_days"].mean().rename("later")], axis=1).dropna()
    routes = means.index
    p_ref, p_later = shares.loc[routes, "ref"], shares.loc[routes, "later"]
    m_ref, m_later = means["ref"], means["later"]
    mix = float(((p_ref - p_later) * m_ref).sum())
    within = float((p_later * (m_ref - m_later)).sum())
    return mix, within


def route_decomposition(cohort, as_of):
    """Decompose the mean delivery-days drop (reference -> later) into route mix and within-route parts.

    Standardises both periods on the state-routes (seller_state->customer_state) observed in both, so the
    mix effect (at reference route means) and the within-route effect (at later shares) add exactly to the
    mean drop. Week-cluster bootstrap gives uncertainty.
    """
    known = cohort["outcome_available_at"].le(as_of)
    ref = add_route(cohort[cohort["split"].eq("train") & known])
    later = add_route(cohort[cohort["split"].eq("test") & known])
    routes = ref[ROUTE].value_counts().index.intersection(later[ROUTE].value_counts().index)
    ref, later = ref[ref[ROUTE].isin(routes)], later[later[ROUTE].isin(routes)]
    drop = float(ref["delivery_days"].mean() - later["delivery_days"].mean())
    mix, within = _mix_within(ref, later)

    ref_weeks, later_weeks = month_label(ref["order_purchase_timestamp"]), month_label(later["order_purchase_timestamp"])
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    draws = []
    ref_by_week = {w: g for w, g in ref.groupby(ref_weeks)}
    later_by_week = {w: g for w, g in later.groupby(later_weeks)}
    ref_keys, later_keys = list(ref_by_week), list(later_by_week)
    for _ in range(BOOTSTRAP_DRAWS):
        pick = lambda pools, keys: pd.concat([pools[k] for k in rng.choice(keys, size=len(keys), replace=True)])
        draw = _mix_within(pick(ref_by_week, ref_keys), pick(later_by_week, later_keys))
        draws.append(draw)
    boot = pd.DataFrame(draws, columns=["mix", "within"])
    boot["drop"] = boot["mix"] + boot["within"]
    quartiles = {c: {"median": round(float(boot[c].median()), 3),
                     "ci_low": round(float(boot[c].quantile(0.025)), 3),
                     "ci_high": round(float(boot[c].quantile(0.975)), 3)} for c in boot.columns}
    return {"reference": "train split", "later": "test split", "unit": "days",
            "state_routes_used": int(len(routes)),
            "orders_retained_share": round((len(ref) + len(later))
                                           / (cohort["split"].isin(["train", "test"]) & known).sum(), 4),
            "mean_drop": round(drop, 3), "route_mix_effect": round(mix, 3),
            "within_route_effect": round(within, 3),
            "bootstrap": {"draws": BOOTSTRAP_DRAWS, "seed": BOOTSTRAP_SEED, "cluster": "purchase week",
                          **{f"bootstrap_{k}": v for k, v in quartiles.items()}}}


# ------------------------------------------------------------------------------------- report

def default_as_of(cohort):
    return cohort["outcome_available_at"].max()


def collect_report(cohort, spec, as_of, root=REPO_ROOT, results_root=DASHBOARD_PHASE2):
    as_of = pd.Timestamp(as_of)
    later_mask = cohort["split"].eq("test")
    month = pd.Series(month_label(cohort["order_purchase_timestamp"]), index=cohort.index)
    windows = {"test_overall": later_mask}
    windows.update({m: later_mask & month.eq(m) for m in sorted(month[later_mask].unique())})

    selected = selected_models(results_root)
    models, contracts, train = fit_selected_models(cohort, spec, selected)
    validation = validation_metrics(train, spec, contracts, selected)
    report = {
        "as_of": str(as_of),
        "reference": {"period": "training split", "cutoff": spec["primary_cutoff"],
                      "orders": int(cohort["split"].eq("train").sum())},
        "selected_models": selected,
        "x_drift": x_drift(cohort, spec, windows, cutoff=as_of),
        "y_drift": y_drift(cohort, as_of),
        "x_to_y": {},
    }
    report["x_to_y"].update(xy_regressions(cohort, models, contracts, train, as_of, windows))
    for stage in ("checkout", "handover"):
        report["x_to_y"][stage].update(validation[stage])
    report["x_to_y"]["late_warning"] = xy_classifier(cohort, models["late_warning"], contracts["late_warning"],
                                                     train, as_of, windows)
    report["x_to_y"]["late_warning"].update(validation["late_warning"])
    report["route_decomposition"] = route_decomposition(cohort, as_of)
    report["input_counts"] = {stage: len(sum(feature_columns(contracts[stage]), []))
                              for stage in ("checkout", "handover", "late_warning")}
    return report


def run_report(as_of=None, root=REPO_ROOT, results_root=DASHBOARD_PHASE2, out_dir=RESULTS):
    cohort, spec, coverage = load_cohort(root)
    as_of = default_as_of(cohort) if as_of is None else pd.Timestamp(as_of)
    report = collect_report(cohort, spec, as_of, root, results_root)
    report["coverage"] = coverage
    ledger = build_ledger(report)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "drift_report.json").write_text(json.dumps(report, indent=1, default=_json_default) + "\n")
    pd.DataFrame(ledger).to_csv(out_dir / "ledger.csv", index=False)
    return report, ledger


def _json_default(value):
    return value.item() if hasattr(value, "item") else str(value)


# ------------------------------------------------------------------------------------- ledger

def _match(reproduced, stated, rel_tol=0.02):
    """yes = differs only in rounding; close = within 2%; no = a finding."""
    if reproduced is None or stated is None:
        return "no"
    stated = float(stated)
    decimals = len(repr(stated).split(".")[1]) if "." in repr(stated) else 0
    if abs(float(reproduced) - stated) <= 0.5 * 10 ** -decimals + 1e-12:
        return "yes"
    if abs(float(reproduced) - stated) <= rel_tol * max(abs(stated), 1e-9):
        return "close"
    return "no"


def _match_pair(reproduced_pair, stated_pair):
    return {"from": _match(reproduced_pair[0], stated_pair[0]),
            "to": _match(reproduced_pair[1], stated_pair[1])}


def _pair_verdict(matches):
    return "yes" if all(v == "yes" for v in matches.values()) else ("close" if "no" not in matches.values() else "no")


def _window(report, stage, window):
    return report["x_drift"][stage]["windows"][window]


def build_ledger(report):
    y, xy, dec = report["y_drift"], report["x_to_y"], report["route_decomposition"]
    monthly = {r["month"]: r for r in y["monthly"]}
    aug, june = monthly.get("2018-08", {}), monthly.get("2018-06", {})
    overall_x = {stage: _window(report, stage, "test_overall") for stage in ("checkout", "handover", "late_warning")}
    example = y["maturity_example"]
    cls = xy["late_warning"]["later_overall"]
    checkout_model = report["selected_models"]["checkout"]["candidate"]
    handover_model = report["selected_models"]["handover"]["candidate"]
    rows = []

    def pct(value):
        return None if value is None else round(value * 100, 2)

    def add(claim, source, stated, reproduced, definition, note=""):
        if isinstance(stated, tuple):
            matches = _match_pair(reproduced, stated)
            rows.append({"claim": claim, "source": source, "stated_value": f"{stated[0]} -> {stated[1]}",
                         "reproduced_value": f"{reproduced[0]} -> {reproduced[1]}",
                         "definition_used": definition, "match": _pair_verdict(matches), "note": note})
        else:
            rows.append({"claim": claim, "source": source, "stated_value": stated,
                         "reproduced_value": reproduced, "definition_used": definition,
                         "match": _match(reproduced, stated), "note": note})

    ks = lambda stage, feature: overall_x[stage][feature]["ks"]
    med = lambda stage, feature: (overall_x[stage][feature]["median_reference"], overall_x[stage][feature]["median_window"])
    add("KS promise_slack", "brief", 0.296, ks("checkout", PROMISE_SLACK),
        "derived promised_days - route_typical_days, train vs test; not a current selected-model input")
    add("promise_slack medians", "brief", (11.30, 7.37), med("checkout", PROMISE_SLACK),
        "train median -> later-period median")
    add("KS route_typical_days", "brief", 0.274, ks("checkout", "route_typical_days"), "train vs test overall")
    add("KS promised_days", "brief", 0.228, ks("checkout", "promised_days"), "train vs test overall")
    add("promised_days medians", "brief", (23.49, 20.36), med("checkout", "promised_days"),
        "train median -> later-period median")
    add("KS total_freight", "brief", 0.177, ks("checkout", "total_freight"), "train vs test overall")
    add("KS seller_ship_days", "brief", 0.158, ks("checkout", "seller_ship_days"), "train vs test overall",
        "0.16 reproduced; NaN-handling variants (imputation) move it only to ~0.16")
    add("KS route_transit_90d_at_handover", "brief", 0.270, ks("handover", "route_transit_90d_at_handover"),
        "train vs test overall")
    add("KS remaining_slack", "brief", 0.182, ks("late_warning", "remaining_slack"),
        "current definition: promised_days + 1 - handover_days - route_transit_90d_at_handover",
        "PDF medians are exactly 1.0 lower: it used the promised date itself as the deadline; KS is shift-invariant")
    add("remaining_slack medians (PDF deadline definition)", "brief", (11.40, 9.01),
        med("late_warning", REMAINING_SLACK_PDF),
        "deadline = the promised date, as in the PDF: promised_days - handover_days - route_transit_90d_at_handover",
        "reproduces the PDF exactly; the current selected-model input adds the 1-day calendar-date deadline")
    add("remaining_slack medians (current definition)", "brief", (11.40, 9.01), med("late_warning", "remaining_slack"),
        "current definition (deadline = promised date + 1 day)", "1.0 higher than the PDF by definition")
    add("KS state_backlog_recent_vs_8wk", "brief", 0.262, ks("late_warning", "state_backlog_recent_vs_8wk"),
        "train vs test overall",
        "per-month KS: May 0.53, June 0.65, July 0.26, August 0.38; July is closest, so the PDF may have quoted "
        "a single month or an earlier version of the feature")
    add("missing approval_days train", "brief", 0.71, round(overall_x["handover"]["approval_days"]["missing_reference"] * 100, 2),
        "share with approval unknown at handover, %")
    add("missing approval_days later", "brief", 3.39, round(overall_x["handover"]["approval_days"]["missing_window"] * 100, 2),
        "share with approval unknown at handover, %")
    add("missing approval_days July", "brief", 8.60,
        round(_window(report, "handover", "2018-07")["approval_days"]["missing_window"] * 100, 2),
        "share with approval unknown at handover, %, July cohort")
    add("mean delivery days train -> later", "brief", (13.29, 8.78),
        (y["reference_period"]["mean_delivery_days"], y["later_period"]["mean_delivery_days"]),
        "mean purchase-to-delivery days, matured orders")
    mar, apr = monthly.get("2018-03", {}), monthly.get("2018-04", {})
    add("mean delivery days Mar -> Apr 2018", "brief", (16.30, 11.50),
        (mar.get("mean_delivery_days"), apr.get("mean_delivery_days")),
        "all cohort orders by purchase month (train and later-arriving)")
    add("promised days June -> August 2018", "brief", (27.78, 15.17),
        (june.get("mean_promised_days"), aug.get("mean_promised_days")),
        "mean promised_days by purchase month, later period")
    add("late rate June -> August 2018", "brief", (1.16, 6.23),
        (pct(june.get("late_rate_matured")), pct(aug.get("late_rate_matured"))),
        "late rate of matured orders by purchase month, %")
    add("May 1-25 orders", "brief", 5964, example["orders"], "cohort purchases 1-25 May 2018")
    add("labelled by 26 May", "brief", 4088, example["labelled_by_as_of"],
        "delivered (outcome recorded) before the Phase 2 cutoff")
    add("labelled mean delivery days", "brief", 7.57, example["labelled_mean_delivery_days"],
        "matured-by-cutoff subset of 1-25 May")
    add("labelled late rate %", "brief", 0.61, pct(example["labelled_late_rate"]),
        "matured-by-cutoff subset of 1-25 May")
    add("full-set mean delivery days", "brief", 11.33, example["full_mean_delivery_days"], "all 1-25 May orders")
    add("full-set late rate %", "brief", 7.39, pct(example["full_late_rate"]), "all 1-25 May orders")
    add("route-mix effect of duration drop", "brief", 0.46, dec["route_mix_effect"],
        "state-route mix effect at reference route means; drop = train mean - test mean")
    add("within-route effect of duration drop", "brief", 3.99, dec["within_route_effect"],
        "within-route mean change at later shares")
    add("mean duration drop", "brief", 4.49, dec["mean_drop"], "train mean - test mean delivery days")
    add("bootstrap route-mix median", "brief", 0.46, dec["bootstrap"]["bootstrap_mix"]["median"],
        "week-cluster bootstrap, 300 draws, seed 42")
    add("bootstrap within-route median", "brief", 3.99, dec["bootstrap"]["bootstrap_within"]["median"],
        "week-cluster bootstrap, 300 draws, seed 42")
    add("bootstrap drop median", "brief", 4.49, dec["bootstrap"]["bootstrap_drop"]["median"],
        "week-cluster bootstrap, 300 draws, seed 42",
        "the PDF's 4.49 matches the point drop (reproduced 4.492); the bootstrap median of the drop is lower "
        "because weeks are resampled independently per period")
    add("checkout MAE train in-sample", "brief", 5.2350, xy["checkout"]["train_in_sample_mae"],
        f"{checkout_model} fit on the training split, scored on it")
    add("checkout MAE validation mean", "brief", 5.8085, xy["checkout"]["validation_mean_mae"],
        "mean over 5 chronological validation windows")
    add("checkout MAE later", "brief", 4.9951, xy["checkout"]["later_overall"]["mae"],
        "refit on train, scored on the test split")
    add("checkout later signed error", "brief", 3.65, xy["checkout"]["later_overall"]["signed_error"],
        "predicted - actual, test split")
    add("checkout later R2", "brief", -0.175, xy["checkout"]["later_overall"]["r2"], "test split")
    add("handover MAE train in-sample", "brief", 4.5730, xy["handover"]["train_in_sample_mae"],
        f"{handover_model} fit on the training split, scored on it")
    add("handover MAE validation mean", "brief", 5.0986, xy["handover"]["validation_mean_mae"],
        "mean over 5 chronological validation windows")
    add("handover MAE later", "brief", 3.9720, xy["handover"]["later_overall"]["mae"],
        "refit on train, scored on the test split")
    add("handover later signed error", "brief", 2.87, xy["handover"]["later_overall"]["signed_error"],
        "predicted - actual, test split")
    add("classifier mean predicted risk", "brief", 6.87, pct(cls["mean_predicted_risk"]),
        "mean predicted late probability on the test split, %")
    add("classifier observed late rate", "brief", 3.49, pct(cls["observed_late_rate"]),
        "late rate on the test split, %")
    add("classifier expected late", "brief", 1321, cls["expected_late"], "mean predicted risk x test orders")
    add("classifier observed late", "brief", 672, cls["late"], "late orders on the test split")
    add("classifier AP pooled", "brief", 0.489, cls["ap_pooled"], "average precision over the pooled test split",
        "Phase 2's own saved test_scores.csv gives the same 0.434; 0.489 could not be reproduced from the "
        "selected model under any AP variant tried (see the .out notes)")
    add("classifier AP month-weighted", "brief", 0.385, cls["ap_month_weighted"],
        "AP with each test month weighted equally (Phase 2 weighted_pr_auc)")
    add("classifier AP validation mean", "brief", 0.392, xy["late_warning"]["validation_mean_ap"],
        "mean over 5 chronological validation windows")
    add("CheckoutTestBias", "docs/phase2/report/generated/numbers.tex", 3.64,
        xy["checkout"]["later_overall"]["signed_error"],
        "checkout later signed error")
    add("checkout inputs", "brief", 20, report["input_counts"]["checkout"], "numeric + categorical contract inputs",
        "current Phase 2 contract has 19; the contract reproduces the saved Phase 2 test metrics, so the PDF "
        "likely counted differently (its refit metrics match this contract to rounding)")
    add("handover inputs", "brief", 23, report["input_counts"]["handover"], "numeric + categorical contract inputs",
        "current Phase 2 contract has 22; see the checkout note")
    add("classifier inputs", "brief", 27, report["input_counts"]["late_warning"], "numeric + categorical contract inputs",
        "current Phase 2 contract has 26; see the checkout note")
    return rows


# ---------------------------------------------------------------------------------------- CLI

def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--as-of", default=None, help="reporting date; defaults to the end of the data")
    args = parser.parse_args()
    report, ledger = run_report(as_of=args.as_of)
    print(f"as_of {report['as_of']}; report and ledger written to {RESULTS}")
    print(pd.DataFrame(ledger)[["claim", "stated_value", "reproduced_value", "match"]].to_string(index=False))


if __name__ == "__main__":
    main()
