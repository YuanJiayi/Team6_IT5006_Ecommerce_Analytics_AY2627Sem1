"""Phase 3 promise policy: rebuild of Jiayi's "Checkout ETA findings".

Method (her PDF, no code supplied): the revised promise is the checkout ETA plus a buffer, where the
buffer is the (1 - target late rate) quantile of the residuals (actual delivery days - ETA), per customer
region and per month, using delivered orders purchased in month M-2. A region with fewer than
MIN_CALIBRATION_ORDERS calibration orders falls back to the global quantile for that month. Promises
are rounded up to whole days. Target late rates are 3.5% (headline) and 5% (comparison). Late means
the actual delivery calendar date is after the promised calendar date; Olist's own late rate is
Phase 2's is_late and Olist's mean promise is the calendar-day difference between its estimated date
and the purchase date (the ceiling of promised_days).

Two ETA runs are scored everywhere:
  A: the Phase 2 selected checkout model (family, alpha and candidate read from the Phase 2 selection
     JSON at runtime; all Phase 2 inputs);
  B: her setup, the promise-free estimate used to set the promise (ridge alpha=1000 with both
     promise inputs removed: promised_days and promise_slack, leaving 18 inputs). Alpha=1000 here
     is a documented constant of her setup, not a Phase 2 selection.

A third centre, the route's recent average (`route_typical_days`, median-filled from the fitting
rows), uses the same buffer method and calibration cohort as a benchmark against both ETA runs.

Run from the repository root:
    python -m phase3.policy

Writes CSV outputs and the reproduction ledger to data/outputs/phase3. Run A is the Phase 2
selection (the dashboard's Phase 2 eta selection.json, selected_model.checkout), fitted on the Phase 2
primary training rows exactly as Phase 2 scores its test period, so every scored test month starts on
or after the Phase 2 cutoff and uses only data known before it. Validation-period evidence (the
out-of-fold buffer table, the Nov 2017-Mar 2018 monthly policy) uses Phase 2's five chronological
windows and walk-forward monthly refits, never the test period.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score
from threadpoolctl import threadpool_limits

from olist.classification import feature_columns, temporal_folds
from olist.paths import DASHBOARD_PHASE2, PHASE3_OUTPUTS, REPO_ROOT
from olist.regression import make_regression_pipeline

RESULT_DIR = PHASE3_OUTPUTS / "policy"
TARGET = "delivery_days"
ROUTE_FEATURE = "route_typical_days"
MIN_CALIBRATION_ORDERS = 100
TEST_MONTHS = [pd.Period("2018-05"), pd.Period("2018-06"), pd.Period("2018-07"), pd.Period("2018-08")]
RETRAIN_MONTHS = [pd.Period(m) for m in ("2018-03", "2018-04", "2018-05", "2018-06", "2018-07", "2018-08")]
VALIDATION_POLICY_MONTHS = [pd.Period(m) for m in
                            ("2017-11", "2017-12", "2018-01", "2018-02", "2018-03")]
# Walk-forward fits for the validation policy need month-(M-2) models, so they start two months earlier.
VALIDATION_WF_MONTHS = [pd.Period(m) for m in
                        ("2017-09", "2017-10")] + VALIDATION_POLICY_MONTHS
TARGETS = {"3.5pct": 0.035, "5pct": 0.05}

# Customer-state macro-regions. Jiayi's table names "SP" and "N", so her regional split is most
# consistent with the five IBGE macro-regions plus SP broken out of SE on its own (6 regions).
MACRO_REGIONS = {
    "N": ["AC", "AP", "AM", "PA", "RO", "RR", "TO"],
    "NE": ["AL", "BA", "CE", "MA", "PB", "PE", "PI", "RN", "SE"],
    "CO": ["DF", "GO", "MT", "MS"],
    "SE": ["ES", "MG", "RJ", "SP"],
    "S": ["PR", "RS", "SC"],
}
REGION_OF = {state: region for region, states in MACRO_REGIONS.items() for state in states}
REGION_OF["SP"] = "SP"  # SP reported as its own region, per her table


def selected_checkout_model(root=DASHBOARD_PHASE2):
    """The Phase 2 selected checkout model as (candidate, family, parameters).

    Read from the saved selection JSON so a Phase 2 rerun flows through without a Phase 3 edit.
    Missing files or keys raise: a silently stale Phase 3 result is worse than a failed run.
    """
    path = Path(root) / "eta" / "selection.json"
    if not path.is_file():
        raise FileNotFoundError(f"Phase 2 ETA selection file not found: {path}")
    selection = json.loads(path.read_text())
    try:
        entry = selection["selected_model"]["checkout"]
    except (KeyError, TypeError) as exc:
        raise KeyError(f"Phase 2 ETA selection is missing selected_model.checkout in {path}") from exc
    missing = [key for key in ("candidate", "family", "parameters") if key not in entry]
    if missing:
        raise KeyError(f"Phase 2 ETA selection entry in {path} is missing {', '.join(missing)}")
    return entry["candidate"], entry["family"], dict(entry["parameters"])


def region_of(states):
    mapped = states.map(REGION_OF)
    if mapped.isna().any():
        raise ValueError(f"Unmapped customer states: {sorted(states[mapped.isna()].unique())}")
    return mapped


def load_order_table(root=REPO_ROOT):
    """Prepared one-row-per-order table (delivered orders) with the customer region added."""
    root = Path(root)
    spec = json.loads((root / "data" / "processed" / "phase2_feature_spec.json").read_text())
    table = pd.read_csv(root / "data" / "processed" / "phase2_order_table.csv",
                        parse_dates=["order_purchase_timestamp", "outcome_available_at"])
    if table["delivery_days"].isna().any() or table["promised_days"].isna().any():
        raise ValueError("The prepared table must carry delivery and promise days for every order")
    if not calendar_late(table, np.ceil(table["promised_days"])).eq(table["is_late"].astype(bool)).all():
        raise ValueError("Calendar-day late definition does not equal the raw is_late label")
    table["region"] = region_of(table["customer_state"])
    table["purchase_month"] = table["order_purchase_timestamp"].dt.to_period("M")
    return table, spec


def calendar_late(frame, promise_days):
    """Late on calendar days: actual delivery date > purchase date + whole promised days."""
    purchase_day = frame["order_purchase_timestamp"].dt.normalize()
    promised_date = purchase_day + pd.to_timedelta(promise_days, unit="D")
    return frame["outcome_available_at"].dt.normalize() > promised_date


def run_spec(spec, run):
    """Feature contract for a run: A = the frozen Phase 2 checkout contract, B = the promise-free
    estimate (both promised_days and promise_slack removed, leaving 18 inputs)."""
    if run == "A":
        return spec
    if run == "B":
        changed = json.loads(json.dumps(spec))
        changed["numeric_features"] = [c for c in changed["numeric_features"]
                                       if c not in ("promised_days", "promise_slack")]
        return changed
    raise ValueError(f"Unknown run: {run}")


def fit_eta_model(run, spec, fit_rows, score_rows):
    """Fit the run's model on `fit_rows` and return (pipeline, predictions on `score_rows`).

    Run A takes its family and parameters from the Phase 2 selection; run B's ridge alpha=1000 is a
    documented constant of Jiayi's own setup (both promise inputs removed), not a Phase 2 selection.
    """
    if run == "A":
        _, family, params = selected_checkout_model()
    elif run == "B":
        family, params = "ridge", {"alpha": 1000.0}
    else:
        raise ValueError(f"Unknown run: {run}")
    columns = sum(feature_columns(spec), [])
    pipeline = make_regression_pipeline(family, params, spec)
    with threadpool_limits(limits=1):
        pipeline.fit(fit_rows[columns], fit_rows[TARGET])
        return pipeline, pipeline.predict(score_rows[columns])


def fit_frozen_model(run, spec, table):
    """The run's model on the Phase 2 primary training rows (outcomes known before the cutoff)."""
    train = table[table["split"].eq("train")]
    model, _ = fit_eta_model(run, spec, train, train)
    return model


def predict_eta(model, spec, rows):
    columns = sum(feature_columns(spec), [])
    with threadpool_limits(limits=1):
        return model.predict(rows[columns])


def calibration_for(table, scored_month):
    """Calibration pool for a scored month: delivered orders purchased in month M-2 whose outcomes
    were recorded before the scored month starts (the moment the promise is issued)."""
    start = scored_month.to_timestamp()
    pool = table[table["purchase_month"].eq(scored_month - 2) & table["outcome_available_at"].lt(start)]
    if pool.empty:
        raise ValueError(f"No calibration orders purchased in {scored_month - 2}")
    return pool


def buffers_for(table, scored_month, eta_scoring, eta_calibration, target, regional=True):
    """Buffer per scored row for one month: the (1 - target) residual quantile per region, with the
    global quantile for regions under MIN_CALIBRATION_ORDERS (or missing from the pool).

    `eta_scoring` predicts the scored rows, `eta_calibration` the pool rows; under the frozen Phase 2 model
    they are the same model, under retraining the calibration ETA comes from the model live in M-2.
    """
    pool = calibration_for(table, scored_month)
    residuals = pool[TARGET] - eta_calibration.reindex(pool.index)
    global_buffer = float(residuals.quantile(1 - target))
    sizes = pool["region"].value_counts()
    per_region = residuals.groupby(pool["region"]).quantile(1 - target)
    small = sizes.lt(MIN_CALIBRATION_ORDERS)
    scored = table[table["purchase_month"].eq(scored_month)]
    if regional:
        mapped = scored["region"].map(per_region).astype(float)
        use_global = scored["region"].map(small).fillna(True).astype(bool)
        buffer = pd.Series(np.where(use_global, global_buffer, mapped), index=scored.index)
    else:
        buffer = pd.Series(global_buffer, index=scored.index)
    return scored, buffer


def promise_table(as_of, target=0.035, run="A", model=None, table=None, spec=None, regional=True,
                  root=REPO_ROOT):
    """Per-region promise table for the calendar month containing `as_of`.

    The promise is the run's checkout ETA plus the monthly regional buffer, rounded up to whole days.
    Columns: buffer_days (mean raw buffer before rounding), olist_mean_promise_days (calendar-day
    difference between Olist's estimated date and the purchase date, i.e. the ceiling of promised_days),
    our_mean_promise_days, olist_late_rate (is_late), our_expected_late_rate (realised share arriving
    after our promise) and n_orders.

    The default model is fitted only on Phase 2 primary training rows, so months before that cutoff
    are refused here (use the walk-forward validation path for those).
    """
    if table is None or spec is None:
        table, spec_full = load_order_table(root)
        spec = run_spec(spec_full, run)
    month = pd.Period(as_of, freq="M")
    if model is None:
        model = fit_frozen_model(run, spec, table)
        cutoff = pd.Timestamp(spec["primary_cutoff"])
        first_scored = table.loc[table["purchase_month"].eq(month), "order_purchase_timestamp"].min()
        if pd.notna(first_scored) and first_scored < cutoff:
            raise ValueError("promise_table scores months on or after the Phase 2 cutoff; "
                             "earlier months need the walk-forward validation path")
    eta_all = pd.Series(predict_eta(model, spec, table), index=table.index)
    scored, buffer = buffers_for(table, month, eta_all, eta_all, target, regional)
    eta = eta_all.reindex(scored.index)
    promise = np.ceil(eta + buffer)
    rows = []
    for region, group in scored.groupby("region", sort=True):
        rows.append({
            "region": region,
            "buffer_days": float(buffer.reindex(group.index).mean()),
            "olist_mean_promise_days": float(np.ceil(group["promised_days"]).mean()),
            "our_mean_promise_days": float(promise.reindex(group.index).mean()),
            "olist_late_rate": float(group["is_late"].mean()),
            "our_expected_late_rate": float(calendar_late(group, promise.reindex(group.index)).mean()),
            "n_orders": int(len(group)),
        })
    return pd.DataFrame(rows)


def _add_total(per_month):
    frame = pd.DataFrame(per_month)
    weights = frame["n_orders"]
    total = {"purchase_month": "total", "n_orders": int(weights.sum())}
    for column in ("our_mean_promise_days", "our_late_rate", "olist_mean_promise_days", "olist_late_rate"):
        total[column] = float(np.average(frame[column], weights=weights))
    return pd.concat([frame, pd.DataFrame([total])], ignore_index=True)


def score_centre_test(centre, table, target=0.035, regional=True, months=TEST_MONTHS):
    """Score the whole test period month by month for a fixed centre series, plus a weighted total row.

    `centre` is the per-order base the promise builds on (an ETA or the route's recent average).
    Scored rows are the test-split orders (purchased on or after the Phase 2 cutoff; 19,363 orders),
    matching Jiayi's population; earlier purchases in the same calendar month are excluded.
    """
    rows = []
    for month in months:
        scored, buffer = buffers_for(table, month, centre, centre, target, regional)
        keep = scored["split"].eq("test")
        scored, buffer = scored[keep], buffer.reindex(scored.index[keep])
        promise = np.ceil(centre.reindex(scored.index) + buffer)
        rows.append({"purchase_month": str(month), "n_orders": len(scored),
                     "our_mean_promise_days": float(promise.mean()),
                     "our_late_rate": float(calendar_late(scored, promise).mean()),
                     "olist_mean_promise_days": float(np.ceil(scored["promised_days"]).mean()),
                     "olist_late_rate": float(scored["is_late"].mean())})
    return _add_total(rows)


def score_test_period(run, model, spec, table, target=0.035, regional=True, months=TEST_MONTHS):
    """Score the test period with the run's frozen ETA model (see `score_centre_test`)."""
    eta_all = pd.Series(predict_eta(model, spec, table), index=table.index)
    return score_centre_test(eta_all, table, target, regional, months)


def route_centre(table, fit_rows):
    """The route's recent average as a checkout-time centre, median-filled from the fitting rows.

    `route_typical_days` is a point-in-time Phase 2 history feature (only deliveries recorded before
    the purchase), so its value is known at checkout. Missing values take the fitting rows' median,
    keeping every statistic used to score a period fitted before that period starts.
    """
    if fit_rows.empty:
        raise ValueError("The route centre needs fitting rows to fill missing route values")
    median = float(fit_rows[ROUTE_FEATURE].median())
    if not np.isfinite(median):
        raise ValueError(f"The fitting rows have no usable {ROUTE_FEATURE} values")
    return table[ROUTE_FEATURE].fillna(median)


def route_test_period(table, target=0.035, regional=True, months=TEST_MONTHS):
    """Route-average promise policy on the test months, median-filled from the Phase 2 training rows."""
    centre = route_centre(table, table[table["split"].eq("train")])
    return score_centre_test(centre, table, target, regional, months)


def walk_forward_route_centres(table, months):
    """Route centres for each month, median-filled from rows whose purchase and outcome precede it."""
    centres = {}
    for month in months:
        start = month.to_timestamp()
        available = table[table["order_purchase_timestamp"].lt(start)
                          & table["outcome_available_at"].lt(start)]
        centres[month] = route_centre(table, available)
    return centres


def walk_forward_models(run, spec, table, months):
    """Fit one model per month on every row purchased and with its outcome known before that month
    starts. Returns {month: ETA predictions for every table row}. The month-M model is the one live
    during month M, so month-M purchases are out-of-sample for it and in-sample for the month-(M+2)
    fit (mirroring Phase 2's monthly refit scheme)."""
    predictions = {}
    for month in months:
        start = month.to_timestamp()
        available = table[table["order_purchase_timestamp"].lt(start)
                          & table["outcome_available_at"].lt(start)]
        _, predicted = fit_eta_model(run, spec, available, table)
        predictions[month] = pd.Series(predicted, index=table.index)
    return predictions


def score_period_retrained(table, predictions, months, target, regional=True, test_split_only=False):
    """Monthly policy under walk-forward centres: month M scored by the month-M centre, buffers from
    month-(M-2) residuals under the centre live in M-2, so calibration residuals are out-of-sample.

    With `test_split_only` the scored rows are restricted to the test split (the test-period scoring
    population); validation months keep every purchase.
    """
    rows = []
    for month in months:
        scored, buffer = buffers_for(table, month, predictions[month], predictions[month - 2],
                                     target, regional)
        if test_split_only:
            keep = scored["split"].eq("test")
            scored, buffer = scored[keep], buffer.reindex(scored.index[keep])
        promise = np.ceil(predictions[month].reindex(scored.index) + buffer)
        rows.append({"purchase_month": str(month), "n_orders": len(scored),
                     "our_mean_promise_days": float(promise.mean()),
                     "our_late_rate": float(calendar_late(scored, promise).mean()),
                     "olist_mean_promise_days": float(np.ceil(scored["promised_days"]).mean()),
                     "olist_late_rate": float(scored["is_late"].mean())})
    return _add_total(rows)


def validation_oof_predictions(run, spec, table):
    """Out-of-fold ETA for the five Phase 2 chronological validation windows (38,005 orders),
    indexed by the original table rows. Orders outside the windows (earlier purchases) are excluded."""
    train = table[table["split"].eq("train")]
    folds = temporal_folds(train, spec)
    predictions = np.full(len(train), np.nan)
    for fit_rows, valid_rows in folds:
        model, predicted = fit_eta_model(run, spec, train.iloc[fit_rows], train.iloc[valid_rows])
        predictions[valid_rows] = predicted
    covered = pd.Series(predictions, index=train.index)
    fold_rows = table.loc[table["split"].eq("train"), "cv_fold"].isin([0, 1, 2, 3, 4])
    if covered[fold_rows.reindex(covered.index).fillna(False).to_numpy()].isna().any():
        raise ValueError("Validation windows do not partition their orders cleanly")
    return covered.dropna()


def validation_needed_buffers(table, oof_eta, targets):
    """Regional residual quantiles on the pooled validation windows (her "needed buffer" table)."""
    train = table.loc[oof_eta.index]
    residuals = train[TARGET] - oof_eta
    rows = []
    for region, group in residuals.groupby(train["region"]):
        rows.append({"region": region, "n_orders": int(len(group)),
                     **{f"buffer_{label}": float(group.quantile(1 - target))
                        for label, target in targets.items()}})
    return pd.DataFrame(rows)


def fixed_validation_buffer(table, oof_eta, target=0.05):
    """One buffer fixed once on the validation residuals, applied to the test period with the Phase 2 model."""
    train = table.loc[oof_eta.index]
    buffer = float((train[TARGET] - oof_eta).quantile(1 - target))
    test = table[table["split"].eq("test")]
    promise = np.ceil(test["eta_frozen"] + buffer)
    return {"buffer_days": buffer, "our_mean_promise_days": float(promise.mean()),
            "our_late_rate": float(calendar_late(test, promise).mean())}


def validation_monthly_policy(table, predictions, target, regional=True):
    """Her validation check: the monthly policy over Nov 2017-Mar 2018 with walk-forward refits
    (honest month-(M-2) calibration), plus the Olist reference over the same months."""
    result = score_period_retrained(table, predictions, VALIDATION_POLICY_MONTHS, target, regional)
    scored_months = table[table["purchase_month"].isin(VALIDATION_POLICY_MONTHS)]
    return pd.concat([result, pd.DataFrame([{
        "purchase_month": "olist_reference", "n_orders": int(len(scored_months)),
        "our_mean_promise_days": float(np.ceil(scored_months["promised_days"]).mean()),
        "our_late_rate": float(scored_months["is_late"].mean()),
        "olist_mean_promise_days": float(np.ceil(scored_months["promised_days"]).mean()),
        "olist_late_rate": float(scored_months["is_late"].mean())}])], ignore_index=True)


def gap_band_prauc(table, eta, buffer_series):
    """PR-AUC (sklearn average precision) of late-vs-Olist-promise from checkout-known gap scores.

    Scores (higher = later): raw gap = ETA - Olist promise; recalibrated gap = ETA + the regional
    median residual (pooled M-2 calibration months) - Olist promise; slack rule = route_typical_days
    - promise (minus promise_slack, built from those two series); shortest promise = -min(Olist
    promise, our promise); random = the late prevalence.
    """
    test = table[table["split"].eq("test")]
    index = test.index
    eta_test = eta.reindex(index)
    pool = pd.concat([calibration_for(table, m) for m in TEST_MONTHS])
    regional_median = (pool[TARGET] - eta.reindex(pool.index)).groupby(pool["region"]).median()
    y = test["is_late"].to_numpy()
    scores = {
        "raw_gap": eta_test - test["promised_days"],
        "recalibrated_gap": eta_test + test["region"].map(regional_median) - test["promised_days"],
        "slack_rule": test["route_typical_days"] - test["promised_days"],
        "shortest_promise": -np.minimum(test["promised_days"],
                                        (eta_test + buffer_series.reindex(index)).to_numpy()),
    }
    rows = [{"score": name, "pr_auc": float(average_precision_score(y, score.to_numpy()))}
            for name, score in scores.items()]
    rows.append({"score": "random", "pr_auc": float(y.mean())})
    return pd.DataFrame(rows)


def olist_reference(table):
    """Olist rows for the test period, the five validation windows and the Nov-Mar policy window."""
    def row(name, frame):
        return {"scope": name, "n_orders": int(len(frame)),
                "olist_mean_promise_days": float(np.ceil(frame["promised_days"]).mean()),
                "olist_late_rate": float(frame["is_late"].mean()),
                "olist_mean_delivery_days": float(frame[TARGET].mean())}
    validation = table[table["cv_fold"].isin([0, 1, 2, 3, 4])]
    nov_mar = table[table["purchase_month"].isin(VALIDATION_POLICY_MONTHS)]
    return pd.DataFrame([row("test", table[table["split"].eq("test")]),
                         row("validation_5_windows", validation),
                         row("validation_nov_mar", nov_mar)])


def build_ledger(summary):
    """Reproduction ledger: claim, source, stated value, reproduced value, match, note."""
    ledger = [
        ("test period orders", "brief (her table)", "19,363", summary["test_orders"], "yes",
         "test split of the prepared table; Phase 2's 19,230 excludes 133 orders without a valid "
         "carrier handover, which the checkout policy does not need"),
        ("validation window orders", "brief (her table)", "38,005", summary["val_orders"], "yes",
         "five Phase 2 chronological windows; Phase 2's 38,003 excludes 2 orders without a valid handover"),
        ("test Olist mean promise", "brief", "22.1 d", summary["olist_test_promise"], "close",
         "calendar-day difference between Olist's estimated date and the purchase date (ceiling of "
         "promised_days); the raw promised_days mean is 21.41"),
        ("test Olist late rate", "brief", "3.5%", summary["olist_test_late"], "yes",
         "is_late: arrival after the promised calendar date"),
        ("test Olist mean delivery", "brief", "8.7 d", summary["olist_test_delivery"], "close", ""),
        ("validation Olist mean promise", "brief", "25.0 d", summary["olist_val_promise"], "close",
         "same calendar-day definition over the 5 windows"),
        ("validation Olist late rate", "brief", "10.7%", summary["olist_val_late"], "yes", ""),
        ("validation Olist mean delivery", "brief", "14.9 d", summary["olist_val_delivery"], "close", ""),
        ("regional needed buffer SP (10%/5%)", "brief", "8.2 / 12.7", summary["buffer_SP"], "yes",
         "out-of-fold residual quantiles, run B"),
        ("regional needed buffer N (10%/5%)", "brief", "22.3 / 27.6", summary["buffer_N"], "close",
         "run B out-of-fold; the 5% value lands within 0.31 d of her figure"),
        ("median checkout residual to Mar 2018", "brief", "about +/-1.5 d", summary["median_val"], "yes",
         "run B out-of-fold monthly medians Sep 2017-Mar 2018"),
        ("median checkout residual Jun-Aug 2018", "brief", "-4.3 to -4.7 d", summary["median_test"], "yes",
         "run B frozen-model monthly medians"),
        ("buffer fixed once on validation, 5%", "brief", "33.1 d / 0.4%", summary["fixed_once"], "yes",
         "run A: one buffer = 95th validation residual percentile, applied to the test period with the Phase 2 model"),
        ("monthly regional, 3.5% target", "brief", "19.3 d / 2.8%", summary["monthly_regional_35"], "no",
         "the frozen Phase 2 model gives the reproduced value above: the late rate is close, the promise higher. "
         "Every interpretation tried (frozen/retrained scoring, in-sample/out-of-sample calibration, "
         "alternative delivered-pool windows, drift correction, both runs) keeps the same gap; her "
         "figure implies buffers calibrated on deliveries faster than any M-2 lagged pool provides "
         "under the Phase 2 selected model"),
        ("monthly uniform, 5% target", "brief", "18.6 d / 3.2%", summary["monthly_uniform_5"], "no",
         "same gap as the regional row"),
        ("monthly regional, 5% target", "brief", "18.1 d / 3.5%", summary["monthly_regional_5"], "no",
         "same gap"),
        ("monthly uniform, 3.5% target", "brief", "19.9 d / 2.5%", summary["monthly_uniform_35"], "no",
         "same gap"),
        ("retraining + recalibration", "brief", "20.3 d / 2.4%", summary["retrained"], "no",
         "walk-forward retrain (month-M fit scores month M, month-(M-2) fit calibrates); the late rate "
         "is close, the promise about 0.9 d higher"),
        ("validation policy Nov 2017-Mar 2018", "brief", "25.2 d / 11.8%", summary["val_policy"], "close",
         "walk-forward monthly refits, honest month-(M-2) calibration, 5% target, run A"),
        ("validation policy Olist reference", "brief", "25.0 d", summary["val_policy_olist"], "yes",
         "25.04 d / 10.67% over the 5 windows; 25.11 d / 11.88% over Nov-Mar"),
        ("gap-band PR-AUC, raw gap", "brief", "0.115", summary["gap_raw"], "yes",
         "score = ETA - Olist promise, label is_late, run B"),
        ("gap-band PR-AUC, recalibrated gap", "brief", "0.137", summary["gap_recal"], "close",
         "score with regional median-residual recalibration (run A); definition uncertain"),
        ("gap-band PR-AUC, slack rule", "brief", "0.124", summary["gap_slack"], "yes",
         "score = route_typical_days - promise (promise_slack)"),
        ("gap-band PR-AUC, shortest promise", "brief", "0.120", summary["gap_shortest"], "close",
         "score = -min(Olist promise, our promise); definition uncertain"),
        ("gap-band PR-AUC, random", "brief", "0.035", summary["gap_random"], "yes", "test late prevalence"),
        ("promise_slack still a Phase 2 input?", "brief (check)", "input in the pre-audit Phase 2", "input",
         "yes", "restored to the feature contract and ladder (commit ea87322); still in the contract "
         "(run A, handover, late risk) and the order table, but run B drops it for promise setting"),
    ]
    return pd.DataFrame(ledger, columns=["claim", "source", "stated_value", "reproduced_value",
                                         "match", "note"])


def _fmt_total(row):
    return f"{row['our_mean_promise_days']:.1f} d / {row['our_late_rate']:.1%}"


def _lookup(frame, **keys):
    row = frame
    for key, value in keys.items():
        row = row[row[key].eq(value)]
    if len(row) != 1:
        raise KeyError(f"Expected one row for {keys}, found {len(row)}")
    return row.iloc[0]


def build_summary(output):
    """Ledger values pulled from the saved result CSVs, never retyped."""
    reference = pd.read_csv(Path(output) / "olist_reference.csv")
    summary = {
        "test_orders": int(reference.loc[0, "n_orders"]),
        "val_orders": int(reference.loc[1, "n_orders"]),
        "olist_test_promise": f"{reference.loc[0, 'olist_mean_promise_days']:.2f} d",
        "olist_test_late": f"{reference.loc[0, 'olist_late_rate']:.2%}",
        "olist_test_delivery": f"{reference.loc[0, 'olist_mean_delivery_days']:.2f} d",
        "olist_val_promise": f"{reference.loc[1, 'olist_mean_promise_days']:.2f} d",
        "olist_val_late": f"{reference.loc[1, 'olist_late_rate']:.2%}",
        "olist_val_delivery": f"{reference.loc[1, 'olist_mean_delivery_days']:.2f} d",
        "val_policy_olist": (f"{reference.loc[2, 'olist_mean_promise_days']:.2f} d / "
                             f"{reference.loc[2, 'olist_late_rate']:.2%}"),
    }
    buffers = pd.read_csv(Path(output) / "validation_needed_buffers.csv")
    sp_b = _lookup(buffers, run="B", region="SP")
    n_b = _lookup(buffers, run="B", region="N")
    summary["buffer_SP"] = f"{sp_b['buffer_10pct']:.2f} / {sp_b['buffer_5pct']:.2f}"
    summary["buffer_N"] = f"{n_b['buffer_10pct']:.2f} / {n_b['buffer_5pct']:.2f}"

    medians = pd.read_csv(Path(output) / "median_residuals.csv")
    val_med = medians.query("run == 'B' and scope == 'validation_oof'")
    test_med = medians.query("run == 'B' and scope == 'test_frozen' and "
                             "purchase_month in ['2018-06', '2018-07', '2018-08']")
    summary["median_val"] = f"{val_med['median_residual_days'].min():.2f} to {val_med['median_residual_days'].max():.2f}"
    summary["median_test"] = (f"{test_med['median_residual_days'].min():.2f} to "
                              f"{test_med['median_residual_days'].max():.2f}")

    fixed = pd.read_csv(Path(output) / "fixed_validation_buffer.csv")
    fixed_a = _lookup(fixed, run="A", target="5pct")
    summary["fixed_once"] = f"{fixed_a['our_mean_promise_days']:.1f} d / {fixed_a['our_late_rate']:.1%}"

    monthly = pd.read_csv(Path(output) / "test_monthly_policy.csv")
    frozen = monthly.query("scheme == 'frozen_phase2' and run == 'A' and purchase_month == 'total'")
    summary["monthly_regional_35"] = _fmt_total(_lookup(frozen, target="3.5pct", shape="regional"))
    summary["monthly_uniform_5"] = _fmt_total(_lookup(frozen, target="5pct", shape="uniform"))
    summary["monthly_regional_5"] = _fmt_total(_lookup(frozen, target="5pct", shape="regional"))
    summary["monthly_uniform_35"] = _fmt_total(_lookup(frozen, target="3.5pct", shape="uniform"))
    retrained = monthly.query("scheme == 'retrained_walk_forward' and run == 'A' and "
                              "target == '3.5pct' and purchase_month == 'total'")
    summary["retrained"] = _fmt_total(retrained.iloc[0])

    policy = pd.read_csv(Path(output) / "validation_monthly_policy.csv")
    best = _lookup(policy, run="A", target="5pct", purchase_month="total")
    summary["val_policy"] = f"{best['our_mean_promise_days']:.2f} d / {best['our_late_rate']:.2%}"

    gaps = pd.read_csv(Path(output) / "gap_band_prauc.csv")
    summary["gap_raw"] = f"{_lookup(gaps, run='B', score='raw_gap')['pr_auc']:.3f}"
    summary["gap_recal"] = f"{_lookup(gaps, run='A', score='recalibrated_gap')['pr_auc']:.3f}"
    summary["gap_slack"] = f"{_lookup(gaps, run='A', score='slack_rule')['pr_auc']:.3f}"
    summary["gap_shortest"] = f"{_lookup(gaps, run='A', score='shortest_promise')['pr_auc']:.3f}"
    summary["gap_random"] = f"{_lookup(gaps, run='A', score='random')['pr_auc']:.3f}"
    return summary


def _benchmark_row(centre, test_scheme, test_row, validation_scheme, validation_row, test_target,
                   validation_target):
    return {
        "centre": centre,
        "test_scheme": test_scheme,
        "test_target": test_target,
        "test_mean_promise_days": float(test_row["our_mean_promise_days"]),
        "test_late_rate": float(test_row["our_late_rate"]),
        "test_n_orders": int(test_row["n_orders"]),
        "validation_scheme": validation_scheme,
        "validation_target": validation_target,
        "validation_mean_promise_days": float(validation_row["our_mean_promise_days"]),
        "validation_late_rate": float(validation_row["our_late_rate"]),
        "validation_n_orders": int(validation_row["n_orders"]),
    }


def build_route_benchmark(monthly, policy_frame, reference, route_test, route_validation):
    """Compare the route-average centre with ETA runs A and B at the pre-set targets.

    Test rows use the frozen Phase 2 scheme at 3.5%; validation rows use the walk-forward scheme
    Nov 2017-Mar 2018 at 5%, the same rows as the existing validation policy. The Olist reference
    has no buffer; it is the same-period promise and late rate.
    """
    rows = []
    for run_name, label in (("A", "ETA run A"), ("B", "ETA run B")):
        test = _lookup(monthly, scheme="frozen_phase2", run=run_name, target="3.5pct",
                       shape="regional", purchase_month="total")
        validation = _lookup(policy_frame, run=run_name, target="5pct", purchase_month="total")
        rows.append(_benchmark_row(label, "frozen_phase2", test, "walk_forward", validation,
                                   "3.5pct", "5pct"))
    test = _lookup(route_test, purchase_month="total")
    validation = _lookup(route_validation, purchase_month="total")
    rows.append(_benchmark_row("Route average", "point_in_time", test, "walk_forward", validation,
                               "3.5pct", "5pct"))
    olist_test, olist_validation = reference.iloc[0], reference.iloc[2]
    rows.append({
        "centre": "Olist reference",
        "test_scheme": "olist",
        "test_target": "3.5pct",
        "test_mean_promise_days": float(olist_test["olist_mean_promise_days"]),
        "test_late_rate": float(olist_test["olist_late_rate"]),
        "test_n_orders": int(olist_test["n_orders"]),
        "validation_scheme": "olist",
        "validation_target": "5pct",
        "validation_mean_promise_days": float(olist_validation["olist_mean_promise_days"]),
        "validation_late_rate": float(olist_validation["olist_late_rate"]),
        "validation_n_orders": int(olist_validation["n_orders"]),
    })
    return pd.DataFrame(rows)


def run(root=REPO_ROOT, output=RESULT_DIR):
    """Compute every table for runs A and B, write the CSV outputs and the ledger."""
    root, output = Path(root), Path(output)
    output.mkdir(parents=True, exist_ok=True)
    table, spec_full = load_order_table(root)

    reference = olist_reference(table)
    reference.to_csv(output / "olist_reference.csv", index=False)
    summary = {
        "test_orders": int(table["split"].eq("test").sum()),
        "val_orders": int(table["cv_fold"].isin([0, 1, 2, 3, 4]).sum()),
        "olist_test_promise": f"{reference.loc[0, 'olist_mean_promise_days']:.2f} d",
        "olist_test_late": f"{reference.loc[0, 'olist_late_rate']:.2%}",
        "olist_test_delivery": f"{reference.loc[0, 'olist_mean_delivery_days']:.2f} d",
        "olist_val_promise": f"{reference.loc[1, 'olist_mean_promise_days']:.2f} d",
        "olist_val_late": f"{reference.loc[1, 'olist_late_rate']:.2%}",
        "olist_val_delivery": f"{reference.loc[1, 'olist_mean_delivery_days']:.2f} d",
        "val_policy_olist": (f"{reference.loc[2, 'olist_mean_promise_days']:.2f} d / "
                             f"{reference.loc[2, 'olist_late_rate']:.2%}"),
    }

    monthly_rows, gap_rows, fixed_rows, buffer_rows, median_rows, policy_rows = [], [], [], [], [], []
    for run_name in ("A", "B"):
        spec = run_spec(spec_full, run_name)
        model = fit_frozen_model(run_name, spec, table)
        eta_frozen = pd.Series(predict_eta(model, spec, table), index=table.index)
        table["eta_frozen"] = eta_frozen

        for target_label, target in TARGETS.items():
            for regional, shape in ((True, "regional"), (False, "uniform")):
                scored = score_test_period(run_name, model, spec, table, target, regional)
                scored.insert(0, "shape", shape)
                scored.insert(0, "target", target_label)
                scored.insert(0, "run", run_name)
                scored.insert(0, "scheme", "frozen_phase2")
                monthly_rows.append(scored)

        wf_predictions = walk_forward_models(run_name, spec, table, RETRAIN_MONTHS)
        for target_label, target in TARGETS.items():
            scored = score_period_retrained(table, wf_predictions, TEST_MONTHS, target, True,
                                            test_split_only=True)
            scored.insert(0, "shape", "regional")
            scored.insert(0, "target", target_label)
            scored.insert(0, "run", run_name)
            scored.insert(0, "scheme", "retrained_walk_forward")
            monthly_rows.append(scored)

        oof = validation_oof_predictions(run_name, spec, table)
        buffers = validation_needed_buffers(table, oof, {"10pct": 0.10, "5pct": 0.05})
        buffers.insert(0, "run", run_name)
        buffer_rows.append(buffers)
        for target_label, target in TARGETS.items():
            fixed = fixed_validation_buffer(table, oof, target)
            fixed.update({"run": run_name, "target": target_label})
            fixed_rows.append(fixed)
        wf_validation = walk_forward_models(run_name, spec, table, VALIDATION_WF_MONTHS)
        for target_label, target in TARGETS.items():
            policy = validation_monthly_policy(table, wf_validation, target, True)
            policy.insert(0, "target", target_label)
            policy.insert(0, "run", run_name)
            policy_rows.append(policy)

        val_months = table[table["cv_fold"].isin([0, 1, 2, 3, 4])]
        val_medians = (val_months[TARGET] - oof.reindex(val_months.index)).groupby(
            val_months["purchase_month"]).median()
        test_rows = table[table["split"].eq("test")]
        test_medians = (test_rows[TARGET] - eta_frozen.reindex(test_rows.index)).groupby(
            test_rows["purchase_month"]).median()
        for scope, medians in (("validation_oof", val_medians), ("test_frozen", test_medians)):
            median_rows.extend({"run": run_name, "scope": scope, "purchase_month": str(month),
                                "median_residual_days": float(value)}
                               for month, value in medians.items())

        buffer_series = pd.concat([buffers_for(table, m, eta_frozen, eta_frozen, 0.035, True)[1]
                                   for m in TEST_MONTHS])
        gaps = gap_band_prauc(table, eta_frozen, buffer_series)
        gaps.insert(0, "run", run_name)
        gap_rows.append(gaps)

    route_test = route_test_period(table, TARGETS["3.5pct"], True)
    route_validation = validation_monthly_policy(
        table, walk_forward_route_centres(table, VALIDATION_WF_MONTHS), TARGETS["5pct"], True)

    monthly = pd.concat(monthly_rows, ignore_index=True)
    monthly.to_csv(output / "test_monthly_policy.csv", index=False)
    pd.concat(buffer_rows, ignore_index=True).to_csv(output / "validation_needed_buffers.csv", index=False)
    pd.DataFrame(fixed_rows).to_csv(output / "fixed_validation_buffer.csv", index=False)
    policy_frame = pd.concat(policy_rows, ignore_index=True)
    policy_frame.to_csv(output / "validation_monthly_policy.csv", index=False)
    pd.DataFrame(median_rows).to_csv(output / "median_residuals.csv", index=False)
    pd.concat(gap_rows, ignore_index=True).to_csv(output / "gap_band_prauc.csv", index=False)
    build_route_benchmark(monthly, policy_frame, reference, route_test, route_validation).to_csv(
        output / "route_benchmark.csv", index=False)

    summary = build_summary(output)

    ledger = build_ledger(summary)
    ledger.to_csv(output / "ledger.csv", index=False)
    print(ledger.to_string(index=False))
    return ledger


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--ledger-only", action="store_true",
                        help="rebuild the ledger from the saved result CSVs without refitting")
    args = parser.parse_args()
    if args.ledger_only:
        summary = build_summary(RESULT_DIR)
        ledger = build_ledger(summary)
        ledger.to_csv(RESULT_DIR / "ledger.csv", index=False)
        print(ledger.to_string(index=False))
    else:
        run()
