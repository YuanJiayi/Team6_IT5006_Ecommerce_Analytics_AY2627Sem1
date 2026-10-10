"""Promise engine of docs/phase2_final_spec.md: delivery-time forecast plus a rolling empirical buffer.

Run from the repository root:
    python phase2_promise.py

The forecast mu predicts `delivery_days`; the buffer turns it into a whole-day promise
    promise = max(1, ceil(mu + q * max(mu, 3)))
where q is, for each purchase day D, the level-L quantile of r = (need - mu) / max(mu, 3) over orders purchased in
[D-90, D-45) days and already delivered before D. Calibration orders that were also fitting orders use out-of-fold mu.
Candidates and the fixed level L are chosen on the five validation windows; the test period is scored once at that L.
"""

from __future__ import annotations

import argparse
import copy
import json
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from phase2_classification import feature_columns, load_primary_training, temporal_folds
from phase2_eta import add_stage_features
from phase2_regression import RouteBaseline, assert_fit_precedes, make_regression_pipeline, regression_metrics  # noqa: F401

ROOT = Path(__file__).resolve().parent
RESULT_DIR = ROOT / "results" / "phase2" / "promise"
TARGET = "delivery_days"
LEVELS = np.round(np.arange(0.8, 0.9951, 0.0025), 4)
CAL_FAR, CAL_NEAR, MIN_SCALE = 90, 45, 3.0
TARGET_ONTIME = 0.95
TIE_MARGIN_DAYS = 0.1
OOF_FOLDS = 3

CANDIDATES = [
    ("mean_baseline", "mean", {}),
    ("route_baseline", "route", {}),
    ("linear_baseline", "linear", {}),
    ("ridge_1", "ridge", {"alpha": 1.0}),
    ("ridge_10", "ridge", {"alpha": 10.0}),
    ("ridge_100", "ridge", {"alpha": 100.0}),
    ("tree_baseline", "tree", {"max_depth": None, "min_samples_leaf": 1}),
    ("tree_depth6", "tree", {"max_depth": 6, "min_samples_leaf": 50}),
    ("forest_leaf20", "forest", {"max_depth": None, "min_samples_leaf": 20}),
]
BASELINES = {"mean_baseline", "route_baseline"}
SIMPLICITY = ["linear_baseline", "ridge_1", "ridge_10", "ridge_100", "tree_depth6", "tree_baseline", "forest_leaf20"]
PROMISE_FIELDS = ["promised_days", "promise_slack"]
TRANSIT = "route_transit_90d"


# ---------------------------------------------------------------- data

def load_promise_table(root=ROOT):
    """Full order table (all splits) with route_transit_90d, calendar-day need/promise and day numbers.

    Returns (table, spec, train_rows) where train_rows are the positions of `split == "train"` rows in `table`.
    """
    root = Path(root)
    _, spec = load_primary_training(root)
    table = pd.read_csv(root / "data/phase2_order_table.csv",
                        parse_dates=["order_purchase_timestamp", "outcome_available_at"])
    orders = pd.read_csv(root / "data/olist_orders_dataset.csv",
                         usecols=["order_id", "order_approved_at", "order_delivered_carrier_date",
                                  "order_estimated_delivery_date"],
                         parse_dates=["order_approved_at", "order_delivered_carrier_date",
                                      "order_estimated_delivery_date"])
    full = add_stage_features(table, orders)
    estimated = orders.set_index("order_id")["order_estimated_delivery_date"].reindex(full["order_id"]).to_numpy()
    full = attach_calendar_days(full, pd.Series(estimated))
    train_rows = np.flatnonzero(full[spec["primary_split"]].eq("train").to_numpy())
    return full, spec, train_rows


def attach_calendar_days(full, estimated):
    """Add purchase_day, delivery_day (integer day numbers), need and olist_promise; check they reproduce is_late."""
    day = lambda s: pd.Series(pd.to_datetime(s)).dt.normalize().to_numpy().astype("datetime64[D]").astype(np.int64)
    full = full.copy()
    full["purchase_day"] = day(full["order_purchase_timestamp"])
    full["delivery_day"] = day(full["outcome_available_at"])
    full["need"] = full["delivery_day"] - full["purchase_day"]
    full["olist_promise"] = day(estimated) - full["purchase_day"]
    if not ((full["need"] > full["olist_promise"]).astype(int) == full["is_late"]).all():
        raise ValueError("need > calendar-day promise does not reproduce is_late")
    return full


def promise_spec(spec, variant="final"):
    """Spec feature set: minus the promise fields, plus route_transit_90d. Variants are input checks only."""
    changed = copy.deepcopy(spec)
    numeric = [c for c in spec["numeric_features"] if c not in PROMISE_FIELDS]
    if variant == "with_promise_fields":
        numeric = list(spec["numeric_features"])
    if variant != "without_transit":
        numeric = numeric + [TRANSIT]
    changed["numeric_features"] = numeric
    return changed


# ---------------------------------------------------------------- buffer

def buffer_promises(mu, need, purchase_day, delivery_day, target, levels=LEVELS):
    """Promise matrix (len(target), len(levels)) for the target rows.

    For each purchase day D of the target rows the calibration set is every row purchased in
    [D-90, D-45) and delivered before D; `mu` must be finite on those rows and on the target rows.
    """
    target = np.asarray(target)
    order = np.argsort(purchase_day, kind="stable")
    sorted_days = purchase_day[order]
    out = np.full((len(target), len(levels)), np.nan)
    days = purchase_day[target]
    for day in np.unique(days):
        lo = np.searchsorted(sorted_days, day - CAL_FAR, side="left")
        hi = np.searchsorted(sorted_days, day - CAL_NEAR, side="left")
        cal = order[lo:hi]
        cal = cal[delivery_day[cal] < day]
        if not len(cal):
            raise ValueError(f"Empty calibration set for purchase day {day}")
        if not np.isfinite(mu[cal]).all():
            raise ValueError("Calibration rows without a forecast")
        score = (need[cal] - mu[cal]) / np.maximum(mu[cal], MIN_SCALE)
        q = np.quantile(score, levels)
        rows = np.flatnonzero(days == day)
        m = mu[target[rows]]
        out[rows] = np.maximum(np.ceil(m[:, None] + q[None, :] * np.maximum(m, MIN_SCALE)[:, None]), 1)
    return out


def frontier(promise, need):
    """On-time rate and mean promise at each level."""
    return (promise >= need[:, None]).mean(axis=0), promise.mean(axis=0)


def interpolate(on_time, mean_promise, goal):
    """Mean promise at exactly `goal` on-time, interpolated along the level grid; NaN outside the observed range."""
    order = np.argsort(on_time, kind="stable")
    return float(np.interp(goal, on_time[order], mean_promise[order], left=np.nan, right=np.nan))


def fixed_level(mean_on_time, goal=TARGET_ONTIME):
    """Index of the smallest level whose mean validation on-time rate reaches `goal`."""
    ok = np.flatnonzero(np.asarray(mean_on_time) >= goal)
    if not len(ok):
        raise ValueError(f"No level reaches {goal:.0%} mean validation on-time")
    return int(ok[0])


# ---------------------------------------------------------------- forecasts

def forecast(full, fit, target, extra, family, params, spec):
    """mu for `target` rows and `extra` rows (calibration rows outside the target), all fitted on `fit` rows only.

    Extra rows that are fitting rows get out-of-fold mu from 3 time-ordered folds over the fitting rows.
    Returns an array over `full` positions, NaN elsewhere.
    """
    columns = sum(feature_columns(spec), [])
    X, y = full[columns], full[TARGET]
    mu = np.full(len(full), np.nan)
    in_fit = np.zeros(len(full), dtype=bool)
    in_fit[fit] = True
    outside = np.union1d(target, extra[~in_fit[extra]])
    with threadpool_limits(limits=1 if family in ("mean", "route", "linear", "ridge") else None):
        model = make_regression_pipeline(family, params, spec).fit(X.iloc[fit], y.iloc[fit])
        mu[outside] = model.predict(X.iloc[outside])
        ordered = fit[np.argsort(full["order_purchase_timestamp"].to_numpy()[fit], kind="stable")]
        for held in np.array_split(ordered, OOF_FOLDS):
            rows = np.intersect1d(held, extra)
            if not len(rows):
                continue
            rest = np.setdiff1d(ordered, held)
            fold_model = make_regression_pipeline(family, params, spec).fit(X.iloc[rest], y.iloc[rest])
            mu[rows] = fold_model.predict(X.iloc[rows])
    return mu


def calibration_universe(full, target):
    """Rows that can serve as calibration for some day of `target` (superset: delivery filter applied later)."""
    days = full["purchase_day"].to_numpy()
    d = days[target]
    ok = (days >= d.min() - CAL_FAR) & (days < d.max() - CAL_NEAR)
    ok &= full["delivery_day"].to_numpy() < d.max()
    return np.flatnonzero(ok)


def run_window(full, fit, target, family, params, spec):
    """Forecast and promise matrix for one scoring window. Returns (mu over target rows, promise matrix)."""
    extra = calibration_universe(full, target)
    mu = forecast(full, fit, target, extra, family, params, spec)
    promise = buffer_promises(mu, full["need"].to_numpy(), full["purchase_day"].to_numpy(),
                              full["delivery_day"].to_numpy(), target)
    return mu[target], promise


# ---------------------------------------------------------------- selection

def select_candidate(promise_at_95):
    """Spec rule on mean-over-windows promise days at 95%: lowest mean; within 0.1 day, the simplest wins."""
    scores = {n: v for n, v in promise_at_95.items() if n not in BASELINES}
    if set(scores) != set(SIMPLICITY):
        raise ValueError("Scores must cover exactly the selectable candidates")
    best = min(scores, key=scores.get)
    chosen = [n for n in SIMPLICITY if scores[n] - scores[best] < TIE_MARGIN_DAYS][0]
    if chosen == best:
        reason = f"{best} has the lowest mean promise at 95% on time ({scores[best]:.3f} days) and is the simplest within {TIE_MARGIN_DAYS} day"
    else:
        reason = (f"{best} has the lowest mean promise at 95% ({scores[best]:.3f}); {chosen} ({scores[chosen]:.3f}) is "
                  f"within {TIE_MARGIN_DAYS} day and simpler, so it is chosen")
    return chosen, reason


# ---------------------------------------------------------------- runs

def validation_run(full, spec, train_rows, folds, name, family, params):
    """Per-window metrics and frontiers for one candidate (validation windows only)."""
    metrics, fronts = [], []
    started = time.monotonic()
    train = full.iloc[train_rows].reset_index(drop=True)
    for window, (fit, valid) in enumerate(folds):
        assert_fit_precedes(train, fit, valid)
        fit_rows, valid_rows = train_rows[fit], train_rows[valid]
        mu, promise = run_window(full, fit_rows, valid_rows, family, params, spec)
        need = full["need"].to_numpy()[valid_rows]
        on_time, mean_promise = frontier(promise, need)
        metrics.append({"candidate": name, "window": window, "fit_orders": len(fit), "validation_orders": len(valid),
                        **regression_metrics(full[TARGET].to_numpy()[valid_rows], mu),
                        "promise_at_95": interpolate(on_time, mean_promise, TARGET_ONTIME)})
        fronts.append(pd.DataFrame({"candidate": name, "window": window, "level": LEVELS,
                                    "on_time": on_time, "mean_promise": mean_promise}))
    print(f"{name}: mean promise@95={np.nanmean([m['promise_at_95'] for m in metrics]):.3f}; "
          f"{time.monotonic() - started:.1f}s", flush=True)
    return metrics, pd.concat(fronts, ignore_index=True)


def mean_front(fronts, name):
    """Window-averaged on-time rate by level for one candidate."""
    f = fronts[fronts["candidate"].eq(name)]
    return f.groupby("level", sort=True)["on_time"].mean().to_numpy()


def test_run(full, spec, train_rows, name, family, params, level_index):
    """Single refit on all train rows; score test rows. Returns (order frame, full-grid test promise matrix)."""
    test_rows = np.flatnonzero(full[spec["primary_split"]].eq("test").to_numpy())
    mu, promise = run_window(full, train_rows, test_rows, family, params, spec)
    frame = full.iloc[test_rows][["order_id", "order_purchase_timestamp", "need", "olist_promise", "is_late"]].copy()
    frame["mu"] = mu
    frame["promise_at_L"] = promise[:, level_index]
    return frame.reset_index(drop=True), promise


def test_summary(frame, promise, level_index):
    need = frame["need"].to_numpy()
    p = promise[:, level_index]
    month = frame["order_purchase_timestamp"].dt.strftime("%Y-%m")
    per_month = pd.DataFrame({"month": month, "on_time": p >= need, "promise": p, "buffer": p - frame["mu"]}) \
        .groupby("month").agg(orders=("promise", "size"), on_time=("on_time", "mean"),
                              mean_promise=("promise", "mean"), mean_buffer=("buffer", "mean")).reset_index()
    return {"mean_promise": float(p.mean()), "median_promise": float(np.median(p)),
            "on_time": float((p >= need).mean()), "mean_buffer": float((p - frame["mu"]).mean()),
            "orders": int(len(p))}, per_month


def run(root=ROOT):
    started = time.monotonic()
    out = Path(root) / "results" / "phase2" / "promise"
    out.mkdir(parents=True, exist_ok=True)
    full, base_spec, train_rows = load_promise_table(root)
    spec = promise_spec(base_spec)
    train = full.iloc[train_rows].reset_index(drop=True)
    folds = temporal_folds(train, base_spec)
    print(f"orders {len(full)}; train {len(train_rows)}; windows {len(folds)}", flush=True)

    metrics, fronts = [], []
    for name, family, params in CANDIDATES:
        m, f = validation_run(full, spec, train_rows, folds, name, family, params)
        metrics += m
        fronts.append(f)
    fronts = pd.concat(fronts, ignore_index=True)
    metrics = pd.DataFrame(metrics)
    metrics.to_csv(out / "validation_by_window.csv", index=False)
    fronts.to_csv(out / "validation_frontiers.csv", index=False)
    summary = metrics.groupby("candidate", sort=False).agg(
        mean_mae=("mae", "mean"), mean_rmse=("rmse", "mean"), mean_r2=("r2", "mean"),
        mean_promise_at_95=("promise_at_95", "mean"))
    if summary["mean_promise_at_95"].isna().any():
        raise ValueError("A window never reaches 95% on time inside the level grid")
    summary.to_csv(out / "validation_summary.csv")

    chosen, reason = select_candidate(summary["mean_promise_at_95"].to_dict())
    algorithms = {n: (f, p) for n, f, p in CANDIDATES}
    fixed = {name: fixed_level(mean_front(fronts, name)) for name in (chosen, "route_baseline")}
    window_on_time = {n: fronts[fronts["candidate"].eq(n) & fronts["level"].eq(LEVELS[fixed[n]])]
                      .set_index("window")["on_time"].round(5).to_dict() for n in fixed}
    selection = {
        "selected": chosen, "reason": reason,
        "fixed_level": float(LEVELS[fixed[chosen]]),
        "mean_validation_on_time_at_fixed_level": float(mean_front(fronts, chosen)[fixed[chosen]]),
        "route_baseline_fixed_level": float(LEVELS[fixed["route_baseline"]]),
        "route_baseline_mean_validation_on_time": float(mean_front(fronts, "route_baseline")[fixed["route_baseline"]]),
        "validation_on_time_by_window": window_on_time,
        "level_grid": [float(LEVELS[0]), float(LEVELS[-1]), float(LEVELS[1] - LEVELS[0])],
    }
    (out / "selection.json").write_text(json.dumps(selection, indent=2))
    print(reason, f"| L={selection['fixed_level']}", flush=True)

    # Input checks: selected candidate only, validation only, reported not used to override selection.
    checks = []
    family, params = algorithms[chosen]
    for variant in ("with_promise_fields", "without_transit"):
        m, _ = validation_run(full, promise_spec(base_spec, variant), train_rows, folds, f"{chosen}:{variant}", family, params)
        checks += m
    checks = pd.DataFrame(checks)
    base = metrics[metrics["candidate"].eq(chosen)].assign(candidate=f"{chosen}:final")
    pd.concat([base, checks]).groupby("candidate", sort=False).agg(
        mean_mae=("mae", "mean"), mean_rmse=("rmse", "mean"), mean_r2=("r2", "mean"),
        mean_promise_at_95=("promise_at_95", "mean")).to_csv(out / "input_checks.csv")

    # Test, once, at the fixed levels.
    results, frontiers, months = {}, {}, {}
    for name in (chosen, "route_baseline"):
        fam, par = algorithms[name]
        frame, promise = test_run(full, spec, train_rows, name, fam, par, fixed[name])
        results[name], months[name] = test_summary(frame, promise, fixed[name])
        on_time, mean_promise = frontier(promise, frame["need"].to_numpy())
        frontiers[name] = (on_time, mean_promise)
        if name == chosen:
            frame[["order_id", "mu", "promise_at_L", "need"]].to_csv(out / "test_predictions.csv", index=False)
    olist_on_time = float(1 - frame["is_late"].mean())
    olist = {"mean_promise": float(frame["olist_promise"].mean()), "median_promise": float(frame["olist_promise"].median()),
             "on_time": olist_on_time}
    per_month_olist = frame.assign(month=frame["order_purchase_timestamp"].dt.strftime("%Y-%m")) \
        .groupby("month").agg(olist_on_time=("is_late", lambda s: 1 - s.mean()),
                              olist_mean_promise=("olist_promise", "mean")).reset_index()
    matched = {n: interpolate(*frontiers[n], olist_on_time) for n in frontiers}
    matched["olist_on_time"] = olist_on_time
    result_months = months[chosen].merge(months["route_baseline"], on="month", suffixes=("_engine", "_route")) \
        .merge(per_month_olist, on="month")
    result_months.to_csv(out / "test_per_month.csv", index=False)
    pd.DataFrame({"level": LEVELS,
                  "engine_on_time": frontiers[chosen][0], "engine_mean_promise": frontiers[chosen][1],
                  "route_on_time": frontiers["route_baseline"][0], "route_mean_promise": frontiers["route_baseline"][1],
                  "olist_on_time": olist_on_time, "olist_mean_promise": olist["mean_promise"]}) \
        .to_csv(out / "test_frontier.csv", index=False)

    val_gap = summary.loc["route_baseline", "mean_promise_at_95"] - summary.loc[chosen, "mean_promise_at_95"]
    test_test = {
        "selected": results[chosen], "route_baseline": results["route_baseline"], "olist": olist,
        "matched_reliability_mean_promise": {"selected": matched[chosen], "route_baseline": matched["route_baseline"],
                                             "olist": olist["mean_promise"], "olist_on_time": olist_on_time},
        "gate_1": {"selected_beats_route_on_validation_promise_at_95": bool(val_gap > 0),
                   "validation_gap_days": float(val_gap),
                   "test_matched_selected_shorter_than_olist": bool(matched[chosen] < olist["mean_promise"]),
                   "test_matched_gap_days": float(olist["mean_promise"] - matched[chosen])},
    }
    (out / "test_results.json").write_text(json.dumps(test_test, indent=2))
    print(json.dumps(test_test, indent=2))
    print(f"runtime {time.monotonic() - started:.0f}s", flush=True)



# ---------------------------------------------------------------- adaptive level (spec: Changes after fitting, 1)

# Extended after the first adaptive run selected the grid edge (0.05); see the spec change log.
GAMMAS = [0.0, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2]
ALPHA_TARGET = 0.05
LEVEL_RANGE = (0.50, 0.995)
BURN_IN_DAYS = 120


def adaptive_promises(mu, need, purchase_day, delivery_day, start, end, gamma, alpha0=ALPHA_TARGET):
    """Daily-adaptive buffer (adaptive conformal inference) over purchase days start..end inclusive.

    The level on day D is clip(1 - alpha_D); the quantile of the score is taken directly at that level over the same
    calibration set as `buffer_promises`. After day D, alpha_{D+1} = alpha_D + gamma * (0.05 - e_D), where e_D is the
    late share among orders whose promise date (purchase day + promise) was D - 1; alpha is unchanged when there are
    none. Lateness of those orders is known on day D. Returns (promise over all rows, NaN outside the range;
    trace frame with day, level, e, n_informing).
    """
    order = np.argsort(purchase_day, kind="stable")
    sorted_days = purchase_day[order]
    promise = np.full(len(mu), np.nan)
    due = defaultdict(list)  # promise date -> rows promised for that date
    alpha, trace = alpha0, []
    for day in range(int(start), int(end) + 1):
        level = float(np.clip(1 - alpha, *LEVEL_RANGE))
        rows = order[np.searchsorted(sorted_days, day, "left"):np.searchsorted(sorted_days, day, "right")]
        if len(rows):
            lo = np.searchsorted(sorted_days, day - CAL_FAR, side="left")
            hi = np.searchsorted(sorted_days, day - CAL_NEAR, side="left")
            cal = order[lo:hi]
            cal = cal[delivery_day[cal] < day]
            if not len(cal):
                raise ValueError(f"Empty calibration set for purchase day {day}")
            if not np.isfinite(mu[cal]).all() or not np.isfinite(mu[rows]).all():
                raise ValueError("Rows without a forecast")
            q = np.quantile((need[cal] - mu[cal]) / np.maximum(mu[cal], MIN_SCALE), level)
            promise[rows] = np.maximum(np.ceil(mu[rows] + q * np.maximum(mu[rows], MIN_SCALE)), 1)
            for date, r in zip((purchase_day[rows] + promise[rows]).astype(np.int64), rows):
                due[int(date)].append(r)
        informing = np.array(due.get(day - 1, []), dtype=np.int64)
        e = float((need[informing] > promise[informing]).mean()) if len(informing) else np.nan
        trace.append({"day": day, "level": level, "e": e, "n_informing": len(informing)})
        if len(informing):
            alpha += gamma * (ALPHA_TARGET - e)
    return promise, pd.DataFrame(trace)


def adaptive_window(full, fit, target, family, params, spec, gammas=GAMMAS):
    """Adaptive runs for one scoring window (burn-in 120 days before it). Returns {gamma: (promise at target, trace)}."""
    days = full["purchase_day"].to_numpy()
    start, end = int(days[target].min()) - BURN_IN_DAYS, int(days[target].max())
    # Every cohort order purchased from the earliest calibration day to the window end needs a forecast.
    extra = np.flatnonzero((days >= start - CAL_FAR) & (days <= end))
    mu = forecast(full, fit, target, extra, family, params, spec)
    need, delivery = full["need"].to_numpy(), full["delivery_day"].to_numpy()
    out = {}
    for gamma in gammas:
        promise, trace = adaptive_promises(mu, need, days, delivery, start, end, gamma)
        out[gamma] = (promise[target], trace)
    return mu[target], out


def run_adaptive(root=ROOT):
    started = time.monotonic()
    out = Path(root) / "results" / "phase2" / "promise"
    selection = json.loads((out / "selection.json").read_text())
    chosen = selection["selected"]
    full, base_spec, train_rows = load_promise_table(root)
    spec = promise_spec(base_spec)
    train = full.iloc[train_rows].reset_index(drop=True)
    folds = temporal_folds(train, base_spec)
    algorithms = {n: (f, p) for n, f, p in CANDIDATES}
    need_all = full["need"].to_numpy()
    models = {"engine": chosen, "route": "route_baseline"}

    rows, traces = [], {}
    for label, name in models.items():
        family, params = algorithms[name]
        for window, (fit, valid) in enumerate(folds):
            assert_fit_precedes(train, fit, valid)
            valid_rows = train_rows[valid]
            _, runs = adaptive_window(full, train_rows[fit], valid_rows, family, params, spec)
            for gamma, (promise, trace) in runs.items():
                need = need_all[valid_rows]
                rows.append({"model": label, "candidate": name, "gamma": gamma, "window": window,
                             "on_time": float((promise >= need).mean()), "mean_promise": float(promise.mean()),
                             "mean_level": float(trace.loc[trace["day"] >= full["purchase_day"].to_numpy()[valid_rows].min(), "level"].mean())})
                traces[(label, gamma, "validation", window)] = trace
        print(f"{label} validation done {time.monotonic() - started:.0f}s", flush=True)
    validation = pd.DataFrame(rows)
    validation.to_csv(out / "adaptive_validation.csv", index=False)
    mean = validation.groupby(["model", "gamma"]).agg(on_time=("on_time", "mean"), mean_promise=("mean_promise", "mean")).reset_index()
    chosen_gamma = {}
    for label in models:
        m = mean[mean["model"].eq(label) & (mean["on_time"] >= TARGET_ONTIME)]
        if m.empty:
            raise ValueError(f"No gamma reaches {TARGET_ONTIME:.0%} mean validation on-time for {label}")
        chosen_gamma[label] = float(m.loc[m["mean_promise"].idxmin(), "gamma"])
    (out / "adaptive_selection.json").write_text(json.dumps({
        "gamma_grid": GAMMAS, "selected_gamma": chosen_gamma,
        "mean_validation": {l: mean[mean["model"].eq(l)].set_index("gamma")[["on_time", "mean_promise"]].to_dict("index") for l in models}},
        indent=2))
    print("gamma", chosen_gamma, flush=True)

    # Test, once, with gamma fixed.
    test_rows = np.flatnonzero(full[base_spec["primary_split"]].eq("test").to_numpy())
    frame = full.iloc[test_rows][["order_id", "order_purchase_timestamp", "need", "olist_promise", "is_late"]].reset_index(drop=True)
    results, month_tables, test_traces = {}, [], []
    for label, name in models.items():
        family, params = algorithms[name]
        mu, runs = adaptive_window(full, train_rows, test_rows, family, params, spec, gammas=[chosen_gamma[label]])
        promise, trace = runs[chosen_gamma[label]]
        frame[f"mu_{label}"], frame[f"promise_adaptive_{label}"] = mu, promise
        on = promise >= frame["need"].to_numpy()
        results[label] = {"gamma": chosen_gamma[label], "mean_promise": float(promise.mean()),
                          "median_promise": float(np.median(promise)), "on_time": float(on.mean())}
        month = frame["order_purchase_timestamp"].dt.strftime("%Y-%m")
        day_level = trace.set_index("day")["level"]
        level = day_level.reindex(full["purchase_day"].to_numpy()[test_rows]).to_numpy()
        month_tables.append(pd.DataFrame({"month": month, "on_time": on, "mean_promise": promise, "level": level})
                            .groupby("month").agg(on_time=("on_time", "mean"), mean_promise=("mean_promise", "mean"),
                                                  mean_level=("level", "mean")).add_prefix(f"{label}_"))
        test_traces.append(trace.assign(model=label, split="test", window=0))
    results["olist"] = {"mean_promise": float(frame["olist_promise"].mean()),
                        "median_promise": float(frame["olist_promise"].median()), "on_time": float(1 - frame["is_late"].mean())}
    olist_month = frame.assign(month=frame["order_purchase_timestamp"].dt.strftime("%Y-%m")).groupby("month").agg(
        olist_on_time=("is_late", lambda s: 1 - s.mean()), olist_mean_promise=("olist_promise", "mean"), orders=("need", "size"))
    pd.concat(month_tables + [olist_month], axis=1).reset_index().to_csv(out / "adaptive_test_per_month.csv", index=False)
    val_traces = [t.assign(model=l, split="validation", window=w) for (l, g, s, w), t in traces.items() if g == chosen_gamma[l]]
    pd.concat(val_traces + test_traces).to_csv(out / "adaptive_level_trace.csv", index=False)
    frame.drop(columns=["order_purchase_timestamp", "olist_promise", "is_late"]).to_csv(out / "adaptive_test_predictions.csv", index=False)
    gate = {"engine_shorter_than_olist": results["engine"]["mean_promise"] < results["olist"]["mean_promise"],
            "engine_on_time_at_least_olist": results["engine"]["on_time"] >= results["olist"]["on_time"],
            "engine_shorter_than_route": results["engine"]["mean_promise"] < results["route"]["mean_promise"]}
    (out / "adaptive_test_results.json").write_text(json.dumps({**results, "gate": gate}, indent=2))
    print(json.dumps({**results, "gate": gate}, indent=2))
    print(f"adaptive runtime {time.monotonic() - started:.0f}s", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--adaptive-only", action="store_true", help="skip the fixed-level run (needs its selection.json)")
    if not parser.parse_args().adaptive_only:
        run()
    run_adaptive()
