"""Checkout ETA -> promised date: what the checkout regression is worth as a promise-setting tool.

Run from the repository root with ``python src/experiments/checkout_promise/run.py``.
The ETA is the ridge (alpha=100) checkout model, the selected Phase 2 setting, without promised_days and promise_slack, so the promise it
produces does not reuse Olist's promise. A promise is ceil(ETA + buffer), where the buffer is a quantile of
recent residuals (actual calendar days - ETA). Outputs go to data/outputs/experiments/checkout_promise/.

This does not change the selected Phase 2 models. Validation results use the chronological windows; every
test-period comparison is descriptive, made after the policy was fixed on validation.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from olist.classification import feature_columns, load_primary_training, temporal_folds  # noqa: E402
from olist.paths import EXPERIMENT_OUTPUTS, RAW_DIR, REPO_ROOT  # noqa: E402
from olist.regression import make_regression_pipeline  # noqa: E402

HERE = EXPERIMENT_OUTPUTS / "checkout_promise"
ALPHA = 100.0
PROMISE_FEATURES = ("promised_days", "promise_slack")
MIN_GROUP = 100           # monthly cohort orders a group needs for its own quantile
REGION = {**{s: "N" for s in "AC AP AM PA RO RR TO".split()},
          **{s: "NE" for s in "AL BA CE MA PB PE PI RN SE".split()},
          **{s: "CW" for s in "DF GO MT MS".split()}, **{s: "SE" for s in "ES MG RJ".split()},
          "SP": "SP", **{s: "S" for s in "PR RS SC".split()}}
VAL_MONTHS = pd.period_range("2017-11", "2018-03", freq="M")
TEST_MONTHS = pd.period_range("2018-05", "2018-08", freq="M")
RNG = np.random.RandomState(42)


# ---------------------------------------------------------------- data and predictions

def load_orders(spec):
    table = pd.read_csv(REPO_ROOT / "data/processed/phase2_order_table.csv",
                        parse_dates=["order_purchase_timestamp", "outcome_available_at"])
    raw = pd.read_csv(RAW_DIR / "olist_orders_dataset.csv",
                      usecols=["order_id", "order_estimated_delivery_date", "order_delivered_customer_date"],
                      parse_dates=["order_estimated_delivery_date", "order_delivered_customer_date"])
    reviews = pd.read_csv(RAW_DIR / "olist_order_reviews_dataset.csv",
                          usecols=["order_id", "review_score"]).groupby("order_id").review_score.mean()
    t = table.merge(raw, on="order_id", how="left")
    t["review"] = t.order_id.map(reviews)
    day = t.order_purchase_timestamp.dt.normalize()
    # Calendar days, so that "late" matches Olist's is_late label exactly.
    t["promise_cal"] = (t.order_estimated_delivery_date.dt.normalize() - day).dt.days
    t["actual_cal"] = (t.order_delivered_customer_date.dt.normalize() - day).dt.days
    if not ((t.actual_cal > t.promise_cal) == t.is_late.astype(bool)).all():
        raise ValueError("calendar-day lateness does not reproduce is_late")
    t["cregion"] = t.customer_state.map(REGION)
    t["sregion"] = t.seller_state.map(REGION)
    t["month"] = t.order_purchase_timestamp.dt.to_period("M")
    return t


def ridge(spec, columns):
    s = dict(spec)
    s["numeric_features"] = [c for c in spec["numeric_features"] if c in columns]
    s["categorical_features"] = [c for c in spec["categorical_features"] if c in columns]
    return make_regression_pipeline("ridge", {"alpha": ALPHA}, s)


def predict_all(t, spec):
    """Out-of-fold ETA for validation windows, a gap-period ETA, and a single-fit ETA for test."""
    train, _ = load_primary_training(REPO_ROOT)
    num, cat = feature_columns(spec)
    full = num + cat
    noprom = [c for c in full if c not in PROMISE_FEATURES]
    y = spec["regression_target"]
    folds = temporal_folds(train, spec)
    window = pd.Series(-1, index=train.index)
    oof = {k: pd.Series(np.nan, index=train.index) for k in ("eta_full", "eta")}
    for k, (fit, valid) in enumerate(folds):
        window.iloc[valid] = k
        for name, cols in (("eta_full", full), ("eta", noprom)):
            oof[name].iloc[valid] = ridge(spec, cols).fit(train.iloc[fit][cols], train.iloc[fit][y]).predict(
                train.iloc[valid][cols])
    val = train.assign(window=window, **oof)
    val = val[val.window.ge(0)]
    # Orders purchased after the last validation window and before the test cutoff: fitted only on outcomes
    # known when the gap starts, so the monthly cohorts that straddle it carry honest residuals.
    # Orders bought before the test cutoff but delivered after it are outside the primary training split, yet
    # many were delivered before a later month starts. They are the slow ones, so leaving them out of the
    # monthly cohorts would shrink those buffers (about 1,900 May orders for July's buffer).
    gap_start = val.order_purchase_timestamp.max()
    late_outcome = t[t.split.eq("unavailable_at_cutoff")]
    gap = pd.concat([train[train.order_purchase_timestamp > gap_start], late_outcome], ignore_index=True)
    known = train[train.outcome_available_at < gap_start]
    gap["eta"] = ridge(spec, noprom).fit(known[noprom], known[y]).predict(gap[noprom])
    test = t[t.split.eq("test")].copy()
    test["eta"] = ridge(spec, noprom).fit(train[noprom], train[y]).predict(test[noprom])
    test["eta_full"] = ridge(spec, full).fit(train[full], train[y]).predict(test[full])
    keep = ["order_id", "eta"]
    parts = [val[keep + ["eta_full", "window"]].assign(src="val"), gap[keep].assign(src="gap"),
             test[keep + ["eta_full"]].assign(src="test")]
    d = pd.concat(parts, ignore_index=True).merge(t, on="order_id")
    d["resid"] = d.actual_cal - d.eta
    return d, noprom


# ---------------------------------------------------------------- policy helpers

def cohort(d, month):
    """Orders purchased in month M-2 and already delivered when month M starts."""
    st = month.start_time
    return d[d.order_purchase_timestamp.ge(st - pd.DateOffset(months=2))
             & d.order_purchase_timestamp.lt(st - pd.DateOffset(months=1))
             & d.order_delivered_customer_date.lt(st)]


def region_key(coh, frame):
    return coh.cregion, frame.cregion


def no_group(coh, frame):
    return pd.Series("all", index=coh.index), pd.Series("all", index=frame.index)


def group_buffer(resid, kc, ke, level, stat="quantile"):
    """Per-group residual quantile (or median), falling back to the cohort-wide value for small groups."""
    size = resid.groupby(kc).size()
    by = resid.groupby(kc)
    q = (by.quantile(level) if stat == "quantile" else by.median())[size >= MIN_GROUP]
    overall = resid.quantile(level) if stat == "quantile" else resid.median()
    return ke.map(q).fillna(overall).to_numpy()


def monthly_promises(d, src, months, target, key=region_key, base="eta"):
    """Monthly recalibrated promises for the src orders purchased in months."""
    rows = []
    for m in months:
        e = d[d.src.eq(src) & d.month.eq(m)]
        coh = cohort(d, m)
        assert coh.order_delivered_customer_date.max() < m.start_time
        kc, ke = key(coh, e)
        buf = group_buffer(coh.actual_cal - coh[base], kc, ke, 1 - target)
        rows.append(e.assign(promise_new=np.ceil(e[base].to_numpy() + buf), buffer=buf, cohort_n=len(coh)))
    return pd.concat(rows)


def score(frame, promise="promise_new"):
    return {"mean_promise": frame[promise].mean(), "late_rate": (frame.actual_cal > frame[promise]).mean()}


def frontier(d, src, months, key=region_key, base="eta", targets=np.r_[0.005:0.30:0.005]):
    f = [(T, *score(monthly_promises(d, src, months, T, key, base)).values()) for T in targets]
    return pd.DataFrame(f, columns=["target", "mean_promise", "late_rate"]).sort_values("late_rate")


def length_at(f, late):
    """Mean promise needed for a late rate; NaN when the frontier never reaches it."""
    if late < f.late_rate.min() or late > f.late_rate.max():
        return np.nan
    return float(np.interp(late, f.late_rate, f.mean_promise))


def shift_frontier(frame, base, grid=np.arange(-25, 30, 0.25)):
    """One buffer for every order (tuned on the same data): the in-sample baseline of evidence 1."""
    a = frame.actual_cal.to_numpy()
    rows = [(b, np.mean(a > np.ceil(frame[base] + b)), np.mean(np.ceil(frame[base] + b))) for b in grid]
    return pd.DataFrame(rows, columns=["buffer", "late_rate", "mean_promise"]).sort_values("late_rate")


# ---------------------------------------------------------------- analyses

def accuracy_and_state(d):
    rows = []
    for src in ("val", "test"):
        f = d[d.src.eq(src)]
        early = f.promise_cal - f.actual_cal
        route = f.route_typical_days.fillna(d[d.src.eq("val")].route_typical_days.median())
        rows.append({"split": src, "orders": len(f), "late_rate": f.is_late.mean(),
                     "mean_promise": f.promise_cal.mean(), "mean_actual": f.actual_cal.mean(),
                     "median_days_early": early.median(), "share_7plus_days_early": (early >= 7).mean(),
                     "mae_promise_as_forecast": (f.promised_days - f.delivery_days).abs().mean(),
                     "mae_route_typical": (route - f.delivery_days).abs().mean(),
                     "mae_ridge_full": (f.eta_full - f.delivery_days).abs().mean(),
                     "mae_ridge_no_promise": (f.eta - f.delivery_days).abs().mean(),
                     "bias_no_promise": (f.eta - f.delivery_days).mean()})
    return pd.DataFrame(rows)


def reviews(t):
    r = t.dropna(subset=["review"]).copy()
    r["days_early"] = r.promise_cal - r.actual_cal
    bins = [-1000, -8, -4, -1, 0, 3, 7, 14, 21, 1000]
    labels = ["late >7", "late 4-7", "late 1-3", "on day", "early 1-3", "early 4-7", "early 8-14",
              "early 15-21", "early >21"]
    r["arrival"] = pd.cut(r.days_early, bins=bins, labels=labels)
    by_arrival = r.groupby("arrival", observed=True).agg(
        orders=("review", "size"), mean_score=("review", "mean"),
        bad_review_rate=("review", lambda s: (s <= 2).mean())).reset_index()
    on_time = r[r.days_early >= 0].assign(speed=lambda x: pd.cut(x.actual_cal, [0, 5, 10, 15, 25, 200]).astype(str))
    by_speed = on_time.pivot_table(index="speed", columns="arrival", values="review", aggfunc="mean",
                                   observed=True).reset_index()
    late = r.groupby(r.is_late.astype(bool)).review.agg(mean_score="mean", bad_review_rate=lambda s: (s <= 2).mean())
    return by_arrival, by_speed, late


def uniform_baseline(d):
    """Evidence 1: one buffer for all orders, at Olist's own late rate, pooled and per validation window."""
    val = d[d.src.eq("val")]
    f = shift_frontier(val, "eta")
    pooled = {"olist_late": val.is_late.mean(), "olist_promise": val.promise_cal.mean(),
              "uniform_buffer": float(np.interp(val.is_late.mean(), f.late_rate, f.buffer)),
              "uniform_promise": float(np.interp(val.is_late.mean(), f.late_rate, f.mean_promise))}
    rows = []
    for w, g in val.groupby("window"):
        fw = shift_frontier(g, "eta")
        rows.append({"window": int(w), "orders": len(g), "olist_late": g.is_late.mean(),
                     "olist_promise": g.promise_cal.mean(),
                     "uniform_promise": float(np.interp(g.is_late.mean(), fw.late_rate, fw.mean_promise))})
    windows = pd.DataFrame(rows)
    saved = windows.olist_promise - windows.uniform_promise
    pooled.update(saved_mean=saved.mean(), saved_se=saved.std(ddof=1) / np.sqrt(len(saved)))
    # Trade-off near Olist's operating point: late-rate change for a promise one day shorter.
    fl = f.sort_values("mean_promise")
    at = lambda x: float(np.interp(x, fl.mean_promise, fl.late_rate))
    pooled["late_pp_per_day_shorter"] = 100 * (at(pooled["olist_promise"] - 1) - at(pooled["olist_promise"]))
    return pooled, windows


def region_buffers(d):
    """Evidence 2: what Olist adds per customer region vs the buffer the ETA needs (validation)."""
    val = d[d.src.eq("val")]
    t = val.groupby("cregion").apply(lambda g: pd.Series({
        "orders": len(g), "eta_mean": g.eta.mean(), "olist_buffer": (g.promise_cal - g.eta).mean(),
        "needed_buffer_10pct_late": g.resid.quantile(0.90), "needed_buffer_5pct_late": g.resid.quantile(0.95),
        "olist_late_rate": g.is_late.mean(), "olist_mean_promise": g.promise_cal.mean(),
        "actual_p90": g.actual_cal.quantile(0.9)}), include_groups=False).reset_index()
    dq = pd.qcut(val.distance_km, 5, labels=False)
    by_dist = val.groupby(dq).agg(p=("promise_cal", "mean"), a=("actual_cal", lambda s: s.quantile(0.9)))
    route = val.seller_state + ">" + val.customer_state
    m = val.promise_cal.groupby(route).transform("mean")
    busy = val[route.map(route.value_counts()).ge(100)]
    br = busy.seller_state + ">" + busy.customer_state
    dm = lambda c: busy[c] - busy.groupby(br)[c].transform("mean")
    facts = {"corr_region_promise_vs_p90": t.olist_mean_promise.corr(t.actual_p90),
             "corr_distance_quintile_promise_vs_p90": by_dist.p.corr(by_dist.a),
             "route_share_of_promise_variance": float(((m - val.promise_cal.mean()) ** 2).sum()
                                                      / ((val.promise_cal - val.promise_cal.mean()) ** 2).sum()),
             "within_route_corr_olist": dm("promise_cal").corr(dm("actual_cal")),
             "within_route_corr_eta": dm("eta").corr(dm("actual_cal"))}
    return t, facts


def static_groupings(d):
    """Evidence 2: regional late-rate spread for buffers set once on validation, in and out of sample."""
    val = d[d.src.eq("val")].copy()
    dcut = val.distance_km.quantile([.2, .4, .6, .8]).to_numpy()
    ecut = val.eta.quantile([1 / 3, 2 / 3]).to_numpy()
    val["all"] = "all"
    val["dist_q"] = pd.Series(np.digitize(val.distance_km.fillna(val.distance_km.median()), dcut),
                              index=val.index).astype(str)
    val["route_region"] = val.sregion.fillna("?") + ">" + val.cregion
    val["region_x_eta"] = val.cregion + "_" + pd.Series(np.digitize(val.eta, ecut), index=val.index).astype(str)
    names = {"all": "Uniform buffer", "cregion": "Customer region", "dist_q": "Distance quintile",
             "route_region": "Seller region -> customer region", "region_x_eta": "Region x ETA tercile"}

    def promise(frame, ref, g, q):
        qs = ref.groupby(g).resid.quantile(q)[ref.groupby(g).size() >= 200]
        return np.ceil(frame.eta + frame[g].map(qs).fillna(ref.resid.quantile(q)))

    def solve(frame, ref, g, target):
        lo, hi = 0.3, 0.999
        for _ in range(40):
            q = (lo + hi) / 2
            lo, hi = (q, hi) if (frame.actual_cal > promise(frame, ref, g, q)).mean() > target else (lo, q)
        return q

    spread = lambda frame, N: (lambda lr: lr.max() - lr.min())((frame.actual_cal > N).groupby(frame.cregion).mean())
    calib, held = val[val.window <= 2], val[val.window >= 3]
    rows = [{"grouping": "Olist status quo", "groups": np.nan, "in_sample_spread": spread(val, val.promise_cal),
             "out_of_sample_spread": spread(held, held.promise_cal)}]
    for g, name in names.items():
        n_in = promise(val, val, g, solve(val, val, g, val.is_late.mean()))
        n_out = promise(held, calib, g, solve(calib, calib, g, calib.is_late.mean()))
        rows.append({"grouping": name, "groups": val[g].nunique(), "in_sample_spread": spread(val, n_in),
                     "out_of_sample_spread": spread(held, n_out)})
    return pd.DataFrame(rows)


def monthly_recalibration(d):
    """Evidence 3: test months, buffer fixed on validation vs recalibrated monthly (uniform and by region)."""
    val, test = d[d.src.eq("val")], d[d.src.eq("test")]
    f = shift_frontier(val, "eta")
    fixed_buffer = float(np.interp(0.05, f.late_rate, f.buffer))
    fixed = test.assign(promise_new=np.ceil(test.eta + fixed_buffer))
    policies = {"Olist": test.assign(promise_new=test.promise_cal),
                "Buffer fixed on validation (5% target)": fixed}
    for target in (0.05, 0.035):
        policies[f"Monthly, uniform buffer ({target:.1%} target)"] = monthly_promises(d, "test", TEST_MONTHS,
                                                                                    target, no_group)
        policies[f"Monthly, region buffer ({target:.1%} target)"] = monthly_promises(d, "test", TEST_MONTHS,
                                                                                   target)
    overall = pd.DataFrame([{"policy": k, **score(v)} for k, v in policies.items()])
    by_month = pd.concat([v.groupby("month").apply(lambda g: pd.Series({
        **score(g), "mean_buffer": g.buffer.mean() if "buffer" in g else np.nan,
        "cohort_orders": g.cohort_n.iloc[0] if "cohort_n" in g else np.nan}), include_groups=False)
        .reset_index().assign(policy=k) for k, v in policies.items()])
    by_month["month"] = by_month["month"].astype(str)
    return overall, by_month, fixed_buffer


def retrain_vs_recalibrate(d, t, spec, noprom, fixed_buffer, target=0.05):
    """Evidence 4: monthly retraining vs monthly buffer recalibration, test months."""
    y = spec["regression_target"]
    fit = lambda frame: ridge(spec, noprom).fit(frame[noprom], frame[y])
    known_at = lambda st: t[t.order_purchase_timestamp.lt(st) & t.outcome_available_at.lt(st)]
    single_recal = monthly_promises(d, "test", TEST_MONTHS, target, no_group)
    rows = []
    for m in TEST_MONTHS:
        st = m.start_time
        g = d[d.src.eq("test") & d.month.eq(m)]
        eta_rt = fit(known_at(st)).predict(g[noprom])
        # The retrained model's buffer needs residuals on orders it did not fit: refit before the cohort starts.
        coh = known_at(st)
        coh = coh[coh.order_purchase_timestamp.ge(st - pd.DateOffset(months=2))
                  & coh.order_purchase_timestamp.lt(st - pd.DateOffset(months=1))]
        old = fit(known_at(st - pd.DateOffset(months=2)))
        buf_rt = (coh.actual_cal - old.predict(coh[noprom])).quantile(1 - target)
        s = single_recal[single_recal.month.eq(m)]
        for name, N in {"Single model + fixed buffer": np.ceil(g.eta + fixed_buffer),
                        "Monthly retrain + fixed buffer": np.ceil(eta_rt + fixed_buffer),
                        "Single model + monthly recalibrated buffer": s.set_index("order_id").promise_new
                        .reindex(g.order_id).to_numpy(),
                        "Monthly retrain + monthly recalibration": np.ceil(eta_rt + buf_rt)}.items():
            N = np.asarray(N)
            rows.append({"month": str(m), "policy": name, "orders": len(g), "mean_promise": N.mean(),
                         "late_rate": np.mean(g.actual_cal.to_numpy() > N),
                         "retrained_bias": float(np.mean(eta_rt - g[y]))})
    by_month = pd.DataFrame(rows)
    overall = by_month.groupby("policy", sort=False).apply(lambda x: pd.Series({
        "mean_promise": np.average(x.mean_promise, weights=x.orders),
        "late_rate": np.average(x.late_rate, weights=x.orders)}), include_groups=False).reset_index()
    return overall, by_month


def gap_signal(d):
    """Evidence 5: Olist promise - ETA as a checkout late-risk ranking."""
    val = d[d.src.eq("val")].assign(gap=lambda x: x.promise_cal - x.eta)
    bins = pd.cut(val.gap, [-100, 4, 8, 12, 16, 100], labels=["<4", "4-8", "8-12", "12-16", ">16"])
    table = val.groupby(bins, observed=True).agg(
        share=("gap", lambda s: len(s) / len(val)), late_rate=("is_late", "mean"),
        mean_review=("review", "mean")).reset_index().rename(columns={"gap": "gap_days"})
    # Test: the same monthly region correction as the buffer, here as a median shift of the ETA.
    shifted = []
    for m in TEST_MONTHS:
        e = d[d.src.eq("test") & d.month.eq(m)]
        coh = cohort(d, m)
        shifted.append(e.assign(shift=group_buffer(coh.resid, coh.cregion, e.cregion, 0.5, stat="median")))
    test = pd.concat(shifted)
    rankings = []
    for split, frame in (("val", val), ("test", test)):
        frame = frame.assign(gap=frame.promise_cal - frame.eta)
        scores = {"Gap, raw ETA": frame.gap, "Slack rule": frame.promise_slack,
                  "Shortest promise first": frame.promise_cal}
        if split == "test":
            scores["Gap, ETA recalibrated by region"] = frame.promise_cal - (frame.eta + frame["shift"])
        k = int(0.1 * len(frame))
        for name, s in scores.items():
            top = s.nsmallest(k).index
            rankings.append({"split": split, "ranking": name, "pr_auc": average_precision_score(frame.is_late, -s),
                             "late_caught_top10": frame.loc[top, "is_late"].sum() / frame.is_late.sum(),
                             "base_rate": frame.is_late.mean()})
    return table, pd.DataFrame(rankings)


def drift_gauge(d):
    """Evidence 6: monthly median residual of the checkout ETA, with a bootstrap SE."""
    rows = []
    for m, g in d.groupby("month"):
        r = g.resid.to_numpy()
        se = np.std([np.median(RNG.choice(r, len(r))) for _ in range(300)])
        rows.append({"month": str(m), "orders": len(r), "median_residual": np.median(r), "bootstrap_se": se,
                     "residual_sd": r.std()})
    return pd.DataFrame(rows)


def handover_gauge(spec):
    """Evidence 6: checkout vs handover residuals on the handover cohort (both ridge alpha=100)."""
    from olist.eta import TARGET, chronological_folds, fit_predict, load_stage_table, stage_spec
    table, s, _ = load_stage_table(REPO_ROOT)
    s.setdefault("random_state", 42)
    train = table[table[s["primary_split"]].eq("train")].reset_index(drop=True)
    test = table[table[s["primary_split"]].eq("test")].reset_index(drop=True)
    folds = chronological_folds(train, s)
    checkout = stage_spec(s, "checkout")
    checkout["numeric_features"] = [c for c in checkout["numeric_features"] if c not in PROMISE_FEATURES]
    rows = []
    for name, sp in (("checkout", checkout), ("handover", stage_spec(s, "handover"))):
        oof = pd.Series(np.nan, index=train.index)
        for fit, valid in folds:
            oof.iloc[valid] = fit_predict("ridge", {"alpha": ALPHA}, sp, train.iloc[fit], train.iloc[valid])
        pred = fit_predict("ridge", {"alpha": ALPHA}, sp, train, test)
        for frame, p in ((train, oof), (test, pred)):
            rows.append(pd.DataFrame({"stage": name, "month": frame.order_purchase_timestamp.dt.to_period("M"),
                                      "resid": frame[TARGET] - p, "handover_days": frame.handover_days}).dropna())
    r = pd.concat(rows)
    out = r.groupby(["month", "stage"]).resid.median().unstack("stage").add_prefix("median_residual_")
    out["median_handover_days"] = r[r.stage.eq("checkout")].groupby("month").handover_days.median()
    out["orders"] = r[r.stage.eq("checkout")].groupby("month").size()
    out = out.reset_index()
    out["month"] = out["month"].astype(str)
    return out


def replace_vs_threshold(d, target=0.035):
    """Replace every promise vs only change orders whose new promise differs a lot from Olist's (test)."""
    o = monthly_promises(d, "test", TEST_MONTHS, target)
    ol, ours, a = o.promise_cal, o.promise_new, o.actual_cal
    diff = ours - ol
    rules = {"Olist": diff < -999, "Replace all": diff > -999,
             **{f"Ours only if it differs by >{k} days": diff.abs() > k for k in (3, 5, 7, 10)},
             "Ours only if ours is >4 days longer": diff > 4, "Ours only if ours is 1-4 days longer": diff.between(1, 4),
             "min(Olist, ours)": diff < 0, "max(Olist, ours)": diff > 0,
             "Lengthen all, shorten only if >=5 days shorter": (diff > 0) | (diff <= -5),
             "Lengthen all, shorten only if >=10 days shorter": (diff > 0) | (diff <= -10)}
    # A rule is only worse if replacing every promise, with the target moved, is shorter at the same late rate.
    curve = frontier(d, "test", TEST_MONTHS, targets=np.r_[0.005:0.15:0.0025])
    rows = []
    for name, use in rules.items():
        N = ol.where(~use, ours)
        late = (a > N).mean()
        rows.append({"policy": name, "mean_promise": N.mean(), "late_rate": late,
                     "orders_changed": (N != ol).mean(),
                     "replace_all_promise_at_same_late": length_at(curve, late)})
    bins = pd.cut(diff, [-100, -10, -5, -1, 0, 4, 7, 100],
                  labels=["<=-10", "-9..-5", "-4..-1", "0", "+1..+4", "+5..+7", ">+7"])
    by_diff = pd.DataFrame({"diff": bins, "olist_late": a > ol, "ours_late": a > ours}).groupby(
        "diff", observed=True).agg(orders=("olist_late", "size"), olist_late=("olist_late", "sum"),
                                   ours_late=("ours_late", "sum")).reset_index()
    by_diff["late_orders_avoided"] = by_diff.olist_late - by_diff.ours_late
    targets = pd.DataFrame([{"target": T, **score(monthly_promises(d, "test", TEST_MONTHS, T))}
                            for T in (0.035, 0.03, 0.025, 0.02)])
    return pd.DataFrame(rows), by_diff, targets


def groupings_monthly(d):
    """Which factor to group the buffer by, under the same monthly recalibration (validation and test)."""
    def cat(col):
        return lambda coh, e: (coh[col].astype(str), e[col].astype(str))

    def binned(col, n):
        def key(coh, e):
            cuts = coh[col].quantile(np.arange(1, n) / n).to_numpy()
            dig = lambda x: pd.Series(np.digitize(x[col].fillna(cuts[0]), cuts), index=x.index).astype(str)
            return dig(coh), dig(e)
        return key

    def cross(col, other):
        def key(coh, e):
            a, b = other(coh, e)
            return coh[col].astype(str) + "_" + a, e[col].astype(str) + "_" + b
        return key

    d = d.assign(route=d.sregion.fillna("?") + ">" + d.cregion)
    groupings = {"None (uniform)": no_group, "Customer region (6)": region_key,
                 "Customer state (27)": cat("customer_state"), "Seller region": cat("sregion"),
                 "Route region pair": cat("route"), "Same state y/n": cat("same_state"),
                 "Product category": cat("product_category"), "ETA quintile": binned("eta", 5),
                 "Distance quintile": binned("distance_km", 5), "Weight quintile": binned("total_weight_g", 5),
                 "Freight quintile": binned("total_freight", 5),
                 "Seller ship-days quintile": binned("seller_ship_days", 5),
                 "Customer region x ETA tercile": cross("cregion", binned("eta", 3)),
                 "Customer region x same state": cross("cregion", cat("same_state"))}
    targets = np.r_[0.001:0.12:0.004]
    rows = []
    for name, key in groupings.items():
        fv = frontier(d, "val", VAL_MONTHS, key, targets=targets)
        ft = frontier(d, "test", TEST_MONTHS, key, targets=targets)
        rows.append({"grouping": name, "val_promise_at_5pct": length_at(fv, 0.05),
                     "val_promise_at_8pct": length_at(fv, 0.08), "test_promise_at_3.5pct": length_at(ft, 0.035),
                     "test_promise_at_2.8pct": length_at(ft, 0.028)})
    return pd.DataFrame(rows)


def seasonal(d):
    """The monthly policy over the validation months, where the Black Friday peak sits."""
    val = d[d.src.eq("val") & d.month.isin(VAL_MONTHS)]
    olist_late = val.is_late.mean()
    pooled = {"olist_late": olist_late, "olist_promise": val.promise_cal.mean()}
    for name, key in (("uniform", no_group), ("region", region_key)):
        pooled[f"{name}_promise_at_olist_late"] = length_at(frontier(d, "val", VAL_MONTHS, key), olist_late)
    o = monthly_promises(d, "val", VAL_MONTHS, 0.10, no_group)
    by_month = pd.DataFrame([{
        "month": str(m), "buffer": g.buffer.iloc[0], **score(g), "olist_late_rate": g.is_late.mean(),
        "olist_mean_promise": g.promise_cal.mean(), "cohort_median_residual": cohort(d, m).resid.median(),
        "month_median_residual": g.resid.median()} for m, g in o.groupby("month")])
    return pooled, by_month


def benchmarks(d):
    """The same monthly region buffer applied to no-model centres, to separate the model from recalibration."""
    d = d.assign(base_route=d.route_typical_days.fillna(d.route_typical_days.median()),
                 base_olist=d.promise_cal.astype(float), base_const=0.0)
    lates = {"val": d[d.src.eq("val") & d.month.isin(VAL_MONTHS)].is_late.mean(), "test": d[d.src.eq("test")].is_late.mean()}
    rows = []
    for base, name in (("base_const", "No estimate (constant per region)"),
                       ("base_route", "Route historical average + buffer"),
                       ("base_olist", "Olist promise, recalibrated"), ("eta", "Our ETA + buffer")):
        fv = frontier(d, "val", VAL_MONTHS, base=base)
        ft = frontier(d, "test", TEST_MONTHS, base=base, targets=np.r_[0.005:0.60:0.005])
        rows.append({"centre": name, "val_promise_at_olist_late": length_at(fv, lates["val"]),
                     "val_promise_at_5pct": length_at(fv, 0.05),
                     "test_promise_at_olist_late": length_at(ft, lates["test"]),
                     "test_promise_at_2.8pct": length_at(ft, 0.028)})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- figures

INK, INK2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
OURS, OLIST = "#2a78d6", "#eb6834"


def figures(buffers, by_month):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    def style(ax):
        ax.set_facecolor(SURFACE)
        for side in ("top", "right", "left"):
            ax.spines[side].set_visible(False)
        ax.spines["bottom"].set_color(INK2)
        ax.tick_params(colors=INK2, length=0)
        ax.yaxis.grid(True, color=GRID, linewidth=0.8)
        ax.set_axisbelow(True)

    order = ["SP", "CW", "S", "SE", "NE", "N"]
    b = buffers.set_index("cregion").loc[order]
    fig, ax = plt.subplots(figsize=(8, 4.2), facecolor=SURFACE)
    x, w = np.arange(len(order)), 0.38
    ax.bar(x - w / 2 - 0.01, b.olist_buffer, w, color=OLIST, label="Buffer Olist adds (promise − ETA)")
    ax.bar(x + w / 2 + 0.01, b.needed_buffer_10pct_late, w, color=OURS, label="Buffer needed for ~10% late")
    for i, (o, n, late) in enumerate(zip(b.olist_buffer, b.needed_buffer_10pct_late, b.olist_late_rate)):
        ax.text(i - w / 2, o + 0.4, f"{o:.0f}", ha="center", color=INK2, fontsize=9)
        ax.text(i + w / 2, n + 0.4, f"{n:.0f}", ha="center", color=INK2, fontsize=9)
        ax.text(i, -3.2, f"Olist late {late:.0%}", ha="center", color=INK2, fontsize=8)
    ax.set_xticks(x, order, color=INK)
    ax.set_ylabel("Days added to the ETA", color=INK2)
    ax.set_ylim(0, 26)
    ax.set_title("Olist adds a near-constant buffer; the buffer each region needs differs",
                 loc="left", color=INK, fontsize=11)
    style(ax)
    ax.legend(frameon=False, loc="upper left", fontsize=9, labelcolor=INK)
    fig.tight_layout()
    fig.savefig(HERE / "fig_region_buffers.png", dpi=200)
    plt.close(fig)

    m = by_month[by_month.policy.isin(["Olist", "Monthly, region buffer (3.5% target)"])]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(9, 3.8), facecolor=SURFACE)
    for policy, colour, label in (("Olist", OLIST, "Olist"),
                                  ("Monthly, region buffer (3.5% target)", OURS, "ETA + monthly region buffer")):
        s = m[m.policy.eq(policy)]
        a1.plot(s.month, s.mean_promise, color=colour, linewidth=2, marker="o", markersize=6, label=label)
        a2.plot(s.month, 100 * s.late_rate, color=colour, linewidth=2, marker="o", markersize=6, label=label)
    a1.set_title("Mean promise (days)", loc="left", color=INK, fontsize=11)
    a2.set_title("Late rate (%)", loc="left", color=INK, fontsize=11)
    a1.set_ylim(0, None)
    a2.set_ylim(0, None)
    for ax in (a1, a2):
        style(ax)
    a1.legend(frameon=False, fontsize=9, labelcolor=INK, loc="lower left")
    fig.suptitle("Test period (2018): monthly buffer recalibration follows the drift", x=0.01, ha="left",
                 color=INK, fontsize=12)
    fig.tight_layout()
    fig.savefig(HERE / "fig_monthly_recalibration.png", dpi=200)
    plt.close(fig)


# ---------------------------------------------------------------- main

def main():
    HERE.mkdir(parents=True, exist_ok=True)
    spec = json.loads((REPO_ROOT / "data/processed/phase2_feature_spec.json").read_text())
    t = load_orders(spec)
    d, noprom = predict_all(t, spec)
    save = lambda frame, name: frame.to_csv(HERE / f"{name}.csv", index=False)
    summary = {}

    state = accuracy_and_state(d)
    save(state, "accuracy_and_state")
    by_arrival, by_speed, late_reviews = reviews(t)
    save(by_arrival, "reviews_by_arrival")
    save(by_speed, "reviews_by_speed_on_time")
    summary["bad_review_rate"] = {"late": late_reviews.loc[True, "bad_review_rate"],
                                  "on_time": late_reviews.loc[False, "bad_review_rate"]}
    print("accuracy and current state done", flush=True)

    pooled, windows = uniform_baseline(d)
    save(windows, "uniform_buffer_windows")
    summary["uniform_baseline"] = pooled
    bad_per_100 = pooled["late_pp_per_day_shorter"] * (summary["bad_review_rate"]["late"]
                                                       - summary["bad_review_rate"]["on_time"])
    summary["bad_reviews_per_100_orders_per_day_shorter"] = bad_per_100
    buffers, facts = region_buffers(d)
    save(buffers, "region_buffers")
    summary["olist_differentiation"] = facts
    save(static_groupings(d), "static_groupings")
    print("evidence 1-2 done", flush=True)

    overall, by_month, fixed_buffer = monthly_recalibration(d)
    save(overall, "monthly_recalibration")
    save(by_month, "monthly_recalibration_by_month")
    summary["fixed_validation_buffer_5pct"] = fixed_buffer
    rt_overall, rt_month = retrain_vs_recalibrate(d, t, spec, noprom, fixed_buffer)
    save(rt_overall, "retrain_vs_recalibrate")
    save(rt_month, "retrain_vs_recalibrate_by_month")
    print("evidence 3-4 done", flush=True)

    bins, rankings = gap_signal(d)
    save(bins, "gap_bins")
    save(rankings, "gap_rankings")
    save(drift_gauge(d), "drift_gauge")
    save(handover_gauge(spec), "drift_gauge_checkout_vs_handover")
    print("evidence 5-6 done", flush=True)

    rules, by_diff, targets = replace_vs_threshold(d)
    save(rules, "replace_vs_threshold")
    save(by_diff, "replace_by_difference")
    save(targets, "target_sweep")
    pooled_s, months_s = seasonal(d)
    save(months_s, "seasonal_validation_months")
    summary["seasonal_validation"] = pooled_s
    save(groupings_monthly(d), "grouping_factors")
    save(benchmarks(d), "benchmarks")
    print("policy comparisons done", flush=True)

    final = overall.set_index("policy")
    olist = final.loc["Olist"]
    curve = frontier(d, "test", TEST_MONTHS, targets=np.r_[0.005:0.15:0.0025])
    by_length = curve.sort_values("mean_promise")
    summary["headline_test"] = {"olist": olist.to_dict(),
                                "region_3.5pct": final.loc["Monthly, region buffer (3.5% target)"].to_dict(),
                                # Read off the target sweep after the fact: what the policy can reach.
                                "promise_at_olist_late_rate": length_at(curve, olist.late_rate),
                                "late_rate_at_olist_promise": float(np.interp(olist.mean_promise, by_length.mean_promise,
                                                                              by_length.late_rate)),
                                "promise_at_2.8pct_late": length_at(curve, 0.028)}
    (HERE / "summary.json").write_text(json.dumps(summary, indent=2, default=float))
    figures(buffers, by_month)
    print(json.dumps(summary["headline_test"], indent=2, default=float))


if __name__ == "__main__":
    main()
