"""Exploratory post-hoc climate audit; seed 42; does not edit frozen artifacts.

Run: it5006-proj/bin/python experiments/phase2_marketplace_climate_audit.py
Outputs plain CSV tables in experiments/phase2_marketplace_climate_audit/.
"""

from pathlib import Path
import math
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score
from threadpoolctl import threadpool_limits
from xgboost import XGBClassifier

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from phase2_classification import (  # noqa: E402
    load_primary_training, temporal_folds, make_pipeline, feature_columns,
)

OUT = ROOT / "experiments/phase2_marketplace_climate_audit"
CLIMATE = ["climate_late_7d", "climate_late_30d", "climate_shock", "demand_pressure"]
SEED = 42


def window(event_ns, values, query_ns, days):
    hi = np.searchsorted(event_ns, query_ns, side="left")
    lo = np.searchsorted(event_ns, query_ns - np.int64(pd.Timedelta(days=days).value), side="left")
    sums = np.r_[0., np.cumsum(values, dtype=float)]
    counts = hi - lo
    totals = sums[hi] - sums[lo]
    return np.divide(totals, counts, out=np.full(len(query_ns), np.nan), where=counts > 0), counts, hi


def add_climate(table):
    raw = pd.read_csv(ROOT / "data/olist_orders_dataset.csv",
                      usecols=["order_id", "order_purchase_timestamp", "order_delivered_customer_date",
                               "order_estimated_delivery_date"],
                      parse_dates=["order_purchase_timestamp", "order_delivered_customer_date",
                                   "order_estimated_delivery_date"])
    assert raw.order_id.is_unique and table.order_id.is_unique
    complete = raw.dropna(subset=["order_delivered_customer_date", "order_estimated_delivery_date"]).copy()
    complete = complete.loc[complete.order_delivered_customer_date.ge(complete.order_purchase_timestamp)]
    complete = complete.sort_values("order_delivered_customer_date", kind="stable")
    event_ns = complete.order_delivered_customer_date.to_numpy(dtype="datetime64[ns]").astype("int64")
    labels = (complete.order_delivered_customer_date.dt.normalize()
              > complete.order_estimated_delivery_date.dt.normalize()).to_numpy(dtype=float)
    query_ns = table.order_purchase_timestamp.to_numpy(dtype="datetime64[ns]").astype("int64")
    rate7, count7, hi7 = window(event_ns, labels, query_ns, 7)
    rate30, count30, hi30 = window(event_ns, labels, query_ns, 30)
    assert np.array_equal(hi7, hi30)
    valid = hi7 > 0
    assert (event_ns[hi7[valid] - 1] < query_ns[valid]).all()
    # A completed delivery may enter only after its outcome became known.
    assert (event_ns[np.searchsorted(event_ns, query_ns, side="left") - 1][valid] < query_ns[valid]).all()
    purchases = raw.order_purchase_timestamp.dropna().to_numpy(dtype="datetime64[ns]").astype("int64")
    purchases.sort()
    zeros = np.zeros(len(purchases))
    _, volume7, p_hi = window(purchases, zeros, query_ns, 7)
    _, volume90, _ = window(purchases, zeros, query_ns, 90)
    valid = p_hi > 0
    assert (purchases[p_hi[valid] - 1] < query_ns[valid]).all()
    result = table.copy()
    result[CLIMATE[0]] = rate7
    result[CLIMATE[1]] = rate30
    result[CLIMATE[2]] = np.divide(rate7, rate30, out=np.full(len(result), np.nan), where=rate30 > 0)
    base = volume90 * (7 / 90)
    result[CLIMATE[3]] = np.divide(volume7, base, out=np.full(len(result), np.nan), where=base > 0)
    pd.DataFrame({"feature": CLIMATE, "missing_orders": [int(result[c].isna().sum()) for c in CLIMATE],
                  "source_events": [len(complete)] * 3 + [len(purchases)]}).to_csv(OUT / "feature_coverage.csv", index=False)
    return result


def pipeline(spec, family, with_climate):
    feature_set = {"numeric": list(spec["numeric_features"]) + (CLIMATE if with_climate else []),
                   "categorical": list(spec["categorical_features"])}
    model = make_pipeline("linear", {"C": .1, "class_weight": None}, spec,
                          feature_set=feature_set, log_features=spec["linear_log_candidates"])
    if family == "xgboost":
        model.set_params(model=XGBClassifier(n_estimators=200, max_depth=3, learning_rate=.05,
                         subsample=.8, colsample_bytree=.8, min_child_weight=10,
                         objective="binary:logistic", eval_metric="logloss", tree_method="hist",
                         n_jobs=4, random_state=SEED))
    numeric, categorical = feature_columns(spec, feature_set=feature_set)
    return model, numeric + categorical


def fit_predict(train, score, spec, family, with_climate):
    model, columns = pipeline(spec, family, with_climate)
    assert train.is_late.nunique() == 2
    with threadpool_limits(limits=4):
        model.fit(train[columns], train.is_late)
        probs = model.predict_proba(score[columns])[:, 1]
    return probs


def top_lift(y, p):
    count = math.ceil(.1 * len(y))
    picked = np.argsort(-p, kind="stable")[:count]
    return float(np.mean(y[picked]) / np.mean(y))


def metrics(frame):
    y = frame.is_late.to_numpy(dtype=int)
    p = frame.probability.to_numpy(dtype=float)
    rate = float(y.mean())
    return dict(orders=len(y), late=int(y.sum()), late_rate=rate,
                roc_auc=float(roc_auc_score(y, p)), pr_auc=float(average_precision_score(y, p)),
                pr_auc_over_rate=float(average_precision_score(y, p) / rate),
                top10_lift=top_lift(y, p), mean_predicted_risk=float(p.mean()))


def within_month_auc(frame):
    total = 0.
    pairs = 0
    for _, month in frame.groupby(frame.order_purchase_timestamp.dt.to_period("M")):
        y = month.is_late.to_numpy(dtype=int)
        if len(np.unique(y)) != 2:
            continue
        weight = int(y.sum() * (len(y) - y.sum()))
        total += weight * roc_auc_score(y, month.probability)
        pairs += weight
    return total / pairs if pairs else np.nan


def main():
    OUT.mkdir(exist_ok=True)
    train, spec = load_primary_training(ROOT)
    assert spec["random_state"] == SEED and spec["primary_cutoff"] == "2018-05-26"
    table = pd.read_csv(ROOT / "data/phase2_order_table.csv",
                        parse_dates=["order_purchase_timestamp", "outcome_available_at"])
    table = add_climate(table)
    train = table.loc[table.split.eq("train")].reset_index(drop=True)
    test = table.loc[table.split.eq("test")].reset_index(drop=True)
    assert len(train) == 75099 and len(test) == 19363
    folds = temporal_folds(train, spec)
    predictions = []
    for family in ("logistic", "xgboost"):
        for added in (False, True):
            name = f"{family}_{'climate' if added else 'base'}"
            for fold, (fit_idx, val_idx) in enumerate(folds, start=1):
                fit = train.iloc[fit_idx]
                val = train.iloc[val_idx].copy()
                start = val.order_purchase_timestamp.min()
                assert fit.outcome_available_at.lt(start).all()
                val["probability"] = fit_predict(fit, val, spec, family, added)
                val["model"] = name
                val["period"] = f"validation_{fold}"
                predictions.append(val[["order_id", "order_purchase_timestamp", "is_late", "probability", "model", "period"]])
            assert train.outcome_available_at.lt(pd.Timestamp(spec["primary_cutoff"])).all()
            scored = test.copy()
            scored["probability"] = fit_predict(train, scored, spec, family, added)
            scored["model"] = name
            scored["period"] = "test"
            predictions.append(scored[["order_id", "order_purchase_timestamp", "is_late", "probability", "model", "period"]])
    pred = pd.concat(predictions, ignore_index=True)
    pred.to_csv(OUT / "predictions.csv", index=False)
    rows = []
    for (model, period), part in pred.groupby(["model", "period"]):
        rows.append(dict(model=model, period=period, **metrics(part), within_month_auc=within_month_auc(part)))
    for model, part in pred.loc[pred.period.ne("test")].groupby("model"):
        rows.append(dict(model=model, period="validation_pooled", **metrics(part),
                         within_month_auc=within_month_auc(part)))
    scores = pd.DataFrame(rows)
    means = scores.loc[scores.period.str.fullmatch(r"validation_[1-5]")].groupby("model")[
        ["roc_auc", "pr_auc", "pr_auc_over_rate", "top10_lift", "mean_predicted_risk", "within_month_auc"]].mean().reset_index()
    means.insert(1, "period", "validation_mean")
    pd.concat([scores, means], ignore_index=True).to_csv(OUT / "scores.csv", index=False)
    diagnostic = []
    for model, all_test in pred.loc[pred.period.eq("test")].groupby("model"):
        jja = all_test.loc[all_test.order_purchase_timestamp.dt.month.isin((6, 7, 8))]
        diagnostic.append(dict(model=model, period="June-August pooled", **metrics(jja),
                               within_month_auc=within_month_auc(jja)))
        for month, part in jja.groupby(jja.order_purchase_timestamp.dt.strftime("%Y-%m")):
            diagnostic.append(dict(model=model, period=month, **metrics(part),
                                   within_month_auc=within_month_auc(part)))
    pd.DataFrame(diagnostic).to_csv(OUT / "test_months.csv", index=False)
    decomposition = []
    for model, all_test in pred.loc[pred.period.eq("test")].groupby("model"):
        jja = all_test.loc[all_test.order_purchase_timestamp.dt.month.isin((6, 7, 8))]
        pos, neg = int(jja.is_late.sum()), int(len(jja) - jja.is_late.sum())
        all_pairs = pos * neg
        within_pairs = sum(int(g.is_late.sum() * (len(g) - g.is_late.sum()))
                           for _, g in jja.groupby(jja.order_purchase_timestamp.dt.month))
        pooled = roc_auc_score(jja.is_late, jja.probability)
        inside = within_month_auc(jja)
        outside = (pooled * all_pairs - inside * within_pairs) / (all_pairs - within_pairs)
        decomposition.append(dict(model=model, all_pairs=all_pairs, within_pairs=within_pairs,
                                  within_pair_weight=within_pairs / all_pairs,
                                  pooled_auc=pooled, within_month_auc=inside,
                                  cross_month_auc=outside,
                                  within_contribution=(within_pairs / all_pairs) * inside,
                                  cross_contribution=(1 - within_pairs / all_pairs) * outside))
    pd.DataFrame(decomposition).to_csv(OUT / "regime_decomposition.csv", index=False)
    # The following is an impossible-at-cutoff information sensitivity, not a deployable fit.
    extra = table.loc[table.split.eq("unavailable_at_cutoff")].copy()
    assert len(extra) == 2008
    assert extra.order_purchase_timestamp.lt(pd.Timestamp(spec["primary_cutoff"])).all()
    assert extra.outcome_available_at.ge(pd.Timestamp(spec["primary_cutoff"])).all()
    augmented = pd.concat([train, extra], ignore_index=True)
    scored = test.copy()
    scored["probability"] = fit_predict(augmented, scored, spec, "logistic", False)
    original = pred.loc[pred.model.eq("logistic_base") & pred.period.eq("test")]
    original = original.set_index("order_id").loc[scored.order_id]
    assert np.array_equal(original.is_late.to_numpy(), scored.is_late.to_numpy())
    eligibility = [dict(training="eligible at boundary", training_orders=len(train), **metrics(original.reset_index())),
                   dict(training="plus 2008 future-known outcomes", training_orders=len(augmented), **metrics(scored))]
    pd.DataFrame(eligibility).to_csv(OUT / "eligibility_sensitivity.csv", index=False)
    print(f"Wrote post hoc climate tables to {OUT}")


if __name__ == "__main__":
    main()
