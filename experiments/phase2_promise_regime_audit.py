"""Exploratory post-hoc promise-regime audit; seed 42; does not edit frozen artifacts.

Designed after the later test was scored, so no result here replaces the frozen checkout model.
Compares the frozen logistic model (C=0.1) with two recency features meant to track promise-policy shifts,
and a monthly refit of each during the test period.

Run: it5006-proj/bin/python experiments/phase2_promise_regime_audit.py
Outputs plain CSV tables in experiments/phase2_promise_regime_audit/.
"""

from pathlib import Path
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from phase2_classification import load_primary_training, temporal_folds, make_pipeline, feature_columns  # noqa: E402
from phase2_features import as_of_counts_and_sums  # noqa: E402

OUT = ROOT / "experiments/phase2_promise_regime_audit"
REGIME = ["promise_vs_recent", "recent_route_slack"]
SEED = 42


def windowed_shrunk_mean(keys, values, times, query_keys, query_times, days, strength):
    """Per-key mean of history in [query - days, query), shrunk toward the all-key mean of the same window."""
    shift = pd.Timedelta(days=days)
    query_times = pd.to_datetime(np.asarray(query_times))

    def window(k, qk):
        n_now, s_now = as_of_counts_and_sums(k, values, times, qk, query_times)
        n_old, s_old = as_of_counts_and_sums(k, values, times, qk, query_times - shift)
        return n_now - n_old, s_now - s_old

    n_key, s_key = window(keys, query_keys)
    n_all, s_all = window(np.zeros(len(values), dtype=int), np.zeros(len(query_times), dtype=int))
    overall = np.divide(s_all, n_all, out=np.full(len(n_all), np.nan), where=n_all > 0)
    return np.where(n_key > 0, (s_key + overall * strength) / (n_key + strength), overall)


def add_regime(table):
    orders = pd.read_csv(ROOT / "data/olist_orders_dataset.csv",
                         usecols=["order_id", "customer_id", "order_purchase_timestamp",
                                  "order_estimated_delivery_date"],
                         parse_dates=["order_purchase_timestamp", "order_estimated_delivery_date"])
    customers = pd.read_csv(ROOT / "data/olist_customers_dataset.csv", usecols=["customer_id", "customer_state"])
    orders = orders.merge(customers, on="customer_id", how="left", validate="one_to_one").dropna(
        subset=["order_estimated_delivery_date", "customer_state"])
    # Every purchase shows a promise at checkout, whatever its later status, so all orders are usable history.
    promise = (orders.order_estimated_delivery_date - orders.order_purchase_timestamp).dt.total_seconds() / 86400
    recent_promise = windowed_shrunk_mean(orders.customer_state, promise, orders.order_purchase_timestamp,
                                          table.customer_state, table.order_purchase_timestamp, 30, 20)
    # Delivered orders enter route history only once their delivery is known (outcome_available_at).
    route = table.seller_state.astype(str) + ">" + table.customer_state.astype(str)
    recent_route = windowed_shrunk_mean(route, table.delivery_days, table.outcome_available_at,
                                        route, table.order_purchase_timestamp, 60, 20)
    result = table.copy()
    result["promise_vs_recent"] = result.promised_days - recent_promise
    result["recent_route_slack"] = result.promised_days - recent_route
    return result


def fit_predict(fit, score, spec, with_regime):
    feature_set = {"numeric": list(spec["numeric_features"]) + (REGIME if with_regime else []),
                   "categorical": list(spec["categorical_features"])}
    model = make_pipeline("linear", {"C": .1, "class_weight": None}, spec,
                          feature_set=feature_set, log_features=spec["linear_log_candidates"])
    numeric, categorical = feature_columns(spec, feature_set=feature_set)
    assert fit.is_late.nunique() == 2
    with threadpool_limits(limits=4):
        model.fit(fit[numeric + categorical], fit.is_late)
        return model.predict_proba(score[numeric + categorical])[:, 1]


def top10_precision(y, p):
    picked = np.argsort(-p, kind="stable")[:int(np.ceil(.1 * len(y)))]
    return float(y[picked].mean())


def within_month_auc(frame):
    """ROC-AUC over late/on-time pairs from the same purchase month only."""
    total = pairs = 0.
    for _, month in frame.groupby(frame.order_purchase_timestamp.dt.to_period("M")):
        y = month.is_late.to_numpy(dtype=int)
        if len(np.unique(y)) == 2:
            weight = y.sum() * (len(y) - y.sum())
            total += weight * roc_auc_score(y, month.probability)
            pairs += weight
    return total / pairs


def scores(frame):
    y, p = frame.is_late.to_numpy(dtype=int), frame.probability.to_numpy(dtype=float)
    return dict(orders=len(y), late=int(y.sum()), late_rate=float(y.mean()), mean_predicted=float(p.mean()),
                pr_auc=float(average_precision_score(y, p)), roc_auc=float(roc_auc_score(y, p)),
                top10_precision=top10_precision(y, p))


def main():
    OUT.mkdir(exist_ok=True)
    _, spec = load_primary_training(ROOT)
    assert spec["random_state"] == SEED and spec["primary_cutoff"] == "2018-05-26"
    table = add_regime(pd.read_csv(ROOT / "data/phase2_order_table.csv",
                                   parse_dates=["order_purchase_timestamp", "outcome_available_at"]))
    train = table.loc[table.split.eq("train")].reset_index(drop=True)
    test = table.loc[table.split.eq("test")].reset_index(drop=True)
    assert len(train) == 75099 and len(test) == 19363
    cutoff = pd.Timestamp(spec["primary_cutoff"])
    # Monthly refits: each test block is scored by a model fitted on every order whose outcome was known
    # before the block started, including earlier test orders. The first block reuses the primary boundary.
    blocks = [(cutoff, pd.Timestamp("2018-07-01")), (pd.Timestamp("2018-07-01"), pd.Timestamp("2018-08-01")),
              (pd.Timestamp("2018-08-01"), pd.Timestamp("2018-09-01"))]
    assert sum(test.order_purchase_timestamp.between(a, b, inclusive="left").sum() for a, b in blocks) == len(test)
    keep = ["order_id", "order_purchase_timestamp", "is_late", "probability", "model", "setup", "period"]
    predictions = []
    for with_regime in (False, True):
        name = "regime" if with_regime else "frozen"
        for fold, (fit_idx, val_idx) in enumerate(temporal_folds(train, spec), start=1):
            fit, val = train.iloc[fit_idx], train.iloc[val_idx].copy()
            assert fit.outcome_available_at.lt(val.order_purchase_timestamp.min()).all()
            val["probability"] = fit_predict(fit, val, spec, with_regime)
            predictions.append(val.assign(model=name, setup="validation", period=f"validation_{fold}")[keep])
        scored = test.copy()
        scored["probability"] = fit_predict(train, scored, spec, with_regime)
        predictions.append(scored.assign(model=name, setup="single_fit", period="test")[keep])
        for start, end in blocks:
            fit = table.loc[table.order_purchase_timestamp.lt(start) & table.outcome_available_at.lt(start)]
            block = test.loc[test.order_purchase_timestamp.between(start, end, inclusive="left")].copy()
            block["probability"] = fit_predict(fit, block, spec, with_regime)
            predictions.append(block.assign(model=name, setup="monthly_refit", period="test")[keep])
    pred = pd.concat(predictions, ignore_index=True)
    # The frozen single fit must reproduce the reported test probabilities exactly.
    frozen = pd.read_csv(ROOT / "reports/phase2_evidence/test_predictions.csv")
    mine = pred.loc[pred.model.eq("frozen") & pred.setup.eq("single_fit")].set_index("order_id")
    gap = np.abs(mine.loc[frozen.order_id, "probability"].to_numpy() - frozen.probability.to_numpy()).max()
    print(f"max absolute difference from reported frozen test probabilities: {gap:.2e}")
    assert gap < 1e-9
    pred.to_csv(OUT / "predictions.csv", index=False)

    rows = []
    for (model, setup, period), part in pred.groupby(["model", "setup", "period"]):
        rows.append(dict(model=model, setup=setup, period=period, **scores(part),
                         within_month_roc_auc=within_month_auc(part) if period == "test" else np.nan))
    table_scores = pd.DataFrame(rows)
    means = (table_scores.loc[table_scores.setup.eq("validation")].groupby("model")[["pr_auc", "roc_auc"]]
             .mean().reset_index().assign(setup="validation", period="validation_mean"))
    pd.concat([table_scores, means], ignore_index=True).to_csv(OUT / "scores.csv", index=False)
    monthly = []
    for (model, setup), part in pred.loc[pred.period.eq("test")].groupby(["model", "setup"]):
        for month, m in part.groupby(part.order_purchase_timestamp.dt.strftime("%Y-%m")):
            if m.is_late.nunique() == 2:
                monthly.append(dict(model=model, setup=setup, month=month, **scores(m)))
    pd.DataFrame(monthly).to_csv(OUT / "test_months.csv", index=False)
    drift = (table.groupby([table.split, table.order_purchase_timestamp.dt.to_period("M").astype(str)])
             [["promised_days", "promise_slack", "promise_vs_recent", "recent_route_slack", "is_late"]]
             .median().rename(columns={"is_late": "median_is_late"}))
    drift["late_rate"] = table.groupby([table.split, table.order_purchase_timestamp.dt.to_period("M").astype(str)]
                                       ).is_late.mean()
    drift.drop(columns="median_is_late").reset_index().to_csv(OUT / "promise_drift.csv", index=False)
    print(f"Wrote post hoc promise-regime tables to {OUT}")


if __name__ == "__main__":
    main()
