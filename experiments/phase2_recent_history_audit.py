"""Post hoc, read-only comparison of recent histories and a handover update.

The held-out period has already been opened. Outputs are diagnostics, not a new
independent model-selection result. Run from the repository root with the local
project Python environment. Every history value is available strictly before
the purchase or handover being predicted.
"""

from __future__ import annotations

import copy
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from phase2_classification import feature_columns, load_primary_training, temporal_folds
from phase2_regression import make_regression_pipeline


OUT = ROOT / "experiments" / "phase2_recent_history_audit"
HORIZONS = (30, 90, 180)


def asof_mean(keys, values, available, query_keys, queries, days=None, strength=20):
    """Shrunk keyed mean of events available in [query-days, query)."""
    hist = pd.DataFrame({
        "key": np.asarray(keys, dtype=object),
        "value": np.asarray(values, dtype=float),
        "time": np.asarray(pd.to_datetime(available)),
    }).dropna(subset=["key", "value", "time"]).sort_values("time", kind="stable")
    query = pd.DataFrame({"key": np.asarray(query_keys, dtype=object),
                          "time": np.asarray(pd.to_datetime(queries))})
    result = np.full(len(query), np.nan)
    span = None if days is None else np.timedelta64(days, "D")

    def summarize(h, qt):
        times = h.time.to_numpy(dtype="datetime64[ns]")
        vals = h.value.to_numpy(dtype=float)
        end = np.searchsorted(times, qt, side="left")
        start = np.zeros(len(qt), dtype=int) if span is None else np.searchsorted(times, qt - span, side="left")
        sums = np.r_[0.0, np.cumsum(vals)]
        return end - start, sums[end] - sums[start]

    all_count, all_sum = summarize(hist, query.time.to_numpy(dtype="datetime64[ns]"))
    overall = np.divide(all_sum, all_count, out=np.full(len(query), np.nan), where=all_count > 0)
    groups = {key: group for key, group in hist.groupby("key", sort=False)}
    for key, part in query.groupby("key", sort=False):
        rows = part.index.to_numpy()
        if key not in groups:
            result[rows] = overall[rows]
            continue
        count, total = summarize(groups[key], part.time.to_numpy(dtype="datetime64[ns]"))
        prior = overall[rows]
        if strength == 0:
            result[rows] = np.divide(total, count, out=prior.copy(), where=count > 0)
        else:
            result[rows] = np.divide(total + strength * prior, count + strength,
                                     out=np.full(len(rows), np.nan), where=np.isfinite(prior))
    return result


def scores(actual, predicted):
    actual, predicted = np.asarray(actual), np.asarray(predicted)
    return {"orders": len(actual), "mae": mean_absolute_error(actual, predicted),
            "bias_pred_minus_actual": float(np.mean(predicted - actual))}


def handover_experiment(table):
    valid = table.handover_time.notna() & table.handover_time.ge(table.order_purchase_timestamp)
    valid &= table.handover_time.lt(table.outcome_available_at)
    stage = table[valid & table.split.isin(["train", "test"])].copy()
    stage["elapsed_to_handover"] = (stage.handover_time - stage.order_purchase_timestamp).dt.total_seconds() / 86400
    val_saved = pd.read_csv(ROOT / "results/phase2/regression/validation_predictions.csv")
    test_saved = pd.read_csv(ROOT / "results/phase2/regression/test_predictions.csv")
    saved = pd.concat([val_saved[["order_id", "predicted"]].rename(columns={"predicted": "checkout"}),
                       test_saved[["order_id", "predicted_single"]].rename(columns={"predicted_single": "checkout"})])
    stage = stage.merge(saved, on="order_id", how="inner", validate="one_to_one")
    rows = []
    stage["period"] = np.where(stage.split.eq("test"), "opened_test_post_hoc",
                               "validation_" + (stage.cv_fold.fillna(-1).astype(int) + 1).astype(str))
    for period, part in stage.groupby("period", sort=True):
        rows.append({"variant": "checkout_frozen", "period": period,
                     **scores(part.delivery_days, part.checkout)})
        for name in ("all", "180", "90", "30", "global90", "global30"):
            predicted = part.elapsed_to_handover + part[f"handover_shipping_{name}"].to_numpy()
            if not np.isfinite(predicted).all():
                raise ValueError(f"Missing handover forecast for {period}, {name}")
            rows.append({"variant": f"handover_shipping_{name}", "period": period,
                         **scores(part.delivery_days, predicted)})
            if period == "opened_test_post_hoc":
                months = part.order_purchase_timestamp.dt.to_period("M").astype(str)
                for month in sorted(months.unique()):
                    which = months.eq(month).to_numpy()
                    rows.append({"variant": f"handover_shipping_{name}", "period": f"test_{month}",
                                 **scores(part.delivery_days.to_numpy()[which], predicted[which])})
        if period == "opened_test_post_hoc":
            months = part.order_purchase_timestamp.dt.to_period("M").astype(str)
            for month in sorted(months.unique()):
                which = months.eq(month).to_numpy()
                rows.append({"variant": "checkout_frozen", "period": f"test_{month}",
                             **scores(part.delivery_days.to_numpy()[which], part.checkout.to_numpy()[which])})
    result = pd.DataFrame(rows)
    result.to_csv(OUT / "handover.csv", index=False)
    return result


def main():
    OUT.mkdir(exist_ok=True)
    table = pd.read_csv(ROOT / "data/phase2_order_table.csv",
                        parse_dates=["order_purchase_timestamp", "outcome_available_at"])
    raw = pd.read_csv(ROOT / "data/olist_orders_dataset.csv",
                      usecols=["order_id", "order_delivered_carrier_date"],
                      parse_dates=["order_delivered_carrier_date"])
    table = table.merge(raw, on="order_id", validate="one_to_one")
    table = table.rename(columns={"order_delivered_carrier_date": "handover_time"})
    table["route"] = table.seller_state.astype(str) + "->" + table.customer_state.astype(str)
    shipping = (table.outcome_available_at - table.handover_time).dt.total_seconds() / 86400
    shipping = shipping.where(shipping.ge(0) & table.handover_time.ge(table.order_purchase_timestamp))
    table["shipping_days"] = shipping
    reproduced_route = asof_mean(
        table.route, table.delivery_days, table.outcome_available_at,
        table.route, table.order_purchase_timestamp, strength=20)
    if not np.allclose(reproduced_route, table.route_typical_days, equal_nan=True, atol=1e-9):
        raise ValueError("All-history route calculation does not reproduce the saved feature")
    for horizon in HORIZONS:
        table[f"route_delivery{horizon}"] = asof_mean(
            table.route, table.delivery_days, table.outcome_available_at,
            table.route, table.order_purchase_timestamp, horizon)
    table["global_delivery90"] = asof_mean(
        np.zeros(len(table)), table.delivery_days, table.outcome_available_at,
        np.zeros(len(table)), table.order_purchase_timestamp, 90, strength=0)
    table["route_shipping90"] = asof_mean(
        table.route, table.shipping_days, table.outcome_available_at,
        table.route, table.order_purchase_timestamp, 90)
    for horizon in (None, *HORIZONS):
        name = "all" if horizon is None else str(horizon)
        table[f"handover_shipping_{name}"] = asof_mean(
            table.route, table.shipping_days, table.outcome_available_at,
            table.route, table.handover_time, horizon)
    for horizon in (30, 90):
        table[f"handover_shipping_global{horizon}"] = asof_mean(
            np.zeros(len(table)), table.shipping_days, table.outcome_available_at,
            np.zeros(len(table)), table.handover_time, horizon, strength=0)

    _, spec = load_primary_training(ROOT)
    original_route = table.route_typical_days.copy()
    original_slack = table.promise_slack.copy()
    checkout_results = []
    for label in ["frozen", "route30", "route90", "route180", "global90", "shipping90", "weighted90"]:
        if label.startswith("route") and label != "route_shipping90":
            table["route_typical_days"] = table[f"route_delivery{label[5:]}"]
            table["promise_slack"] = table.promised_days - table.route_typical_days
        else:
            table["route_typical_days"] = original_route
            table["promise_slack"] = original_slack
        result = checkout_experiment_one(table, spec, label)
        checkout_results.extend(result)
    pd.DataFrame(checkout_results).to_csv(OUT / "checkout.csv", index=False)
    handover = handover_experiment(table)
    checkout = pd.DataFrame(checkout_results)
    summary = checkout[checkout.period.str.startswith("validation")].groupby("variant", sort=False).mae.mean().rename("validation_mae")
    test_summary = checkout[checkout.period.eq("opened_test_post_hoc")].set_index("variant")
    print(summary.to_frame().join(test_summary[["mae", "bias_pred_minus_actual"]].rename(
        columns={"mae": "test_mae", "bias_pred_minus_actual": "test_bias"})).round(3).to_string())
    print("\nHandover results:\n", handover[~handover.period.str.startswith("test_")].pivot(
        index="variant", columns="period", values="mae").round(3).to_string())


def checkout_experiment_one(table, spec, variant):
    train = table[table.split.eq("train")].reset_index(drop=True)
    test = table[table.split.eq("test")].reset_index(drop=True)
    changed_spec = copy.deepcopy(spec)
    if variant in ("global90", "shipping90"):
        changed_spec["numeric_features"].append(
            "global_delivery90" if variant == "global90" else "route_shipping90")
    cols = sum(feature_columns(changed_spec), [])
    rows = []
    for w, (fit_idx, valid_idx) in enumerate(temporal_folds(train, spec)):
        fit, valid = train.iloc[fit_idx], train.iloc[valid_idx]
        start = valid.order_purchase_timestamp.min()
        pred = fit_predict(fit, valid, start, cols, changed_spec, variant)
        rows.append({"variant": variant, "period": f"validation_{w+1}",
                     "fit_orders": len(fit), **scores(valid.delivery_days, pred)})
    pred = fit_predict(train, test, pd.Timestamp(spec["primary_cutoff"]), cols, changed_spec, variant)
    rows.append({"variant": variant, "period": "opened_test_post_hoc",
                 "fit_orders": len(train), **scores(test.delivery_days, pred)})
    months = test.order_purchase_timestamp.dt.to_period("M").astype(str)
    for month in sorted(months.unique()):
        which = months.eq(month).to_numpy()
        rows.append({"variant": variant, "period": f"test_{month}", "fit_orders": len(train),
                     **scores(test.delivery_days.to_numpy()[which], pred[which])})
    return rows


def fit_predict(fit, predict, start, cols, spec, variant):
    model = make_regression_pipeline("ridge", {"alpha": 100.0}, spec)
    options = {}
    if variant == "weighted90":
        age = (start - fit.order_purchase_timestamp).dt.total_seconds() / 86400
        options["model__sample_weight"] = np.power(0.5, age / 90).to_numpy()
    with threadpool_limits(limits=1):
        model.fit(fit[cols], fit.delivery_days, **options)
        return model.predict(predict[cols])


if __name__ == "__main__":
    main()
