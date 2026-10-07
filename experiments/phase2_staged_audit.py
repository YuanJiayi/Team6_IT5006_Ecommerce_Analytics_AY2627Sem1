"""Post-hoc staged late-delivery audit; leaves the frozen checkout selection untouched.

Run: it5006-proj/bin/python experiments/phase2_staged_audit.py
All exploratory test comparisons, feature removals, and permutations use seed 42.
The results are diagnostics, not a second untouched final evaluation.
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from phase2_classification import (  # noqa: E402
    feature_columns, load_primary_training, make_pipeline, temporal_folds,
)

OUT = ROOT / "experiments/phase2_staged_audit"
SEED = 42
N_BOOT = 1000
SHARES = (.05, .10)
STAGES = ("checkout", "approval", "handover")
ADDED = {
    "checkout": [],
    "approval": ["approval_delay_days"],
    "handover": ["approval_delay_known_at_handover", "handover_after_approval_days",
                 "days_left_at_handover"],
}
UNI = {
    "promised_days": -1, "same_state": -1, "distance_km": 1,
    "route_typical_days": 1, "seller_ship_days": 1,
    "handover_after_approval_days": 1, "days_left_at_handover": -1,
}


def load():
    train, spec = load_primary_training(ROOT)
    table = pd.read_csv(ROOT / "data/phase2_order_table.csv",
                        parse_dates=["order_purchase_timestamp", "outcome_available_at"])
    raw = pd.read_csv(ROOT / "data/olist_orders_dataset.csv",
                      usecols=["order_id", "order_purchase_timestamp", "order_approved_at",
                               "order_delivered_carrier_date", "order_delivered_customer_date",
                               "order_estimated_delivery_date"],
                      parse_dates=["order_purchase_timestamp", "order_approved_at",
                                   "order_delivered_carrier_date", "order_delivered_customer_date",
                                   "order_estimated_delivery_date"])
    raw = raw.drop(columns="order_purchase_timestamp")
    frame = table.merge(raw, on="order_id", validate="one_to_one")
    purchase = frame.order_purchase_timestamp
    approval = frame.order_approved_at
    handover = frame.order_delivered_carrier_date
    delivery = frame.order_delivered_customer_date
    promise = frame.order_estimated_delivery_date

    # A stage exists only if its event happened before the observed outcome.
    stage1 = approval.notna() & approval.ge(purchase) & approval.le(delivery)
    stage2 = handover.notna() & handover.ge(purchase) & handover.le(delivery)
    approval_at_handover = stage2 & approval.notna() & approval.ge(purchase) & approval.le(handover)
    frame["eligible_checkout"] = True
    frame["eligible_approval"] = stage1
    frame["eligible_handover"] = stage2
    frame["approval_delay_days"] = ((approval - purchase).dt.total_seconds() / 86400).where(stage1)
    frame["approval_delay_known_at_handover"] = (
        (approval - purchase).dt.total_seconds() / 86400).where(approval_at_handover)
    frame["handover_after_approval_days"] = (
        (handover - approval).dt.total_seconds() / 86400).where(approval_at_handover)
    frame["days_left_at_handover"] = (promise.dt.normalize() - handover.dt.normalize()).dt.days.where(stage2)
    frame["late_days"] = (delivery.dt.normalize() - promise.dt.normalize()).dt.days
    frame["late_more_than_3"] = frame.late_days.gt(3).astype(int)
    frame["late_more_than_7"] = frame.late_days.gt(7).astype(int)

    # The source has no creation timestamp for promise, payment, basket, profile,
    # or catalog fields. Their Stage 0 availability is an explicit assumption.
    assert frame.loc[frame.approval_delay_days.notna(), "order_approved_at"].le(
        frame.loc[frame.approval_delay_days.notna(), "order_delivered_customer_date"]).all()
    assert frame.loc[frame.approval_delay_known_at_handover.notna(), "order_approved_at"].le(
        frame.loc[frame.approval_delay_known_at_handover.notna(), "order_delivered_carrier_date"]).all()
    assert frame.loc[frame.handover_after_approval_days.notna(), "order_approved_at"].le(
        frame.loc[frame.handover_after_approval_days.notna(), "order_delivered_carrier_date"]).all()
    assert frame.loc[frame.days_left_at_handover.notna(), "order_delivered_carrier_date"].le(
        frame.loc[frame.days_left_at_handover.notna(), "order_delivered_customer_date"]).all()
    assert set(spec["numeric_features"] + spec["categorical_features"]).isdisjoint({
        "order_approved_at", "order_delivered_carrier_date", "order_delivered_customer_date",
        "outcome_available_at", "is_late", "late_days", "late_more_than_3", "late_more_than_7"})
    assert frame["order_id"].is_unique
    assert spec["primary_cutoff"] == "2018-05-26" and spec["random_state"] == SEED
    assert frame.loc[frame.split.eq("train"), "order_id"].reset_index(drop=True).equals(train.order_id)
    return frame, spec


def columns(spec, stage, remove=()):
    drop = set(remove)
    numeric = [c for c in [*spec["numeric_features"], *ADDED[stage]] if c not in drop]
    categorical = [c for c in spec["categorical_features"] if c not in drop]
    return {"numeric": numeric, "categorical": categorical}


def fit_model(train, spec, stage, target="is_late", remove=()):
    feature_set = columns(spec, stage, remove)
    numeric, categorical = feature_columns(spec, feature_set=feature_set)
    model = make_pipeline("linear", {"C": .1, "class_weight": None}, spec,
                          feature_set=feature_set, log_features=spec["linear_log_candidates"])
    with threadpool_limits(limits=1):
        model.fit(train[numeric + categorical], train[target])
    return model, numeric + categorical


def probability(model, frame, features):
    with threadpool_limits(limits=1):
        return model.predict_proba(frame[features])[:, 1]


def rank(y, score, share):
    y = np.asarray(y, dtype=int)
    score = np.asarray(score, dtype=float)
    n = max(1, math.ceil(share * len(y)))
    ids = np.argsort(-score, kind="stable")[:n]
    found = int(y[ids].sum())
    precision = found / n
    return {"reviewed": n, "late_found": found, "precision": precision,
            "recall": found / y.sum() if y.sum() else float("nan"),
            "lift": precision / y.mean() if y.mean() else float("nan")}


def metrics(y, score):
    y = np.asarray(y, dtype=int)
    score = np.asarray(score, dtype=float)
    if y.min() == y.max():
        return {"orders": len(y), "late": int(y.sum()), "late_rate": float(y.mean()),
                "pr_auc": float("nan"), "ap_over_rate": float("nan"), "roc_auc": float("nan")}
    ap = average_precision_score(y, score)
    result = {"orders": len(y), "late": int(y.sum()), "late_rate": float(y.mean()),
              "pr_auc": float(ap), "ap_over_rate": float(ap / y.mean()),
              "roc_auc": float(roc_auc_score(y, score)), "mean_predicted": float(score.mean())}
    for share in SHARES:
        suffix = str(round(share * 100))
        result.update({f"{key}_at_{suffix}": value for key, value in rank(y, score, share).items()})
    return result


def boot_ci(frame, *, cluster=None, n=N_BOOT, seed=SEED):
    """Two-sided percentile intervals, conditional on already-fitted scores."""
    rng = np.random.default_rng(seed)
    y = frame.is_late.to_numpy(dtype=int)
    score = frame.probability.to_numpy(dtype=float)
    if cluster is None:
        groups = None
    else:
        group_values = frame[cluster].to_numpy()
        groups = [np.flatnonzero(group_values == value) for value in pd.unique(group_values)]
    draws = []
    for _ in range(n):
        if groups is None:
            ids = rng.integers(0, len(frame), len(frame))
        else:
            ids = np.concatenate([groups[k] for k in rng.integers(0, len(groups), len(groups))])
        if np.unique(y[ids]).size < 2:
            continue
        draws.append(metrics(y[ids], score[ids]))
    keys = ["pr_auc", "ap_over_rate", "roc_auc", "precision_at_5", "recall_at_5",
            "lift_at_5", "precision_at_10", "recall_at_10", "lift_at_10"]
    return {name: tuple(np.quantile([row[name] for row in draws], [.025, .975])) for name in keys}


def fold_indices(train, spec):
    return temporal_folds(train, spec)


def score_stage(frame, spec, stage):
    train = frame.loc[frame.split.eq("train")].reset_index(drop=True)
    test = frame.loc[frame.split.eq("test")].copy()
    folds = fold_indices(train, spec)
    result = []
    fits = []
    for fold, (fit_all, valid_all) in enumerate(folds):
        fit = train.iloc[fit_all].loc[lambda d: d[f"eligible_{stage}"]]
        valid = train.iloc[valid_all].loc[lambda d: d[f"eligible_{stage}"]]
        assert fit.outcome_available_at.lt(valid.order_purchase_timestamp.min()).all()
        assert fit.order_purchase_timestamp.lt(valid.order_purchase_timestamp.min()).all()
        model, features = fit_model(fit, spec, stage)
        scored = valid[["order_id", "order_purchase_timestamp", "is_late",
                        "late_more_than_3", "late_more_than_7"]].copy()
        scored["probability"] = probability(model, valid, features)
        scored["fold"] = fold
        scored["stage"] = stage
        result.append(scored)
        fits.append({"stage": stage, "fit": f"validation_{fold + 1}",
                     "fit_orders": len(fit), "train_late_rate": float(fit.is_late.mean()),
                     "eval_orders": len(valid), "eval_late_rate": float(valid.is_late.mean()),
                     "mean_predicted": float(scored.probability.mean())})
    fit = train.loc[train[f"eligible_{stage}"]]
    assert fit.order_purchase_timestamp.lt(pd.Timestamp(spec["primary_cutoff"])).all()
    assert fit.outcome_available_at.lt(pd.Timestamp(spec["primary_cutoff"])).all()
    test = test.loc[test[f"eligible_{stage}"]].copy()
    model, features = fit_model(fit, spec, stage)
    scored = test[["order_id", "order_purchase_timestamp", "is_late",
                   "late_more_than_3", "late_more_than_7"]].copy()
    scored["probability"] = probability(model, test, features)
    scored["fold"] = -1
    scored["stage"] = stage
    fits.append({"stage": stage, "fit": "final", "fit_orders": len(fit),
                 "train_late_rate": float(fit.is_late.mean()), "eval_orders": len(test),
                 "eval_late_rate": float(test.is_late.mean()),
                 "mean_predicted": float(scored.probability.mean())})
    return pd.concat(result, ignore_index=True), scored.reset_index(drop=True), fits, model, features


def period_results(validation, test):
    rows = []
    for fold, group in validation.groupby("fold", sort=True):
        rows.append({"stage": group.stage.iloc[0], "period": f"validation_{fold + 1}",
                     **metrics(group.is_late, group.probability)})
    rows.append({"stage": test.stage.iloc[0], "period": "test",
                 **metrics(test.is_late, test.probability)})
    return rows


def within_month_auc(groups):
    numerator, denominator = 0.0, 0
    for group in groups:
        y = group.is_late.to_numpy(dtype=int)
        if y.min() == y.max():
            continue
        pairs = int(y.sum() * (len(y) - y.sum()))
        numerator += pairs * roc_auc_score(y, group.probability)
        denominator += pairs
    return numerator / denominator


def monthly(test):
    test = test.copy()
    test["month"] = test.order_purchase_timestamp.dt.to_period("M").astype(str)
    test["week"] = test.order_purchase_timestamp.dt.to_period("W").astype(str)
    rows = []
    groups = []
    for month in ("2018-06", "2018-07", "2018-08"):
        group = test.loc[test.month.eq(month)]
        groups.append(group)
        point = metrics(group.is_late, group.probability)
        ci = boot_ci(group)
        rows.append({"stage": test.stage.iloc[0], "month": month, **point,
                     **{f"{key}_{end}": value[i] for key, value in ci.items()
                        for i, end in enumerate(("low", "high"))}})
    rng = np.random.default_rng(SEED)
    draws = []
    for _ in range(N_BOOT):
        sampled = [g.iloc[rng.integers(0, len(g), len(g))] for g in groups]
        draws.append(within_month_auc(sampled))
    combined = {"stage": test.stage.iloc[0], "months": "June-August 2018",
                "pair_weighted_roc_auc": within_month_auc(groups),
                "low": float(np.quantile(draws, .025)), "high": float(np.quantile(draws, .975))}
    return rows, combined


def rolling(frame, spec, stage):
    rows = []
    for month in ("2018-06", "2018-07", "2018-08"):
        start = pd.Timestamp(month + "-01")
        end = start + pd.offsets.MonthBegin(1)
        fit = frame.loc[frame.order_purchase_timestamp.lt(start)
                        & frame.outcome_available_at.lt(start)
                        & frame[f"eligible_{stage}"]]
        valid = frame.loc[frame.split.eq("test")
                          & frame.order_purchase_timestamp.ge(start)
                          & frame.order_purchase_timestamp.lt(end)
                          & frame[f"eligible_{stage}"]]
        assert fit.outcome_available_at.lt(start).all()
        assert fit.order_purchase_timestamp.lt(start).all()
        model, features = fit_model(fit, spec, stage)
        score = probability(model, valid, features)
        rows.append({"stage": stage, "month": month, "fit_orders": len(fit),
                     "fit_late_rate": float(fit.is_late.mean()),
                     **metrics(valid.is_late, score)})
    return rows


def severity(frame, spec, stage, target):
    train = frame.loc[frame.split.eq("train")].reset_index(drop=True)
    folds = temporal_folds(train, spec)
    rows = []
    for fold, (fit_all, valid_all) in enumerate(folds):
        fit = train.iloc[fit_all].loc[lambda d: d[f"eligible_{stage}"]]
        valid = train.iloc[valid_all].loc[lambda d: d[f"eligible_{stage}"]]
        assert fit.outcome_available_at.lt(valid.order_purchase_timestamp.min()).all()
        model, features = fit_model(fit, spec, stage, target=target)
        score = probability(model, valid, features)
        rows.append({"stage": stage, "target": target, "period": f"validation_{fold + 1}",
                     **metrics(valid[target], score)})
    fit = train.loc[train[f"eligible_{stage}"]]
    test = frame.loc[frame.split.eq("test") & frame[f"eligible_{stage}"]]
    model, features = fit_model(fit, spec, stage, target=target)
    score = probability(model, test, features)
    rows.append({"stage": stage, "target": target, "period": "test",
                 **metrics(test[target], score)})
    return rows


def filled(series, train):
    return pd.to_numeric(series, errors="coerce").fillna(
        pd.to_numeric(train, errors="coerce").median()).to_numpy(dtype=float)


def score_simple(fit, evaluation, spec, stage):
    """Fixed directions and day-scale weights; no outcome-based rule tuning."""
    output = {}
    available = set(columns(spec, stage)["numeric"])
    for feature, direction in UNI.items():
        if feature in available:
            output[feature] = direction * filled(evaluation[feature], fit[feature])
    p = lambda name: filled(evaluation[name], fit[name])
    rule = (p("route_typical_days") + p("seller_ship_days") + p("distance_km") / 500
            - p("promised_days") - p("same_state"))
    if stage == "handover":
        rule = rule + p("handover_after_approval_days") - p("days_left_at_handover")
    output["fixed_rule"] = rule
    return output


def baselines(frame, spec, stage):
    train = frame.loc[frame.split.eq("train")].reset_index(drop=True)
    rows = []
    for fold, (fit_all, valid_all) in enumerate(temporal_folds(train, spec)):
        fit = train.iloc[fit_all].loc[lambda d: d[f"eligible_{stage}"]]
        valid = train.iloc[valid_all].loc[lambda d: d[f"eligible_{stage}"]]
        for name, score in score_simple(fit, valid, spec, stage).items():
            m = metrics(valid.is_late, score)
            rows.append({"stage": stage, "feature": name, "period": f"validation_{fold + 1}",
                         "orders": m["orders"], "roc_auc": m["roc_auc"],
                         "ap_over_rate": m["ap_over_rate"]})
    fit = train.loc[train[f"eligible_{stage}"]]
    test = frame.loc[frame.split.eq("test") & frame[f"eligible_{stage}"]]
    for name, score in score_simple(fit, test, spec, stage).items():
        m = metrics(test.is_late, score)
        rows.append({"stage": stage, "feature": name, "period": "test",
                     "orders": m["orders"], "roc_auc": m["roc_auc"],
                     "ap_over_rate": m["ap_over_rate"]})
    return rows


def availability(spec):
    source = {
        "promised_days": ("orders.estimated_delivery_date + purchase_timestamp", "purchase (assumed)", True),
        "promise_slack": ("promised_days + earlier delivered route history", "purchase (assumed promise)", True),
        "route_typical_days": ("earlier delivered orders + customer/seller states", "purchase (as-of history)", True),
        "seller_ship_days": ("earlier carrier handovers + item seller_id", "purchase (as-of history)", True),
        "distance_km": ("customer/seller ZIP prefixes + geolocation centroids", "purchase (assumed profiles/map)", True),
        "same_state": ("customers.customer_state + sellers.seller_state", "purchase (assumed profiles)", True),
        "purchase_hour": ("orders.order_purchase_timestamp", "purchase", False),
        "n_items": ("order_items.order_item_id", "purchase (assumed basket)", True),
        "n_products": ("order_items.product_id", "purchase (assumed basket)", True),
        "n_sellers": ("order_items.seller_id", "purchase (assumed basket)", True),
        "total_price": ("order_items.price", "purchase (assumed basket)", True),
        "total_freight": ("order_items.freight_value", "purchase (assumed quote)", True),
        "total_weight_g": ("products.product_weight_g + order_items", "purchase (assumed catalog)", True),
        "total_volume_cm3": ("products dimensions + order_items", "purchase (assumed catalog)", True),
        "max_installments": ("payments.payment_installments (all rows)", "purchase (assumed submitted payments)", True),
        "customer_state": ("customers.customer_state", "purchase (assumed address)", True),
        "seller_state": ("sellers.seller_state + order_items", "purchase (assumed profile)", True),
        "product_category": ("products.product_category_name + order_items", "purchase (assumed catalog)", True),
        "payment_type": ("payments.payment_type (first sequence)", "purchase (assumed submitted payments)", True),
        "purchase_dayofweek": ("orders.order_purchase_timestamp", "purchase", False),
        "approval_delay_days": ("orders.order_approved_at - purchase_timestamp", "approval", False),
        "approval_delay_known_at_handover": ("orders.order_approved_at - purchase_timestamp", "handover only if approval <= handover", False),
        "handover_after_approval_days": ("orders.carrier_date - approved_at", "handover only if approval <= handover", False),
        "days_left_at_handover": ("orders.estimated_delivery_date - carrier_date", "handover (assumed promise)", True),
    }
    rows = []
    for stage in STAGES:
        for name in [*columns(spec, stage)["numeric"], *columns(spec, stage)["categorical"]]:
            raw, timestamp, uncertain = source[name]
            rows.append({"stage": stage, "input": name, "source_column": raw,
                         "depends_on_timestamp": timestamp, "availability_uncertain": uncertain})
    return pd.DataFrame(rows)


def removals(frame, spec, stage, original_metrics):
    train = frame.loc[frame.split.eq("train") & frame[f"eligible_{stage}"]]
    test = frame.loc[frame.split.eq("test") & frame[f"eligible_{stage}"]]
    flags = availability(spec)
    flagged = flags.loc[flags.stage.eq(stage) & flags.availability_uncertain, "input"].tolist()
    groups = {
        "promise_bundle": ["promised_days", "promise_slack", "days_left_at_handover"],
        "payment_bundle": ["payment_type", "max_installments"],
    }
    candidates = {name: [name] for name in flagged}
    candidates.update(groups)
    rows = []
    for name, omitted in candidates.items():
        omitted = [f for f in omitted if f in [*columns(spec, stage)["numeric"],
                                                *columns(spec, stage)["categorical"]]]
        if not omitted:
            continue
        model, features = fit_model(train, spec, stage, remove=omitted)
        point = metrics(test.is_late, probability(model, test, features))
        rows.append({"stage": stage, "removed": name, "columns": ",".join(omitted),
                     "test_roc_auc": point["roc_auc"], "test_ap_over_rate": point["ap_over_rate"],
                     "test_top10_lift": point["lift_at_10"],
                     "delta_roc_vs_full": point["roc_auc"] - original_metrics["roc_auc"],
                     "delta_ap_ratio_vs_full": point["ap_over_rate"] - original_metrics["ap_over_rate"],
                     "delta_top10_lift_vs_full": point["lift_at_10"] - original_metrics["lift_at_10"]})
    return rows


def stage2_sanity(frame, spec, model, features, original):
    test = frame.loc[frame.split.eq("test") & frame.eligible_handover].copy()
    assert test.order_id.reset_index(drop=True).equals(original.order_id.reset_index(drop=True))
    after_promise = test.order_delivered_carrier_date.dt.normalize().gt(
        test.order_estimated_delivery_date.dt.normalize())
    assert test.loc[after_promise, "is_late"].eq(1).all()
    rng = np.random.default_rng(SEED)
    base = metrics(test.is_late, original.probability)
    groupings = {
        "promise": ["promised_days", "promise_slack", "days_left_at_handover"],
        "route_and_seller_history": ["route_typical_days", "seller_ship_days"],
        "geography": ["customer_state", "seller_state", "same_state", "distance_km"],
        "basket": ["n_items", "n_products", "n_sellers", "total_price", "total_freight"],
        "catalog": ["product_category", "total_weight_g", "total_volume_cm3"],
        "payment": ["payment_type", "max_installments"],
        "purchase_time": ["purchase_hour", "purchase_dayofweek"],
        "approval_delay": ["approval_delay_known_at_handover"],
        "handover_delay": ["handover_after_approval_days"],
    }
    importance = []
    for name, cols in groupings.items():
        changes = []
        for _ in range(5):
            shuffled = test[features].copy()
            perm = rng.permutation(len(test))
            for col in cols:
                shuffled[col] = shuffled[col].to_numpy()[perm]
            score = probability(model, shuffled, features)
            changes.append(base["pr_auc"] - average_precision_score(test.is_late, score))
        importance.append({"group": name, "columns": ",".join(cols),
                           "mean_ap_decrease": float(np.mean(changes)),
                           "min_ap_decrease": float(np.min(changes)),
                           "max_ap_decrease": float(np.max(changes))})
    perturbations = []
    month = test.order_purchase_timestamp.dt.to_period("M").astype(str)
    for name, cols in [("handover_delay_only", ["handover_after_approval_days"]),
                       ("handover_delay_and_days_left", ["handover_after_approval_days",
                                                        "days_left_at_handover"])]:
        shuffled = test[features].copy()
        for _, ids in month.groupby(month).groups.items():
            ids = np.asarray(list(ids))
            # Test retains the merged table's index; map to row positions.
            positions = test.index.get_indexer(ids)
            perm = rng.permutation(positions)
            for col in cols:
                values = shuffled[col].to_numpy().copy()
                values[positions] = values[perm]
                shuffled[col] = values
        point = metrics(test.is_late, probability(model, shuffled, features))
        perturbations.append({"shuffle": name, "seed": SEED, "roc_auc": point["roc_auc"],
                              "pr_auc": point["pr_auc"], "ap_over_rate": point["ap_over_rate"],
                              "top10_lift": point["lift_at_10"]})
    pre_promise = test.loc[~after_promise]
    pre_promise_scores = original.loc[original.order_id.isin(pre_promise.order_id),
                                      ["order_id", "probability"]]
    pre_promise = pre_promise.merge(pre_promise_scores, on="order_id", validate="one_to_one")
    return importance, perturbations, {
        "test_stage2_orders": len(test),
        "handover_after_promised_date": int(after_promise.sum()),
        "fraction": float(after_promise.mean()),
        "all_late_after_promise": bool(test.loc[after_promise, "is_late"].eq(1).all()),
        "before_or_on_promise_metrics": metrics(pre_promise.is_late, pre_promise.probability),
    }


def coverage(frame):
    rows = []
    for population, group in frame.groupby("split", sort=False):
        p, a, h, d = (group.order_purchase_timestamp, group.order_approved_at,
                      group.order_delivered_carrier_date, group.order_delivered_customer_date)
        rows.append({"population": population, "orders": len(group),
                     "stage1_invalid_or_missing_approval": int((~group.eligible_approval).sum()),
                     "stage1_approval_after_delivery": int(a.gt(d).sum()),
                     "stage2_invalid_or_missing_handover": int((~group.eligible_handover).sum()),
                     "stage2_handover_before_purchase": int(h.lt(p).sum()),
                     "stage2_handover_after_delivery": int(h.gt(d).sum()),
                     "stage2_approval_not_yet_known": int((group.eligible_handover
                                                          & group.approval_delay_known_at_handover.isna()).sum()),
                     "stage2_days_left_missing_on_eligible": int((group.eligible_handover
                                                                & group.days_left_at_handover.isna()).sum())})
    return rows


def sequential_cohort_ladder(frame, spec):
    """Sensitivity: identical fitting and scoring orders at all three timestamps."""
    train = frame.loc[frame.split.eq("train")].reset_index(drop=True)
    sequential_train = (train.eligible_approval & train.eligible_handover
                        & train.order_approved_at.le(train.order_delivered_carrier_date))
    test = frame.loc[frame.split.eq("test")].copy()
    sequential_test = (test.eligible_approval & test.eligible_handover
                       & test.order_approved_at.le(test.order_delivered_carrier_date))
    rows = []
    for fold, (fit_all, valid_all) in enumerate(temporal_folds(train, spec)):
        fit = train.iloc[fit_all[sequential_train.iloc[fit_all].to_numpy()]]
        valid = train.iloc[valid_all[sequential_train.iloc[valid_all].to_numpy()]]
        assert fit.outcome_available_at.lt(valid.order_purchase_timestamp.min()).all()
        for stage in STAGES:
            model, features = fit_model(fit, spec, stage)
            score = probability(model, valid, features)
            rows.append({"stage": stage, "period": f"validation_{fold + 1}",
                         "fit_orders": len(fit), **metrics(valid.is_late, score)})
    fit = train.loc[sequential_train]
    valid = test.loc[sequential_test]
    for stage in STAGES:
        model, features = fit_model(fit, spec, stage)
        score = probability(model, valid, features)
        rows.append({"stage": stage, "period": "test", "fit_orders": len(fit),
                     **metrics(valid.is_late, score)})
    return rows


def save(name, rows):
    pd.DataFrame(rows).to_csv(OUT / name, index=False)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    frame, spec = load()
    save("coverage.csv", coverage(frame))
    availability(spec).to_csv(OUT / "feature_availability.csv", index=False)
    (OUT / "stage0b_status.txt").write_text(
        "Stage 0b not fitted: shipping_limit_date exists in order_items, but the CSV "
        "has no timestamp showing when that deadline was set or revised. The "
        "preparation notebook marks it as post-purchase information. Its availability "
        "at checkout cannot be asserted, so using it would violate the stated stage rule.\n")

    predictions = {}
    final_models = {}
    period_rows, fit_rows, ci_rows, monthly_rows, within_rows = [], [], [], [], []
    for stage in STAGES:
        print(f"Scoring fixed {stage} model...", flush=True)
        val, test, fits, model, features = score_stage(frame, spec, stage)
        val.to_csv(OUT / f"{stage}_validation_predictions.csv", index=False)
        test.to_csv(OUT / f"{stage}_test_predictions.csv", index=False)
        predictions[stage] = (val, test)
        final_models[stage] = (model, features)
        period_rows.extend(period_results(val, test))
        fit_rows.extend(fits)
        test = test.copy()
        test["week"] = test.order_purchase_timestamp.dt.to_period("W").astype(str)
        for method, cluster in [("order", None), ("purchase_week", "week")]:
            ci = boot_ci(test, cluster=cluster)
            ci_rows.extend({"stage": stage, "method": method, "metric": key,
                            "low": values[0], "high": values[1]} for key, values in ci.items())
        month_results, within = monthly(test)
        monthly_rows.extend(month_results)
        within_rows.append(within)
    save("information_ladder.csv", period_rows)
    save("intercept_check.csv", fit_rows)
    save("test_intervals.csv", ci_rows)
    save("within_month.csv", monthly_rows)
    save("within_month_pair_weighted_auc.csv", within_rows)

    # Exact same evaluated Stage 2 rows for a transparent coverage-controlled comparison.
    eligible_test_ids = set(predictions["handover"][1].order_id)
    eligible_val_ids = set(predictions["handover"][0].order_id)
    matched = []
    for stage in STAGES:
        val, test = predictions[stage]
        for fold, group in val.loc[val.order_id.isin(eligible_val_ids)].groupby("fold", sort=True):
            matched.append({"stage": stage, "period": f"validation_{fold + 1}",
                            **metrics(group.is_late, group.probability)})
        group = test.loc[test.order_id.isin(eligible_test_ids)]
        matched.append({"stage": stage, "period": "test",
                        **metrics(group.is_late, group.probability)})
    save("common_stage2_cohort.csv", matched)

    print("Sequential common-cohort sensitivity...", flush=True)
    save("sequential_same_train_and_test.csv", sequential_cohort_ladder(frame, spec))

    # Confirm that the frozen checkout model's score was reproduced, not retuned.
    checkout_val, checkout_test = predictions["checkout"]
    saved = pd.read_csv(ROOT / "results/phase2/classification/selected_validation_predictions.csv",
                        float_precision="round_trip")
    check = checkout_val.merge(saved[["order_id", "probability"]], on="order_id",
                               suffixes=("_live", "_saved"), validate="one_to_one")
    assert len(check) == 38005
    assert np.max(np.abs(check.probability_live - check.probability_saved)) < 1e-9
    assert abs(average_precision_score(checkout_test.is_late, checkout_test.probability) - .065127) < 1e-5

    roll_rows = []
    for stage in ("checkout", "handover"):
        print(f"Rolling refits for {stage}...", flush=True)
        roll_rows.extend(rolling(frame, spec, stage))
    save("rolling_refit.csv", roll_rows)

    baseline_rows = []
    for stage in STAGES:
        baseline_rows.extend(baselines(frame, spec, stage))
    save("simple_baselines.csv", baseline_rows)

    severity_rows = []
    for stage in ("checkout", "handover"):
        for target in ("late_more_than_3", "late_more_than_7"):
            print(f"Severity {stage}, {target}...", flush=True)
            severity_rows.extend(severity(frame, spec, stage, target))
    save("severity.csv", severity_rows)

    # One-input removals are exploratory test diagnostics, not model selection.
    removal_rows = []
    for stage in ("checkout", "handover"):
        original = metrics(predictions[stage][1].is_late, predictions[stage][1].probability)
        print(f"Availability removals for {stage}...", flush=True)
        removal_rows.extend(removals(frame, spec, stage, original))
    save("availability_removals.csv", removal_rows)

    print("Stage 2 permutation and shuffle checks...", flush=True)
    model, features = final_models["handover"]
    importance, shuffled, count = stage2_sanity(frame, spec, model, features,
                                                 predictions["handover"][1])
    save("stage2_grouped_permutation.csv", importance)
    save("stage2_handover_shuffle.csv", shuffled)
    (OUT / "stage2_sanity_counts.json").write_text(json.dumps(count, indent=2) + "\n")

    summary = {
        "seed": SEED, "bootstrap_replicates": N_BOOT,
        "boundary": spec["primary_cutoff"], "logistic_C": .1,
        "stage0b": "not fitted because checkout availability cannot be asserted",
        "prediction_timestamps": {"checkout": "purchase (after submitted payment, assumed)",
                                  "approval": "order_approved_at",
                                  "handover": "order_delivered_carrier_date"},
        "stage2_approval_handling": "impute from fit median if approval is after handover",
        "missing_stage_events": "exclude from that stage; no prediction timestamp exists",
        "test_analysis_status": "all new staged and diagnostic test comparisons are exploratory/post hoc",
    }
    (OUT / "run_metadata.json").write_text(json.dumps(summary, indent=2) + "\n")
    print("Completed staged audit; tables saved in", OUT, flush=True)


if __name__ == "__main__":
    main()
