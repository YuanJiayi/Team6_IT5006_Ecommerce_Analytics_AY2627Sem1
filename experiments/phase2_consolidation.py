"""Post-test Phase 2 experiments and report tables.

These analyses were proposed after the original checkout classification test was
opened. They do not replace its selected model, cutoff, or reported test score.
Run: it5006-proj/bin/python experiments/phase2_consolidation.py
"""

from __future__ import annotations

import copy
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, QuantileRegressor
from sklearn.metrics import average_precision_score, mean_absolute_error, mean_pinball_loss, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from phase2_classification import feature_columns, load_primary_training, make_pipeline, temporal_folds
from phase2_regression import make_regression_pipeline

OUT = ROOT / "experiments" / "phase2_consolidation"
QUANTILE_FEATURES = [
    "route_typical_days", "seller_ship_days", "distance_km", "same_state",
    "purchase_hour", "n_items", "n_products", "n_sellers", "total_price",
    "total_freight", "total_weight_g", "total_volume_cm3", "max_installments",
]


def top_precision(y, score, fraction=0.10):
    y, score = np.asarray(y), np.asarray(score)
    k = max(1, int(np.ceil(len(y) * fraction)))
    return float(y[np.argsort(-score, kind="stable")[:k]].mean())


def classification_metrics(y, score):
    return {
        "orders": len(y), "late_rate": float(np.mean(y)),
        "ap": float(average_precision_score(y, score)),
        "roc_auc": float(roc_auc_score(y, score)),
        "precision_at_10pct": top_precision(y, score),
    }


def business_rule_and_slices(table):
    val = pd.read_csv(ROOT / "results/phase2/classification/selected_validation_predictions.csv")
    test = pd.read_csv(ROOT / "reports/phase2_evidence/test_predictions.csv")
    context = table.drop(columns=["is_late"])
    val = val.merge(context, on="order_id", how="left", validate="one_to_one")
    test = test.drop(columns=["order_purchase_timestamp"]).merge(context, on="order_id", how="left", validate="one_to_one")
    rows = []
    for label, part in [(f"validation_{f+1}", val[val.fold.eq(f)]) for f in sorted(val.fold.unique())] + [("opened_test_post_hoc", test)]:
        y = part.is_late.to_numpy()
        score = part.route_typical_days - part.promised_days
        if not np.isfinite(score).all():
            score = score.fillna(float(score.median()))
        for name, s in [("historical_route_minus_promise_rule", score), ("frozen_logistic", part.probability)]:
            rows.append({"period": label, "variant": name, **classification_metrics(y, s)})
    for month, part in test.groupby(test.order_purchase_timestamp.dt.to_period("M")):
        if part.is_late.nunique() < 2:
            continue
        score = (part.route_typical_days - part.promised_days).fillna(
            float((part.route_typical_days - part.promised_days).median()))
        for name, s in [("historical_route_minus_promise_rule", score), ("frozen_logistic", part.probability)]:
            rows.append({"period": f"test_month_{month}", "variant": name,
                         **classification_metrics(part.is_late.to_numpy(), s)})
    pd.DataFrame(rows).to_csv(OUT / "checkout_rule.csv", index=False)

    # Descriptive error slices use saved, already-open test predictions only.
    test = test.assign(month=test.order_purchase_timestamp.dt.to_period("M").astype(str),
                       predicted_late=test.probability.ge(0.07878485347563163),
                       top_10pct=False)
    top = np.argsort(-test.probability.to_numpy(), kind="stable")[:int(np.ceil(len(test)*.1))]
    test.iloc[top, test.columns.get_loc("top_10pct")] = True
    test["promise_band"] = pd.cut(test.promised_days, [-np.inf, 14, 21, 30, np.inf], labels=["<=14", "15-21", "22-30", ">30"])
    test["distance_band"] = pd.cut(test.distance_km, [-np.inf, 100, 500, 1000, np.inf], labels=["<=100", "101-500", "501-1000", ">1000"])
    slices = []
    for field in ("month", "promise_band", "distance_band", "n_sellers"):
        for key, part in test.groupby(field, observed=True, dropna=False):
            if len(part) < 100:
                continue
            found = part.is_late.eq(1) & part.top_10pct
            slices.append({"dimension": field, "group": str(key), "orders": len(part),
                           "late": int(part.is_late.sum()), "late_rate": part.is_late.mean(),
                           "top10_selected": int(part.top_10pct.sum()),
                           "top10_late_found": int(found.sum()),
                           "late_missed_by_top10": int(part.is_late.sum() - found.sum())})
    pd.DataFrame(slices).to_csv(OUT / "classification_error_slices.csv", index=False)

    reg = pd.read_csv(ROOT / "results/phase2/regression/test_predictions.csv")
    reg = reg.merge(table.drop(columns=["delivery_days"]), on="order_id", how="left", validate="one_to_one")
    reg["month"] = reg.order_purchase_timestamp.dt.to_period("M").astype(str)
    reg["distance_band"] = pd.cut(reg.distance_km, [-np.inf, 100, 500, 1000, np.inf], labels=["<=100", "101-500", "501-1000", ">1000"])
    reg["promise_band"] = pd.cut(reg.promised_days, [-np.inf, 14, 21, 30, np.inf], labels=["<=14", "15-21", "22-30", ">30"])
    slices = []
    for field in ("month", "promise_band", "distance_band", "same_state"):
        for key, part in reg.groupby(field, observed=True, dropna=False):
            if len(part) < 100:
                continue
            error = part.predicted_single - part.delivery_days
            slices.append({"dimension": field, "group": str(key), "orders": len(part),
                           "mae": float(error.abs().mean()), "bias_pred_minus_actual": float(error.mean()),
                           "mean_actual_days": float(part.delivery_days.mean())})
    pd.DataFrame(slices).to_csv(OUT / "regression_error_slices.csv", index=False)


def quantile_models(train, test, folds):
    """Two numeric-feature 90th-percentile candidates, excluding the Olist promise."""
    results = []
    prediction_rows = []
    for name in ("linear_quantile", "shallow_boosted_quantile"):
        for period, fit, valid in [(f"validation_{i+1}", fit, valid) for i, (fit, valid) in enumerate(folds)] + [("opened_test_post_hoc", np.arange(len(train)), None)]:
            fit_rows = train.iloc[fit]
            hold = train.iloc[valid] if valid is not None else test
            if name == "linear_quantile":
                model = Pipeline([("impute", SimpleImputer(strategy="median")),
                                  ("scale", StandardScaler()),
                                  ("model", QuantileRegressor(quantile=.9, alpha=.05, solver="highs"))])
            else:
                model = Pipeline([("impute", SimpleImputer(strategy="median")),
                                  ("model", HistGradientBoostingRegressor(
                                      loss="quantile", quantile=.9, max_iter=100,
                                      max_leaf_nodes=15, min_samples_leaf=100,
                                      learning_rate=.05, random_state=42))])
            with threadpool_limits(limits=1):
                model.fit(fit_rows[QUANTILE_FEATURES], fit_rows.delivery_days)
                predicted = np.maximum(0, model.predict(hold[QUANTILE_FEATURES]))
            for variant, q in [(name, predicted), ("olist_promise", hold.promised_days.to_numpy())]:
                # Round before normalizing: fractional-day CSV arithmetic can
                # land a few nanoseconds before an exact Olist midnight.
                promised_dates = (hold.order_purchase_timestamp + pd.to_timedelta(q, unit="D")).dt.round("s").dt.normalize()
                actual_dates = hold.outcome_available_at.dt.normalize()
                if variant == "olist_promise" and not np.array_equal(
                    (actual_dates.to_numpy() <= promised_dates.to_numpy()),
                    hold.is_late.eq(0).to_numpy(),
                ):
                    raise AssertionError("Olist promise coverage must match calendar-date lateness")
                results.append({"period": period, "variant": variant, "orders": len(hold),
                                "calendar_date_coverage": float(np.mean(actual_dates.to_numpy() <= promised_dates.to_numpy())),
                                "pinball_90": float(mean_pinball_loss(hold.delivery_days, q, alpha=.9)),
                                "mean_predicted_days": float(np.mean(q)),
                                "mae": float(mean_absolute_error(hold.delivery_days, q))})
                prediction_rows.append(pd.DataFrame({"order_id": hold.order_id.to_numpy(),
                                                     "period": period, "variant": variant,
                                                     "actual_delivery_days": hold.delivery_days.to_numpy(),
                                                     "predicted_promise_days": q,
                                                     "on_or_before_promised_date": actual_dates.to_numpy() <= promised_dates.to_numpy()}))
            print("quantile", name, period, flush=True)
    pd.DataFrame(results).to_csv(OUT / "promise_quantile.csv", index=False)
    pd.concat(prediction_rows, ignore_index=True).to_csv(OUT / "promise_quantile_predictions.csv", index=False)


def refresh_promise_calendar_coverage(table):
    """Recheck saved forecasts against the exact calendar-date target rule."""
    path = OUT / "promise_quantile_predictions.csv"
    data = pd.read_csv(path)
    context = table[["order_id", "order_purchase_timestamp", "outcome_available_at", "is_late"]]
    data = data.drop(columns=["on_or_before_promised_date"]).merge(context, on="order_id", validate="many_to_one")
    promised_dates = (data.order_purchase_timestamp + pd.to_timedelta(data.predicted_promise_days, unit="D")).dt.round("s").dt.normalize()
    actual_dates = data.outcome_available_at.dt.normalize()
    data["on_or_before_promised_date"] = actual_dates.le(promised_dates)
    olist = data.variant.eq("olist_promise")
    if not data.loc[olist, "on_or_before_promised_date"].eq(data.loc[olist, "is_late"].eq(0)).all():
        raise AssertionError("Calendar-date promise and late labels disagree")
    summary = pd.read_csv(OUT / "promise_quantile.csv")
    coverage = data.groupby(["period", "variant"]).on_or_before_promised_date.mean()
    summary["calendar_date_coverage"] = [coverage.loc[(p, v)] for p, v in zip(summary.period, summary.variant)]
    summary.to_csv(OUT / "promise_quantile.csv", index=False)
    data.drop(columns=["order_purchase_timestamp", "outcome_available_at", "is_late"]).to_csv(path, index=False)


def wls_check(train, spec, folds):
    cols = sum(feature_columns(spec), [])
    rows = []
    for i, (fit, valid) in enumerate(folds):
        fitting, hold = train.iloc[fit], train.iloc[valid]
        # Variance tends to grow with expected route duration. Weights are based
        # only on the known, point-in-time route predictor, never the target.
        route = fitting.route_typical_days.fillna(fitting.route_typical_days.median()).clip(lower=1)
        weights = 1 / route
        weights /= weights.mean()
        for name, use_weights in (("ridge_100", False), ("route_weighted_ridge_100", True)):
            model = make_regression_pipeline("ridge", {"alpha": 100.0}, spec)
            with threadpool_limits(limits=1):
                model.fit(fitting[cols], fitting.delivery_days,
                          **({"model__sample_weight": weights} if use_weights else {}))
                predicted = np.maximum(0, model.predict(hold[cols]))
            rows.append({"period": f"validation_{i+1}", "variant": name,
                         "mae": mean_absolute_error(hold.delivery_days, predicted),
                         "bias_pred_minus_actual": float(np.mean(predicted - hold.delivery_days))})
        print("wls validation", i+1, flush=True)
    pd.DataFrame(rows).to_csv(OUT / "wls_validation.csv", index=False)


def promise_availability_sensitivity(train, test, spec, folds):
    """Remove both the stored promise and its derived slack from checkout inputs."""
    numeric = [c for c in spec["numeric_features"] if c not in {"promised_days", "promise_slack"}]
    categorical = list(spec["categorical_features"])
    features = {"numeric": numeric, "categorical": categorical}
    columns = numeric + categorical
    rows = []
    for period, fit, valid in [(f"validation_{i+1}", fit, valid) for i, (fit, valid) in enumerate(folds)] + [("opened_test_post_hoc", np.arange(len(train)), None)]:
        fitting = train.iloc[fit]
        hold = train.iloc[valid] if valid is not None else test
        model = make_pipeline("linear", {"C": .1, "class_weight": None}, spec,
                              feature_set=features, log_features=spec["linear_log_candidates"])
        with threadpool_limits(limits=1):
            model.fit(fitting[columns], fitting.is_late)
            probability = model.predict_proba(hold[columns])[:, 1]
        rows.append({"period": period, "variant": "no_stored_promise_or_slack",
                     **classification_metrics(hold.is_late.to_numpy(), probability)})
        print("promise availability", period, flush=True)
    pd.DataFrame(rows).to_csv(OUT / "promise_availability.csv", index=False)


def regression_reporting_checks(train, spec, folds):
    """Resolve open clipping, redundant-slack and final-coefficient checks."""
    val = pd.read_csv(ROOT / "results/phase2/regression/validation_predictions.csv")
    test = pd.read_csv(ROOT / "results/phase2/regression/test_predictions.csv")
    clip_rows = []
    for period, part, prediction in [("validation_pooled", val, "predicted"),
                                     ("opened_test_post_hoc", test, "predicted_single")]:
        actual = part["actual"].to_numpy() if period == "validation_pooled" else part.delivery_days.to_numpy()
        original = part[prediction].to_numpy()
        for name, p in (("original", original), ("clipped_at_zero", np.maximum(0, original))):
            clip_rows.append({"period": period, "variant": name, "orders": len(actual),
                              "negative_predictions": int(np.sum(original < 0)),
                              "mae": float(mean_absolute_error(actual, p))})
    pd.DataFrame(clip_rows).to_csv(OUT / "prediction_clip.csv", index=False)

    reduced = copy.deepcopy(spec)
    reduced["numeric_features"] = [c for c in spec["numeric_features"] if c != "promise_slack"]
    cols = sum(feature_columns(reduced), [])
    rows = []
    for i, (fit, valid) in enumerate(folds):
        fitting, hold = train.iloc[fit], train.iloc[valid]
        model = make_regression_pipeline("ridge", {"alpha": 100.0}, reduced)
        with threadpool_limits(limits=1):
            model.fit(fitting[cols], fitting.delivery_days)
            prediction = model.predict(hold[cols])
        rows.append({"period": f"validation_{i+1}", "variant": "ridge_without_promise_slack",
                     "mae": mean_absolute_error(hold.delivery_days, prediction)})
    pd.DataFrame(rows).to_csv(OUT / "drop_slack_validation.csv", index=False)

    ridge = make_regression_pipeline("ridge", {"alpha": 100.0}, reduced)
    with threadpool_limits(limits=1):
        ridge.fit(train[cols], train.delivery_days)
    names = ridge.named_steps["prepare"].get_feature_names_out()
    pd.DataFrame({"feature": names, "coefficient_days": ridge.named_steps["model"].coef_}).to_csv(
        OUT / "regression_final_coefficients_without_slack.csv", index=False)

    full_cols = sum(feature_columns(spec), [])
    logistic = make_pipeline("linear", {"C": .1, "class_weight": None}, spec,
                             log_features=spec["linear_log_candidates"])
    with threadpool_limits(limits=1):
        logistic.fit(train[full_cols], train.is_late)
    names = logistic.named_steps["prepare"].get_feature_names_out()
    coefficients = logistic.named_steps["model"].coef_.ravel()
    pd.DataFrame({"feature": names, "log_odds_coefficient": coefficients,
                  "odds_ratio_per_transformed_unit": np.exp(coefficients)}).to_csv(
        OUT / "classification_final_odds_ratios.csv", index=False)
    in_sample = logistic.predict_proba(train[full_cols])[:, 1]
    pd.DataFrame([{ "period": "final_fit_in_sample", "variant": "frozen_logistic",
                    **classification_metrics(train.is_late.to_numpy(), in_sample)}]).to_csv(
        OUT / "classification_train_score.csv", index=False)


def classification_v2(train, test, table, spec, folds):
    """Regression-gap classifier specified before the regression test was scored.

    This is a secondary analysis: its design followed the classification test.
    """
    cols = sum(feature_columns(spec), [])
    rows = []

    def score_period(fitting, hold):
        fitting = fitting.sort_values("order_purchase_timestamp")
        boundary = int(len(fitting) * .8)
        early, later = fitting.iloc[:boundary], fitting.iloc[boundary:]
        if early.outcome_available_at.max() >= later.order_purchase_timestamp.min():
            # Rebuild the inner fit so every training outcome predates the first
            # row used to train the gap classifier.
            early = early[early.outcome_available_at.lt(later.order_purchase_timestamp.min())]
        inner = make_regression_pipeline("ridge", {"alpha":100.0}, spec)
        with threadpool_limits(limits=1):
            inner.fit(early[cols], early.delivery_days)
            inner_gap = inner.predict(later[cols]) - later.promised_days.to_numpy()
            clf = LogisticRegression(C=1, random_state=42).fit(inner_gap.reshape(-1, 1), later.is_late)
            full = make_regression_pipeline("ridge", {"alpha":100.0}, spec)
            full.fit(fitting[cols], fitting.delivery_days)
            gap = full.predict(hold[cols]) - hold.promised_days.to_numpy()
            probability = clf.predict_proba(gap.reshape(-1, 1))[:, 1]
        return probability

    for period, fit, valid in [(f"validation_{i+1}", fit, valid) for i, (fit, valid) in enumerate(folds)] + [("opened_test_post_hoc", np.arange(len(train)), None)]:
        fitting = train.iloc[fit]
        hold = train.iloc[valid] if valid is not None else test
        probability = score_period(fitting, hold)
        rows.append({"period": period, "variant": "regression_gap_logistic_v2_single",
                     **classification_metrics(hold.is_late.to_numpy(), probability)})
        print("classification v2", period, flush=True)

    monthly_parts = []
    boundaries = [pd.Timestamp("2018-05-26"), pd.Timestamp("2018-07-01"),
                  pd.Timestamp("2018-08-01"), pd.Timestamp("2018-09-01")]
    for start, end in zip(boundaries[:-1], boundaries[1:]):
        available = table[table.order_purchase_timestamp.lt(start) & table.outcome_available_at.lt(start)]
        hold = test[test.order_purchase_timestamp.ge(start) & test.order_purchase_timestamp.lt(end)]
        if hold.empty:
            continue
        predicted = score_period(available, hold)
        monthly_parts.append(pd.DataFrame({"order_id": hold.order_id.to_numpy(),
                                           "is_late": hold.is_late.to_numpy(),
                                           "probability": predicted}))
        rows.append({"period": f"monthly_refit_{start.date()}",
                     "variant": "regression_gap_logistic_v2_monthly",
                     **classification_metrics(hold.is_late.to_numpy(), predicted)})
        print("classification v2 monthly", start.date(), flush=True)
    monthly = pd.concat(monthly_parts, ignore_index=True)
    if len(monthly) != len(test) or monthly.order_id.nunique() != len(test):
        raise AssertionError("Monthly version 2 must score each later-period order once")
    rows.append({"period": "opened_test_post_hoc_monthly_refit",
                 "variant": "regression_gap_logistic_v2_monthly",
                 **classification_metrics(monthly.is_late.to_numpy(), monthly.probability.to_numpy())})
    pd.DataFrame(rows).to_csv(OUT / "classification_v2.csv", index=False)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    train, spec = load_primary_training(ROOT)
    table = pd.read_csv(ROOT / "data/phase2_order_table.csv",
                        parse_dates=["order_purchase_timestamp", "outcome_available_at"])
    test = table[table.split.eq("test")].copy()
    folds = temporal_folds(train, spec)
    business_rule_and_slices(table)
    wls_check(train, spec, folds)
    promise_availability_sensitivity(train, test, spec, folds)
    regression_reporting_checks(train, spec, folds)
    classification_v2(train, test, table, spec, folds)
    quantile_models(train, test, folds)
    refresh_promise_calendar_coverage(table)
    (OUT / "metadata.json").write_text(json.dumps({
        "status": "post_hoc_after_checkout_test_opened",
        "train_orders": len(train), "test_orders": len(test),
        "prediction_point": "checkout unless separately marked handover",
        "quantile_features": QUANTILE_FEATURES,
        "quantile_model_scope": "reduced numeric-only comparison; exploratory",
        "frozen_models_unchanged": True,
    }, indent=2) + "\n")


if __name__ == "__main__":
    main()
