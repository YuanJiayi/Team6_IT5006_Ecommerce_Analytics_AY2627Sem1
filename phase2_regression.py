"""Phase 2 regression pipeline for delivery_days, following docs/phase2_regression_spec.md.

Run from the repository root:
    python phase2_regression.py              validation stage only (no test rows are scored or read)
    python phase2_regression.py --score-test the spec's test step; run once, after the selection is recorded
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, OrdinalEncoder, StandardScaler
from sklearn.tree import DecisionTreeRegressor
from threadpoolctl import threadpool_limits

from phase2_classification import feature_columns, hour_components, hour_names, load_primary_training, temporal_folds


ROOT = Path(__file__).resolve().parent
RESULT_DIR = ROOT / "results" / "phase2" / "regression"
TIE_MARGIN_DAYS = 0.05
MONTHLY_REFIT_DATES = ["2018-05-26", "2018-07-01", "2018-08-01"]

# Fixed by the spec before fitting. Order is the spec table.
CANDIDATES = [
    ("mean_baseline", "mean", {}),
    ("route_baseline", "route", {}),
    ("linear_baseline", "linear", {}),
    ("ridge_1", "ridge", {"alpha": 1.0}),
    ("ridge_10", "ridge", {"alpha": 10.0}),
    ("ridge_100", "ridge", {"alpha": 100.0}),
    ("tree_baseline", "tree", {"max_depth": None, "min_samples_leaf": 1}),
    ("tree_depth6", "tree", {"max_depth": 6, "min_samples_leaf": 50}),
    ("tree_depth12", "tree", {"max_depth": 12, "min_samples_leaf": 20}),
    ("forest_baseline", "forest", {"max_depth": None, "min_samples_leaf": 1}),
    ("forest_depth12", "forest", {"max_depth": 12, "min_samples_leaf": 20}),
    ("forest_leaf20", "forest", {"max_depth": None, "min_samples_leaf": 20}),
]
BASELINES = {"mean_baseline", "route_baseline"}
# Simplicity order for the tie rule: linear before tree before forest; within a family, fewer parameters first
# (a smaller or more constrained tree has fewer parameters; ridge alphas share a parameter count, so declared order).
SIMPLICITY = ["linear_baseline", "ridge_1", "ridge_10", "ridge_100",
              "tree_depth6", "tree_depth12", "tree_baseline",
              "forest_depth12", "forest_leaf20", "forest_baseline"]
LINEAR_FAMILIES = ("linear", "ridge")


class RouteBaseline(BaseEstimator, RegressorMixin):
    """Predict route_typical_days; missing values take the median of the fitting data."""

    def fit(self, X, y=None):
        self.median_ = float(X["route_typical_days"].median())
        return self

    def predict(self, X):
        return X["route_typical_days"].fillna(self.median_).to_numpy(dtype=float)


def make_regression_pipeline(family, params, spec):
    """Regression Pipeline; all preprocessing is fitted on whatever data `fit` receives."""
    seed = spec["random_state"]
    if family == "mean":
        return Pipeline([("model", DummyRegressor(strategy="mean"))])
    if family == "route":
        return Pipeline([("model", RouteBaseline())])
    numeric, categorical = feature_columns(spec)
    if family == "boost":
        # Gradient boosting (LightGBM-style histogram trees); categories are ordinal codes handled natively, unseen -> NaN.
        prepare = ColumnTransformer([
            ("numeric", SimpleImputer(strategy="median", keep_empty_features=True), numeric),
            ("category", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=np.nan,
                                        encoded_missing_value=np.nan), categorical),
        ], remainder="drop")
        mask = [False] * len(numeric) + [True] * len(categorical)
        return Pipeline([("prepare", prepare),
                         ("model", HistGradientBoostingRegressor(categorical_features=mask, early_stopping=False,
                                      random_state=seed, **params))])
    linear = family in LINEAR_FAMILIES
    branches, logged = [], []
    if linear:
        logged = [c for c in numeric if c in set(spec["linear_log_candidates"])]
        numeric = [c for c in numeric if c != "purchase_hour" and c not in logged]
        branches.append(("hour", FunctionTransformer(hour_components, feature_names_out=hour_names),
                         ["purchase_hour"]))
    numeric_steps = [("impute", SimpleImputer(strategy="median", keep_empty_features=True))]
    if linear:
        numeric_steps.append(("scale", StandardScaler()))
    branches.append(("numeric", Pipeline(numeric_steps), numeric))
    if logged:
        branches.append(("logged", Pipeline([
            ("impute", SimpleImputer(strategy="median", keep_empty_features=True)),
            ("log", FunctionTransformer(np.log1p, feature_names_out="one-to-one")),
            ("scale", StandardScaler()),
        ]), logged))
    branches.append(("category", OneHotEncoder(handle_unknown="ignore"), categorical))
    preparation = ColumnTransformer(branches, remainder="drop")
    if family == "linear":
        model = LinearRegression()
    elif family == "ridge":
        model = Ridge(random_state=seed, **params)
    elif family == "tree":
        model = DecisionTreeRegressor(random_state=seed, **params)
    elif family == "forest":
        model = RandomForestRegressor(n_estimators=200, n_jobs=4, random_state=seed, **{"max_features": "sqrt", **params})
    else:
        raise ValueError(f"Unknown model family: {family}")
    return Pipeline([("prepare", preparation), ("model", model)])


def regression_metrics(y, predicted):
    y, predicted = np.asarray(y, dtype=float), np.asarray(predicted, dtype=float)
    return {"mae": float(mean_absolute_error(y, predicted)),
            "rmse": float(np.sqrt(mean_squared_error(y, predicted))),
            "r2": float(r2_score(y, predicted)),
            "mean_signed_error": float(np.mean(predicted - y))}


def assert_fit_precedes(frame, fit, valid):
    """Every fitting order was purchased and its outcome known strictly before the window's first purchase."""
    start = frame.iloc[valid]["order_purchase_timestamp"].min()
    fitting = frame.iloc[fit]
    assert fitting["outcome_available_at"].lt(start).all(), "Fitting outcome not available before window start"
    assert fitting["order_purchase_timestamp"].lt(start).all(), "Fitting purchase not before window start"
    assert np.intersect1d(fit, valid).size == 0, "Fitting and validation rows overlap"


def evaluate_candidate(train, spec, folds, name, family, params):
    columns = sum(feature_columns(spec), [])
    X, y = train[columns], train[spec["regression_target"]]
    rows, predictions = [], []
    started = time.monotonic()
    for window, (fit, valid) in enumerate(folds):
        assert_fit_precedes(train, fit, valid)
        pipeline = make_regression_pipeline(family, params, spec)
        with threadpool_limits(limits=1):
            pipeline.fit(X.iloc[fit], y.iloc[fit])
            predicted = pipeline.predict(X.iloc[valid])
        rows.append({"candidate": name, "algorithm": family, "window": window, "fit_orders": len(fit),
                     "validation_orders": len(valid), **regression_metrics(y.iloc[valid], predicted)})
        predictions.append(pd.DataFrame({"order_id": train.iloc[valid]["order_id"].to_numpy(), "window": window,
                                         "actual": y.iloc[valid].to_numpy(), "predicted": predicted}))
    print(f"{name}: mean validation MAE={np.mean([r['mae'] for r in rows]):.4f}; "
          f"{time.monotonic() - started:.1f}s", flush=True)
    return rows, pd.concat(predictions, ignore_index=True)


def summarize(rows):
    """Mean window metrics per candidate in declared order."""
    return pd.DataFrame(rows).groupby("candidate", sort=False).agg(
        mean_mae=("mae", "mean"), mean_rmse=("rmse", "mean"), mean_r2=("r2", "mean"),
        mean_signed_error=("mean_signed_error", "mean"))


def select_candidate(mean_mae):
    """Spec rule: lowest mean MAE among non-baselines; candidates within 0.05 days of it lose to the simplest.

    `mean_mae` maps candidate name to mean validation MAE. Returns (name, reason).
    """
    scores = {n: v for n, v in mean_mae.items() if n not in BASELINES}
    if set(scores) != set(SIMPLICITY):
        raise ValueError("Scores must cover exactly the selectable candidates")
    best = min(scores, key=scores.get)
    near = [n for n in SIMPLICITY if scores[n] - scores[best] < TIE_MARGIN_DAYS]
    chosen = near[0]
    if chosen == best:
        reason = (f"{best} has the lowest mean MAE ({scores[best]:.4f}) and is also the simplest of "
                  f"{len(near)} candidate(s) within {TIE_MARGIN_DAYS} days")
    else:
        reason = (f"{best} has the lowest mean MAE ({scores[best]:.4f}); {chosen} ({scores[chosen]:.4f}) is within "
                  f"{TIE_MARGIN_DAYS} days and simpler, so it is chosen")
    return chosen, reason


def interpret(train, spec, folds, name, family, params, output):
    """Selected model on validation windows only: coefficients (linear) or permutation importance (tree)."""
    columns = sum(feature_columns(spec), [])
    X, y = train[columns], train[spec["regression_target"]]
    rows = []
    for window, (fit, valid) in enumerate(folds):
        pipeline = make_regression_pipeline(family, params, spec)
        with threadpool_limits(limits=1 if family in LINEAR_FAMILIES else None):
            pipeline.fit(X.iloc[fit], y.iloc[fit])
        if family in LINEAR_FAMILIES:
            names = pipeline.named_steps["prepare"].get_feature_names_out()
            rows.extend({"window": window, "feature": n, "coefficient": float(c)}
                        for n, c in zip(names, pipeline.named_steps["model"].coef_))
        else:
            result = permutation_importance(pipeline, X.iloc[valid], y.iloc[valid],
                                            scoring="neg_mean_absolute_error", n_repeats=5,
                                            random_state=spec["random_state"], n_jobs=1)
            rows.extend({"window": window, "feature": f, "importance_mean": float(m), "importance_std": float(s)}
                        for f, m, s in zip(columns, result.importances_mean, result.importances_std))
    filename = "coefficients.csv" if family in LINEAR_FAMILIES else "permutation_importance.csv"
    pd.DataFrame(rows).to_csv(output / filename, index=False)
    return filename


def run_validation(root=ROOT):
    root = Path(root)
    output = root / "results/phase2/regression"
    output.mkdir(parents=True, exist_ok=True)
    train, spec = load_primary_training(root)
    folds = temporal_folds(train, spec)
    all_rows, predictions = [], {}
    for name, family, params in CANDIDATES:
        rows, predicted = evaluate_candidate(train, spec, folds, name, family, params)
        all_rows.extend(rows)
        predictions[name] = predicted
    summary = summarize(all_rows)
    selected, reason = select_candidate(summary["mean_mae"].to_dict())
    family, params = next((f, p) for n, f, p in CANDIDATES if n == selected)
    interpretation = interpret(train, spec, folds, selected, family, params, output)

    pd.DataFrame(all_rows).to_csv(output / "cv_metrics.csv", index=False)
    summary.to_csv(output / "cv_summary.csv")
    predictions[selected].to_csv(output / "validation_predictions.csv", index=False)
    selection = {
        "selected_candidate": selected, "algorithm": family, "parameters": params, "reason": reason,
        "rule": f"lowest mean validation MAE over 5 windows; candidates within {TIE_MARGIN_DAYS} days of the lowest "
                "lose to the simplest (linear, tree, forest; fewer parameters first); baselines are reference only",
        "simplicity_order": SIMPLICITY,
        "mean_scores": summary.to_dict(orient="index"),
        "interpretation_file": interpretation,
        "train_orders": len(train), "random_state": spec["random_state"],
        "test_status": "not scored",
    }
    (output / "selection.json").write_text(json.dumps(selection, indent=2) + "\n")
    print(f"Selected {selected}: {reason}. Test not scored.", flush=True)
    return summary, selection


def score_test(root=ROOT):
    """Spec test step for the recorded selection: single refit, then monthly refits. Run only via --score-test."""
    root = Path(root)
    output = root / "results/phase2/regression"
    selection = json.loads((output / "selection.json").read_text())
    if selection["test_status"] != "not scored":
        raise RuntimeError("Test already scored; the spec allows a single test scoring")
    _, spec = load_primary_training(root)
    table = pd.read_csv(root / "data/phase2_order_table.csv",
                        parse_dates=["order_purchase_timestamp", "outcome_available_at"])
    columns = sum(feature_columns(spec), [])
    target = spec["regression_target"]
    family, params = selection["algorithm"], selection["parameters"]

    def fit_predict(fit_rows, test_rows):
        pipeline = make_regression_pipeline(family, params, spec)
        pipeline.fit(fit_rows[columns], fit_rows[target])
        return pipeline.predict(test_rows[columns])

    train = table[table[spec["primary_split"]].eq("train")]
    test = table[table[spec["primary_split"]].eq("test")].reset_index(drop=True)
    if len(train) != selection["train_orders"] or test["order_id"].isin(train["order_id"]).any():
        raise ValueError("Unexpected train/test split")
    test["predicted_single"] = fit_predict(train, test)
    test["predicted_monthly"] = np.nan
    dates = [pd.Timestamp(d) for d in MONTHLY_REFIT_DATES]
    for i, date in enumerate(dates):
        # All table rows purchased and with outcome known before the refit date.
        available = table[table["order_purchase_timestamp"].lt(date) & table["outcome_available_at"].lt(date)]
        later = dates[i + 1] if i + 1 < len(dates) else pd.Timestamp.max
        window = test["order_purchase_timestamp"].ge(date) & test["order_purchase_timestamp"].lt(later)
        assert available["outcome_available_at"].max() < date
        test.loc[window, "predicted_monthly"] = fit_predict(available, test[window])
    if test["predicted_monthly"].isna().any():
        raise ValueError("Some test orders precede the first monthly refit")

    month = test["order_purchase_timestamp"].dt.to_period("M").astype(str)
    rows = []
    for scheme in ("single", "monthly"):
        predicted = test[f"predicted_{scheme}"]
        rows.append({"scheme": scheme, "scope": "overall", "orders": len(test),
                     **regression_metrics(test[target], predicted)})
        for m, index in test.groupby(month).groups.items():
            rows.append({"scheme": scheme, "scope": m, "orders": len(index),
                         **regression_metrics(test.loc[index, target], predicted.loc[index])})
    pd.DataFrame(rows).to_csv(output / "test_metrics.csv", index=False)
    test[["order_id", target, "predicted_single", "predicted_monthly"]].to_csv(
        output / "test_predictions.csv", index=False)
    selection["test_status"] = "scored"
    (output / "selection.json").write_text(json.dumps(selection, indent=2) + "\n")
    return pd.DataFrame(rows)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--score-test", action="store_true",
                        help="score the held-out test once (requires a recorded validation selection)")
    if parser.parse_args().score_test:
        print(score_test().to_string(index=False))
    else:
        run_validation()
