"""Reproducible Phase 2 classification validation; no holdout evaluation or final fit.

Run from the repository: python phase2_classification.py

Order of work (all on primary training and validation rows only):
1. Run the 13 declared candidates on the original feature set (including purchase_month).
2. Take the best linear and the best tree-based configuration and add one feature change at a time:
   drop purchase_month, add route features, add seller speed, log-transform skewed inputs (linear only).
3. Re-run all 13 candidates on the final feature set and select the highest mean PR-AUC.
4. Save the evidence the review notebook displays (ladder, split comparison, feature signal, importance, ranking).
"""

from __future__ import annotations

import hashlib
import json
import platform
import time
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.inspection import permutation_importance
from sklearn.metrics import (
    average_precision_score, confusion_matrix, f1_score, matthews_corrcoef,
    precision_score, recall_score, roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, OrdinalEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier
from threadpoolctl import threadpool_limits


ROOT = Path(__file__).resolve().parent
RESULT_DIR = ROOT / "results" / "phase2" / "classification"
MODEL_FAMILIES = {"dummy": "baseline", "linear": "linear", "tree": "tree_based", "forest": "tree_based"}

# Fixed before looking at classification validation scores. Order breaks exact ties.
CANDIDATES = [
    ("prior_baseline", "dummy", {}),
    ("logistic_baseline", "linear", {"C": 1.0, "class_weight": None}),
    ("logistic_c01", "linear", {"C": 0.1, "class_weight": None}),
    ("logistic_c10", "linear", {"C": 10.0, "class_weight": None}),
    ("logistic_balanced", "linear", {"C": 1.0, "class_weight": "balanced"}),
    ("tree_baseline", "tree", {"max_depth": None, "min_samples_leaf": 1}),
    ("tree_depth6", "tree", {"max_depth": 6, "min_samples_leaf": 50}),
    ("tree_depth12", "tree", {"max_depth": 12, "min_samples_leaf": 20}),
    ("tree_balanced", "tree", {"max_depth": 12, "min_samples_leaf": 20, "class_weight": "balanced"}),
    ("forest_baseline", "forest", {"max_depth": None, "min_samples_leaf": 1}),
    ("forest_depth12", "forest", {"max_depth": 12, "min_samples_leaf": 20}),
    ("forest_leaf20", "forest", {"max_depth": None, "min_samples_leaf": 20}),
    ("forest_balanced", "forest", {"max_depth": None, "min_samples_leaf": 20, "class_weight": "balanced_subsample"}),
]


def load_primary_training(root=ROOT):
    root = Path(root)
    spec = json.loads((root / "data/phase2_feature_spec.json").read_text())
    table = pd.read_csv(
        root / "data/phase2_order_table.csv",
        parse_dates=["order_purchase_timestamp", "outcome_available_at"],
    )
    # Select the agreed training population before any statistics or fitting.
    train = table.loc[table[spec["primary_split"]].eq("train")].reset_index(drop=True)
    if not train["order_id"].is_unique:
        raise ValueError("Primary training must have unique order IDs")
    cutoff = pd.Timestamp(spec["primary_cutoff"])
    if not (train["order_purchase_timestamp"].lt(cutoff).all()
            and train["outcome_available_at"].lt(cutoff).all()):
        raise ValueError("Primary training contains an order unavailable at the cutoff")
    features = spec["numeric_features"] + spec["categorical_features"]
    forbidden = {spec["classification_target"], spec["regression_target"], "order_id",
                 "outcome_available_at", "order_purchase_timestamp", "furthest_seller_id", "split", "cv_fold",
                 "random_split", "random_cv_fold"}
    if len(features) != len(set(features)) or forbidden.intersection(features):
        raise ValueError("Invalid predictor contract")
    return train, spec


def temporal_folds(train, spec):
    """Return positional indices compatible with sklearn's explicit CV interface."""
    end = pd.Timestamp(spec["primary_cutoff"]) - pd.Timedelta(days=spec["validation_maturity_days"])
    folds = []
    previous_end = None
    for fold in range(spec["n_folds"]):
        valid = np.flatnonzero(train[spec["primary_fold"]].eq(fold).to_numpy())
        if not len(valid):
            raise ValueError(f"Empty validation fold {fold}")
        starts = train.iloc[valid]["order_purchase_timestamp"]
        start = starts.min()
        if starts.max() >= end or (previous_end is not None and previous_end >= start):
            raise ValueError("Validation folds overlap in time or violate the maturity gap")
        fit = np.flatnonzero((train["order_purchase_timestamp"].lt(start)
                              & train["outcome_available_at"].lt(start)).to_numpy())
        if not len(fit) or np.intersect1d(fit, valid).size:
            raise ValueError("Invalid training fold")
        if train.iloc[fit][spec["classification_target"]].nunique() != 2:
            raise ValueError("Training fold needs both classes")
        if train.iloc[valid][spec["classification_target"]].nunique() != 2:
            raise ValueError("Validation fold needs both classes for ROC-AUC")
        folds.append((fit, valid))
        previous_end = starts.max()
    return folds


def hour_components(values):
    hours = np.asarray(values, dtype=float).reshape(-1)
    angle = hours * 2 * np.pi / 24
    return np.column_stack([np.sin(angle), np.cos(angle)])


def hour_names(transformer, input_names):
    return np.array(["hour_sin", "hour_cos"])


def feature_columns(spec, include_payment=True, feature_set=None):
    """Numeric and categorical inputs; `feature_set` ({"numeric", "categorical"}) overrides the saved contract."""
    numeric = feature_set["numeric"] if feature_set else spec["numeric_features"]
    categorical = feature_set["categorical"] if feature_set else spec["categorical_features"]
    omitted = set() if include_payment else set(spec["payment_sensitivity_features"])
    return [c for c in numeric if c not in omitted], [c for c in categorical if c not in omitted]


def feature_stages(spec):
    """Cumulative feature sets for the ladder. The last one is the saved contract."""
    history = spec["history_features"]["features"]
    route = ["route_typical_days", "promise_slack"]
    numeric = spec["numeric_features"]
    retired = list(spec["retired_features"])
    base = [c for c in numeric if c not in history]
    return {
        "original": {"numeric": base, "categorical": spec["categorical_features"] + retired},
        "no_month": {"numeric": base, "categorical": spec["categorical_features"]},
        "route": {"numeric": [c for c in numeric if c in base or c in route],
                  "categorical": spec["categorical_features"]},
        "seller": {"numeric": numeric, "categorical": spec["categorical_features"]},
    }


def make_pipeline(family, params, spec, include_payment=True, feature_set=None, log_features=()):
    """Model Pipeline. `log_features` are log1p-transformed before scaling, for the linear family only."""
    numeric, categorical = feature_columns(spec, include_payment, feature_set)
    seed = spec["random_state"]
    if family == "boost":
        # Gradient boosting (LightGBM-style histogram trees); categories are ordinal codes handled natively, unseen -> NaN.
        prepare = ColumnTransformer([
            ("numeric", SimpleImputer(strategy="median", keep_empty_features=True), numeric),
            ("category", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=np.nan,
                                        encoded_missing_value=np.nan), categorical),
        ], remainder="drop")
        mask = [False] * len(numeric) + [True] * len(categorical)
        return Pipeline([("prepare", prepare),
                         ("model", HistGradientBoostingClassifier(categorical_features=mask, early_stopping=False,
                                      random_state=seed, **params))])
    branches = []
    logged = []
    if family == "linear":
        logged = [c for c in numeric if c in set(log_features)]
        numeric = [c for c in numeric if c != "purchase_hour" and c not in logged]
        branches.append(("hour", FunctionTransformer(hour_components, feature_names_out=hour_names),
                         ["purchase_hour"]))
    numeric_steps = [("impute", SimpleImputer(strategy="median", keep_empty_features=True))]
    if family == "linear":
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
    if family == "dummy":
        model = DummyClassifier(strategy="prior")
    elif family == "linear":
        model = LogisticRegression(solver="lbfgs", max_iter=3000, random_state=seed, **params)
    elif family == "tree":
        model = DecisionTreeClassifier(random_state=seed, **params)
    elif family == "forest":
        model = RandomForestClassifier(n_estimators=200, n_jobs=4, random_state=seed,
                                       **{"max_features": "sqrt", **params})
    else:
        raise ValueError(f"Unknown model family: {family}")
    return Pipeline([("prepare", preparation), ("model", model)])


def metrics(y, probability, threshold=0.5):
    predicted = np.asarray(probability) >= threshold
    tn, fp, fn, tp = confusion_matrix(y, predicted, labels=[0, 1]).ravel()
    return {
        "average_precision": float(average_precision_score(y, probability)),
        "roc_auc": float(roc_auc_score(y, probability)),
        "precision": float(precision_score(y, predicted, zero_division=0)),
        "recall": float(recall_score(y, predicted, zero_division=0)),
        "f1": float(f1_score(y, predicted, zero_division=0)),
        "mcc": float(matthews_corrcoef(y, predicted)),
        "true_negative": int(tn), "false_positive": int(fp),
        "false_negative": int(fn), "true_positive": int(tp),
    }


def mcc_threshold(y, probability):
    """Maximise MCC over distinct score boundaries; ties favour fewer alerts."""
    y, probability = np.asarray(y, dtype=int), np.asarray(probability, dtype=float)
    if len(y) != len(probability) or len(np.unique(y)) != 2:
        raise ValueError("Threshold selection needs aligned predictions and both classes")
    if not np.isfinite(probability).all() or not np.isin(y, [0, 1]).all():
        raise ValueError("Invalid labels or probabilities")
    order = np.argsort(-probability, kind="stable")
    scores, labels = probability[order], y[order]
    last = np.r_[np.flatnonzero(scores[:-1] != scores[1:]), len(scores) - 1]
    tp = np.cumsum(labels, dtype=float)[last]
    fp = last + 1 - tp
    fn = labels.sum() - tp
    tn = len(labels) - labels.sum() - fp
    denominator = np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    mcc = np.divide(tp * tn - fp * fn, denominator, out=np.zeros_like(tp), where=denominator != 0)
    # Include the always-negative option, including when all probabilities equal 1.
    choices = np.r_[np.nextafter(scores[0], np.inf), scores[last]]
    return float(choices[np.argmax(np.r_[0.0, mcc])])


def evaluate_candidate(train, spec, folds, name, family, params, include_payment=True,
                       feature_set=None, log_features=(), stage="final_grid", feature_set_name="final"):
    rows, predictions = [], []
    numeric, categorical = feature_columns(spec, include_payment, feature_set)
    X, y = train[numeric + categorical], train[spec["classification_target"]]
    started = time.monotonic()
    for fold, (fit, valid) in enumerate(folds):
        pipeline = make_pipeline(family, params, spec, include_payment, feature_set, log_features)
        with threadpool_limits(limits=1):
            pipeline.fit(X.iloc[fit], y.iloc[fit])
            probability = pipeline.predict_proba(X.iloc[valid])[:, 1]
        rows.append({"candidate": name, "algorithm": family, "model_family": MODEL_FAMILIES[family],
                     "stage": stage, "feature_set": feature_set_name,
                     "log_transform": bool(len(log_features)) and family == "linear",
                     "include_payment": include_payment,
                     "fold": fold, "fit_orders": len(fit), "validation_orders": len(valid),
                     "validation_late_rate": float(y.iloc[valid].mean()),
                     **metrics(y.iloc[valid], probability)})
        predictions.append(pd.DataFrame({
            "order_id": train.iloc[valid]["order_id"].to_numpy(),
            "fold": fold, "is_late": y.iloc[valid].to_numpy(), "probability": probability,
        }))
    elapsed = time.monotonic() - started
    print(f"[{stage}] {name}: mean validation AP={np.mean([r['average_precision'] for r in rows]):.4f}; {elapsed:.1f}s", flush=True)
    return rows, pd.concat(predictions, ignore_index=True), elapsed


def summarize(rows):
    """Mean fold metrics per candidate, best mean average precision first (ties keep declared order)."""
    return pd.DataFrame(rows).groupby("candidate", sort=False).agg(
        mean_average_precision=("average_precision", "mean"),
        std_average_precision=("average_precision", "std"),
        mean_roc_auc=("roc_auc", "mean"),
        mean_precision_at_05=("precision", "mean"),
        mean_recall_at_05=("recall", "mean"),
        mean_f1_at_05=("f1", "mean"),
        mean_mcc_at_05=("mcc", "mean"),
    ).sort_values("mean_average_precision", ascending=False, kind="stable")


def risk_ranking(predictions, shares=(0.1, 0.2, 0.3)):
    """Review the riskiest `share` of orders within each validation period.

    Recall is the share of late orders in the flagged set; lift is how many times more often a flagged order is late
    than a typical order in that period. Random flagging would give recall equal to the share and lift of 1.
    """
    rows = []
    for share in shares:
        totals = {"flagged": 0, "caught": 0, "late": 0, "orders": 0}
        for fold, group in predictions.groupby("fold"):
            order = np.argsort(-group["probability"].to_numpy(), kind="stable")
            labels = group["is_late"].to_numpy()[order]
            flagged = int(np.ceil(share * len(labels)))
            counts = {"flagged": flagged, "caught": int(labels[:flagged].sum()),
                      "late": int(labels.sum()), "orders": len(labels)}
            rows.append({"scope": f"fold_{fold}", "top_share": share, **counts,
                         **_ranking_rates(counts)})
            for key, value in counts.items():
                totals[key] += value
        rows.append({"scope": "all_folds", "top_share": share, **totals, **_ranking_rates(totals)})
    return pd.DataFrame(rows)


def _ranking_rates(counts):
    precision = counts["caught"] / counts["flagged"]
    return {"recall": counts["caught"] / counts["late"], "precision": precision,
            "lift": precision / (counts["late"] / counts["orders"])}


LADDER_MODEL_STEPS = [
    # (step, line, candidate, compared with)
    (1, "baseline", "prior_baseline", None),
    (2, "linear", "logistic_baseline", "prior_baseline"),
    (3, "linear", "logistic_c01", "logistic_baseline"),
    (4, "linear", "logistic_balanced", "logistic_baseline"),
    (5, "tree_based", "tree_baseline", "prior_baseline"),
    (6, "tree_based", "tree_depth12", "tree_baseline"),
    (7, "tree_based", "forest_baseline", "tree_baseline"),
    (8, "tree_based", "forest_depth12", "forest_baseline"),
]
LADDER_LABELS = {
    "prior_baseline": "Baseline: always predict the training late rate",
    "logistic_baseline": "Logistic regression, default settings",
    "logistic_c01": "Logistic regression, stronger regularisation (C=0.1)",
    "logistic_balanced": "Logistic regression, class-weighted (vs default)",
    "tree_baseline": "Decision tree, default settings",
    "tree_depth12": "Decision tree, constrained (depth 12, leaf 20)",
    "forest_baseline": "Random forest, default settings",
    "forest_depth12": "Random forest, constrained (depth 12, leaf 20)",
}
FEATURE_STEPS = [
    (9, "no_month", "Drop purchase_month", "original"),
    (10, "route", "Add route promise slack and route typical days", "no_month"),
    (11, "seller", "Add seller ship speed", "route"),
    (12, "seller_log", "Log-transform skewed inputs (linear model only)", "seller"),
]


def build_ladder(metric_rows, winners):
    """One row per ladder step and fold, plus a summary comparing each step with the step it builds on."""
    frame = pd.DataFrame(metric_rows)

    def folds_of(stage, candidate):
        part = frame[(frame["stage"] == stage) & (frame["candidate"] == candidate)].sort_values("fold")
        if part.empty:
            raise KeyError((stage, candidate))
        return part

    entries = []
    for step, line, candidate, compare in LADDER_MODEL_STEPS:
        entries.append({"step": step, "line": line, "label": LADDER_LABELS[candidate],
                        "stage": "original_grid", "candidate": candidate,
                        "compare": ("original_grid", compare) if compare else None})
    for line, winner in winners.items():
        previous = ("original_grid", winner)
        for step, feature_stage, label, _ in FEATURE_STEPS:
            if feature_stage == "seller_log" and line != "linear":
                continue
            key = (f"feature_{feature_stage}", f"{winner}@{feature_stage}")
            entries.append({"step": step, "line": line, "label": label, "stage": key[0],
                            "candidate": key[1], "compare": previous})
            previous = key
    per_fold, summary = [], []
    for entry in sorted(entries, key=lambda e: (e["step"], e["line"])):
        part = folds_of(entry["stage"], entry["candidate"])
        ratio = (part["average_precision"] / part["validation_late_rate"]).to_numpy()
        for (_, row), value in zip(part.iterrows(), ratio):
            per_fold.append({"step": entry["step"], "line": entry["line"], "label": entry["label"],
                             "candidate": entry["candidate"], "fold": row["fold"],
                             "validation_late_rate": row["validation_late_rate"],
                             "average_precision": row["average_precision"], "roc_auc": row["roc_auc"],
                             "ap_over_late_rate": value})
        record = {"step": entry["step"], "line": entry["line"], "label": entry["label"],
                  "candidate": entry["candidate"],
                  "mean_average_precision": part["average_precision"].mean(),
                  "mean_roc_auc": part["roc_auc"].mean(),
                  "mean_ap_over_late_rate": ratio.mean()}
        if entry["compare"]:
            other = folds_of(*entry["compare"])
            record.update({
                "compared_with": entry["compare"][1],
                "delta_average_precision": part["average_precision"].mean() - other["average_precision"].mean(),
                "delta_roc_auc": part["roc_auc"].mean() - other["roc_auc"].mean(),
                "folds_improved_ap": int((part["average_precision"].to_numpy()
                                          > other["average_precision"].to_numpy()).sum()),
                "folds_improved_auc": int((part["roc_auc"].to_numpy() > other["roc_auc"].to_numpy()).sum()),
            })
        summary.append(record)
    return pd.DataFrame(per_fold), pd.DataFrame(summary)


def feature_signal(train, spec, folds):
    """Single-feature ranking power (ROC-AUC) of every numeric input on each validation fold."""
    y = train[spec["classification_target"]].to_numpy()
    rows = []
    for fold, (fit, valid) in enumerate(folds):
        for feature in spec["numeric_features"]:
            values = train[feature].fillna(train.iloc[fit][feature].median()).to_numpy()
            rows.append({"feature": feature, "fold": fold,
                         "roc_auc": float(roc_auc_score(y[valid], values[valid]))})
    return pd.DataFrame(rows)


def monthly_late_rate(train, spec):
    month = train["order_purchase_timestamp"].dt.to_period("M").astype(str)
    return (train.groupby(month)[spec["classification_target"]].agg(orders="size", late_orders="sum", late_rate="mean")
            .reset_index(names="purchase_month"))


def logistic_odds_ratios(train, spec, folds, params, feature_set, log_features):
    """Standardised coefficients and odds ratios of the linear model, refitted on each validation fold's training rows."""
    numeric, categorical = feature_columns(spec, True, feature_set)
    X, y = train[numeric + categorical], train[spec["classification_target"]]
    rows = []
    for fold, (fit, _) in enumerate(folds):
        pipeline = make_pipeline("linear", params, spec, True, feature_set, log_features)
        with threadpool_limits(limits=1):
            pipeline.fit(X.iloc[fit], y.iloc[fit])
        names = pipeline.named_steps["prepare"].get_feature_names_out()
        coefficients = pipeline.named_steps["model"].coef_[0]
        rows.extend({"fold": fold, "feature": name, "coefficient": float(c), "odds_ratio": float(np.exp(c))}
                    for name, c in zip(names, coefficients))
    return pd.DataFrame(rows)


def forest_permutation_importance(train, spec, folds, family, params, feature_set, repeats=5):
    """How much validation PR-AUC falls when one input is shuffled, per fold."""
    numeric, categorical = feature_columns(spec, True, feature_set)
    X, y = train[numeric + categorical], train[spec["classification_target"]]
    rows = []
    for fold, (fit, valid) in enumerate(folds):
        pipeline = make_pipeline(family, params, spec, True, feature_set)
        pipeline.fit(X.iloc[fit], y.iloc[fit])
        result = permutation_importance(pipeline, X.iloc[valid], y.iloc[valid], scoring="average_precision",
                                        n_repeats=repeats, random_state=spec["random_state"], n_jobs=1)
        rows.extend({"fold": fold, "feature": feature, "importance_mean": float(m), "importance_std": float(sd)}
                    for feature, m, sd in zip(X.columns, result.importances_mean, result.importances_std))
    return pd.DataFrame(rows)


def run_classification(root=ROOT):
    root = Path(root)
    output = root / "results/phase2/classification"
    output.mkdir(parents=True, exist_ok=True)
    train, spec = load_primary_training(root)
    folds = temporal_folds(train, spec)
    stages = feature_stages(spec)
    log_candidates = spec["linear_log_candidates"]
    all_rows, settings, elapsed_times = [], {}, {}

    def run(name, family, params, stage, set_name, include_payment=True, log=(), folds=folds, keep=None):
        rows, oof, elapsed = evaluate_candidate(train, spec, folds, name, family, params, include_payment,
                                                stages[set_name], log, stage, set_name)
        all_rows.extend(rows)
        if keep is not None:
            keep[name] = oof
        return rows, elapsed

    def record(name, family, params, set_name, include_payment=True, log=(), **extra):
        settings[name] = {"algorithm": family, "model_family": MODEL_FAMILIES[family], "parameters": params,
                          "feature_set": set_name, "include_payment": include_payment,
                          "log_transformed": list(log) if family == "linear" else [], **extra}

    # 1. Declared candidates on the original feature set (month included).
    for name, family, params in CANDIDATES:
        _, elapsed = run(name, family, params, "original_grid", "original")
    original = summarize([r for r in all_rows if r["stage"] == "original_grid"])
    winners = {}
    for line in ("linear", "tree_based"):
        winners[line] = next(n for n in original.index
                             if next(c for c in CANDIDATES if c[0] == n)[1] != "dummy"
                             and MODEL_FAMILIES[next(c for c in CANDIDATES if c[0] == n)[1]] == line)

    # 2. One feature change at a time, hyperparameters held fixed so only the features differ.
    candidate = {name: (family, params) for name, family, params in CANDIDATES}
    for line, winner in winners.items():
        family, params = candidate[winner]
        for _, set_name, _, _ in FEATURE_STEPS:
            if set_name == "seller_log":
                if family != "linear":
                    continue
                run(f"{winner}@seller_log", family, params, "feature_seller_log", "seller", log=log_candidates)
            else:
                run(f"{winner}@{set_name}", family, params, f"feature_{set_name}", set_name)

    # The log transform is adopted for the linear model only if it does not lower mean validation PR-AUC.
    frame = pd.DataFrame(all_rows)
    linear_winner = winners["linear"]
    with_log = frame[frame["candidate"] == f"{linear_winner}@seller_log"]["average_precision"].mean()
    without_log = frame[frame["candidate"] == f"{linear_winner}@seller"]["average_precision"].mean()
    adopt_log = bool(with_log >= without_log)
    final_log = tuple(log_candidates) if adopt_log else ()

    # 3. Re-run every declared candidate on the final feature set and select from them.
    predictions = {}
    for name, family, params in CANDIDATES:
        _, elapsed = run(name, family, params, "final_grid", "seller", log=final_log, keep=predictions)
        elapsed_times[name] = elapsed
        record(name, family, params, "seller", log=final_log)
    final_rows = [r for r in all_rows if r["stage"] == "final_grid"]
    final_summary = summarize(final_rows)

    # 4. Payment inputs: paired sensitivity check for each algorithm's best final configuration.
    for algorithm in ("linear", "tree", "forest"):
        best = next(n for n in final_summary.index if candidate[n][0] == algorithm)
        family, params = candidate[best]
        name = best + "_no_payment"
        _, elapsed = run(name, family, params, "payment_check", "seller", include_payment=False, log=final_log)
        elapsed_times[name] = elapsed
        record(name, family, params, "seller", False, final_log, paired_with=best)
    payment_rows = pd.DataFrame([r for r in all_rows if r["stage"] in ("final_grid", "payment_check")])
    payment_summary = summarize(payment_rows.to_dict("records"))
    sensitivity = []
    for name, configuration in settings.items():
        if "paired_with" in configuration:
            paired = configuration["paired_with"]
            sensitivity.append({
                "algorithm": configuration["algorithm"], "with_payment_candidate": paired,
                "with_payment_ap": payment_summary.loc[paired, "mean_average_precision"],
                "without_payment_ap": payment_summary.loc[name, "mean_average_precision"],
                "delta_without_minus_with": payment_summary.loc[name, "mean_average_precision"]
                                            - payment_summary.loc[paired, "mean_average_precision"]})

    selected = final_summary.index[0]
    oof = predictions[selected]
    threshold = mcc_threshold(oof["is_late"], oof["probability"])
    threshold_rows = [{"rule": label, "threshold": value, **metrics(oof["is_late"], oof["probability"], value)}
                      for label, value in [("default_0.5", 0.5), ("validation_selected_mcc", threshold)]]
    family_winners = {line: next(n for n in final_summary.index if settings[n]["model_family"] == line)
                      for line in ("linear", "tree_based")}

    # 5. Evidence for the review notebook.
    ladder, ladder_summary = build_ladder([r for r in all_rows if r["stage"] != "final_grid"], winners)
    eligible = np.flatnonzero(train[spec["primary_fold"]].ge(0).to_numpy())
    labels = train[spec["classification_target"]].to_numpy()
    shuffled = [(eligible[fit], eligible[valid]) for fit, valid in
                StratifiedKFold(spec["n_folds"], shuffle=True, random_state=spec["random_state"])
                .split(eligible, labels[eligible])]
    for line in ("linear", "tree_based"):
        name = family_winners[line]
        family, params = candidate[name]
        run(name, family, params, "random_split_check", "seller", log=final_log, folds=shuffled)
    split_rows = pd.DataFrame([r for r in all_rows if r["stage"] in ("final_grid", "random_split_check")
                               and r["candidate"] in family_winners.values()])
    split_rows["split"] = np.where(split_rows["stage"].eq("final_grid"), "time_based", "random")
    split_rows["ap_over_late_rate"] = split_rows["average_precision"] / split_rows["validation_late_rate"]
    split_comparison = split_rows.groupby(["candidate", "model_family", "split"], sort=False).agg(
        mean_average_precision=("average_precision", "mean"), mean_roc_auc=("roc_auc", "mean"),
        mean_ap_over_late_rate=("ap_over_late_rate", "mean")).reset_index()

    final_set = stages["seller"]
    linear_name, tree_name = family_winners["linear"], family_winners["tree_based"]
    odds = logistic_odds_ratios(train, spec, folds, candidate[linear_name][1], final_set, final_log)
    importance = forest_permutation_importance(train, spec, folds, candidate[tree_name][0],
                                               candidate[tree_name][1], final_set)
    correlation = train[spec["numeric_features"]].corr(method="spearman")

    fold_dates = []
    for fold, (fit, valid) in enumerate(folds):
        dates = train.iloc[valid]["order_purchase_timestamp"]
        fold_dates.append({"fold": fold, "fit_orders": len(fit), "validation_orders": len(valid),
                           "validation_start": str(dates.min()), "validation_end": str(dates.max()),
                           "latest_training_outcome": str(train.iloc[fit]["outcome_available_at"].max()),
                           "late_rate": float(train.iloc[valid][spec["classification_target"]].mean())})

    pd.DataFrame(all_rows).to_csv(output / "cv_metrics.csv", index=False)
    final_summary.assign(seconds=pd.Series(elapsed_times)).to_csv(output / "cv_summary.csv")
    oof.to_csv(output / "selected_validation_predictions.csv", index=False)
    pd.DataFrame(threshold_rows).to_csv(output / "threshold_metrics.csv", index=False)
    pd.DataFrame(fold_dates).to_csv(output / "folds.csv", index=False)
    ladder.to_csv(output / "ladder.csv", index=False)
    ladder_summary.to_csv(output / "ladder_summary.csv", index=False)
    split_comparison.to_csv(output / "split_comparison.csv", index=False)
    pd.DataFrame(sensitivity).to_csv(output / "payment_sensitivity.csv", index=False)
    monthly_late_rate(train, spec).to_csv(output / "monthly_late_rate.csv", index=False)
    feature_signal(train, spec, folds).to_csv(output / "feature_signal.csv", index=False)
    correlation.to_csv(output / "feature_correlation.csv")
    odds.to_csv(output / "logistic_odds_ratios.csv", index=False)
    importance.to_csv(output / "forest_permutation_importance.csv", index=False)
    risk = risk_ranking(oof)
    risk.to_csv(output / "risk_ranking.csv", index=False)
    by_fold = [{"fold": fold, **metrics(group["is_late"], group["probability"], threshold)}
               for fold, group in oof.groupby("fold")]
    pd.DataFrame(by_fold).to_csv(output / "selected_threshold_by_fold.csv", index=False)

    selection = {
        "status": "validation_only_pending_review", "selection_metric": "mean fold average precision",
        "pr_auc_definition": "sklearn average_precision_score (not trapezoidal PR area)",
        "selected_candidate": selected, "selected_threshold": threshold,
        "threshold_rule": "maximise pooled validation MCC; exact ties favour fewer alerts",
        "threshold_caveat": "selected and measured on the same validation predictions; not an unbiased final score",
        "family_winners": family_winners,
        "selection_scope": "highest mean average precision among the 13 declared candidates on the final feature set; "
                           "payment checks are reported separately",
        "winners_on_original_features": winners,
        "feature_stages": stages,
        "log_transform": {"columns": list(log_candidates), "adopted": adopt_log,
                          "rule": "adopt for the linear model if it does not lower mean validation PR-AUC",
                          "mean_ap_with_log": float(with_log), "mean_ap_without_log": float(without_log)},
        "candidates": settings,
        "fixed_settings": {
            "logistic": {"solver": "lbfgs", "max_iter": 3000},
            "forest": {"n_estimators": 200, "max_features": "sqrt", "n_jobs": 4},
            "numeric_imputation": "training-fold median", "categorical_encoding": "one-hot; ignore unknown",
            "linear_numeric_scaling": "StandardScaler", "linear_hour": "sine/cosine; period 24",
        },
        "random_state": spec["random_state"], "train_orders": len(train), "validation_orders": len(oof),
        "primary_test_evaluated": False, "final_model_fitted": False,
        "environment": {"python": platform.python_version(), "sklearn": sklearn.__version__,
                        "pandas": pd.__version__, "numpy": np.__version__},
        "input_sha256": {name: hashlib.sha256((root / "data" / name).read_bytes()).hexdigest()
                         for name in ("phase2_order_table.csv", "phase2_feature_spec.json")},
        "implementation_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "features_module_sha256": hashlib.sha256((Path(__file__).parent / "phase2_features.py").read_bytes()).hexdigest(),
    }
    (output / "selection.json").write_text(json.dumps(selection, indent=2) + "\n")
    print(f"Selected {selected}; validation MCC threshold {threshold:.6f}. No final model or holdout scores.", flush=True)
    return final_summary, pd.DataFrame(threshold_rows), selection


if __name__ == "__main__":
    run_classification()
