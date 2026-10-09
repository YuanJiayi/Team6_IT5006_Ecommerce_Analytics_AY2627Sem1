"""Two-stage delivery-time estimate: a checkout forecast and an update at carrier handover.

Follows docs/phase2_two_stage_eta.md. Run from the repository root:
    python phase2_eta.py

The checkout stage is the frozen Phase 2 regression (same 20 inputs). The handover stage predicts the same target,
`delivery_days`, once the seller has handed the parcel to the carrier, so it may also use the time already elapsed.
Both stages are scored on the same orders: the five chronological validation windows (primary), the later test
period (post hoc: it had been opened before this stage was designed) and a random same-period split (secondary).
Nothing here edits the frozen feature contract, the order table or results/phase2/regression/.
"""

from __future__ import annotations

import copy
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingRegressor, VotingRegressor
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "experiments"))

from phase2_classification import feature_columns, load_primary_training, temporal_folds  # noqa: E402
from phase2_recent_history_audit import asof_mean  # noqa: E402
from phase2_regression import assert_fit_precedes, make_regression_pipeline, regression_metrics  # noqa: E402

RESULT_DIR = ROOT / "results" / "phase2" / "eta"
TARGET = "delivery_days"
TIE_MARGIN_DAYS = 0.05
TRANSIT_WINDOW_DAYS = 90
ROUTE_PRIOR_STRENGTH = 20
SECONDS_PER_DAY = 86400

# Inputs added to the frozen checkout contract at each rung. Every rung is scored on the same orders.
STAGE_FEATURES = {
    "checkout": [],
    "checkout_transit": ["route_transit_90d"],
    "handover_elapsed": ["route_transit_90d", "approval_days", "handover_days"],
    "handover": ["approval_days", "handover_days", "route_transit_90d_at_handover"],
}
CHECKOUT_STAGE, HANDOVER_STAGE = "checkout", "handover"
LADDER_MODEL = ("ridge", {"alpha": 100.0})  # the frozen checkout selection, held fixed along the ladder

# Fixed before fitting. Each family starts from its simplest variant.
CANDIDATES = [
    ("mean_baseline", "mean", {}),
    ("elapsed_plus_route_baseline", "rule", {}),
    ("linear_baseline", "linear", {}),
    ("ridge_1", "ridge", {"alpha": 1.0}),
    ("ridge_10", "ridge", {"alpha": 10.0}),
    ("ridge_100", "ridge", {"alpha": 100.0}),
    ("tree_baseline", "tree", {"max_depth": None, "min_samples_leaf": 1}),
    ("tree_depth6", "tree", {"max_depth": 6, "min_samples_leaf": 50}),
    ("forest_leaf20", "forest", {"max_depth": None, "min_samples_leaf": 20}),
    ("boost_baseline", "boost", {}),
    ("boost_slow", "boost", {"learning_rate": 0.05, "max_iter": 300, "min_samples_leaf": 40}),
]
BASELINES = {"mean_baseline", "elapsed_plus_route_baseline"}
SIMPLICITY = ["linear_baseline", "ridge_1", "ridge_10", "ridge_100", "tree_depth6", "tree_baseline",
              "forest_leaf20", "boost_baseline", "boost_slow"]
FAMILY_OF = {"linear": "linear", "ridge": "linear", "tree": "tree_based", "forest": "tree_based",
             "boost": "tree_based"}


def days_between(later, earlier):
    return (later - earlier).dt.total_seconds() / SECONDS_PER_DAY


def add_stage_features(table, orders):
    """Add handover-stage inputs to the prepared order table. Returns a new frame.

    `orders` needs order_id, order_approved_at and order_delivered_carrier_date. Every added value is known at
    the stated prediction point: route transit history uses only deliveries recorded strictly before the purchase
    (checkout) or the carrier handover (handover stage), and an approval recorded after handover is treated as unknown.
    """
    raw = orders[["order_id", "order_approved_at", "order_delivered_carrier_date"]]
    out = table.merge(raw, on="order_id", how="left", validate="one_to_one")
    purchase, delivered = out["order_purchase_timestamp"], out["outcome_available_at"]
    handover = out["order_delivered_carrier_date"]
    out["handover_time"] = handover
    out["route"] = out["seller_state"].astype(str) + "->" + out["customer_state"].astype(str)
    out["handover_valid"] = handover.notna() & handover.ge(purchase) & handover.lt(delivered)

    out["handover_days"] = days_between(handover, purchase).where(out["handover_valid"])
    approval = days_between(out["order_approved_at"], purchase)
    known = out["order_approved_at"].le(handover) & approval.ge(0)
    out["approval_days"] = approval.where(known)

    # Carrier-to-customer days of completed deliveries: history only, never an input for its own order.
    transit = days_between(delivered, handover).where(out["handover_valid"])
    out["transit_days"] = transit
    out["route_transit_90d"] = asof_mean(out["route"], transit, delivered, out["route"], purchase,
                                         TRANSIT_WINDOW_DAYS, ROUTE_PRIOR_STRENGTH)
    at_handover = asof_mean(out["route"], transit, delivered, out["route"],
                            handover.where(out["handover_valid"], purchase), TRANSIT_WINDOW_DAYS, ROUTE_PRIOR_STRENGTH)
    out["route_transit_90d_at_handover"] = np.where(out["handover_valid"], at_handover, np.nan)
    return out


def load_stage_table(root=ROOT):
    """Prepared order table with stage inputs, restricted to orders with a usable carrier handover."""
    root = Path(root)
    _, spec = load_primary_training(root)
    table = pd.read_csv(root / "data/phase2_order_table.csv",
                        parse_dates=["order_purchase_timestamp", "outcome_available_at"])
    orders = pd.read_csv(root / "data/olist_orders_dataset.csv",
                         usecols=["order_id", "order_approved_at", "order_delivered_carrier_date"],
                         parse_dates=["order_approved_at", "order_delivered_carrier_date"])
    staged = add_stage_features(table, orders)
    cohort = staged[staged["handover_valid"]].reset_index(drop=True)
    coverage = {"orders": int(len(staged)), "handover_cohort": int(len(cohort)),
                "excluded_no_valid_handover": int(len(staged) - len(cohort))}
    return cohort, spec, coverage


def stage_spec(spec, stage):
    changed = copy.deepcopy(spec)
    changed["numeric_features"] = spec["numeric_features"] + STAGE_FEATURES[stage]
    return changed


class ElapsedPlusRoute:
    """Rule baseline: days already elapsed at handover plus the route's recent carrier-to-customer average."""

    def fit(self, X, y=None):
        self.median_ = float(X["route_transit_90d_at_handover"].median())
        return self

    def predict(self, X):
        return (X["handover_days"] + X["route_transit_90d_at_handover"].fillna(self.median_)).to_numpy(dtype=float)


def make_boost(params, spec):
    numeric, categorical = feature_columns(spec)
    prepare = ColumnTransformer([
        ("numeric", SimpleImputer(strategy="median", keep_empty_features=True), numeric),
        ("category", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1), categorical),
    ], remainder="drop")
    return Pipeline([("prepare", prepare),
                     ("model", HistGradientBoostingRegressor(random_state=spec["random_state"], **params))])


def make_stage_pipeline(family, params, spec):
    """All preprocessing sits inside the Pipeline, so it is fitted on the fitting rows only."""
    if family == "rule":
        return ElapsedPlusRoute()
    if family == "boost":
        return make_boost(params, spec)
    if family == "voting":
        members = [(name, make_stage_pipeline(f, p, spec)) for name, f, p in params["members"]]
        return VotingRegressor(members)
    return make_regression_pipeline(family, params, spec)


def chronological_folds(train, spec):
    folds = temporal_folds(train, spec)
    for fit, valid in folds:
        assert_fit_precedes(train, fit, valid)
    return folds


def random_folds(train, spec):
    fold = train[spec["secondary_fold"]]
    return [(np.flatnonzero(fold.ne(k).to_numpy()), np.flatnonzero(fold.eq(k).to_numpy()))
            for k in range(spec["n_folds"])]


def fit_predict(family, params, spec, fit_rows, score_rows):
    columns = sum(feature_columns(spec), [])
    pipeline = make_stage_pipeline(family, params, spec)
    with threadpool_limits(limits=1 if family in ("linear", "ridge") else None):
        pipeline.fit(fit_rows[columns], fit_rows[TARGET])
        return pipeline.predict(score_rows[columns])


def cross_validate(train, spec, folds, family, params, label):
    rows = []
    started = time.monotonic()
    for window, (fit, valid) in enumerate(folds):
        predicted = fit_predict(family, params, spec, train.iloc[fit], train.iloc[valid])
        rows.append({**label, "window": window, "fit_orders": len(fit), "validation_orders": len(valid),
                     **regression_metrics(train.iloc[valid][TARGET], predicted)})
    print(f"{' / '.join(map(str, label.values()))}: mean R2={np.mean([r['r2'] for r in rows]):.4f}, "
          f"mean MAE={np.mean([r['mae'] for r in rows]):.4f}; {time.monotonic() - started:.1f}s", flush=True)
    return rows


def summarize(rows, keys):
    return (pd.DataFrame(rows).groupby(keys, sort=False)
            .agg(mean_mae=("mae", "mean"), mean_rmse=("rmse", "mean"), mean_r2=("r2", "mean"),
                 min_r2=("r2", "min"), mean_signed_error=("mean_signed_error", "mean")).reset_index())


def select_candidate(mean_mae):
    """Same rule as the checkout regression: lowest mean validation MAE; within 0.05 days the simplest wins."""
    scores = {n: v for n, v in mean_mae.items() if n in SIMPLICITY}
    if set(scores) != set(SIMPLICITY):
        raise ValueError("Scores must cover exactly the selectable candidates")
    best = min(scores, key=scores.get)
    chosen = next(n for n in SIMPLICITY if scores[n] - scores[best] < TIE_MARGIN_DAYS)
    reason = (f"{best} has the lowest mean validation MAE ({scores[best]:.4f})"
              + ("" if chosen == best else f"; {chosen} ({scores[chosen]:.4f}) is within {TIE_MARGIN_DAYS} days "
                                           "and simpler, so it is chosen"))
    return chosen, best, reason


def score_later_test(table, spec, entries):
    """Refit on all primary training orders and score the later period, overall and by purchase month."""
    train = table[table[spec["primary_split"]].eq("train")]
    test = table[table[spec["primary_split"]].eq("test")].reset_index(drop=True)
    month = test["order_purchase_timestamp"].dt.to_period("M").astype(str)
    rows = []
    for label, family, params, changed in entries:
        predicted = fit_predict(family, params, changed, train, test)
        rows.append({**label, "scope": "overall", "orders": len(test), **regression_metrics(test[TARGET], predicted)})
        for m, index in test.groupby(month).groups.items():
            rows.append({**label, "scope": m, "orders": len(index),
                         **regression_metrics(test.loc[index, TARGET], predicted[index])})
    return rows


def score_random_benchmark(table, spec, entries):
    """Same-period benchmark: random 5-fold validation and random test from the saved secondary split."""
    train = table[table[spec["secondary_split"]].eq("train")].reset_index(drop=True)
    test = table[table[spec["secondary_split"]].eq("test")]
    folds = random_folds(train, spec)
    rows = []
    for label, family, params, changed in entries:
        cv = summarize(cross_validate(train, changed, folds, family, params, {**label, "split": "random"}),
                       list(label)).iloc[0]
        predicted = fit_predict(family, params, changed, train, test)
        scored = regression_metrics(test[TARGET], predicted)
        rows.append({**label, "cv_mean_mae": cv["mean_mae"], "cv_mean_rmse": cv["mean_rmse"],
                     "cv_mean_r2": cv["mean_r2"], "test_orders": len(test),
                     **{f"test_{k}": v for k, v in scored.items()}})
    return rows


def interpret(train, spec, folds, family, params, output, stem):
    """Validation windows only: coefficients for a linear model, permutation importance otherwise."""
    columns = sum(feature_columns(spec), [])
    rows = []
    for window, (fit, valid) in enumerate(folds):
        pipeline = make_stage_pipeline(family, params, spec)
        pipeline.fit(train.iloc[fit][columns], train.iloc[fit][TARGET])
        if family in ("linear", "ridge"):
            names = pipeline.named_steps["prepare"].get_feature_names_out()
            rows.extend({"window": window, "feature": n, "coefficient": float(c)}
                        for n, c in zip(names, pipeline.named_steps["model"].coef_))
        else:
            result = permutation_importance(pipeline, train.iloc[valid][columns], train.iloc[valid][TARGET],
                                            scoring="neg_mean_absolute_error", n_repeats=5,
                                            random_state=spec["random_state"], n_jobs=1)
            rows.extend({"window": window, "feature": f, "importance_mean": float(m), "importance_std": float(s)}
                        for f, m, s in zip(columns, result.importances_mean, result.importances_std))
    name = f"{stem}_{'coefficients' if family in ('linear', 'ridge') else 'permutation_importance'}.csv"
    pd.DataFrame(rows).to_csv(output / name, index=False)
    return name


def run(root=ROOT):
    root = Path(root)
    output = root / "results/phase2/eta"
    output.mkdir(parents=True, exist_ok=True)
    table, spec, coverage = load_stage_table(root)
    train = table[table[spec["primary_split"]].eq("train")].reset_index(drop=True)
    folds = chronological_folds(train, spec)
    coverage.update(train_orders=int(len(train)), test_orders=int(table[spec["primary_split"]].eq("test").sum()))

    # 1. Information ladder: one fixed model, inputs added by prediction point.
    ladder_rows = []
    for stage in STAGE_FEATURES:
        ladder_rows += cross_validate(train, stage_spec(spec, stage), folds, *LADDER_MODEL,
                                      {"stage": stage, "model": "ridge_100"})
    pd.DataFrame(ladder_rows).to_csv(output / "ladder_windows.csv", index=False)
    ladder = summarize(ladder_rows, ["stage", "model"])
    ladder.to_csv(output / "ladder_summary.csv", index=False)

    # 2. Model comparison at the handover stage.
    handover_spec = stage_spec(spec, HANDOVER_STAGE)
    cv_rows = []
    for name, family, params in CANDIDATES:
        cv_rows += cross_validate(train, handover_spec, folds, family, params,
                                  {"candidate": name, "algorithm": family})
    summary = summarize(cv_rows, ["candidate", "algorithm"])
    scores = summary.set_index("candidate")["mean_mae"].to_dict()
    selected, lowest, reason = select_candidate(scores)
    lookup = {name: (family, params) for name, family, params in CANDIDATES}
    winners = {}
    for group in ("linear", "tree_based"):
        members = [n for n in SIMPLICITY if FAMILY_OF[lookup[n][0]] == group]
        winners[group] = min(members, key=scores.get)

    # 3. Optional ensemble: equal-weight average of the best linear and best tree-based candidate.
    voting = {"members": [(n, *lookup[n]) for n in winners.values()]}
    cv_rows += cross_validate(train, handover_spec, folds, "voting", voting,
                              {"candidate": "voting_family_winners", "algorithm": "voting"})
    pd.DataFrame(cv_rows).to_csv(output / "cv_metrics.csv", index=False)
    summary = summarize(cv_rows, ["candidate", "algorithm"])
    summary.to_csv(output / "cv_summary.csv", index=False)
    best_r2 = summary.loc[summary["mean_r2"].idxmax(), "candidate"]

    # 4. Later period (post hoc) and the random same-period benchmark, for both stages.
    selected_model = lookup[selected]
    tree_model = lookup[winners["tree_based"]]
    entries = [
        ({"stage": CHECKOUT_STAGE, "candidate": "ridge_100"}, *LADDER_MODEL, stage_spec(spec, CHECKOUT_STAGE)),
        ({"stage": CHECKOUT_STAGE, "candidate": winners["tree_based"]}, *tree_model, stage_spec(spec, CHECKOUT_STAGE)),
        ({"stage": HANDOVER_STAGE, "candidate": selected}, *selected_model, handover_spec),
        ({"stage": HANDOVER_STAGE, "candidate": winners["tree_based"]}, *tree_model, handover_spec),
        ({"stage": HANDOVER_STAGE, "candidate": "voting_family_winners"}, "voting", voting, handover_spec),
    ]
    entries = [e for i, e in enumerate(entries) if e[0] not in [x[0] for x in entries[:i]]]
    pd.DataFrame(score_later_test(table, spec, entries)).to_csv(output / "test_metrics.csv", index=False)
    pd.DataFrame(score_random_benchmark(table, spec, entries)).to_csv(output / "random_benchmark.csv", index=False)

    # 5. Interpretation on validation windows.
    files = [interpret(train, handover_spec, folds, *selected_model, output, "selected")]
    if winners["tree_based"] != selected:
        files.append(interpret(train, handover_spec, folds, *tree_model, output, "tree_winner"))

    selection = {
        "prediction_points": {"checkout": spec["prediction_point"],
                              "handover": "when the carrier records receiving the parcel from the seller"},
        "target": TARGET, "population": "delivered orders with a carrier handover after purchase and before delivery",
        "coverage": coverage, "stage_features": STAGE_FEATURES,
        "selected_candidate": selected, "lowest_mae_candidate": lowest, "highest_r2_candidate": best_r2,
        "reason": reason,
        "rule": f"lowest mean validation MAE over {spec['n_folds']} chronological windows; candidates within "
                f"{TIE_MARGIN_DAYS} days of the lowest lose to the simplest; baselines and the ensemble are reference only",
        "family_winners": winners, "simplicity_order": SIMPLICITY,
        "interpretation_files": files, "random_state": spec["random_state"],
        "test_status": "post hoc: the later period was opened before the handover stage was designed",
    }
    (output / "selection.json").write_text(json.dumps(selection, indent=2) + "\n")
    print(f"\nSelected {selected}: {reason}. Highest mean validation R2: {best_r2}.", flush=True)
    return ladder, summary, selection


if __name__ == "__main__":
    ladder, summary, _ = run()
    print(ladder.round(4).to_string(index=False))
    print(summary.round(4).to_string(index=False))
