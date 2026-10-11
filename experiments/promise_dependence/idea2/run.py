"""Exploratory slack-anchored late-risk model, with validation-only tuning.

Run from repository root: it5006-proj/bin/python experiments/promise_dependence/idea2/run.py
The already-inspected later period is diagnostic only.
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import expit
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from threadpoolctl import threadpool_limits

REPO = Path(__file__).resolve().parents[3]
SOURCE = REPO / "phase2_final"
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(SOURCE))

from phase2_classification import feature_columns, make_pipeline  # noqa: E402
from phase2_eta import chronological_folds  # noqa: E402
from phase2_handover_classifier import (  # noqa: E402
    LABEL, fit_scores, load_cohort, monthly_flags, top_flags, variant_spec,
)

PRIMARY_CAPS = (0.5, 1.0)
# Added as post-hoc sensitivity after seeing August failure of the primary
# validation-selected candidate; never eligible for the primary selection.
EXPLORATORY_CAPS = (0.05, 0.1, 0.2)
CAPS = (*EXPLORATORY_CAPS, *PRIMARY_CAPS)
CS = (0.001, 0.01, 0.1)
TOP_SHARE = 0.10
SHORT_PROMISE = 14.5
# Removing the two components of the slack identity prevents exact reconstruction
# of promised_days inside the correction. The remaining features describe the
# parcel, route and seller, not Olist's promise policy or slack transformations.
EXCLUDE = {
    "promised_days", "promise_slack", "remaining_slack", "slack_shortfall",
    "past_promise_at_handover", "handover_days", "route_transit_90d_at_handover",
}


def source_hashes() -> dict[str, str]:
    paths = [
        HERE / "run.py", SOURCE / "phase2_handover_classifier.py",
        SOURCE / "phase2_classification.py", SOURCE / "phase2_eta.py",
        SOURCE / "data/phase2_feature_spec.json", SOURCE / "data/phase2_order_table.csv",
        SOURCE / "data/olist_orders_dataset.csv", SOURCE / "data/olist_customers_dataset.csv",
    ]
    return {str(path.relative_to(REPO)): hashlib.sha256(path.read_bytes()).hexdigest() for path in paths}


def correction_spec(spec: dict) -> dict:
    out = variant_spec(spec)
    out["numeric_features"] = [c for c in out["numeric_features"] if c not in EXCLUDE]
    assert not EXCLUDE.intersection(out["numeric_features"] + out["categorical_features"])
    return out


def slack_values(fit: pd.DataFrame, score: pd.DataFrame) -> np.ndarray:
    median = fit["route_transit_90d_at_handover"].median()
    return -(score["promised_days"] - score["handover_days"]
             - score["route_transit_90d_at_handover"].fillna(median)).to_numpy(dtype=float)


def base_fit(fit: pd.DataFrame) -> tuple[float, float]:
    """Calibrate log-odds against slack alone, using fitting labels only."""
    x = slack_values(fit, fit).reshape(-1, 1)
    with threadpool_limits(limits=1):
        model = LogisticRegression(C=np.inf, max_iter=1000).fit(x, fit[LABEL])
    return float(model.intercept_[0]), float(model.coef_[0, 0])


def offset_logit(intercept: float, slope: float, slack: np.ndarray) -> np.ndarray:
    return intercept + slope * slack


def offset_fit(x, y: np.ndarray, base: np.ndarray, c: float) -> np.ndarray:
    """Convex L2-penalized logistic fit with fixed base offset and no extra intercept."""
    y = np.asarray(y, dtype=float)
    def objective(beta):
        logits = base + np.asarray(x @ beta).reshape(-1)
        residual = expit(logits) - y
        loss = np.logaddexp(0, logits).sum() - np.dot(y, logits) + np.dot(beta, beta) / (2 * c)
        grad = np.asarray(x.T @ residual).reshape(-1) + beta / c
        return float(loss), grad
    result = minimize(objective, np.zeros(x.shape[1]), jac=True, method="L-BFGS-B",
                      options={"maxiter": 500, "ftol": 1e-9})
    if not result.success:
        raise RuntimeError(f"Offset logistic fit failed: {result.message}")
    return np.asarray(result.x)


def correction_scores(base: np.ndarray, x, beta: np.ndarray, cap: float) -> np.ndarray:
    adjustment = np.clip(np.asarray(x @ beta).reshape(-1), -cap, cap)
    return expit(base + adjustment)


def metrics(y, scores, *, period, ranking, n_fit, cap=None, c=None, months=None):
    yy = np.asarray(y, dtype=bool)
    flagged = top_flags(scores, TOP_SHARE) if months is None else monthly_flags(scores, months, TOP_SHARE)
    return {"period": period, "ranking": ranking, "fit_orders": n_fit, "orders": len(yy),
            "late": int(yy.sum()), "reviews": int(flagged.sum()), "caught": int((yy & flagged).sum()),
            "pr_auc": float(average_precision_score(yy, scores)),
            "roc_auc": float(roc_auc_score(yy, scores)),
            "recall_top10": float((yy & flagged).sum() / yy.sum()),
            "precision_top10": float((yy & flagged).sum() / flagged.sum()),
            "cap_log_odds": cap, "correction_c": c}


def fit_period(fit: pd.DataFrame, score: pd.DataFrame, spec: dict):
    intercept, slope = base_fit(fit)
    fit_slack, score_slack = slack_values(fit, fit), slack_values(fit, score)
    fit_base = offset_logit(intercept, slope, fit_slack)
    score_base = offset_logit(intercept, slope, score_slack)
    spec_c = correction_spec(spec)
    numeric, categorical = feature_columns(spec_c)
    prep = make_pipeline("linear", {"C": 0.01}, spec_c,
                         log_features=spec["linear_log_candidates"]).named_steps["prepare"]
    x_fit = prep.fit_transform(fit[numeric + categorical])
    x_score = prep.transform(score[numeric + categorical])
    y_fit = fit[LABEL].to_numpy()
    betas = {c: offset_fit(x_fit, y_fit, fit_base, c) for c in CS}
    predictions = {("anchored", c, cap): correction_scores(score_base, x_score, betas[c], cap)
                   for c in CS for cap in CAPS}
    predictions[("calibrated_slack", None, None)] = expit(score_base)
    predictions[("raw_slack_rule", None, None)] = score_slack
    predictions[("full_logistic_c001", None, None)] = fit_scores(
        "linear", {"C": 0.01, "class_weight": None}, variant_spec(spec), fit, score,
        spec["linear_log_candidates"])
    return predictions, {"intercept": intercept, "slope": slope, "n_fit": len(fit),
                         "max_known_outcome": str(fit["outcome_available_at"].max()),
                         "min_scored_purchase": str(score["order_purchase_timestamp"].min())}


def main():
    HERE.mkdir(parents=True, exist_ok=True)
    table, spec, _ = load_cohort(SOURCE)
    train = table[table[spec["primary_split"]].eq("train")].reset_index(drop=True)
    test = table[table[spec["primary_split"]].eq("test")].reset_index(drop=True)
    folds = chronological_folds(train, spec)
    rows = []
    fit_audit = []
    for fold, (fit_ix, valid_ix) in enumerate(folds):
        fit, valid = train.iloc[fit_ix], train.iloc[valid_ix]
        pred, audit = fit_period(fit, valid, spec)
        assert fit["outcome_available_at"].max() < valid["order_purchase_timestamp"].min()
        fit_audit.append({"period": f"W{fold+1}", **audit})
        for (name, c, cap), scores in pred.items():
            rows.append(metrics(valid[LABEL], scores, period=f"W{fold+1}", ranking=name,
                                n_fit=len(fit), cap=cap, c=c))
        print(f"W{fold+1}: fit {len(fit)} score {len(valid)}", flush=True)
    valid = pd.DataFrame(rows)
    valid.to_csv(HERE / "validation_windows.csv", index=False)
    summary = valid.groupby(["ranking", "correction_c", "cap_log_odds"], dropna=False, as_index=False)[
        ["pr_auc", "recall_top10", "precision_top10"]].mean().sort_values(
            ["pr_auc", "recall_top10"], ascending=False)
    summary.to_csv(HERE / "validation_summary.csv", index=False)
    # Selection does not use any later-period label. PR-AUC is the existing
    # model-selection metric; simpler/smaller corrections break exact ties.
    candidates = summary.loc[summary["ranking"].eq("anchored")
                             & summary["cap_log_odds"].isin(PRIMARY_CAPS)].copy()
    candidates = candidates.sort_values(["pr_auc", "correction_c", "cap_log_odds"],
                                        ascending=[False, True, True])
    winner = candidates.iloc[0]
    chosen_c, chosen_cap = float(winner["correction_c"]), float(winner["cap_log_odds"])
    (HERE / "selection.json").write_text(json.dumps({"criterion": "mean validation PR-AUC",
                                                       "primary_cap_grid": PRIMARY_CAPS,
                                                       "post_hoc_sensitivity_caps": EXPLORATORY_CAPS,
                                                       "correction_c": chosen_c, "cap_log_odds": chosen_cap}, indent=2))
    print(f"Validation selection: C={chosen_c}, cap={chosen_cap}; PR-AUC={winner['pr_auc']:.4f}", flush=True)
    # The final period is already opened. This is a diagnostic, not model selection.
    pred, audit = fit_period(train, test, spec)
    fit_audit.append({"period": "later", **audit})
    periods = {
        "later_all": np.ones(len(test), dtype=bool),
        "later_august_purchase": test["order_purchase_timestamp"].dt.to_period("M").eq("2018-08").to_numpy(),
        "later_august_short": (test["order_purchase_timestamp"].dt.to_period("M").eq("2018-08")
                               & test["promised_days"].lt(SHORT_PROMISE)).to_numpy(),
        "later_august_short_pre_promise": (test["order_purchase_timestamp"].dt.to_period("M").eq("2018-08")
                                           & test["promised_days"].lt(SHORT_PROMISE)
                                           & test["past_promise_at_handover"].eq(0)).to_numpy(),
    }
    keep = {("anchored", chosen_c, chosen_cap),
            ("calibrated_slack", None, None), ("raw_slack_rule", None, None),
            ("full_logistic_c001", None, None)}
    keep.update(("anchored", chosen_c, cap) for cap in EXPLORATORY_CAPS)
    for period, mask in periods.items():
        for (name, c, cap), scores in pred.items():
            if (name, c, cap) in keep:
                months = (test.loc[mask, "order_purchase_timestamp"].dt.to_period("M").astype(str).to_numpy()
                          if period == "later_all" else None)
                rows.append(metrics(test.loc[mask, LABEL], scores[mask], period=period,
                                    ranking=name, n_fit=len(train), cap=cap, c=c, months=months))
    pd.DataFrame(rows[len(valid):]).to_csv(HERE / "later_diagnostic.csv", index=False)
    pd.DataFrame(fit_audit).to_csv(HERE / "fit_audit.csv", index=False)
    (HERE / "source_sha256.json").write_text(json.dumps(source_hashes(), indent=2))
    print(summary.to_string(index=False), flush=True)
    print(pd.DataFrame(rows[len(valid):]).to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
