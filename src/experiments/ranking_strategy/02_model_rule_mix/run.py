"""Test two fixed mixes of the late-warning model and the remaining-slack rule against the gates.

Run from the repository root:
    python src/experiments/ranking_strategy/02_model_rule_mix/run.py

Candidates and gates are declared in src/experiments/ranking_strategy/EVALUATION_CRITERIA.md:
- mix_50_50: mean of the model percentile and the rule percentile
- short_switch: rule percentile for promises under the fitting orders' 10th percentile, model percentile otherwise

Percentiles compare each score with the same score on that period's fitting orders, so they are known at handover.
The script stops if the baselines do not reproduce Phase 2 and experiment 01. Writes CSVs to data/outputs/experiments/ranking_strategy/02_model_rule_mix/.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score
from threadpoolctl import threadpool_limits

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[2]))

from olist.paths import EXPERIMENT_OUTPUTS, REPO_ROOT  # noqa: E402

OUT = EXPERIMENT_OUTPUTS / "ranking_strategy" / HERE.name
OUT.mkdir(parents=True, exist_ok=True)
MODELS = EXPERIMENT_OUTPUTS / "ranking_strategy" / "models"

from olist.classification import feature_columns, make_pipeline  # noqa: E402
from olist.eta import CANDIDATES as ETA_CANDIDATES, PHASE2_REFERENCE, stage_spec  # noqa: E402
from olist.handover_classifier import (  # noqa: E402
    CANDIDATES as CLS_CANDIDATES, LABEL, PROMISE, TOP_SHARE, baseline_scores, chronological_folds, gate_verdict,
    load_cohort, monthly_flags, paired_bootstrap, top_flags, variant_spec,
)

MIXES = ["mix_50_50", "short_switch"]
# From simplest to most complex, the order used when several candidates pass.
CANDIDATES = ["mix_50_50", "short_switch", "model"]
RANKINGS = ["slack_rule", *CANDIDATES, "regression_gap", "deadline"]
SHORT_SHARE = 0.10
SHORT_WINDOWS = ["W1", "W2", "W4", "W5"]  # W3 has only 3 late short-promise orders
REPORT_CAUGHT = {"W1": (164, 132), "W2": (341, 271), "W3": (126, 112), "W4": (274, 227), "W5": (469, 357),
                 "test": (377, 398)}
EXPERIMENT_01_CAUGHT = {"regression_gap": 411, "deadline": 359}
BOOTSTRAPS, SEED = 1000, 42
TOLERANCE = 1e-6


def lookup(candidates, name):
    return next((family, params) for n, family, params in candidates if n == name)


def rule_scores(fit_rows, rows):
    median = fit_rows["route_transit_90d_at_handover"].median()
    return -(rows[PROMISE] - rows["handover_days"] - rows["route_transit_90d_at_handover"].fillna(median)).to_numpy()


def percentile(reference, scores):
    """Share of fitting orders with a score at or below each order's score."""
    return np.searchsorted(np.sort(reference), scores, side="right") / len(reference)


def all_rankings(fit_rows, rows, model_fit, model_rows, spec, regression_model):
    """Every ranking for `rows`; `model_fit` and `model_rows` hold the model's scores on fitting and scored orders."""
    family, params = regression_model
    base = baseline_scores(fit_rows, rows, stage_spec(spec, "handover"), family, params)
    rule = rule_scores(fit_rows, rows)
    if not np.allclose(rule, base["slack_rule"]):
        raise AssertionError("Rule scores differ from Phase 2's slack rule")
    model_pct = percentile(model_fit, model_rows)
    rule_pct = percentile(rule_scores(fit_rows, fit_rows), rule)
    short = (rows[PROMISE] < fit_rows[PROMISE].quantile(SHORT_SHARE)).to_numpy()
    return {"slack_rule": rule, "model": model_rows, "regression_gap": base["regression_gap"],
            "deadline": (rows["handover_days"] - rows[PROMISE]).to_numpy(dtype=float),
            "mix_50_50": (model_pct + rule_pct) / 2, "short_switch": np.where(short, rule_pct, model_pct)}


def pr_auc(y, scores):
    return average_precision_score(y, scores) if y.any() else np.nan


def metric_row(period, name, rows, scores, flags, short_cut, daily_flags=None):
    y = rows[LABEL].to_numpy().astype(bool)
    not_past = rows["past_promise_at_handover"].ne(1).to_numpy()
    short = (rows[PROMISE] < short_cut).to_numpy() & not_past
    row = {"period": period, "ranking": name, "orders": len(y), "late": int(y.sum()), "pr_auc": pr_auc(y, scores),
           "roc_auc": roc_auc_score(y, scores) if y.any() else np.nan, "reviewed": int(flags.sum()),
           "caught": int((y & flags).sum()), "recall": (y & flags).sum() / y.sum() if y.any() else np.nan,
           "precision": (y & flags).sum() / flags.sum(),
           "late_not_past": int((y & not_past).sum()), "caught_not_past": int((y & flags & not_past).sum()),
           "pr_auc_not_past": pr_auc(y[not_past], scores[not_past]),
           "late_short_not_past": int((y & short).sum()), "pr_auc_short_not_past": pr_auc(y[short], scores[short])}
    if daily_flags is not None:
        row["caught_daily"] = int((y & daily_flags).sum())
    return row


def test_bootstrap(y, scores, months, not_past):
    """95% intervals for paired test differences, resampling orders within each purchase month."""
    rng = np.random.default_rng(SEED)
    groups = [np.flatnonzero(months == m) for m in np.unique(months)]
    pairs = [(name, "slack_rule") for name in RANKINGS[1:]]
    pairs += [("short_switch", "mix_50_50"), ("model", "mix_50_50"), ("model", "short_switch")]
    draws = []
    for _ in range(BOOTSTRAPS):
        index = np.concatenate([rng.choice(g, len(g)) for g in groups])
        draw = {}
        for name in scores:
            flags = monthly_flags(scores[name][index], months[index], TOP_SHARE)
            draw[name, "caught"] = (y[index] & flags).sum()
            draw[name, "caught_not_past"] = (y[index] & flags & not_past[index]).sum()
            draw[name, "pr_auc"] = average_precision_score(y[index], scores[name][index])
        draws.append(draw)
    rows = []
    for a, b in pairs:
        for metric in ("caught", "caught_not_past", "pr_auc"):
            gaps = np.array([d[a, metric] - d[b, metric] for d in draws], dtype=float)
            rows.append({"a": a, "b": b, "metric": metric, "ci_low": np.percentile(gaps, 2.5),
                         "ci_high": np.percentile(gaps, 97.5)})
    return pd.DataFrame(rows)


def above_zero(intervals, a, b):
    part = intervals[intervals["a"].eq(a) & intervals["b"].eq(b) & intervals["metric"].isin(["pr_auc", "caught"])]
    return bool(part["ci_low"].gt(0).any())


def evaluate_gates(metrics, windows, intervals):
    by_period = metrics.set_index(["period", "ranking"])
    rule_test = by_period.loc[("test", "slack_rule")]
    rows = []
    for name in CANDIDATES:
        value = paired_bootstrap(windows, name, "slack_rule")
        gaps = np.array([by_period.loc[(w, name), "pr_auc_short_not_past"] - by_period.loc[(w, "slack_rule"), "pr_auc_short_not_past"]
                         for w in SHORT_WINDOWS])
        short_gap, short_se = gaps.mean(), gaps.std(ddof=1) / np.sqrt(len(gaps))
        test_row = by_period.loc[("test", name)]
        at_least = all(test_row[m] >= rule_test[m] for m in ("pr_auc", "caught", "caught_not_past"))
        rows.append({"candidate": name,
                     "val_pr_auc_diff": value["pr_auc"]["difference"], "val_pr_auc_ci_low": value["pr_auc"]["ci_low"],
                     "val_pr_auc_ci_high": value["pr_auc"]["ci_high"],
                     "val_recall_diff": value["recall_top10"]["difference"], "val_recall_ci_low": value["recall_top10"]["ci_low"],
                     "val_recall_ci_high": value["recall_top10"]["ci_high"], "gate_a": gate_verdict(value),
                     "short_pr_auc_diff": short_gap, "short_paired_se": short_se, "gate_b": bool(short_gap >= -short_se),
                     "test_pr_auc_diff": test_row["pr_auc"] - rule_test["pr_auc"], "test_caught_diff": test_row["caught"] - rule_test["caught"],
                     "test_caught_not_past_diff": test_row["caught_not_past"] - rule_test["caught_not_past"],
                     "test_at_least_rule": at_least, "test_clear_lead": above_zero(intervals, name, "slack_rule")})
        rows[-1]["gate_c"] = bool(at_least and rows[-1]["test_clear_lead"])
        rows[-1]["passes_all"] = bool(rows[-1]["gate_a"] and rows[-1]["gate_b"] and rows[-1]["gate_c"])
    gates = pd.DataFrame(rows)
    winner = "slack_rule"
    for name in gates.loc[gates["passes_all"], "candidate"]:
        if winner == "slack_rule" or above_zero(intervals, name, winner):
            winner = name
    return gates, winner


def check_baselines(metrics, gates, short_cut):
    for period, expected in REPORT_CAUGHT.items():
        part = metrics[metrics["period"].eq(period)].set_index("ranking")["caught"]
        got = (int(part["model"]), int(part["slack_rule"]))
        print(f"{period}: model, rule caught {got} (report {expected})", flush=True)
        if got != expected:
            raise AssertionError(f"{period} does not reproduce the report: {got} vs {expected}")
    test = metrics[metrics["period"].eq("test")].set_index("ranking")
    for name, expected in EXPERIMENT_01_CAUGHT.items():
        if int(test.loc[name, "caught"]) != expected:
            raise AssertionError(f"{name} does not reproduce experiment 01")
    saved = json.loads((REPO_ROOT / "data/outputs/phase2/handover/selection.json").read_text())["variants"]["primary"]
    saved = next(v for k, v in saved.items() if isinstance(v, dict) and "model_minus_slack_rule" in v)["model_minus_slack_rule"]
    model = gates.set_index("candidate").loc["model"]
    if abs(model["val_pr_auc_diff"] - saved["pr_auc"]["difference"]) > TOLERANCE or abs(model["val_pr_auc_ci_low"] - saved["pr_auc"]["ci_low"]) > TOLERANCE:
        raise AssertionError("Gate A does not reproduce the saved Phase 2 interval")
    short = pd.read_csv(REPO_ROOT / "data/outputs/phase2/handover/short_promise_model_vs_rule.csv").set_index("period")
    if abs(short_cut - short.loc["W1", "short_cut_days"]) > TOLERANCE:
        raise AssertionError("Short-promise cutoff differs from Phase 2")
    for window in SHORT_WINDOWS:
        part = metrics[metrics["period"].eq(window)].set_index("ranking")["pr_auc_short_not_past"]
        if abs(part["model"] - short.loc[window, "pr_auc_model"]) > TOLERANCE or abs(part["slack_rule"] - short.loc[window, "pr_auc_rule"]) > TOLERANCE:
            raise AssertionError(f"{window} short-promise PR-AUC differs from Phase 2")


def run():
    table, spec, _ = load_cohort(REPO_ROOT)
    train = table[table[spec["primary_split"]].eq("train")].reset_index(drop=True)
    test = table[table[spec["primary_split"]].eq("test")].reset_index(drop=True)
    folds = chronological_folds(train, spec)
    # The Phase 2 regression (ridge alpha=100), not the Phase 3 selection: this experiment compares Phase 2 rankings.
    regression_model = lookup(ETA_CANDIDATES, PHASE2_REFERENCE)
    family, params = lookup(CLS_CANDIDATES, "logistic_c0.01")
    spec_v = variant_spec(spec)
    columns = sum(feature_columns(spec_v), [])
    short_cut = float(train[PROMISE].quantile(SHORT_SHARE))
    metrics, windows = [], []

    for number, (fit, valid) in enumerate(folds, start=1):
        fit_rows, valid_rows = train.iloc[fit], train.iloc[valid].reset_index(drop=True)
        pipeline = make_pipeline(family, params, spec_v, log_features=spec["linear_log_candidates"])
        with threadpool_limits(limits=1):
            pipeline.fit(fit_rows[columns], fit_rows[LABEL])
        scores = all_rankings(fit_rows, valid_rows, pipeline.predict_proba(fit_rows[columns])[:, 1],
                              pipeline.predict_proba(valid_rows[columns])[:, 1], spec, regression_model)
        metrics += [metric_row(f"W{number}", name, valid_rows, scores[name], top_flags(scores[name], TOP_SHARE), short_cut)
                    for name in RANKINGS]
        windows.append({"y": valid_rows[LABEL].to_numpy(), **scores})

    manifest = json.loads((MODELS / "phase2_final_models_manifest.json").read_text())["models"]["late_warning"]
    baseline = joblib.load(MODELS / manifest["file"])
    scores = all_rankings(train, test, baseline.predict_proba(train[columns])[:, 1],
                          baseline.predict_proba(test[columns])[:, 1], spec, regression_model)
    months = test["order_purchase_timestamp"].dt.to_period("M").astype(str).to_numpy()
    days = test["handover_time"].dt.strftime("%Y-%m-%d").to_numpy()
    flags = {name: monthly_flags(scores[name], months, TOP_SHARE) for name in RANKINGS}
    daily = {name: monthly_flags(scores[name], days, TOP_SHARE) for name in RANKINGS}
    y = test[LABEL].to_numpy().astype(bool)
    metrics += [metric_row("test", name, test, scores[name], flags[name], short_cut, daily[name]) for name in RANKINGS]
    for month in np.unique(months):
        part = months == month
        metrics += [metric_row(month, name, test[part], scores[name][part], flags[name][part], short_cut, daily[name][part])
                    for name in RANKINGS]
    metrics = pd.DataFrame(metrics)

    intervals = test_bootstrap(y, scores, months, test["past_promise_at_handover"].ne(1).to_numpy())
    gates, winner = evaluate_gates(metrics, windows, intervals)
    check_baselines(metrics, gates, short_cut)

    ties = pd.DataFrame([{"ranking": name, "unique_score_share": len(np.unique(scores[name])) / len(y),
                          "caught_ties_reversed": int((y[::-1] & monthly_flags(scores[name][::-1], months[::-1], TOP_SHARE)).sum())}
                         for name in RANKINGS])
    metrics.to_csv(OUT / "metrics.csv", index=False)
    intervals.to_csv(OUT / "test_bootstrap.csv", index=False)
    gates.to_csv(OUT / "gates.csv", index=False)
    ties.to_csv(OUT / "test_ties.csv", index=False)

    valid_part = metrics[metrics["period"].str.fullmatch("W[1-5]")]
    print(valid_part.groupby("ranking", sort=False).agg(pr_auc=("pr_auc", "mean"), roc_auc=("roc_auc", "mean"), caught=("caught", "sum"),
                                                        caught_not_past=("caught_not_past", "sum")).round(3).to_string())
    print(metrics[~metrics["period"].str.fullmatch("W[1-5]")].round(3).to_string(index=False))
    print(valid_part.pivot(index="ranking", columns="period", values="pr_auc").round(3).to_string())
    print(gates.round(4).T.to_string())
    print(intervals.round(3).to_string(index=False))
    print(ties.round(3).to_string(index=False))
    print(f"Winner under the gates: {winner}", flush=True)


if __name__ == "__main__":
    run()
