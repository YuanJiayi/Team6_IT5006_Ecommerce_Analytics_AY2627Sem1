"""Add the regression ranking (handover estimate minus promised days) to the period shortfalls and the weekly test intervals.

Run from the repository root:
    python src/experiments/ranking_strategy/16_regression_comparator/run.py

Declared in src/experiments/ranking_strategy/EVALUATION_CRITERIA.md. Adds no candidates and changes no scores: the regression ranking is the
one scored in experiment 01, and the other four rankings are those of experiments 13 and 15. Stops if earlier results do
not reproduce. Writes CSVs to data/outputs/experiments/ranking_strategy/16_regression_comparator/.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
EXPERIMENTS = HERE.parent
sys.path.insert(0, str(HERE.parents[2]))

from olist.paths import EXPERIMENT_OUTPUTS  # noqa: E402

OUT = EXPERIMENT_OUTPUTS / "ranking_strategy" / HERE.name
OUT.mkdir(parents=True, exist_ok=True)
OUTPUTS = EXPERIMENT_OUTPUTS / "ranking_strategy"
sys.path.insert(0, str(EXPERIMENTS))

from task4_common import LABEL, load_data  # noqa: E402


def load_run(name, folder):
    spec = importlib.util.spec_from_file_location(name, EXPERIMENTS / folder / "run.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


exp01 = load_run("experiment_01", "01_ranking_baselines")
exp08 = load_run("experiment_08", "08_final_comparison")
exp15 = load_run("experiment_15", "15_weekly_intervals")

REGRESSION = "regression_gap"
RANKINGS = ["slack_rule", "model", exp15.BLEND, exp15.DEADLINE, REGRESSION]
PAIRS = [(exp15.BLEND, REGRESSION), ("slack_rule", REGRESSION), ("model", REGRESSION), (REGRESSION, exp15.DEADLINE)]
EXPECTED_REGRESSION = (0.500, 411)
TOLERANCE = 1e-3


def shortfalls():
    four = pd.read_csv(OUTPUTS / "13_deadline_comparison" / "periods.csv")
    first = pd.read_csv(OUTPUTS / "01_ranking_baselines" / "metrics.csv")
    first = first[first["ranking"].eq(REGRESSION) & ~first["period"].eq("test")]
    regression = pd.DataFrame({"period": first["period"], "ranking": REGRESSION, "pr_auc": first["pr_auc"],
                               "pr_auc_not_past": first["pr_auc_not_past_promise"], "recall_top10": first["recall"]})
    periods = pd.concat([four, regression], ignore_index=True)
    used = periods[~periods["period"].isin(exp08.EXCLUDED_FROM_SHORTFALLS)]
    rows = []
    for metric in exp08.SHORTFALL_METRICS:
        table = used.pivot(index="period", columns="ranking", values=metric)[RANKINGS]
        behind = table.rsub(table.max(axis=1), axis=0)
        rows += [{"metric": metric, "ranking": name, "largest_shortfall": behind[name].max(), "period_of_largest": behind[name].idxmax(),
                  "mean_shortfall": behind[name].mean(), "periods_best": int(behind[name].eq(0).sum())} for name in RANKINGS]
    return periods, pd.DataFrame(rows)


def intervals(y, scores, months, weeks):
    point = {name: exp15.list_values(y, s, months) for name, s in scores.items()}
    blocks, picks = exp15.week_blocks(weeks)
    draws = []
    for number, pick in enumerate(picks, start=1):
        index = np.concatenate([blocks[k] for k in pick])
        draws.append({name: exp15.list_values(y[index], s[index], months[index]) for name, s in scores.items()})
        if number % 500 == 0:
            print(f"Resampling round {number} of {exp15.BOOTSTRAPS}", flush=True)
    rows = []
    for a, b in PAIRS:
        for metric in ("pr_auc", "caught"):
            gaps = np.array([d[a][metric] - d[b][metric] for d in draws], dtype=float)
            rows.append({"a": a, "b": b, "metric": metric, "a_minus_b": point[a][metric] - point[b][metric],
                         "ci_low": np.percentile(gaps, 2.5), "ci_high": np.percentile(gaps, 97.5)})
    return point, pd.DataFrame(rows)


def run():
    periods, behind = shortfalls()
    periods.to_csv(OUT / "periods.csv", index=False)
    behind.to_csv(OUT / "period_shortfalls.csv", index=False)
    pd.set_option("display.width", 200)
    print(periods.pivot(index="period", columns="ranking", values="pr_auc")[RANKINGS].round(3).to_string())
    print(behind[behind["metric"].eq("pr_auc")].round(3).to_string(index=False), flush=True)

    data = load_data()
    train, test, spec = data["train"], data["test"], data["spec"]
    regression_model = exp01.lookup(exp01.ETA_CANDIDATES, exp01.PHASE2_REFERENCE)
    scores = {exp15.BLEND: exp15.exp10.monthly_scores(data, test, exp15.exp10.TEST_START)[exp15.BLEND],
              "slack_rule": exp15.shared.rule_scores(train, test), "model": exp15.saved_model_scores(data, test),
              exp15.DEADLINE: (test["handover_days"] - test[exp15.PROMISE]).to_numpy(dtype=float),
              REGRESSION: exp01.rankings_for(train, test, spec, regression_model)[REGRESSION]}
    y = test[LABEL].to_numpy().astype(bool)
    months = test["order_purchase_timestamp"].dt.to_period("M").astype(str).to_numpy()
    weeks = test["order_purchase_timestamp"].dt.to_period("W").astype(str).to_numpy()
    point, pairs = intervals(y, scores, months, weeks)
    for name, (pr_auc, caught) in {**exp15.EXPECTED_LIST, REGRESSION: EXPECTED_REGRESSION}.items():
        if abs(point[name]["pr_auc"] - pr_auc) > TOLERANCE or point[name]["caught"] != caught:
            raise AssertionError(f"{name} does not reproduce earlier test results: {point[name]}")
    print("Earlier test results reproduce", flush=True)
    pairs.to_csv(OUT / "test_intervals.csv", index=False)
    print(pairs.round(3).to_string(index=False))


if __name__ == "__main__":
    run()
