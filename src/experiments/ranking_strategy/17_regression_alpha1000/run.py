"""Rescore the regression ranking with the ridge penalty the group report now selects (alpha = 1000).

Run from the repository root:
    python src/experiments/ranking_strategy/17_regression_alpha1000/run.py

Declared in src/experiments/ranking_strategy/EVALUATION_CRITERIA.md. Experiments 01 and 16 used the Phase 2 handover ridge (alpha = 100);
the Phase 3 report selects alpha = 1000 at both regression stages. This reruns alpha = 100 first and stops unless it
reproduces experiment 01, then scores alpha = 1000 on the same windows, test months, shortfalls and weekly intervals as
experiment 16. Writes CSVs to data/outputs/experiments/ranking_strategy/17_regression_alpha1000/.
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

from task4_common import LABEL, TOP_SHARE, load_data, top_flags, validation_rows  # noqa: E402


def load_run(name, folder):
    spec = importlib.util.spec_from_file_location(name, EXPERIMENTS / folder / "run.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


exp01 = load_run("experiment_01", "01_ranking_baselines")
exp08 = load_run("experiment_08", "08_final_comparison")
exp15 = load_run("experiment_15", "15_weekly_intervals")
exp16 = load_run("experiment_16", "16_regression_comparator")

REGRESSION = exp16.REGRESSION
ALPHAS = {"alpha_100": 100.0, "alpha_1000": 1000.0}
EXPECTED_100 = {"validation_pr_auc": 0.323, "validation_caught": 1067, "test_pr_auc": 0.500, "test_caught": 411}
TOLERANCE = 1e-3


def regression_periods(data, alpha):
    """PR-AUC, ROC-AUC and top-10% catches of the regression ranking in each window, the test set and each test month."""
    model = ("ridge", {"alpha": alpha})
    rows = []
    for number, (fit_rows, valid_rows) in enumerate(validation_rows(data["train"], data["folds"]), start=1):
        score = exp01.rankings_for(fit_rows, valid_rows, data["spec"], model)[REGRESSION]
        rows.append(exp01.metric_row(f"W{number}", REGRESSION, valid_rows, score, top_flags(score, TOP_SHARE)))
    test = data["test"]
    score = exp01.rankings_for(data["train"], test, data["spec"], model)[REGRESSION]
    months = test["order_purchase_timestamp"].dt.to_period("M").astype(str).to_numpy()
    flags = exp15.monthly_flags(score, months, TOP_SHARE)
    rows.append(exp01.metric_row("test", REGRESSION, test, score, flags))
    for month in np.unique(months):
        part = months == month
        rows.append(exp01.metric_row(month, REGRESSION, test[part], score[part], flags[part]))
    return pd.DataFrame(rows), score


def summary_values(frame):
    windows = frame[frame["period"].str.fullmatch("W[1-5]")]
    test = frame[frame["period"].eq("test")].iloc[0]
    return {"validation_pr_auc": windows["pr_auc"].mean(), "validation_roc_auc": windows["roc_auc"].mean(),
            "validation_caught": int(windows["caught"].sum()), "test_pr_auc": test["pr_auc"], "test_caught": int(test["caught"])}


def shortfalls(frame):
    four = pd.read_csv(OUTPUTS / "13_deadline_comparison" / "periods.csv")
    part = frame[~frame["period"].eq("test")]
    regression = pd.DataFrame({"period": part["period"], "ranking": REGRESSION, "pr_auc": part["pr_auc"],
                               "pr_auc_not_past": part["pr_auc_not_past_promise"], "recall_top10": part["recall"]})
    periods = pd.concat([four, regression], ignore_index=True)
    used = periods[~periods["period"].isin(exp08.EXCLUDED_FROM_SHORTFALLS)]
    table = used.pivot(index="period", columns="ranking", values="pr_auc")[exp16.RANKINGS]
    behind = table.rsub(table.max(axis=1), axis=0)
    gaps = table.loc[[f"W{n}" for n in range(1, 6)], REGRESSION] - table.loc[[f"W{n}" for n in range(1, 6)], "model"]
    paired = {"gap_to_model": gaps.mean(), "paired_se": gaps.std(ddof=1) / np.sqrt(5), "windows_behind_model": int((gaps < 0).sum())}
    rows = [{"ranking": name, "largest_shortfall": behind[name].max(), "period_of_largest": behind[name].idxmax(),
             "periods_best": int(behind[name].eq(0).sum())} for name in exp16.RANKINGS]
    return table, pd.DataFrame(rows), paired


def run():
    data = load_data()
    results = {}
    for label, alpha in ALPHAS.items():
        frame, score = regression_periods(data, alpha)
        values = summary_values(frame)
        print(f"{label}: {values}", flush=True)
        if label == "alpha_100":
            for key, expected in EXPECTED_100.items():
                if abs(values[key] - expected) > (TOLERANCE if isinstance(expected, float) else 0):
                    raise AssertionError(f"alpha = 100 does not reproduce experiment 01: {key} = {values[key]}")
            print("alpha = 100 reproduces experiment 01", flush=True)
        results[label] = (frame, score, values)

    frame, regression_score, values = results["alpha_1000"]
    frame.to_csv(OUT / "regression_periods.csv", index=False)
    table, behind, paired = shortfalls(frame)
    table.to_csv(OUT / "period_pr_auc.csv")
    behind.to_csv(OUT / "period_shortfalls.csv", index=False)
    pd.DataFrame([{**values, **paired}]).to_csv(OUT / "summary.csv", index=False)
    pd.set_option("display.width", 200)
    print(table.round(3).to_string())
    print(behind.round(3).to_string(index=False))
    print(paired, flush=True)

    train, test = data["train"], data["test"]
    scores = {exp15.BLEND: exp15.exp10.monthly_scores(data, test, exp15.exp10.TEST_START)[exp15.BLEND],
              "slack_rule": exp15.shared.rule_scores(train, test), "model": exp15.saved_model_scores(data, test),
              exp15.DEADLINE: (test["handover_days"] - test[exp15.PROMISE]).to_numpy(dtype=float), REGRESSION: regression_score}
    y = test[LABEL].to_numpy().astype(bool)
    months = test["order_purchase_timestamp"].dt.to_period("M").astype(str).to_numpy()
    weeks = test["order_purchase_timestamp"].dt.to_period("W").astype(str).to_numpy()
    point, pairs = exp16.intervals(y, scores, months, weeks)
    for name, (pr_auc, caught) in exp15.EXPECTED_LIST.items():
        if abs(point[name]["pr_auc"] - pr_auc) > TOLERANCE or point[name]["caught"] != caught:
            raise AssertionError(f"{name} does not reproduce earlier test results: {point[name]}")
    pairs.to_csv(OUT / "test_intervals.csv", index=False)
    print(pairs.round(3).to_string(index=False))


if __name__ == "__main__":
    run()
