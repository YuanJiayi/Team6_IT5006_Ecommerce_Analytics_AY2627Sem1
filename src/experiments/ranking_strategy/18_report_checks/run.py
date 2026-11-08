"""Make two report numbers reproducible: review-list quality (precision, recall, F1, MCC) and a weekly refit interval.

Run from the repository root:
    python src/experiments/ranking_strategy/18_report_checks/run.py

Declared in src/experiments/ranking_strategy/EVALUATION_CRITERIA.md. Adds no candidates and changes no scores: it rescores the five rankings
of experiment 17 and stops unless their late orders caught match the report. Writes CSVs to data/outputs/experiments/ranking_strategy/18_report_checks/.
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

from task4_common import LABEL, PROMISE, TOP_SHARE, load_data, monthly_flags, saved_model_scores, shared, top_flags, validation_rows  # noqa: E402


def load_run(name, folder):
    spec = importlib.util.spec_from_file_location(name, EXPERIMENTS / folder / "run.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


exp01 = load_run("experiment_01", "01_ranking_baselines")
exp15 = load_run("experiment_15", "15_weekly_intervals")
exp10 = exp15.exp10

REGRESSION_MODEL = ("ridge", {"alpha": 1000.0})
RANKINGS = ["deadline", "regression_gap", "slack_rule", "model", exp10.REFRESHED]
EXPECTED_CAUGHT = {"validation": {"deadline": 596, "regression_gap": 1122, "slack_rule": 1099, "model": 1374, exp10.REFRESHED: 1341},
                   "test": {"deadline": 359, "regression_gap": 412, "slack_rule": 398, "model": 377, exp10.REFRESHED: 411,
                            exp10.SINGLE: 403}}
# Table 17 of the report (precision, recall, F1, MCC), checked to three decimals.
REPORTED = {"validation": {"deadline": (0.157, 0.147, 0.152, 0.054), "regression_gap": (0.295, 0.277, 0.286, 0.204),
                           "slack_rule": (0.289, 0.271, 0.280, 0.197), "model": (0.361, 0.339, 0.350, 0.275),
                           exp10.REFRESHED: (0.353, 0.331, 0.341, 0.266)},
            "test": {"deadline": (0.187, 0.534, 0.277, 0.275), "regression_gap": (0.214, 0.613, 0.317, 0.325),
                     "slack_rule": (0.207, 0.592, 0.307, 0.312), "model": (0.196, 0.561, 0.290, 0.292),
                     exp10.REFRESHED: (0.214, 0.612, 0.317, 0.324)}}


def list_quality(period, name, y, flags):
    """Precision, recall, F1 and MCC of a review list, with the list counted as predicted late."""
    n, listed, late = len(y), int(flags.sum()), int(y.sum())
    tp = int((y & flags).sum())
    fp, fn = listed - tp, late - tp
    tn = n - tp - fp - fn
    precision, recall = tp / listed, tp / late
    mcc = (tp * tn - fp * fn) / np.sqrt(float(listed) * (n - listed) * late * (n - late))
    return {"period": period, "ranking": name, "orders": n, "late": late, "listed": listed, "caught": tp, "precision": precision,
            "recall": recall, "f1": 2 * precision * recall / (precision + recall), "mcc": mcc}


def validation_lists(data):
    ys, flags = [], {name: [] for name in RANKINGS}
    for fit_rows, rows in validation_rows(data["train"], data["folds"]):
        scores = {"slack_rule": shared.rule_scores(fit_rows, rows), "model": exp10.exp04.phase2_scores(data, fit_rows, rows)[1],
                  exp10.REFRESHED: exp10.monthly_scores(data, rows, rows["handover_time"].min())[exp10.REFRESHED],
                  "deadline": (rows["handover_days"] - rows[PROMISE]).to_numpy(dtype=float),
                  "regression_gap": exp01.rankings_for(fit_rows, rows, data["spec"], REGRESSION_MODEL)["regression_gap"]}
        ys.append(rows[LABEL].to_numpy().astype(bool))
        for name in RANKINGS:
            flags[name].append(top_flags(scores[name], TOP_SHARE))
        print(f"Window {len(ys)} scored", flush=True)
    y = np.concatenate(ys)
    return [list_quality("validation", name, y, np.concatenate(flags[name])) for name in RANKINGS]


def test_scores(data):
    train, test = data["train"], data["test"]
    return {"slack_rule": shared.rule_scores(train, test), "model": saved_model_scores(data, test),
            exp10.REFRESHED: exp10.monthly_scores(data, test, exp10.TEST_START)[exp10.REFRESHED],
            exp10.SINGLE: exp10.single_fit_mix(train, test, saved_model_scores(data, train), saved_model_scores(data, test)),
            "deadline": (test["handover_days"] - test[PROMISE]).to_numpy(dtype=float),
            "regression_gap": exp01.rankings_for(train, test, data["spec"], REGRESSION_MODEL)["regression_gap"]}


def refit_interval(y, scores, months, weeks):
    a, b = exp10.REFRESHED, exp10.SINGLE
    point = {name: exp15.list_values(y, scores[name], months) for name in (a, b)}
    blocks, picks = exp15.week_blocks(weeks)
    draws = []
    for pick in picks:
        index = np.concatenate([blocks[k] for k in pick])
        draws.append({name: exp15.list_values(y[index], scores[name][index], months[index]) for name in (a, b)})
    rows = []
    for metric in ("caught", "pr_auc"):
        gaps = np.array([d[a][metric] - d[b][metric] for d in draws], dtype=float)
        rows.append({"a": a, "b": b, "metric": metric, "a_minus_b": point[a][metric] - point[b][metric],
                     "ci_low": np.percentile(gaps, 2.5), "ci_high": np.percentile(gaps, 97.5)})
    return pd.DataFrame(rows)


def run():
    data = load_data()
    rows = validation_lists(data)

    test = data["test"]
    scores = test_scores(data)
    y = test[LABEL].to_numpy().astype(bool)
    months = test["order_purchase_timestamp"].dt.to_period("M").astype(str).to_numpy()
    weeks = test["order_purchase_timestamp"].dt.to_period("W").astype(str).to_numpy()
    rows += [list_quality("test", name, y, monthly_flags(scores[name], months, TOP_SHARE)) for name in RANKINGS]
    quality = pd.DataFrame(rows)

    single_caught = int((y & monthly_flags(scores[exp10.SINGLE], months, TOP_SHARE)).sum())
    caught = {(r["period"], r["ranking"]): r["caught"] for r in rows} | {("test", exp10.SINGLE): single_caught}
    for period, expected in EXPECTED_CAUGHT.items():
        for name, value in expected.items():
            if caught[(period, name)] != value:
                raise AssertionError(f"{period} {name}: {caught[(period, name)]} late caught, report has {value}")
    print("Late orders caught reproduce the report", flush=True)

    quality.to_csv(OUT / "list_quality.csv", index=False)
    pd.set_option("display.width", 200)
    print(quality.round(3).to_string(index=False))

    interval = refit_interval(y, scores, months, weeks)
    interval.to_csv(OUT / "refit_interval.csv", index=False)
    print(interval.round(4).to_string(index=False))

    mismatches = [(r["period"], r["ranking"], m, round(r[m], 3), v)
                  for r in rows for m, v in zip(("precision", "recall", "f1", "mcc"), REPORTED[r["period"]][r["ranking"]])
                  if abs(round(r[m], 3) - v) > 1e-9]
    if mismatches:
        raise AssertionError(f"List quality differs from the report's Table 17: {mismatches}")
    print("List quality matches the report's Table 17", flush=True)


if __name__ == "__main__":
    run()
