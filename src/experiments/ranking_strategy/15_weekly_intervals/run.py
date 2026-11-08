"""Recompute the report's test intervals with weekly blocks and 2,000 resamples, to match the team's convention.

Run from the repository root:
    python src/experiments/ranking_strategy/15_weekly_intervals/run.py

Declared in src/experiments/ranking_strategy/EVALUATION_CRITERIA.md. Adds no rankings and changes no scores: it rescores the four test
rankings compared in experiment 13 (the blend with monthly refits from experiment 10) and reads the saved daily replays of
experiments 12 and 13, and stops if their point results do not reproduce. Writes intervals.csv to data/outputs/experiments/ranking_strategy/15_weekly_intervals/.
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

from task4_common import LABEL, PROMISE, TOP_SHARE, load_data, monthly_flags, saved_model_scores, shared  # noqa: E402

_spec = importlib.util.spec_from_file_location("experiment_10", EXPERIMENTS / "10_mix_monthly_refits" / "run.py")
exp10 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(exp10)

BOOTSTRAPS, SEED = 2000, 42
BLEND, DEADLINE = exp10.REFRESHED, "deadline"
LIST_PAIRS = [(BLEND, "slack_rule"), (BLEND, "model"), (BLEND, DEADLINE), ("slack_rule", "model"), ("slack_rule", DEADLINE)]
QUEUE_PAIRS = [("B_mix_refits", "B_rule"), ("B_mix_refits", "B_deadline"), ("B_mix_refits", "A_mix_refits"), ("B_mix_refits", "B_mix")]
QUEUE_COUNTS = ["caught_before_date", "caught_1plus_days", "caught_3plus_days", "caught_5plus_days"]
EXPECTED_LIST = {BLEND: (0.522, 411), "slack_rule": (0.519, 398), "model": (0.489, 377), DEADLINE: (0.487, 359)}
EXPECTED_QUEUE = {"B_mix_refits": 256, "B_rule": 300, "B_deadline": 481, "A_mix_refits": 241, "B_mix": 250}
TOLERANCE = 1e-3


def week_blocks(weeks):
    """Row positions for each week, and 2,000 draws of weeks with replacement."""
    keys = np.unique(weeks)
    blocks = [np.flatnonzero(weeks == k) for k in keys]
    rng = np.random.default_rng(SEED)
    return blocks, [rng.integers(0, len(keys), len(keys)) for _ in range(BOOTSTRAPS)]


def list_values(y, score, months):
    return {"pr_auc": shared.average_precision_score(y, score), "caught": int((y & monthly_flags(score, months, TOP_SHARE)).sum())}


def list_intervals(y, scores, months, weeks):
    point = {name: list_values(y, s, months) for name, s in scores.items()}
    blocks, picks = week_blocks(weeks)
    draws = []
    for number, pick in enumerate(picks, start=1):
        index = np.concatenate([blocks[k] for k in pick])
        draws.append({name: list_values(y[index], s[index], months[index]) for name, s in scores.items()})
        if number % 200 == 0:
            print(f"Resampling round {number} of {BOOTSTRAPS}", flush=True)
    rows = []
    for a, b in LIST_PAIRS:
        for metric in ("pr_auc", "caught"):
            gaps = np.array([d[a][metric] - d[b][metric] for d in draws], dtype=float)
            rows.append({"scope": "list", "a": a, "b": b, "metric": metric, "a_minus_b": point[a][metric] - point[b][metric],
                         "ci_low": np.percentile(gaps, 2.5), "ci_high": np.percentile(gaps, 97.5)})
    return point, rows


def queue_intervals(test_frame):
    test_frame = test_frame.assign(week=pd.to_datetime(test_frame["morning"]).dt.to_period("W").astype(str))
    weekly = test_frame.groupby(["week", "version"])[[*QUEUE_COUNTS, "days_left_total"]].sum()
    keys = weekly.index.get_level_values("week").unique()
    rng = np.random.default_rng(SEED)
    picks = [rng.integers(0, len(keys), len(keys)) for _ in range(BOOTSTRAPS)]
    table = {name: weekly.xs(name, level="version").reindex(keys, fill_value=0).to_numpy(dtype=float) for name in EXPECTED_QUEUE}
    columns = [*QUEUE_COUNTS, "days_left_total"]
    caught, days = columns.index("caught_before_date"), columns.index("days_left_total")
    rows = []
    for a, b in QUEUE_PAIRS:
        for metric in [*QUEUE_COUNTS, "mean_days_left"]:
            def value(name, pick=slice(None)):
                part = table[name][pick].sum(axis=0)
                return part[days] / part[caught] if metric == "mean_days_left" else part[columns.index(metric)]
            gaps = np.array([value(a, p) - value(b, p) for p in picks])
            rows.append({"scope": "daily_queue", "a": a, "b": b, "metric": metric, "a_minus_b": value(a) - value(b),
                         "ci_low": np.percentile(gaps, 2.5), "ci_high": np.percentile(gaps, 97.5)})
    return rows, len(keys)


def run():
    saved = pd.concat([pd.read_csv(OUTPUTS / "12_rerank_monthly_refits" / "mornings.csv", dtype={"morning": str}),
                       pd.read_csv(OUTPUTS / "13_deadline_comparison" / "deadline_mornings.csv", dtype={"morning": str})])
    test_frame = saved[saved["period"].eq("test")]
    got = test_frame.groupby("version")["caught_before_date"].sum()
    for name, expected in EXPECTED_QUEUE.items():
        if int(got[name]) != expected:
            raise AssertionError(f"{name} does not reproduce the daily replay: {got[name]}")
    queue_rows, queue_weeks = queue_intervals(test_frame)
    print(f"Daily replay reproduces; {queue_weeks} weeks of mornings", flush=True)

    data = load_data()
    train, test = data["train"], data["test"]
    scores = {BLEND: exp10.monthly_scores(data, test, exp10.TEST_START)[BLEND], "slack_rule": shared.rule_scores(train, test),
              "model": saved_model_scores(data, test), DEADLINE: (test["handover_days"] - test[PROMISE]).to_numpy(dtype=float)}
    y = test[LABEL].to_numpy().astype(bool)
    months = test["order_purchase_timestamp"].dt.to_period("M").astype(str).to_numpy()
    weeks = test["order_purchase_timestamp"].dt.to_period("W").astype(str).to_numpy()
    point, list_rows = list_intervals(y, scores, months, weeks)
    for name, (pr_auc, caught) in EXPECTED_LIST.items():
        if abs(point[name]["pr_auc"] - pr_auc) > TOLERANCE or point[name]["caught"] != caught:
            raise AssertionError(f"{name} does not reproduce earlier test results: {point[name]}")
    print(f"Test lists reproduce; {len(np.unique(weeks))} purchase weeks", flush=True)

    intervals = pd.DataFrame(list_rows + queue_rows)
    intervals.to_csv(OUT / "intervals.csv", index=False)
    pd.set_option("display.width", 200)
    print(intervals.round(3).to_string(index=False))


if __name__ == "__main__":
    run()
