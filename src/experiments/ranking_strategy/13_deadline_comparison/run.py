"""Fill the remaining gaps for closest to deadline against the final strategy.

Run from the repository root:
    python src/experiments/ranking_strategy/13_deadline_comparison/run.py

Declared in src/experiments/ranking_strategy/EVALUATION_CRITERIA.md. Reads the saved results of experiments 01, 10 and 12, and stops if
earlier results do not reproduce. Writes CSVs to data/outputs/experiments/ranking_strategy/13_deadline_comparison/.
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


exp08 = load_run("experiment_08", "08_final_comparison")
exp10 = load_run("experiment_10", "10_mix_monthly_refits")
exp11 = load_run("experiment_11", "11_daily_rerank")
exp12 = load_run("experiment_12", "12_rerank_monthly_refits")

BLEND, DEADLINE = exp10.REFRESHED, "deadline"
LIST_PAIRS = [(BLEND, DEADLINE), ("slack_rule", DEADLINE), ("model", DEADLINE)]
A_DEADLINE, B_DEADLINE = "A_deadline", "B_deadline"
EARLIER = ["B_mix_refits", "B_mix", "A_mix", "B_rule"]
QUEUE_PAIRS = [("B_mix_refits", B_DEADLINE), ("B_mix_refits", A_DEADLINE), ("B_rule", B_DEADLINE), (B_DEADLINE, A_DEADLINE)]
LEAD_DAYS = exp11.LEAD_DAYS
SHORTFALL_RANKINGS = ["slack_rule", "model", BLEND, DEADLINE]
EXPECTED_TEST = {BLEND: (0.522, 411, 377), DEADLINE: (0.487, 359, 339), "slack_rule": (0.519, 398, 386), "model": (0.489, 377, 358)}
EXPECTED_SIMULATION = {"test": {"B_mix_refits": 256, "B_rule": 300}, "validation": {"B_mix_refits": 1287, "B_rule": 1429}}
TOLERANCE = 1e-3


def simulate_deadline(period, orders, start, end):
    handover = orders["handover_time"].to_numpy(dtype="datetime64[ns]")
    delivered = orders["outcome_available_at"].to_numpy(dtype="datetime64[ns]")
    purchase = orders["order_purchase_timestamp"].to_numpy(dtype="datetime64[ns]")
    promised_day = orders["promised_day"].to_numpy(dtype="datetime64[ns]")
    promised_days = orders[PROMISE].to_numpy(dtype=float)
    handover_day = orders["handover_time"].dt.normalize().to_numpy(dtype="datetime64[ns]")
    at_handover = (orders["handover_days"] - orders[PROMISE]).to_numpy(dtype=float)
    y = orders[LABEL].to_numpy().astype(bool)
    reviewed = {A_DEADLINE: np.zeros(len(orders), dtype=bool), B_DEADLINE: np.zeros(len(orders), dtype=bool)}
    rows = []
    for day in pd.date_range(start, end):
        batch = handover_day == day.to_datetime64()
        count = int(top_flags(np.zeros(int(batch.sum())), TOP_SHARE).sum()) if batch.any() else 0
        morning = (day + pd.Timedelta(days=1)).to_datetime64()
        review = (day + pd.Timedelta(days=1, hours=9)).to_datetime64()
        before_date = promised_day >= morning
        still_open = delivered > review
        days_left = promised_days - exp11.days_between(review, purchase)
        queue = (handover < morning) & still_open & before_date
        lists = {A_DEADLINE: (at_handover, batch & still_open & before_date), B_DEADLINE: (-days_left, queue)}
        for name, (score, pool) in lists.items():
            picked = exp11.choose(score, pool & ~reviewed[name], count)
            reviewed[name][picked] = True
            late = y[picked]
            rows.append({"period": period, "morning": str(day.date() + pd.Timedelta(days=1)), "version": name, "capacity": count,
                         "reviewed": len(picked), "caught_before_date": int(late.sum()),
                         "days_left_total": float(days_left[picked][late].sum()), "queue_size": int(queue.sum()),
                         **{f"caught_{days}plus_days": int((days_left[picked][late] >= days).sum()) for days in LEAD_DAYS}})
    return pd.DataFrame(rows)


def summarise(frame, label, versions):
    rows = []
    for name in versions:
        part = frame[frame["version"].eq(name)]
        caught = int(part["caught_before_date"].sum())
        rows.append({"scope": label, "version": name, "reviewed": int(part["reviewed"].sum()), "caught_before_date": caught,
                     "precision": caught / max(int(part["reviewed"].sum()), 1), "mean_days_left": part["days_left_total"].sum() / max(caught, 1),
                     **{f"caught_{days}plus_days": int(part[f"caught_{days}plus_days"].sum()) for days in LEAD_DAYS}})
    return rows


def queue_intervals(test_frame):
    rng = np.random.default_rng(shared.SEED)
    size = test_frame["morning"].nunique()
    picks = [rng.integers(0, size, size) for _ in range(shared.BOOTSTRAPS)]
    rows = []
    for metric in ["caught_before_date", *[f"caught_{days}plus_days" for days in LEAD_DAYS]]:
        table = test_frame.pivot(index="morning", columns="version", values=metric)
        for a, b in QUEUE_PAIRS:
            gap = (table[a] - table[b]).to_numpy(dtype=float)
            draws = np.array([gap[p].sum() for p in picks])
            rows.append({"metric": metric, "a": a, "b": b, "a_minus_b": gap.sum(), "ci_low": np.percentile(draws, 2.5), "ci_high": np.percentile(draws, 97.5)})
    return pd.DataFrame(rows)


def list_intervals(y, scores, months, days, not_past):
    names = [BLEND, "slack_rule", "model", DEADLINE]
    rng = np.random.default_rng(shared.SEED)
    groups = [np.flatnonzero(months == m) for m in np.unique(months)]
    point = {name: exp08.test_values(y, scores[name], months, days, not_past) for name in names}
    draws = []
    for number in range(shared.BOOTSTRAPS):
        index = np.concatenate([rng.choice(g, len(g)) for g in groups])
        draws.append({name: exp08.test_values(y[index], scores[name][index], months[index], days[index], not_past[index]) for name in names})
        if (number + 1) % 100 == 0:
            print(f"Resampling round {number + 1} of {shared.BOOTSTRAPS}", flush=True)
    rows = []
    for a, b in LIST_PAIRS:
        for metric in point[a]:
            gaps = np.array([d[a][metric] - d[b][metric] for d in draws], dtype=float)
            rows.append({"a": a, "b": b, "metric": metric, "a_minus_b": float(point[a][metric] - point[b][metric]),
                         "ci_low": np.percentile(gaps, 2.5), "ci_high": np.percentile(gaps, 97.5)})
    return pd.DataFrame(point).rename_axis("metric").reset_index(), pd.DataFrame(rows)


def shortfalls():
    saved = pd.read_csv(OUTPUTS / "10_mix_monthly_refits" / "periods.csv")
    saved = saved[saved["ranking"].isin([BLEND, "slack_rule", "model"])][["period", "ranking", *exp08.SHORTFALL_METRICS]]
    first = pd.read_csv(OUTPUTS / "01_ranking_baselines" / "metrics.csv")
    first = first[first["ranking"].eq(DEADLINE) & ~first["period"].eq("test")]
    deadline = pd.DataFrame({"period": first["period"], "ranking": DEADLINE, "pr_auc": first["pr_auc"],
                             "pr_auc_not_past": first["pr_auc_not_past_promise"], "recall_top10": first["recall"]})
    periods = pd.concat([saved, deadline], ignore_index=True)
    used = periods[~periods["period"].isin(exp08.EXCLUDED_FROM_SHORTFALLS)]
    rows = []
    for metric in exp08.SHORTFALL_METRICS:
        table = used.pivot(index="period", columns="ranking", values=metric)[SHORTFALL_RANKINGS]
        behind = table.rsub(table.max(axis=1), axis=0)
        rows += [{"metric": metric, "ranking": name, "largest_shortfall": behind[name].max(), "period_of_largest": behind[name].idxmax(),
                  "mean_shortfall": behind[name].mean(), "periods_best": int(behind[name].eq(0).sum())} for name in SHORTFALL_RANKINGS]
    return periods, pd.DataFrame(rows)


def run():
    data = load_data(exp11.add_promised_day)
    table, train = data["table"], data["train"]
    saved = pd.read_csv(OUTPUTS / "12_rerank_monthly_refits" / "mornings.csv", dtype={"morning": str})
    frames = []
    for number, (_, rows) in enumerate(validation_rows(train, data["folds"]), start=1):
        start = rows["order_purchase_timestamp"].min().normalize() + pd.Timedelta(days=1)
        end = rows["order_purchase_timestamp"].max().normalize()
        frames.append(simulate_deadline(f"W{number}", exp11.span_orders(table, start, end), start, end))
    start = pd.Timestamp(data["spec"]["primary_cutoff"])
    end = data["test"]["order_purchase_timestamp"].max().normalize()
    frames.append(simulate_deadline("test", exp11.span_orders(table, start, end), start, end))
    deadline_frame = pd.concat(frames, ignore_index=True)

    capacity = saved[saved["version"].eq("B_rule")].set_index(["period", "morning"])["capacity"]
    for name in (A_DEADLINE, B_DEADLINE):
        got = deadline_frame[deadline_frame["version"].eq(name)].set_index(["period", "morning"])["capacity"]
        if not got.sort_index().equals(capacity.sort_index()):
            raise AssertionError(f"{name} capacity differs from experiment 12")
    combined = pd.concat([saved, deadline_frame], ignore_index=True)
    versions = [*EARLIER, A_DEADLINE, B_DEADLINE]
    test_frame = combined[combined["period"].eq("test")]
    validation = combined[~combined["period"].eq("test")]
    rows = summarise(validation, "validation", versions) + summarise(test_frame, "test", versions)
    for period, part in validation.groupby("period"):
        rows += summarise(part, period, versions)
    for month, part in test_frame.groupby(test_frame["morning"].str[:7]):
        rows += summarise(part, month, versions)
    summary = pd.DataFrame(rows)
    for scope, expected in EXPECTED_SIMULATION.items():
        got = summary[summary["scope"].eq(scope)].set_index("version")["caught_before_date"]
        for name, value in expected.items():
            if int(got[name]) != value:
                raise AssertionError(f"{scope} {name} does not reproduce experiment 12: {got[name]}")
    print("Experiment 12 reproduces", flush=True)
    deadline_frame.to_csv(OUT / "deadline_mornings.csv", index=False)
    summary.to_csv(OUT / "simulation_summary.csv", index=False)
    queue = queue_intervals(test_frame)
    queue.to_csv(OUT / "simulation_intervals.csv", index=False)
    periods, behind = shortfalls()
    periods.to_csv(OUT / "periods.csv", index=False)
    behind.to_csv(OUT / "period_shortfalls.csv", index=False)
    print(summary[summary["scope"].isin(["test", "validation"])].round(3).to_string(index=False))
    print(summary[summary["version"].isin(["B_mix_refits", "B_rule", A_DEADLINE, B_DEADLINE]) & ~summary["scope"].isin(["test", "validation"])][["scope", "version", "caught_before_date", "caught_3plus_days", "mean_days_left"]].round(2).to_string(index=False))
    print(queue.round(1).to_string(index=False))
    print(behind.round(4).to_string(index=False), flush=True)

    test = data["test"]
    scores = {BLEND: exp10.monthly_scores(data, test, exp10.TEST_START)[BLEND], "slack_rule": shared.rule_scores(train, test),
              "model": saved_model_scores(data, test), DEADLINE: (test["handover_days"] - test[PROMISE]).to_numpy(dtype=float)}
    y = test[LABEL].to_numpy().astype(bool)
    not_past = test["past_promise_at_handover"].ne(1).to_numpy()
    months = test["order_purchase_timestamp"].dt.to_period("M").astype(str).to_numpy()
    days = test["handover_time"].dt.strftime("%Y-%m-%d").to_numpy()
    values, pairs = list_intervals(y, scores, months, days, not_past)
    indexed = values.set_index("metric")
    for name, (pr_auc, monthly, daily) in EXPECTED_TEST.items():
        got = (indexed.loc["monthly_10%_caught", name], indexed.loc["daily_10%_caught", name])
        if abs(indexed.loc["pr_auc", name] - pr_auc) > TOLERANCE or got != (monthly, daily):
            raise AssertionError(f"{name} does not reproduce earlier test results: {indexed.loc['pr_auc', name]}, {got}")
    print("Earlier test results reproduce", flush=True)
    values.to_csv(OUT / "test_values.csv", index=False)
    pairs.to_csv(OUT / "test_pairs.csv", index=False)
    print(values.round(3).to_string(index=False))
    print(pairs.round(3).to_string(index=False))


if __name__ == "__main__":
    run()
