"""Test the recommended setup together: the 50/50 mix, re-ranked each morning, with monthly refits.

Run from the repository root:
    python Phase3_Winnie/experiments/12_rerank_monthly_refits/run.py

Declared in Phase3_Winnie/EVALUATION_CRITERIA.md. Combines the simulation in experiment 11 with the refits in experiment 10.
The script stops if experiment 11's versions do not reproduce. Writes CSVs next to this file.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

from task4_common import BASELINE_CONFIG, LABEL, PROMISE, TOP_SHARE, exp03, load_data, saved_model_scores, shared, top_flags, validation_rows  # noqa: E402


def load_run(name, folder):
    spec = importlib.util.spec_from_file_location(name, HERE.parent / folder / "run.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


exp05 = load_run("experiment_05", "05_recent_monthly_refits")
exp11 = load_run("experiment_11", "11_daily_rerank")
exp04 = exp11.exp04

A_REFIT, B_REFIT = "A_mix_refits", "B_mix_refits"
REFERENCES = ["A_mix", "B_mix", "B_rule"]
VERSIONS = [*REFERENCES, A_REFIT, B_REFIT]
PAIRS = [(B_REFIT, "B_mix"), (A_REFIT, "A_mix"), (B_REFIT, A_REFIT), (B_REFIT, "B_rule")]
LEAD_DAYS = exp11.LEAD_DAYS
EXPECTED_11 = {"test": {"A_mix": 232, "B_mix": 250, "B_rule": 300}, "validation": {"A_mix": 1123, "B_mix": 1287, "B_rule": 1429}}


def refit_parts(data, orders, start):
    """Model percentile and slack score at handover from the latest refit, and each refit's rule scale."""
    table = data["table"]
    dates = exp05.refit_dates(start, orders["handover_time"].max())
    model_pct, rule_scores, handover_mix = (np.full(len(orders), np.nan) for _ in range(3))
    scales = []
    for begin, end in zip(dates, [*dates[1:], pd.Timestamp.max]):
        known = table[table["order_purchase_timestamp"].lt(begin) & table["handover_time"].lt(begin) & table["outcome_available_at"].lt(begin)]
        pipeline, columns = exp03.fit_config(BASELINE_CONFIG, known, begin, data["spec"], data["spec_base"], data["params"])
        rule_ref = shared.rule_scores(known, known)
        scales.append((begin.to_datetime64(), rule_ref))
        take = orders["handover_time"].ge(begin) & orders["handover_time"].lt(end)
        if not take.any():
            continue
        where = np.flatnonzero(take.to_numpy())
        part = orders.loc[take]
        model_pct[where] = shared.percentile(pipeline.predict_proba(known[columns])[:, 1], pipeline.predict_proba(part[columns])[:, 1])
        rule_scores[where] = shared.rule_scores(known, part)
        handover_mix[where] = 0.5 * model_pct[where] + 0.5 * shared.percentile(rule_ref, rule_scores[where])
    if np.isnan(handover_mix).any():
        raise AssertionError(f"{np.isnan(handover_mix).sum()} orders have no refit before their handover")
    return model_pct, rule_scores, handover_mix, scales


def simulate_refits(period, data, orders, start, end):
    model_pct, rule_scores, handover_mix, scales = refit_parts(data, orders, start)
    handover = orders["handover_time"].to_numpy(dtype="datetime64[ns]")
    delivered = orders["outcome_available_at"].to_numpy(dtype="datetime64[ns]")
    purchase = orders["order_purchase_timestamp"].to_numpy(dtype="datetime64[ns]")
    promised_day = orders["promised_day"].to_numpy(dtype="datetime64[ns]")
    promised_days = orders[PROMISE].to_numpy(dtype=float)
    handover_day = orders["handover_time"].dt.normalize().to_numpy(dtype="datetime64[ns]")
    y = orders[LABEL].to_numpy().astype(bool)
    reviewed = {A_REFIT: np.zeros(len(orders), dtype=bool), B_REFIT: np.zeros(len(orders), dtype=bool)}
    rows = []
    for day in pd.date_range(start, end):
        batch = handover_day == day.to_datetime64()
        count = int(top_flags(np.zeros(int(batch.sum())), TOP_SHARE).sum()) if batch.any() else 0
        morning = (day + pd.Timedelta(days=1)).to_datetime64()
        review = (day + pd.Timedelta(days=1, hours=9)).to_datetime64()
        before_date = promised_day >= morning
        still_open = delivered > review
        days_left = promised_days - exp11.days_between(review, purchase)
        updated_rule = -np.minimum(-rule_scores, days_left)
        rule_ref = [ref for date, ref in scales if date <= review][-1]
        updated_mix = 0.5 * model_pct + 0.5 * shared.percentile(rule_ref, updated_rule)
        queue = (handover < morning) & still_open & before_date
        lists = {A_REFIT: (handover_mix, batch & still_open & before_date), B_REFIT: (updated_mix, queue)}
        for name, (score, pool) in lists.items():
            picked = exp11.choose(score, pool & ~reviewed[name], count)
            reviewed[name][picked] = True
            late = y[picked]
            rows.append({"period": period, "morning": str(day.date() + pd.Timedelta(days=1)), "version": name, "capacity": count,
                         "reviewed": len(picked), "caught_before_date": int(late.sum()),
                         "days_left_total": float(days_left[picked][late].sum()), "queue_size": int(queue.sum()),
                         **{f"caught_{days}plus_days": int((days_left[picked][late] >= days).sum()) for days in LEAD_DAYS}})
    return pd.DataFrame(rows)


def summarise(frame, label):
    rows = []
    for name in VERSIONS:
        part = frame[frame["version"].eq(name)]
        caught = int(part["caught_before_date"].sum())
        rows.append({"scope": label, "version": name, "capacity": int(part["capacity"].sum()), "reviewed": int(part["reviewed"].sum()), "caught_before_date": caught,
                     "precision": caught / max(int(part["reviewed"].sum()), 1), "mean_days_left": part["days_left_total"].sum() / max(caught, 1),
                     **{f"caught_{days}plus_days": int(part[f"caught_{days}plus_days"].sum()) for days in LEAD_DAYS}})
    return rows


def morning_intervals(test_frame):
    rng = np.random.default_rng(shared.SEED)
    size = test_frame["morning"].nunique()
    picks = [rng.integers(0, size, size) for _ in range(shared.BOOTSTRAPS)]
    rows = []
    for metric in ["caught_before_date", *[f"caught_{days}plus_days" for days in LEAD_DAYS]]:
        table = test_frame.pivot(index="morning", columns="version", values=metric)
        for a, b in PAIRS:
            gap = (table[a] - table[b]).to_numpy(dtype=float)
            draws = np.array([gap[p].sum() for p in picks])
            rows.append({"metric": metric, "a": a, "b": b, "a_minus_b": gap.sum(), "ci_low": np.percentile(draws, 2.5), "ci_high": np.percentile(draws, 97.5)})
    return pd.DataFrame(rows)


def run():
    data = load_data(exp11.add_promised_day)
    table, train = data["table"], data["train"]
    frames = []
    for number, (fit_rows, rows) in enumerate(validation_rows(train, data["folds"]), start=1):
        start = rows["order_purchase_timestamp"].min().normalize() + pd.Timedelta(days=1)
        end = rows["order_purchase_timestamp"].max().normalize()
        orders = exp11.span_orders(table, start, end)
        model_fit, model_scores = exp04.phase2_scores(data, fit_rows, orders)
        single, _ = exp11.simulate(f"W{number}", orders, start, end, model_scores, shared.rule_scores(fit_rows, orders), model_fit, shared.rule_scores(fit_rows, fit_rows))
        frames += [single[single["version"].isin(REFERENCES)], simulate_refits(f"W{number}", data, orders, start, end)]
        print(f"W{number} simulated", flush=True)
    validation = pd.concat(frames, ignore_index=True)

    start = pd.Timestamp(data["spec"]["primary_cutoff"])
    end = data["test"]["order_purchase_timestamp"].max().normalize()
    orders = exp11.span_orders(table, start, end)
    single, _ = exp11.simulate("test", orders, start, end, saved_model_scores(data, orders), shared.rule_scores(train, orders),
                               saved_model_scores(data, train), shared.rule_scores(train, train))
    test_frame = pd.concat([single[single["version"].isin(REFERENCES)], simulate_refits("test", data, orders, start, end)], ignore_index=True)

    rows = summarise(validation, "validation")
    for period, part in validation.groupby("period"):
        rows += summarise(part, period)
    rows += summarise(test_frame, "test")
    for month, part in test_frame.groupby(test_frame["morning"].str[:7]):
        rows += summarise(part, month)
    summary = pd.DataFrame(rows)
    for scope, expected in EXPECTED_11.items():
        got = summary[summary["scope"].eq(scope)].set_index("version")["caught_before_date"]
        for name, value in expected.items():
            if int(got[name]) != value:
                raise AssertionError(f"{scope} {name} does not reproduce experiment 11: {got[name]}")
    print("Experiment 11 reproduces", flush=True)
    intervals = morning_intervals(test_frame)

    pd.concat([validation, test_frame], ignore_index=True).to_csv(HERE / "mornings.csv", index=False)
    summary.to_csv(HERE / "summary.csv", index=False)
    intervals.to_csv(HERE / "test_intervals.csv", index=False)

    print(summary.round(3).to_string(index=False))
    print(intervals.round(1).to_string(index=False))
    main = intervals[intervals["metric"].eq("caught_before_date") & intervals["a"].eq(B_REFIT) & intervals["b"].eq("B_mix")].iloc[0]
    print(f"Declared decision, recommendation changes: {bool(main['ci_high'] < 0)}", flush=True)


if __name__ == "__main__":
    run()
