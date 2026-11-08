"""Check whether the Phase 2 50/50 mix holds up when the model is refitted each month.

Run from the repository root:
    python src/experiments/ranking_strategy/10_mix_monthly_refits/run.py

Declared in src/experiments/ranking_strategy/EVALUATION_CRITERIA.md. Refits use all history with known outcomes, as in experiment 05 and the
Phase 2 replay. The percentile scale is either refreshed at each refit or frozen from the first refit of a scored period.
The script stops if earlier results do not reproduce. Writes CSVs to data/outputs/experiments/ranking_strategy/10_mix_monthly_refits/.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[2]))

from olist.paths import EXPERIMENT_OUTPUTS  # noqa: E402

OUT = EXPERIMENT_OUTPUTS / "ranking_strategy" / HERE.name
OUT.mkdir(parents=True, exist_ok=True)
sys.path.insert(0, str(HERE.parent))

from task4_common import (  # noqa: E402
    BASELINE_CONFIG, LABEL, TOP_SHARE, evaluate, exp03, load_data, monthly_flags, saved_model_scores, shared, top_flags, validation_rows,
)


def load_run(name, folder):
    spec = importlib.util.spec_from_file_location(name, HERE.parent / folder / "run.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


exp05 = load_run("experiment_05", "05_recent_monthly_refits")
exp08 = load_run("experiment_08", "08_final_comparison")
exp04 = exp08.exp04

REFRESHED, FROZEN, SINGLE, MONTHLY_MODEL = "mix_refreshed", "mix_frozen", "mix_single_fit", "monthly_all_history"
PAIRS = [(REFRESHED, SINGLE), (REFRESHED, "slack_rule"), (FROZEN, SINGLE)]
SHORTFALL_RANKINGS = ["slack_rule", "model", SINGLE, REFRESHED, FROZEN]
TEST_START = exp05.TEST_START
EXPECTED_TEST = {SINGLE: (0.5167, 403), MONTHLY_MODEL: (0.4930, 389)}
EXPECTED_VALIDATION_CAUGHT = {SINGLE: 1342}
EXPECTED_HANDOVER_MONTH_CAUGHT = {MONTHLY_MODEL: 379}
TOLERANCE = 1e-3


def half_and_half(model_ref, model_values, rule_ref, rule_values):
    return 0.5 * shared.percentile(model_ref, model_values) + 0.5 * shared.percentile(rule_ref, rule_values)


def monthly_scores(data, rows, start):
    """The refitted model's scores and both mix versions for `rows`, using the latest refit before each handover."""
    table = data["table"]
    dates = exp05.refit_dates(start, rows["handover_time"].max())
    out = {name: np.full(len(rows), np.nan) for name in (MONTHLY_MODEL, REFRESHED, FROZEN)}
    first = None
    for begin, end in zip(dates, [*dates[1:], pd.Timestamp.max]):
        take = rows["handover_time"].ge(begin) & rows["handover_time"].lt(end)
        known = table[table["order_purchase_timestamp"].lt(begin) & table["handover_time"].lt(begin) & table["outcome_available_at"].lt(begin)]
        pipeline, columns = exp03.fit_config(BASELINE_CONFIG, known, begin, data["spec"], data["spec_base"], data["params"])
        model_ref = pipeline.predict_proba(known[columns])[:, 1]
        rule_ref = shared.rule_scores(known, known)
        if first is None:
            first = {"model": model_ref, "rule": rule_ref, "fit": known}
        if not take.any():
            continue
        where = np.flatnonzero(take.to_numpy())
        part = rows.loc[take]
        model_values = pipeline.predict_proba(part[columns])[:, 1]
        out[MONTHLY_MODEL][where] = model_values
        out[REFRESHED][where] = half_and_half(model_ref, model_values, rule_ref, shared.rule_scores(known, part))
        out[FROZEN][where] = half_and_half(first["model"], model_values, first["rule"], shared.rule_scores(first["fit"], part))
    for name, values in out.items():
        if np.isnan(values).any():
            raise AssertionError(f"{name}: {np.isnan(values).sum()} orders have no refit before their handover")
    return out


def single_fit_mix(fit_rows, rows, model_fit, model_rows):
    return half_and_half(model_fit, model_rows, shared.rule_scores(fit_rows, fit_rows), shared.rule_scores(fit_rows, rows))


def extra_intervals(y, scores, months, days, not_past):
    names = sorted({name for pair in PAIRS for name in pair})
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
    for a, b in PAIRS:
        for metric in point[a]:
            gaps = np.array([d[a][metric] - d[b][metric] for d in draws], dtype=float)
            rows.append({"a": a, "b": b, "metric": metric, "a_minus_b": float(point[a][metric] - point[b][metric]),
                         "ci_low": np.percentile(gaps, 2.5), "ci_high": np.percentile(gaps, 97.5)})
    values = pd.DataFrame(point).rename_axis("metric").reset_index()
    return values, pd.DataFrame(rows)


def shortfalls(periods):
    rows = []
    used = periods[~periods["period"].isin(exp08.EXCLUDED_FROM_SHORTFALLS)]
    for metric in exp08.SHORTFALL_METRICS:
        table = used.pivot(index="period", columns="ranking", values=metric)[SHORTFALL_RANKINGS]
        behind = table.rsub(table.max(axis=1), axis=0)
        rows += [{"metric": metric, "ranking": name, "largest_shortfall": behind[name].max(), "period_of_largest": behind[name].idxmax(),
                  "mean_shortfall": behind[name].mean()} for name in SHORTFALL_RANKINGS]
    return pd.DataFrame(rows)


def run():
    data = load_data()
    train, test = data["train"], data["test"]
    valid_scores = {name: [] for name in (REFRESHED, FROZEN, SINGLE, MONTHLY_MODEL)}
    periods = []
    for number, (fit_rows, rows) in enumerate(validation_rows(train, data["folds"]), start=1):
        model_fit, model_rows = exp04.phase2_scores(data, fit_rows, rows)
        scores = {**monthly_scores(data, rows, rows["handover_time"].min()), SINGLE: single_fit_mix(fit_rows, rows, model_fit, model_rows)}
        for name, values in scores.items():
            valid_scores[name].append(values)
        scores.update({"slack_rule": shared.rule_scores(fit_rows, rows), "model": model_rows})
        y = rows[LABEL].to_numpy().astype(bool)
        not_past = rows["past_promise_at_handover"].ne(1).to_numpy()
        periods += [exp08.period_row(f"W{number}", name, y, scores[name], top_flags(scores[name], TOP_SHARE), not_past) for name in SHORTFALL_RANKINGS]
    print("Validation scored", flush=True)

    test_scores = {**monthly_scores(data, test, TEST_START),
                   SINGLE: single_fit_mix(train, test, saved_model_scores(data, train), saved_model_scores(data, test))}
    all_scores = evaluate(OUT, data, valid_scores, test_scores, {})

    metrics = pd.read_csv(OUT / "test_metrics.csv")
    whole = metrics[metrics["period"].eq("test")].set_index("ranking")
    for name, (pr_auc, caught) in EXPECTED_TEST.items():
        if abs(whole.loc[name, "pr_auc"] - pr_auc) > TOLERANCE or int(whole.loc[name, "caught"]) != caught:
            raise AssertionError(f"{name} does not reproduce earlier test results")
    summary = pd.read_csv(OUT / "validation_summary.csv").set_index("ranking")
    for name, caught in EXPECTED_VALIDATION_CAUGHT.items():
        if int(summary.loc[name, "caught"]) != caught:
            raise AssertionError(f"{name} does not reproduce earlier validation results")
    y = test[LABEL].to_numpy().astype(bool)
    handover_months = test["handover_time"].dt.to_period("M").astype(str).to_numpy()
    for name, caught in EXPECTED_HANDOVER_MONTH_CAUGHT.items():
        if int((y & monthly_flags(all_scores[name], handover_months, TOP_SHARE)).sum()) != caught:
            raise AssertionError(f"{name} does not reproduce the Phase 2 replay")
    print("Earlier test and validation results reproduce", flush=True)

    not_past = test["past_promise_at_handover"].ne(1).to_numpy()
    months = test["order_purchase_timestamp"].dt.to_period("M").astype(str).to_numpy()
    days = test["handover_time"].dt.strftime("%Y-%m-%d").to_numpy()
    for month in np.unique(months):
        part = months == month
        periods += [exp08.period_row(month, name, y[part], all_scores[name][part], top_flags(all_scores[name][part], TOP_SHARE), not_past[part])
                    for name in SHORTFALL_RANKINGS]
    periods = pd.DataFrame(periods)
    behind = shortfalls(periods)
    periods.to_csv(OUT / "periods.csv", index=False)
    behind.to_csv(OUT / "period_shortfalls.csv", index=False)
    print(behind.round(4).to_string(index=False), flush=True)

    values, pairs = extra_intervals(y, all_scores, months, days, not_past)
    values.to_csv(OUT / "test_values.csv", index=False)
    pairs.to_csv(OUT / "test_pairs.csv", index=False)
    print(values.round(3).to_string(index=False))
    print(pairs.round(3).to_string(index=False))


if __name__ == "__main__":
    run()
