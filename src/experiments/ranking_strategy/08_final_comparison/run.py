"""Compare the four remaining rankings side by side, adding no candidates and no tuning.

Run from the repository root:
    python src/experiments/ranking_strategy/08_final_comparison/run.py

Rankings: remaining-slack rule, Phase 2 model, Phase 2 50/50 mix, relative changing ratio. Reports what
src/experiments/ranking_strategy/EVALUATION_CRITERIA.md declares: direct comparisons, review capacity, orders not past their date, daily
lists and performance across periods. The script stops if earlier results do not reproduce. Writes CSVs to data/outputs/experiments/ranking_strategy/08_final_comparison/.
"""

from __future__ import annotations

import importlib.util
import itertools
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

from task4_common import LABEL, load_data, monthly_flags, saved_model_scores, shared, top_flags, validation_rows  # noqa: E402

_spec = importlib.util.spec_from_file_location("experiment_04", HERE.parent / "04_mix_ratios" / "run.py")
exp04 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(exp04)
exp07 = exp04.exp07

RANKINGS = ["slack_rule", "model", "phase2_w0.5", "relative_changing"]
SHARES = [0.01, 0.02, 0.05, 0.10, 0.20]
TOP_ORDERS = [50, 100, 200, 500, 1000]
SHORTFALL_METRICS = ["pr_auc", "pr_auc_not_past", "recall_top10"]
EXCLUDED_FROM_SHORTFALLS = {"2018-05"}
TOLERANCE = 1e-3
EXPECTED_TEST = {"slack_rule": (0.5194, 398, 386), "model": (0.4887, 377, 358), "phase2_w0.5": (0.5167, 403, 373), "relative_changing": (0.518, 401, 393)}
EXPECTED_VALIDATION_CAUGHT = {"slack_rule": 1099, "model": 1374, "phase2_w0.5": 1342, "relative_changing": 1322}
EXPECTED_TOP_200 = {"slack_rule": 88, "phase2_w0.5": 70}


def scores_for(data, spec_relative, fit_rows, rows, model_fit, model_rows):
    relative_fit, relative_rows, _ = exp04.relative_scores(data, spec_relative, fit_rows, rows)
    phase2_pct, rule_pct = exp04.percentiles(data, fit_rows, rows, model_fit, model_rows)
    relative_pct, _ = exp04.percentiles(data, fit_rows, rows, relative_fit, relative_rows)
    phase2_mix = exp04.mixes(data, fit_rows, rows, phase2_pct, rule_pct, "phase2")[0]
    relative_mix = exp04.mixes(data, fit_rows, rows, relative_pct, rule_pct, "relative")[0]
    return {"slack_rule": shared.rule_scores(fit_rows, rows), "model": model_rows, "phase2_w0.5": phase2_mix["phase2_w0.5"],
            "relative_changing": relative_mix["relative_changing"]}


def test_values(y, score, months, days, not_past):
    """Every test measure for one ranking; called on the test orders and on each resample."""
    out = {"pr_auc": shared.average_precision_score(y, score),
           "pr_auc_not_past": shared.average_precision_score(y[not_past], score[not_past])}
    open_y, open_score, open_months = y[not_past], score[not_past], months[not_past]
    for share in SHARES:
        monthly = monthly_flags(score, months, share)
        daily = monthly_flags(score, days, share)
        out[f"monthly_{share:.0%}_caught"] = (y & monthly).sum()
        out[f"daily_{share:.0%}_caught"] = (y & daily).sum()
        out[f"daily_{share:.0%}_caught_not_past"] = (y & daily & not_past).sum()
        out[f"not_past_list_{share:.0%}_caught"] = (open_y & monthly_flags(open_score, open_months, share)).sum()
    order = np.argsort(-open_score, kind="stable")
    for size in TOP_ORDERS:
        out[f"not_past_top_{size}_caught"] = open_y[order[:size]].sum()
    return out


def pair_intervals(y, scores, months, days, not_past):
    rng = np.random.default_rng(shared.SEED)
    groups = [np.flatnonzero(months == m) for m in np.unique(months)]
    point = {name: test_values(y, scores[name], months, days, not_past) for name in RANKINGS}
    draws = []
    for _ in range(shared.BOOTSTRAPS):
        index = np.concatenate([rng.choice(g, len(g)) for g in groups])
        draws.append({name: test_values(y[index], scores[name][index], months[index], days[index], not_past[index]) for name in RANKINGS})
    rows = []
    for a, b in itertools.combinations(RANKINGS, 2):
        for metric in point[a]:
            gaps = np.array([d[a][metric] - d[b][metric] for d in draws], dtype=float)
            rows.append({"a": a, "b": b, "metric": metric, "a_minus_b": float(point[a][metric] - point[b][metric]),
                         "ci_low": np.percentile(gaps, 2.5), "ci_high": np.percentile(gaps, 97.5)})
    values = pd.DataFrame(point).rename_axis("metric").reset_index()
    return values, pd.DataFrame(rows)


def period_row(period, name, y, score, flags, not_past):
    return {"period": period, "ranking": name, "late": int(y.sum()), "pr_auc": shared.average_precision_score(y, score),
            "pr_auc_not_past": shared.average_precision_score(y[not_past], score[not_past]), "recall_top10": (y & flags).sum() / y.sum()}


def shortfalls(periods):
    rows = []
    used = periods[~periods["period"].isin(EXCLUDED_FROM_SHORTFALLS)]
    for metric in SHORTFALL_METRICS:
        table = used.pivot(index="period", columns="ranking", values=metric)
        behind = table.rsub(table.max(axis=1), axis=0)
        for name in RANKINGS:
            rows.append({"metric": metric, "ranking": name, "largest_shortfall": behind[name].max(), "period_of_largest": behind[name].idxmax(),
                         "mean_shortfall": behind[name].mean(), "periods_best": int(behind[name].eq(0).sum())})
    return pd.DataFrame(rows)


def check_earlier_results(values, validation):
    table = values.set_index("metric")
    for name, (pr_auc, monthly, daily) in EXPECTED_TEST.items():
        got = (table.loc["monthly_10%_caught", name], table.loc["daily_10%_caught", name])
        if abs(table.loc["pr_auc", name] - pr_auc) > TOLERANCE or got != (monthly, daily):
            raise AssertionError(f"{name} does not reproduce earlier test results: {got}")
    for name, expected in EXPECTED_TOP_200.items():
        if table.loc["not_past_top_200_caught", name] != expected:
            raise AssertionError(f"{name} does not reproduce the earlier top-200 count")
    caught = validation[validation["share"].eq(0.10)].groupby("ranking")["caught"].sum()
    for name, expected in EXPECTED_VALIDATION_CAUGHT.items():
        if caught[name] != expected:
            raise AssertionError(f"{name} does not reproduce earlier validation results: {caught[name]}")
    print("Earlier test and validation results reproduce", flush=True)


def run():
    data = load_data(exp07.add_relative_inputs)
    train, test = data["train"], data["test"]
    spec_relative = exp07.variant(data, True)
    capacity, periods = [], []
    for number, (fit_rows, rows) in enumerate(validation_rows(train, data["folds"]), start=1):
        model_fit, model_rows = exp04.phase2_scores(data, fit_rows, rows)
        scores = scores_for(data, spec_relative, fit_rows, rows, model_fit, model_rows)
        y = rows[LABEL].to_numpy().astype(bool)
        not_past = rows["past_promise_at_handover"].ne(1).to_numpy()
        for name in RANKINGS:
            capacity += [{"period": f"W{number}", "ranking": name, "share": share, "late": int(y.sum()), "caught": int((y & top_flags(scores[name], share)).sum())}
                         for share in SHARES]
            periods.append(period_row(f"W{number}", name, y, scores[name], top_flags(scores[name], 0.10), not_past))
    validation = pd.DataFrame(capacity)

    scores = scores_for(data, spec_relative, train, test, saved_model_scores(data, train), saved_model_scores(data, test))
    y = test[LABEL].to_numpy().astype(bool)
    not_past = test["past_promise_at_handover"].ne(1).to_numpy()
    months = test["order_purchase_timestamp"].dt.to_period("M").astype(str).to_numpy()
    days = test["handover_time"].dt.strftime("%Y-%m-%d").to_numpy()
    for month in np.unique(months):
        part = months == month
        for name in RANKINGS:
            periods.append(period_row(month, name, y[part], scores[name][part], top_flags(scores[name][part], 0.10), not_past[part]))
    periods = pd.DataFrame(periods)

    values, pairs = pair_intervals(y, scores, months, days, not_past)
    check_earlier_results(values, validation)
    behind = shortfalls(periods)

    values.to_csv(OUT / "test_values.csv", index=False)
    pairs.to_csv(OUT / "test_pairs.csv", index=False)
    validation.to_csv(OUT / "validation_capacity.csv", index=False)
    periods.to_csv(OUT / "periods.csv", index=False)
    behind.to_csv(OUT / "period_shortfalls.csv", index=False)

    print(values.round(3).to_string(index=False))
    print(validation.groupby(["share", "ranking"], sort=False)["caught"].sum().unstack().round(0).to_string())
    for metric in SHORTFALL_METRICS:
        print(metric)
        print(periods.pivot(index="period", columns="ranking", values=metric)[RANKINGS].round(3).to_string())
    print(behind.round(4).to_string(index=False))
    print(pairs.round(3).to_string(index=False))


if __name__ == "__main__":
    run()
