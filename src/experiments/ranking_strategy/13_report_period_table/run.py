"""Period-by-period comparison of the five late-warning rankings in the final report (Table 18).

Run from the repository root:
    python src/experiments/ranking_strategy/13_report_period_table/run.py

Rankings: remaining-slack rule, Phase 2 model, 50/50 blend of their percentiles, the regression ranking (predicted
delivery days minus promised days, using the Phase 3 selected handover ridge from data/outputs/phase2/eta/) and
closest to deadline. Periods are the five validation windows and each test purchase month; shortfalls are measured
against the best ranking in each period, excluding the partial May 2018. Uses the scoring helpers of experiments 02
and 04, so the rule, model and blend match experiment 08. Writes CSVs to
data/outputs/experiments/ranking_strategy/13_report_period_table/.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[2]))

from olist.paths import EXPERIMENT_OUTPUTS, PHASE2_OUTPUTS  # noqa: E402

OUT = EXPERIMENT_OUTPUTS / "ranking_strategy" / HERE.name
OUT.mkdir(parents=True, exist_ok=True)
sys.path.insert(0, str(HERE.parent))

from olist.eta import CANDIDATES as ETA_CANDIDATES  # noqa: E402
from task4_common import LABEL, load_data, saved_model_scores, shared, validation_rows  # noqa: E402

_spec = importlib.util.spec_from_file_location("experiment_04", HERE.parent / "04_mix_ratios" / "run.py")
exp04 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(exp04)

RANKINGS = {"slack_rule": "Remaining-slack rule", "model": "Phase 2 model", "mix_50_50": "50/50 blend",
            "regression_gap": "Regression ranking", "deadline": "Closest to deadline"}
EXCLUDED_FROM_SHORTFALLS = {"2018-05"}
# Experiment 08's saved PR-AUC for the rule, model and blend; the same scores must come out here.
EXPERIMENT_08 = {"slack_rule": "slack_rule", "model": "model", "mix_50_50": "phase2_w0.5"}


def run():
    selected = json.loads((PHASE2_OUTPUTS / "eta/selection.json").read_text())["selected_candidate"]
    regression_model = next((f, p) for n, f, p in ETA_CANDIDATES if n == selected)
    data = load_data()
    train, test = data["train"], data["test"]
    rows_out = []

    def add(period, rows, scores):
        y = rows[LABEL].to_numpy().astype(bool)
        for name in RANKINGS:
            rows_out.append({"period": period, "ranking": name, "orders": len(rows), "late": int(y.sum()),
                             "pr_auc": shared.average_precision_score(y, scores[name])})

    for number, (fit_rows, rows) in enumerate(validation_rows(train, data["folds"]), start=1):
        model_fit, model_rows = exp04.phase2_scores(data, fit_rows, rows)
        add(f"W{number}", rows, shared.all_rankings(fit_rows, rows, model_fit, model_rows, data["spec"], regression_model))

    scores = shared.all_rankings(train, test, saved_model_scores(data, train), saved_model_scores(data, test),
                                 data["spec"], regression_model)
    months = test["order_purchase_timestamp"].dt.to_period("M").astype(str).to_numpy()
    for month in np.unique(months):
        part = months == month
        add(month, test[part].reset_index(drop=True), {k: v[part] for k, v in scores.items()})
    add("test", test, scores)

    periods = pd.DataFrame(rows_out)
    table = periods.pivot(index="period", columns="ranking", values="pr_auc")[list(RANKINGS)]
    saved = pd.read_csv(OUT.parent / "08_final_comparison" / "periods.csv").pivot(
        index="period", columns="ranking", values="pr_auc")
    for ours, theirs in EXPERIMENT_08.items():
        common = saved.index.intersection(table.index)
        gap = float((table.loc[common, ours] - saved.loc[common, theirs]).abs().max())
        print(f"{ours} vs experiment 08 {theirs}: max PR-AUC difference {gap:.2e}", flush=True)

    used = table.drop(index=["test", *EXCLUDED_FROM_SHORTFALLS], errors="ignore")
    behind = used.rsub(used.max(axis=1), axis=0)
    shortfalls = pd.DataFrame([{"ranking": name, "largest_shortfall": behind[name].max(),
                                "period_of_largest": behind[name].idxmax(), "periods_best": int(behind[name].eq(0).sum())}
                               for name in RANKINGS])
    periods.to_csv(OUT / "periods.csv", index=False)
    table.to_csv(OUT / "pr_auc_table.csv")
    shortfalls.to_csv(OUT / "shortfalls.csv", index=False)
    pd.Series({"regression_model": selected}).to_json(OUT / "settings.json", indent=2)
    print(table.round(3).to_string())
    print(shortfalls.round(3).to_string(index=False))


if __name__ == "__main__":
    run()
