"""Compare four rankings of handover orders before trying any combination.

Run from the repository root:
    python src/experiments/ranking_strategy/01_ranking_baselines/run.py

Rankings, all scored at carrier handover:
- model: the Phase 2 late-warning logistic regression
- slack_rule: least remaining slack first (promised days - elapsed days - recent route transit), the report's benchmark
- deadline: least time to the promised date first (promised days - elapsed days), with no expected transit
- regression_gap: largest handover delivery estimate minus promised days first

Validation lists take the top 10% of each window and test lists the top 10% of each purchase month, as in the report.
The script stops if the model and slack rule do not reproduce the report's counts. Writes CSVs to data/outputs/experiments/ranking_strategy/01_ranking_baselines/.
"""

from __future__ import annotations

import itertools
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[2]))

from olist.paths import EXPERIMENT_OUTPUTS, REPO_ROOT  # noqa: E402

OUT = EXPERIMENT_OUTPUTS / "ranking_strategy" / HERE.name
OUT.mkdir(parents=True, exist_ok=True)
MODELS = EXPERIMENT_OUTPUTS / "ranking_strategy" / "models"

from olist.classification import feature_columns  # noqa: E402
from olist.eta import CANDIDATES as ETA_CANDIDATES, PHASE2_REFERENCE, stage_spec  # noqa: E402
from olist.handover_classifier import (  # noqa: E402
    CANDIDATES as CLS_CANDIDATES, LABEL, PROMISE, TOP_SHARE, baseline_scores, chronological_folds, fit_scores,
    load_cohort, monthly_flags, top_flags, variant_spec,
)

RANKINGS = ["model", "slack_rule", "deadline", "regression_gap"]
REPORT_CAUGHT = {"W1": (164, 132), "W2": (341, 271), "W3": (126, 112), "W4": (274, 227), "W5": (469, 357),
                 "test": (377, 398)}
BOOTSTRAPS, SEED = 1000, 42


def lookup(candidates, name):
    return next((family, params) for n, family, params in candidates if n == name)


def rankings_for(fit_rows, score_rows, spec, regression_model):
    reg_family, reg_params = regression_model
    base = baseline_scores(fit_rows, score_rows, stage_spec(spec, "handover"), reg_family, reg_params)
    return {"slack_rule": base["slack_rule"], "regression_gap": base["regression_gap"],
            "deadline": (score_rows["handover_days"] - score_rows[PROMISE]).to_numpy(dtype=float)}


def metric_row(period, name, rows, scores, flags):
    y = rows[LABEL].to_numpy().astype(bool)
    open_rows = rows["past_promise_at_handover"].eq(0).to_numpy()
    return {"period": period, "ranking": name, "orders": len(y), "late": int(y.sum()),
            "pr_auc": average_precision_score(y, scores) if y.any() else np.nan,
            "roc_auc": roc_auc_score(y, scores) if y.any() else np.nan,
            "reviewed": int(flags.sum()), "caught": int((y & flags).sum()),
            "precision": (y & flags).sum() / flags.sum(), "recall": (y & flags).sum() / y.sum() if y.any() else np.nan,
            "late_not_past_promise": int((y & open_rows).sum()), "caught_not_past_promise": int((y & flags & open_rows).sum()),
            "pr_auc_not_past_promise": (average_precision_score(y[open_rows], scores[open_rows])
                                        if (y & open_rows).any() else np.nan)}


def check_report(metrics):
    for period, expected in REPORT_CAUGHT.items():
        part = metrics[metrics["period"].eq(period)].set_index("ranking")["caught"]
        got = (int(part["model"]), int(part["slack_rule"]))
        print(f"{period}: model, rule caught {got} (report {expected})", flush=True)
        if got != expected:
            raise AssertionError(f"{period} does not reproduce the report: {got} vs {expected}")


def overlap_rows(period, y, flags):
    rows = []
    for a, b in itertools.combinations(RANKINGS, 2):
        rows.append({"period": period, "a": a, "b": b, "flagged_both": int((flags[a] & flags[b]).sum()),
                     "caught_both": int((y & flags[a] & flags[b]).sum()), "caught_only_a": int((y & flags[a] & ~flags[b]).sum()),
                     "caught_only_b": int((y & flags[b] & ~flags[a]).sum()), "caught_either": int((y & (flags[a] | flags[b])).sum())})
    return rows


def bootstrap(y, scores, months):
    """95% intervals for paired test differences, resampling orders within each purchase month."""
    rng = np.random.default_rng(SEED)
    groups = [np.flatnonzero(months == m) for m in np.unique(months)]
    draws = []
    for _ in range(BOOTSTRAPS):
        index = np.concatenate([rng.choice(g, len(g)) for g in groups])
        draw = {}
        for name in RANKINGS:
            draw[name, "caught"] = (y[index] & monthly_flags(scores[name][index], months[index], TOP_SHARE)).sum()
            draw[name, "pr_auc"] = average_precision_score(y[index], scores[name][index])
        draws.append(draw)
    rows = []
    for a, b in itertools.combinations(RANKINGS, 2):
        for metric in ("caught", "pr_auc"):
            gaps = np.array([d[a, metric] - d[b, metric] for d in draws], dtype=float)
            rows.append({"a": a, "b": b, "metric": metric, "mean_a_minus_b": gaps.mean(),
                         "ci_low": np.percentile(gaps, 2.5), "ci_high": np.percentile(gaps, 97.5)})
    return rows


def run():
    table, spec, _ = load_cohort(REPO_ROOT)
    train = table[table[spec["primary_split"]].eq("train")].reset_index(drop=True)
    test = table[table[spec["primary_split"]].eq("test")].reset_index(drop=True)
    folds = chronological_folds(train, spec)
    # The Phase 2 regression (ridge alpha=100), not the Phase 3 selection: this experiment compares Phase 2 rankings.
    regression_model = lookup(ETA_CANDIDATES, PHASE2_REFERENCE)
    cls_family, cls_params = lookup(CLS_CANDIDATES, "logistic_c0.01")
    spec_v = variant_spec(spec)
    metrics = []

    for number, (fit, valid) in enumerate(folds, start=1):
        fit_rows, valid_rows = train.iloc[fit], train.iloc[valid].reset_index(drop=True)
        scores = rankings_for(fit_rows, valid_rows, spec, regression_model)
        scores["model"] = fit_scores(cls_family, cls_params, spec_v, fit_rows, valid_rows, spec["linear_log_candidates"])
        metrics += [metric_row(f"W{number}", name, valid_rows, scores[name], top_flags(scores[name], TOP_SHARE))
                    for name in RANKINGS]

    manifest = json.loads((MODELS / "phase2_final_models_manifest.json").read_text())["models"]["late_warning"]
    baseline = joblib.load(MODELS / manifest["file"])
    scores = rankings_for(train, test, spec, regression_model)
    scores["model"] = baseline.predict_proba(test[sum(feature_columns(spec_v), [])])[:, 1]
    months = test["order_purchase_timestamp"].dt.to_period("M").astype(str).to_numpy()
    flags = {name: monthly_flags(scores[name], months, TOP_SHARE) for name in RANKINGS}
    y = test[LABEL].to_numpy().astype(bool)
    metrics += [metric_row("test", name, test, scores[name], flags[name]) for name in RANKINGS]
    for month in np.unique(months):
        part = months == month
        metrics += [metric_row(month, name, test[part], scores[name][part], flags[name][part]) for name in RANKINGS]

    metrics = pd.DataFrame(metrics)
    check_report(metrics)
    metrics.to_csv(OUT / "metrics.csv", index=False)
    august = months == "2018-08"
    overlap = overlap_rows("test", y, flags) + overlap_rows("2018-08", y[august], {n: f[august] for n, f in flags.items()})
    pd.DataFrame(overlap).to_csv(OUT / "test_overlap.csv", index=False)
    pd.DataFrame(bootstrap(y, scores, months)).to_csv(OUT / "test_bootstrap.csv", index=False)

    windows = metrics[metrics["period"].str.fullmatch("W[1-5]")]
    print(windows.groupby("ranking", sort=False)[["pr_auc", "roc_auc", "caught", "caught_not_past_promise"]]
          .agg({"pr_auc": "mean", "roc_auc": "mean", "caught": "sum", "caught_not_past_promise": "sum"}).to_string())
    print(metrics[~metrics["period"].str.fullmatch("W[1-5]")].to_string(index=False))


if __name__ == "__main__":
    run()
