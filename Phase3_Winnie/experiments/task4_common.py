"""Shared scoring and gate checks for the Task 4 experiments from 05 onwards.

Applies Phase3_Winnie/EVALUATION_CRITERIA.md using the helpers from experiments 02 and 03, and stops if the model and
remaining-slack rule do not reproduce the report's counts.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

HERE = Path(__file__).resolve().parent
PHASE3 = HERE.parent
PHASE2 = PHASE3.parent / "Phase2"
sys.path.insert(0, str(PHASE2))

from phase2_classification import feature_columns, make_pipeline  # noqa: E402
from phase2_handover_classifier import (  # noqa: E402
    CANDIDATES as CLS_CANDIDATES, LABEL, PROMISE, TOP_SHARE, chronological_folds, load_cohort, monthly_flags, top_flags,
    variant_spec,
)

_spec = importlib.util.spec_from_file_location("experiment_03", HERE / "03_improve_logistic" / "run.py")
exp03 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(exp03)
shared = exp03.shared

BASELINE_CONFIG = {"name": exp03.BASELINE, "C": 0.01, "training": ("all", None), "signals": "base"}


def load_data(prepare=None):
    """The cohort, split and folded as in Phase 2; `prepare` may add inputs calculated from `load_cohort` columns."""
    table, spec, _ = load_cohort(PHASE2)
    if prepare is not None:
        table = prepare(table)
    train = table[table[spec["primary_split"]].eq("train")].reset_index(drop=True)
    test = table[table[spec["primary_split"]].eq("test")].reset_index(drop=True)
    params = shared.lookup(CLS_CANDIDATES, "logistic_c0.01")[1]
    return {"table": table, "spec": spec, "train": train, "test": test, "folds": chronological_folds(train, spec),
            "params": params, "spec_base": variant_spec(spec), "short_cut": float(train[PROMISE].quantile(shared.SHORT_SHARE))}


def make_pipeline_for(data, spec_v=None):
    """The Phase 2 late-warning pipeline (C = 0.01), for `spec_v` or the Phase 2 inputs. Fits on a single thread."""
    pipeline = make_pipeline("linear", data["params"], spec_v or data["spec_base"], log_features=data["spec"]["linear_log_candidates"])
    threadpool_limits(limits=1)
    return pipeline


def validation_rows(train, folds):
    return [(train.iloc[fit], train.iloc[valid].reset_index(drop=True)) for fit, valid in folds]


def saved_model_scores(data, rows):
    manifest = json.loads((PHASE3 / "models/phase2_final_models_manifest.json").read_text())["models"]["late_warning"]
    return joblib.load(PHASE3 / "models" / manifest["file"]).predict_proba(rows[sum(feature_columns(data["spec_base"]), [])])[:, 1]


def evaluate(out_dir, data, valid_scores, test_scores, roles, test_references=()):
    """Write and print validation, test and gate results.

    `valid_scores` maps a ranking to one score array per validation window, `test_scores` maps a ranking to its test
    scores, and `roles` names the candidates that go through the gates.
    """
    out_dir = Path(out_dir)
    train, test, short_cut = data["train"], data["test"], data["short_cut"]
    pairs_rows = validation_rows(train, data["folds"])
    baseline, rule_scores = [], []
    for fit_rows, valid_rows in pairs_rows:
        pipeline, columns = exp03.fit_config(BASELINE_CONFIG, fit_rows, valid_rows["order_purchase_timestamp"].min(),
                                             data["spec"], data["spec_base"], data["params"])
        baseline.append(pipeline.predict_proba(valid_rows[columns])[:, 1])
        rule_scores.append(shared.rule_scores(fit_rows, valid_rows))
    all_valid = {"slack_rule": rule_scores, "model": baseline, **valid_scores}
    rows = []
    for name, per_window in all_valid.items():
        for number, ((_, valid_rows), s) in enumerate(zip(pairs_rows, per_window), start=1):
            rows.append({"config": name, **shared.metric_row(f"W{number}", name, valid_rows, s, top_flags(s, TOP_SHARE), short_cut)})
    valid_frame = pd.DataFrame(rows)
    for number in range(1, 6):
        part = valid_frame[valid_frame["period"].eq(f"W{number}")].set_index("config")["caught"]
        if (int(part["model"]), int(part["slack_rule"])) != shared.REPORT_CAUGHT[f"W{number}"]:
            raise AssertionError(f"W{number} does not reproduce the report")

    base_values = exp03.window_pr(valid_frame, "model")
    summary = []
    for name in all_valid:
        values = exp03.window_pr(valid_frame, name)
        part = valid_frame[valid_frame["config"].eq(name)].set_index("period")
        summary.append({"ranking": name, "pr_auc": values.mean(), "pr_auc_minus_model": (values - base_values).mean(),
                        "paired_se_vs_model": exp03.paired_se(values, base_values) if name != "model" else 0.0,
                        "caught": int(part["caught"].sum()), "caught_not_past": int(part["caught_not_past"].sum()),
                        "pr_auc_short_not_past": part.loc[shared.SHORT_WINDOWS, "pr_auc_short_not_past"].mean()})
    summary = pd.DataFrame(summary)

    y = test[LABEL].to_numpy().astype(bool)
    months = test["order_purchase_timestamp"].dt.to_period("M").astype(str).to_numpy()
    days = test["handover_time"].dt.strftime("%Y-%m-%d").to_numpy()
    scores = {"slack_rule": shared.rule_scores(train, test), "model": saved_model_scores(data, test), **test_scores}
    flags = {name: monthly_flags(s, months, TOP_SHARE) for name, s in scores.items()}
    daily = {name: monthly_flags(s, days, TOP_SHARE) for name, s in scores.items()}
    metrics = [shared.metric_row("test", name, test, scores[name], flags[name], short_cut, daily[name]) for name in scores]
    for month in np.unique(months):
        part = months == month
        metrics += [shared.metric_row(month, name, test[part], scores[name][part], flags[name][part], short_cut, daily[name][part])
                    for name in scores]
    metrics = pd.DataFrame(metrics)
    whole = metrics[metrics["period"].eq("test")].set_index("ranking")["caught"]
    if (int(whole["model"]), int(whole["slack_rule"])) != shared.REPORT_CAUGHT["test"]:
        raise AssertionError("Test baselines do not reproduce the report")

    pairs = [(name, "slack_rule") for name in scores if name != "slack_rule"] + [(name, "model") for name in scores if name not in ("slack_rule", "model")]
    intervals = exp03.test_bootstrap(y, scores, months, test["past_promise_at_handover"].ne(1).to_numpy(), pairs)
    valid_y = [valid_rows[LABEL].to_numpy() for _, valid_rows in pairs_rows]
    windows = {name: [{"y": t, "candidate": s, "slack_rule": r} for t, s, r in zip(valid_y, valid_scores[name], rule_scores)] for name in roles}
    gates = exp03.gate_rows(roles, roles, windows, valid_frame, valid_frame[valid_frame["config"].eq("slack_rule")],
                            metrics[metrics["period"].eq("test")], intervals)

    valid_frame.to_csv(out_dir / "validation_windows.csv", index=False)
    summary.to_csv(out_dir / "validation_summary.csv", index=False)
    metrics.to_csv(out_dir / "test_metrics.csv", index=False)
    intervals.to_csv(out_dir / "test_bootstrap.csv", index=False)
    gates.to_csv(out_dir / "gates.csv", index=False)

    print(summary.round(4).to_string(index=False))
    print(valid_frame.pivot(index="config", columns="period", values="pr_auc").round(3).to_string())
    print(metrics.round(3).to_string(index=False))
    print(gates.round(4).T.to_string())
    print(intervals.round(3).to_string(index=False))
    return scores
