"""Test other mix ratios of a logistic model and the remaining-slack rule.

Run from the repository root:
    python Phase3_Winnie/experiments/04_mix_ratios/run.py

Declared in Phase3_Winnie/EVALUATION_CRITERIA.md. Two families, the Phase 2 model (main) and the promise-relative model
from experiment 07 (exploratory), each with fixed ratios chosen on validation and a ratio that follows the recent share
of short promises. The script stops if references do not reproduce experiments 02 and 07. Writes CSVs next to this file.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

from task4_common import (  # noqa: E402
    BASELINE_CONFIG, LABEL, PROMISE, evaluate, exp03, feature_columns, load_data, saved_model_scores, shared, validation_rows,
)

_spec = importlib.util.spec_from_file_location("experiment_07", HERE.parent / "07_promise_relative_inputs" / "run.py")
exp07 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(exp07)

MODEL_WEIGHTS = [round(w / 10, 1) for w in range(1, 10)]
FAMILIES = {"phase2": "main", "relative": "exploratory"}
SHORT_LOOKBACK = pd.Timedelta(days=30)
RULE_WEIGHT_SCALE = 2.0
EXPERIMENT_02_MIX = {"caught": 403, "pr_auc": 0.5167}
EXPERIMENT_07_CAUGHT = 406
TOLERANCE = 1e-3


def recent_short_share(table, rows, cut):
    """Share of orders purchased in the 30 days before each order's handover with a promise under `cut`."""
    ordered = table.sort_values("order_purchase_timestamp")
    times = ordered["order_purchase_timestamp"].to_numpy(dtype="datetime64[ns]")
    counts = np.concatenate([[0], np.cumsum((ordered[PROMISE] < cut).to_numpy(dtype=int))])
    now = rows["handover_time"].to_numpy(dtype="datetime64[ns]")
    end = np.searchsorted(times, now, side="left")
    start = np.searchsorted(times, now - SHORT_LOOKBACK.to_timedelta64(), side="left")
    size = end - start
    return np.divide(counts[end] - counts[start], size, out=np.zeros(len(rows), dtype=float), where=size > 0)


def percentiles(data, fit_rows, rows, model_fit, model_rows):
    return shared.percentile(model_fit, model_rows), shared.percentile(shared.rule_scores(fit_rows, fit_rows), shared.rule_scores(fit_rows, rows))


def mixes(data, fit_rows, rows, model_pct, rule_pct, family):
    out = {f"{family}_w{w}": w * model_pct + (1 - w) * rule_pct for w in MODEL_WEIGHTS}
    share = recent_short_share(data["table"], rows, fit_rows[PROMISE].quantile(shared.SHORT_SHARE))
    rule_weight = np.minimum(1.0, RULE_WEIGHT_SCALE * share)
    out[f"{family}_changing"] = (1 - rule_weight) * model_pct + rule_weight * rule_pct
    return out, share


def phase2_scores(data, fit_rows, rows):
    pipeline, columns = exp03.fit_config(BASELINE_CONFIG, fit_rows, rows["order_purchase_timestamp"].min(), data["spec"], data["spec_base"], data["params"])
    return pipeline.predict_proba(fit_rows[columns])[:, 1], pipeline.predict_proba(rows[columns])[:, 1]


def relative_scores(data, spec_v, fit_rows, rows):
    pipeline, values = exp07.fit_scores(data, spec_v, fit_rows, rows)
    return pipeline.predict_proba(fit_rows[sum(feature_columns(spec_v), [])])[:, 1], values, pipeline


def choose_ratio(frame, family):
    """Best fixed ratio on validation; it replaces 50/50 only if the gain exceeds one paired standard error."""
    base = exp03.window_pr(frame, f"{family}_w0.5")
    audit = []
    for w in MODEL_WEIGHTS:
        values = exp03.window_pr(frame, f"{family}_w{w}")
        audit.append({"family": family, "model_weight": w, "mean_pr_auc": values.mean(), "gain_over_50_50": (values - base).mean(),
                      "paired_se": exp03.paired_se(values, base) if w != 0.5 else 0.0})
    audit = pd.DataFrame(audit)
    best = audit.loc[audit["mean_pr_auc"].idxmax()]
    chosen = float(best["model_weight"]) if best["gain_over_50_50"] > best["paired_se"] else 0.5
    return chosen, audit


def validation_frame(data, windows, scores):
    rows = []
    for name, per_window in scores.items():
        for number, ((_, valid_rows), s) in enumerate(zip(windows, per_window), start=1):
            rows.append({"config": name, "period": f"W{number}", "pr_auc": shared.average_precision_score(valid_rows[LABEL], s)})
    return pd.DataFrame(rows)


def run():
    data = load_data(exp07.add_relative_inputs)
    train, test = data["train"], data["test"]
    spec_relative = exp07.variant(data, True)
    windows = validation_rows(train, data["folds"])

    # Validation only, up to the point the candidates are fixed.
    valid_scores = {}
    for fit_rows, rows in windows:
        model_fit, model_rows = phase2_scores(data, fit_rows, rows)
        relative_fit, relative_rows, _ = relative_scores(data, spec_relative, fit_rows, rows)
        window = {}
        for family, fit_values, values in (("phase2", model_fit, model_rows), ("relative", relative_fit, relative_rows)):
            model_pct, rule_pct = percentiles(data, fit_rows, rows, fit_values, values)
            window.update(mixes(data, fit_rows, rows, model_pct, rule_pct, family)[0])
        window["relative_model"] = relative_rows
        for name, values in window.items():
            valid_scores.setdefault(name, []).append(values)
    frame = validation_frame(data, windows, valid_scores)
    chosen, audits = {}, []
    for family in FAMILIES:
        chosen[family], audit = choose_ratio(frame, family)
        audits.append(audit)
    audits = pd.concat(audits, ignore_index=True)
    audits.to_csv(HERE / "ratio_selection.csv", index=False)

    roles = {"phase2_changing": "main_changing", "relative_changing": "exploratory_changing",
             f"relative_w{chosen['relative']}": "exploratory_fixed"}
    if chosen["phase2"] != 0.5:
        roles[f"phase2_w{chosen['phase2']}"] = "main_fixed"
    (HERE / "selection.json").write_text(json.dumps({"chosen_model_weight": chosen, "candidates": roles}, indent=2) + "\n")
    print(audits.round(4).to_string(index=False))
    print(f"Candidates fixed before test scoring: {roles}", flush=True)

    # Test scoring, once per declared candidate.
    model_fit, model_rows = saved_model_scores(data, train), saved_model_scores(data, test)
    relative_fit, relative_rows, _ = relative_scores(data, spec_relative, train, test)
    phase2_pct, rule_pct = percentiles(data, train, test, model_fit, model_rows)
    relative_pct, _ = percentiles(data, train, test, relative_fit, relative_rows)
    phase2_mixes, share = mixes(data, train, test, phase2_pct, rule_pct, "phase2")
    relative_mixes, _ = mixes(data, train, test, relative_pct, rule_pct, "relative")
    pool = {**phase2_mixes, **relative_mixes}
    test_scores = {name: pool[name] for name in roles}
    test_scores["phase2_w0.5"] = pool["phase2_w0.5"]
    test_scores["relative_model"] = relative_rows
    months = test["order_purchase_timestamp"].dt.to_period("M").astype(str)
    pd.DataFrame({"month": months, "recent_short_share": share}).groupby("month").agg(mean_share=("recent_short_share", "mean")).to_csv(HERE / "test_short_share.csv")
    print(pd.DataFrame({"month": months, "share": share, "rule_weight": np.minimum(1.0, RULE_WEIGHT_SCALE * share)}).groupby("month").mean().round(3).to_string(), flush=True)

    keep = {name: valid_scores[name] for name in {*test_scores}}
    evaluate(HERE, data, keep, test_scores, roles)

    metrics = pd.read_csv(HERE / "test_metrics.csv")
    whole = metrics[metrics["period"].eq("test")].set_index("ranking")
    if int(whole.loc["phase2_w0.5", "caught"]) != EXPERIMENT_02_MIX["caught"] or abs(whole.loc["phase2_w0.5", "pr_auc"] - EXPERIMENT_02_MIX["pr_auc"]) > TOLERANCE:
        raise AssertionError("The 50/50 mix does not reproduce experiment 02")
    if int(whole.loc["relative_model", "caught"]) != EXPERIMENT_07_CAUGHT:
        raise AssertionError("The relative model does not reproduce experiment 07")
    print("References reproduce experiments 02 and 07", flush=True)


if __name__ == "__main__":
    run()
