"""Try to improve the late-warning logistic regression, choosing on validation only.

Run from the repository root:
    python Phase3_Winnie/experiments/03_improve_logistic/run.py

Settings, selection and gates are declared in Phase3_Winnie/EVALUATION_CRITERIA.md:
- Stage 1: C and training data (all history, recent windows, recency weights), one-standard-error rule
- Stage 2: the unused ZIP-3 and seller history signals, kept only if the gain exceeds one paired standard error
- Candidates: the selected configuration and the best recent-data configuration

Saved Phase 2 models are read, never overwritten. Candidate models and CSVs are written next to this file.
"""

from __future__ import annotations

import importlib.util
import itertools
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

HERE = Path(__file__).resolve().parent
PHASE3 = HERE.parents[1]
PHASE2 = PHASE3.parent / "Phase2"
sys.path.insert(0, str(PHASE2))

from phase2_classification import feature_columns, make_pipeline  # noqa: E402
from phase2_handover_classifier import (  # noqa: E402
    CANDIDATES as CLS_CANDIDATES, EXTENSION_INPUTS, LABEL, PROMISE, TOP_SHARE, chronological_folds, gate_verdict,
    load_cohort, monthly_flags, paired_bootstrap, top_flags, variant_spec,
)

# run.py from experiment 02 supplies the shared metric and rule helpers; its folder name cannot be imported directly.
_shared = importlib.util.spec_from_file_location("experiment_02", HERE.parent / "02_model_rule_mix" / "run.py")
shared = importlib.util.module_from_spec(_shared)
_shared.loader.exec_module(shared)

C_GRID = [0.0001, 0.0003, 0.001, 0.003, 0.01]  # lower C is simpler
TRAINING = [("all", None), ("window", 365), ("window", 180), ("halflife", 180), ("halflife", 90)]  # simplest first
ZIP3, SELLER = EXTENSION_INPUTS
SIGNALS = {"base": [], "zip3": [ZIP3], "seller": [SELLER], "both": [ZIP3, SELLER]}
BASELINE = "c0.01_all_base"
TEST_REFERENCE = pd.Timestamp("2018-05-26")
SHORT_WINDOWS = shared.SHORT_WINDOWS
WEIGHTS_TO_REPORT = ["numeric__remaining_slack", "numeric__route_transit_90d_at_handover", "numeric__promised_days",
                     "numeric__handover_days", "numeric__state_backlog_recent_vs_8wk"]


def config_name(c, training, signals="base"):
    kind, days = training
    return f"c{c}_{kind if days is None else f'{kind}{days}'}_{signals}"


def stage_1_configs():
    return [{"name": config_name(c, training), "C": c, "training": training, "signals": "base"}
            for training in TRAINING for c in C_GRID]


def fit_config(config, fit_rows, reference, spec, spec_base, params):
    """Fit one configuration on orders purchased before `reference`, with its window or recency weights."""
    spec_v = variant_spec(spec, SIGNALS[config["signals"]]) if config["signals"] != "base" else spec_base
    columns = sum(feature_columns(spec_v), [])
    kind, days = config["training"]
    ages = (reference - fit_rows["order_purchase_timestamp"]).dt.total_seconds() / 86400
    rows, fit_params = fit_rows, {}
    if kind == "window":
        rows = fit_rows[ages <= days]
    elif kind == "halflife":
        weights = 0.5 ** (ages.to_numpy() / days)
        fit_params = {"model__sample_weight": weights / weights.mean()}
    pipeline = make_pipeline("linear", {**params, "C": config["C"]}, spec_v, log_features=spec["linear_log_candidates"])
    with threadpool_limits(limits=1):
        pipeline.fit(rows[columns], rows[LABEL], **fit_params)
    return pipeline, columns


def validate_config(config, train, folds, spec, spec_base, params, short_cut):
    rows, scores = [], []
    for number, (fit, valid) in enumerate(folds, start=1):
        fit_rows, valid_rows = train.iloc[fit], train.iloc[valid].reset_index(drop=True)
        reference = valid_rows["order_purchase_timestamp"].min()
        pipeline, columns = fit_config(config, fit_rows, reference, spec, spec_base, params)
        s = pipeline.predict_proba(valid_rows[columns])[:, 1]
        scores.append(s)
        rows.append({"config": config["name"], **shared.metric_row(f"W{number}", config["name"], valid_rows, s, top_flags(s, TOP_SHARE), short_cut)})
    mean_pr = np.mean([r["pr_auc"] for r in rows])
    print(f"{config['name']}: mean PR-AUC {mean_pr:.4f}, caught {sum(r['caught'] for r in rows)}", flush=True)
    return rows, scores


def window_pr(rows, name):
    return rows[rows["config"].eq(name)].set_index("period")["pr_auc"].sort_index().to_numpy()


def paired_se(a, b):
    return (a - b).std(ddof=1) / np.sqrt(len(a))


def select_stage_1(configs, rows):
    means = {c["name"]: window_pr(rows, c["name"]).mean() for c in configs}
    best = max(means, key=means.get)
    audit = []
    chosen = None
    for config in configs:
        name = config["name"]
        se = paired_se(window_pr(rows, best), window_pr(rows, name)) if name != best else 0.0
        within = bool(means[best] - means[name] <= se)
        audit.append({"config": name, "mean_pr_auc": means[name], "gap_to_best": means[best] - means[name], "paired_se": se, "within_one_se": within})
        if within and chosen is None:
            chosen = config
    return chosen, best, pd.DataFrame(audit)


def select_stage_2(stage_1, signal_configs, rows):
    base = window_pr(rows, stage_1["name"])
    audit, chosen, best_gain = [], stage_1, 0.0
    for config in signal_configs:
        values = window_pr(rows, config["name"])
        gain, se = (values - base).mean(), paired_se(values, base)
        audit.append({"config": config["name"], "gain_over_stage_1": gain, "paired_se": se, "accepted": bool(gain > se)})
        if gain > se and gain > best_gain:
            chosen, best_gain = config, gain
    return chosen, pd.DataFrame(audit)


def test_bootstrap(y, scores, months, not_past, pairs):
    rng = np.random.default_rng(shared.SEED)
    groups = [np.flatnonzero(months == m) for m in np.unique(months)]
    draws = []
    for _ in range(shared.BOOTSTRAPS):
        index = np.concatenate([rng.choice(g, len(g)) for g in groups])
        draw = {}
        for name in scores:
            flags = monthly_flags(scores[name][index], months[index], TOP_SHARE)
            draw[name, "caught"] = (y[index] & flags).sum()
            draw[name, "caught_not_past"] = (y[index] & flags & not_past[index]).sum()
            draw[name, "pr_auc"] = shared.average_precision_score(y[index], scores[name][index])
        draws.append(draw)
    rows = []
    for a, b in pairs:
        for metric in ("caught", "caught_not_past", "pr_auc"):
            gaps = np.array([d[a, metric] - d[b, metric] for d in draws], dtype=float)
            rows.append({"a": a, "b": b, "metric": metric, "ci_low": np.percentile(gaps, 2.5), "ci_high": np.percentile(gaps, 97.5)})
    return pd.DataFrame(rows)


def gate_rows(candidates, roles, valid_windows, valid_rows_metrics, rule_rows, test_metrics, intervals):
    rule_test = test_metrics.set_index("ranking").loc["slack_rule"]
    rule_short = rule_rows.set_index("period")["pr_auc_short_not_past"]
    rows = []
    for name in candidates:
        value = paired_bootstrap(valid_windows[name], "candidate", "slack_rule")
        part = valid_rows_metrics[valid_rows_metrics["config"].eq(name)].set_index("period")["pr_auc_short_not_past"]
        gaps = np.array([part[w] - rule_short[w] for w in SHORT_WINDOWS])
        short_gap, short_se = gaps.mean(), gaps.std(ddof=1) / np.sqrt(len(gaps))
        row = test_metrics.set_index("ranking").loc[name]
        at_least = all(row[m] >= rule_test[m] for m in ("pr_auc", "caught", "caught_not_past"))
        lead = shared.above_zero(intervals, name, "slack_rule")
        gate_a, gate_b, gate_c = gate_verdict(value), bool(short_gap >= -short_se), bool(at_least and lead)
        rows.append({"candidate": name, "role": roles[name],
                     "val_pr_auc_diff": value["pr_auc"]["difference"], "val_pr_auc_ci_low": value["pr_auc"]["ci_low"],
                     "val_recall_diff": value["recall_top10"]["difference"], "val_recall_ci_low": value["recall_top10"]["ci_low"],
                     "gate_a": gate_a, "short_pr_auc_diff": short_gap, "short_paired_se": short_se, "gate_b": gate_b,
                     "test_pr_auc_diff": row["pr_auc"] - rule_test["pr_auc"], "test_caught_diff": row["caught"] - rule_test["caught"],
                     "test_caught_not_past_diff": row["caught_not_past"] - rule_test["caught_not_past"],
                     "test_at_least_rule": at_least, "test_clear_lead": lead, "gate_c": gate_c, "passes_all": gate_a and gate_b and gate_c})
    return pd.DataFrame(rows)


def run():
    table, spec, _ = load_cohort(PHASE2)
    train = table[table[spec["primary_split"]].eq("train")].reset_index(drop=True)
    test = table[table[spec["primary_split"]].eq("test")].reset_index(drop=True)
    folds = chronological_folds(train, spec)
    params = shared.lookup(CLS_CANDIDATES, "logistic_c0.01")[1]
    spec_base = variant_spec(spec)
    short_cut = float(train[PROMISE].quantile(shared.SHORT_SHARE))

    # Validation only, up to the point the candidates are fixed.
    configs = stage_1_configs()
    results, scores = [], {}
    for config in configs:
        rows, scores[config["name"]] = validate_config(config, train, folds, spec, spec_base, params, short_cut)
        results += rows
    frame = pd.DataFrame(results)
    stage_1, best, stage_1_audit = select_stage_1(configs, frame)

    signal_configs = [{**stage_1, "name": config_name(stage_1["C"], stage_1["training"], signals), "signals": signals}
                      for signals in ("zip3", "seller", "both")]
    for config in signal_configs:
        rows, scores[config["name"]] = validate_config(config, train, folds, spec, spec_base, params, short_cut)
        results += rows
    frame = pd.DataFrame(results)
    selected, stage_2_audit = select_stage_2(stage_1, signal_configs, frame)

    recent = [c for c in configs if c["training"][0] != "all"]
    best_recent = max(recent, key=lambda c: window_pr(frame, c["name"]).mean())
    baseline_rows = frame[frame["config"].eq(BASELINE)].set_index("period")["caught"]
    rule_rows, valid_rule, valid_y, valid_tables = [], [], [], []
    for number, (fit, valid) in enumerate(folds, start=1):
        fit_rows, valid_rows = train.iloc[fit], train.iloc[valid].reset_index(drop=True)
        rule = shared.rule_scores(fit_rows, valid_rows)
        rule_rows.append(shared.metric_row(f"W{number}", "slack_rule", valid_rows, rule, top_flags(rule, TOP_SHARE), short_cut))
        valid_rule.append(rule)
        valid_y.append(valid_rows[LABEL].to_numpy())
        expected_model, expected_rule = shared.REPORT_CAUGHT[f"W{number}"]
        if baseline_rows[f"W{number}"] != expected_model or rule_rows[-1]["caught"] != expected_rule:
            raise AssertionError(f"W{number} does not reproduce the report")
    rule_rows = pd.DataFrame(rule_rows)

    roles = {}
    for config, role in ((selected, "selected"), (best_recent, "best_recent")):
        if config["name"] != BASELINE and config["name"] not in roles:
            roles[config["name"]] = role
        elif config["name"] in roles:
            roles[config["name"]] += f"+{role}"
    candidates = {c["name"]: c for c in [selected, best_recent] if c["name"] in roles}
    print(f"Stage 1 best {best}; Stage 1 choice {stage_1['name']}; selected {selected['name']}; best recent {best_recent['name']}", flush=True)
    print(f"Candidates fixed before test scoring: {roles}", flush=True)

    frame.to_csv(HERE / "validation_windows.csv", index=False)
    stage_1_audit.to_csv(HERE / "stage_1_selection.csv", index=False)
    stage_2_audit.to_csv(HERE / "stage_2_selection.csv", index=False)
    (HERE / "selection.json").write_text(json.dumps({"stage_1_best": best, "stage_1_choice": stage_1["name"], "selected": selected["name"],
                                                     "best_recent": best_recent["name"], "candidates": roles}, indent=2) + "\n")

    # Test scoring, once per declared candidate.
    manifest = json.loads((PHASE3 / "models/phase2_final_models_manifest.json").read_text())["models"]["late_warning"]
    saved = joblib.load(PHASE3 / "models" / manifest["file"])
    test_scores = {"slack_rule": shared.rule_scores(train, test), "model": saved.predict_proba(test[sum(feature_columns(spec_base), [])])[:, 1]}
    weights = {"model": pd.Series(saved.steps[-1][1].coef_[0], index=saved[:-1].get_feature_names_out())}
    for name, config in candidates.items():
        pipeline, columns = fit_config(config, train, TEST_REFERENCE, spec, spec_base, params)
        test_scores[name] = pipeline.predict_proba(test[columns])[:, 1]
        weights[name] = pd.Series(pipeline.steps[-1][1].coef_[0], index=pipeline[:-1].get_feature_names_out())
        joblib.dump(pipeline, HERE / f"{name}.joblib")
    months = test["order_purchase_timestamp"].dt.to_period("M").astype(str).to_numpy()
    days = test["handover_time"].dt.strftime("%Y-%m-%d").to_numpy()
    flags = {name: monthly_flags(s, months, TOP_SHARE) for name, s in test_scores.items()}
    daily = {name: monthly_flags(s, days, TOP_SHARE) for name, s in test_scores.items()}
    metrics = [shared.metric_row("test", name, test, test_scores[name], flags[name], short_cut, daily[name]) for name in test_scores]
    for month in np.unique(months):
        part = months == month
        metrics += [shared.metric_row(month, name, test[part], test_scores[name][part], flags[name][part], short_cut, daily[name][part])
                    for name in test_scores]
    metrics = pd.DataFrame(metrics)
    whole = metrics[metrics["period"].eq("test")].set_index("ranking")["caught"]
    if (int(whole["model"]), int(whole["slack_rule"])) != shared.REPORT_CAUGHT["test"]:
        raise AssertionError("Test baselines do not reproduce the report")

    pairs = [(name, "slack_rule") for name in test_scores if name != "slack_rule"] + [(name, "model") for name in candidates]
    pairs += list(itertools.combinations(candidates, 2))
    intervals = test_bootstrap(test[LABEL].to_numpy().astype(bool), test_scores, months, test["past_promise_at_handover"].ne(1).to_numpy(), pairs)
    valid_windows = {name: [{"y": y, "candidate": s, "slack_rule": r} for y, s, r in zip(valid_y, scores[name], valid_rule)] for name in candidates}
    gates = gate_rows(candidates, roles, valid_windows, frame, rule_rows, metrics[metrics["period"].eq("test")], intervals) if candidates else pd.DataFrame()

    metrics.to_csv(HERE / "test_metrics.csv", index=False)
    intervals.to_csv(HERE / "test_bootstrap.csv", index=False)
    gates.to_csv(HERE / "gates.csv", index=False)
    pd.DataFrame(weights).reindex(WEIGHTS_TO_REPORT).to_csv(HERE / "model_weights.csv")

    print(stage_1_audit.round(4).to_string(index=False))
    print(stage_2_audit.round(4).to_string(index=False))
    print(frame.pivot(index="config", columns="period", values="pr_auc").loc[[BASELINE, *candidates]].round(3).to_string())
    print(metrics.round(3).to_string(index=False))
    print(gates.round(4).T.to_string())
    print(intervals.round(3).to_string(index=False))
    print(pd.DataFrame(weights).reindex(WEIGHTS_TO_REPORT).round(3).to_string())


if __name__ == "__main__":
    run()
