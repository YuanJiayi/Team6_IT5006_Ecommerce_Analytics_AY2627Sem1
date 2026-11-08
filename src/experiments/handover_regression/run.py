"""Handover decision study; uses the shared olist package without changing it.

Run from the repository root: python src/experiments/handover_regression/run.py
Outputs go to data/outputs/experiments/handover_regression/.
All new comparisons are exploratory: the historical test period has already been seen.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from olist.paths import EXPERIMENT_OUTPUTS, REPO_ROOT  # noqa: E402

OUTPUT_DIR = EXPERIMENT_OUTPUTS / "handover_regression"
ALPHAS = [0.01, 0.1, 1, 10, 100, 1000, 10000, 100000]
SPEEDUPS = [1, 2, 3]
SHARES = [0.05, 0.10, 0.20]


def paired_comparison(plain, regularised):
    gain = np.asarray(plain) - np.asarray(regularised)
    return {"mean_gain_days": float(gain.mean()), "sd_gain_days": float(gain.std(ddof=1)),
            "paired_se_days": float(gain.std(ddof=1) / np.sqrt(len(gain))),
            "plain_within_one_se": bool(gain.mean() <= gain.std(ddof=1) / np.sqrt(len(gain))),
            "window_gains_days": gain.tolist()}


def decision_rows(predictions):
    """Next-day 09:00 review of the previous day's handovers, using status then known.

    Calendar promise is exclusive midnight at the start of the following day.
    Completed orders and already-missed promises are outside the prevention queue.
    """
    out = predictions.copy()
    out["review_time"] = out.handover_time.dt.normalize() + pd.Timedelta(days=1, hours=9)
    out["deadline"] = out.order_estimated_delivery_date.dt.normalize() + pd.Timedelta(days=1)
    out["open_at_review"] = out.outcome_available_at.gt(out.review_time)
    out["before_deadline"] = out.review_time.lt(out.deadline)
    out["eligible"] = out.open_at_review & out.before_deadline
    for name in ["frozen_ridge", "selected", "route_rule"]:
        # The regression predicts total days; subtract elapsed time to get remaining days.
        arrival = out.order_purchase_timestamp + pd.to_timedelta(out[name], unit="D")
        out[name + "_gap"] = (arrival - out.deadline).dt.total_seconds() / 86400
    out["actual_gap"] = (out.outcome_available_at - out.deadline).dt.total_seconds() / 86400
    return out


def rank_queue(day, strategy, speedup):
    """No actual delivery/label enters ranking. All strategies fill the same capacity."""
    if strategy == "nearest_deadline":
        return day.sort_values(["deadline", "order_id"], kind="stable")
    gap = "route_rule_gap" if strategy == "route_slack" else "frozen_ridge_gap"
    if strategy == "recoverable_ridge":
        ranked = day.assign(recovery_band=~day[gap].between(0, speedup, inclusive="both"))
        # Among predicted recoverable orders, larger delay first; all others follow.
        return ranked.sort_values(["recovery_band", gap, "order_id"], ascending=[True, False, True])
    return day.sort_values([gap, "order_id"], ascending=[False, True])


def queue_metrics(predictions):
    frame = decision_rows(predictions)
    coverage = frame.groupby("period").agg(handovers=("order_id", "size"),
        open_at_review=("open_at_review", "sum"), before_deadline=("before_deadline", "sum"),
        eligible=("eligible", "sum"))
    rows = []
    for period, cohort in frame.groupby("period", sort=False):
        eligible = cohort[cohort.eligible]
        for share in SHARES:
            for speedup in SPEEDUPS:
                for strategy in ["nearest_deadline", "route_slack", "largest_predicted_delay",
                                 "recoverable_ridge", "random_expected"]:
                    selected = caught = prevented = oracle = 0.0
                    for _, day in eligible.groupby("review_time", sort=True):
                        k = int(np.ceil(share * len(day)))
                        late = day.actual_gap.ge(0)
                        # Assumption: each selected order's arrival moves earlier by exactly d days,
                        # bounded below by review time. With an unexpired promise this is equivalent.
                        recoverable = late & day.actual_gap.lt(speedup)
                        oracle += min(k, int(recoverable.sum()))
                        selected += k
                        if strategy == "random_expected":
                            caught += k * late.mean()
                            prevented += k * recoverable.mean()
                        else:
                            chosen = rank_queue(day, strategy, speedup).head(k)
                            caught += chosen.actual_gap.ge(0).sum()
                            prevented += (chosen.actual_gap.ge(0) & chosen.actual_gap.lt(speedup)).sum()
                    nlate = int(eligible.actual_gap.ge(0).sum())
                    nrecoverable = int((eligible.actual_gap.ge(0) & eligible.actual_gap.lt(speedup)).sum())
                    rows.append(dict(period=period, share=share, assumed_days_saved=speedup,
                        strategy=strategy, eligible=len(eligible), late_in_eligible=nlate,
                        scenario_recoverable=nrecoverable, reviewed=selected, late_reviewed=caught,
                        precision=caught / selected if selected else np.nan,
                        recall=caught / nlate if nlate else np.nan,
                        hypothetical_prevented=prevented,
                        hypothetical_prevented_per_100_reviews=100 * prevented / selected if selected else np.nan,
                        hindsight_oracle_prevented=oracle))
    return coverage.reset_index(), pd.DataFrame(rows)


def run(root=REPO_ROOT, output=OUTPUT_DIR):
    from olist.eta import CANDIDATES, load_stage_table, stage_spec, chronological_folds, fit_predict, make_stage_pipeline
    from olist.regression import regression_metrics

    output.mkdir(parents=True, exist_ok=True)
    table, spec, coverage = load_stage_table(root)
    promise = pd.read_csv(root / "data/raw/olist_orders_dataset.csv", usecols=["order_id", "order_estimated_delivery_date"],
                          parse_dates=["order_estimated_delivery_date"])
    table = table.merge(promise, on="order_id", validate="one_to_one")
    train = table[table[spec["primary_split"]].eq("train")].reset_index(drop=True)
    test = table[table[spec["primary_split"]].eq("test")].reset_index(drop=True)
    folds = chronological_folds(train, spec)
    changed = stage_spec(spec, "handover")
    configurations = []
    for name, family, params in CANDIDATES:
        if family in ("tree", "forest", "boost"):
            model = make_stage_pipeline(family, params, changed).named_steps["model"]
            configurations.append(dict(candidate=name, estimator=type(model).__name__,
                explicit_overrides=params, effective_settings=model.get_params(deep=False)))
    (output / "submitted_tree_configurations.json").write_text(json.dumps(configurations, indent=2) + "\n")
    candidates = [("linear", "linear", {})] + [(f"ridge_{a:g}", "ridge", {"alpha": a}) for a in ALPHAS]
    scores, predictions = [], {}
    for name, family, params in candidates:
        predictions[name] = []
        for w, (fit, valid) in enumerate(folds):
            p = fit_predict(family, params, changed, train.iloc[fit], train.iloc[valid])
            predictions[name].append(p)
            scores.append(dict(candidate=name, window=w + 1, **regression_metrics(train.iloc[valid].delivery_days, p)))
        print(name, np.mean([r["mae"] for r in scores if r["candidate"] == name]), flush=True)
    cv = pd.DataFrame(scores)
    pivot = cv.pivot(index="window", columns="candidate", values="mae")
    best_ridge = pivot.drop(columns="linear").mean().idxmin()
    comparison = paired_comparison(pivot.linear, pivot[best_ridge])
    selected = "linear" if comparison["plain_within_one_se"] else best_ridge
    family, params = next((f, p) for n, f, p in candidates if n == selected)
    audit = {"extended_best_ridge": best_ridge, "selected": selected,
             "extended_plain_vs_best": comparison,
             "frozen_plain_vs_ridge100": paired_comparison(pivot.linear, pivot.ridge_100)}
    old = pd.read_csv(root / "data/outputs/phase2/eta/cv_metrics.csv")
    audit["submitted_plain_vs_ridge100"] = {}
    for stage in ["checkout", "handover"]:
        part = old[old.stage.eq(stage)].pivot(index="window", columns="candidate", values="mae")
        audit["submitted_plain_vs_ridge100"][stage] = paired_comparison(part.linear_baseline, part.ridge_100)
    keys = ["order_id", "order_purchase_timestamp", "handover_time", "outcome_available_at",
            "order_estimated_delivery_date", "delivery_days", "handover_days"]
    frames, metrics = [], []
    for w, (fit, valid) in enumerate(folds):
        rows = train.iloc[valid][keys].copy()
        rows["period"] = f"validation_{w+1}"
        rows["frozen_ridge"] = predictions["ridge_100"][w]
        rows["selected"] = predictions[selected][w]
        rows["route_rule"] = fit_predict("rule", {}, changed, train.iloc[fit], train.iloc[valid])
        frames.append(rows)
    rows = test[keys].copy()
    rows["period"] = "test_descriptive"
    for name, fam, par in [("frozen_ridge", "ridge", {"alpha": 100}), ("selected", family, params),
                            ("route_rule", "rule", {})]:
        rows[name] = fit_predict(fam, par, changed, train, test)
    frames.append(rows)
    pred = pd.concat(frames, ignore_index=True)
    for period, part in pred.groupby("period", sort=False):
        for name in ["frozen_ridge", "selected", "route_rule"]:
            metrics.append(dict(period=period, model=name, orders=len(part),
                **regression_metrics(part.delivery_days, part[name]),
                negative_remaining=int((part[name] < part.handover_days).sum()),
                p90_absolute_error=float(np.quantile(np.abs(part.delivery_days - part[name]), .9))))
    cover, queue = queue_metrics(pred)
    cv.to_csv(output / "extended_grid_windows.csv", index=False)
    cv.groupby("candidate").mean(numeric_only=True).drop(columns="window").to_csv(output / "extended_grid_summary.csv")
    pd.DataFrame(metrics).to_csv(output / "prediction_metrics.csv", index=False)
    cover.to_csv(output / "queue_coverage.csv", index=False)
    queue.to_csv(output / "queue_scenarios.csv", index=False)
    pred.to_csv(output / "predictions.csv.gz", index=False, compression={"method": "gzip", "mtime": 0})
    audit.update(coverage=coverage, alphas=ALPHAS, queue_model="frozen ridge alpha=100, not selected using queue results",
        selection_rule="Tune ridge on mean validation MAE; prefer plain linear if within paired one SE of best ridge",
        scope="Exploratory extension on previously used validation/test periods; no independent confirmatory test",
        evaluation_unit="purchase-cohort validation windows; previous-day handovers reviewed next morning at 09:00",
        se_limitation="Five overlapping training windows; paired SE is descriptive, not an independent-sample significance test",
        versions={"sklearn": sklearn.__version__, "pandas": pd.__version__, "numpy": np.__version__},
        source_sha256={str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in [root / "src/olist/eta.py", root / "src/olist/regression.py", root / "data/processed/phase2_order_table.csv"]})
    (output / "selection_audit.json").write_text(json.dumps(audit, indent=2) + "\n")
    print(json.dumps(audit, indent=2), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=REPO_ROOT)
    parser.add_argument("--output", type=Path, default=OUTPUT_DIR)
    args = parser.parse_args()
    run(args.root.resolve(), args.output.resolve())
