"""Exploratory as-of carrier-handover replay of logistic retraining cadence.

Run from repository root with ``it5006-proj/bin/python experiments/monthly_retraining/run.py``.
This does not change the saved Phase 2 selection or report.
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

REPO = Path(__file__).resolve().parents[2]
SOURCE = REPO / "phase2_final"
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(SOURCE))

from phase2_handover_classifier import (  # noqa: E402
    add_remaining_slack, fit_scores, load_cohort, variant_spec,
)

REFIT_START = pd.Timestamp("2017-09-22")
VALID_END = pd.Timestamp("2018-03-26")
TEST_START = pd.Timestamp("2018-05-26")
CAPACITY = 0.10
SHORT_PROMISE = 14.5


def refit_dates(mode: str, phase: str) -> list[pd.Timestamp]:
    if phase == "validation":
        first, last = REFIT_START, VALID_END
        if mode == "monthly":
            return [first, *pd.date_range("2017-10-01", last, freq="MS")]
        if mode == "quarterly":
            return [first, pd.Timestamp("2017-12-22"), pd.Timestamp("2018-03-22")]
    else:
        first, last = TEST_START, pd.Timestamp("2018-09-12")
        if mode == "monthly":
            return [first, *pd.date_range("2018-06-01", last, freq="MS")]
        if mode == "frozen":
            return [first]
    raise ValueError((mode, phase))


def training_mask(table: pd.DataFrame, date: pd.Timestamp) -> pd.Series:
    """Every fitted label and handover feature existed strictly before refit."""
    return (table["order_purchase_timestamp"].lt(date)
            & table["handover_time"].lt(date)
            & table["outcome_available_at"].lt(date))


def fit_replay(table: pd.DataFrame, scored: pd.DataFrame, spec: dict,
               dates: list[pd.Timestamp], policy: str, phase: str) -> tuple[np.ndarray, list[dict]]:
    spec_v = variant_spec(spec)
    result = np.full(len(scored), np.nan)
    audit = []
    for start, end in zip(dates, [*dates[1:], pd.Timestamp.max]):
        take = scored["handover_time"].ge(start) & scored["handover_time"].lt(end)
        if not take.any():
            continue
        fit = table.loc[training_mask(table, start)]
        target = scored.loc[take]
        assert fit["outcome_available_at"].max() < start
        assert fit["handover_time"].max() < start
        assert target["handover_time"].min() >= start
        result[np.flatnonzero(take.to_numpy())] = fit_scores(
            "linear", {"C": 0.01, "class_weight": None}, spec_v,
            fit, target, spec["linear_log_candidates"],
        )
        audit.append({"phase": phase, "policy": policy, "refit_at": start,
                      "score_before": end if end != pd.Timestamp.max else "end",
                      "fit_orders": len(fit), "fit_late_rate": fit["is_late"].mean(),
                      "fit_short_promise_share": fit["promised_days"].lt(SHORT_PROMISE).mean(),
                      "scored_orders": len(target),
                      "scored_late_rate": target["is_late"].mean(),
                      "fit_latest_outcome": fit["outcome_available_at"].max()})
        print(f"{phase} {policy} {start.date()}: train={len(fit)} score={len(target)}", flush=True)
    if not np.isfinite(result).all():
        raise AssertionError(f"{phase} {policy}: {np.isnan(result).sum()} scores missing")
    return result, audit


def slack_scores(table: pd.DataFrame, scored: pd.DataFrame, cutoff: pd.Timestamp) -> np.ndarray:
    # The saved rule uses a fit-sample median only for unknown route history.
    median = table.loc[training_mask(table, cutoff), "route_transit_90d_at_handover"].median()
    slack = (scored["promised_days"] - scored["handover_days"]
             - scored["route_transit_90d_at_handover"].fillna(median))
    return -slack.to_numpy(dtype=float)


def metric_rows(scored: pd.DataFrame, scores: dict[str, np.ndarray], phase: str) -> pd.DataFrame:
    """Same scored orders and top-10% capacity per handover month for each policy."""
    frame = scored.copy()
    frame["handover_month"] = frame["handover_time"].dt.to_period("M").astype(str)
    frame["purchase_month"] = frame["order_purchase_timestamp"].dt.to_period("M").astype(str)
    frame["short_promise"] = frame["promised_days"].lt(SHORT_PROMISE)
    y = frame["is_late"].to_numpy(dtype=int)
    rows = []
    for name, values in scores.items():
        frame[name] = values
        for cohort, mask in [
            ("all", np.ones(len(frame), dtype=bool)),
            ("short_promise", frame["short_promise"].to_numpy()),
            ("long_promise", ~frame["short_promise"].to_numpy()),
            *[(f"purchase_{month}", frame["purchase_month"].eq(month).to_numpy())
              for month in sorted(frame["purchase_month"].unique())],
            *[(f"handover_{month}", frame["handover_month"].eq(month).to_numpy())
              for month in sorted(frame["handover_month"].unique())],
        ]:
            local = frame.loc[mask]
            if len(local) == 0:
                continue
            yy, ss = local["is_late"].to_numpy(dtype=int), local[name].to_numpy()
            # Operational flags are set on the full monthly queue, then sliced to the cohort.
            flagged = frame[f"{name}_flag"].to_numpy()[mask] if f"{name}_flag" in frame else None
            rows.append({"phase": phase, "policy": name, "cohort": cohort,
                         "orders": len(local), "late": int(yy.sum()), "late_rate": yy.mean(),
                         "pr_auc": average_precision_score(yy, ss) if yy.sum() else np.nan,
                         "roc_auc": roc_auc_score(yy, ss) if len(np.unique(yy)) == 2 else np.nan,
                         "reviewed": int(flagged.sum()) if flagged is not None else np.nan,
                         "caught": int((flagged & yy.astype(bool)).sum()) if flagged is not None else np.nan})
    return pd.DataFrame(rows)


def add_flags(scored: pd.DataFrame, scores: dict[str, np.ndarray]) -> pd.DataFrame:
    out = scored.copy()
    month = out["handover_time"].dt.to_period("M").astype(str).to_numpy()
    for name, values in scores.items():
        flags = np.zeros(len(out), dtype=bool)
        for m in np.unique(month):
            ix = np.flatnonzero(month == m)
            chosen = ix[np.argsort(-values[ix], kind="stable")[:int(np.ceil(CAPACITY * len(ix)))]]
            flags[chosen] = True
        out[f"{name}_flag"] = flags
    return out


def evaluate(table: pd.DataFrame, spec: dict, phase: str) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if phase == "validation":
        scored = table.loc[table["cv_fold"].isin(range(5))
                           & table["handover_time"].ge(REFIT_START)
                           & table["handover_time"].lt(VALID_END)].copy().reset_index(drop=True)
        policies = ["monthly", "quarterly"]
    else:
        scored = table.loc[table["split"].eq("test")].copy().reset_index(drop=True)
        policies = ["monthly", "frozen"]
    scores = {"slack_rule": slack_scores(table, scored, REFIT_START if phase == "validation" else TEST_START)}
    audit = []
    for policy in policies:
        scores[policy], records = fit_replay(table, scored, spec, refit_dates(policy, phase), policy, phase)
        audit.extend(records)
    flagged = add_flags(scored, scores)
    metrics = metric_rows(flagged, scores, phase)
    score_table = flagged[["order_id", "order_purchase_timestamp", "handover_time", "promised_days",
                           "is_late", *[f"{name}_flag" for name in scores]]].copy()
    for name, values in scores.items():
        score_table[name] = values
    score_table["phase"] = phase
    return metrics, pd.DataFrame(audit), score_table


def main() -> None:
    table, spec, _ = load_cohort(SOURCE)
    metrics, audits, scores = [], [], []
    for phase in ["validation", "test"]:
        m, a, s = evaluate(table, spec, phase)
        metrics.append(m)
        audits.append(a)
        scores.append(s)
    HERE.mkdir(exist_ok=True)
    pd.concat(metrics, ignore_index=True).to_csv(HERE / "metrics.csv", index=False)
    pd.concat(audits, ignore_index=True).to_csv(HERE / "fit_audit.csv", index=False)
    pd.concat(scores, ignore_index=True).to_csv(HERE / "scores.csv.gz", index=False)
    later = scores[1].copy()
    elapsed = (later["handover_time"] - later["order_purchase_timestamp"]).dt.total_seconds() / 86400
    before_promise = elapsed.le(later["promised_days"])
    august = later["order_purchase_timestamp"].dt.to_period("M").astype(str).eq("2018-08")
    short = later["promised_days"].lt(SHORT_PROMISE)
    subgroups = {"before_promise": before_promise, "short_promise": short,
                 "short_before_promise": short & before_promise, "august_purchase": august,
                 "august_short": august & short,
                 "august_short_before_promise": august & short & before_promise}
    subgroup_rows = []
    for cohort, mask in subgroups.items():
        subset = later.loc[mask]
        y = subset["is_late"].to_numpy(dtype=int)
        n = int(np.ceil(CAPACITY * len(subset)))
        for policy in ("slack_rule", "monthly", "frozen"):
            order = np.argsort(-subset[policy].to_numpy(), kind="stable")
            subgroup_rows.append({"cohort": cohort, "policy": policy, "orders": len(subset),
                                  "late": int(y.sum()), "reviewed_at_subgroup_10pct": n,
                                  "caught_at_subgroup_10pct": int(y[order[:n]].sum()),
                                  "pr_auc": average_precision_score(y, subset[policy])})
    pd.DataFrame(subgroup_rows).to_csv(HERE / "later_subgroups.csv", index=False)
    paths = [SOURCE / "phase2_handover_classifier.py", SOURCE / "phase2_eta.py",
             SOURCE / "phase2_classification.py", SOURCE / "data/phase2_order_table.csv",
             SOURCE / "data/phase2_feature_spec.json"]
    (HERE / "source_sha256.json").write_text(json.dumps(
        {str(p.relative_to(REPO)): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}, indent=2) + "\n")
    print(pd.concat(metrics).query("cohort == 'all'").to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
