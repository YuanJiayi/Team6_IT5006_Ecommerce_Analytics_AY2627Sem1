"""Exploratory five-window ablation of raw promise inputs in handover risk ranking.

Run from repository root with ``python experiments/promise_dependence/idea1/run.py``.
Writes only beside this file; never changes the report or saved selection.
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
SRC = ROOT / "phase2_final"
sys.path.insert(0, str(SRC))
from phase2_eta import chronological_folds  # noqa: E402
from phase2_handover_classifier import (LABEL, PROMISE, fit_scores, load_cohort,
                                        monthly_flags, top_flags, variant_spec)  # noqa: E402

C = {"C": 0.01, "class_weight": None}
DROP_RAW = {"promised_days", "promise_slack"}
# The literal drop still permits exact reconstruction of the raw promise through
# remaining_slack + handover_days + route_transit_90d_at_handover.
DROP_EXACT_RECONSTRUCTION = DROP_RAW | {"handover_days", "route_transit_90d_at_handover"}
VARIANTS = {"selected_full": set(), "drop_raw_only": DROP_RAW,
            "slack_only_promise_path": DROP_EXACT_RECONSTRUCTION}
TOP = 0.10
SHORT_PROMISE = 14.5  # diagnostic subgroup identified before this ablation from opened later period


def sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def spec_without(spec, dropped):
    out = variant_spec(spec)
    out["numeric_features"] = [x for x in out["numeric_features"] if x not in dropped]
    assert DROP_RAW.isdisjoint(out["numeric_features"]) if dropped else True
    assert "remaining_slack" in out["numeric_features"]
    assert {"past_promise_at_handover", "slack_shortfall"}.issubset(out["numeric_features"])
    return out


def rule_scores(fit_rows, rows):
    median = fit_rows["route_transit_90d_at_handover"].median()
    return -(rows[PROMISE] - rows["handover_days"]
             - rows["route_transit_90d_at_handover"].fillna(median)).to_numpy()


def metrics(y, score, flags):
    y = np.asarray(y, dtype=bool)
    return {"orders": len(y), "late": int(y.sum()), "reviewed": int(flags.sum()),
            "caught": int((y & flags).sum()), "recall_top10": float((y & flags).sum() / y.sum()),
            "precision_top10": float((y & flags).sum() / flags.sum()),
            "pr_auc": float(average_precision_score(y, score)),
            "roc_auc": float(roc_auc_score(y, score))}


def run():
    table, spec, coverage = load_cohort(SRC)
    train = table[table[spec["primary_split"]].eq("train")].reset_index(drop=True)
    test = table[table[spec["primary_split"]].eq("test")].reset_index(drop=True)
    folds = chronological_folds(train, spec)
    specs = {name: spec_without(spec, dropped) for name, dropped in VARIANTS.items()}
    log_features = spec["linear_log_candidates"]
    records, score_parts = [], []

    for i, (fit, valid) in enumerate(folds, 1):
        fitting, scoring = train.iloc[fit], train.iloc[valid]
        period = f"W{i}"
        scores = {name: fit_scores("linear", C, changed, fitting, scoring, log_features)
                  for name, changed in specs.items()}
        scores["slack_rule"] = rule_scores(fitting, scoring)
        score_parts.append(pd.DataFrame({"period": period, "order_id": scoring["order_id"].to_numpy(),
                                         "is_late": scoring[LABEL].to_numpy(), **scores}))
        y = scoring[LABEL].to_numpy()
        for name, score in scores.items():
            records.append({"period": period, "ranking": name,
                            **metrics(y, score, top_flags(score, TOP))})
        print(period, {name: round(average_precision_score(y, score), 4) for name, score in scores.items()}, flush=True)

    # Opened later period: one 26-May fit, fixed throughout. Never use for selection.
    later = {name: fit_scores("linear", C, changed, train, test, log_features)
             for name, changed in specs.items()}
    later["slack_rule"] = rule_scores(train, test)
    months = test["order_purchase_timestamp"].dt.to_period("M").astype(str).to_numpy()
    y = test[LABEL].to_numpy()
    for name, score in later.items():
        records.append({"period": "later", "ranking": name,
                        **metrics(y, score, monthly_flags(score, months, TOP))})
        for month in np.unique(months):
            mask = months == month
            records.append({"period": month, "ranking": name,
                            **metrics(y[mask], score[mask], top_flags(score[mask], TOP))})
        # Subgroup-only top decile: equal capacity within the diagnostic subgroup.
        for group, mask in {"august_short": (months == "2018-08") & (test[PROMISE].to_numpy() < SHORT_PROMISE),
                            "august_short_prepromise": (months == "2018-08")
                            & (test[PROMISE].to_numpy() < SHORT_PROMISE)
                            & (test["past_promise_at_handover"].to_numpy() == 0)}.items():
            records.append({"period": group, "ranking": name,
                            **metrics(y[mask], score[mask], top_flags(score[mask], TOP))})
    score_parts.append(pd.DataFrame({"period": "later", "order_id": test["order_id"].to_numpy(),
                                     "month": months, "is_late": y,
                                     "promised_days": test[PROMISE].to_numpy(),
                                     "past_promise_at_handover": test["past_promise_at_handover"].to_numpy(),
                                     **later}))
    pd.DataFrame(records).to_csv(HERE / "metrics.csv", index=False)
    pd.concat(score_parts, ignore_index=True).to_csv(HERE / "scores.csv.gz", index=False, compression="gzip")
    sources = [SRC / "phase2_handover_classifier.py", SRC / "phase2_classification.py",
               SRC / "phase2_eta.py", SRC / "data/phase2_feature_spec.json",
               SRC / "data/phase2_order_table.csv", SRC / "data/olist_orders_dataset.csv",
               SRC / "data/olist_customers_dataset.csv", SRC / "experiments/phase2_recent_history_audit.py"]
    (HERE / "source_sha256.json").write_text(json.dumps({str(p.relative_to(ROOT)): sha(p) for p in sources}, indent=2) + "\n")
    (HERE / "feature_contract.json").write_text(json.dumps({name: changed["numeric_features"] + changed["categorical_features"]
                                                             for name, changed in specs.items()}, indent=2) + "\n")
    print(f"Cohort {coverage}; completed", flush=True)


if __name__ == "__main__":
    run()
