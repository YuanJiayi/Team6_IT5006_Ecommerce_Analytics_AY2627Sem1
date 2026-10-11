"""Validation-only ablation of the three engineered handover slack inputs.

Run from the repository root:
    python Phase2/experiments/phase2_three_feature_validation_ablation.py

Keeps the selected logistic configuration, relative buyer-state backlog, all other inputs, and the original
five chronological validation folds fixed. No later-period rows are scored.
"""

import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from phase2_eta import chronological_folds  # noqa: E402
from phase2_handover_classifier import (CANDIDATES, LABEL, RELATIVE_BACKLOG_INPUT,  # noqa: E402
                                        fit_scores, load_cohort, ranking_metrics, variant_spec)

OUT = ROOT / "results" / "phase2" / "handover" / "three_feature_validation_ablation.csv"
ENGINEERED = {"remaining_slack", "past_promise_at_handover", "slack_shortfall"}


def main():
    selection = json.loads((OUT.parent / "selection.json").read_text())
    selected = selection["variants"]["primary"]["selected"]
    assert selected == "logistic_c0.01", f"Unexpected selected model: {selected}"
    family, params = next((f, p) for name, f, p in CANDIDATES if name == selected)
    assert family == "linear" and params == {"C": 0.01, "class_weight": None}

    table, spec, _ = load_cohort(ROOT)
    train = table.loc[table[spec["primary_split"]].eq("train")].reset_index(drop=True)
    assert train[spec["primary_split"]].eq("train").all()
    folds = chronological_folds(train, spec)
    assert len(folds) == 5

    rows = []
    for name in ("without_all_three", "full"):
        features = variant_spec(spec)
        assert ENGINEERED.issubset(features["numeric_features"])
        assert RELATIVE_BACKLOG_INPUT in features["numeric_features"]
        if name == "without_all_three":
            features["numeric_features"] = [f for f in features["numeric_features"] if f not in ENGINEERED]
        for window, (fit, valid) in enumerate(folds, 1):
            fit_rows, valid_rows = train.iloc[fit], train.iloc[valid]
            scores = fit_scores(family, params, features, fit_rows, valid_rows,
                                spec["linear_log_candidates"])
            metrics = ranking_metrics(valid_rows[LABEL].to_numpy(), scores)
            rows.append({"variant": name, "window": window,
                         "fit_orders": len(fit), "validation_orders": len(valid),
                         "pr_auc": metrics["pr_auc"], "roc_auc": metrics["roc_auc"],
                         "recall_top10": metrics["recall_top10"]})

    result = pd.DataFrame(rows)
    result.to_csv(OUT, index=False)
    print(result.groupby("variant", sort=False)[["pr_auc", "roc_auc", "recall_top10"]].mean().to_string())
    print(f"Wrote {OUT}")


if __name__ == "__main__":
    main()
