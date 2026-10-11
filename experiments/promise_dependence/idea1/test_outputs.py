"""Focused integrity checks for the paired ablation outputs."""
import json
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
RANKERS = {"selected_full", "drop_raw_only", "slack_only_promise_path", "slack_rule"}


def main():
    metrics = pd.read_csv(HERE / "metrics.csv")
    scores = pd.read_csv(HERE / "scores.csv.gz")
    contract = json.loads((HERE / "feature_contract.json").read_text())
    assert set(metrics.ranking) == RANKERS
    assert set(scores.period) == {"W1", "W2", "W3", "W4", "W5", "later"}
    assert not scores.duplicated(["period", "order_id"]).any()
    assert np.isfinite(scores[list(RANKERS)].to_numpy(dtype=float)).all()
    for _, group in metrics.groupby("period"):
        assert len(group) == len(RANKERS)
        for column in ("orders", "late", "reviewed"):
            assert group[column].nunique() == 1, (group.period.iloc[0], column)
    for period in scores.period.unique():
        subset = scores[scores.period == period]
        if period.startswith("W"):
            assert int(metrics[(metrics.period == period) & (metrics.ranking == "selected_full")].orders.iloc[0]) == len(subset)
    assert {"promised_days", "promise_slack"}.isdisjoint(contract["drop_raw_only"])
    assert {"handover_days", "route_transit_90d_at_handover"}.isdisjoint(contract["slack_only_promise_path"])
    assert {"remaining_slack", "past_promise_at_handover", "slack_shortfall"}.issubset(contract["slack_only_promise_path"])
    # Reference to the selected model's already saved five-window mean; checks reproduction rather than selection.
    original = metrics[(metrics.period.str.startswith("W")) & (metrics.ranking == "selected_full")]
    assert abs(original.pr_auc.mean() - 0.3823) < 0.0001
    print("All paired-cohort, equal-capacity, feature-contract, and reproduction checks passed.")


if __name__ == "__main__":
    main()
