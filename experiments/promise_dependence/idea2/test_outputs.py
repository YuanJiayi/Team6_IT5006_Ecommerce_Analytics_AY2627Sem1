"""Focused consistency checks for the saved exploratory offset experiment."""

import json
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]


def main():
    from run import EXCLUDE, PRIMARY_CAPS, correction_spec  # noqa: PLC0415
    spec = json.loads((REPO / "phase2_final/data/phase2_feature_spec.json").read_text())
    features = correction_spec(spec)
    assert not EXCLUDE.intersection(features["numeric_features"] + features["categorical_features"])

    selection = json.loads((HERE / "selection.json").read_text())
    assert selection["cap_log_odds"] in PRIMARY_CAPS
    assert selection["correction_c"] in (0.001, 0.01, 0.1)

    valid = pd.read_csv(HERE / "validation_windows.csv")
    assert set(valid.period) == {f"W{i}" for i in range(1, 6)}
    assert valid.groupby(["period", "ranking", "correction_c", "cap_log_odds"],
                         dropna=False).size().eq(1).all()
    assert valid["reviews"].eq(valid["orders"].mul(0.1).apply(lambda x: int(-(-x // 1)))).all()

    later = pd.read_csv(HERE / "later_diagnostic.csv")
    for period in later.period.unique():
        group = later[later.period.eq(period)]
        assert group.reviews.nunique() == 1
        assert group.orders.nunique() == 1
        assert group.late.nunique() == 1
    assert len(later.loc[later.period.eq("later_august_short_pre_promise")]) == 7

    audit = pd.read_csv(HERE / "fit_audit.csv", parse_dates=["max_known_outcome", "min_scored_purchase"])
    assert (audit.max_known_outcome < audit.min_scored_purchase).all()

    print("Offset experiment output checks passed")


if __name__ == "__main__":
    main()
