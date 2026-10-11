"""Focused invariant checks for the completed cadence replay."""

from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent


def main():
    scores = pd.read_csv(HERE / "scores.csv.gz", parse_dates=["handover_time"])
    audit = pd.read_csv(HERE / "fit_audit.csv", parse_dates=["refit_at", "fit_latest_outcome"])
    metrics = pd.read_csv(HERE / "metrics.csv")
    assert (audit["fit_latest_outcome"] < audit["refit_at"]).all()
    for phase, pair in (("validation", ["monthly", "quarterly"]),
                        ("test", ["monthly", "frozen"])):
        subset = scores.loc[scores["phase"].eq(phase)]
        assert subset["order_id"].is_unique
        assert np.isfinite(subset[pair]).all().all()
        whole = metrics.loc[metrics["phase"].eq(phase) & metrics["cohort"].eq("all")]
        assert whole["orders"].nunique() == whole["late"].nunique() == whole["reviewed"].nunique() == 1
    initial = scores.loc[scores["phase"].eq("test")
                         & scores["handover_time"].lt("2018-06-01")]
    assert np.allclose(initial["monthly"], initial["frozen"])
    print("cadence replay output invariants passed")


if __name__ == "__main__":
    main()
