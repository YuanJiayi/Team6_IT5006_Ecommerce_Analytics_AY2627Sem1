"""Checks that the saved checkout-promise outputs still carry the numbers quoted in the findings doc."""

import json
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parents[3] / "data" / "outputs" / "experiments" / "checkout_promise"


def close(value, expected, tol):
    assert abs(value - expected) <= tol, (value, expected)


def main():
    summary = json.loads((HERE / "summary.json").read_text())
    head = summary["headline_test"]
    close(head["olist"]["mean_promise"], 22.05, 0.05)
    close(head["olist"]["late_rate"], 0.035, 0.001)
    # Includes the slow orders delivered after the cutoff in the monthly cohorts (July's buffer depends on them).
    close(head["region_3.5pct"]["mean_promise"], 21.63, 0.1)
    close(head["region_3.5pct"]["late_rate"], 0.0192, 0.002)
    close(head["promise_at_olist_late_rate"], 18.61, 0.1)
    close(head["late_rate_at_olist_promise"], 0.0181, 0.002)
    close(head["promise_at_2.8pct_late"], 19.60, 0.1)
    # The recommended policy is shorter and less late than Olist on test.
    assert head["region_3.5pct"]["mean_promise"] < head["olist"]["mean_promise"]
    assert head["region_3.5pct"]["late_rate"] < head["olist"]["late_rate"]

    state = pd.read_csv(HERE / "accuracy_and_state.csv").set_index("split")
    close(state.loc["val", "mae_ridge_no_promise"], 5.78, 0.02)
    close(state.loc["val", "mae_ridge_full"], 5.76, 0.02)

    months = pd.read_csv(HERE / "monthly_recalibration_by_month.csv")
    recal = months[months.policy.str.startswith("Monthly")]
    assert (recal.cohort_orders >= 100).all()
    # July's cohort (May purchases) must include the slow orders delivered between the cutoff and 1 July.
    assert (recal[recal.month.eq("2018-07")].cohort_orders > 6500).all()

    rules = pd.read_csv(HERE / "replace_vs_threshold.csv").set_index("policy")
    best = rules.loc["Replace all"]
    threshold = rules[rules.index.str.startswith("Ours only if it differs")]
    assert (threshold.late_rate > best.late_rate).all()
    # At the same late rate, replacing every promise (target moved) is shorter than any threshold rule.
    assert (threshold.mean_promise > threshold.replace_all_promise_at_same_late).all()

    seasonal = summary["seasonal_validation"]
    close(seasonal["region_promise_at_olist_late"], seasonal["olist_promise"], 0.5)
    print("checkout promise output checks passed")


if __name__ == "__main__":
    main()
