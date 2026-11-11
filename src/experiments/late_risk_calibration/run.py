"""Calibration of the selected late-risk classifier: predicted late probability against the actual late rate.

Run from the repository root: python src/experiments/late_risk_calibration/run.py

The selected Phase 2 classifier (data/outputs/phase2/handover/selection.json) is refitted on each chronological
validation window's fitting rows and scored on the window. The test period uses the saved test scores in
data/outputs/phase2/handover/test_scores.csv, after checking that a refit on all primary training orders reproduces
them. Calibration is summarised as the mean predicted probability divided by the actual late rate, overall, per
period and per score decile. Outputs go to data/outputs/experiments/late_risk_calibration/.

This is a diagnostic. It does not change the model, which is used as a ranking (percentiles and top shares), not as
a probability.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from olist.eta import chronological_folds  # noqa: E402
from olist.handover_classifier import CANDIDATES, LABEL, fit_scores, load_cohort, variant_spec  # noqa: E402
from olist.paths import EXPERIMENT_OUTPUTS, PHASE2_OUTPUTS  # noqa: E402

OUT = EXPERIMENT_OUTPUTS / "late_risk_calibration"
DECILES = 10


def calibration_row(scope, period, y, p):
    return {"scope": scope, "period": period, "orders": len(y), "late": int(y.sum()),
            "actual_late_rate": float(y.mean()), "mean_predicted": float(p.mean()),
            "predicted_to_actual": float(p.mean() / y.mean())}


def decile_rows(scope, y, p):
    frame = pd.DataFrame({"y": y, "p": p})
    frame["decile"] = pd.qcut(frame["p"].rank(method="first"), DECILES, labels=False) + 1
    out = frame.groupby("decile").agg(orders=("y", "size"), actual_late_rate=("y", "mean"),
                                      mean_predicted=("p", "mean")).reset_index()
    out["predicted_to_actual"] = out["mean_predicted"] / out["actual_late_rate"]
    return out.assign(scope=scope)[["scope", "decile", "orders", "actual_late_rate", "mean_predicted",
                                    "predicted_to_actual"]]


def run():
    OUT.mkdir(parents=True, exist_ok=True)
    selected = json.loads((PHASE2_OUTPUTS / "handover/selection.json").read_text())["variants"]["primary"]["selected"]
    family, params = next((f, p) for n, f, p in CANDIDATES if n == selected)
    table, spec, _ = load_cohort()
    spec_v, log_features = variant_spec(spec), spec["linear_log_candidates"]
    train = table[table[spec["primary_split"]].eq("train")].reset_index(drop=True)
    test = table[table[spec["primary_split"]].eq("test")].reset_index(drop=True)

    rows, val_y, val_p = [], [], []
    for window, (fit, valid) in enumerate(chronological_folds(train, spec), start=1):
        fit_rows, rows_v = train.iloc[fit], train.iloc[valid]
        p = fit_scores(family, params, spec_v, fit_rows, rows_v, log_features)
        y = rows_v[LABEL].to_numpy().astype(float)
        rows.append({**calibration_row("validation", f"W{window}", y, p),
                     "fitting_late_rate": float(fit_rows[LABEL].mean())})
        val_y.append(y)
        val_p.append(p)
    val_y, val_p = np.concatenate(val_y), np.concatenate(val_p)
    rows.append(calibration_row("validation", "W1-W5 pooled", val_y, val_p))

    saved = pd.read_csv(PHASE2_OUTPUTS / "handover/test_scores.csv")
    refit = fit_scores(family, params, spec_v, train, test, log_features)
    merged = saved.merge(pd.DataFrame({"order_id": test["order_id"], "refit": refit}), on="order_id",
                         validate="one_to_one")
    gap = float(np.abs(merged["refit"] - merged["score"]).max())
    if len(merged) != len(test) or gap > 1e-9:
        raise AssertionError(f"Refit does not reproduce the saved test scores (max difference {gap:.2e})")
    y_t, p_t = saved["is_late"].to_numpy().astype(float), saved["score"].to_numpy()
    for month, part in saved.groupby("month"):
        rows.append(calibration_row("test", str(month), part["is_late"].to_numpy().astype(float),
                                    part["score"].to_numpy()))
    rows.append({**calibration_row("test", "overall", y_t, p_t), "fitting_late_rate": float(train[LABEL].mean())})

    summary = pd.DataFrame(rows)
    deciles = pd.concat([decile_rows("validation W1-W5 pooled", val_y, val_p), decile_rows("test", y_t, p_t)])
    summary.to_csv(OUT / "summary.csv", index=False)
    deciles.to_csv(OUT / "deciles.csv", index=False)
    test_row = summary[(summary.scope == "test") & (summary.period == "overall")].iloc[0]
    val_row = summary[summary.period == "W1-W5 pooled"].iloc[0]
    headline = {
        "model": selected,
        "test_mean_predicted": test_row["mean_predicted"], "test_actual_late_rate": test_row["actual_late_rate"],
        "test_predicted_to_actual": test_row["predicted_to_actual"],
        "training_late_rate": test_row["fitting_late_rate"],
        "validation_pooled_predicted_to_actual": val_row["predicted_to_actual"],
        "test_deciles_overpredicted": int((deciles[deciles.scope == "test"].predicted_to_actual > 1).sum()),
        "refit_max_abs_difference_from_saved_scores": gap,
    }
    (OUT / "summary.json").write_text(json.dumps(headline, indent=2) + "\n")
    print(summary.round(4).to_string(index=False))
    print(deciles.round(4).to_string(index=False))
    print(json.dumps(headline, indent=2))


if __name__ == "__main__":
    run()
