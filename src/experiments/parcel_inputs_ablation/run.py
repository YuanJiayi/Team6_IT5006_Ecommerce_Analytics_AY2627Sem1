"""Validation-only ablation of the parcel inputs (price, freight, weight, volume) in the checkout ridge.

Run from the repository root: python src/experiments/parcel_inputs_ablation/run.py
Uses the selected checkout model from data/outputs/phase2/eta/selection.json and the five chronological windows.
Each variant drops a set of inputs and is compared with the full model on the same windows (paired by window).
Outputs go to data/outputs/experiments/parcel_inputs_ablation/.
"""

from __future__ import annotations

import copy
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from olist.eta import (CANDIDATES, CHECKOUT_STAGE, TARGET, chronological_folds, fit_predict,  # noqa: E402
                       load_stage_table, stage_spec)
from olist.paths import EXPERIMENT_OUTPUTS, PHASE2_OUTPUTS  # noqa: E402
from olist.regression import regression_metrics  # noqa: E402
from olist.selection import paired_gap  # noqa: E402

OUT = EXPERIMENT_OUTPUTS / "parcel_inputs_ablation"
DROPS = {
    "full": [],
    "without_all_four": ["total_price", "total_freight", "total_weight_g", "total_volume_cm3"],
    "without_freight": ["total_freight"],
    "without_price": ["total_price"],
    "without_weight_volume": ["total_weight_g", "total_volume_cm3"],
}


def run():
    OUT.mkdir(parents=True, exist_ok=True)
    selected = json.loads((PHASE2_OUTPUTS / "eta/selection.json").read_text())["checkout_selected_candidate"]
    family, params = next((f, p) for n, f, p in CANDIDATES if n == selected)
    table, spec, _ = load_stage_table()
    train = table[table[spec["primary_split"]].eq("train")].reset_index(drop=True)
    folds = chronological_folds(train, spec)
    base = stage_spec(spec, CHECKOUT_STAGE)

    rows = []
    for variant, dropped in DROPS.items():
        spec_v = copy.deepcopy(base)
        spec_v["numeric_features"] = [c for c in base["numeric_features"] if c not in dropped]
        for window, (fit, valid) in enumerate(folds):
            predicted = fit_predict(family, params, spec_v, train.iloc[fit], train.iloc[valid])
            rows.append({"variant": variant, "window": window, **regression_metrics(train.iloc[valid][TARGET], predicted)})
    windows = pd.DataFrame(rows)
    windows.to_csv(OUT / "validation_windows.csv", index=False)

    mae = windows.pivot(index="window", columns="variant", values="mae")
    summary = []
    for variant in DROPS:
        gap, se = paired_gap(mae[variant], mae["full"], higher_is_better=False)
        summary.append({"model": selected, "variant": variant, "dropped": " ".join(DROPS[variant]),
                        "mean_mae": float(mae[variant].mean()), "mae_increase": gap, "paired_se": se,
                        "worse_in_windows": int((mae[variant] > mae["full"]).sum())})
    summary = pd.DataFrame(summary)
    summary.to_csv(OUT / "summary.csv", index=False)
    print(summary.round(4).to_string(index=False))


if __name__ == "__main__":
    run()
