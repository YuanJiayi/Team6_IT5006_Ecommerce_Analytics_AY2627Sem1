"""Validation-only check of the unregularized handover logistic baseline.

Run from the Phase2 folder:
    python experiments/phase2_unregularized_logistic_audit.py

Reuses the saved validation scores for the original candidate grid and fits only
the new baseline on the same five chronological folds. It never opens the test
period or changes the canonical handover results.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from phase2_eta import chronological_folds  # noqa: E402
from phase2_handover_classifier import (GROUPS, LABEL, fit_scores, load_cohort,  # noqa: E402
                                        ranking_metrics, variant_spec)
from phase2_selection import two_stage_select  # noqa: E402


def main():
    out = ROOT / "results/phase2/handover/unregularized_baseline_audit"
    out.mkdir(parents=True, exist_ok=True)
    saved = pd.read_csv(ROOT / "results/phase2/handover/validation_windows.csv")
    saved = saved[saved.candidate.ne("logistic_unregularized")]
    table, spec, _ = load_cohort(ROOT)
    train = table[table[spec["primary_split"]].eq("train")].reset_index(drop=True)
    folds = chronological_folds(train, spec)
    assert len(folds) == 5
    rows, report = [], {}
    for variant, variant_features in (("primary", variant_spec(spec)),
                                      ("checkout", variant_spec(spec, handover=False))):
        old = saved[saved.variant.eq(variant)]
        expected = old[old.candidate.eq("logistic_c1")].sort_values("window")
        assert expected.window.tolist() == list(range(5))
        assert expected.validation_orders.tolist() == [len(valid) for _, valid in folds]
        variant_rows = []
        for window, (fit, valid) in enumerate(folds):
            valid_rows = train.iloc[valid]
            probabilities = fit_scores(
                "linear", {"C": np.inf, "class_weight": None}, variant_features,
                train.iloc[fit], valid_rows, spec["linear_log_candidates"])
            variant_rows.append({
                "variant": variant, "candidate": "logistic_unregularized", "window": window,
                "fit_orders": len(fit), "validation_orders": len(valid),
                **ranking_metrics(valid_rows[LABEL].to_numpy(), probabilities),
            })
        rows.extend(variant_rows)
        combined = pd.concat([old, pd.DataFrame(variant_rows)], ignore_index=True)
        per_window = combined.pivot(index="window", columns="candidate", values="pr_auc").to_dict("list")
        original_groups = [
            (group, "logistic_c1" if group == "linear" else baseline,
             [name for name in configs if name != "logistic_unregularized"])
            for group, baseline, configs in GROUPS
        ]
        original_choice, _ = two_stage_select(original_groups, per_window, higher_is_better=True)
        choice, details = two_stage_select(GROUPS, per_window, higher_is_better=True)
        means = combined.groupby("candidate")["pr_auc"].mean()
        report[variant] = {
            "unregularized_mean_pr_auc": float(means["logistic_unregularized"]),
            "default_c1_mean_pr_auc": float(means["logistic_c1"]),
            "selected_c001_mean_pr_auc": float(means["logistic_c0.01"]),
            "original_selected": original_choice,
            "selected_with_unregularized_baseline": choice,
            "stage2_lineup": details["lineup"],
            "stage2": details["stage2"],
        }
        print(f"{variant}: unregularized PR-AUC {report[variant]['unregularized_mean_pr_auc']:.6f}; "
              f"selected {choice} (previously {original_choice})", flush=True)
    pd.DataFrame(rows).to_csv(out / "validation_windows.csv", index=False)
    (out / "comparison.json").write_text(json.dumps({
        "sklearn_version": sklearn.__version__,
        "baseline_parameters": {"solver": "lbfgs", "C": "infinity (no regularization)",
                                "class_weight": None, "max_iter": 3000},
        "variants": report,
    }, indent=2) + "\n")
    print(f"Saved validation-only evidence to {out}")


if __name__ == "__main__":
    main()
