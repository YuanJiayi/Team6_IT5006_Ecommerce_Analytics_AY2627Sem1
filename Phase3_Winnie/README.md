# Phase 3 (Winnie): final Phase 2 models

The three models presented in the [Phase 2 report](../Phase2/reports/phase2_report_v4.tex), refitted and saved for later work. This folder is Winnie's Phase 3 workspace, separate from `Phase2/`, and nothing there is changed: the build script imports the Phase 2 code and reads its data and saved results.

| File | Model | Output |
| --- | --- | --- |
| [`models/phase2_final_checkout_delivery_days_ridge.joblib`](models/phase2_final_checkout_delivery_days_ridge.joblib) | Ridge regression (α = 100), checkout inputs | Predicted purchase-to-delivery days |
| [`models/phase2_final_handover_delivery_days_ridge.joblib`](models/phase2_final_handover_delivery_days_ridge.joblib) | Ridge regression (α = 100), checkout inputs plus handover progress | Updated purchase-to-delivery days at carrier handover |
| [`models/phase2_final_late_warning_logistic.joblib`](models/phase2_final_late_warning_logistic.joblib) | Regularised logistic regression (C = 0.01), handover inputs | Probability of arriving after the promised calendar date |
| [`models/phase2_final_models_manifest.json`](models/phase2_final_models_manifest.json) | Input columns and output of each model | |

Each model is fitted once on the 75,039 primary training orders, whose outcomes were recorded before the 26 May 2018 test cutoff, as in the report's test scoring.

## Rebuild

From the repository root, with the Phase 2 requirements installed:

```
python Phase3_Winnie/rebuild_phase2_final_models.py
```

The script stops before saving if a refit does not reproduce the report's saved test results. The late-warning scores match exactly; regression metrics agree within 0.0001 days.

| Test-set check | Report | Refit |
| --- | --- | --- |
| Checkout MAE (days) | 5.00 | 5.00 |
| Handover MAE (days) | 3.97 | 3.97 |
| Late-warning PR-AUC / ROC-AUC | 0.49 / 0.86 | 0.489 / 0.863 |
| Top 10% of each month: recall, precision, late orders caught | 56%, 19.6%, 377 | 56%, 19.6%, 377 |

## Use

Build inputs with `load_cohort`, which computes the point-in-time features (remaining slack, recent route transit, regional backlog). Do not compute them by hand.

```python
import json, sys
import joblib

sys.path.insert(0, "Phase2")
from phase2_handover_classifier import load_cohort

table, spec, _ = load_cohort("Phase2")
m = json.load(open("Phase3_Winnie/models/phase2_final_models_manifest.json"))["models"]
cols = lambda k: m[k]["numeric_inputs"] + m[k]["categorical_inputs"]

days = joblib.load("Phase3_Winnie/models/phase2_final_handover_delivery_days_ridge.joblib").predict(table[cols("handover")])
risk = joblib.load("Phase3_Winnie/models/phase2_final_late_warning_logistic.joblib").predict_proba(table[cols("late_warning")])[:, 1]
```

Load with the pinned scikit-learn version recorded in the manifest.
