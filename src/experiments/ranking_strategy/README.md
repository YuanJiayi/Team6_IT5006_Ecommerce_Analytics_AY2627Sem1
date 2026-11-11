# Late-order ranking strategy (Phase 3, Task 4)

Winnie's Phase 3 experiments on how to rank handed-over parcels for review: the Phase 2 late-warning model, the remaining-slack rule, mixes of the two, promise-relative inputs, monthly refits and a daily re-ranking simulation. Candidates, gates and metrics are declared in advance in [EVALUATION_CRITERIA.md](EVALUATION_CRITERIA.md).

Code lives here; results and saved models are in [`data/outputs/experiments/ranking_strategy/`](../../../data/outputs/experiments/ranking_strategy/), one folder per experiment with the same name as its code folder.

| Code | What it tests |
| --- | --- |
| `01_ranking_baselines/run.py` | The rule, the Phase 2 model, closest to deadline and the regression ranking (ridge alpha=100), before any mix |
| `02_model_rule_mix/run.py` | Two fixed mixes of the model and the remaining-slack rule |
| `03_improve_logistic/run.py` | Regularisation, training window and signal choices for the logistic model |
| `07_promise_relative_inputs/run.py` | Promise-relative inputs in place of the Phase 2 inputs |
| `04_mix_ratios/run.py` | Mix ratios for the Phase 2 and promise-relative models (uses 07) |
| `05_recent_monthly_refits/run.py` | Monthly refits on recent history only |
| `06_not_past_training/run.py` | The model fitted only on orders still before their promised date at handover |
| `08_final_comparison/run.py` | Final comparison of rule, model, 50/50 mix and changing ratio (uses 04) |
| `10_mix_monthly_refits/run.py` | The 50/50 mix under monthly refits |
| `11_daily_rerank/run.py` | Daily 9am re-ranking simulation, following `src/reporting/complete_daily_replay.py` |
| `12_rerank_monthly_refits/run.py` | Daily re-ranking combined with monthly refits |
| `13_deadline_comparison/run.py` | Closest to deadline against the final strategy: daily replay, shortfalls and test comparisons (uses 01, 10 and 12) |
| `14_gmv_estimate/run.py` | Yearly item value (GMV) of late parcels the blend's review list would reach, scaled from the test catch rates to September 2017–August 2018 (uses 10) |
| `15_weekly_intervals/run.py` | Test intervals from 2,000 resamples of whole weeks, for ranked lists and the daily replay (uses 12 and 13) |
| `16_regression_comparator/run.py` | The regression ranking added to the shortfalls and weekly intervals (uses 01, 13 and 15) |
| `17_regression_alpha1000/run.py` | The regression ranking rescored with the Phase 3 selection, ridge alpha=1000 (uses 01, 13, 15 and 16) |
| `18_report_checks/run.py` | Review-list precision, recall, F1 and MCC for five rankings, and the weekly interval for monthly refits against a single fit |
| `19_class_weights_insample/run.py` | Balanced class weights at each C of the extended grid, and the Phase 2 classifier's in-sample PR-AUC and ROC-AUC |
| `task4_common.py` | Shared scoring and gate checks for experiments 05 onwards |

### Where the final report's late-warning numbers come from

| Report content | Experiment |
| --- | --- |
| Phase 3 variants (Table 7): regularisation, recency, ZIP-3 and seller history | 03 |
| Promise-relative inputs; fitted only on orders before their date | 07; 06 |
| Monthly refits; blend weights and promise-varying weight | 05; 02 and 04 |
| Recommended blend with monthly refits | 10 |
| Rule, closest to deadline and regression comparators (Tables 7 and 8) | 01, 13, 17 |
| Review shares of 1, 2, 5 and 20% | 08 |
| Daily replay (Table 9) | 11, 12, 13 |
| Test intervals | 15, 17, 18 |
| Review-list quality (Table 17) | 18 |
| Class weights (Section 5.3, Table 18) and the classifier's training scores (Table 4) | 19 |
| Presentation only: ~2,400 late parcels and ~R$0.46M a year reached while still rescuable (58% catch rate by value) | 14 |
| PR-AUC by period and largest shortfalls (Table 19) | 13_deadline_comparison (rule, model, blend with monthly refits, closest to deadline) and 17 (regression ranking) |

Run each script from the repository root, in the order above, for example:

```bash
python src/experiments/ranking_strategy/02_model_rule_mix/run.py
```

Each script stops if its baselines do not reproduce the Phase 2 results or the earlier experiments. Experiments 01–12, 13_deadline_comparison, 16 and the rebuild script use the Phase 2 regression (ridge alpha=100) as their reference; 17 and 18 use the Phase 3 selection (alpha=1000) for the regression ranking.

## Final Phase 2 models used as the baseline

The three models presented in the Phase 2 report, refitted and saved for this work in [`data/outputs/experiments/ranking_strategy/models/`](../../../data/outputs/experiments/ranking_strategy/models/). They are separate from the saved models in `data/outputs/models/`.

| File | Model | Output |
| --- | --- | --- |
| `phase2_final_checkout_delivery_days_ridge.joblib` | Ridge regression (α = 100), checkout inputs | Predicted purchase-to-delivery days |
| `phase2_final_handover_delivery_days_ridge.joblib` | Ridge regression (α = 100), checkout inputs plus handover progress | Updated purchase-to-delivery days at carrier handover |
| `phase2_final_late_warning_logistic.joblib` | Regularised logistic regression (C = 0.01), handover inputs | Probability of arriving after the promised calendar date |
| `phase2_final_models_manifest.json` | Input columns and output of each model | |

Each model is fitted once on the 75,039 primary training orders, whose outcomes were recorded before the 26 May 2018 test cutoff, as in the report's test scoring.

### Rebuild

```bash
python src/experiments/ranking_strategy/rebuild_phase2_final_models.py
```

The script stops before saving if a refit does not reproduce the report's saved test results. The late-warning scores match exactly; regression metrics agree within 0.0001 days.

| Test-set check | Report | Refit |
| --- | --- | --- |
| Checkout MAE (days) | 5.00 | 5.00 |
| Handover MAE (days) | 3.97 | 3.97 |
| Late-warning PR-AUC / ROC-AUC | 0.49 / 0.86 | 0.489 / 0.863 |
| Top 10% of each month: recall, precision, late orders caught | 56%, 19.6%, 377 | 56%, 19.6%, 377 |

### Use

Build inputs with `load_cohort`, which computes the point-in-time features (remaining slack, recent route transit, regional backlog). Do not compute them by hand.

```python
import json
import joblib

from olist.handover_classifier import load_cohort
from olist.paths import EXPERIMENT_OUTPUTS, REPO_ROOT

models = EXPERIMENT_OUTPUTS / "ranking_strategy" / "models"
table, spec, _ = load_cohort(REPO_ROOT)
m = json.loads((models / "phase2_final_models_manifest.json").read_text())["models"]
cols = lambda k: m[k]["numeric_inputs"] + m[k]["categorical_inputs"]

days = joblib.load(models / "phase2_final_handover_delivery_days_ridge.joblib").predict(table[cols("handover")])
risk = joblib.load(models / "phase2_final_late_warning_logistic.joblib").predict_proba(table[cols("late_warning")])[:, 1]
```

Load with the pinned scikit-learn version recorded in the manifest.
