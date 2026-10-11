# Phase 2 final pipeline and evidence

This folder holds the **active Phase 2 pipeline**. Start with the [version 4 LaTeX source](reports/phase2_report_v4.tex), the submission draft: teammates' edited v1, corrected and condensed with the slack-shape inputs, the remaining-slack benchmark and the short-promise drift analysis. [Version 3](reports/phase2_report_v3.tex) is a uniformly reworded alternative; [version 2](reports/phase2_report_v2.tex) and [version 1](reports/phase2_report.tex) (with its [PDF](reports/phase2_report.pdf)) are kept for reference. Only v4's numbers are refreshed. Then follow the chain below. The preparation, feature, model, evidence and test code in this folder is the working source. The scripts run from this folder without depending on their old locations in the repository.

| Stage | Current work in this folder | What it establishes |
| --- | --- | --- |
| Raw data and preparation | `data/olist_*.csv`, `data/product_category_name_translation.csv`, `notebooks/phase2/data_prep.ipynb`, `phase2_features.py` | Order joins, cleaning, one-row-per-order construction, point-in-time history features, and split assignments. |
| Prepared contract | `data/phase2_order_table.csv`, `data/phase2_feature_spec.json`, `quality/ge_validation.py`, `quality/ge_validation.json` | Saved modelling table, allowed inputs and targets, and the separate Great Expectations schema/value audit. |
| Regression | `notebooks/phase2/regression.ipynb`, `phase2_eta.py`, with `phase2_regression.py`, `phase2_classification.py`, `phase2_selection.py`, and `experiments/phase2_recent_history_audit.py` as imported helpers | Checkout forecast updated at carrier handover; chronological validation, model selection and later-period results in `results/phase2/eta/`. |
| Classification | `notebooks/phase2/classification.ipynb`, `phase2_handover_classifier.py`, using the shared helpers above | Late-order risk ranking at handover; validation and later-period results in `results/phase2/handover/`. |
| Additional report checks | `experiments/phase2_unregularized_logistic_audit.py`, `reports/phase2_train_fit_evidence.py`, `reports/phase2_daily_queue_evidence.py`, `reports/phase2_complete_daily_replay.py` | Plain logistic comparison, train-fit diagnostics, daily queue sensitivity and retrospective workload replay. Their saved outputs are in `results/phase2/handover/` and `results/phase2/final/`. |
| Report evidence | `reports/phase2_final_evidence.py`, `reports/generated/`, `results/phase2/final/` | Tables, figures and numbers that the evidence script inserts into marked blocks in the report source. |
| Verification | `tests/test_phase2_*.py`, `requirements.txt` | Focused checks for the feature contract, point-in-time construction, temporal evaluation and selected models. |

## Reproduction order

Run commands from **this folder's root**. The preparation notebook reads the raw Olist tables and product-category translation, then writes the prepared CSV and feature spec. The review CSV is also included because the report's customer-experience context uses it. Model and diagnostic scripts use only the paths in this folder; the final evidence script updates `reports/phase2_report_v4.tex` here.

1. Install the Phase 2 dependencies from `requirements.txt`. The separate GX audit uses `quality/requirements.txt`.
2. Run `notebooks/phase2/data_prep.ipynb` from a fresh kernel if rebuilding the prepared table. This **overwrites** its included prepared CSV and feature spec, so preserve the snapshot if comparing exact saved results.
3. Optionally run `python quality/ge_validation.py --out quality/ge_validation_reproduced.json` on the rebuilt table. Its saved JSON in this folder is the teammate's prior run, not a validation performed by this packaging step.
4. Read `notebooks/phase2/regression.ipynb` and `notebooks/phase2/classification.ipynb` for concise, executed comparisons of the saved candidates and selections. The training scripts are the source of the model fits; run `python phase2_eta.py`, then `python phase2_handover_classifier.py` when a refit is needed. They rebuild the comparisons and write their saved result directories. Rerun the notebooks afterward to display the refreshed evidence.
5. For the report's extra checks, run the four scripts listed in the **Additional report checks** row and `python experiments/phase2_slack_shape_ablation.py`. Finally run `python reports/phase2_final_evidence.py` to refresh the marked report blocks and `reports/generated/`.

For a quick verification without refitting models, run `python -m unittest discover -s tests -p 'test_phase2_*.py'` from this folder.

The test period had already been inspected during report development. Reproducing saved test scores does not make it a fresh holdout. The model-selection claims in the report use the chronological validation windows.

## Submission handoff

The report PDF and source, its logo, the three Phase 2 notebooks, training and evidence scripts, feature contract, saved results and focused tests are all here. The PDF has a cover, seven main-report pages, and references and appendices. After editing the LaTeX source, regenerate the PDF and check that the GitHub link, author details, figures and declaration render correctly before submission. The notebooks are readable without retraining; the preparation notebook writes the included prepared table when executed.

## Scope and provenance

The folder includes code used by the final models and the specific diagnostics shown in the report. `phase2_classification.py` and `phase2_regression.py` remain because the selected scripts import their preprocessing, folds, and metrics; their older standalone checkout experiments are **not** additional final models. The checkout-classification notebook, promise-engine trials, old report drafts and unrelated explorations are excluded.

`quality/ge_validation.py` and `quality/ge_validation.json` were copied from `origin/add-great-expectations` at commit `6f346d30ebe241dde006d3db69d2c1d47eda11fa`. The saved JSON records a run on the schema-v2 table from the `josh` branch. That branch's committed table and feature spec are different Git objects from the present files, so rerun the audit on this snapshot before treating the saved pass counts as a fresh check of these exact bytes.

The raw Olist CSVs also remain at repository root because Phase 1 still reads them; this folder has its own focused dependency list. The report and logo are in this folder's `reports/` directory. Other Phase 2 files outside this folder are historical experiments or drafts, not inputs to this report. Teammates will see the move on GitHub only after it is committed and pushed.
