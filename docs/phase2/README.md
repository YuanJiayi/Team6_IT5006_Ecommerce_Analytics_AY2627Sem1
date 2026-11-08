# Phase 2 — keeping delivery promises at Olist

[Final report source](report/phase2_report_v4.tex) · [Evidence map](report_evidence_map.md) · [Assumptions](phase2_assumptions.md)

The code, notebooks, data and saved results supporting the report are spread across the repository by role. The model scripts perform the fits; the notebooks explain the preparation and display model comparisons. All commands below run from the repository root.

## What is where

| Path | Purpose |
| --- | --- |
| [`data/raw/`](../../data/raw/), [`data/processed/`](../../data/processed/) | Olist source CSVs, the prepared one-row-per-order table, and the feature specification. |
| [`notebooks/phase2_modelling/`](../../notebooks/phase2_modelling/) | Preparation, regression and classification narratives. The regression and classification notebooks read saved model outputs. |
| [`src/olist/features.py`](../../src/olist/features.py) | Order features, history features and point-in-time construction. |
| [`src/olist/eta.py`](../../src/olist/eta.py), [`src/olist/handover_classifier.py`](../../src/olist/handover_classifier.py) | Selected regression and classifier comparisons, temporal windows, test scores and selection records. The other modules in `src/olist/` are imported helpers. |
| [`src/experiments/`](../../src/experiments/) | The specific ablations and refresh checks cited by the report, including slack shape, regional backlog and monthly retraining. |
| [`data/outputs/phase2/`](../../data/outputs/phase2/), [`data/outputs/experiments/`](../../data/outputs/experiments/) | Saved metrics, predictions and daily replay outputs read by the report generator. |
| [`src/reporting/final_evidence.py`](../../src/reporting/final_evidence.py) | Refreshes marked numbers, tables and figures in the LaTeX report from saved results. |
| [`src/quality/`](../../src/quality/) | Great Expectations contract audit; its saved result is [`data/outputs/quality/ge_validation.json`](../../data/outputs/quality/ge_validation.json). |
| [`tests/`](../../tests/) | Point-in-time, feature, model and selection checks (`test_phase2_*.py`). |

## Reproduce

1. Install the dependencies in `requirements.txt`, then `pip install -e .` so `olist` is importable (or prefix commands with `PYTHONPATH=src`). Great Expectations is separately pinned in `src/quality/requirements.txt`.
2. To rebuild the prepared table, execute `notebooks/phase2_modelling/data_prep.ipynb`. This overwrites the included CSV and feature spec in `data/processed/`; retain a copy if you need to compare with the submitted snapshot.
3. To refit the models, run `python -m olist.eta` and `python -m olist.handover_classifier` in that order. Rerun `notebooks/phase2_modelling/regression.ipynb` and `classification.ipynb` to inspect refreshed outputs.
4. To reproduce specific report checks, run the scripts in `src/experiments/` and `src/reporting/` shown in [the evidence map](report_evidence_map.md).
5. Run `python src/reporting/final_evidence.py` to refresh generated LaTeX blocks from the saved outputs. Run `python -m unittest discover -s tests -p 'test_phase2_*.py'` for focused tests. Run `python src/quality/ge_validation.py` for the GX table contract.

> **Phase 3 update.** After the Phase 2 feedback, `olist.eta` searches ridge alpha from 0.01 to 100,000 and selects alpha = 1000 at both stages, so `data/outputs/phase2/eta/`, the regression-ranking comparisons in `data/outputs/phase2/handover/` and the deployment models now reflect alpha = 1000. The submitted Phase 2 report and its `docs/phase2/report/generated/` files, and `data/outputs/phase2/final/`, still show the alpha = 100 results; they are kept as submitted (the earlier outputs are in git history), so do not rerun `final_evidence.py` expecting to reproduce the Phase 2 PDF.


The report compares checkout and carrier-handover predictions on distinct information sets. Its five chronological validation windows drive model choice. The later test period reports performance under subsequent conditions; the monthly refresh and drift analyses are descriptive diagnostics.

The data include only eventual deliveries with a valid carrier handover for the final model comparison. See the report for cohort exclusions and assumptions about checkout visibility of the recorded promise, payments and static ZIP coordinates.

## Package verification (12 October 2026)

The 43 focused Phase 2 tests passed from the then `Phase2/` folder. The report evidence script ran, and the open LaTeX source compiled in the document editor. GX Core 1.24.0 passed 35/35 contract checks on the packaged table and feature spec; its JSON records their SHA-256 hashes. The full regression entry point was rerun. Its selected ridge results were stable; the unpruned decision-tree baseline shifted by a few hundredths of a day, so the saved regression output and report table were refreshed together. The classifier's selected logistic validation result and initial tree/forest candidates were rerun; its full ensemble sweep was not repeated during packaging. The included saved classifier outputs remain the source for its report figures.

After the Phase 3 restructure (26 October 2026) the same 43 tests pass from the repository root, the evidence script reproduces all 245 report macros unchanged, and `data_prep.ipynb` rebuilds the prepared table and spec byte-for-byte.
