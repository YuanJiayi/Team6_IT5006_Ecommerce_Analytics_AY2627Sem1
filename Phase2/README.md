# Phase 2 — keeping delivery promises at Olist

[Final report source](reports/phase2_report_v4.tex) · [Evidence map](docs/report_evidence_map.md)

This folder contains the code, notebooks, data and saved results supporting the report. The model scripts perform the fits; the notebooks explain the preparation and display model comparisons. All commands below run from `Phase2/`.

## What is where

| Path | Purpose |
| --- | --- |
| [`data/`](data/) | Olist source CSVs, the prepared one-row-per-order table, and the feature specification. |
| [`notebooks/`](notebooks/) | Preparation, regression and classification narratives. The regression and classification notebooks read saved model outputs. |
| [`phase2_features.py`](phase2_features.py) | Order features, history features and point-in-time construction. |
| [`phase2_eta.py`](phase2_eta.py), [`phase2_handover_classifier.py`](phase2_handover_classifier.py) | Selected regression and classifier comparisons, temporal windows, test scores and selection records. Other `phase2_*.py` files are imported helpers. |
| [`experiments/`](experiments/) | The specific ablations and refresh checks cited by the report, including slack shape, regional backlog and monthly retraining. |
| [`results/phase2/`](results/phase2/) | Saved metrics, predictions and daily replay outputs read by the report generator. |
| [`reports/phase2_final_evidence.py`](reports/phase2_final_evidence.py) | Refreshes marked numbers, tables and figures in the LaTeX report from saved results. |
| [`quality/`](quality/) | Great Expectations contract audit and its saved result for this packaged table. |
| [`tests/`](tests/) | Point-in-time, feature, model and selection checks. |

## Reproduce

1. Install the dependencies in `requirements.txt`. Great Expectations is separately pinned in `quality/requirements.txt`.
2. To rebuild the prepared table, execute `notebooks/data_prep.ipynb` from `Phase2/`. This overwrites the included CSV and feature spec; retain a copy if you need to compare with the submitted snapshot.
3. To refit the models, run `python phase2_eta.py` and `python phase2_handover_classifier.py` in that order. Rerun `notebooks/regression.ipynb` and `notebooks/classification.ipynb` to inspect refreshed outputs.
4. To reproduce specific report checks, run the scripts in `experiments/` and `reports/` shown in [the evidence map](docs/report_evidence_map.md). The monthly retraining experiment is inside this folder and the report generator reads its saved metrics here.
5. Run `python reports/phase2_final_evidence.py` to refresh generated LaTeX blocks from the saved outputs. From `Phase2/`, run `python -m unittest discover -s tests -p 'test_phase2_*.py'` for focused tests. Run `python quality/ge_validation.py --out quality/ge_validation.json` for the GX table contract.

The report compares checkout and carrier-handover predictions on distinct information sets. Its five chronological validation windows drive model choice. The later test period is a separate evaluation period, and the monthly refresh and drift analyses describe that period after it was available; they do not constitute a new untouched test.

The data include only eventual deliveries with a valid carrier handover for the final model comparison. See the report for cohort exclusions and assumptions about checkout visibility of the recorded promise, payments and static ZIP coordinates.
