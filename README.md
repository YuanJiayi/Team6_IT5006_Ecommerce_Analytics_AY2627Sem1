# IT5006 Group 6 — Olist delivery analytics

We predict how long an Olist order will take to arrive (regression) and flag which parcels are at risk of missing their promised date when the carrier picks them up (classification). The stakeholder is Olist's operations team. Phase 1 explored the data and built a dashboard. Phase 2 built and compared the models. Phase 3 deploys them as an application.

## Repository layout

```
data/
  raw/            Olist source CSVs (Canvas copy, unchanged)
  geo/            Brazil boundary and state GeoJSON for the maps
  processed/      Prepared tables: Phase 1 consolidated line items, Phase 2 order table and feature spec
  outputs/        Saved results: saved models, phase1 map, phase2 model outputs, experiment and data-quality results
notebooks/
  phase1_eda/     Exploratory analysis notebooks
  phase2_modelling/  Data preparation, regression and classification notebooks
src/
  olist/          Shared package: features, model training and selection, EDA helpers, map layers, paths
  experiments/    Ablations and refresh checks cited by the Phase 2 report
  reporting/      Scripts that generate report numbers, tables and figures from saved outputs
  quality/        Great Expectations contract for the prepared table
deployment/
  olist_dashboard/   The deployed Phase 3 app (Olist dashboard: Overview, Actions, Models, Ask)
  phase1_dashboard/  Phase 1 Streamlit EDA dashboard
docs/
  phase1/         Submitted Phase 1 report, marker feedback, validation notes
  phase2/         Phase 2 report source, evidence map, assumptions
  reference/      Course project description
tests/            Unit and data-contract tests
```

## Setup

Python 3.11 or later. From the repository root:

```bash
python3 -m venv it5006-proj
```

```bash
source it5006-proj/bin/activate
```

```bash
pip install -r requirements.txt
```

```bash
pip install -e .
```

The last command makes the `olist` and `phase3` packages in `src/` importable from any folder. Scripts under `src/experiments/`, `src/reporting/` and the dashboard also add `src/` to the import path themselves.

## Common commands

All commands run from the repository root.

| Task | Command |
| --- | --- |
| Run all tests | `python -m unittest discover -s tests` |
| Run the Olist dashboard (deployed app) | `bash deployment/olist_dashboard/run_local.sh` |
| Run the Phase 1 dashboard | `streamlit run deployment/phase1_dashboard/app.py` |
| Rebuild the dashboard's model store | `python -m phase3.registry` |
| Build open-parcel rows | `python -m phase3.build_open_orders` |
| Refit the delivery-time models | `python -m olist.eta` |
| Refit the late-risk classifier | `python -m olist.handover_classifier` |
| Save the chosen models to `data/outputs/models/` | `python -m olist.export_models` |
| Refresh report numbers and figures | `python src/reporting/final_evidence.py` |
| Check the prepared table contract | `python src/quality/ge_validation.py` (needs `src/quality/requirements.txt`) |

Notebooks find `data/` from any folder inside the repository, so open them from `notebooks/`.

## Phase documents

- [Phase 1: exploratory analysis and dashboard](docs/phase1/README.md)
- [Phase 2: delivery-time estimation and late-order prioritisation](docs/phase2/README.md)
- [Deployment](deployment/README.md)
- [Project description](docs/reference/IT5006%20Project%20Description%20-%20AY%202026_27%20Semester%201.pdf)
