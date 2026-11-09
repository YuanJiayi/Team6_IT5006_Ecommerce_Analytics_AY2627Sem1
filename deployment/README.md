# Deployment

Two dashboards live here. The deployed Phase 3 app is `olist_dashboard/`; the Phase 1 Streamlit EDA dashboard is kept in `phase1_dashboard/` because the Phase 1 report links to it.

| Folder | Service | Status |
| --- | --- | --- |
| [`olist_dashboard/`](olist_dashboard/) | The deployed Phase 3 app: the Olist dashboard with Overview, Actions, Models and Ask. It loads its models from its own MLflow store in `deployment/olist_dashboard/mlflow/`, rebuilt by `python -m phase3.registry`. | Runs. |
| [`phase1_dashboard/`](phase1_dashboard/) | Phase 1 Streamlit EDA dashboard. Not the deployed Phase 3 app. Kept because the Phase 1 report links to it. | Runs. |

**Saved models:** the three chosen models as standalone files, with `model_metadata.json` (the inputs each model expects, scores, an example) and `feature_stats.json` (training data summary for drift checks). Made by `python -m olist.export_models`. They are in `data/outputs/models/`.

| File | Model | Output |
| --- | --- | --- |
| `eta_checkout.pkl` | Ridge (alpha 1000), checkout inputs | Days from purchase to delivery, predicted at checkout |
| `eta_handover.pkl` | Ridge (alpha 1000), checkout and handover inputs | Updated delivery days once the carrier has the parcel |
| `late_risk_handover.pkl` | Logistic regression (C 0.01), handover inputs | Probability of arriving after the promised date; review the top 14% each month |
| `model_metadata.json` | Input columns, categories, selection reason, test metrics, example input, environment | |
| `feature_stats.json` | Training mean, spread and missing share of every numeric input, for drift checks | |

Loading a pipeline needs the `olist` package importable and scikit-learn 1.9.1, the version that saved it. The history inputs (route and seller averages, remaining slack) are computed before the pipeline from earlier orders only, so a caller must supply them; they are not raw order fields.

The dashboard imports shared feature code from [`src/olist/`](../src/olist/) and [`src/phase3/`](../src/phase3/) instead of copying it, so prediction-time features are computed exactly as in training.

## Run the Olist dashboard

From the repository root:

```bash
bash deployment/olist_dashboard/run_local.sh
```

`deployment/olist_dashboard/README.md` covers the analytics database, the Ask provider settings, the model store and the checks.

## Run the Phase 1 dashboard locally

From the repository root:

```bash
pip install -r deployment/phase1_dashboard/requirements.txt
streamlit run deployment/phase1_dashboard/app.py
```

`deployment/phase1_dashboard/run_dashboard.sh` does the same inside the repository's `it5006-proj` virtual environment. Streamlit reads its theme from `.streamlit/config.toml` at the repository root, so run it from there. On Streamlit Community Cloud, set the main file path to `deployment/phase1_dashboard/app.py`.
