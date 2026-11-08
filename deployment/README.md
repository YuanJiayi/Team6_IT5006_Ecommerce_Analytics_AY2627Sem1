# Deployment

The Phase 3 deployment follows the course's two-service pattern. A model API loads the saved pipelines and serves predictions. A separate Streamlit app is the user interface and calls the API over HTTP.

| Folder | Service | Status |
| --- | --- | --- |
| `api/` | FastAPI prediction service for Render: `main.py`, `requirements.txt`, `Procfile`, `.python-version`, and [`models/`](api/models/) holding the saved pipelines, `model_metadata.json` and `feature_stats.json`. | Models saved; service to be built. |
| [`app/`](app/) | Streamlit app for Streamlit Community Cloud. It currently holds the Phase 1 EDA dashboard; prediction tabs that call the API will be added. | Dashboard runs. |

## Saved models

`python -m olist.export_models` (from the repository root) refits each selected Phase 2 model on all training orders, saves it as one scikit-learn Pipeline with joblib, and reloads every file to check it. It follows the L6 lecture's Step 7 (save pipelines and metadata) and Step 8 (verify saved pipelines).

| File | Model | Output |
| --- | --- | --- |
| `eta_checkout.pkl` | Ridge (alpha 1000), checkout inputs | Days from purchase to delivery, predicted at checkout |
| `eta_handover.pkl` | Ridge (alpha 1000), checkout and handover inputs | Updated delivery days once the carrier has the parcel |
| `late_risk_handover.pkl` | Logistic regression (C 0.01), handover inputs | Probability of arriving after the promised date; review the top 14% each month |
| `model_metadata.json` | Input columns, categories, selection reason, test metrics, example input, environment | |
| `feature_stats.json` | Training mean, spread and missing share of every numeric input, for drift checks | |

Loading a pipeline needs the `olist` package importable and scikit-learn 1.9.1, the version that saved it. The history inputs (route and seller averages, remaining slack) are computed before the pipeline from earlier orders only, so the API must supply them; they are not raw order fields.

Both services import shared feature code from [`src/olist/`](../src/olist/) instead of copying it, so prediction-time features are computed exactly as in training.

## Run the dashboard locally

From the repository root:

```bash
pip install -r deployment/app/requirements.txt
streamlit run deployment/app/app.py
```

`deployment/app/run_dashboard.sh` does the same inside the repository's `it5006-proj` virtual environment. Streamlit reads its theme from `.streamlit/config.toml` at the repository root, so run it from there. On Streamlit Community Cloud, set the main file path to `deployment/app/app.py`.
