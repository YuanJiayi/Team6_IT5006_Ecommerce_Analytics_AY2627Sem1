# Deployment

The Phase 3 deployment follows the course's two-service pattern. A model API loads the saved pipelines and serves predictions. A separate Streamlit app is the user interface and calls the API over HTTP.

| Folder | Service | Status |
| --- | --- | --- |
| `api/` | FastAPI prediction service for Render: `main.py`, `requirements.txt`, `Procfile`, `.python-version`, and `models/` holding the saved pipelines (`*.pkl`), `model_metadata.json` and `feature_stats.json`. | To be built. |
| [`app/`](app/) | Streamlit app for Streamlit Community Cloud. It currently holds the Phase 1 EDA dashboard; prediction tabs that call the API will be added. | Dashboard runs. |

Both services import shared feature code from [`src/olist/`](../src/olist/) instead of copying it, so prediction-time features are computed exactly as in training.

## Run the dashboard locally

From the repository root:

```bash
pip install -r deployment/app/requirements.txt
streamlit run deployment/app/app.py
```

`deployment/app/run_dashboard.sh` does the same inside the repository's `it5006-proj` virtual environment. Streamlit reads its theme from `.streamlit/config.toml` at the repository root, so run it from there. On Streamlit Community Cloud, set the main file path to `deployment/app/app.py`.
