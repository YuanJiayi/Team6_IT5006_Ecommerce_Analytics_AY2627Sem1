# Phase 1 — exploratory analysis and dashboard

[Original Phase 1 report](report/Team6_Phase1_IT5006_AY2627Sem1-2.pdf) · [Marker feedback](report/Phase%201%20Report%20feedback.txt) · [Notebook and dashboard validation](docs/validation.md)

The report is the submitted Phase 1 artifact. The notebooks and dashboard here are the supporting analysis. The validation note records later checks and corrections to notebook calculations; read it alongside the historical PDF where a figure differs.

## Contents

| Path | Purpose |
| --- | --- |
| [`notebooks/`](notebooks/) | Seven EDA notebooks covering overview, delivery lead time, reviews and route distance, capacity, products and payments, plus the getting-started notebook. Saved plots make the analysis readable without a rerun. |
| [`app.py`](app.py), [`dashboard_data.py`](dashboard_data.py), [`geomapper.py`](geomapper.py), [`map_layers/`](map_layers/) | Streamlit dashboard and geographic layers. |
| [`data/`](data/) and [`brazil_order_density_map.html`](brazil_order_density_map.html) | Olist source tables, consolidated data and Brazil map files used by the notebooks and dashboard. |
| [`tests/`](tests/) | Focused dashboard checks. |
| [`docs/validation.md`](docs/validation.md) | Evidence and corrections recorded after the Phase 1 submission. |

## Run

From `Phase1/`, install `requirements.txt`, then run `streamlit run app.py` for the dashboard. `run_dashboard.sh` is an optional local setup script. Open notebooks from `Phase1/notebooks/` and run them with `Phase1/` as the working directory. The raw data are checked in so the notebooks can be reproduced without a separate download.

The notebooks are exploratory, not a single automated pipeline. The submitted PDF is preserved as an original artifact rather than rewritten to imply its figures changed after submission.
