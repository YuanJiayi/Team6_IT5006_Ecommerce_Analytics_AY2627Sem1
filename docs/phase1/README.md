# Phase 1 — exploratory analysis and dashboard

[Original Phase 1 report](report/Team6_Phase1_IT5006_AY2627Sem1-2.pdf) · [Marker feedback](report/Phase%201%20Report%20feedback.txt) · [Notebook and dashboard validation](validation.md)

The report is the submitted Phase 1 artifact. The notebooks and dashboard are the supporting analysis. The validation note records later checks and corrections to notebook calculations; read it alongside the historical PDF where a figure differs. The validation note describes the Phase 1 folder layout at the time; the files now live at the paths below.

## Contents

Paths are relative to the repository root.

| Path | Purpose |
| --- | --- |
| [`notebooks/phase1_eda/`](../../notebooks/phase1_eda/) | Seven EDA notebooks covering overview, delivery lead time, reviews and route distance, capacity, products and payments, plus the getting-started notebook. Saved plots make the analysis readable without a rerun. |
| [`deployment/app/app.py`](../../deployment/app/app.py) | Streamlit dashboard. |
| [`src/olist/eda.py`](../../src/olist/eda.py), [`src/olist/geomapper.py`](../../src/olist/geomapper.py), [`src/olist/map_layers/`](../../src/olist/map_layers/) | Dashboard data helpers (plain pandas) and geographic layers. |
| [`data/raw/`](../../data/raw/), [`data/processed/smartcommerce_consolidated.csv`](../../data/processed/smartcommerce_consolidated.csv), [`data/geo/`](../../data/geo/) and [`data/outputs/phase1/brazil_order_density_map.html`](../../data/outputs/phase1/brazil_order_density_map.html) | Olist source tables, consolidated data and Brazil map files used by the notebooks and dashboard. |
| [`tests/test_dashboard_*.py`](../../tests/) | Focused dashboard checks. |
| [`validation.md`](validation.md) | Evidence and corrections recorded after the Phase 1 submission. |

## Run

From the repository root, install `requirements.txt`, then run `streamlit run deployment/app/app.py` for the dashboard. `deployment/app/run_dashboard.sh` is an optional local setup script. Open notebooks from `notebooks/phase1_eda/`; they locate `data/raw/` and `data/processed/` from any folder inside the repository. To rebuild the map, run `PYTHONPATH=src python -m olist.geomapper`.

The notebooks are exploratory, not a single automated pipeline. The submitted PDF is preserved as an original artifact rather than rewritten to imply its figures changed after submission.
