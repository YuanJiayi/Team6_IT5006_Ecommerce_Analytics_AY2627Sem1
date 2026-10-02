# Phase 2 modelling notebooks

The shared `data_prep.ipynb` notebook builds the Phase 2 order table, defines both evaluation splits, and records the rationale and limits. Add classification and regression notebooks here as the team defines the Phase 2 problems. Build modelling tables from the raw Olist CSVs in `data/`, with one row per order and features available at the prediction point. Reusable preprocessing code can be added after the modelling design is agreed.

Use the primary chronological split and expanding validation indices in [data_prep.ipynb](data_prep.ipynb) for model selection and future-order claims. The random split is a separate secondary benchmark. Complete the notebook's remaining leakage audit before fitting models.
