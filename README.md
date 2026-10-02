# Team6_IT5006_Ecommerce_Analytics_AY2627Sem1
Phase / Milestone 2: Analytics Implementation - Problem Definition, Model Building & Evaluation

## Repository layout

- `app.py`, `dashboard_data.py`, `geomapper.py`, and `map_layers/`: the shared Phase 1 dashboard and map code.
- `data/`: original Olist tables and the prepared, item-level Phase 1 dataset.
- `notebooks/phase1/`: Phase 1 exploratory notebooks and historical EDA snippets.
- `notebooks/phase2/`: new modelling notebooks as they are developed.
- `ref/`: project brief, submitted Phase 1 report, and feedback.
- `tests/`: checks for the dashboard's delivery and review rules.

Run the dashboard from the repository root with `./run_dashboard.sh`. To regenerate its map, run `python geomapper.py` from the root with the dependencies in `requirements.txt` installed. Local Phase 1 notebooks can be launched from the root or from `notebooks/phase1/`; their setup cells locate the root `data/` directory. The EDA starter notebook also retains a Google Colab Drive example: set its Drive path to your own data folder when running in Colab. Its final save cell writes `smartcommerce_consolidated_rebuilt.csv` into that folder for review, leaving the checked-in dataset intact.

## Phase 1 data carried into Phase 2

See [the Phase 1 validation audit](docs/phase1_validation.md) for corrected figures, reproducibility checks, and limits on the findings carried into modelling.

The Phase 1 PDF preserves the original submission; the notebooks now have refreshed outputs from the checked-in data. The dashboard and checked-in `data/smartcommerce_consolidated.csv` count delivery on the estimated **calendar date** as on time; the old timestamp rule incorrectly marked 1,292 such orders late. Duplicate reviews use the latest creation time, answer time, and review ID as tie-breakers. Use the audit's corrected definitions and figures when citing Phase 1 in the Phase 2 report.

For modelling, build a fresh **one-row-per-order** table from the raw Olist CSVs. The consolidated CSV is an item-level EDA artifact: its `delivery_days` value is rounded down to whole days, it has outcome fields that would leak into purchase-time predictions, and its single `seller_id` cannot represent every seller in a multi-seller order. Use `customer_unique_id` for customer history; the Phase 1 count of 95,420 refers to distinct people among the 98,666 orders with recorded items, not every row in the customers table.

## Phase 2 evaluation decision

See [the Phase 2 evaluation protocol](milestone2/evaluation_protocol.md). The primary test holds out orders purchased on or after 2018-05-26, about the latest 20% by purchase date, to assess performance on a later period. Primary training uses earlier orders whose outcomes were known by that date, with five expanding-time validation windows. The teammate's seeded, stratified random 80/20 split is retained as a separate secondary benchmark for orders drawn from the same historical mix. Both tasks use the same partitions within each experiment. The prepared table, split metadata, and loading examples are in [the Phase 2 data preparation notebook](milestone2/data_prep.ipynb).

Deadline: Sunday, 11 October 2026 at 23:59 (end of Week 8)
Weight: 40%
Report length: 6–8 pages (excludes cover page, references, appendices)

## Problem Scoping (Required)

Each team must define 1 or 2 analytics problems maximum, submitted as part of this phase:

| Requirement | Details |
| --- | --- |
| Number of problems | 1 or 2 |
| Problem types | **both classification and regression** - via two problems (one of each), or one problem with dual framing |
| Scope | Must be achievable with Olist data and course techniques |
| Justification | Business stakeholder, target variable definition, and success criteria |

## Modelling Requirements - Quality Over Quantity

Focus on doing a small number of model families well, not trying many algorithms superficially. Many scikit-learn algorithms come in matched Classifier/Regressor pairs - reuse the same family across your classification and regression task(s) wherever sensible.

**Model Family Budget (2–3 families total for the whole project)**. The families and algorithms below are illustrative examples, not a fixed list - pick any algorithm(s) that reasonably belong to a family you choose, as long as you stay within the 2–3 family budget.

| Family (example) | Example classification model(s) | Example regression model(s) |
| --- | --- | --- |
| A - Linear | e.g. `LogisticRegression` | e.g. `LinearRegression` / `Ridge` |
| B - Tree-based | e.g. `DecisionTreeClassifier` / `RandomForestClassifier` | e.g. `DecisionTreeRegressor` / `RandomForestRegressor` |
| C - Ensemble (optional, advanced) | e.g. `VotingClassifier` / `StackingClassifier` (built from A + B) | e.g. `VotingRegressor` / `StackingRegressor` (built from A + B) |

Don't jump straight to your most complex model. Within each family, start with its simplest variant as your baseline before adding complexity - e.g. plain `LogisticRegression` / `LinearRegression` before `Ridge`/regularised versions, or a single `DecisionTree` before `RandomForest`.

| Guideline | Details |
| --- | --- |
| Model families | **2–3 total** across the entire project (e.g. Linear + Tree-based, optionally + Ensemble - see examples above, other families are welcome) |
| Baseline | Use the **simplest variant within each family** as your starting point (see tip above); every more complex model must be shown to beat it |
| Train–test discipline | Clear train/test split (or train/validation/test); no data leakage |
| Model selection | Compare families systematically using appropriate metrics; explain your final choice |
| Cross-validation | Use CV for robust performance estimates and hyperparameter tuning |
| Reproducibility | Set random seeds; document all preprocessing steps in pipelines |

## Deliverables

- Problem statement(s) and stakeholder context (1–2 problems max)
- Data preprocessing and cleaning methodology
- Feature engineering strategy
- Model descriptions and training process (2–3 model families total)
- Hyperparameter tuning methodology
- **Classification metrics**: Precision, Recall, F1-Score, AUC-ROC / PR-AUC (as appropriate for imbalance)
- **Regression metrics**: MAE, RMSE, R² (as appropriate)
- Cross-validation results
- Feature importance / interpretability analysis
- Model comparison and final selection with justification
- Ensemble or stacking results (if used)
- Actionable insights for your e-commerce stakeholder
- Discussion of limitations and constraints


## Submission Format

- Technical report as PDF with GitHub repository link
- GitHub repository with Jupyter notebooks and Python scripts
- Model performance summary tables (in report)
