# Team6_IT5006_Ecommerce_Analytics_AY2627Sem1
Phase / Milestone 2: Analytics Implementation - Problem Definition, Model Building & Evaluation

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
