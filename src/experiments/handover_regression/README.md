# Handover regression: analysis checkpoint (Phase 3)

Open [the analysis notebook](../../../notebooks/phase3_investigations/handover_regression.ipynb) for the executed analysis, supporting tables, charts and conclusions.

The notebook covers:

- Paired one-standard-error comparisons of ridge and plain linear regression.
- An extended handover regularisation grid and later-period performance.
- Remaining-time estimates and shortfalls against the existing promise.
- Equal-capacity review queues and hypothetical carrier speed-up scenarios.
- Full settings for the submitted regression tree models.
- Implications and remaining evidence requirements for decision support.

Run the notebook from top to bottom with the repository's `requirements.txt` installed. It finds the repository automatically and imports the shared `olist` package from `src/`. Outputs are saved in the notebook; rerunning the fitting cell refreshes only `data/outputs/experiments/handover_regression/`. The same fit runs from the repository root with:

```bash
python src/experiments/handover_regression/run.py
```

`run.py` contains the shared fitting and queue-evaluation functions. `test_analyse.py` checks calendar-date handling, prediction-time ranking, equal review capacity and paired standard errors:

```bash
python -m unittest discover -s src/experiments/handover_regression -p "test_*.py"
```

Compact result tables are tracked; the larger order-level `predictions.csv.gz` is regenerated locally.

The historical periods have already been examined, and carrier speed-ups are assumptions. The notebook distinguishes forecast accuracy, ranking performance and hypothetical recovery from measured intervention effects.
