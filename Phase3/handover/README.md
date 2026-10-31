# Handover regression: analysis checkpoint

Open [analysis.ipynb](analysis.ipynb) for the executed analysis, supporting tables, charts and conclusions.

The notebook covers:

- Paired one-standard-error comparisons of ridge and plain linear regression.
- An extended handover regularisation grid and later-period performance.
- Remaining-time estimates and shortfalls against the existing promise.
- Equal-capacity review queues and hypothetical carrier speed-up scenarios.
- Full settings for the submitted regression tree models.
- Implications and remaining evidence requirements for decision support.

Use a Python environment with the packages in `requirements.txt`, then run the notebook from top to bottom. It finds the repository automatically and reads the included `Phase2/` package. Outputs are saved in the notebook; rerunning the fitting cell refreshes only `Phase3/handover/results/`.

`analyse.py` contains shared fitting and queue-evaluation functions. `test_analyse.py` checks calendar-date handling, prediction-time ranking, equal review capacity and paired standard errors. Compact result tables are tracked; the larger order-level predictions are regenerated locally.

The historical periods have already been examined, and carrier speed-ups are assumptions. The notebook distinguishes forecast accuracy, ranking performance and hypothetical recovery from measured intervention effects.
