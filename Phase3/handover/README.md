# Regression at carrier handover: decision-support investigation

Pratik's Phase 3 work, 31 October 2026. This is a first experimental checkpoint, not a change to the submitted model or a deployment recommendation.

## Business question

**Which parcels warrant a carrier recovery check, and how much faster would delivery need to be to meet the existing promise?**

The classifier owner's task is estimating/ranking lateness risk. This regression work supplies a complementary quantity: estimated time still needed, compared with time still available. A parcel with a high late-risk score may already be too far behind for a feasible intervention; a smaller expected delay might be recoverable. That hypothesis must be tested, not assumed.

At handover, the promise remains unchanged. Compute:

- Predicted arrival = purchase timestamp + predicted total delivery days.
- Remaining time = predicted arrival minus the decision time.
- Time budget = end of the existing promised calendar day minus the decision time.
- Estimated shortfall = remaining time minus time budget. A positive shortfall indicates how many days of improvement the point forecast suggests are needed.

For example, an estimated five days remaining with three days left to the promise suggests checking whether the carrier can save more than two days. It does **not** establish that such a service exists or that it will work. The recommendation is to ask for a feasible option and its cost, not automatically purchase an upgrade. Negative remaining-time predictions are invalid for an open parcel and should trigger a fallback/review, not display an arrival in the past.

## First results

### Responding to the regression selection feedback

Using the submitted five-window results [1], let each paired gain be plain-linear MAE minus ridge MAE. Report the mean gain and `SD(gain) / sqrt(5)`:

| Stage | Ridge alpha | Mean MAE gain (days) | Paired SE (days) | Plain linear within one SE? |
|---|---:|---:|---:|---|
| Checkout | 100 | 0.05178 | 0.00772 | No |
| Handover | 100 | 0.04764 | 0.00404 | No |

Thus the original ridge choice survives comparison with plain linear under the project's adapted paired one-SE rule. This is a small absolute gain: about 69 minutes at handover. It is not a statistical significance claim; five windows with overlapping fitting samples do not provide five independent replications. Full per-window differences and SDs are in `results/selection_audit.json`.

The handover extension tries alpha = 0.01, 0.1, 1, 10, 100, 1,000, 10,000 and 100,000, plus plain linear. Preprocessing and folds remain those of the submitted package [2]. Ridge is tuned on mean validation MAE; plain linear is preferred if within one paired SE of the best ridge.

| Model | Mean validation MAE | Later-period MAE | Later-period bias |
|---|---:|---:|---:|
| Plain linear | 5.146 | Not rescored here | — |
| Submitted ridge, alpha 100 | 5.099 | 3.972 | +2.867 |
| Extended ridge, alpha 1,000 | 5.055 | 4.199 | +3.120 |
| Elapsed time + recent route transit rule | 5.304 | 3.762 | +2.447 |

The extended optimum is inside the grid. Its validation gain over plain linear is 0.09142 days (paired SE 0.01513); over alpha 100 it is 0.04377 days (SE 0.01612). However, it performs worse in the later period. Do not select a replacement by switching to whichever candidate looks best on that already-seen test. The drift investigation remains necessary. These are exploratory results on reused validation and test periods, not a new independent test.

### Does the estimate help a review queue?

Replay the previous day's handovers at 09:00 next morning. Include only parcels still open with their promised calendar day not yet over. All strategies receive the same daily capacity, rounded up from 5%, 10% or 20% of that day's eligible cohort. The rounding means the aggregate share is slightly above the nominal share.

Compared strategies:

1. **Nearest deadline:** ascending time until the promised day ends.
2. **Route slack:** descending expected lateness using elapsed time plus recent route transit.
3. **Largest predicted delay:** descending estimated shortfall from the frozen alpha-100 regression.
4. **Predicted recovery band:** first consider predicted shortfalls between zero and the assumed speed-up, then fill remaining slots by largest predicted delay. This does not leave capacity unused.
5. **Random:** exact expected counts from a random queue of the same daily size.

The forecast remains fixed at handover; the next-morning decision only updates elapsed time and observed delivery status. Ranking never uses actual future delivery. Outcomes are used afterward to evaluate the queue. No classifier is retrained or compared here; these results are a handoff to that workstream.

At the illustrative 10% share, assuming every reviewed parcel could arrive **exactly two days earlier**:

| Strategy | Late orders reviewed, validation pooled | Hypothetically prevented misses, validation pooled | Late orders reviewed, later period | Hypothetically prevented misses, later period |
|---|---:|---:|---:|---:|
| Nearest deadline | 416 | 112 | 183 | 87 |
| Route slack | 917 | 151 | 225 | 92 |
| Largest predicted delay | 869 | 153 | 238 | 93 |
| Predicted recovery band | 868 | 156 | 234 | 93 |
| Random expectation | 412.2 | 78.5 | 48.1 | 17.9 |

Each strategy reviews 3,890 parcels across validation and 1,936 in the later period. Pooling is descriptive and differs from equal-weight averaging across validation windows. The later-period prevention cohort has 18,999 eligible parcels, including 474 eventual late deliveries. Its 231 exclusions comprise parcels already delivered and/or past their promised day at review. These counts intentionally differ from the full Phase 2 cohort and its monthly ranking lists [3].

The forecast queue improves on nearest-deadline ranking, but its two-day scenario gain over route slack is only **two** hypothetical prevented misses across validation and **one** in the later period. The recovery-band rule supplies no convincing operational improvement over the simpler forecast ranking. We should not present either as a proven winning intervention strategy.

For one-, two-, and three-day speed-ups respectively, the later-period forecast queue has 63, 93 and 119 hypothetical prevented misses; route slack has 61, 92 and 116. All capacity/speed-up combinations are in `results/queue_scenarios.csv`. The hindsight oracle column is a deliberately unattainable ceiling using actual outcomes, not a deployable strategy.

## Limits that change the recommendation

- **No intervention evidence:** Olist has no expedited-service options, carrier identifiers, prices, success probabilities or treatment outcomes. The scenario assumes a fixed additive speed-up for every reviewed parcel, arrival no earlier than review time, no resource contention and no effects on other parcels. It cannot establish cost savings, causal impact or real preventability. With an independent success probability q, expected successful recoveries would be q times the scenario count; q is not estimated here.
- **Forecast resolution:** the frozen model's later-period MAE is 3.97 days and its 90th-percentile absolute error is 8.23 days. Its precision is inadequate for confidently distinguishing one-day from two-day interventions for an individual order. Those error summaries are not calibrated individual prediction intervals. At handover, 157 of 19,230 later-period predictions imply negative remaining time; an application must handle these explicitly.
- **Selection versus operations:** the queue uses the submitted alpha-100 model throughout. No queue result selected its model or hyperparameters. The extended ridge fit is a separate model-selection experiment.
- **Population:** eventual deliveries with valid handovers only. Cancelled and permanently unfulfilled orders remain outside scope.
- **Replay coverage:** these are daily queues within the original purchase-based validation/test cohorts, not complete live-marketplace daily queues. They do not carry unreviewed parcels forward or measure staffing requirements. The full-population daily replay in Phase 2 is a different analysis [3].
- **Timing and date conventions:** timestamps use the dataset's recorded local clock. Arrival anywhere on the promised calendar date is on time; the boundary is the next midnight. This avoids treating afternoon arrivals on the promised day as late.
- **Data drift:** both model overprediction and the simple route rule's stronger later-period MAE argue against presenting the point forecast as a guarantee. Recent-history features update as outcomes become known, while fitted model coefficients stay frozen for each validation/test fit.

## Proposed handover contribution to the team demo

Show an order card with the unchanged promised date, handover time, estimated remaining days, time left to the deadline, estimated shortfall, and the simple route-history estimate. Use an explicit warning/fallback for impossible forecasts and indicate the model's historical error.

Keep two actions distinct: before the promise expires, ask the carrier to assess a recovery option; after it expires, escalate and communicate the delay rather than count the parcel as preventable. The card should say "carrier check recommended" rather than "expedite will save this order".

Next investigations: assess forecast uncertainty using only prior completed outcomes; compare model and route-rule disagreement cases; combine the classifier's priority list with estimated shortfall and carrier feasibility. Any choice based on this historical period needs prospective evaluation before an impact claim. Retraining schedules belong with the drift workstream; changing customer promises belongs with the checkout workstream.

## Reproduce

The `pratik` branch is based on the submitted main snapshot `ea87322`, with this investigation committed on top. This study adds only `Phase3/handover/` and does not modify the submitted Phase 2 outputs. The initial experiment used a temporary export; the local `Phase2/` sources and data now match that export. The commands below retain the pinned export for exact reproducibility.

From the repository root:

```sh
mkdir -p /tmp/it5006-submitted-ea87322
git archive ea87322a7b62f95314d811df8010156faa0a1470 Phase2 | tar -x -C /tmp/it5006-submitted-ea87322
python -m venv /tmp/it5006-handover-env
/tmp/it5006-handover-env/bin/python -m pip install -r Phase3/handover/requirements.txt
/tmp/it5006-handover-env/bin/python Phase3/handover/analyse.py --phase2-root /tmp/it5006-submitted-ea87322/Phase2
/tmp/it5006-handover-env/bin/python -m unittest discover -s Phase3/handover -p 'test_*.py' -v
```

The script defaults to the repository's `Phase2/`, so on this branch you can omit `--phase2-root`. Full predictions are regenerated as `results/predictions.csv.gz` and ignored by Git; compact metrics and audit records are tracked. The audit records dependency versions and source/data hashes. Tiny differences from submitted numbers reflect rerunning the numerical pipeline; compare rounded metrics, not bitwise identity.

`results/submitted_tree_configurations.json` records actual estimator settings, including defaults, for the submitted regression candidates. The explicit alternatives were unrestricted versus depth-6/min-leaf-50 decision trees; a 200-tree forest with square-root feature sampling and min-leaf-20; and default histogram gradient boosting versus learning-rate-0.05/300-iteration/min-leaf-40 boosting. The JSON supplies all remaining settings for the full-configuration appendix requested by the evaluator. This study does not expand the tree grid.

## Numbered evidence references

[1] Submitted regression window metrics, fixed snapshot: [Phase2/results/phase2/eta/cv_metrics.csv](https://github.com/YuanJiayi/Team6_IT5006_Ecommerce_Analytics_AY2627Sem1/blob/ea87322a7b62f95314d811df8010156faa0a1470/Phase2/results/phase2/eta/cv_metrics.csv).

[2] Submitted feature construction, model definitions and chronological fitting: [Phase2/phase2_eta.py](https://github.com/YuanJiayi/Team6_IT5006_Ecommerce_Analytics_AY2627Sem1/blob/ea87322a7b62f95314d811df8010156faa0a1470/Phase2/phase2_eta.py), [Phase2/phase2_regression.py](https://github.com/YuanJiayi/Team6_IT5006_Ecommerce_Analytics_AY2627Sem1/blob/ea87322a7b62f95314d811df8010156faa0a1470/Phase2/phase2_regression.py), and the paired selection implementation [Phase2/phase2_selection.py](https://github.com/YuanJiayi/Team6_IT5006_Ecommerce_Analytics_AY2627Sem1/blob/ea87322a7b62f95314d811df8010156faa0a1470/Phase2/phase2_selection.py).

[3] Submitted report and evidence mapping: [Phase2/docs/report_evidence_map.md](https://github.com/YuanJiayi/Team6_IT5006_Ecommerce_Analytics_AY2627Sem1/blob/ea87322a7b62f95314d811df8010156faa0a1470/Phase2/docs/report_evidence_map.md).

[4] New calculations: [`analyse.py`](analyse.py), [`results/selection_audit.json`](results/selection_audit.json), [`results/prediction_metrics.csv`](results/prediction_metrics.csv), and [`results/queue_scenarios.csv`](results/queue_scenarios.csv). All numerical extension and queue results above are generated by this study [4]. These references support project calculations; they are not external evidence that expediting is effective.
