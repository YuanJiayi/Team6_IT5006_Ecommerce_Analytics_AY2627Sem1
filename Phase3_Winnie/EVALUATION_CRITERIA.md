# Task 4: how a ranking wins

Task 4 asks whether the late-warning model, a simple rule or a mix gives the best ranked review list. These criteria are fixed before experiments 02 to 04 run, so that results cannot change what counts as a win.

**The remaining-slack rule is the default.** This applies the same principle as the Phase 2 one-standard-error rule: prefer the simpler option unless a more complex one is clearly better. The rule needs no training or retraining, and it has the better result in the most recent period. The evidence is not one-sided: the model caught more late orders in all five validation windows (1,374 against 1,099), and the rule's test lead is small (398 against 377).

## Baseline (test set, 26 May to August 2018)

Each list is the top 10% of each purchase month. Source: `experiments/01_ranking_baselines/metrics.csv`.

| Ranking | Role | PR-AUC | ROC-AUC | Late caught (of 672) | Recall | Precision | Late caught, not past date (of 474) | PR-AUC, not past date |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Remaining-slack rule | Default | 0.519 | 0.876 | 398 | 59.2% | 20.7% | 200 | 0.227 |
| Logistic model | Candidate | 0.489 | 0.863 | 377 | 56.1% | 19.6% | 179 | 0.165 |
| Regression estimate minus promise | Reference | 0.500 | 0.879 | 411 | 61.2% | 21.4% | 217 | 0.263 |
| Closest to deadline | Reference | 0.487 | 0.833 | 359 | 53.4% | 18.7% | 161 | 0.142 |

"Not past date" means the promised date had not passed at carrier handover.

## Gates

A candidate replaces the rule only if it passes all three, in order.

| Gate | Data | Passes when |
| --- | --- | --- |
| **A. Beats the rule in validation** | Five validation windows, top 10% of each | The 95% paired bootstrap interval for candidate minus rule is above zero for both mean PR-AUC and mean top-10% recall (Phase 2's `paired_bootstrap` and `gate_verdict`) |
| **B. Holds up on short promises** | Validation orders with promises under 14.52 days (training 10th percentile, as in Phase 2) and not past their date | Mean PR-AUC over W1, W2, W4 and W5 is no more than one paired standard error below the rule's. W3 is left out because it has only 3 late orders |
| **C. Clearly beats the rule on test** | Test set, top 10% of each purchase month | PR-AUC, late caught and late caught not past date are all at least the rule's, and the 95% interval for candidate minus rule is above zero for PR-AUC or late caught |

Test intervals use 1,000 resamples of orders within each purchase month (seed 42), as in experiment 01.

Gate B exists because validation contains few short promises, so Gate A alone favours the model. The paired standard error is the standard deviation of the four window differences (candidate minus rule) divided by the square root of 4. It matches the one-standard-error approach in Phase 2, but four windows with small late counts make it approximate.

If several candidates pass, the simplest wins unless a more complex one has a 95% interval above zero against it on test PR-AUC or late caught. From simplest to most complex: the rule, a fixed mix, a switch, a refitted model.

**Check against known results:** on the Phase 2 results, the logistic model passes Gate A and Gate B (0.0035 below the rule, within one paired standard error of 0.0068) and fails Gate C, so the rule stays the default.

## Rules for using the test set

- Candidates and their settings are declared before test scoring. Tuning uses validation only.
- Each declared candidate is scored on test once and is not changed afterwards.
- Test results are reported for every declared candidate, including any that fail Gate A or B. A candidate that failed an earlier gate cannot win.

## Experiment 02 candidates

Neither candidate has settings to tune, and both can be calculated at handover without using later orders.

Scores are put on a common scale first. Each order's model score and rule score become percentiles against the same scores on that period's fitting orders. The fitting orders are each window's training rows in validation and the 75,039 training orders on test.

| Candidate | Score |
| --- | --- |
| **50/50 mix** | Mean of the model percentile and the rule percentile |
| **Short-promise switch** | Rule percentile for promises under the fitting orders' 10th percentile, model percentile otherwise |

The regression estimate and closest-to-deadline rankings stay as references. Neither is mixed in experiment 02.

## Experiment 03 candidates

Experiment 03 tries to improve the logistic model itself. All choices are made on validation PR-AUC only. Phase 2 feedback pointed out that C = 0.01 lies on the boundary of the values tried, so the grid extends below it.

**Stage 1: regularisation and training data (25 configurations).**

| Setting | Values |
| --- | --- |
| C | 0.0001, 0.0003, 0.001, 0.003, 0.01 |
| Training data | All history; last 365 days; last 180 days; recency weights with a 180-day half-life; recency weights with a 90-day half-life |

Age is measured in days from purchase to the start of the period being scored. Recency weights are scaled to average 1 so that C stays comparable.

Stage 1 applies the one-standard-error rule: take the simplest configuration whose mean PR-AUC is within one paired standard error of the best. From simplest to most complex, training data runs in the order listed above, and within each lower C is simpler.

**Stage 2: added signals.** Three versions of the Stage 1 choice are fitted, adding `zip3_transit_90d_at_handover`, `seller_handover_90d_at_handover` or both. A version replaces the Stage 1 choice only if its mean PR-AUC gain exceeds one paired standard error. If several do, the largest gain is kept.

**Candidates for the gates, at most two:**

- **Selected:** the choice after Stage 2.
- **Best recent-data version:** the Stage 1 configuration with recent-only or weighted training that has the highest mean PR-AUC. It is included even if the one-standard-error rule prefers all history, because validation contains little of the promise shift that recent-data training is meant to address.

A candidate identical to the Phase 2 model is not rescored as new, as its gate results are already known. Each candidate is refitted on the training orders with the same settings, using 26 May 2018 as the reference date.

## Experiments 05 to 07 candidates

Each experiment sends one fixed candidate to the gates, using C = 0.01 and the Phase 2 inputs unless stated. These are three more looks at a test set we have already seen, so a pass would be weaker evidence than a single declared test. Experiment 04 stays reserved for mix ratios.

| Experiment | Idea | Declared candidate |
| --- | --- | --- |
| **05** Recent-only monthly refits | Learn from short-promise orders as they are delivered | Refit at the start of each scored period and each calendar month, on orders with known outcomes that were purchased in the last 90 or 180 days. The window with the higher mean validation PR-AUC goes to test |
| **06** Training on orders not past their date | Focus the model on orders still in contention | Fit only on orders not past their date at handover. Orders already past their date always rank above all others |
| **07** Inputs relative to the promise | Use inputs that depend less on absolute promise length | Replace `promised_days`, `promise_slack`, `handover_days`, `route_transit_90d_at_handover`, `remaining_slack` and `slack_shortfall` with the three inputs below |

**Experiment 05 references:** monthly refits on all history are also scored on test, for comparison with the Phase 2 replay (379 late caught, 370 for the single fit and 394 for the rule, by handover month). They do not go through the gates.

**Experiment 07 inputs**, an approved exception to the rule against computing features by hand. They are calculated only from `load_cohort` columns and clipped to between -5 and 5:

- `promise_used_share`: `handover_days` / `promised_days`
- `slack_relative`: `remaining_slack` / `promised_days`
- `route_recent_vs_typical`: `route_transit_90d_at_handover` / `route_typical_days`

Note added after the run: the name `route_recent_vs_typical` is misleading. Its top is recent carrier time only, while `route_typical_days` is the route's whole purchase-to-delivery time, so it does not compare like with like (training median 0.72). The input is unchanged, as declared.

Experiment 07 also reports a version that adds these inputs to all existing inputs, on validation only.

## Experiment 04 candidates

Experiment 04 tests other mix ratios, using percentiles as in experiment 02. There are two families, each mixed with the remaining-slack rule:

- **Main family:** the Phase 2 model.
- **Exploratory family:** the promise-relative model from experiment 07. It is carried forward partly because of its test results, so its findings carry less weight. Its short-promise validation score gives some independent support.

**Fixed ratios.** The model's weight is tried at 0.1, 0.2, ... 0.9 on validation. The best ratio replaces 50/50 only if its mean PR-AUC gain over 50/50 exceeds one paired standard error, so that validation's preference for the model does not decide the ratio on its own.

**Ratio that changes with promise length.** The rule's weight is the smaller of 1 and 2 × s, and the model has the remainder. s is the share of orders purchased in the 30 days before an order's handover whose promises fall under the fitting orders' 10th percentile. So the rule has about 20% of the weight when 10% of promises are short, and all of it once 50% are. Promises are known at checkout, so this does not wait for deliveries. The doubling is a judgement call, fixed here without tuning. s uses this delivered cohort, because the prepared table does not hold other orders.

**Candidates for the gates:**

| Family | Candidates |
| --- | --- |
| Main | The chosen fixed ratio, if it is not 50/50 (50/50 already went through the gates in experiment 02), and the changing ratio |
| Exploratory | The chosen fixed ratio, which may be 50/50, and the changing ratio |

The experiment 02 mix and the experiment 07 model are also scored on test as references, and must reproduce their earlier results. There is only one large promise shift in the data, so the changing ratio cannot be shown to work in general.

## Experiment 08: final comparison

Experiment 08 compares four existing rankings side by side: the remaining-slack rule, the Phase 2 model, the Phase 2 50/50 mix and the relative changing ratio. It adds no candidates and no tuning. It does not use the gates and cannot change their results.

All test intervals use the resampling above, for each pair of rankings.

| Question | What is reported |
| --- | --- |
| **Direct comparisons** | Every pair on PR-AUC, PR-AUC on orders not past their date, and late caught in the top 10% of each purchase month |
| **Review capacity** | Late caught at 1%, 2%, 5%, 10% and 20% of each purchase month, on test and validation (top share of each window) |
| **Orders staff can still help** | Lists built only from orders not past their date, at the same shares of each purchase month, and at the top 50, 100, 200, 500 and 1,000 of the whole test period |
| **Daily lists** | Late caught at the same shares of each handover day, for all late orders and those not past their date |
| **Performance across periods** | PR-AUC, PR-AUC on orders not past their date, and top-10% recall in W1 to W5 and June to August 2018, with each ranking's shortfall behind the best of the four. May 2018 is shown but left out of the shortfalls, as it has only 2 late orders |

## Experiment 09: agreement tiers

Experiment 09 asks whether staff should first review orders flagged by both the remaining-slack rule and the Phase 2 model. It is a way of using existing rankings, so it does not use the gates and cannot change their results. The idea came from a test-set result, so the findings are exploratory.

**Setup.** As on the proposed dashboard, orders already past their date are handled separately. The rule and the model each build a top-10% list from the remaining orders of each handover day.

**Tiers.** Orders on both lists come first, then orders on one list, then the rest. Within each tier, orders are ordered by the Phase 2 50/50 mix score. Rule-only and model-only orders share a tier, because placing one above the other would rely on the test-set figures that suggested this idea. Their late rates are reported separately.

**Reported, on validation and test:**

- Orders, late orders and share late in each tier, overall and for each period (W1 to W5 and each test month)
- Late orders caught when reviewing 1%, 2%, 5%, 10% and 20% of each day's remaining orders, and PR-AUC on those orders, for the tiered ranking and the four rankings in experiment 08
- On test, 95% intervals for the tiered ranking against the Phase 2 50/50 mix and the rule, and for the share late on both lists against one list

## Experiment 10: the 50/50 mix with monthly refits

Experiment 10 checks whether the recommended Phase 2 50/50 mix holds up when the model is retrained each month. It adds no candidates and no tuning, and it does not use the gates.

Each refit follows experiment 05 with all history: at the start of each scored period and each calendar month, the model is refitted on orders purchased and handed over before that date with outcomes already known. Each order is scored with the latest refit before its handover.

| Version | Percentile scale |
| --- | --- |
| **Refreshed** | Recalculated at each refit from that refit's fitting orders. This is the version we would recommend |
| **Frozen** | Kept from the first refit of each scored period |

The single-fit 50/50 mix, the rule, and the model with monthly refits alone are shown for comparison. On test, the refreshed version gets 95% intervals against the single-fit 50/50 mix and the rule, at the review shares in experiment 08. Largest shortfalls behind the best ranking are reported as in experiment 08, across the rule, the Phase 2 model and the 50/50 versions.

## Experiment 11: re-ranking open orders each morning

Experiment 11 asks whether a dashboard should re-rank all open orders each morning (option B) instead of ranking only the previous day's handovers (option A). It adds no new model and no tuning, and it does not use the gates.

**Simulation.** It follows the report's replay in `Phase2/reports/phase2_complete_daily_replay.py`: the same periods, a review at 9am the day after each handover day, and the promised date itself counting as not yet missed. Orders delivered by 9am or already past their date are not ranked. An order leaves the queue once reviewed.

**Same workload.** Each morning, every version can review the top-10% count of the previous day's handovers (rounded up).

| Version | Orders ranked | Score |
| --- | --- | --- |
| **A: rule** | Previous day's handovers still open and before their date | Slack at handover |
| **A: 50/50** | Same | Phase 2 50/50 mix at handover |
| **Queue, fixed: rule** | All open orders before their date | Slack at handover |
| **Queue, fixed: 50/50** | Same | Phase 2 50/50 mix at handover |
| **B: rule** | Same | Updated slack |
| **B: 50/50** | Same | Mix with the rule part updated and the model part fixed at handover |

**Updated slack** = days left to the promised date − max(0, route's recent carrier time at handover − days since handover). This equals the slack at handover until a parcel outlasts its route's usual time, and equals the days left after that. Route history is kept from handover, so only time passing changes the score. The model part stays fixed because the model was trained on the moment of handover. The percentile scale also stays as it was at handover, and validation uses each window's fitted model.

**Main measure:** late orders reviewed while still before their date. Also reported: reviews used, precision, and days left when those late orders were reviewed, for each validation window, the test period and each test month.

**Decision declared in advance:** option B is recommended only if B: 50/50 beats A: 50/50 on the main measure on test, with a 95% interval above zero, and also has a higher total across the validation windows. Otherwise, the recommendation stays with option A.

**Added after the first run:** late orders reviewed with at least 1, 3 and 5 days left. These were not declared in advance. They were added because the updated rule reviewed late orders with fewer days left, and they do not change the declared decision.

**Intervals** resample mornings 1,000 times (seed 42) and compare versions on the same mornings. They treat mornings as independent, although a queue carries orders over, so they are approximate. The simulation only holds delivered orders, so a real queue would also contain orders that were never delivered, and a dashboard may show more orders than staff can review.

## Experiment 12: daily re-ranking with monthly refits

Experiment 12 tests the recommended setup as a whole: the 50/50 mix, re-ranked each morning, with the model refitted monthly. It adds no candidates and no tuning, and it does not use the gates.

It combines the simulation in experiment 11 with the refits in experiment 10 (all history, refreshed percentile scale). Each order's model percentile is fixed from the latest refit before its handover. Each morning, its updated slack is converted to a percentile on the scale of the latest refit before that morning.

**Reported:** the same measures as experiment 11, for B: 50/50 and A: 50/50 with monthly refits, alongside A: 50/50, B: 50/50 and B: rule from experiment 11. On test, intervals resample mornings as in experiment 11.

**Decision declared in advance:** the recommendation changes only if the 95% interval for B: 50/50 with refits minus B: 50/50 lies wholly below zero on late orders reviewed before their date. In that case, the recommendation would not include monthly refits until that is explained.

## Experiment 13: closest to deadline against the final strategy

Experiment 13 fills the remaining gaps for closest to deadline, an existing reference ranking. It adds no candidates and no tuning, and it does not use the gates.

**Direct comparisons.** On test, 95% intervals for the 50/50 with monthly refits, the rule and the Phase 2 model, each against closest to deadline, on the measures and resampling in experiment 08.

**Daily simulation.** Closest to deadline runs under the rules and the same review capacity as experiment 11, in two versions. New parcels only ranks the previous day's handovers by days left at handover. Re-ranked ranks all open orders by days left each morning. Both are compared with the recommended and fallback versions from experiment 12, with intervals that resample mornings.

**Shortfalls.** Each ranking's shortfall behind the best of the rule, the Phase 2 model, the 50/50 with monthly refits and closest to deadline, across W1 to W5 and June to August 2018, as in experiment 08.

## Reporting

Every experiment reports its own folder's results against the baseline above, for validation, the whole test set and each test month:

- PR-AUC, the metric used to choose in validation
- ROC-AUC, for comparison with the Phase 2 report
- Late orders caught, recall and precision in the top 10%
- Late orders caught in the top 10% of each handover day on test, the daily list staff would see (reported only, not used in the gates)
- Late orders caught and PR-AUC on orders not past their date
- PR-AUC on short-promise orders not past their date

## Limits

- **The test set is no longer untouched.** We have seen its results for these rankings, including how the model's score changes when input groups are removed from it. The data ends in August 2018, so no unseen period remains.
- **Any win is exploratory.** It suggests what to try on later data, but it does not show how the ranking would perform in live use.
