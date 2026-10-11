# Handover classifier retraining cadence: exploratory as-of replay

## Question and setup

Does refreshing the selected `logistic_c0.01` classifier each month improve ranking over a three-month refresh, and does it address the later August drop relative to the remaining-slack rule? This is an **exploratory** follow-up to an already inspected later period. It does not replace the report's saved model selection or create a fresh holdout.

The script uses the existing handover feature builder and selected logistic pipeline. At a refit timestamp it includes an order only if its purchase, carrier handover, and recorded delivery outcome all precede that timestamp. The fitted preprocessing also sees only those orders. Predictions use the model active at each order's *handover*, not purchase. The route-history inputs remain as-of each order's handover. The same scored orders and exactly the same top-10% review count per **handover month** are used for every policy. This 10% is a fixed comparison capacity, not the report's separately selected 14% policy.

Validation scores 36,877 orders purchased in the five existing validation folds and handed over from 22 September 2017 through 25 March 2018. Both policies first fit at 22 September; monthly refreshes on each following first of month, and three-month refreshes on 22 December and 22 March. This explicit replay is necessary because the five original validation folds have unequal lengths and are not a monthly retraining schedule. For the 19,230 later test orders, the frozen policy fits on 26 May 2018 and stays fixed; monthly fits on 26 May, then 1 June, 1 July, 1 August and 1 September (the last scores only two September handovers). Monthly fits can use earlier later-period deliveries once their outcomes are actually known; this is prequential adaptation, not a fixed untouched test. The later set was already inspected before this question was posed.

## Validation comparison

| Policy | PR-AUC | Late orders caught in top 10% per handover month | Reviews |
|---|---:|---:|---:|
| Monthly logistic | 0.395 | 1,275 / 3,892 | 3,691 |
| Three-month logistic | 0.377 | 1,212 / 3,892 | 3,691 |
| Remaining-slack rule | 0.330 | 1,074 / 3,892 | 3,691 |

Monthly beats three-month refresh by 63 late orders at equal capacity. Its monthly catch advantage is concentrated in March (+37) and December (+15); it wins five of seven handover months, ties September, and loses January by one. Thus validation favors monthly refresh descriptively, but this comparison has only seven calendar months and was motivated by later-period inspection.

| Handover month | Late | Monthly caught | Three-month caught | Slack caught |
|---|---:|---:|---:|---:|
| 2017-09 | 9 | 4 | 4 | 3 |
| 2017-10 | 142 | 64 | 63 | 50 |
| 2017-11 | 642 | 189 | 185 | 149 |
| 2017-12 | 641 | 249 | 234 | 239 |
| 2018-01 | 394 | 153 | 154 | 129 |
| 2018-02 | 785 | 270 | 263 | 226 |
| 2018-03 | 1,279 | 346 | 309 | 278 |

## Later-period check

| Policy | PR-AUC | Late orders caught in top 10% per handover month | Reviews |
|---|---:|---:|---:|
| Monthly logistic | 0.482 | 375 / 672 | 1,925 |
| Frozen May-26 logistic | 0.477 | 364 / 672 | 1,925 |
| Remaining-slack rule | 0.519 | 394 / 672 | 1,925 |

The monthly refresh recovers **11** late orders relative to the frozen model, but the slack rule still catches **19 more** than monthly. For **August handovers**, at 697 reviews among 6,965 orders and 467 late orders, monthly catches **282**, frozen **273**, and slack **304**. For **August purchases**, at an equal diagnostic top 10% of 631 orders, monthly catches **229**, frozen **221**, and slack **249** of 393 late orders. The August improvement is real within this replay but too small to overturn the rule comparison.

Promise shortening arrived faster than label feedback. Promises under 14.5 days comprised 9.9% of orders with known outcomes at the 26 May fit and only 10.7% at the 1 August fit, whereas **55.6% of August purchases** in the scored cohort had such short promises. The 1 August refresh therefore had almost no August outcomes from which to learn that month's new promise pattern. This supports feedback lag as a *plausible mechanism* for the small refresh gain, but it does not prove it is the only cause.

At an equal diagnostic 10% review of the **3,505 August short-promise purchases**, monthly catches **196** of 355 late orders, frozen **192**, and slack **207**. For the **3,358 August short-promise orders still before their promised time at handover**, monthly catches **67** of 208 late orders, frozen **61**, and slack **73**, each reviewing 336. The latter comparison matters more to an intervention that must occur before a promise is missed. These subgroup top-10% counts are diagnostic rankings within the subgroup, distinct from the operational full-month queue counts above.

## Decision and boundaries

Monthly refresh is feasible and improves this classifier's historical ranking over freezing it for three months. It does **not** solve the August short-promise shift or beat the simple slack ranking in the later period, including the before-promise subgroup. A monthly-refit claim alone would therefore not justify changing the report's business recommendation or replacing the rule. To learn a new promise regime immediately, the system would need a signal available at handover, a change-aware rule, or a policy response that does not wait for delivery labels. None of those effects is established by this replay.

The replay differs from the report's original five-fold scores: it fixes calendar refit dates, scores by handover date, limits validation to handovers before 26 March, and uses a 10% comparison capacity. The later test period and August shift were known during experiment design. The data contain only realized deliveries; these rankings cannot establish whether staff interventions prevent late arrivals.

## Reproduce and inspect

From repository root: `it5006-proj/bin/python experiments/monthly_retraining/run.py`, then `it5006-proj/bin/python experiments/monthly_retraining/test_outputs.py`. The run saves `metrics.csv` (all periods and month/subgroup metrics), `fit_audit.csv` (refit counts and latest known outcome), `later_subgroups.csv` (equal-capacity subgroup diagnostics), `scores.csv.gz` (order-level paired scores/flags), and `source_sha256.json` (source/data hashes). The source code and all new outputs are confined to this experiment folder. No report, selection artifact, branch, or remote was changed.
