# Idea 1: remove raw promise inputs from the handover classifier

## Setup

Exploratory feature ablation of the already selected logistic regression (`C=0.01`, no class weighting). All variants use the same 38,003 validation handover orders in the original five chronological windows, with each fold's model and preprocessing fitted only on outcomes known before that window. The comparator is the existing remaining-slack rule. The ranking workload is fixed at the top 10% **within each validation window**. A single fit on the original training cohort scores the already inspected later period; later workload is top 10% **within each purchase month**. August subgroups are diagnostics ranked independently within each subgroup at an equal 10% capacity. They are not alternative operational queues or a new selection criterion.

Three logistic input contracts were compared:

1. `selected_full`: the saved `variant_spec(spec)` inputs.
2. `drop_raw_only`: remove `promised_days` and `promise_slack`; retain `remaining_slack`, `past_promise_at_handover`, `slack_shortfall`, and all other inputs.
3. `slack_only_promise_path`: additionally remove `handover_days` and `route_transit_90d_at_handover`, so the raw promise cannot be algebraically reconstructed from the numeric inputs. This is a stronger, deliberately costly constraint, although it still retains the requested `past_promise_at_handover` indicator as another promise-relative signal.

Both ablations leave the original preprocessing pipeline, log transforms, categorical inputs, regularisation, label definition, and scored rows unchanged. No new model or capacity was selected using the later period.

## Exact reconstruction audit

The literal requested drop **does not impose the stated constraint**. The remaining inputs satisfy

`promised_days = remaining_slack + handover_days + route_transit_90d_at_handover`.

This holds to floating-point tolerance for 96,235 of 96,272 handover orders (99.96%); the other 37 have missing recent-route transit. Also, before removal, `promise_slack = promised_days - route_typical_days`, another direct reconstruction path. The selected full and literal-drop risk scores have Pearson correlation 0.99977 on validation and 0.99973 later. Dropping those two columns mostly re-parameterises the same linear information.

The stronger variant removes both `handover_days` and recent-route transit as separate predictors. This prevents *exact* reconstruction, but loses two legitimate signals: how long the order had already taken and the route's current carrier history. Other order/route variables can still correlate with Olist's promise; the variant does not prove independence from promise policy. `past_promise_at_handover` is a second promise-relative input that is not a function of spare time alone; `slack_shortfall` is a function of spare time.

## Five-window validation

All rankings review 3,803 of the same 38,003 orders; 4,053 were late. PR-AUC and ROC-AUC are arithmetic means across five windows; catches are totals at the equal top-10% workload.

| Ranking | Mean PR-AUC | Mean ROC-AUC | Late orders caught |
|---|---:|---:|---:|
| Selected full logistic | 0.3823 | 0.7737 | 1,315 |
| Drop raw promise columns | 0.3819 | 0.7744 | 1,321 |
| Block exact promise reconstruction | 0.3777 | 0.7717 | 1,306 |
| Remaining-slack rule | 0.3279 | 0.6971 | 1,099 |

The literal drop is effectively tied with the full model. The stronger constraint sacrifices a little average validation PR-AUC and 9 catches relative to full, although it remains substantially better than the rule in these windows. Its best individual window is W5 (PR-AUC 0.4938 versus full 0.4834), so the mean hides some variation; it is not a consistent validation improvement.

## Already inspected later period

The later period contains 19,230 scored handovers and 672 late orders. Each ranking reviews 1,924 orders in the monthly top-10% lists.

| Ranking | Later PR-AUC | Later caught | August caught (631 reviews) | August short-promise, pre-promise caught (336 reviews) |
|---|---:|---:|---:|---:|
| Selected full logistic | 0.4772 | 363 | 221 | 61 |
| Drop raw promise columns | 0.4782 | 363 | 222 | 61 |
| Block exact promise reconstruction | 0.4808 | 362 | 223 | 58 |
| Remaining-slack rule | 0.5194 | 398 | 249 | 73 |

The apparently higher later PR-AUC for the stronger variant does not translate into more late orders found at this workload. It still trails the rule, especially in August. Among August purchases with promises under 14.5 days and still before the promised date at handover, the constrained variant finds 58 of 208 late orders; the rule finds 73 at the same 336 reviews. This subgroup and cutoff were motivated by prior inspection, so these are descriptive, not fresh validation evidence.

## Conclusion

Idea 1 alone does **not** repair the August regime shift. Merely deleting the two raw promise columns changes almost nothing because the logistic model can reconstruct promise length from its other inputs. Removing the exact reconstruction path weakens the validation model slightly and still fails to beat the rule in the later, short-promise period. The results do not support replacing the saved classifier or claiming a robust promise-invariant model. A structural rule-first design is a distinct hypothesis; it needs its own like-for-like validation comparison.

## Reproduce

From the repository root, run `it5006-proj/bin/python experiments/promise_dependence/idea1/run.py`, then `it5006-proj/bin/python experiments/promise_dependence/idea1/test_outputs.py`. The folder contains `metrics.csv`, paired order-level `scores.csv.gz`, `feature_contract.json`, and `source_sha256.json`. Existing source files and saved report artifacts were read only.
