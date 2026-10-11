# Idea 2: slack rule with a bounded logistic correction

## Design

This is an exploratory test of a rule-first late-warning classifier at carrier handover. It uses the **same handover cohort and five chronological validation windows** as the saved classifier. Each fold fits only orders whose delivery outcome was known before its validation purchases began; preprocessing fits inside that fold. The later 26 May–August period had already been inspected and is diagnostic only.

The model first fits a one-variable logistic calibration on the rule score, `-remaining_slack` (with the saved fit-only median for unknown route transit):

`base_logit = intercept + slope × (-remaining_slack)`.

It then fits an L2-penalized logistic correction with that base logit as a **fixed offset**, no additional intercept, and the final correction clipped to a cap:

`risk = sigmoid(base_logit + clip(Xβ, -cap, cap))`.

The correction may distinguish similar-slack parcels but cannot freely replace the slack score. At the final fit, the base slope is about 0.141 log-odds per slack day. A cap of 0.5 permits a maximum pairwise correction difference of 1.0 log-odds, enough to reverse roughly seven days of base-score separation; it is a substantial correction. The model excludes `promised_days`, `promise_slack`, `remaining_slack`, `slack_shortfall`, `past_promise_at_handover`, `handover_days`, and `route_transit_90d_at_handover` from the correction. In particular, including both `handover_days` and `route_transit_90d_at_handover` would allow exact reconstruction of `promised_days` from remaining slack; excluding them avoids that identity. Remaining order, seller, and route inputs still provide contextual signal and may correlate with promise policy, so this is a constrained test rather than proof of policy independence.

The **primary grid** was `C ∈ {0.001, 0.01, 0.1}` and correction log-odds cap `∈ {0.5, 1.0}`. Mean five-window PR-AUC selected `C=0.001`, cap `0.5`, with smaller corrections breaking exact ties. The selected candidate was fixed before diagnostic later-period scoring. The raw slack rule and existing full `logistic C=0.01` were fitted and scored on exactly the same orders.

## Validation results

| Ranking | Mean PR-AUC | Mean recall at top 10% | Late orders caught across windows |
|---|---:|---:|---:|
| Existing full logistic C=0.01 | **0.382** | **0.360** | 1,315 |
| Selected bounded correction | 0.374 | 0.357 | 1,314 |
| Raw remaining-slack rule | 0.328 | 0.304 | 1,099 |

The bounded correction adds information to the rule: it improves PR-AUC and catches 215 more late orders at equal aggregate review count. But it does **not** improve on the existing logistic model. Its PR-AUC is lower in every one of the five windows; top-10 catch is nearly tied overall. This weakens a case for replacing the report's selected model with this particular correction.

## Already-opened later period

At a fixed top-10% review share **within each purchase month** across the later period (1,924 reviews), the selected correction catches **364 of 672** late orders, full logistic **363**, and slack **398**. PR-AUC is **0.310 / 0.477 / 0.519**, respectively. The correction is badly affected by the changed promise regime and does not rescue the August ranking.

Within the **3,358 August short-promise orders that were still before their promised date at handover** (208 late, 336 reviews per ranking), the selected correction catches **50**, full logistic **61**, and slack **73**. This is a subgroup top-10% diagnostic, not the full-month operational queue; numbers should be compared only against the same subgroup and capacity. Across all 3,505 August short-promise orders, the corresponding catches are **138 / 192 / 207** at 351 reviews.

After the selected cap of 0.5 failed in August, we ran an **explicitly post-hoc sensitivity** with smaller caps 0.05, 0.1, and 0.2. At fixed `C=0.001`, they caught **77, 81, and 75** respectively in the August short/pre-promise subgroup, compared with slack's 73. But mean validation PR-AUC fell to **0.337, 0.345, and 0.358** from the primary selected candidate's 0.374. Those caps were introduced after seeing August, so their later-period results cannot support a validation-selected policy or a fresh-test claim.

## Decision

The offset design makes slack central in a transparent way, but the primary validation-selected correction is not preferable to the current model on validation or to the simple rule in the shifted later regime. The tighter post-hoc caps suggest that constraining model adjustments can help under the new regime, but they sacrifice validation ranking and need genuinely new future data before operational claims. No report, final model, selection artifact, commit, or remote was changed.

## Reproduce

From the repository root:

`it5006-proj/bin/python experiments/promise_dependence/idea2/run.py`

`it5006-proj/bin/python experiments/promise_dependence/idea2/test_outputs.py`

Outputs are `validation_windows.csv`, `validation_summary.csv`, `later_diagnostic.csv`, `fit_audit.csv`, `selection.json`, and `source_sha256.json` in this folder. The focused output checks passed. `source_sha256.json` records the exact source and data hashes used for the run.
