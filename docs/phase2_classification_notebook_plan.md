# Plan: Phase 2 classification notebook

Hand-off brief for whoever builds `notebooks/phase2/classification.ipynb`. Read the whole file first. Sections 1 and 2 are hard rules; section 3 is the content.

## 0. Why this notebook exists, in one paragraph

We predict, at checkout, whether a delivered order will arrive after Olist's promised date. The result is modest, and the notebook's job is to show *why* honestly: lateness is mostly decided after checkout, and bad periods come and go. A reader should leave understanding (a) what we tried, (b) what each step added, (c) why the numbers are modest, and (d) how the model is still useful (ranking orders by risk).

Story in six sentences, to be echoed in the notebook's opening summary (numbers come from the saved results; the ones here are orientation):
1. Late delivery is uncommon (about 7% in training) and its rate swings by month (1% to 19%).
2. Starting from a "just guess the average" baseline, a simple regularised logistic regression is a clear first step up (PR-AUC 0.107 to 0.181). Single trees and forests start behind it, and class weighting does not help.
3. Month was removed as a feature: with at most one earlier year of each month, models memorised last year's quirks. Dropping it helped both models.
4. Three features built from past orders (route promise slack and typical days, seller ship speed) each added a little. The tree-based model gained more from them, so on the final features the logistic regression (about 0.200) and the forest (about 0.200) are effectively tied.
5. Random vs time-based split: the same models look much better on a random split (PR-AUC about 0.30 vs 0.20; ROC-AUC about 0.76 vs 0.68), because random splits mix time periods. The time-based number is the honest one.
6. The model is better used to rank orders by risk than to make yes/no calls: reviewing the riskiest 10% of orders catches about 24% of late ones, about 2.3 times better than random.

## 1. Readability rules (hard requirements)

The reader may not be a data scientist. The notebook must make sense top to bottom with no outside knowledge.

- **Every section opens with "The question" (one sentence) and closes with "What we learned" (one or two sentences).**
- **Plain English first, term second.** Write "how well the model ranks late orders above on-time ones (ROC-AUC)", not "ROC-AUC" alone.
- **One short glossary box near the top.** Define once: PR-AUC (average precision), ROC-AUC, MCC, lift, recall, precision, baseline. Two lines each, with the baseline value for context (for example, "random guessing scores 0.50 on ROC-AUC").
- **No project-internal jargon.** Do not use "Chunk 2", "Chunk 5", "maturity gap", "outcome-aware fit indices" or file hashes in the narrative. Say what the thing is: "we leave a 60-day gap between training and validation so recent orders whose delivery status was not final yet do not distort the results".
- **Figure titles state the finding**, not the chart type. Good: "Late rate swings from 1% to 19% by month". Bad: "Late rate by month".
- **One idea per figure.** Maximum one table per section unless it is the main ladder table. No walls of columns: keep the 4 to 6 columns the reader needs.
- **Consistent colours.** One colour for the baseline, one for the selected model, one for "late". Do not recolour between sections.
- **Show, do not narrate code.** Training and metric code live in `phase2_classification.py`. Notebook cells only load saved results and plot them. Hide plumbing; keep cells short.
- **Every number in the text comes from a saved file**, not typed by hand, so text and tables cannot drift. Use f-strings.
- **Be honest about limits in plain words.** Threshold metrics selected on the same data are optimistic; say so. Differences of about 0.003 PR-AUC are not meaningful; say so.
- **Run top to bottom with no errors** in a fresh kernel (`RERUN_TRAINING=False`), and check it renders cleanly with outputs saved.

## 2. Do the computing first (prerequisites, not the notebook's job)

The notebook is token-heavy to author and review. Anything computational must be finished *before* it is written, so the notebook only reads files. Order of work:

1. **Update the data contract** (`notebooks/phase2/data_prep.ipynb`, `data/phase2_feature_spec.json`, `docs/phase2_assumptions.md`):
   - Remove `purchase_month` from the features (reason: at most one earlier year of each month; decision made on validation evidence before any test use).
   - Add `route_typical_days` and `promise_slack` (`promised_days` minus `route_typical_days`). Route is seller state to customer state. Use only orders **delivered before each purchase**, shrunk toward the overall mean (prior weight 20).
   - Add `seller_ship_days`: the seller's average days from purchase to carrier handover, using only orders **handed to the carrier before each purchase**, shrunk toward the overall mean (prior weight 10). Use the same seller definition as the order table (state which seller is used for multi-seller orders and note it as a simplification).
   - Log-transform the skewed values (price, freight, distance, weight, volume) **for the linear model only**, inside its pipeline.
   - Bump `schema_version`, add the three features to the checkout-time audit with their "as of purchase" reasoning, and add a leakage test that proves each history feature uses no information from the same order or later.
   - Do not add the peak-season flag. It was tested and hurt both models; record that in the assumptions log.
2. **Extend `phase2_classification.py`** so one run saves everything the notebook needs (details in section 4). Keep fitting and metrics in the script, not the notebook.
3. **Update tests** (`tests/`), rerun the script, and regenerate `results/phase2/classification/`. Confirm all tests pass.
4. **Regenerate `selection.json` hashes** (the script does this) so the notebook's integrity checks pass.
5. **Only then** build the notebook.

Constraints for all of the above:
- Use only primary training and validation rows. The test period and the random benchmark stay untouched until final evaluation. Do not fit a final model.
- The ladder steps in section 3 are pre-declared. Do not add feature experiments beyond them. If something extra looks worth trying, list it as a follow-up.
- The order table is shared with the regression task. Any change to it must be backward compatible for regression, and the README's "Changes from the original data-preparation notebook" section should be updated.

## 3. Notebook structure

### 0. Summary (top of notebook)
Four or five bullet points: the question, the headline result with the baseline for context, the main limitation, and the practical use. Link to the glossary box.

### 1. The problem
- What we predict and when (at checkout, among delivered orders).
- Who uses it (operations team) and what they would do with a flag.
- What we deliberately leave out (no information from after checkout) and why.
- A small figure showing the timeline of what is and is not known at checkout.

### 2. The data we are working with
- Late rate overall, and a **bar chart of late rate by month** (title states the swing).
- What "late" means (calendar date after the estimated date).
- Short bridge to `data_prep.ipynb` for the full preprocessing.

### 3. Why testing is harder than it looks (time matters)
- Explain training on the past and testing on the future, in plain words.
- **Table: the same models on a random split vs a time-based split** (this is the reader's "aha"): e.g. ROC-AUC 0.76 vs 0.64. Regenerate from saved results.
- One diagram or short list of the five validation periods, with their late rates.
- State clearly: we use the time-based split, because it answers the real question.

### 4. Correlation and signal in the features (two plots, mirroring the professor's notebook)
Cover the original features and the three new ones, so the reader sees why they were added.
- **Plot A: correlation heatmap (Spearman)** of the numeric features including the engineered ones. Annotate pairs above |0.4| that are kept on purpose, and the new features' relationships to the originals (for example `promise_slack` vs `promised_days`, `route_typical_days` vs `distance_km`). Use Spearman because the features are skewed.
- **Plot B: how the features differ between late and on-time orders.** Distribution plots (density or box) for the new features, split by late vs on-time, using training rows only. Add a small bar chart of each feature's single-feature ranking power (univariate ROC-AUC) with one marker per validation fold, so the reader sees which features are consistent. Explain that `promise_slack` has a sign that looks backwards: a looser promise means *fewer* late orders, so its single-feature AUC is below 0.5.
- Takeaway sentence: most features are individually weak; that is why modest accuracy is expected.

### 5. The ladder: does each step earn its place?
This is the main section and answers the brief's "every more complex model must beat the baseline, with an honest before/after".
- **One table**, one row per step, each compared with the step before. Columns: PR-AUC, ROC-AUC, PR-AUC as a multiple of that period's late rate, and folds improved vs previous step (x/5).
- Steps (pre-declared, in this order; these are the `step` numbers in `ladder_summary.csv`):
  1. Baseline: always predict the training late rate
  2. Logistic regression, default
  3. Logistic regression, stronger regularisation (C=0.1), compared with step 2
  4. Logistic regression, class-weighted, compared with step 2 (the unadjusted vs weighted comparison Phase 1 asked for)
  5. Decision tree, default
  6. Decision tree, constrained, compared with step 5
  7. Random forest, default, compared with step 5 (forest vs single tree)
  8. Random forest, constrained, compared with step 7
  9. Drop `purchase_month` (one row per line: the best linear and the best tree-based model)
  10. Add route promise slack and route typical days
  11. Add seller ship speed
  12. Log-transform skewed inputs (linear line only)
  The feature steps (9 to 12) hold the model settings fixed, so only the features change. After step 12 the script re-runs all 13 declared candidates on the final features; the final choice comes from that run (`cv_summary.csv`).
- **Figure: a step chart** of PR-AUC down the ladder with the baseline as a reference line, one line for linear and one for tree-based.
- **Figure: PR-AUC per validation period**, baseline vs selected model, to show it beats the baseline in every period but varies a lot.
- A sentence per step in the table, saying whether it helped, did not, or hurt. Treat differences of about 0.003 or less as no change.
- Payment-feature check: results with and without `payment_type`/`max_installments` in one short table.

### 6. Linear vs tree-based: what changed along the ladder
Short, plain-language explanation with the evidence:
- At the start the logistic regression led (steps 2 to 8) and the forest and tree models trailed. Trees learn rules that are specific to a period (for example month x state x category), which do not carry over to the next period.
- The route and seller features gave the forest more than they gave the logistic model (steps 10 and 11), because they put the "is this promise generous for this route?" comparison directly into the data. On the final features the two are effectively tied (a gap of about 0.0004 PR-AUC is not meaningful).
- Say plainly that the selected candidate wins by a tiny margin and that a combination of the two (the voting comparison later) is the natural next test.
- Link back to the random-split table in section 3: on a random split the forest is ahead, which fits the "trees learn period-specific rules" explanation.

### 7. What the model learned (interpretation)
- **Logistic regression: odds-ratio chart** for the top features (as in the professor's section 7.2), with a plain-language example ("a longer-than-usual promise for this route is associated with lower odds of lateness").
- **Random forest: permutation importance** chart.
- Short note that correlated features share importance, so read related features together.

### 8. Using the model: rank orders by risk
- Explain that a fixed yes/no cut-off is brittle because the late rate changes by period.
- **Table and chart: "if the team reviews the riskiest 10%, 20%, 30% of orders"**: share of late orders caught (recall) and lift vs random, pooled and per period.
- **Precision-recall curve** with the baseline late rate as a reference.
- Secondary: the MCC-chosen threshold and its confusion matrix, with the caveat that it is selected and measured on the same data, so it is optimistic.
- Plain-language reading: how many alerts, how many correct, how many missed.

### 9. How this compares with others
Short paragraph with 3 practitioner Olist projects (chronological splits, test ROC-AUC roughly 0.66 to 0.73, PR-AUC roughly 0.08 to 0.11). Cite as practitioner projects, not peer-reviewed research. Say they use somewhat different setups and populations, so this is a sanity check, not a leaderboard.
- ArmutS/olist-delivery-risk: ROC-AUC 0.733, average precision 0.113, 5.3% late, prediction at order approval.
- HaneenDahbour/olist-late-delivery-mlops: validation ROC-AUC 0.776 falling to 0.658 on test, 4.3% late.
- junxuanc/olist-delay-prediction-analysis: random split, ROC-AUC 0.70 (not comparable).

### 10. Limits and what comes next
Plain-language list: delivered orders only; checkout-time availability is assumed; period shift; no Black Friday in the test period; residual censoring; validation scores are model-selection evidence, not the final score. Then what comes next: regression, a voting comparison, and the single final test evaluation.

### 11. Reproducibility (short)
Seeds, versions, where saved results live, how to rerun. This is the only place technical hashes appear.

## 4. Files the script must save for the notebook

All under `results/phase2/classification/`. The notebook reads these and never trains.

| File | Contents |
|---|---|
| `ladder.csv` | Per step, per fold: PR-AUC, ROC-AUC, late rate, plus the step description and what changed |
| `ladder_summary.csv` | Per step: mean PR-AUC, mean ROC-AUC, PR-AUC multiple of the late rate, folds improved vs previous step |
| `split_comparison.csv` | Same models on the random split vs the time-based split (mean PR-AUC and ROC-AUC) |
| `monthly_late_rate.csv` | Late rate and order count by purchase month (delivered orders; training period) |
| `feature_signal.csv` | Per feature, per fold: single-feature ROC-AUC |
| `feature_correlation.csv` | Spearman matrix for the numeric features, training rows only |
| `logistic_odds_ratios.csv` | Odds ratios (standardised) from the selected logistic model, fitted on the training rows |
| `forest_permutation_importance.csv` | Permutation importance on validation rows, with spread across repeats |
| `risk_ranking.csv` | Recall and lift for the riskiest 10%, 20%, 30% of orders, pooled and per fold |
| `selection.json` | As now, extended with the ladder definition, feature list and hashes |

## 5. Definition of done

- A fresh-kernel run completes with no errors and no unexplained warnings.
- Every figure has a finding-style title and every section has "The question" and "What we learned".
- A non-specialist classmate can read it top to bottom and explain the result and its limits back in a minute.
- Every number in the text is generated from a saved file.
- Tests pass, `selection.json` hashes match, and no test-period score exists anywhere.
- `docs/phase2_classification_review.md` and `notebooks/phase2/README.md` are updated so they no longer describe the old feature set or the "Chunk" labels.

## 6. Status and notes for the builder

**Done (prerequisites 1 to 4 in section 2):**
- Feature contract is schema version 2: 20 inputs, `purchase_month` retired (still in the table as a reference column), history features built through `phase2_features.py`, unit and brute-force tests in `tests/`. The table also keeps `furthest_seller_id` as a reference column (never a model input) so the seller feature can be audited.
- `phase2_classification.py` has been rerun; every file in section 4 exists in `results/phase2/classification/`. The log transform was adopted for the linear model (`selection.json`, key `log_transform`). Selected candidate: `logistic_c01`.
- All tests in `tests/test_phase2_*.py` pass.

**Things the builder must know:**
- **The existing `classification.ipynb` is out of date and will fail its hash check.** Rebuild it; do not patch it. The three PNGs from the old notebook (`candidate_comparison.png`, `fold_comparison.png`, `threshold_review.png`) were stale and have been deleted. Figures live inside the notebook; export image files from it only if the report needs them.
- **Pooled vs mean-of-folds numbers differ.** `cv_summary.csv` and the ladder use the mean of the five per-window scores. `threshold_metrics.csv` and its AP/ROC-AUC are pooled over all 38,005 validation orders. Always say which one a number is.
- **Odds ratios:** `logistic_odds_ratios.csv` includes one row per one-hot category level, and the biggest values are rare states. For the chart, show the numeric features (they are per one standard deviation) and summarise states separately or leave them out, and say that categories are relative and noisy.
- **`promise_slack` has a below-0.5 single-feature AUC** (a looser promise means fewer late orders). Explain the sign; do not present it as a mistake.
- **The step 4 comparison is class-weighted vs default at C=1** (an unadjusted vs weighted comparison, as asked for in Phase 1 feedback). Weighting lowered PR-AUC in all five windows; say so.
- **Step 9 is less consistent for the logistic model** (better in 3 of 5 windows, mean gain about 0.009) than for the forest (5 of 5). Do not overstate it. Step 10 for the logistic model is tiny (about 0.001). Report mean changes alongside "windows improved".
- **The risk-ranking numbers flag the top share within each validation window** (not one global cutoff), which is how a team with fixed capacity would work.
- Keep the "regenerate every number from files" rule: none of the figures above should be typed into the notebook.

## 7. Sources for the story (do not re-derive)

The reasoning above came from diagnostic experiments run during review, using primary training and validation rows only. Treat the numbers as indicative and regenerate them from the final script. Headline findings: random vs time-based split gap in ROC-AUC; dropping `purchase_month` improved both models; route promise slack, route typical days and seller ship speed each helped a little and consistently; recent-order-volume, seller late rate and platform recent late rate and the peak-season flag did not help; even giving a model post-purchase carrier-handover timing only reached ROC-AUC about 0.72, which is the ceiling argument for why checkout-time prediction is hard.
