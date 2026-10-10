# Great Expectations - text for the Phase 2 LaTeX report

Run on the **`josh` branch table** (schema v2: 96,470 orders, 20 inputs, 75,099 fitting / 19,363 test / 2,008 `unavailable_at_cutoff`), which is the table behind the report's results. GX Core 1.24.0. Output: `results/ge_validation.json`.

Reproduce (from a checkout of `josh` or after merging it):
```
python milestone2/ge_validation.py                      # default paths data/phase2_order_table.csv, data/phase2_feature_spec.json
```
The same script also runs on the older `main` table (schema v1; 30/30 pass).

## Main text (Data and preparation, after Table 1)
```latex
Before modelling, the prepared order table is validated with Great Expectations (GX Core 1.24; \cite{gx}). The suite encodes the cohort definition (96,470 unique orders, non-null targets), the frozen feature contract (an exact column set, so no approval, handover, delivery or review column can enter as an input), plausibility ranges and category sets; all 35 expectations pass (Appendix~\ref{app:gx}). Separate checks confirm the maturity rule: every fitting order's outcome was recorded before the 26 May cutoff, and the 2,008 orders whose outcome arrived later were held out. Running the same feature checks on each purchase month shows drift: in the three test months, 5 of 21 monthly-mean checks leave the range seen in training. GX checks values and schema only; the as-of timing of route and seller histories is verified separately by the future-truncation test.
```

## Appendix C
See the LaTeX in the report (section "Data-validation suite (Great Expectations)"). Groups and results:

| Group | Result |
|---|---|
| Cohort contract (rows, unique ids, non-null targets, labels) | 8/8 |
| Leakage guard (exact column set; no forbidden-name column) | 1/1 |
| Plausibility (targets, 13 purchase-time inputs, 3 history features) | 16/16 |
| Category sets (states, payment types, weekday, month) | 5/5 |
| Documented missingness (distance 477, weight 16, category, route history 266, seller handover 186) | 5/5 |
| Maturity and split checks (pandas) | all pass |
| Monthly drift, test months Jun-Aug 2018 | 5/21 fail |

Failed monthly checks: June freight 25.6 (training range 20.6-23.9); July promised days 19.7 (19.8-31.8) and freight 26.0; August promised days 15.2 and past seller handover time 2.9 days (3.1-4.4). Training months pass by construction; descriptive range checks, not a test of cause.

## Reference
```latex
\bibitem{gx} Great Expectations. (2026). \emph{GX Core documentation, version 1.24.0}. \url{https://docs.greatexpectations.io/docs/core/introduction/}.
```

## Link to the report's drift story
Promised days fall from 27.8 (June) to 19.7 (July) to 15.2 (August) while the training range is 19.8-31.8, and past seller handover time drops below the training range in August. This is consistent with the fixed alert cutoff flagging 9.5% of validation orders but 42.9% of test orders. Present it as a pre-scoring gate for ops, not as proof of the cause.
