# Phase 1 validation for Phase 2

This audit uses the checked-in raw Olist CSVs. The submitted Phase 1 PDF remains a historical artifact; use the corrected values below when discussing Phase 2. Re-running a notebook confirms that its code executes and refreshes its plots, but it does not by itself prove every interpretation. The key report claims below were also recalculated directly from the raw tables.

## Checks performed

- Re-executed all seven Phase 1 notebooks from the repository root. All code cells completed without errors. The starter notebook's final CSV-writing cell was deliberately left unexecuted; its saved output is blank so the checked-in data cannot be overwritten by this audit.
- Refreshed the notebooks' 53 saved image outputs and inspected contact sheets of all 53. These visual checks caught an overlapping chart annotation, which was removed from the freight-delay figures.
- Loaded the dashboard in Streamlit's app test: six tabs, 45 Altair charts, no Python exceptions. Decoded all 49 chart datasets as Arrow streams, checked that each is nonempty, and rendered and visually screened all 45 charts outside the responsive browser layout. Five chart datasets initially failed decoding because they included unused pandas interval columns; the dashboard data helpers now drop those columns before display. These checks establish that chart data is present and serializable; they are not independent recalculations of every plotted value or a browser layout test.
- Independently recomputed the report's principal cohort, sales, timing, rating, and stage figures from the raw tables. The audit ran with pandas 3.0.6 and NumPy 2.5.3.

## Corrected figures and interpretations

| Submitted Phase 1 claim | Recalculation / disposition |
| --- | --- |
| 96,470 eligible deliveries are 97.0% of the 98,666 item-bearing orders. | 96,470 / 98,666 = **97.8%**. The 97.0% figure uses all 99,441 orders as denominator. |
| 54.1% of late reviewed deliveries have low ratings, versus 9.2% on-time; risk ratio 5.9. | With delivery on the estimated **calendar date** counted on time: **62.4% versus 9.3%; risk ratio 6.73** among 95,824 reviewed, eligible item-bearing orders. This is an association, not a causal effect. |
| Distance sextiles have late rates of 6.3% to 11.6%. | **4.4% to 10.2%** under the corrected calendar-date rule. The 5.9-to-18.5-day recorded duration comparison remains, but the prepared CSV stores whole days and selects one seller per multi-seller order. Rebuild this feature at one-order grain for Phase 2. |
| Freight-to-price sextiles are about 7.8% to 8.4% late. | **6.5% to 7.1%**. The weak separation remains; its predictive value still requires an out-of-sample model. |
| Correlation of lateness and review score is -0.365; days-late association with low rating is 0.374. | The rerun gives **Pearson r = -0.392** for the former and **Spearman rho = 0.402** for the latter with the calendar-date rule. |
| Multi-seller orders average 8.66 delivery days versus 11.77 for single-seller orders. | These numbers compare **multi-item, delivered orders only** (1,275 versus 8,361), not all orders. Calendar-date late rates for these groups are 1.02% versus 5.94%. Product, location, timing, and selection differences have not been adjusted for. |

The prepared `smartcommerce_consolidated.csv` stores `delivery_days` as an integer from `.dt.days`. For the 96,470 eligible orders, mean exact purchase-to-delivery duration from the raw timestamps is **12.56 days**, versus **12.09 recorded whole days**. Phase 2 regression should use exact elapsed days built from raw timestamps. Phase 2 lateness classification should compare calendar dates, with delivery on the estimated date counted on time.

## Figures confirmed and limits on carryover

The raw tables reproduce **112,650 items, 98,666 orders with items, 95,420 distinct customers in that cohort, 3,095 sellers, and R$13.59 million in product sales**. The January–August order counts are 22,693 in 2017 and 53,774 in 2018; November 2017 has 7,451 orders. The top ten item categories account for 63.6% of items, the top 10% of sellers for 67.6% of product sales, and 3.05% of observed customers have repeat purchases. On the item-bearing reviewed cohort, mean review score is 4.10 and low ratings are 14.2%.

In the reviewed-order comparison, the low-rating risk ratio is **2.18** for multiple versus single items and **3.44** for multiple versus single sellers, consistent with the report's rounded 2.2 and 3.4. Orders delivered more than seven calendar days late have a **79.26%** low-rating rate, versus **8.98%** for orders delivered more than seven days early (risk ratio **8.83**, rather than the report's 8.7). These are retrospective associations and cannot serve as purchase-time features.

The selected cohort's weekly order volume peaks at **2,915** in the week of 20 November 2017; its retrospective backlog proxy peaks at **4,429** the following week. The volume/median-delivery correlation is about **0.12**. Shipping accounts for about **74%** of mean recorded lead time after excluding incomplete or negative stages. The backlog proxy excludes orders outside the selected eventually delivered cohort, and none of these observational patterns establishes a warehouse or carrier bottleneck.

The Phase 1 report's seller-volume decile late-rate range was not traced to a reproducible calculation in the checked-in notebooks or dashboard; do not carry that numeric claim into Phase 2 without a separately specified denominator and recomputation. Likewise, the source notebooks mix whole-day and exact elapsed-time measures and some figures use a single seller or city centroid for a multi-seller order. These choices are acceptable to describe as historical EDA, but they must not become Phase 2 feature definitions by default.

For Phase 2, build a fresh one-row-per-order table from raw CSVs, define the prediction time before selecting features, aggregate all sellers and items explicitly, and keep post-purchase timestamps, delivered duration, and review outcomes out of purchase-time predictors. Treat the revised notebook figures as descriptive evidence; validate feature value with held-out data.
