# Recent-history and handover diagnostic

Run from the repository root:

```sh
it5006-proj/bin/python experiments/phase2_recent_history_audit.py
it5006-proj/bin/python tests/test_phase2_recent_history_audit.py -q
```

The script writes `checkout.csv` and `handover.csv`. It does not change the
frozen model, feature contract, notebook, or original results. All new comparisons
were designed after the later-period test had been opened, so their test scores
are **post hoc diagnostics**, not independent evidence of selected-model
performance. The five original validation windows remain the primary comparison
for these candidates.

## Checkout forecast

All candidates retain the previously selected ridge model (`alpha=100`) and the
same outcome-aware chronological folds. Each history statistic uses only earlier
orders whose delivery was recorded **strictly before** the order's purchase.
Route statistics are shrunk toward the corresponding recent all-route average
using 20 pseudo-orders. `route30/90/180` replace the original all-history route
mean and recompute `promise_slack`. `shipping90` adds a 90-day mean of completed
carrier-to-customer time for the route. `global90` adds the all-route 90-day
delivery mean. `weighted90` gives recently purchased fitting orders greater
weight in ridge, with a 90-day half-life; its preprocessing remains unweighted.

| Variant | Mean validation MAE | Opened test MAE | Opened test bias |
| --- | ---: | ---: | ---: |
| Frozen model | 5.810 | 5.001 | +3.659 |
| Recent route delivery, 30 days | 5.763 | 4.501 | +2.929 |
| Recent route delivery, 90 days | 5.829 | 4.802 | +3.351 |
| Recent route delivery, 180 days | 5.827 | 5.019 | +3.677 |
| Added recent route shipping, 90 days | **5.679** | **4.490** | +2.889 |
| Added global delivery trend, 90 days | 5.904 | 4.967 | +3.608 |
| Recently weighted ridge, 90-day half-life | 5.692 | 4.994 | +3.570 |

The recent route-shipping feature lowered MAE in all five validation windows
(improvements of 0.035–0.303 days). Its opened-test improvement is larger but
must be treated as exploratory. A global recent average alone did not improve
validation. The frozen row reproduces the saved evaluation.

## Forecast update at carrier handover

On orders with a valid handover before delivery, predict total days as **observed
purchase-to-handover days + historical carrier-to-customer days**. The historical
shipping mean is calculated as of handover, using only deliveries completed
strictly earlier. The handover and saved checkout forecasts are compared on the
same 19,230 later-period orders and corresponding validation cohorts. This is a
later prediction point; it does not improve what could have been known at checkout.

| Variant | Mean validation MAE | Opened test MAE |
| --- | ---: | ---: |
| Saved checkout model, same handover cohort | 5.809 | 4.991 |
| All-history route shipping | 5.375 | 4.144 |
| Recent route shipping, 180 days | 5.320 | 4.656 |
| Recent route shipping, 90 days | **5.304** | **3.762** |
| Recent route shipping, 30 days | 5.503 | 2.916 |
| Recent global shipping, 90 days | 6.556 | 4.638 |
| Recent global shipping, 30 days | 6.725 | 3.798 |

The 90-day route baseline improved over checkout in every validation window
(0.383–0.675 days). Its opened-test improvement was 1.229 days on the same
cohort. The 30-day baseline looks best only on the already-open test, so its
test result must not be used to select it. Route-specific history mattered:
global recent shipping alone was worse in validation.

These results support testing a recent carrier-to-customer signal at checkout
and offering a simple handover update. They do not identify the real-world cause
of the faster carrier leg or establish performance on a new future period.
