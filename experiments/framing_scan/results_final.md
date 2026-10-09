# Final corrected evaluations (scripts: final_promise.py, final_reviews.py)

# PART 1
```
Olist realised on-time, mean of 5 val windows: 89.9%
M2 target for goal 95: conformal level 0.9425 -> val mean on-time 94.96%
M2 target for goal olistval: conformal level 0.8775 -> val mean on-time 89.81%
R0 target for goal 95: conformal level 0.91 -> val mean on-time 94.94%
R0 target for goal olistval: conformal level 0.8225 -> val mean on-time 89.90%
R1 target for goal 95: conformal level 0.89 -> val mean on-time 94.94%
R1 target for goal olistval: conformal level 0.8025 -> val mean on-time 89.91%

## TEST pooled (preset targets)
method     goal  level  val_mean  val_ontime  test_mean  test_median  test_ontime  test_late     n
    M2       95 0.9425   28.1722      0.9489    17.0812         16.0       0.9591        792 19363
    M2 olistval 0.8775   22.7908      0.8968    13.8977         13.0       0.9128       1688 19363
    R0       95 0.9100   29.6374      0.9487    17.6079         15.0       0.9478       1011 19363
    R0 olistval 0.8225   23.9205      0.8979    14.0214         12.0       0.8954       2026 19363
    R1       95 0.8900   30.1930      0.9484    17.8636         16.0       0.9519        931 19363
    R1 olistval 0.8025   24.2345      0.8973    14.6644         13.0       0.9136       1673 19363
Olist actual on test: mean 22.05 median 21 on-time 96.52% late 674 of 19363

## TEST per month (on-time %, mean promise days)
         cfg                May26-31                      Jun                      Jul                      Aug
       M2/95 96.2% / 22.5d (late 29) 97.5% / 19.6d (late 154) 95.9% / 17.6d (late 252) 94.4% / 13.5d (late 357)
 M2/olistval 90.9% / 18.3d (late 69) 94.7% / 16.0d (late 326) 91.4% / 14.3d (late 530) 88.0% / 11.0d (late 763)
       R0/95 97.5% / 26.3d (late 19) 96.3% / 20.6d (late 226) 93.8% / 17.2d (late 383) 94.0% / 14.1d (late 383)
 R0/olistval 94.2% / 21.0d (late 44) 93.2% / 16.5d (late 414) 87.5% / 13.6d (late 767) 87.4% / 11.2d (late 801)
       R1/95 98.3% / 27.9d (late 13) 96.9% / 21.0d (late 186) 94.0% / 17.3d (late 372) 94.3% / 14.2d (late 360)
 R1/olistval 95.5% / 22.2d (late 34) 94.4% / 16.9d (late 342) 89.5% / 14.3d (late 644) 89.7% / 12.0d (late 653)
Olist actual  99.7% / 37.1d (late 2)  98.8% / 28.4d (late 71) 96.6% / 20.3d (late 208) 93.8% / 15.8d (late 393)

## VAL per window on-time at preset target
        cfg  val0  val1  val2  val3  val4
      M2/95 94.9% 93.0% 91.7% 97.6% 97.6%
M2/olistval 90.4% 87.0% 82.8% 94.4% 94.5%
      R0/95 94.8% 94.0% 90.9% 98.0% 97.0%
R0/olistval 90.1% 88.4% 81.9% 95.9% 93.2%
      R1/95 95.1% 94.5% 89.5% 97.7% 97.9%
R1/olistval 89.9% 88.8% 79.9% 95.8% 95.1%

## Point-estimate error (days; target = needed whole days)
 m  val MAE  val RMSE  test MAE  test RMSE
M2     5.66      8.98      3.87       5.43
R0     6.35      9.19      4.89       6.53
R1     6.31      9.29      5.50       6.96

## Matched-on-time frontier numbers (from results_geo.md, test pooled, interpolated on test frontier)
       d@95  d@97  d@Olist-ontime(96.5)  MAE
R0     17.95 20.69 19.88 4.89
R1     17.71 20.68 19.75 5.50
M2     16.21 18.60 17.75 3.87
Olist  22.05```

```
# PART 2 results

## (i) Late orders: rule 'not delivered by promised date' -> contact at promised date
Test delivered orders (purchase>=2018-05-26): 19363; late: 674 (3.5%); reviewed late: 658; bad-review rate among reviewed late: 47.0% (on-time reviewed: 8.4%)
All test bad reviews (delivered, first review<=2): 1864; late orders hold 309 (16.6%); of those answered AFTER end of promised day: 309 (16.6% of all bad); after 00:00 of promised day: 309
On-time bad reviews: 1555 (83.4%); answered after delivery: 1553 (83.3% of all bad)
Pop (on-time, answered after delivery) size in test: 18604

## (ii) On-time orders, decision at delivery. Validation (mean of 5 windows)
                                          base     PR    ROC    p@5    r@5   p@10   r@10
logit C=0.03                             0.098  0.263  0.705  0.388  0.197  0.312  0.318
logit C=0.3                              0.098  0.260  0.702  0.384  0.195  0.307  0.313
hgb d3 lr.05 it150                       0.098  0.270  0.696  0.414  0.211  0.315  0.321
hgb leaf15 lr.05 it200                   0.098  0.266  0.692  0.408  0.208  0.309  0.315
hgb leaf31 lr.03 it300                   0.098  0.265  0.693  0.400  0.204  0.309  0.315
n_items>=2                               0.098  0.181  0.619  0.290  0.147  0.285  0.289
n_sellers>=2                             0.098  0.161  0.546  0.222  0.114  0.161  0.165
seller shrunk bad rate                   0.098  0.156  0.608  0.216  0.111  0.185  0.189
combined (multi flag, then seller rate)  0.098  0.218  0.672  0.345  0.175  0.295  0.299
days early (less early = riskier)        0.098  0.108  0.510  0.138  0.069  0.128  0.129
chosen model: hgb d3 lr.05 it150; chosen rule (val PR-AUC): combined (multi flag, then seller rate)

## TEST (purchases >= 2018-05-26, once; monthly refit)
                                                     n   base     PR    ROC    p@5    r@5   p@10   r@10
MODEL hgb d3 lr.05 it150                       18604.0  0.083  0.203  0.681  0.301  0.180  0.240  0.288
MODEL(other family) logit C=0.03               18604.0  0.083  0.204  0.677  0.313  0.187  0.246  0.294
rule: n_items>=2                               18604.0  0.083  0.143  0.593  0.230  0.138  0.220  0.263
rule: n_sellers>=2                             18604.0  0.083  0.135  0.546  0.209  0.125  0.139  0.167
rule: seller shrunk bad rate                   18604.0  0.083  0.133  0.606  0.194  0.116  0.154  0.185
rule: combined (multi flag, then seller rate)  18604.0  0.083  0.176  0.658  0.266  0.159  0.233  0.279
rule: days early (less early = riskier)        18604.0  0.083  0.078  0.476  0.065  0.039  0.070  0.084

Test by month (model vs best rule):
  month    n  base  PR model  PR rule  r@10 model  r@10 rule
2018-05  756 0.081     0.265    0.201       0.279      0.295
2018-06 5996 0.094     0.225    0.228       0.311      0.306
2018-07 5908 0.083     0.169    0.137       0.255      0.237
2018-08 5944 0.074     0.223    0.162       0.275      0.280
Bootstrap (500) model minus best rule: recall@10 0.011 [-0.003, 0.025]; PR-AUC 0.027 [0.016, 0.039]

Permutation importance (test PR-AUC drop), top 10:
n_items          0.0814
n_sellers        0.0210
handover_d       0.0049
cat_c            0.0043
sel_bad_max      0.0040
deliv_dow        0.0029
sel_bad_min      0.0028
carrier_leg_d    0.0027
promise_days     0.0027
prod_bad_max     0.0024

## Combined system (i)+(ii), test: late orders contacted at end of promised day + top q% of ALL on-time delivered orders contacted at delivery
all test bad reviews = 1864; on-time delivered test orders = 18689; late = 674; rule = combined (multi flag, then seller rate)
        system  contacts  bad reached MODEL  bad reached RULE  MODEL share of all bad  RULE share of all bad
     late only       674                309               309                   0.166                  0.166
 late + top 5%      1608                589               556                   0.316                  0.298
late + top 10%      2543                756               739                   0.406                  0.396
Random-contact baseline for top 10%: expected reached 464```

## Notes / caveats
- Targets chosen on validation (grid step 0.0025) so mean of 5 val-window on-time = 95% (and = Olist val 89.9%). Test on-time differs from target because test is an easier regime (Olist 96.5% vs 89.9% val); conformal calibration (90,45 day lag) does not fully adapt.
- Late contact time = end of promised day; all 309 late bad reviews were answered after it.
- Combined system scores ALL on-time delivered test orders (including unreviewed / answered-before-delivery, which cannot be known at delivery); models are trained on the reviewed-after-delivery population. Bad reviews of non-delivered orders are excluded from the universe.
- Classifier uses first answered review per order; history features use reviews answered strictly before delivery time.
