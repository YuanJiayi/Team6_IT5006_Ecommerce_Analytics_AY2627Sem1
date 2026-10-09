# results_cls: Task A (taskA.py) and Task B (miss_fixed.py)

## Task A
n handed-over delivered orders: 96469; dist NaN share 0.005; zip5 chain coverage n_z5>0: 0.833

## Validation mean PR-AUC, all candidates

```
clf gbm d3 all [180]        0.3685
clf gbm d3 all [all]        0.3675
clf gbm d3 rel [all]        0.3653
clf gbm d3 rel [180]        0.3641
reg gbm q80 [180]           0.3590
reg gbm q80 [all]           0.3570
reg gbm q90 [all]           0.3503
reg gbm q90 [180]           0.3500
clf logit rel [180]         0.3453
reg gbm sq [180]            0.3397
clf logit all [180]         0.3391
clf logit rel [all]         0.3386
reg gbm sq [all]            0.3355
clf logit all [all]         0.3305
rule leg_base+0.5*spread    0.3280
rule leg_base+0*spread      0.3237
rule leg_z5+1*spread        0.3156
rule leg_z3+1*spread        0.3146
rule leg_z5+0.5*spread      0.3117
rule leg_z3+0.5*spread      0.3116
reg ridge [180]             0.3111
rule leg_base+1*spread      0.3044
reg ridge [all]             0.3018
rule leg_z5+0*spread        0.2578
rule leg_z3+0*spread        0.2569
```

Selected on validation mean PR-AUC: {'(a) state-route slack rule': 'rule leg_base+0*spread', '(b) best zip/spread slack rule': 'rule leg_base+0.5*spread', '(c) best regression->slack': 'reg gbm q80 [180]', '(d) best classifier': 'clf gbm d3 all [180]'}

## VALIDATION (mean of 5 monthly windows); base late rate 0.102
                                PR-AUC  ROC-AUC  top5% prec  top5% recall  top10% prec  top10% recall
(a) state-route slack rule       0.324    0.725       0.420         0.234        0.310          0.333
(b) best zip/spread slack rule   0.328    0.746       0.439         0.240        0.337          0.355
(c) best regression->slack       0.359    0.762       0.457         0.256        0.351          0.378
(d) best classifier              0.368    0.779       0.458         0.256        0.344          0.369

## TEST (2018-05-26 on; scores pooled, top-k taken within each month); base late rate 0.033
                                PR-AUC  ROC-AUC  top5% prec  top5% recall  top10% prec  top10% recall
(a) state-route slack rule       0.452    0.881       0.362         0.500        0.224          0.618
(b) best zip/spread slack rule   0.233    0.867       0.260         0.360        0.211          0.583
(c) best regression->slack       0.313    0.880       0.326         0.451        0.232          0.642
(d) best classifier              0.326    0.819       0.281         0.388        0.190          0.524

## TEST per month: PR-AUC / top-10% recall (n, late rate)
                               May26-31 (n=610, 0.028) Jun (n=6066, 0.015) Jul (n=5985, 0.023) Aug (n=7010, 0.067)
(a) state-route slack rule               0.763 / 94.1%       0.253 / 55.1%       0.207 / 50.4%       0.593 / 65.3%
(b) best zip/spread slack rule           0.599 / 88.2%       0.135 / 53.9%       0.104 / 43.9%       0.400 / 62.3%
(c) best regression->slack               0.623 / 94.1%       0.292 / 57.3%       0.250 / 51.1%       0.495 / 68.3%
(d) best classifier                      0.698 / 82.4%       0.310 / 55.1%       0.345 / 52.5%       0.419 / 50.7%

## VALIDATION per window top-10% recall (selected)
                                  Jan    Feb    Mar    Apr May1-25
(a) state-route slack rule      34.2%  29.6%  24.8%  42.2%   35.9%
(b) best zip/spread slack rule  39.2%  33.4%  28.5%  42.7%   33.5%
(c) best regression->slack      39.0%  35.8%  27.4%  47.4%   39.3%
(d) best classifier             41.5%  35.4%  27.1%  46.1%   34.4%
## Task B (leak-fixed backlog)
# Seller shipping-limit miss prediction

Population 97643 orders; train n=77807, test n=19575. Assumption: shipping_limit_date (max over items) is known at approval. Seller = first item's seller. Backlog (leak-fixed) counts ALL seller orders approved in prior 30d and not handed to carrier at t (null carrier = unshipped).

Best rule (val PR-AUC): A prior miss rate; best model: HGB d3

| method | val base | val PR | val ROC | val P@5 | val R@5 | val P@10 | val R@10 | test base | test PR | test ROC | test P@5 | test R@5 | test P@10 | test R@10 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| A prior miss rate | 0.087 | 0.244 | 0.758 | 0.318 | 0.183 | 0.287 | 0.334 | 0.082 | 0.155 | 0.648 | 0.221 | 0.135 | 0.203 | 0.249 |
| B time allowed (short=risky) | 0.087 | 0.090 | 0.495 | 0.086 | 0.046 | 0.095 | 0.102 | 0.082 | 0.082 | 0.501 | 0.064 | 0.039 | 0.080 | 0.098 |
| C hand days - allowed | 0.087 | 0.177 | 0.655 | 0.249 | 0.135 | 0.200 | 0.217 | 0.082 | 0.113 | 0.583 | 0.141 | 0.086 | 0.135 | 0.165 |
| LogReg | 0.087 | 0.268 | 0.763 | 0.372 | 0.220 | 0.304 | 0.358 | 0.082 | 0.194 | 0.679 | 0.296 | 0.181 | 0.237 | 0.291 |
| HGB d3 | 0.087 | 0.291 | 0.780 | 0.397 | 0.233 | 0.312 | 0.364 | 0.082 | 0.213 | 0.719 | 0.314 | 0.192 | 0.254 | 0.311 |

## Test PR-AUC by month

```
         A prior miss rate  HGB d3       n   base
2018-05              0.209   0.192   772.0  0.052
2018-06              0.131   0.212  6139.0  0.068
2018-07              0.186   0.250  6112.0  0.089
2018-08              0.166   0.185  6551.0  0.091
2018-09                NaN     NaN     1.0  0.000
```

## Lateness link (top-10% flagged by best model, test)

share of ALL test late deliveries that are seller misses inside the best-model top-10% queue: 0.136; late in queue regardless of miss: 0.169; seller-miss share of all late: 0.454
overall late rate 0.035; flagged top-10% late rate 0.059; late rate if seller missed 0.193 vs not missed 0.021

## Permutation importance (test, drop in PR-AUC)

```
backlog         0.0744
allowed         0.0387
s_hand          0.0371
wday            0.0170
seller_state    0.0155
s_miss          0.0153
s_m90           0.0099
hour            0.0089
```