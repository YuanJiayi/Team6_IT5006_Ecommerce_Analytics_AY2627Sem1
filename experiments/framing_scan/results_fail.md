# Task 1: fulfilment failure at approval

All orders 99441; without items 775 ({'unavailable': np.int64(603), 'canceled': np.int64(164), 'created': np.int64(5), 'invoiced': np.int64(2), 'shipped': np.int64(1)}); with items but not approved 14
fail (canceled/unavailable) orders overall 1234; of which have no items (no seller; cannot be scored) 767; never approved 141
Dataset end (last purchase) 2018-10-17 17:30:18; stuck cutoff 2018-08-18 17:30:18
Scored population: approved orders with items = 98652
order_status
delivered      96464
shipped         1106
canceled         461
invoiced         312
processing       301
unavailable        6
approved           2
fail in scored pop 467 (0.0047); fail2 2182; excl2 6
features: 34 numeric + 4 categorical

# ===== Target `fail` (positives in scored pop 467, rows used 98652) =====

## Validation (mean of 5 monthly windows), all candidates
```
                          base     pr    roc     p5     r5    p10    r10
seller prior fail rate   0.005  0.023  0.619  0.006  0.068  0.007  0.124
new-seller flag          0.005  0.010  0.562  0.010  0.140  0.007  0.174
seller backlog           0.005  0.004  0.457  0.000  0.000  0.003  0.057
seller prior miss rate   0.005  0.007  0.533  0.004  0.056  0.005  0.121
product prior fail rate  0.005  0.007  0.583  0.008  0.094  0.007  0.135
approval delay           0.005  0.010  0.515  0.002  0.026  0.004  0.077
logit C=0.01             0.005  0.027  0.648  0.014  0.197  0.011  0.270
logit C=0.1              0.005  0.037  0.654  0.014  0.174  0.011  0.290
logit C=1                0.005  0.019  0.657  0.012  0.155  0.011  0.255
hgb d2                   0.005  0.027  0.673  0.015  0.190  0.014  0.361
hgb d3                   0.005  0.019  0.690  0.019  0.239  0.013  0.317
hgb d5 l2=10             0.005  0.022  0.686  0.017  0.209  0.013  0.340
```
Selected on validation PR-AUC: best rule = seller prior fail rate; best model = logit C=0.1

## TEST (2018-05-26 on; n=19702, positives=87)
```
                               base     pr    roc     p5     r5    p10    r10
rule: seller prior fail rate  0.004  0.019  0.704  0.018  0.207  0.011  0.253
rule: new-seller flag         0.004  0.006  0.517  0.011  0.126  0.006  0.126
model: logit C=0.1            0.004  0.013  0.768  0.013  0.149  0.018  0.402
```

## TEST per month
```
                                                                    May26-31                                               Jun                                               Jul                                               Aug     Sep-Oct
rule: seller prior fail rate  PR 0.005 | R@5 0.00 | R@10 0.00 (n=773, pos=1)  PR 0.039 | R@5 0.20 | R@10 0.33 (n=6157, pos=15)  PR 0.038 | R@5 0.23 | R@10 0.28 (n=6164, pos=39)  PR 0.016 | R@5 0.16 | R@10 0.19 (n=6607, pos=32)  n=1, pos=0
rule: new-seller flag         PR 0.005 | R@5 0.00 | R@10 0.00 (n=773, pos=1)  PR 0.003 | R@5 0.07 | R@10 0.07 (n=6157, pos=15)  PR 0.009 | R@5 0.13 | R@10 0.13 (n=6164, pos=39)  PR 0.008 | R@5 0.16 | R@10 0.16 (n=6607, pos=32)  n=1, pos=0
model: logit C=0.1            PR 0.012 | R@5 0.00 | R@10 0.00 (n=773, pos=1)  PR 0.009 | R@5 0.20 | R@10 0.47 (n=6157, pos=15)  PR 0.019 | R@5 0.13 | R@10 0.44 (n=6164, pos=39)  PR 0.012 | R@5 0.12 | R@10 0.31 (n=6607, pos=32)  n=1, pos=0
```

## Permutation importance (test, drop in PR-AUC), top 10
```
s_ord7        0.0065
s_age         0.0046
s_hand        0.0026
catg          0.0019
same_state    0.0019
prod_n        0.0017
max_price     0.0010
namelen       0.0010
vol           0.0009
s_fail90      0.0006
```

## Review impact
Whole dataset: orders with a review 98673; bad (1-2 star) 14494 (0.147); bad reviews on fail orders 970 = 6.692% of all bad reviews; bad-review rate on fail orders 0.808 (n=1200), on others 0.139
Stuck-old orders with review 1635: bad rate 0.763; fail+stuck share of all bad reviews 15.296%
Test pop: reviewed 19604, bad 2077; bad on fail orders 73 = 3.51% of test bad reviews
5% review queue on test (bad = score<=2 on reviewed orders)
```
                              queue  fail_caught  fail_recall  bad_caught  share_of_all_bad  bad_rate_in_queue  bad_rate_overall
rule: seller prior fail rate  986.0         18.0        0.207       140.0             0.067              0.142             0.106
rule: new-seller flag         986.0         11.0        0.126        93.0             0.045              0.095             0.106
model: logit C=0.1            986.0         13.0        0.149       132.0             0.064              0.135             0.106
```

# ===== Target `fail2` (positives in scored pop 2182, rows used 98646) =====

## Validation (mean of 5 monthly windows), all candidates
```
                          base     pr    roc     p5     r5    p10    r10
seller prior fail rate   0.021  0.027  0.558  0.021  0.051  0.023  0.110
new-seller flag          0.021  0.024  0.523  0.031  0.075  0.024  0.119
seller backlog           0.021  0.021  0.497  0.015  0.037  0.019  0.094
seller prior miss rate   0.021  0.028  0.538  0.038  0.098  0.031  0.160
product prior fail rate  0.021  0.024  0.529  0.024  0.056  0.022  0.106
approval delay           0.021  0.023  0.508  0.017  0.041  0.019  0.095
logit C=0.01             0.021  0.036  0.591  0.051  0.127  0.046  0.230
logit C=0.1              0.021  0.039  0.600  0.050  0.126  0.045  0.225
logit C=1                0.021  0.038  0.600  0.049  0.125  0.042  0.213
hgb d2                   0.021  0.043  0.625  0.058  0.146  0.048  0.240
hgb d3                   0.021  0.044  0.630  0.059  0.147  0.048  0.239
hgb d5 l2=10             0.021  0.046  0.635  0.057  0.141  0.049  0.240
```
Selected on validation PR-AUC: best rule = seller prior miss rate; best model = hgb d5 l2=10

## TEST (2018-05-26 on; n=19696, positives=275)
```
                               base     pr    roc     p5     r5    p10    r10
rule: seller prior miss rate  0.014  0.019  0.587  0.015  0.055  0.020  0.142
rule: new-seller flag         0.014  0.016  0.524  0.019  0.069  0.016  0.113
model: hgb d5 l2=10           0.014  0.029  0.651  0.025  0.091  0.028  0.204
```

## TEST per month
```
                                                                    May26-31                                               Jun                                                Jul                                               Aug     Sep-Oct
rule: seller prior miss rate  PR 0.019 | R@5 0.33 | R@10 0.67 (n=773, pos=3)  PR 0.041 | R@5 0.07 | R@10 0.17 (n=6157, pos=60)  PR 0.026 | R@5 0.05 | R@10 0.16 (n=6164, pos=114)  PR 0.016 | R@5 0.03 | R@10 0.10 (n=6602, pos=98)  n=0, pos=0
rule: new-seller flag         PR 0.003 | R@5 0.00 | R@10 0.00 (n=773, pos=3)  PR 0.011 | R@5 0.05 | R@10 0.12 (n=6157, pos=60)  PR 0.023 | R@5 0.07 | R@10 0.10 (n=6164, pos=114)  PR 0.019 | R@5 0.09 | R@10 0.13 (n=6602, pos=98)  n=0, pos=0
model: hgb d5 l2=10           PR 0.171 | R@5 0.33 | R@10 0.33 (n=773, pos=3)  PR 0.035 | R@5 0.10 | R@10 0.13 (n=6157, pos=60)  PR 0.037 | R@5 0.11 | R@10 0.22 (n=6164, pos=114)  PR 0.023 | R@5 0.09 | R@10 0.19 (n=6602, pos=98)  n=0, pos=0
```

## Permutation importance (test, drop in PR-AUC), top 10
```
s_hand            0.0065
s_ord             0.0057
weight            0.0045
customer_state    0.0031
catg              0.0017
s_ord7            0.0015
vol               0.0010
seller_state      0.0008
prod_n            0.0007
max_price         0.0004
```

---

# Task 2: carrier delay alert

n delivered+handed orders 96469; late rate 0.0677
features: full 49 numeric + 4 categorical; full+rel 58

## Validation (mean over 5 monthly windows; late rate 0.102), all candidates, sorted by PR-AUC
```
                                       pr    roc     r5    r10  r10_min
(b) clf gbm_clf rel-only [all]      0.367  0.783  0.244  0.379    0.264
(b) clf gbm_clf rel-only [180]      0.365  0.783  0.242  0.368    0.259
(b) clf gbm_clf rel-only [90]       0.359  0.779  0.238  0.365    0.264
(b) clf logit full+rel [all,w60]    0.358  0.779  0.239  0.359    0.254
(b) clf gbm_clf full [all]          0.358  0.776  0.238  0.363    0.265
(b) clf gbm_clf full+rel [all]      0.357  0.778  0.239  0.359    0.245
(b) clf logit full+rel [180]        0.357  0.776  0.239  0.356    0.262
(a) leg gbm_q0.8 [90]               0.355  0.753  0.260  0.369    0.247
(a) leg gbm_q0.9 [all]              0.355  0.770  0.247  0.372    0.273
(b) clf logit full+rel [90]         0.355  0.775  0.235  0.360    0.258
(b) clf gbm_clf full+rel [90]       0.354  0.774  0.237  0.358    0.248
(a) leg gbm_q0.8 [all]              0.354  0.754  0.253  0.372    0.277
(a) leg gbm_q0.8 [180]              0.354  0.753  0.258  0.366    0.263
(b) clf gbm_clf full [180]          0.353  0.769  0.240  0.361    0.267
(b) clf gbm_clf full [all,w60]      0.352  0.768  0.243  0.353    0.254
(b) clf gbm_clf full+rel [all,w60]  0.352  0.773  0.238  0.359    0.232
(b) clf logit full [all,w60]        0.351  0.777  0.240  0.360    0.255
(b) clf gbm_clf full+rel [180]      0.351  0.776  0.228  0.349    0.230
(a) leg gbm_q0.9 [all,w60]          0.350  0.765  0.250  0.373    0.269
(a) leg gbm_q0.9 [90]               0.350  0.765  0.249  0.373    0.263
(b) clf logit full [180]            0.350  0.775  0.238  0.362    0.264
(b) clf logit rel-only [180]        0.349  0.762  0.231  0.353    0.263
(a) leg gbm_q0.9 [180]              0.348  0.764  0.245  0.369    0.267
(b) clf logit full+rel [all]        0.347  0.773  0.233  0.346    0.246
(b) clf gbm_clf full [90]           0.347  0.761  0.234  0.354    0.250
(b) clf logit rel-only [90]         0.346  0.761  0.232  0.359    0.245
(b) clf logit full [90]             0.346  0.774  0.236  0.362    0.260
(a) leg gbm_q0.8 [all,w60]          0.345  0.745  0.247  0.366    0.264
(b) clf logit rel-only [all]        0.339  0.760  0.230  0.346    0.246
(b) clf logit full [all]            0.338  0.767  0.232  0.357    0.246
(a) leg gbm_sq [all]                0.329  0.718  0.234  0.333    0.239
(a) leg gbm_sq [180]                0.329  0.717  0.236  0.336    0.242
rule slack leg_base+0.5*spread      0.328  0.746  0.240  0.355    0.285
(c) days_late gbm_sq [180]          0.328  0.724  0.230  0.337    0.210
rule slack leg_base+0*spread        0.324  0.725  0.234  0.333    0.248
(c) days_late gbm_sq [all]          0.323  0.723  0.230  0.337    0.195
(a) leg gbm_sq [90]                 0.320  0.706  0.231  0.323    0.213
(c) days_late gbm_sq [all,w60]      0.320  0.721  0.222  0.331    0.197
(c) days_late gbm_sq [90]           0.317  0.718  0.229  0.327    0.204
rule slack leg_z5+1*spread          0.316  0.741  0.237  0.356    0.271
rule slack leg_z3+1*spread          0.315  0.738  0.242  0.350    0.271
(a) leg ridge [180]                 0.314  0.720  0.219  0.313    0.214
(a) leg ridge [90]                  0.314  0.721  0.221  0.315    0.225
rule slack leg_z5+0.5*spread        0.312  0.725  0.231  0.329    0.226
rule slack leg_z3+0.5*spread        0.312  0.721  0.229  0.328    0.223
(c) days_late ridge [90]            0.311  0.724  0.215  0.315    0.222
(c) days_late ridge [180]           0.310  0.724  0.208  0.314    0.216
(a) leg ridge [all]                 0.305  0.709  0.214  0.303    0.191
rule slack leg_base+1*spread        0.304  0.749  0.214  0.342    0.278
(c) days_late ridge [all]           0.299  0.715  0.203  0.306    0.196
rule slack leg_z5+seller/cat resid  0.270  0.695  0.192  0.283    0.168
rule slack leg_z5+0*spread          0.258  0.685  0.179  0.262    0.160
rule slack leg_z3+0*spread          0.257  0.681  0.176  0.254    0.152
```
Selected on validation mean PR-AUC: {'state-route slack rule (old)': 'rule slack leg_base+0*spread', 'best rule': 'rule slack leg_base+0.5*spread', '(a) leg regression -> slack': '(a) leg gbm_q0.8 [90]', '(b) direct classifier': '(b) clf gbm_clf rel-only [all]', '(c) days-late regression': '(c) days_late gbm_sq [180]', 'overall best model': '(b) clf gbm_clf rel-only [all]'}

## TEST (2018-05-26 on, pooled; top-k within each month)
```
                               base     pr    roc     p5     r5    p10    r10
state-route slack rule (old)  0.036  0.452  0.881  0.362  0.500  0.224  0.618
best rule                     0.036  0.233  0.867  0.260  0.360  0.211  0.583
(a) leg regression -> slack   0.036  0.428  0.878  0.368  0.508  0.232  0.642
(b) direct classifier         0.036  0.538  0.882  0.376  0.520  0.225  0.621
(c) days-late regression      0.036  0.487  0.885  0.360  0.497  0.223  0.615
overall best model            0.036  0.538  0.882  0.376  0.520  0.225  0.621
```

## Validation for the selected
```
                              val PR  val R@5  val R@10  val min-window R@10
state-route slack rule (old)   0.324    0.234     0.333                0.248
best rule                      0.328    0.240     0.355                0.285
(a) leg regression -> slack    0.355    0.260     0.369                0.247
(b) direct classifier          0.367    0.244     0.379                0.264
(c) days-late regression       0.328    0.230     0.337                0.210
overall best model             0.367    0.244     0.379                0.264
```

## TEST per month (n, late rate: 610/0.028, 6066/0.015, 5985/0.023, 7010/0.067)
```
                                           May26-31                    Jun                    Jul                    Aug
state-route slack rule (old)  PR 0.763 | R@10 94.1%  PR 0.253 | R@10 55.1%  PR 0.207 | R@10 50.4%  PR 0.593 | R@10 65.3%
best rule                     PR 0.599 | R@10 88.2%  PR 0.135 | R@10 53.9%  PR 0.104 | R@10 43.9%  PR 0.400 | R@10 62.3%
(a) leg regression -> slack   PR 0.722 | R@10 94.1%  PR 0.322 | R@10 55.1%  PR 0.282 | R@10 48.9%  PR 0.648 | R@10 69.4%
(b) direct classifier         PR 0.757 | R@10 94.1%  PR 0.336 | R@10 58.4%  PR 0.343 | R@10 54.0%  PR 0.640 | R@10 64.0%
(c) days-late regression      PR 0.808 | R@10 94.1%  PR 0.333 | R@10 51.7%  PR 0.316 | R@10 48.2%  PR 0.629 | R@10 66.2%
overall best model            PR 0.757 | R@10 94.1%  PR 0.336 | R@10 58.4%  PR 0.343 | R@10 54.0%  PR 0.640 | R@10 64.0%
```

## VALIDATION per window top-10% recall (selected)
```
                                Jan    Feb    Mar    Apr May1-25
state-route slack rule (old)  34.2%  29.6%  24.8%  42.2%   35.9%
best rule                     39.2%  33.4%  28.5%  42.7%   33.5%
(a) leg regression -> slack   36.7%  36.0%  24.7%  49.7%   37.3%
(b) direct classifier         41.5%  34.0%  26.4%  46.9%   40.6%
(c) days-late regression      37.7%  31.5%  21.0%  39.6%   38.6%
overall best model            41.5%  34.0%  26.4%  46.9%   40.6%
```

## Permutation importance of (b) clf gbm_clf rel-only [all] (Jun-Aug test, month-fitted-Jun model; drop in PR-AUC; base 0.532), top 10
```
ratio_base      0.2817
ratio_z5        0.0587
slack_sel       0.0497
sp_rel          0.0493
over_limit      0.0456
vol7            0.0246
spread          0.0237
hand_frac       0.0198
slack_leg_z3    0.0197
trend           0.0194
```