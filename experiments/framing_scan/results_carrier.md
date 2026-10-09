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