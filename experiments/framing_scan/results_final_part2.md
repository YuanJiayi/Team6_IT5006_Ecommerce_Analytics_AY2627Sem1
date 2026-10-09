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
Random-contact baseline for top 10%: expected reached 464