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