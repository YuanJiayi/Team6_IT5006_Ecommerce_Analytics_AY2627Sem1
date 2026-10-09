# Task 1: promise setting

Olist validation (pooled Jan-May 2018): mean promise 24.24d, on-time 89.8%, n=33414
Olist test (2018-05-26 to 08-31): mean promise 22.05d, on-time 96.5%, n=19363

Validation selection: mean promise at matched on-time 95% / 97% / Olist-val-realised (interp)
     method      cal  d@95  d@97  d@olist
      route (90, 45) 30.50 35.13    23.90
      route (60, 21) 30.98 35.94    24.53
 gbm_sq_all (90, 45) 31.01 36.36    24.11
 gbm_sq_all (60, 21) 31.10   NaN    24.42
gbm_q90_all (90, 45) 30.45 35.26    24.12
gbm_q90_all (60, 21) 30.79   NaN    24.78
  ridge_all (90, 45) 30.58 35.49    23.79
  ridge_all (60, 21) 31.01 36.31    24.41
 gbm_sq_180 (90, 45) 30.31 35.60    23.69
 gbm_sq_180 (60, 21) 30.54   NaN    24.16
gbm_q90_180 (90, 45) 30.18 34.97    24.15
gbm_q90_180 (60, 21) 30.73   NaN    25.06
  ridge_180 (90, 45) 30.38 35.38    23.63
  ridge_180 (60, 21) 30.71 35.98    24.35

Chosen on validation: route cal=(90, 45); model=gbm_q90_180 cal=(90, 45).  (existing baseline = route, (90, 45))

## Frontier VALIDATION: mean promise days / achieved on-time per target
                                       target 0.85     target 0.9    target 0.93    target 0.95    target 0.97
route(90,45)[existing]               24.1d / 90.1%  27.3d / 93.0%  30.3d / 94.9%  33.3d / 96.3%  37.4d / 97.6%
route(90, 45)[val-best]              24.1d / 90.1%  27.3d / 93.0%  30.3d / 94.9%  33.3d / 96.3%  37.4d / 97.6%
gbm_q90_180(90, 45)[val-best model]  19.9d / 82.4%  22.9d / 88.1%  25.7d / 91.7%  28.3d / 93.7%  32.2d / 96.0%

Interpolated mean promise days at matched on-time (Olist validation realised on-time = 89.8%; Olist promise = 24.2d)
                                     on-time 95%  on-time 97%  Olist on-time
route(90,45)[existing]                     30.50        35.13          23.90
route(90, 45)[val-best]                    30.50        35.13          23.90
gbm_q90_180(90, 45)[val-best model]        30.18        34.97          24.15

## Frontier TEST: mean promise days / achieved on-time per target
                                       target 0.85     target 0.9    target 0.93    target 0.95    target 0.97
route(90,45)[existing]               15.1d / 92.6%  17.4d / 95.5%  19.5d / 97.1%  21.4d / 98.0%  24.6d / 98.7%
route(90, 45)[val-best]              15.1d / 92.6%  17.4d / 95.5%  19.5d / 97.1%  21.4d / 98.0%  24.6d / 98.7%
gbm_q90_180(90, 45)[val-best model]  13.1d / 78.9%  15.1d / 88.9%  16.9d / 94.0%  18.7d / 96.5%  21.3d / 98.0%

Interpolated mean promise days at matched on-time (Olist test realised on-time = 96.5%; Olist promise = 22.1d)
                                     on-time 95%  on-time 97%  Olist on-time
route(90,45)[existing]                     16.94        19.39          18.66
route(90, 45)[val-best]                    16.94        19.39          18.66
gbm_q90_180(90, 45)[val-best model]        17.63        19.40          18.71

VALIDATION: at target .95 share of orders with |model - route promise| >= 3d: 72.4%; model shorter by >=3d: 67.9%; longer by >=3d: 4.5%

TEST: at target .95 share of orders with |model - route promise| >= 3d: 76.7%; model shorter by >=3d: 61.4%; longer by >=3d: 15.3%

Segments VALIDATION: mean promise days at matched 95% on-time (route-best vs model-best; negative diff = model shorter)
                                               segment     n  route  model  diff
distance proxy: short route (route mean dur tercile 1) 11143  19.22  18.88 -0.34
                                   distance proxy: mid 11134  28.73  28.34 -0.39
                                  distance proxy: long 11137  42.33  43.42  1.09
                                      same-state route 12334  19.63  19.21 -0.42
                                     cross-state route 21080  36.48  36.86  0.37
                           seller <20 prior deliveries  7685  30.58  30.39 -0.20
                                     seller >=20 prior 25729  30.48  30.12 -0.36
                       route history thin (<30 in 90d)  2185  37.81  36.83 -0.98
                                    route history rich 31229  29.96  29.72 -0.24
                                          weight > 3kg  6496  33.32  32.31 -1.00

Segments TEST: mean promise days at matched 95% on-time (route-best vs model-best; negative diff = model shorter)
                                               segment     n  route  model  diff
distance proxy: short route (route mean dur tercile 1)  8431  12.82  12.02 -0.79
                                   distance proxy: mid  7284  16.75  17.63  0.88
                                  distance proxy: long  3648  25.85  25.91  0.05
                                      same-state route  7821  12.71  11.96 -0.74
                                     cross-state route 11542  19.62  20.47  0.85
                           seller <20 prior deliveries  5268  15.90  17.12  1.21
                                     seller >=20 prior 14095  17.35  17.82  0.48
                       route history thin (<30 in 90d)  1415  22.98  25.32  2.34
                                    route history rich 17948  16.44  16.95  0.51
                                          weight > 3kg  3196  19.67  19.59 -0.08

# Task 2: delay alert at carrier handover

Validation scope selection (mean per-window PR-AUC): all-history vs trailing 180d
  leg_model: all 0.3086, 180d 0.3026 -> all
  logit: all 0.3317, 180d 0.3320 -> 180
  gbm_clf: all 0.3323, 180d 0.3334 -> 180

## VALIDATION (mean over 5 monthly windows); base late rate ~0.102
                        PR-AUC  ROC-AUC  top5% prec  top5% recall  top10% prec  top10% recall
(a) slack rule           0.323    0.724       0.418         0.204        0.309          0.301
(b) leg model -> slack   0.309    0.707       0.389         0.190        0.284          0.278
(c1) logistic            0.332    0.758       0.403         0.197        0.305          0.298
(c2) GBM classifier      0.333    0.752       0.391         0.191        0.302          0.295

## TEST (4 months, scores pooled; top-k within each month); base late rate ~0.033
                        PR-AUC  ROC-AUC  top5% prec  top5% recall  top10% prec  top10% recall
(a) slack rule           0.500    0.882       0.363         0.501        0.223          0.617
(b) leg model -> slack   0.516    0.875       0.365         0.504        0.221          0.611
(c1) logistic            0.276    0.835       0.270         0.374        0.186          0.513
(c2) GBM classifier      0.330    0.768       0.271         0.375        0.170          0.469

## TEST by month (PR-AUC / top-10% recall)
                             May26-31            Jun            Jul            Aug
(a) slack rule          0.763 / 94.1%  0.306 / 53.9%  0.269 / 50.4%  0.619 / 65.3%
(b) leg model -> slack  0.755 / 94.1%  0.291 / 50.6%  0.300 / 49.6%  0.644 / 65.3%
(c1) logistic           0.527 / 82.4%  0.271 / 50.6%  0.211 / 46.0%  0.427 / 51.8%
(c2) GBM classifier     0.629 / 82.4%  0.299 / 49.4%  0.325 / 48.2%  0.418 / 44.8%
test months base late rate: 0.028, 0.015, 0.023, 0.067