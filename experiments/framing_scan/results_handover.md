# Handover-stage ETA: our M2 vs Pratik (c66c7be)

Scripts: ho_pratik.py (his model, extracted to ./pratik), ho_feat.py (features as-of handover), ho_main.py (our model, val search then test once), ho_report.py, ho_extra.py, ho_pay.py (task 4).
Definitions: target = delivery_days (fractional days purchase->delivery); population = delivered, handover after purchase and before delivery (identical 19,230 test rows to Pratik; order ids and targets match exactly). ETA = elapsed purchase->handover + carrier-leg prediction. Bias-corrected = minus daily median residual of orders handed over 45-90 days earlier and delivered before that day.

Our valid-handover test rows: 19230; Pratik test cohort: 19230; overlap: 19230; ours not in his: 0; his not in ours: 0
target agreement on common test rows: max |dur_ours - delivery_days_his| = 0.0000 days
## Handover-stage ETA accuracy (days; bias = mean(pred - actual), + = over-predict)
```
               pop                                               model                 scope     n   MAE  RMSE   bias    R2
    ours-all-valid                              Ours M2 handover (raw) val mean of 5 windows 33381 5.085 8.331 -1.655 0.324
    ours-all-valid                   Ours M2 handover (bias-corrected) val mean of 5 windows 33381 5.144 8.494 -2.301 0.298
    ours-all-valid         Pratik linear (refit monthly, same windows) val mean of 5 windows 33381 5.258 8.378 -1.019 0.315
    ours-all-valid                              Ours M2 handover (raw)                  test 19230 2.649 4.355  0.916 0.458
    ours-all-valid                   Ours M2 handover (bias-corrected)                  test 19230 2.571 4.356 -1.005 0.458
    ours-all-valid         Pratik linear (refit monthly, same windows)                  test 19230 3.819 5.287  2.704 0.201
    ours-all-valid Pratik linear (his protocol: fit once on his train)                  test 19230 3.888 5.369  2.763 0.176
common-with-Pratik                              Ours M2 handover (raw) val mean of 5 windows 33381 5.085 8.331 -1.655 0.324
common-with-Pratik                   Ours M2 handover (bias-corrected) val mean of 5 windows 33381 5.144 8.494 -2.301 0.298
common-with-Pratik         Pratik linear (refit monthly, same windows) val mean of 5 windows 33381 5.258 8.378 -1.019 0.315
common-with-Pratik                              Ours M2 handover (raw)                  test 19230 2.649 4.355  0.916 0.458
common-with-Pratik                   Ours M2 handover (bias-corrected)                  test 19230 2.571 4.356 -1.005 0.458
common-with-Pratik         Pratik linear (refit monthly, same windows)                  test 19230 3.819 5.287  2.704 0.201
common-with-Pratik Pratik linear (his protocol: fit once on his train)                  test 19230 3.888 5.369  2.763 0.176
```
Pratik reported (his protocol, his 19,230 test rows): MAE 3.888, RMSE 5.369, bias +2.763, R2 0.176. Reproduced here: {'n': 19230, 'MAE': np.float64(3.8879918996934353), 'RMSE': np.float64(5.368982652829696), 'bias': np.float64(2.7631157001249287), 'R2': np.float64(0.1760597082524823)}
## Test by purchase month (common rows)
```
  month    n  ours_raw_MAE  ours_raw_bias  ours_corr_MAE  ours_corr_bias  pratik_MAE  pratik_bias
2018-05  755          3.38          -1.62           3.85           -3.04        4.10         2.29
2018-06 6050          3.02           1.06           2.91           -1.62        4.76         3.65
2018-07 6119          3.06           1.66           2.62            0.01        4.00         2.78
2018-08 6306          1.81           0.35           2.05           -1.15        2.92         1.95
```
## Late-order early warning at handover (promise = Olist estimated date; top-10% per purchase month by ETA - promise)
```
                        model     n  late  top10_recall  top10_prec  PR_AUC  rule_flags  rule_recall  rule_prec  med_warn_to_promise_d  med_warn_to_delivery_d
ours raw ETA (all valid test) 19230   672         0.632       0.221   0.549         574        0.507      0.594                  1.358                   4.463
      ours bias-corrected ETA 19230   672         0.641       0.224   0.552         300        0.371      0.830                  1.376                   4.634
       ours raw (common rows) 19230   672         0.632       0.221   0.549         574        0.507      0.594                  1.358                   4.463
 ours corrected (common rows) 19230   672         0.641       0.224   0.552         300        0.371      0.830                  1.376                   4.634
  Pratik linear (common rows) 19230   672         0.613       0.214   0.504         794        0.528      0.447                  1.366                   4.993
```
## Validation per window (monthly refit; Jan..May1-25)
```
 win    n  ours_MAE  ours_corr_MAE  pratik_MAE  ours_bias  corr_bias  pratik_bias
   0 7069      4.63           4.61        4.69      -1.30      -1.34        -1.39
   1 6555      6.29           6.48        6.12      -4.66      -5.18        -3.86
   2 7002      5.99           6.28        6.16      -3.41      -4.78        -3.27
   3 6791      4.09           4.02        4.49       0.56       0.22         1.74
   4 5964      4.43           4.33        4.83       0.53      -0.43         1.68
```
Test MAE gap Pratik(his) - ours raw: 1.24, 95% bootstrap [1.20, 1.27]
Top-10% recall gap ours raw - Pratik: 0.022, 95% bootstrap [-0.005, 0.048]
# Task 4: payment type / approval delay in the handover sub-model
Neither is in M2 (FH or FL).
## Checkout promise days at 95% on-time (validation windows), M2 handover sub-model variants
```
        base  +pay_type  +approval_delay   +both
val0  28.116     27.893           27.836  27.737
val1  34.667     34.647           34.933  34.969
val2  32.041     32.091           31.995  32.251
val3  22.205     22.257           22.357  22.403
val4  22.871     22.681           22.828  22.856
mean  27.980     27.914           27.990  28.043
```
## M2 point-estimate MAE vs needed days (validation)
```
       base  +pay_type  +approval_delay  +both
val0  5.347      5.331            5.316  5.310
val1  6.778      6.751            6.738  6.730
val2  6.406      6.395            6.381  6.370
val3  4.771      4.743            4.865  4.856
val4  4.922      4.902            4.884  4.878
mean  5.645      5.624            5.637  5.629
```
## Change vs base (days; negative = shorter promise)
```
      +pay_type  +approval_delay  +both
val0     -0.223           -0.280 -0.379
val1     -0.020            0.266  0.302
val2      0.050           -0.045  0.210
val3      0.051            0.152  0.198
val4     -0.190           -0.043 -0.015
mean     -0.066            0.010  0.063
```
Approval delay is not known at checkout, so +approval_delay is a leak test only. No variant moved d@95 by more than 0.07 days (mean) with mixed signs, so none selected and test was not reopened.
