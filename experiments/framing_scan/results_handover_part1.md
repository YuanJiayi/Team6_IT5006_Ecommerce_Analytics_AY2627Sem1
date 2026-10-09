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