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