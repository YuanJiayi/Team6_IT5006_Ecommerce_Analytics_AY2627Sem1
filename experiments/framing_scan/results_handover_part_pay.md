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