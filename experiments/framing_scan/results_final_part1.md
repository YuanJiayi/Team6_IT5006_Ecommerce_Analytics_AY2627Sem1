Olist realised on-time, mean of 5 val windows: 89.9%
M2 target for goal 95: conformal level 0.9425 -> val mean on-time 94.96%
M2 target for goal olistval: conformal level 0.8775 -> val mean on-time 89.81%
R0 target for goal 95: conformal level 0.91 -> val mean on-time 94.94%
R0 target for goal olistval: conformal level 0.8225 -> val mean on-time 89.90%
R1 target for goal 95: conformal level 0.89 -> val mean on-time 94.94%
R1 target for goal olistval: conformal level 0.8025 -> val mean on-time 89.91%

## TEST pooled (preset targets)
method     goal  level  val_mean  val_ontime  test_mean  test_median  test_ontime  test_late     n
    M2       95 0.9425   28.1722      0.9489    17.0812         16.0       0.9591        792 19363
    M2 olistval 0.8775   22.7908      0.8968    13.8977         13.0       0.9128       1688 19363
    R0       95 0.9100   29.6374      0.9487    17.6079         15.0       0.9478       1011 19363
    R0 olistval 0.8225   23.9205      0.8979    14.0214         12.0       0.8954       2026 19363
    R1       95 0.8900   30.1930      0.9484    17.8636         16.0       0.9519        931 19363
    R1 olistval 0.8025   24.2345      0.8973    14.6644         13.0       0.9136       1673 19363
Olist actual on test: mean 22.05 median 21 on-time 96.52% late 674 of 19363

## TEST per month (on-time %, mean promise days)
         cfg                May26-31                      Jun                      Jul                      Aug
       M2/95 96.2% / 22.5d (late 29) 97.5% / 19.6d (late 154) 95.9% / 17.6d (late 252) 94.4% / 13.5d (late 357)
 M2/olistval 90.9% / 18.3d (late 69) 94.7% / 16.0d (late 326) 91.4% / 14.3d (late 530) 88.0% / 11.0d (late 763)
       R0/95 97.5% / 26.3d (late 19) 96.3% / 20.6d (late 226) 93.8% / 17.2d (late 383) 94.0% / 14.1d (late 383)
 R0/olistval 94.2% / 21.0d (late 44) 93.2% / 16.5d (late 414) 87.5% / 13.6d (late 767) 87.4% / 11.2d (late 801)
       R1/95 98.3% / 27.9d (late 13) 96.9% / 21.0d (late 186) 94.0% / 17.3d (late 372) 94.3% / 14.2d (late 360)
 R1/olistval 95.5% / 22.2d (late 34) 94.4% / 16.9d (late 342) 89.5% / 14.3d (late 644) 89.7% / 12.0d (late 653)
Olist actual  99.7% / 37.1d (late 2)  98.8% / 28.4d (late 71) 96.6% / 20.3d (late 208) 93.8% / 15.8d (late 393)

## VAL per window on-time at preset target
        cfg  val0  val1  val2  val3  val4
      M2/95 94.9% 93.0% 91.7% 97.6% 97.6%
M2/olistval 90.4% 87.0% 82.8% 94.4% 94.5%
      R0/95 94.8% 94.0% 90.9% 98.0% 97.0%
R0/olistval 90.1% 88.4% 81.9% 95.9% 93.2%
      R1/95 95.1% 94.5% 89.5% 97.7% 97.9%
R1/olistval 89.9% 88.8% 79.9% 95.8% 95.1%

## Point-estimate error (days; target = needed whole days)
 m  val MAE  val RMSE  test MAE  test RMSE
M2     5.66      8.98      3.87       5.43
R0     6.35      9.19      4.89       6.53
R1     6.31      9.29      5.50       6.96

## Matched-on-time frontier numbers (from results_geo.md, test pooled, interpolated on test frontier)
       d@95  d@97  d@Olist-ontime(96.5)  MAE
R0     17.95 20.69 19.88 4.89
R1     17.71 20.68 19.75 5.50
M2     16.21 18.60 17.75 3.87
Olist  22.05