# Geo experiment results (learned model vs geographic rule), promise days at matched on-time

Olist val: promise 24.24d on-time 89.8%;  Olist test: promise 22.05d on-time 96.5%

Val-selected config (criterion: mean over 5 val windows of d@95): R0=('R0', ('mu',)); R1=('R1', (180, 40, 'sp')); M1=('M1_q90', ('all', 'sp')); M2=('M2', ('all', 'mu')); M3=('M3', ('180', 'sp'))

## Validation (mean of 5 monthly windows)
       d@95%  d@97%  d@Olist-ontime  MAE(point)
R0     29.07  33.24           24.35        6.35
R1     29.16  33.63           24.52        6.31
M1     28.06  32.06           23.64        8.24
M2     28.02  31.95           23.28        5.64
M3     28.12  32.25           23.67        6.04
Olist  24.24  24.24           24.24         NaN

## TEST 2018-05-26..08-31 (evaluated once, pooled)
       d@95%  d@97%  d@Olist-ontime  MAE(point)
R0     17.95  20.69           19.88        4.89
R1     17.71  20.68           19.75        5.50
M1     18.61  21.00           20.24        8.44
M2     16.21  18.60           17.75        3.87
M3     17.45  20.33           19.45        3.86
Olist  22.05  22.05           22.05         NaN

Best model on validation: M2 ('M2', ('all', 'mu'))

## Stability: promise days at 95% on-time per window (R0 / R1 / best model / difference model-R1)
  win    n    R0    R1    M2  diff
 val0 7069 29.25 29.69 28.08 -1.61
 val1 6555 35.59 35.91 34.71 -1.20
 val2 7003 32.66 32.88 32.12 -0.76
 val3 6798 22.18 22.44 22.34 -0.10
 val4 5989 25.65 24.88 22.84 -2.04
test0  760 21.88 21.87 21.13 -0.73
test1 6096 18.53 17.88 16.16 -1.72
test2 6156 18.53 18.43 16.82 -1.60
test3 6351 14.88 14.74 13.79 -0.94

## Test per-month d@95 for every method
    May26-31    Jun    Jul    Aug
R0     21.88  18.53  18.53  14.88
R1     21.87  17.88  18.43  14.74
M1     22.17  17.25  18.54  15.35
M2     21.13  16.16  16.82  13.79
M3     21.89  18.40  18.01  13.98

## Segments VALIDATION: d@95 R1 vs M2 (negative diff = model shorter)
               segment     n    R1    M2  diff
            dist short 11080 19.82 17.47 -2.35
              dist mid 11080 31.77 30.40 -1.37
             dist long 11080 39.92 37.72 -2.20
          capital dest 12640 30.66 28.24 -2.42
         interior dest 20774 30.35 28.68 -1.67
            same-state 12334 19.43 17.30 -2.13
           cross-state 21080 37.14 35.22 -1.92
sparse zip5 (<5 prior) 13822 31.46 29.77 -1.69
      dense zip5 (>=5) 19592 29.80 27.67 -2.13
    seller backlog >=8  6049 33.76 30.44 -3.32
     seller backlog <8 27365 29.70 28.08 -1.62
            weight>3kg  6496 32.70 31.15 -1.55

## Segments TEST: d@95 R1 vs M2 (negative diff = model shorter)
               segment     n    R1    M2  diff
            dist short  6428 12.86 11.29 -1.57
              dist mid  6428 17.11 14.41 -2.70
             dist long  6428 23.04 20.70 -2.34
          capital dest  7663 18.04 16.42 -1.62
         interior dest 11700 17.47 16.14 -1.33
            same-state  7821 12.76 11.39 -1.38
           cross-state 11542 20.92 18.24 -2.69
sparse zip5 (<5 prior)  5378 20.01 18.84 -1.17
      dense zip5 (>=5) 13985 16.81 15.25 -1.56
    seller backlog >=8  1673 24.54 20.25 -4.29
     seller backlog <8 17690 16.69 15.79 -0.90
            weight>3kg  3196 20.66 18.02 -2.64

## Permutation importance (M2, increase in test MAE of point estimate; base MAE 4.153d)
   feature  dMAE
      rl90 0.668
route_vol7 0.654
    rd_std 0.415
 rt_spread 0.208
      dist 0.169
      rd_n 0.164
   backlog 0.125
      rd90 0.120

## Notes
- Month-of-year flag was dropped after a first run (kept as results_geo_with_month.md): with it M2 test d@95 was 15.27 and MAE 2.86, but ablation (ablate.txt) showed the gain came from `month` acting as a 2018 regime proxy (unseen test months fall into the May bin). With <2 years of data it is not justified.
- Cal fixed at (90,45) daily conformal; heteroscedastic scale options 1 / mu / route spread chosen on validation. R1 = route90 -> zip3 -> zip5 hierarchical shrinkage (k=40, 180d zip window, spread-scaled margin) -- it is NOT better than R0.
- Caveat: route_vol7 (and rl90) are top features; route_vol7 may partly proxy time/regime.
