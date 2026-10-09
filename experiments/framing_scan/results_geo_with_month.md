# Geo experiment results (learned model vs geographic rule), promise days at matched on-time

Olist val: promise 24.24d on-time 89.8%;  Olist test: promise 22.05d on-time 96.5%

Val-selected config (criterion: mean over 5 val windows of d@95): R0=('R0', ('mu',)); R1=('R1', (180, 40, 'sp')); M1=('M1_q90', ('all', 'sp')); M2=('M2', ('180', 'mu')); M3=('M3', ('all', 'sp'))

## Validation (mean of 5 monthly windows)
       d@95%  d@97%  d@Olist-ontime  MAE(point)
R0     29.07  33.24           24.35        6.35
R1     29.16  33.63           24.52        6.31
M1     28.07  32.02           23.59        7.90
M2     28.12  30.39           23.18        5.62
M3     28.20  32.34           23.72        6.07
Olist  24.24  24.24           24.24         NaN

## TEST 2018-05-26..08-31 (evaluated once, pooled)
       d@95%  d@97%  d@Olist-ontime  MAE(point)
R0     17.95  20.69           19.88        4.89
R1     17.71  20.68           19.75        5.50
M1     17.17  19.63           18.91        6.78
M2     15.27  17.57           16.78        2.86
M3     17.14  20.03           19.10        4.12
Olist  22.05  22.05           22.05         NaN

Best model on validation: M1 ('M1_q90', ('all', 'sp'))

## Stability: promise days at 95% on-time per window (R0 / R1 / best model / difference model-R1)
  win    n    R0    R1    M1  diff
 val0 7069 29.25 29.69 28.67 -1.02
 val1 6555 35.59 35.91 34.11 -1.80
 val2 7003 32.66 32.88 32.16 -0.72
 val3 6798 22.18 22.44 21.90 -0.55
 val4 5989 25.65 24.88 23.50 -1.38
test0  760 21.88 21.87 20.89 -0.98
test1 6096 18.53 17.88 16.54 -1.33
test2 6156 18.53 18.43 17.65 -0.78
test3 6351 14.88 14.74 14.32 -0.42

## Test per-month d@95 for every method
    May26-31    Jun    Jul    Aug
R0     21.88  18.53  18.53  14.88
R1     21.87  17.88  18.43  14.74
M1     20.89  16.54  17.65  14.32
M2     20.28  15.84  16.30  13.14
M3     22.14  17.59  17.88  13.56

## Segments VALIDATION: d@95 R1 vs M1 (negative diff = model shorter)
               segment     n    R1    M1  diff
            dist short 11080 19.82 18.41 -1.41
              dist mid 11080 31.77 31.91  0.13
             dist long 11080 39.92 38.97 -0.95
          capital dest 12640 30.66 29.50 -1.16
         interior dest 20774 30.35 29.55 -0.80
            same-state 12334 19.43 18.17 -1.26
           cross-state 21080 37.14 36.74 -0.40
sparse zip5 (<5 prior) 13822 31.46 30.50 -0.96
      dense zip5 (>=5) 19592 29.80 28.89 -0.91
    seller backlog >=8  6049 33.76 31.79 -1.97
     seller backlog <8 27365 29.70 29.02 -0.68
            weight>3kg  6496 32.70 31.49 -1.21

## Segments TEST: d@95 R1 vs M1 (negative diff = model shorter)
               segment     n    R1    M1  diff
            dist short  6428 12.86 12.00 -0.86
              dist mid  6428 17.11 15.62 -1.49
             dist long  6428 23.04 21.05 -1.99
          capital dest  7663 18.04 17.32 -0.72
         interior dest 11700 17.47 17.13 -0.35
            same-state  7821 12.76 11.99 -0.77
           cross-state 11542 20.92 18.81 -2.11
sparse zip5 (<5 prior)  5378 20.01 19.30 -0.71
      dense zip5 (>=5) 13985 16.81 16.36 -0.45
    seller backlog >=8  1673 24.54 21.56 -2.98
     seller backlog <8 17690 16.69 16.79  0.11
            weight>3kg  3196 20.66 19.08 -1.58

## Permutation importance (M1, increase in test MAE of point estimate; base MAE 7.565d)
   feature  dMAE
route_vol7 0.658
  mu_route 0.600
 rt_spread 0.191
    z3_leg 0.153
    z5_leg 0.147
    z3_dur 0.088
    rd_std 0.077
  z3_dur30 0.073