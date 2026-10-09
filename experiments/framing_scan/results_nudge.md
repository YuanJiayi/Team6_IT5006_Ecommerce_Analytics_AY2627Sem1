# Nudge policies: classifier vs reminder rule

## Timeline

- Target population (approved, handed over, has items): 97643; miss rate 0.091
- deadline - approval (days) quantiles [0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95]: [2.0, 3.99, 4.0, 6.0, 6.0, 9.0, 11.96]
- share with deadline before approval: 0.0012; allowed<1d 0.003; <2d 0.017
- handover - approval (days), on-time (n=88800) quantiles: [0.22, 0.42, 0.82, 1.58, 3.0, 4.59, 5.64]
- handover - approval (days), missed (n=8843) quantiles: [3.37, 4.18, 5.9, 7.09, 10.75, 16.89, 21.8]
- overshoot H-L for misses (hours) quantiles: [1.4, 2.9, 8.8, 24.1, 92.9, 187.3, 304.2]
- -48h: misses unshipped at deadline-48h = 1.000 (tautological: miss means H>deadline); misses whose deadline-48h falls after approval (full lead available) = 0.953; all orders still unshipped at nudge time = 0.304; miss precision among them = 0.298
- -24h: misses unshipped at deadline-24h = 1.000 (tautological: miss means H>deadline); misses whose deadline-24h falls after approval (full lead available) = 0.997; all orders still unshipped at nudge time = 0.175; miss precision among them = 0.517
- -12h: misses unshipped at deadline-12h = 1.000 (tautological: miss means H>deadline); misses whose deadline-12h falls after approval (full lead available) = 0.998; all orders still unshipped at nudge time = 0.130; miss precision among them = 0.698

Test n=19575, miss rate 0.082, late rate 0.035; test budgets (fraction of orders): {'V12': 0.142, 'V24': 0.214, 'V24/2': 0.107}

## Policy table (test | validation mean of 5 windows). nudges% of orders; recall of misses; precision; median lead (h, deadline - nudge) for caught misses; late% = share of late deliveries among nudged

| policy | test nudge% | recall | prec | lead h | late% | val nudge% | recall | prec | lead h | late% |
|---|---|---|---|---|---|---|---|---|---|---|
| P2 reminder -48h | 43.4 | 0.999 | 0.188 | 48.0 | 0.777 | 28.4 | 1.000 | 0.307 | 48.0 | 0.461 |
| P2 reminder -24h | 21.4 | 0.999 | 0.382 | 24.0 | 0.635 | 16.5 | 1.000 | 0.526 | 24.0 | 0.335 |
| P2 reminder -12h | 14.2 | 0.999 | 0.573 | 12.0 | 0.550 | 12.0 | 1.000 | 0.722 | 12.0 | 0.280 |
| P1 approval clf top V12 | 14.2 | 0.384 | 0.220 | 144.0 | 0.224 | 12.0 | 0.404 | 0.292 | 144.0 | 0.178 |
| P1 approval clf top V24 | 21.4 | 0.487 | 0.186 | 144.0 | 0.297 | 16.5 | 0.484 | 0.255 | 144.0 | 0.232 |
| P1 approval clf top V24/2 | 10.7 | 0.323 | 0.247 | 144.0 | 0.182 | 8.3 | 0.315 | 0.330 | 144.0 | 0.128 |
| P1 approval clf top 5% | 5.0 | 0.191 | 0.312 | 144.0 | 0.105 | 5.0 | 0.231 | 0.393 | 144.0 | 0.092 |
| P3 clf -48h top V12 | 14.2 | 0.604 | 0.347 | 48.0 | 0.365 | 12.0 | 0.675 | 0.489 | 48.0 | 0.250 |
| P3r prior-miss rule -48h top V12 | 14.2 | 0.559 | 0.321 | 48.0 | 0.316 | 12.0 | 0.626 | 0.451 | 48.0 | 0.222 |
| P3 clf -48h top V24 | 21.4 | 0.754 | 0.288 | 48.0 | 0.481 | 16.5 | 0.796 | 0.421 | 48.0 | 0.312 |
| P3r prior-miss rule -48h top V24 | 21.4 | 0.706 | 0.270 | 48.0 | 0.458 | 16.5 | 0.765 | 0.402 | 48.0 | 0.292 |
| P3 clf -48h top V24/2 | 10.7 | 0.514 | 0.393 | 48.0 | 0.306 | 8.3 | 0.547 | 0.575 | 48.0 | 0.189 |
| P3r prior-miss rule -48h top V24/2 | 10.7 | 0.456 | 0.348 | 48.0 | 0.254 | 8.3 | 0.468 | 0.491 | 48.0 | 0.160 |
| P3 clf -24h top V12 | 14.2 | 0.794 | 0.456 | 24.0 | 0.481 | 12.0 | 0.814 | 0.590 | 24.0 | 0.272 |
| P3r prior-miss rule -24h top V12 | 14.2 | 0.787 | 0.452 | 24.0 | 0.478 | 12.0 | 0.806 | 0.581 | 24.0 | 0.253 |
| P3 clf -24h top V24 | 21.4 | 0.999 | 0.382 | 24.0 | 0.635 | 16.5 | 1.000 | 0.526 | 24.0 | 0.335 |
| P3r prior-miss rule -24h top V24 | 21.4 | 0.999 | 0.382 | 24.0 | 0.635 | 16.5 | 1.000 | 0.526 | 24.0 | 0.335 |
| P3 clf -24h top V24/2 | 10.7 | 0.651 | 0.498 | 24.0 | 0.375 | 8.3 | 0.629 | 0.662 | 24.0 | 0.203 |
| P3r prior-miss rule -24h top V24/2 | 10.7 | 0.645 | 0.493 | 24.0 | 0.356 | 8.3 | 0.605 | 0.633 | 24.0 | 0.183 |
| P4 hybrid P1 5% + P2 -24h | 23.6 | 0.999 | 0.346 | 24.0 | 0.639 | 18.8 | 1.000 | 0.461 | 24.0 | 0.351 |

## Test by month: nudge% / recall / precision / late% (policy budgets pooled)

| policy | 2018-05 (n=772, miss 0.052) | 2018-06 (n=6139, miss 0.068) | 2018-07 (n=6112, miss 0.089) | 2018-08 (n=6551, miss 0.091) | 2018-09 (n=1, miss 0.000) |
|---|---|---|---|---|---|
| P1 approval clf top V24 | 12.4 / 0.42 / 0.18 / 0.50 | 20.1 / 0.51 / 0.17 / 0.15 | 21.9 / 0.52 / 0.21 / 0.31 | 23.1 / 0.45 / 0.18 / 0.31 | n/a |
| P2 reminder -24h | 10.5 / 1.00 / 0.49 / 0.00 | 18.0 / 1.00 / 0.38 / 0.37 | 21.5 / 1.00 / 0.41 / 0.58 | 25.6 / 1.00 / 0.36 / 0.71 | n/a |
| P2 reminder -12h | 7.8 / 1.00 / 0.67 / 0.00 | 12.2 / 1.00 / 0.56 / 0.27 | 14.9 / 1.00 / 0.60 / 0.52 | 16.3 / 1.00 / 0.56 / 0.62 | n/a |
| P3 clf -24h top V12 | 9.8 / 1.00 / 0.53 / 0.00 | 11.6 / 0.80 / 0.47 / 0.21 | 14.7 / 0.82 / 0.49 / 0.48 | 16.8 / 0.76 / 0.41 / 0.53 | n/a |
| P3 clf -24h top V24/2 | 8.9 / 0.95 / 0.55 / 0.00 | 8.6 / 0.68 / 0.54 / 0.15 | 11.7 / 0.69 / 0.52 / 0.44 | 11.9 / 0.58 / 0.44 / 0.39 | n/a |
| P3 clf -48h top V24 | 13.6 / 1.00 / 0.38 / 0.00 | 17.5 / 0.79 / 0.31 / 0.23 | 21.7 / 0.78 / 0.32 / 0.51 | 25.5 / 0.69 / 0.25 / 0.51 | n/a |
| P3r prior-miss rule -48h top V24 | 11.8 / 0.82 / 0.36 / 0.00 | 16.8 / 0.69 / 0.28 / 0.25 | 22.7 / 0.73 / 0.29 / 0.49 | 25.6 / 0.69 / 0.24 / 0.48 | n/a |
| P4 hybrid P1 5% + P2 -24h | 13.2 / 1.00 / 0.39 / 0.00 | 19.9 / 1.00 / 0.34 / 0.37 | 23.6 / 1.00 / 0.38 / 0.58 | 28.2 / 1.00 / 0.32 / 0.72 | n/a |