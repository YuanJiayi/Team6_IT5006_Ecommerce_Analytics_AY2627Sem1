# Regression scan (scripts in reg/: c1_handover.py c2_demand.py c3_freight.py c4_seller.py; logs c2.log c4.log)
Split: val = Mar-1..May-26 2018 (train < Mar-1) for setting choice; test = >= 2018-05-26 (train < 05-26), scored once.
| Cand | Metric | Best baseline (val/test) | Best model (val/test) | Rel. improvement (test) |
|---|---|---|---|---|
| 1 Seller handover days | MAE | seller-median+weekday/hour offset 1.417/1.490 (seller median alone 1.537/1.566) | GBM (L1) 1.231/1.264 | 15% (19% vs seller median) |
| 2 Weekly demand, state/category orders & GMV, h=1-4 | WAPE | trailing 8-12wk mean e.g. state orders 0.131/0.234 | GBM/ridge 0.148/0.274 | negative (models worse) |
| 3 Freight (single-seller orders) | MAE R$ | log-linear (weight,dist,states) 6.86/7.05; route x weight-bucket median 7.79/8.0 | GBM 5.27/5.73 | 19% (mean freight R$22.6) |
| 4 Seller 90d GMV | WAPE | last-90d 0.689/0.845 (blend 0.836) | GBM 0.627/0.702 | 16-17%; top-10% capture 0.574->0.569 (none) |
Notes: backlog feature helps handover only slightly (test MAE 1.294 -> 1.264); seller history is the bigger driver (no hist 1.503). P80 SLA: GBM cover 82.4% at mean 3.5d vs baseline 83.4% at 4.0d. Freight: stable ~R$11-15/kg billable, 10.6% of test orders >50% above prediction (flag candidates), under-prediction ~0.1%. Seller GMV test is a single snapshot (n=1879, window ends ~Aug-24, data thins late Aug).
