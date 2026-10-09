# Opportunity scan results
Scripts: inv.py size.py size2.py seller_snap.py approval.py. Raw outputs below.
## Inventory

## olist_customers_dataset.csv rows=99441
   customer_id [str] miss=0.0% card=99441
   customer_unique_id [str] miss=0.0% card=96096
   customer_zip_code_prefix [int64] miss=0.0% card=14994 range=1003..9.999e+04
   customer_city [str] miss=0.0% card=4119
   customer_state [str] miss=0.0% card=27

## olist_geolocation_dataset.csv rows=1000163
   geolocation_zip_code_prefix [int64] miss=0.0% card=19015 range=1001..9.999e+04
   geolocation_lat [float64] miss=0.0% card=717363 range=-36.61..45.07
   geolocation_lng [float64] miss=0.0% card=717615 range=-101.5..121.1
   geolocation_city [str] miss=0.0% card=8011
   geolocation_state [str] miss=0.0% card=27

## olist_order_items_dataset.csv rows=112650
   order_id [str] miss=0.0% card=98666
   order_item_id [int64] miss=0.0% card=21 range=1..21
   product_id [str] miss=0.0% card=32951
   seller_id [str] miss=0.0% card=3095
   shipping_limit_date [str] miss=0.0% card=93318 range=2016-09-19 00:15:34..2020-04-09 22:35:08
   price [float64] miss=0.0% card=5968 range=0.85..6735
   freight_value [float64] miss=0.0% card=6999 range=0..409.7

## olist_order_payments_dataset.csv rows=103886
   order_id [str] miss=0.0% card=99440
   payment_sequential [int64] miss=0.0% card=29 range=1..29
   payment_type [str] miss=0.0% card=5
   payment_installments [int64] miss=0.0% card=24 range=0..24
   payment_value [float64] miss=0.0% card=29077 range=0..1.366e+04

## olist_order_reviews_dataset.csv rows=99224
   review_id [str] miss=0.0% card=98410
   order_id [str] miss=0.0% card=98673
   review_score [int64] miss=0.0% card=5 range=1..5
   review_comment_title [str] miss=88.3% card=4527
   review_comment_message [str] miss=58.7% card=36159
   review_creation_date [str] miss=0.0% card=636 range=2016-10-02 00:00:00..2018-08-31 00:00:00
   review_answer_timestamp [str] miss=0.0% card=98248 range=2016-10-07 18:32:28..2018-10-29 12:27:35

## olist_orders_dataset.csv rows=99441
   order_id [str] miss=0.0% card=99441
   customer_id [str] miss=0.0% card=99441
   order_status [str] miss=0.0% card=8
   order_purchase_timestamp [str] miss=0.0% card=98875 range=2016-09-04 21:15:19..2018-10-17 17:30:18
   order_approved_at [str] miss=0.2% card=90733
   order_delivered_carrier_date [str] miss=1.8% card=81018 range=2016-10-08 10:34:01..2018-09-11 19:48:28
   order_delivered_customer_date [str] miss=3.0% card=95664 range=2016-10-11 13:46:32..2018-10-17 13:22:46
   order_estimated_delivery_date [str] miss=0.0% card=459 range=2016-09-30 00:00:00..2018-11-12 00:00:00

## olist_products_dataset.csv rows=32951
   product_id [str] miss=0.0% card=32951
   product_category_name [str] miss=1.9% card=73
   product_name_lenght [float64] miss=1.9% card=66 range=5..76
   product_description_lenght [float64] miss=1.9% card=2960 range=4..3992
   product_photos_qty [float64] miss=1.9% card=19 range=1..20
   product_weight_g [float64] miss=0.0% card=2204 range=0..4.042e+04
   product_length_cm [float64] miss=0.0% card=99 range=7..105
   product_height_cm [float64] miss=0.0% card=102 range=2..105
   product_width_cm [float64] miss=0.0% card=95 range=6..118

## olist_sellers_dataset.csv rows=3095
   seller_id [str] miss=0.0% card=3095
   seller_zip_code_prefix [int64] miss=0.0% card=2246 range=1001..9.973e+04
   seller_city [str] miss=0.0% card=611
   seller_state [str] miss=0.0% card=23

## product_category_name_translation.csv rows=71
   product_category_name [str] miss=0.0% card=71
   product_category_name_english [str] miss=0.0% card=71
## Sizing
cat yoy 18Q2/17Q2: {'cool_stuff': 0.9, 'perfumaria': 1.08, 'brinquedos': 1.49, 'informatica_acessorios': 1.49, 'ferramentas_jardim': 1.59, 'esporte_lazer': 1.84, 'automotivo': 2.07, 'cama_mesa_banho': 2.32, 'moveis_decoracao': 2.45, 'utilidades_domesticas': 2.74, 'beleza_saude': 2.91, 'relogios_presentes': 3.36}
freight/price quartile bad-rate: [0.14, 0.136, 0.142, 0.146]
bad reviews 14575 with message 0.747
not received 0.266
wrong/defect 0.099
refund 0.098
{'delivered': 96478, 'shipped': 1107, 'canceled': 625, 'unavailable': 609, 'invoiced': 314, 'processing': 301, 'created': 5, 'approved': 2}
orders by month: {Period('2017-01', 'M'): 800, Period('2017-02', 'M'): 1780, Period('2017-03', 'M'): 2682, Period('2017-04', 'M'): 2404, Period('2017-05', 'M'): 3700, Period('2017-06', 'M'): 3245, Period('2017-07', 'M'): 4026, Period('2017-08', 'M'): 4331, Period('2017-09', 'M'): 4285, Period('2017-10', 'M'): 4631, Period('2017-11', 'M'): 7544, Period('2017-12', 'M'): 5673, Period('2018-01', 'M'): 7269, Period('2018-02', 'M'): 6728, Period('2018-03', 'M'): 7211, Period('2018-04', 'M'): 6939, Period('2018-05', 'M'): 6873, Period('2018-06', 'M'): 6167, Period('2018-07', 'M'): 6292, Period('2018-08', 'M'): 6512, Period('2018-09', 'M'): 16}

PAYMENT TYPE: n, share, approval hrs mean/median/p90, unapproved%, cancelled%, unavail%
                 n  appr_mean  appr_med  appr_p90  unappr   canc   unav
ptype                                                                  
boleto       19784     33.125    29.068    67.273   0.002  0.005  0.008
credit_card  75387      4.556     0.271    21.359   0.001  0.006  0.006
debit_card    1527      9.549     0.852    24.038   0.000  0.005  0.004
not_defined      3        NaN       NaN       NaN   1.000  1.000  0.000
voucher       2739      8.352     0.285    24.917   0.026  0.032  0.007
boleto approval >24h share: 0.571 ; >48h 0.182 ; cc >24h 0.074

review response days median/p90 1.67 4.86 ; with comment msg 0.413
score dist {1: 0.115, 2: 0.032, 3: 0.082, 4: 0.193, 5: 0.578}
msg presence by score {1: 0.77, 2: 0.68, 3: 0.43, 4: 0.31, 5: 0.36}

GMV 13591644.0 items 112650 orders w/items 98666
multi-seller orders: 1278 0.013
bad review rate multi vs single: 0.472 0.137
top10% order value > 270.0 bad rate 0.184 vs rest 0.137 share GMV 0.41
installments dist (cc): {0: 0.0, 1: 0.331, 2: 0.162, 3: 0.136, 4: 0.092, 5: 0.068, 6: 0.051, 7: 0.021, 8: 0.056, 9: 0.008, 10: 0.074}

products w/ >1 seller: 1225 of 32951 ; GMV share 0.136
product units median 1.0 ; products w/1 unit 0.55
{'n': 1.0, 'ph': 0.015, 'dl': 0.026, 'nl': 0.027}

sellers 3095 ; median orders 6.0 ; sellers <=3 orders 0.37 ; top10% GMV share 0.675
first-order by quarter: {Period('2016Q3', 'Q-DEC'): 3, Period('2016Q4', 'Q-DEC'): 142, Period('2017Q1', 'Q-DEC'): 552, Period('2017Q2', 'Q-DEC'): 316, Period('2017Q3', 'Q-DEC'): 371, Period('2017Q4', 'Q-DEC'): 427, Period('2018Q1', 'Q-DEC'): 374, Period('2018Q2', 'Q-DEC'): 565, Period('2018Q3', 'Q-DEC'): 345}
new sellers (first<=2018-05-01): 2387 ; orders in first 90d median 4.0 ; 1-order share 0.238
2018-02-26 active last180d: 1626 dormant next90: 549 33.8% their trailing GMV share 0.091
2018-05-26 active last180d: 1896 dormant next90: 634 33.4% their trailing GMV share 0.103

state demand vs sellers
                       sum   size  sellers  gmv_per_seller
customer_state                                           
SP              5202955.0  41375   1849.0          2814.0
RJ              1824093.0  12762    171.0         10667.0
MG              1585308.0  11544    244.0          6497.0
RS               750304.0   5432    129.0          5816.0
PR               683084.0   4998    349.0          1957.0
SC               520553.0   3612    190.0          2740.0
BA               511350.0   3358     19.0         26913.0
DF               302604.0   2125     30.0         10087.0
GO               294592.0   2007     40.0          7365.0
ES               275037.0   2025     23.0         11958.0
PE               262788.0   1648      9.0         29199.0
CE               227255.0   1327     13.0         17481.0
share of GMV same-state: 0.313 ; from SP sellers: 0.644
  File "/Users/joshualum/Documents/VSCode/IT5006_Grp_6/it5006-proj/lib/python3.14/site-packages/pandas/core/indexes/base.py", line 3641, in get_loc
    return self._engine.get_loc(casted_key)
           ~~~~~~~~~~~~~~~~~~~~^^^^^^^^^^^^
  File "pandas/_libs/index.pyx", line 168, in pandas._libs.index.IndexEngine.get_loc
  File "pandas/_libs/index.pyx", line 197, in pandas._libs.index.IndexEngine.get_loc
  File "pandas/_libs/hashtable_class_helper.pxi", line 7668, in pandas._libs.hashtable.PyObjectHashTable.get_item
  File "pandas/_libs/hashtable_class_helper.pxi", line 7676, in pandas._libs.hashtable.PyObjectHashTable.get_item
KeyError: '2017Q2'

The above exception was the direct cause of the following exception:

## Seller snapshot tests
rows 13311 19808 49436 1896

=== A. Seller dormancy (active in last 180d, zero orders next 90d) ===
val n 19808 prevalence 0.328 trailing GMV share of dormant 0.088
test n 1896 prevalence 0.334 trailing GMV share of dormant 0.103
[val] rule: days since last order  PR-AUC 0.732  ROC 0.846  prec@10% 0.857  recall@10% 0.261
[val] rule: -orders in last 90d    PR-AUC 0.677  ROC 0.841  prec@10% 0.826  recall@10% 0.251
[val] rule: -orders in last 30d    PR-AUC 0.739  ROC 0.858  prec@10% 0.857  recall@10% 0.261
[val] rule: -(orders180) tie-> recency PR-AUC 0.664  ROC 0.803  prec@10% 0.799  recall@10% 0.243
[test] rule: days since last order PR-AUC 0.715  ROC 0.835  prec@10% 0.820  recall@10% 0.244
[test] rule: -orders in last 90d   PR-AUC 0.666  ROC 0.829  prec@10% 0.799  recall@10% 0.238
[test] rule: -orders in last 30d   PR-AUC 0.720  ROC 0.847  prec@10% 0.825  recall@10% 0.246
[test] rule: -(orders180) tie-> recency PR-AUC 0.647  ROC 0.785  prec@10% 0.788  recall@10% 0.235
[val] HGB cand0                    PR-AUC 0.768  ROC 0.873  prec@10% 0.883  recall@10% 0.269
[val] HGB cand1                    PR-AUC 0.759  ROC 0.868  prec@10% 0.871  recall@10% 0.265
[val] HGB cand2                    PR-AUC 0.757  ROC 0.866  prec@10% 0.874  recall@10% 0.266
chosen 0
[test] HGB (trained to 2018-02-24) PR-AUC 0.760  ROC 0.864  prec@10% 0.873  recall@10% 0.260
test top10% flagged 189 sellers: 87.3% dormant; their trailing-180d GMV 56,501 = 1.0% of total; mean orders180 flagged 2.3 vs all 22.0
worth-saving sellers (>=10 orders/180d): n 710 prevalence 0.104
  [test, >=10 orders] HGB          PR-AUC 0.502  ROC 0.860  prec@10% 0.577  recall@10% 0.554
  [test, >=10 orders] rule recency PR-AUC 0.468  ROC 0.847  prec@10% 0.521  recall@10% 0.500
new sellers (tenure<=120d): n 623 prevalence 0.316
  [test, new] HGB                  PR-AUC 0.678  ROC 0.822  prec@10% 0.823  recall@10% 0.259
  [test, new] rule recency         PR-AUC 0.632  ROC 0.787  prec@10% 0.710  recall@10% 0.223
perm importance top: {'rec': 0.119, 'ho180': 0.018, 'topcat': 0.012, 'o90': 0.009, 'o30': 0.006, 'score180': 0.004}

=== B. Seller quality deterioration: next-90d bad-review rate ===
n val/test 4634 354 mean next bad rate val/test 0.187 0.108
val share with next bad>=0.25: 0.259 ; of those, trailing bad180< 0.25 : 0.7928452579034941
  [val] regression MAE: platform mean 0.0940  trailing180 rate 0.1080  shrunk(k=20) 0.0941
test share with next bad>=0.25: 0.076 ; of those, trailing bad180< 0.25 : 0.5185185185185185
  [test] regression MAE: platform mean 0.0884  trailing180 rate 0.0882  shrunk(k=20) 0.0825
[val] rule trailing bad rate       PR-AUC 0.438  ROC 0.663  prec@10% 0.555  recall@10% 0.214
[val] rule shrunk bad rate         PR-AUC 0.473  ROC 0.683  prec@10% 0.613  recall@10% 0.236
[val] rule late rate               PR-AUC 0.352  ROC 0.608  prec@10% 0.449  recall@10% 0.173
[test] rule trailing bad rate      PR-AUC 0.335  ROC 0.741  prec@10% 0.343  recall@10% 0.444
[test] rule shrunk bad rate        PR-AUC 0.310  ROC 0.736  prec@10% 0.314  recall@10% 0.407
[test] rule late rate              PR-AUC 0.132  ROC 0.628  prec@10% 0.171  recall@10% 0.222
[val] HGB cand0                    PR-AUC 0.381  ROC 0.628  prec@10% 0.402  recall@10% 0.155
[val] HGB cand1                    PR-AUC 0.381  ROC 0.634  prec@10% 0.395  recall@10% 0.152
[test] HGB cls                     PR-AUC 0.223  ROC 0.748  prec@10% 0.257  recall@10% 0.333
[test] HGB regression MAE 0.0895
[val] HGB regression MAE 0.1012
test top10% flagged: 35 sellers; their next-90d orders 1630.0 mean next bad rate 0.193 vs all 0.108
## Boleto approval
boleto n 19600 mean hrs 33.1 median 29.0
val n=5087 MAE hrs: global median 17.20  median-by-hour 16.55  HGB 14.70   (mean y 32.7)
test n=3585 MAE hrs: global median 18.83  median-by-hour 18.15  HGB 15.93   (mean y 34.1)
