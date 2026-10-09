# Review dissatisfaction results (scripts: rv_base.py, rv_diag.py, rv_model.py)
## Diagnosis
```
on-time delivered reviewed: 89443 bad: 8324 rate 0.093
late: 6381 0.625
share of on-time bad with text: 0.789  good: 0.373
                                                 ontime_bad  ontime_good  late_bad
pattern                                                                           
not received (não recebi/nao chegou)                  0.138        0.009     0.385
partial (recebi apenas/só/faltou/falta/um dos)        0.158        0.009     0.038
wrong/different (errado/diferente/trocado)            0.103        0.005     0.012
defect/broken (defeito/quebrado/danific)              0.084        0.006     0.007
quality (qualidade/ruim/péssim/fraco/falsific)        0.126        0.030     0.066
delay words (atraso/demor/prazo)                      0.067        0.095     0.182
delivery word (entrega)                               0.181        0.098     0.265
return/refund (devol/reembolso/estorno/cancel)        0.130        0.006     0.053
seller contact (vendedor/loja/contato/resposta)       0.104        0.035     0.118
multi-seller on-time bad n 591 partial% 0.448 notrecv% 0.355
multi-item on-time bad n 2237 partial% 0.445 notrecv% 0.223
single item on-time bad n 6087 partial% 0.053 notrecv% 0.106

cause shares of on-time bad reviews
 cause
other comment                  0.251
no comment                     0.211
partial / missing item         0.182
wrong / not as described       0.088
not received (per customer)    0.085
poor quality                   0.079
defective / damaged            0.068
delivery time                  0.037

ns
      size   mean
ns              
1   88194  0.088
2    1190  0.465
3      59  0.644

ni
      size   mean
ni              
1   80407  0.076
2    6935  0.233
3    1224  0.275
4     877  0.323

nprod
         size   mean
nprod              
1      86399  0.085
2       2668  0.321
3        376  0.370

price
                    size   mean
price                         
(0.849, 39.0]     18221  0.078
(39.0, 66.99]     17634  0.081
(66.99, 105.0]    17901  0.092
(105.0, 174.0]    17803  0.096
(174.0, 13440.0]  17884  0.118

freight_share
                   size   mean
freight_share                
(-0.001, 0.103]  17889  0.088
(0.103, 0.155]   17889  0.089
(0.155, 0.214]   17888  0.090
(0.214, 0.303]   17888  0.099
(0.303, 0.955]   17889  0.099

photos_min
              size   mean
photos_min              
(-1, 1]     43768  0.101
(1, 2]      17394  0.088
(2, 4]      16950  0.086
(4, 30]     10101  0.078

desc_mean
                    size   mean
desc_mean                     
(3.999, 314.0]    17767  0.098
(314.0, 501.0]    17604  0.098
(501.0, 739.0]    17566  0.092
(739.0, 1135.0]   17681  0.088
(1135.0, 3992.0]  17595  0.088

sel_bad_max
                    size   mean
sel_bad_max                   
(0.0155, 0.0914]  17931  0.065
(0.0914, 0.116]   17847  0.075
(0.116, 0.133]    17888  0.087
(0.133, 0.167]    17889  0.101
(0.167, 0.654]    17888  0.137

sel_n_max
              size   mean
sel_n_max               
(-1, 0]      5383  0.091
(0, 4]       8631  0.084
(4, 19]     15179  0.088
(19, 99]    27701  0.091
(99, 9999]  32549  0.100

cat_bad_max
                   size   mean
cat_bad_max                  
(0.0254, 0.116]  17898  0.076
(0.116, 0.13]    17880  0.082
(0.13, 0.142]    17888  0.087
(0.142, 0.162]   17888  0.102
(0.162, 0.448]   17889  0.117

promise_days
                     size   mean
promise_days                   
(2.009, 16.619]    17889  0.075
(16.619, 21.401]   17888  0.087
(21.401, 25.186]   17889  0.089
(25.186, 30.251]   17888  0.099
(30.251, 155.135]  17889  0.116

handover_d
                     size   mean
handover_d                     
(-171.213, 0.985]  17889  0.077
(0.985, 1.776]     17888  0.081
(1.776, 2.85]      17889  0.089
(2.85, 4.431]      17888  0.092
(4.431, 55.101]    17889  0.126
                       size   mean
cat                               
office_furniture       1137  0.185
unknown                1272  0.130
home_construction       435  0.126
bed_bath_table         8404  0.120
furniture_decor        5730  0.116
computers_accessories  6062  0.114
telephony              3765  0.108
watches_gifts          5035  0.097

multi-seller on-time n 1249 bad 0.473  bad-with-text share 0.854
multi-seller on-time vs late rates; late share among multi-seller delivered: 0.01 vs single 0.067
  - em partes sou cliente há anos, porém sempre há divergências e problemas na entrega, consta como produto entregue, porém foi entregue apenas 
  - parte do produto foi entregue, falta um relogio, embora nesse email conste que foram entregues os tres que fazem parte do pedido.a nota fisc
  - este foi o pedido
balde com 128 peças - blocos de montar 2 un - r$ 25,00 cada (não foi entregue)
vendido e entregue targaryen
tapete de e
  - so recebi metade dos produtos a nota fiscal veio no valor inferior ao pago .. não recomendo ! dei uma estrela pq eh obrigada a dar pelo meno
  - recebi apenas um cobertor de casal. o de solteiro não é a mercadoria foi deixado do lado de fora de casa sem ninguém que receber.
  - recebi apenas um dos produtos,relógio de perola.mas o outro relógio de led,nem sinal dele.no entanto,foi dado como entregue.como pode?a emba
  - não recomendo srs, efetuei a compra de 2 produtos, entregaram apenas a varredoura feiticeira mod. compact plus, sendo que o parceiro foi a t
  - boa noite.

comprei 3 produtos, e recebi apenas 1, estou aguardando um retorno ref. aos demais produtos comprados.

aguardo retorno brev

share of on-time bad that is multi-item(>=2): 0.269  multi-seller: 0.071 ; of on-time orders: 0.101 0.014
single-item on-time bad share: 0.7312590100913022
other comment                  0.271
no comment                     0.240
wrong / not as described       0.107
not received (per customer)    0.099
poor quality                   0.098
defective / damaged            0.086
partial / missing item         0.053
delivery time                  0.046
```
## Point A
```

===== POINT A: n=97917 base=0.142
VALIDATION (mean of 5 windows)
                                                  base     PR    ROC    p@5    r@5   p@10   r@10
logit C=0.03                                    0.167  0.289  0.647  0.416  0.129  0.348  0.214
logit C=0.3                                     0.167  0.288  0.646  0.415  0.129  0.349  0.214
hgb d3 lr.05 it150                              0.167  0.287  0.642  0.428  0.133  0.346  0.213
hgb leaf15 lr.05 it200                          0.167  0.284  0.639  0.403  0.125  0.347  0.216
hgb leaf31 lr.03 it300                          0.167  0.285  0.639  0.417  0.130  0.345  0.214
seller prior bad rate                           0.167  0.221  0.572  0.293  0.091  0.255  0.156
n_sellers>=2 flag                               0.167  0.203  0.518  0.258  0.080  0.205  0.125
combined (multi-seller flag, then seller rate)  0.167  0.246  0.583  0.364  0.115  0.296  0.184
PR-AUC std across windows: {'logit C=0.03': 0.05, 'logit C=0.3': 0.05, 'hgb d3 lr.05 it150': 0.045, 'hgb leaf15 lr.05 it200': 0.042, 'hgb leaf31 lr.03 it300': 0.04, 'seller prior bad rate': 0.042, 'n_sellers>=2 flag': 0.039, 'combined (multi-seller flag, then seller rate)': 0.031} 
chosen model: logit C=0.03  chosen rule: combined (multi-seller flag, then seller rate)

TEST (purchases >= 2018-05-26, once): train n=75541 test n=19552
                                                       base     PR    ROC    p@5    r@5   p@10   r@10
MODEL logit C=0.03                                    0.106  0.192  0.615  0.302  0.142  0.237  0.224
rule: seller prior bad rate                           0.106  0.155  0.589  0.224  0.106  0.177  0.167
rule: n_sellers>=2 flag                               0.106  0.145  0.532  0.209  0.098  0.156  0.147
rule: combined (multi-seller flag, then seller rate)  0.106  0.184  0.602  0.297  0.140  0.214  0.202
MODEL(other family) hgb d3 lr.05 it150                0.106  0.190  0.620  0.298  0.140  0.237  0.224

TEST by month (model vs best rule):
   month    n  base  PR model  PR rule  r@10 model  r@10 rule  p@10 model  p@10 rule
2018-05  762 0.087     0.268    0.249       0.288      0.273       0.250      0.237
2018-06 6134 0.107     0.215    0.189       0.273      0.203       0.292      0.217
2018-07 6228 0.109     0.171    0.172       0.193      0.205       0.210      0.223
2018-08 6427 0.105     0.191    0.192       0.202      0.192       0.212      0.201
bootstrap model-rule diff: recall@10 0.023 [0.006, 0.040]; PR-AUC 0.008 [-0.002, 0.018]

Permutation importance (test PR-AUC drop), top 10:
 n_sellers        0.0243
sel_bad_max      0.0174
n_items          0.0171
price            0.0110
pay_value        0.0094
sel_n_max        0.0073
prod_bad_max     0.0047
max_price        0.0041
photos_mean      0.0041
freight_share    0.0035
```
## Point B
```

===== POINT B: n=95824 base=0.128
VALIDATION (mean of 5 windows)
                                 base     PR    ROC    p@5    r@5   p@10   r@10
logit C=0.03                   0.155  0.492  0.792  0.647  0.217  0.585  0.384
logit C=0.3                    0.155  0.491  0.792  0.653  0.219  0.586  0.385
hgb d3 lr.05 it150             0.155  0.531  0.794  0.748  0.251  0.630  0.409
hgb leaf15 lr.05 it200         0.155  0.533  0.794  0.745  0.250  0.628  0.407
hgb leaf31 lr.03 it300         0.155  0.530  0.793  0.742  0.249  0.630  0.409
days late                      0.155  0.444  0.684  0.709  0.237  0.569  0.357
late flag (ties by days late)  0.155  0.444  0.684  0.708  0.236  0.567  0.356
late + multi-seller            0.155  0.471  0.715  0.716  0.240  0.608  0.389
PR-AUC std across windows: {'logit C=0.03': 0.113, 'logit C=0.3': 0.113, 'hgb d3 lr.05 it150': 0.129, 'hgb leaf15 lr.05 it200': 0.129, 'hgb leaf31 lr.03 it300': 0.127, 'days late': 0.145, 'late flag (ties by days late)': 0.145, 'late + multi-seller': 0.134} 
chosen model: hgb leaf15 lr.05 it200  chosen rule: late + multi-seller

TEST (purchases >= 2018-05-26, once): train n=73829 test n=19278
                                      base     PR    ROC    p@5    r@5   p@10   r@10
MODEL hgb leaf15 lr.05 it200         0.097  0.314  0.713  0.477  0.247  0.338  0.350
rule: days late                      0.097  0.210  0.554  0.346  0.179  0.204  0.211
rule: late flag (ties by days late)  0.097  0.210  0.554  0.345  0.179  0.203  0.210
rule: late + multi-seller            0.097  0.244  0.597  0.454  0.235  0.262  0.271
MODEL(other family) logit C=0.03     0.097  0.299  0.712  0.455  0.235  0.343  0.355

TEST by month (model vs best rule):
   month    n  base  PR model  PR rule  r@10 model  r@10 rule  p@10 model  p@10 rule
2018-05  758 0.083     0.300    0.252       0.317      0.222       0.263      0.184
2018-06 6072 0.100     0.286    0.198       0.333      0.213       0.333      0.213
2018-07 6118 0.097     0.291    0.235       0.309      0.267       0.299      0.258
2018-08 6330 0.095     0.371    0.322       0.412      0.386       0.393      0.368
bootstrap model-rule diff: recall@10 0.080 [0.064, 0.094]; PR-AUC 0.070 [0.058, 0.081]

Permutation importance (test PR-AUC drop), top 10:
 days_late           0.1458
n_items             0.0627
n_sellers           0.0113
handover_d          0.0048
cat_c               0.0036
customer_state_c    0.0023
carrier_leg_d       0.0023
approve_h           0.0022
sel_bad_max         0.0021
promise_days        0.0021
```
