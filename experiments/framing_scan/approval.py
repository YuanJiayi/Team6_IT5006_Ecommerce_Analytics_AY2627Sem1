import pandas as pd, numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor as HR
from sklearn.metrics import mean_absolute_error as MAE
D="data"
o=pd.read_csv(f"{D}/olist_orders_dataset.csv",parse_dates=["order_purchase_timestamp","order_approved_at"])
it=pd.read_csv(f"{D}/olist_order_items_dataset.csv"); pay=pd.read_csv(f"{D}/olist_order_payments_dataset.csv"); cu=pd.read_csv(f"{D}/olist_customers_dataset.csv"); se=pd.read_csv(f"{D}/olist_sellers_dataset.csv")
a=it.merge(se,on="seller_id").groupby("order_id").agg(p=("price","sum"),f=("freight_value","sum"),n=("price","size"),ss=("seller_state","first"))
pt=pay.groupby("order_id").agg(pt=("payment_type","first"),inst=("payment_installments","max"))
d=o.merge(a,on="order_id").merge(pt,on="order_id").merge(cu[["customer_id","customer_state"]],on="customer_id")
d=d[(d.pt=="boleto")&d.order_approved_at.notna()].copy()
d["y"]=(d.order_approved_at-d.order_purchase_timestamp).dt.total_seconds()/3600
t=d.order_purchase_timestamp; d["hr"]=t.dt.hour+t.dt.minute/60; d["dow"]=t.dt.dayofweek
for c in("ss","customer_state"): d[c]=d[c].astype("category").cat.codes
X=["p","f","n","ss","customer_state","hr","dow"]
d=d.sort_values("order_purchase_timestamp"); B=pd.Timestamp("2018-05-26"); V=pd.Timestamp("2018-02-01")
tr=d[t.loc[d.index]<V]; va=d[(t.loc[d.index]>=V)&(t.loc[d.index]<B)]; trf=d[t.loc[d.index]<B]; te=d[t.loc[d.index]>=B]
print("boleto n",len(d),"mean hrs",d.y.mean().round(1),"median",d.y.median().round(1))
for nm,a_,b_ in (("val",tr,va),("test",trf,te)):
    med=a_.y.median(); mh=a_.groupby(a_.order_purchase_timestamp.dt.hour).y.median()
    r1=MAE(b_.y,np.full(len(b_),med)); r2=MAE(b_.y,b_.order_purchase_timestamp.dt.hour.map(mh).fillna(med))
    m=HR(random_state=0,max_iter=150,learning_rate=.05,max_depth=4,min_samples_leaf=100).fit(a_[X],a_.y); r3=MAE(b_.y,m.predict(b_[X]))
    print(nm,f"n={len(b_)} MAE hrs: global median {r1:.2f}  median-by-hour {r2:.2f}  HGB {r3:.2f}   (mean y {b_.y.mean():.1f})")
