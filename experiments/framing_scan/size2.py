import pandas as pd, numpy as np
D="data"
o=pd.read_csv(f"{D}/olist_orders_dataset.csv",parse_dates=["order_purchase_timestamp","order_delivered_customer_date"])
it=pd.read_csv(f"{D}/olist_order_items_dataset.csv"); pr=pd.read_csv(f"{D}/olist_products_dataset.csv")
rv=pd.read_csv(f"{D}/olist_order_reviews_dataset.csv")
its=it.merge(pr,on="product_id").merge(o[["order_id","order_purchase_timestamp"]],on="order_id")
its["q"]=its.order_purchase_timestamp.dt.to_period("Q")
c=its[(its.q>="2017Q1")&(its.q<="2018Q2")].pivot_table(index="product_category_name",columns="q",values="price",aggfunc="sum").fillna(0)
P=lambda s:pd.Period(s,"Q")
g=(c[P("2018Q2")]/c[P("2017Q2")].replace(0,np.nan)); m=c[P("2017Q2")]>50000
print("cat yoy 18Q2/17Q2:",g[m].round(2).sort_values().to_dict())
x=its.groupby("order_id").agg(p=("price","sum"),f=("freight_value","sum")).join(rv.groupby("order_id").review_score.min()); x["fr"]=x.f/x.p
print("freight/price quartile bad-rate:",x.groupby(pd.qcut(x.fr,4)).review_score.apply(lambda s:(s<=2).mean()).round(3).tolist())
# text: bad reviews w/ message share
b=rv[rv.review_score<=2]; print("bad reviews",len(b),"with message",b.review_comment_message.notna().mean().round(3))
m=b.review_comment_message.dropna().str.lower()
for k,w in {"not received":"não recebi|nao recebi|não chegou|nao chegou|ainda não|aguardando","wrong/defect":"defeito|quebrad|errado|diferente|danificad","refund":"reembolso|devol|estorno|cancel"}.items():
    print(k,m.str.contains(w).mean().round(3))
