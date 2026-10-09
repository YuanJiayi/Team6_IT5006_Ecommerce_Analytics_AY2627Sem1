import pandas as pd, numpy as np
D="data"
T=["order_purchase_timestamp","order_approved_at","order_delivered_carrier_date","order_delivered_customer_date","order_estimated_delivery_date"]
o=pd.read_csv(f"{D}/olist_orders_dataset.csv",parse_dates=T)
it=pd.read_csv(f"{D}/olist_order_items_dataset.csv",parse_dates=["shipping_limit_date"])
pay=pd.read_csv(f"{D}/olist_order_payments_dataset.csv")
rv=pd.read_csv(f"{D}/olist_order_reviews_dataset.csv",parse_dates=["review_creation_date","review_answer_timestamp"])
pr=pd.read_csv(f"{D}/olist_products_dataset.csv"); se=pd.read_csv(f"{D}/olist_sellers_dataset.csv"); cu=pd.read_csv(f"{D}/olist_customers_dataset.csv")
print(o.order_status.value_counts().to_dict())
print("orders by month:",o.groupby(o.order_purchase_timestamp.dt.to_period("M")).size().loc["2017-01":"2018-09"].to_dict())
# payments
pt=pay.groupby("order_id").agg(ptype=("payment_type","first"),nt=("payment_type","nunique"),inst=("payment_installments","max"),val=("payment_value","sum"),npay=("payment_sequential","size"))
o=o.merge(pt,on="order_id",how="left")
o["appr_h"]=(o.order_approved_at-o.order_purchase_timestamp).dt.total_seconds()/3600
print("\nPAYMENT TYPE: n, share, approval hrs mean/median/p90, unapproved%, cancelled%, unavail%")
g=o.groupby("ptype").agg(n=("order_id","size"),appr_mean=("appr_h","mean"),appr_med=("appr_h","median"),appr_p90=("appr_h",lambda s:s.quantile(.9)),unappr=("order_approved_at",lambda s:s.isna().mean()),canc=("order_status",lambda s:(s=="canceled").mean()),unav=("order_status",lambda s:(s=="unavailable").mean()))
print(g.round(3))
bo=o[o.ptype=="boleto"]
print("boleto approval >24h share:",(bo.appr_h>24).mean().round(3),"; >48h",(bo.appr_h>48).mean().round(3),"; cc >24h",(o[o.ptype=="credit_card"].appr_h>24).mean().round(3))
# review response
rv["resp_d"]=(rv.review_answer_timestamp-rv.review_creation_date).dt.total_seconds()/86400
print("\nreview response days median/p90",rv.resp_d.median().round(2),rv.resp_d.quantile(.9).round(2),"; with comment msg",rv.review_comment_message.notna().mean().round(3))
print("score dist",rv.review_score.value_counts(normalize=True).sort_index().round(3).to_dict())
print("msg presence by score",rv.groupby("review_score").review_comment_message.apply(lambda s:s.notna().mean()).round(2).to_dict())
# items
its=it.merge(pr,on="product_id",how="left").merge(se,on="seller_id")
its=its.merge(o[["order_id","order_purchase_timestamp","order_status"]],on="order_id")
gm=its.price.sum(); print("\nGMV",gm.round(0),"items",len(its),"orders w/items",its.order_id.nunique())
# multi seller
ns=its.groupby("order_id").seller_id.nunique(); print("multi-seller orders:",(ns>1).sum(),(ns>1).mean().round(4))
r1=rv.groupby("order_id").review_score.min()
x=pd.DataFrame({"ns":ns}).join(r1); print("bad review rate multi vs single:",(x[x.ns>1].review_score<=2).mean().round(3),(x[x.ns==1].review_score<=2).mean().round(3))
# high value
ov=its.groupby("order_id").price.sum(); x=pd.DataFrame({"v":ov}).join(r1)
q=x.v.quantile(.9); print("top10% order value >",q.round(0),"bad rate",(x[x.v>q].review_score<=2).mean().round(3),"vs rest",(x[x.v<=q].review_score<=2).mean().round(3),"share GMV",(ov[ov>q].sum()/ov.sum()).round(3))
# installments
print("installments dist (cc):",pay[pay.payment_type=="credit_card"].payment_installments.clip(upper=10).value_counts(normalize=True).sort_index().round(3).to_dict())
# same product multi-seller
ps=its.groupby("product_id").seller_id.nunique(); print("\nproducts w/ >1 seller:",(ps>1).sum(),"of",len(ps),"; GMV share",(its[its.product_id.isin(ps[ps>1].index)].price.sum()/gm).round(3))
# listing quality
pm=its.groupby("product_id").agg(n=("order_id","size"),gmv=("price","sum"),cat=("product_category_name","first"),ph=("product_photos_qty","first"),dl=("product_description_lenght","first"),nl=("product_name_lenght","first"))
print("product units median",pm.n.median(),"; products w/1 unit",(pm.n==1).mean().round(3))
print(pm[["n","ph","dl","nl"]].corr(method="spearman").round(3).iloc[0].to_dict())
# sellers
sm=its.groupby("seller_id").agg(n=("order_id","nunique"),gmv=("price","sum"),first=("order_purchase_timestamp","min"),last=("order_purchase_timestamp","max"))
print("\nsellers",len(sm),"; median orders",sm.n.median(),"; sellers <=3 orders",(sm.n<=3).mean().round(3),"; top10% GMV share",(sm.gmv.nlargest(len(sm)//10).sum()/sm.gmv.sum()).round(3))
print("first-order by quarter:",sm["first"].dt.to_period("Q").value_counts().sort_index().to_dict())
# new seller survival: sellers whose first order <= 2018-02-28; success = >=10 orders in first 180d?? compute orders in 90d after first
for dd in [90]:
    f=its.merge(sm[["first"]],left_on="seller_id",right_index=True); f=f[f.order_purchase_timestamp<=f["first"]+pd.Timedelta(days=dd)]
    k=f.groupby("seller_id").order_id.nunique(); old=sm[sm["first"]<="2018-05-01"]; k=k.reindex(old.index)
    print("new sellers (first<=2018-05-01):",len(old),"; orders in first 90d median",k.median(),"; 1-order share",(k<=1).mean().round(3))
# seller dormancy at snapshots
for snap in ["2018-02-26","2018-05-26"]:
    s=pd.Timestamp(snap); a=its[(its.order_purchase_timestamp<s)&(its.order_purchase_timestamp>=s-pd.Timedelta(days=180))]
    act=a.seller_id.unique(); nx=its[(its.order_purchase_timestamp>=s)&(its.order_purchase_timestamp<s+pd.Timedelta(days=90))].seller_id.unique()
    dorm=np.setdiff1d(act,nx); g_=a[a.seller_id.isin(dorm)].price.sum()
    print(snap,"active last180d:",len(act),"dormant next90:",len(dorm),f"{len(dorm)/len(act):.1%}","their trailing GMV share",(g_/a.price.sum()).round(3))
# geography
cs=cu.merge(o[["customer_id","order_id"]],on="customer_id").merge(its.groupby("order_id").price.sum().rename("gmv"),on="order_id")
bs=cs.groupby("customer_state").gmv.agg(["sum","size"]); ss=its.groupby("seller_state").seller_id.nunique().rename("sellers")
bs=bs.join(ss).fillna(0); bs["gmv_per_seller"]=bs["sum"]/bs.sellers.replace(0,np.nan); print("\nstate demand vs sellers\n",bs.sort_values("sum",ascending=False).head(12).round(0))
sh=its.merge(cu.merge(o[["customer_id","order_id"]],on="customer_id")[["order_id","customer_state"]],on="order_id")
print("share of GMV same-state:",(sh[sh.seller_state==sh.customer_state].price.sum()/gm).round(3),"; from SP sellers:",(sh[sh.seller_state=="SP"].price.sum()/gm).round(3))
# category growth
its["m"]=its.order_purchase_timestamp.dt.to_period("Q"); c=its[(its.m>="2017Q1")&(its.m<="2018Q2")].pivot_table(index="product_category_name",columns="m",values="price",aggfunc="sum").fillna(0)
c["g"]=c["2018Q2"]/c["2017Q2"].replace(0,np.nan); print("\ncats with 2017Q2 GMV>50k yoy growth:",c[c["2017Q2"]>50000].g.round(2).sort_values().to_dict())
# freight share
x=its.groupby("order_id").agg(p=("price","sum"),f=("freight_value","sum")).join(r1); x["fr"]=x.f/x.p
print("\nfreight/price quartile bad-rate:",x.groupby(pd.qcut(x.fr,4)).review_score.apply(lambda s:(s<=2).mean()).round(3).tolist())
