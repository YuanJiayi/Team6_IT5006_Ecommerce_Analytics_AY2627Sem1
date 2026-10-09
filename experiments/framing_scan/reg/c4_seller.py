import sys,os; sys.path.insert(0,os.path.dirname(os.path.abspath(__file__)))
from common import *
from scipy.stats import spearmanr
from sklearn.ensemble import HistGradientBoostingRegressor as H
o,it,pr,se,cu=load(); rv=pd.read_csv(f"{D}/olist_order_reviews_dataset.csv")
o=o[o.order_status!="canceled"]
it=it.merge(o[["order_id","order_purchase_timestamp","order_delivered_customer_date","order_delivered_carrier_date"]],on="order_id").merge(pr[["product_id","cat"]],on="product_id",how="left")
it["t"]=it.order_purchase_timestamp; rs=rv.groupby("order_id").review_score.min(); it["rev"]=it.order_id.map(rs)
it["cat"]=it.cat.fillna("unk"); it=it.merge(se[["seller_id","seller_state"]],on="seller_id")
S=it.seller_id.unique()
def snap(s_date,with_y=True):
    s=pd.Timestamp(s_date); h=it[it.t<s]
    def win(d0,d1):
        x=h[(h.t>=s-pd.Timedelta(days=d1))&(h.t<s-pd.Timedelta(days=d0))]
        return x.groupby("seller_id").price.sum(),x.groupby("seller_id").order_id.nunique()
    g30,n30=win(0,30);g90,n90=win(0,90);g180,_=win(0,180);gp,_=win(90,180)
    first=h.groupby("seller_id").t.min();last=h.groupby("seller_id").t.max()
    df=pd.DataFrame({"g30":g30,"g90":g90,"g180":g180,"gprev90":gp,"n30":n30,"n90":n90}).reindex(first.index).fillna(0)
    df["tenure"]=(s-first).dt.days;df["recency"]=(s-last).dt.days
    # reviews/late only for orders delivered before s
    dl=h[h.order_delivered_customer_date<s]; df["rev_mean"]=dl.groupby("seller_id").rev.mean(); df["late_share"]=dl.assign(l=(dl.order_delivered_customer_date>dl.order_delivered_carrier_date+pd.Timedelta(days=15)).astype(int)).groupby("seller_id").l.mean()
    df["top_cat"]=h.groupby("seller_id").cat.agg(lambda x:x.value_counts().index[0]).reindex(df.index).astype("category").cat.codes
    df["state"]=pd.Series(se.set_index("seller_id").seller_state).reindex(df.index).astype("category").cat.codes
    df["n_cat"]=h[h.t>=s-pd.Timedelta(days=180)].groupby("seller_id").cat.nunique().reindex(df.index).fillna(0)
    df["aov"]=df.g90/df.n90.replace(0,np.nan)
    df["trend"]=(df.g90-df.gprev90)/(df.g90+df.gprev90+1)
    df=df[(df.g180>0)]   # active in last 180d
    if with_y:
        f=it[(it.t>=s)&(it.t<s+pd.Timedelta(days=90))].groupby("seller_id").price.sum(); df["y"]=f.reindex(df.index).fillna(0)
    df["snap"]=s; return df
F=["g30","g90","g180","gprev90","n30","n90","tenure","recency","rev_mean","late_share","top_cat","state","n_cat","aov","trend"]
def build(dates): return pd.concat([snap(d) for d in dates])
def ev(x,p,nm):
    y=x.y.values; k=max(1,int(.1*len(x))); top=np.argsort(-p)[:k]
    return dict(WAPE=wape(y,p),spearman=spearmanr(y,p)[0],top10_capture=y[top].sum()/y.sum(),oracle_top10=np.sort(y)[::-1][:k].sum()/y.sum(),zero_share=(y==0).mean(),n=len(x))
tr_dates_val=pd.date_range("2017-03-01","2017-09-01",freq="14D"); val_dates=pd.date_range("2017-12-01","2018-02-23",freq="14D")
tr_dates_te=pd.date_range("2017-03-01","2018-02-23",freq="14D"); te_dates=[pd.Timestamp("2018-05-26")]
Tv=build(tr_dates_val);V=build(val_dates);T2=build(tr_dates_te);Te=build(te_dates)
res={}
cfgs=[dict(learning_rate=.05,max_iter=150,max_depth=3,min_samples_leaf=40),dict(learning_rate=.03,max_iter=200,max_leaf_nodes=8,min_samples_leaf=80,l2_regularization=3)]
def fitpred(tr,te,loss,c):
    # model target ratio to g90+50 scale to reduce skew
    sc=lambda x:x.g90+x.g180/2+50
    m=H(loss=loss,random_state=0,**c).fit(tr[F],tr.y/sc(tr)); return np.clip(m.predict(te[F]),0,None)*sc(te)
for nm,tr,te in [("val",Tv,V),("test",T2,Te)]:
    res[(nm,"last90")]=ev(te,te.g90.values,"")
    res[(nm,"last180/2")]=ev(te,(te.g180/2).values,"")
    res[(nm,"0.5*g90+0.5*g180/2")]=ev(te,(0.5*te.g90+0.25*te.g180).values,"")
bestc=None
for loss in ["absolute_error","squared_error"]:
  for i,c in enumerate(cfgs):
    p=fitpred(Tv,V,loss,c);r=ev(V,p,"");print(loss,i,r)
    if bestc is None or r["WAPE"]<bestc[0]: bestc=(r["WAPE"],loss,c)
_,loss,c=bestc;print("chosen",loss,c)
res[("val","GBM")]=ev(V,fitpred(Tv,V,loss,c),"");res[("test","GBM")]=ev(Te,fitpred(T2,Te,loss,c),"")
print(pd.DataFrame(res).T.round(3).to_string())
