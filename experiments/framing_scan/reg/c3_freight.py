import sys,os; sys.path.insert(0,os.path.dirname(os.path.abspath(__file__)))
from common import *
from sklearn.ensemble import HistGradientBoostingRegressor as H
o,it,pr,se,cu=load()
g=pd.read_csv(f"{D}/olist_geolocation_dataset.csv").groupby("geolocation_zip_code_prefix")[["geolocation_lat","geolocation_lng"]].median()
it=it.merge(pr[["product_id","cat","product_weight_g","product_length_cm","product_height_cm","product_width_cm"]],on="product_id",how="left")
it["vol"]=it.product_length_cm*it.product_height_cm*it.product_width_cm
a=it.groupby("order_id").agg(n=("order_item_id","size"),price=("price","sum"),freight=("freight_value","sum"),w=("product_weight_g","sum"),vol=("vol","sum"),seller_id=("seller_id","first"),ns=("seller_id","nunique"),cat=("cat","first"))
d=o.merge(a,on="order_id").merge(se,on="seller_id").merge(cu,on="customer_id")
d=d[(d.ns==1)&d.w.notna()&(d.w>0)&(d.freight>0)].copy()
for p,c in [("seller_zip_code_prefix","s"),("customer_zip_code_prefix","c")]:
    d=d.join(g.rename(columns={"geolocation_lat":c+"lat","geolocation_lng":c+"lng"}),on=p)
d=d.dropna(subset=["slat","clat"])
R=6371;la1,la2=np.radians(d.slat),np.radians(d.clat);dl=np.radians(d.clng-d.slng)
d["dist"]=R*np.arccos(np.clip(np.sin(la1)*np.sin(la2)+np.cos(la1)*np.cos(la2)*np.cos(dl),-1,1))
d["route"]=d.seller_state+">"+d.customer_state; d["same_state"]=(d.seller_state==d.customer_state).astype(int)
d["volw"]=d.vol/6000*1000  # volumetric weight in g (cm3/6000 kg)
d["bw"]=np.maximum(d.w,d.volw)
d["fpk"]=d.freight/(d.bw/1000)
for c in ["seller_state","customer_state","cat"]: d[c+"_c"]=d[c].astype("category").cat.codes
d=d.sort_values("order_purchase_timestamp");print(len(d),"freight mean",d.freight.mean(),"median",d.freight.median())
tr=d[d.order_purchase_timestamp<"2018-03-01"];va=d[(d.order_purchase_timestamp>="2018-03-01")&(d.order_purchase_timestamp<BOUND)];tr2=d[d.order_purchase_timestamp<BOUND];te=d[d.order_purchase_timestamp>=BOUND]
def mets(x,p): return dict(MAE=np.abs(x.freight-p).mean(),MAPE_med=np.median(np.abs(x.freight-p)/x.freight),WAPE=wape(x.freight,p),within20=(np.abs(x.freight-p)/x.freight<.2).mean())
def b_route(fit,x):   # per-route median freight/kg(billable) * billable kg
    m=fit.groupby("route").fpk.median();gm=fit.fpk.median(); return (x.route.map(m).fillna(gm)*x.bw/1000).values
def b_route_lin(fit,x): # per route median of freight, scaled? : route+weight-bucket median
    fit=fit.assign(wb=pd.qcut(fit.bw,12,duplicates="drop",labels=False));edges=pd.qcut(fit.bw,12,duplicates="drop",retbins=True)[1]
    xb=np.clip(np.searchsorted(edges[1:-1],x.bw),0,len(edges)-2);m=fit.groupby(["route","wb"]).freight.median();mw=fit.groupby("wb").freight.median()
    return np.array([m.get((r,b),mw.get(b,fit.freight.median())) for r,b in zip(x.route,xb)])
F=["w","vol","bw","dist","price","n","same_state","slat","slng","clat","clng","seller_state_c","customer_state_c","cat_c"]
res={}
for nm,fit,x in [("val",tr,va),("test",tr2,te)]:
    res[(nm,"route median/kg")]=mets(x,b_route(fit,x)); res[(nm,"route x weight-bucket median")]=mets(x,b_route_lin(fit,x))
cf=[dict(learning_rate=.05,max_iter=300,max_leaf_nodes=31,min_samples_leaf=30),dict(learning_rate=.05,max_iter=500,max_leaf_nodes=15,min_samples_leaf=20)]
best=None
for c in cf:
    m=H(loss="absolute_error",random_state=0,**c).fit(tr[F],tr.freight);mae=np.abs(va.freight-m.predict(va[F])).mean();print(c,mae)
    if best is None or mae<best[0]:best=(mae,c)
c=best[1];m=H(loss="absolute_error",random_state=0,**c).fit(tr[F],tr.freight);res[("val","GBM")]=mets(va,m.predict(va[F]))
m2=H(loss="absolute_error",random_state=0,**c).fit(tr2[F],tr2.freight);pt=m2.predict(te[F]);res[("test","GBM")]=mets(te,pt)
print(pd.DataFrame(res).T.round(3))
# anomaly view: residual vs prediction band; time stability of price per kg
d["ym"]=d.order_purchase_timestamp.dt.to_period("M");print(d.groupby("ym").fpk.median().round(1).to_string())
te=te.assign(p=pt,r=te.freight-pt);print("test resid >50% over pred:",(te.r>.5*te.p).mean(),"under:",(te.r<-.5*te.p).mean())
print("noise floor: dup same route+weight bucket spread (IQR/median) ->",(d.groupby(["route",pd.qcut(d.bw,12,duplicates='drop')]).freight.apply(lambda s:(s.quantile(.75)-s.quantile(.25))/s.median())).median())
# extra strong simple baseline: log-linear OLS (log bw, log dist, route dummies)
from sklearn.linear_model import Ridge
def lin(fit,x):
    def X(z):
        M=pd.DataFrame({"lb":np.log(z.bw+1),"ld":np.log(z.dist+10),"lbd":np.log(z.bw+1)*np.log(z.dist+10),"lb2":np.log(z.bw+1)**2,"n":z.n,"ss":z.same_state})
        return pd.concat([M.reset_index(drop=True),pd.get_dummies(z.seller_state_c,prefix="s").reset_index(drop=True),pd.get_dummies(z.customer_state_c,prefix="c").reset_index(drop=True)],axis=1)
    xf=X(fit);xx=X(x).reindex(columns=xf.columns,fill_value=0).astype(float)
    m=Ridge(alpha=1).fit(xf.astype(float),np.log(fit.freight.values));return np.exp(m.predict(xx))
for nm,fit,x in [("val",tr,va),("test",tr2,te)]: print(nm,"loglinear",{k:round(v,3) for k,v in mets(x,lin(fit,x)).items()})
