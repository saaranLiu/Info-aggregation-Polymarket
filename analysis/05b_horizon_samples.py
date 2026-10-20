"""main-sample vs horizon-complete estimates with market- and event-cluster SEs (feeds Table 2 of the paper)."""
from study_common import *
import pandas as pd, numpy as np
con = connect(threads=32, mem="200GB")
rows=[]
for grp in ["YesNo","SportsML","OverUnder"]:
    con.execute(f"CREATE OR REPLACE VIEW e AS SELECT * FROM read_parquet('{WORK}/mid_{grp}.parquet')")
    cells=[("pooled","TRUE")]
    if grp=="YesNo": cells += [(c,f"cat='{c}'") for c in ["Politics","Crypto","Sports","Finance"]] + [(f"{c} | {s}", f"cat='{c}' AND size3='{s}'") for c in ["Politics","Crypto"] for s in ["s1 <$100","s2 $100-1k","s3 >=$1k"]]
    for cname,cw in cells:
        for sname,sw in [("all","TRUE"),("horizon","ts_lead_100 IS NOT NULL")]:
            for m,ex in METS.items():
                # cluster sums for market and event
                q=f"""WITH t AS (SELECT condition_id, ev_cluster, {ex} x FROM e WHERE {BASE} AND {LATE} AND {cw} AND {sw}),
                 s AS (SELECT avg(x) mu, count(*) n FROM t),
                 cm AS (SELECT sum(x-(SELECT mu FROM s)) d FROM t GROUP BY condition_id),
                 ce AS (SELECT sum(x-(SELECT mu FROM s)) d FROM t GROUP BY ev_cluster)
                 SELECT (SELECT mu FROM s), (SELECT n FROM s), (SELECT sqrt(sum(d*d))/ (SELECT n FROM s) * sqrt(count(*)/(count(*)-1.0)) FROM cm), (SELECT sqrt(sum(d*d))/(SELECT n FROM s)*sqrt(count(*)/(count(*)-1.0)) FROM ce), (SELECT count(*) FROM ce)"""
                mu,n,sm,se,G=con.execute(q).fetchone()
                rows.append(dict(grp=grp,cell=cname,sample=sname,metric=m,est_bp=mu*1e4,n=n,se_mkt=sm*1e4,se_ev=se*1e4,G_ev=G))
d=pd.DataFrame(rows); d.to_csv(f"{RES}/inf_A_horizon_samples.csv",index=False)
pd.set_option('display.width',250)
d["fmt"]=d.apply(lambda r:f"{r.est_bp:.0f} ({r.se_mkt:.0f}/{r.se_ev:.0f})",axis=1)
print(d.pivot_table(index=["grp","cell","sample"],columns="metric",values="fmt",aggfunc="first")[["eff","info","RS","PI_100","over_100"]].to_string())
print(d[d.metric=="info"].pivot_table(index=["grp","cell"],columns="sample",values="n"))
