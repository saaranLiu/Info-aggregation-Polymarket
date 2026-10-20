"""Within/between decomposition of the information term by order size.

For trades of size class s in market g (n_gs trades),
  mean_gs[D(Y - m)] = Dbar_gs (Y_g - mbar_gs)  -  Cov_gs(D, m)
                      `------ between -----'     `-- within --'
The between term depends on the realised outcome; the within term is a direction-price covariance that is
invariant to Y. Pooled components weight market-size cells by n_gs. Event-cluster SEs use the ratio
estimator over topic-by-resolution-date clusters. Samples: main final-day sample, and the clock common
sample under the scheduled-close cutoff (built in 15_clock_horizon.py; recomputed here from the same rules).

Output: results/wb_F_size.csv
"""
from study_common import *

con = connect()
con.execute(f"""CREATE OR REPLACE TABLE ev AS SELECT condition_id, ts, D::DOUBLE D, p, Y::DOUBLE Y, cat, res_ts::DOUBLE res_ts, tau::DOUBLE tau,
    tau_close, mid_before, mid_before m, ev_cluster, size3, ts_lead_100 FROM '{WORK}/mid_YesNo.parquet'""")
con.execute("""CREATE OR REPLACE TABLE cut AS SELECT condition_id, least(any_value(res_ts), coalesce(any_value(ts + tau_close), any_value(res_ts))) S
    FROM ev GROUP BY 1""")
SAMPLES = {"main": "TRUE", "clock_common_S": "ts + 7200 <= S AND ts_lead_100 IS NOT NULL AND ts_lead_100 <= S"}
rows = []
for sname, sw in SAMPLES.items():
    g = con.sql(f"""SELECT condition_id, any_value(ev_cluster) ev, any_value(cat) cat, size3, count(*) n,
            avg(D) Dbar, any_value(Y) Y, avg(m) mbar, avg(D*(Y-m)) total
        FROM ev JOIN cut USING(condition_id) WHERE {LATE} AND {BASE} AND {sw} GROUP BY condition_id, size3""").df()
    g["between"] = g.Dbar * (g.Y - g.mbar); g["within"] = g.total - g.between
    for cname, cmask in [("pooled", np.ones(len(g), bool)), ("Politics", (g.cat == "Politics").to_numpy()), ("Crypto", (g.cat == "Crypto").to_numpy())]:
        for size in ["all", "s1 <$100", "s2 $100-1k", "s3 >=$1k"]:
            d = g[cmask & ((g.size3 == size).to_numpy() if size != "all" else True)]
            if size == "all":  # recombine size cells within market: components are not additive across cells, recompute
                d = con.sql(f"""SELECT condition_id, any_value(ev_cluster) ev, count(*) n, avg(D) Dbar, any_value(Y) Y, avg(m) mbar, avg(D*(Y-m)) total
                    FROM ev JOIN cut USING(condition_id) WHERE {LATE} AND {BASE} AND {sw}
                    {"" if cname == "pooled" else f"AND cat='{cname}'"} GROUP BY condition_id""").df()
                d["between"] = d.Dbar * (d.Y - d.mbar); d["within"] = d.total - d.between
            N = d.n.sum()
            row = dict(sample=sname, cell=cname, size=size, n=int(N), n_markets=len(d))
            for comp in ["total", "between", "within"]:
                x = d[comp] * d.n; est = x.sum() / N
                s = (x - est * d.n).groupby(d.ev).sum().to_numpy(); G = len(s)
                row[f"{comp}_bp"] = 1e4 * est; row[f"{comp}_se_ev_bp"] = 1e4 * np.sqrt((s ** 2).sum() * G / max(G - 1, 1)) / N
            row["n_ev_clusters"] = G
            rows.append(row)
save(pd.DataFrame(rows), "wb_F_size.csv")
manifest("16_wb_size")
log("done")
