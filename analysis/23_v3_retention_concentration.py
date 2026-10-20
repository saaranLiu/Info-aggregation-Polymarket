"""Horizon retention and the concentration of the settlement margin.

Retention compares, by topic and size, trades whose 100-trade horizon is observed with trades in the same
main window whose horizon is not. The settlement margin needs no future midpoint, so it is comparable
across the two groups. Concentration asks whether the per-dollar margin is carried by a few event clusters:
the mean is recomputed after dropping the clusters with the largest absolute contribution.

Outputs: ret_A_retention.csv, con_A_concentration.csv, con_B_side.csv
"""
from study_common import *

con = connect(threads=32, mem="80GB")
con.execute(f"""CREATE OR REPLACE TABLE ev AS SELECT cat, size3, ev_cluster, D, p, Y, usd, ts, ts_lead_100, mid_after_100,
        (mid_after_100 IS NOT NULL AND ts_lead_100 IS NOT NULL) AS kept, D*(p-Y) AS rs
    FROM '{WORK}/mid_YesNo.parquet' WHERE {LATE} AND {BASE} AND cat IN ('Politics','Crypto')""")
rows = []
for cat in ["Politics", "Crypto"]:
    for size in ["s1 <$100", "s2 $100-1k", "s3 >=$1k"]:
        d = con.sql(f"""SELECT ev_cluster, kept, 1e4*rs AS rs, (ts_lead_100 - ts)/3600.0 AS hours
            FROM ev WHERE cat='{cat}' AND size3='{size}'""").df()
        k, dr = d[d.kept], d[~d.kept]
        def se(s):
            return cr_se_mean(s.rs.to_numpy(float), s.ev_cluster.to_numpy())[0] if len(s) else np.nan
        rows.append(dict(cat=cat, size3=size, n_main=len(d), n_kept=len(k), share_kept=len(k) / len(d),
                         med_hours=float(k.hours.median()) if len(k) else np.nan,
                         p90_hours=float(k.hours.quantile(0.9)) if len(k) else np.nan,
                         rs_kept_bp=float(k.rs.mean()) if len(k) else np.nan, rs_kept_se=se(k),
                         rs_dropped_bp=float(dr.rs.mean()) if len(dr) else np.nan, rs_dropped_se=se(dr)))
        log(f"{cat} {size}: kept {rows[-1]['share_kept']:.3f}, RS {rows[-1]['rs_kept_bp']:.0f} vs {rows[-1]['rs_dropped_bp']:.0f}")
save(pd.DataFrame(rows), "ret_A_retention.csv")

# concentration of the settlement margin on the observed horizon
con.execute(f"""CREATE OR REPLACE TABLE h AS SELECT cat, ev_cluster, D, usd, D*(p-Y) AS rs
    FROM '{WORK}/mid_YesNo.parquet' WHERE {LATE} AND {BASE} AND mid_after_100 IS NOT NULL AND ts_lead_100 IS NOT NULL""")
g = con.sql("""SELECT coalesce(cat, 'Yes/No') cat, ev_cluster, count(*) n, sum(rs) s_trade, sum(usd) u, sum(usd*rs) s_usd
    FROM h GROUP BY GROUPING SETS ((cat, ev_cluster), (ev_cluster))""").df()
out = []
for cat in ["Yes/No", "Politics", "Crypto"]:
    s = g[g.cat == cat]
    for wname, col, den in [("trade", "s_trade", "n"), ("usd", "s_usd", "u")]:
        tot_w, tot_x = s[den].sum(), s[col].sum()
        c = s.assign(contrib=s[col] / tot_w)
        c = c.reindex(c.contrib.abs().sort_values(ascending=False).index)
        def wstat(z):
            W = z[den].sum(); mu = z[col].sum() / W
            r = (z[col] - mu * z[den]).to_numpy(); G = len(z)
            return 1e4 * mu, 1e4 * np.sqrt((r ** 2).sum()) / W * np.sqrt(G / (G - 1))
        full, full_se = wstat(c); d1, _ = wstat(c.iloc[1:]); d5, d5_se = wstat(c.iloc[5:])
        out.append(dict(cat=cat, weight=wname, n_clusters=len(c), mean_bp=full, mean_se_bp=full_se,
                        drop1_bp=d1, drop5_bp=d5, drop5_se_bp=d5_se,
                        top1_abs_share=float(c.contrib.abs().iloc[0] / c.contrib.abs().sum()),
                        top5_abs_share=float(c.contrib.abs().iloc[:5].sum() / c.contrib.abs().sum())))
        log(f"{cat} {wname}: {out[-1]['mean_bp']:.1f} -> drop5 {out[-1]['drop5_bp']:.1f}")
save(pd.DataFrame(out), "con_A_concentration.csv")

side = con.sql("""SELECT coalesce(cat,'Yes/No') cat, CASE WHEN D = 1 THEN 'buy' ELSE 'sell' END side,
        count(*) n, 1e4*avg(rs) rs_bp FROM h GROUP BY GROUPING SETS ((cat, D), (D))""").df()
save(side, "con_B_side.csv")
top = g[g.cat == "Politics"].copy()
top["contrib_bp"] = 1e4 * top.s_usd / top.u.sum()
top = top.reindex(top.contrib_bp.abs().sort_values(ascending=False).index).head(8)
top["usd_share"] = top.u / top.u.sum()
save(top[["ev_cluster", "n", "usd_share", "contrib_bp"]], "con_C_politics_dates.csv")
log("done")
