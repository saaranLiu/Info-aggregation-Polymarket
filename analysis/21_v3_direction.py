"""Direction audit and an alternative tie-break for same-timestamp trades.

The on-chain direction D is the reference-axis aggressor sign. The tick rule classifies a trade by the
sign of its price change relative to the previous trade in the same market; a zero price change inherits
the most recent nonzero tick, and a market's opening trades stay unclassified. Two experiments use the
same trades:

  step 1  replace D, keep the on-chain midpoint;
  step 2  also rebuild the midpoint from tick-rule direction switches.

A separate check rebuilds the on-chain midpoint after ordering same-timestamp trades by taker address
instead of by hash(taker, p, usd).

Outputs: dir_A_agreement.csv, dir_B_metrics.csv, dir_C_order.csv
"""
from study_common import *
import math

con = connect(threads=32, mem="120GB")
log("load")
con.execute(f"""CREATE OR REPLACE TABLE ev AS
    SELECT condition_id, rn, ts, taker, D, p, usd, Y, cat, size3, ev_cluster, tau, mid_before, mid_after_100, ts_lead_100, p_lag_1
    FROM '{WORK}/mid_YesNo.parquet'""")

# tick rule. IGNORE NULLS carries the last nonzero price change forward.
con.execute("""CREATE OR REPLACE TABLE ev2 AS
    SELECT *, last_value(tick IGNORE NULLS) OVER (PARTITION BY condition_id ORDER BY rn
            ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW) AS D_tick
    FROM (SELECT *, CASE WHEN p_lag_1 IS NULL THEN NULL
                         WHEN p > p_lag_1 + 1e-9 THEN 1 WHEN p < p_lag_1 - 1e-9 THEN -1 ELSE NULL END AS tick
          FROM ev)""")
con.execute("DROP TABLE ev; ALTER TABLE ev2 RENAME TO ev")
con.execute("ALTER TABLE ev ADD COLUMN zero_tick BOOLEAN; UPDATE ev SET zero_tick = (tick IS NULL AND p_lag_1 IS NOT NULL)")
log("tick rule assigned")

H = f"{LATE} AND {BASE} AND ts_lead_100 IS NOT NULL AND mid_after_100 IS NOT NULL"
rows = []
for cell, w in [("pooled", "TRUE"), ("Politics | s1 <$100", "cat='Politics' AND size3='s1 <$100'"),
                ("Politics | s3 >=$1k", "cat='Politics' AND size3='s3 >=$1k'"),
                ("Crypto | s1 <$100", "cat='Crypto' AND size3='s1 <$100'"),
                ("Crypto | s3 >=$1k", "cat='Crypto' AND size3='s3 >=$1k'")]:
    r = con.sql(f"""SELECT count(*) n, avg((D_tick IS NOT NULL)::int) share_classified,
            avg(zero_tick::int) share_zero_tick,
            avg((D_tick = D)::int) FILTER (WHERE D_tick IS NOT NULL) share_agree,
            sum(usd*(D_tick = D)::int) FILTER (WHERE D_tick IS NOT NULL) / sum(usd) FILTER (WHERE D_tick IS NOT NULL) share_agree_usd
        FROM ev WHERE {H} AND {w}""").fetchone()
    rows.append(dict(cell=cell, n=r[0], share_classified=r[1], share_zero_tick=r[2], share_agree=r[3], share_agree_usd=r[4]))
    log(f"agree {cell}: {r[3]:.3f} on {r[1]:.3f} classified")
save(pd.DataFrame(rows), "dir_A_agreement.csv")

def metrics(where, m_pre, m_post, tag):
    out = []
    for cell, w in [("pooled", "TRUE"), ("Politics | s1 <$100", "cat='Politics' AND size3='s1 <$100'"),
                    ("Politics | s3 >=$1k", "cat='Politics' AND size3='s3 >=$1k'"),
                    ("Crypto | s1 <$100", "cat='Crypto' AND size3='s1 <$100'")]:
        d = con.sql(f"""SELECT ev_cluster,
                1e4*D*(Y-{m_pre}) info_on, 1e4*D_tick*(Y-{m_pre}) info_tk,
                1e4*D*(p-Y) rs_on, 1e4*D_tick*(p-Y) rs_tk,
                1e4*D*({m_post}-Y) over_on, 1e4*D_tick*({m_post}-Y) over_tk
            FROM ev WHERE {where} AND {w} AND D_tick IS NOT NULL AND {m_pre} IS NOT NULL AND {m_post} IS NOT NULL""").df()
        for k in ["info", "rs", "over"]:
            diff = (d[f"{k}_tk"] - d[f"{k}_on"]).to_numpy(float)
            se, G = cr_se_mean(diff, d.ev_cluster.to_numpy())
            out.append(dict(step=tag, cell=cell, metric=k, n=len(d), n_clusters=G,
                            onchain_bp=float(d[f"{k}_on"].mean()), tick_bp=float(d[f"{k}_tk"].mean()),
                            diff_bp=float(diff.mean()), se_ev_bp=se))
        log(f"{tag} {cell}: n={len(d):,} info diff {out[-3]['diff_bp']:.1f}")
    return out

rows = metrics(H, "mid_before", "mid_after_100", "fixed midpoint")

# step 2: midpoint from tick-rule switches, same availability rule as 00_build.py
log("rebuild tick midpoint")
con.execute("""CREATE OR REPLACE TABLE mids AS
    SELECT a.condition_id, a.rn + 1 AS rn_mid, (a.p + b.p) / 2.0 AS mid
    FROM ev a JOIN ev b ON a.condition_id = b.condition_id AND b.rn = a.rn + 1
    WHERE a.D_tick IS NOT NULL AND b.D_tick IS NOT NULL AND a.D_tick <> b.D_tick""")
con.execute("""CREATE OR REPLACE TABLE ev2 AS SELECT e.*, x.mid AS mid_tick
    FROM ev e ASOF LEFT JOIN mids x ON x.condition_id = e.condition_id AND x.rn_mid <= e.rn - 1""")
con.execute("DROP TABLE ev; ALTER TABLE ev2 RENAME TO ev")
con.execute("""CREATE OR REPLACE TABLE ev2 AS SELECT e.*, x.mid AS mid_tick_100
    FROM ev e ASOF LEFT JOIN mids x ON x.condition_id = e.condition_id AND x.rn_mid <= e.rn + 100""")
con.execute("DROP TABLE ev; ALTER TABLE ev2 RENAME TO ev")
rows += metrics(H, "mid_tick", "mid_tick_100", "rebuilt midpoint")
save(pd.DataFrame(rows), "dir_B_metrics.csv")

# tie-break: order same-timestamp trades by taker address instead of the stored hash order
log("alternative within-timestamp order")
con.execute(f"""CREATE OR REPLACE TABLE alt AS
    SELECT condition_id, ts, taker, D, p, Y, usd, cat, size3, ev_cluster, tau::DOUBLE tau, mid_before, mid_after_100,
           row_number() OVER (PARTITION BY condition_id ORDER BY ts, taker, D) AS rn
    FROM '{WORK}/mid_YesNo.parquet'""")
con.execute("""CREATE OR REPLACE TABLE mids AS
    SELECT a.condition_id, a.rn + 1 AS rn_mid, (a.p + b.p) / 2.0 AS mid
    FROM alt a JOIN alt b ON a.condition_id = b.condition_id AND b.rn = a.rn + 1
    WHERE a.D <> b.D""")
con.execute("""CREATE OR REPLACE TABLE alt2 AS SELECT e.*, x.mid AS mid_alt
    FROM alt e ASOF LEFT JOIN mids x ON x.condition_id = e.condition_id AND x.rn_mid <= e.rn - 1""")
con.execute("DROP TABLE alt; ALTER TABLE alt2 RENAME TO alt")
d = con.sql(f"""SELECT ev_cluster, cat, 1e4*D*(Y-mid_before) info_on, 1e4*D*(Y-mid_alt) info_alt
    FROM alt WHERE {LATE} AND abs(p-0.5) < 0.3 AND mid_before IS NOT NULL AND mid_alt IS NOT NULL
      AND cat IN ('Politics','Crypto')""").df()
out = []
for cell, w in [("Politics+Crypto", np.ones(len(d), bool)), ("Politics", (d.cat == "Politics").to_numpy()),
                ("Crypto", (d.cat == "Crypto").to_numpy())]:
    s = d[w]; diff = (s.info_alt - s.info_on).to_numpy(float)
    se, G = cr_se_mean(diff, s.ev_cluster.to_numpy())
    out.append(dict(cell=cell, n=len(s), n_clusters=G, hash_order_bp=float(s.info_on.mean()),
                    taker_order_bp=float(s.info_alt.mean()), diff_bp=float(diff.mean()), se_ev_bp=se))
    log(f"order {cell}: diff {out[-1]['diff_bp']:.2f} ({se:.2f})")
save(pd.DataFrame(out), "dir_C_order.csv")
log("done")
