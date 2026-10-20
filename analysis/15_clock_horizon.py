"""Pricing residuals at fixed clock horizons, on common samples, under explicit information cutoffs.

For each final-day interior-price Yes/No trade with a defined proxy, the post-trade midpoint at clock
horizon H is the latest switch-pair midpoint completed at or before ts + H. A horizon is admissible
under cutoff K only if ts + H <= K (and ts < K). Cutoffs:
  R   on-chain resolution
  S   scheduled close (metadata close_at), and on-chain resolution if earlier
  S1  min(scheduled close, resolution) - 1 h
  C   price-convergence time: the last trade whose winner-axis price is below 0.95 (all later trades
      price the eventual winner at >= 0.95); combined with S
The common sample for a cutoff keeps trades for which every clock horizon (10 min, 30 min, 1 h, 2 h)
and the 100-trade horizon are admissible, so all horizons are estimated on the same trades.

Outputs (results/):
  clk_A_cells.csv        over/PI/info/RS/eff by cutoff x horizon x cell (pooled, topic, topic x size),
                         event- and market-cluster SEs, on the common sample of each cutoff
  clk_B_sample.csv       sample sizes and composition by cutoff
  clk_C_forward.csv      over_H by resolution period (discovery vs forward periods) for Politics/Crypto small
  clk_D_forward_pred.csv expanding-window forward prediction: price-only logit vs price + D x size x topic,
                         trained on earlier resolution quarters, evaluated on the next quarter
  clk_E_diff.csv         Politics-minus-Crypto small-order residual differences (independent clusters)
"""
from study_common import *
import math

HS = [600, 1800, 3600, 7200]
con = connect()
log("load")
con.execute(f"""CREATE OR REPLACE TABLE ev AS SELECT condition_id, rn, ts, D, p, usd, Y, cat, res_ts::DOUBLE res_ts, tau::DOUBLE tau,
    tau_close, mid_before, ev_cluster, size3, ts_lead_100, mid_after_100, rn_mid_after_100, kind, stale
    FROM '{WORK}/mid_YesNo.parquet'""")
con.execute("""CREATE OR REPLACE TABLE mids AS SELECT condition_id, ts_mid, arg_max(rn_mid, rn_mid) AS rn_mid, arg_max(mid, rn_mid) AS mid FROM (
    SELECT a.condition_id, b.ts AS ts_mid, b.rn AS rn_mid, (a.p + b.p)/2.0 AS mid
    FROM ev a JOIN ev b ON a.condition_id = b.condition_id AND b.rn = a.rn + 1 WHERE a.kind = 'switch') GROUP BY 1, 2""")
# market-level cutoffs
con.execute("""CREATE OR REPLACE TABLE cut AS
    SELECT condition_id, any_value(res_ts) R,
           least(any_value(res_ts), coalesce(any_value(ts + tau_close), any_value(res_ts))) S,
           max(ts) FILTER (WHERE (Y*p + (1-Y)*(1-p)) < 0.95) Cq
    FROM ev GROUP BY 1""")
con.execute("UPDATE cut SET Cq = coalesce(Cq, R)")
log("clock midpoints")
con.execute(f"CREATE OR REPLACE TABLE m AS SELECT * FROM ev WHERE {LATE} AND {BASE}")
for H in HS:
    con.execute(f"""CREATE OR REPLACE TABLE m2 AS SELECT e.*, x.mid AS mid_c{H} FROM m e
        ASOF LEFT JOIN mids x ON x.condition_id = e.condition_id AND x.ts_mid <= e.ts + {H}""")
    con.execute("DROP TABLE m; ALTER TABLE m2 RENAME TO m")
con.execute("""CREATE OR REPLACE TABLE m2 AS SELECT m.*, c.R, c.S, least(c.S, c.R) - 3600 AS S1, least(c.S, c.Cq) AS C
    FROM m JOIN cut c USING(condition_id)""")
con.execute("DROP TABLE m; ALTER TABLE m2 RENAME TO m")
CUTS = {"R": "R", "S": "S", "S1": "S1", "C": "C"}
def common(K):
    return f"ts + 7200 <= {K} AND ts_lead_100 IS NOT NULL AND ts_lead_100 <= {K}"

METRICS = {"info": "D*(Y-mid_before)", "eff": "D*(p-mid_before)", "RS": "D*(p-Y)",
           "over_100": "D*(mid_after_100-Y)", "PI_100": "D*(mid_after_100-mid_before)"}
for H in HS:
    METRICS[f"over_c{H}"] = f"D*(mid_c{H}-Y)"; METRICS[f"PI_c{H}"] = f"D*(mid_c{H}-mid_before)"

CELLS = [("pooled", "TRUE"), ("Politics", "cat='Politics'"), ("Crypto", "cat='Crypto'")] + \
        [(f"{c} | {s}", f"cat='{c}' AND size3='{s}'") for c in ["Politics", "Crypto"] for s in ["s1 <$100", "s2 $100-1k", "s3 >=$1k"]]

def cluster_stats(where, exprs):
    """mean, market-cluster SE, event-cluster SE for several expressions on one sample (one pass)."""
    sel = ", ".join(f"{e} AS x{i}" for i, e in enumerate(exprs))
    d = con.sql(f"SELECT condition_id, ev_cluster, {sel} FROM m WHERE {where}").df()
    out = {}
    for i, name in enumerate(exprs):
        x = d[f"x{i}"].to_numpy(float)
        mu = x.mean(); se_m, Gm = cr_se_mean(x, d.condition_id.to_numpy()); se_e, Ge = cr_se_mean(x, d.ev_cluster.to_numpy())
        out[name] = (mu, se_m, se_e, Gm, Ge, len(x))
    return out

rows, srows = [], []
for kname, K in CUTS.items():
    W = common(K)
    log(f"cutoff {kname}")
    for cname, cw in CELLS:
        st = cluster_stats(f"{W} AND {cw}", list(METRICS.values()))
        for mname, e in METRICS.items():
            mu, sm, se, Gm, Ge, n = st[e]
            rows.append(dict(cutoff=kname, cell=cname, metric=mname, est_bp=1e4 * mu, se_mkt_bp=1e4 * sm, se_ev_bp=1e4 * se,
                             n=n, n_markets=Gm, n_ev_clusters=Ge, p=math.erfc(abs(mu / se) / math.sqrt(2)) if se > 0 else np.nan))
    s = con.sql(f"""SELECT '{kname}' cutoff, cat, count(*) n_main, sum(({W})::int) n_common, count(distinct condition_id) m_main,
            count(distinct condition_id) FILTER (WHERE {W}) m_common,
            avg((ts > ts + tau_close)::int) share_after_close_main
        FROM m WHERE cat IN ('Politics','Crypto') GROUP BY 1,2""").df()
    srows.append(s)
A = pd.DataFrame(rows); save(A, "clk_A_cells.csv"); save(pd.concat(srows), "clk_B_sample.csv")

# ---- E: Politics minus Crypto, small orders, per cutoff and horizon (independent topic-by-day clusters)
er = []
for kname in CUTS:
    for mname in ["over_100"] + [f"over_c{H}" for H in HS]:
        a = A[(A.cutoff == kname) & (A.cell == "Politics | s1 <$100") & (A.metric == mname)].iloc[0]
        b = A[(A.cutoff == kname) & (A.cell == "Crypto | s1 <$100") & (A.metric == mname)].iloc[0]
        d = a.est_bp - b.est_bp; s = math.hypot(a.se_ev_bp, b.se_ev_bp)
        er.append(dict(cutoff=kname, metric=mname, politics_bp=a.est_bp, crypto_bp=b.est_bp, diff_bp=d, se_bp=s, p=math.erfc(abs(d / s) / math.sqrt(2))))
save(pd.DataFrame(er), "clk_E_diff.csv")

# ---- C: forward periods.  Discovery = markets resolving before 2025-07-01; forward = later, split in halves of the remainder
log("forward periods")
PER = """CASE WHEN res_ts < epoch(TIMESTAMP '2025-01-01') THEN 'P1 <2025'
              WHEN res_ts < epoch(TIMESTAMP '2025-07-01') THEN 'P2 2025H1'
              WHEN res_ts < epoch(TIMESTAMP '2026-01-01') THEN 'P3 2025H2'
              ELSE 'P4 2026' END"""
cr = []
for kname in ["R", "S", "C"]:
    W = common(CUTS[kname])
    for cname, cw in [c for c in CELLS if c[0] in ("Politics | s1 <$100", "Crypto | s1 <$100", "Politics", "Crypto", "Politics | s3 >=$1k")]:
        for per in ["P1 <2025", "P2 2025H1", "P3 2025H2", "P4 2026"]:
            st = cluster_stats(f"{W} AND {cw} AND ({PER}) = '{per}'", ["D*(mid_c3600-Y)", "D*(mid_after_100-Y)", "D*(Y-mid_before)"])
            for mname, e in [("over_c3600", "D*(mid_c3600-Y)"), ("over_100", "D*(mid_after_100-Y)"), ("info", "D*(Y-mid_before)")]:
                mu, sm, se, Gm, Ge, n = st[e]
                cr.append(dict(cutoff=kname, cell=cname, period=per, metric=mname, est_bp=1e4 * mu, se_ev_bp=1e4 * se, se_mkt_bp=1e4 * sm,
                               n=n, n_markets=Gm, n_ev_clusters=Ge))
save(pd.DataFrame(cr), "clk_C_forward.csv")

# ---- D: expanding-window forward prediction at the 1-hour clock horizon (cutoff S and C)
log("forward prediction")
from sklearn.linear_model import LogisticRegression
def lg(x): x = np.clip(np.asarray(x, float), 0.005, 0.995); return np.log(x / (1 - x))
QTR = "strftime(date_trunc('quarter', to_timestamp(res_ts)), '%Y-Q') || cast(quarter(to_timestamp(res_ts)) AS VARCHAR)"
dr = []
rng = np.random.default_rng(SEED)
for kname in ["S", "C"]:
    d = con.sql(f"""SELECT condition_id, ev_cluster, cat, size3, D::DOUBLE D, Y::DOUBLE Y, mid_c3600 mh, {QTR} q
        FROM m WHERE {common(CUTS[kname])} AND cat IN ('Politics','Crypto')""").df()
    d["lm"] = lg(d.mh)
    for c in ["Politics", "Crypto"]:
        for s, tag in [("s1 <$100", "s1"), ("s2 $100-1k", "s2"), ("s3 >=$1k", "s3")]:
            d[f"D_{c[:3]}_{tag}"] = d.D * ((d.cat == c) & (d.size3 == s))
    d["is_pol"] = (d.cat == "Politics").astype(float)
    flow = [c for c in d.columns if c.startswith("D_")]
    qs = sorted(d.q.unique()); test_q = [q for q in qs if q >= "2025-Q1"]
    for q in test_q:
        tr, te = d[d.q < q], d[d.q == q]
        if len(te) < 2000 or tr.Y.nunique() < 2: continue
        X0 = ["lm", "is_pol"]; X1 = X0 + flow
        m0 = LogisticRegression(C=1e6, max_iter=1000).fit(tr[X0], tr.Y); m1 = LogisticRegression(C=1e6, max_iter=1000).fit(tr[X1], tr.Y)
        p0 = np.clip(m0.predict_proba(te[X0])[:, 1], 1e-6, 1 - 1e-6); p1 = np.clip(m1.predict_proba(te[X1])[:, 1], 1e-6, 1 - 1e-6)
        y = te.Y.to_numpy()
        for cell, mask in [("Politics+Crypto", np.ones(len(te), bool)), ("Politics", (te.cat == "Politics").to_numpy()),
                           ("Politics | s1 <$100", ((te.cat == "Politics") & (te.size3 == "s1 <$100")).to_numpy()),
                           ("Crypto", (te.cat == "Crypto").to_numpy())]:
            if mask.sum() < 200: continue
            db = ((p0 - y) ** 2 - (p1 - y) ** 2)[mask]
            dl = (-(y * np.log(p0) + (1 - y) * np.log(1 - p0)) + (y * np.log(p1) + (1 - y) * np.log(1 - p1)))[mask]
            dr.append(dict(cutoff=kname, test_quarter=q, cell=cell, n_test=int(mask.sum()), n_train=len(tr),
                           ev_cluster=None, sum_dbrier=db.sum(), sum_dlogloss=dl.sum(),
                           coef_pol_s1=float(m1.coef_[0][X1.index("D_Pol_s1")]), coef_cry_s1=float(m1.coef_[0][X1.index("D_Cry_s1")])))
            # store per-cluster sums for the pooled bootstrap
            g = pd.DataFrame(dict(cl=te.ev_cluster.to_numpy()[mask], db=db, dl=dl, k=1.0)).groupby("cl").sum()
            g["cutoff"], g["cell"], g["q"] = kname, cell, q
            dr[-1]["_clusters"] = g
D = pd.DataFrame([{k: v for k, v in r.items() if k != "_clusters"} for r in dr])
summ = []
for (kname, cell), grp in D.groupby(["cutoff", "cell"]):
    cl = pd.concat([r["_clusters"] for r in dr if r["cutoff"] == kname and r["cell"] == cell])
    n = grp.n_test.sum(); G = len(cl)
    sb, sl, sk = cl.db.to_numpy(), cl.dl.to_numpy(), cl.k.to_numpy()
    Wb = rng.multinomial(G, np.full(G, 1 / G), size=2000).astype(float)
    bb = (Wb @ sb) / (Wb @ sk); bl = (Wb @ sl) / (Wb @ sk)
    summ.append(dict(cutoff=kname, test_quarter="pooled forward", cell=cell, n_test=n, sum_dbrier=sb.sum(), sum_dlogloss=sl.sum(),
                     dbrier=sb.sum() / n, dbrier_lo=np.percentile(bb, 2.5), dbrier_hi=np.percentile(bb, 97.5),
                     dlogloss=sl.sum() / n, dlogloss_lo=np.percentile(bl, 2.5), dlogloss_hi=np.percentile(bl, 97.5), n_clusters=G))
D["dbrier"] = D.sum_dbrier / D.n_test; D["dlogloss"] = D.sum_dlogloss / D.n_test
save(pd.concat([D.drop(columns=["ev_cluster"]), pd.DataFrame(summ)]), "clk_D_forward_pred.csv")
manifest("15_clock_horizon")
log("done")
