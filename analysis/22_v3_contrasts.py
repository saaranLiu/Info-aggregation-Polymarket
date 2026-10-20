"""Direct contrasts the text compares, with event-cluster and two-way intervals.

On the observed 100-trade horizon, and on the one-hour common samples of cutoffs R and S:

  theta1  mean residual of Politics orders below $100
  theta2  Politics large minus Politics small
  theta3  Politics small minus Crypto small

The same contrasts are reported for alignment and for the price response. Standard errors come from one
regression with topic-by-size indicators, so the contrasts share their covariance. The two-way error also
clusters by calendar week. Equivalence of the Crypto small residual uses 50 bp and 25 bp margins.

Output: ctr_A_contrasts.csv
"""
from study_common import *
import math
import statsmodels.api as sm

def cgm(y, X, g1, g2):
    """Cameron-Gelbach-Miller two-way cluster covariance of the OLS coefficients."""
    n, k = X.shape
    b, *_ = np.linalg.lstsq(X, y, rcond=None)
    e = y - X @ b
    bread = np.linalg.inv(X.T @ X)
    def meat(g):
        codes = pd.factorize(g)[0]
        G = int(codes.max()) + 1
        S = np.zeros((G, k))
        np.add.at(S, codes, X * e[:, None])
        return (S.T @ S) * (G / (G - 1)) * ((n - 1) / (n - k)), G
    m1, _ = meat(g1); m2, _ = meat(g2)
    m12, _ = meat(pd.Series(g1).astype(str).to_numpy() + "|" + pd.Series(g2).astype(str).to_numpy())
    V = bread @ (m1 + m2 - m12) @ bread
    return b, V

def contrasts(d, sample, metric):
    d = d.dropna(subset=["y", "ev_cluster", "cal_week"]).copy()
    d = d[d.cat.isin(["Politics", "Crypto"]) & d.size3.isin(["s1 <$100", "s2 $100-1k", "s3 >=$1k"])]
    keys = [(c, s) for c in ["Politics", "Crypto"] for s in ["s1 <$100", "s2 $100-1k", "s3 >=$1k"]]
    X = np.column_stack([((d.cat == c) & (d.size3 == s)).to_numpy(float) for c, s in keys])
    y = d.y.to_numpy(float)
    b, V = cgm(y, X, d.ev_cluster.to_numpy(), d.cal_week.to_numpy())
    fit = sm.OLS(y, X).fit(cov_type="cluster", cov_kwds={"groups": pd.factorize(d.ev_cluster)[0]})
    want = {"theta1 Politics small": (0, None), "theta2 Politics large-small": (2, 0), "theta3 Politics-Crypto small": (0, 3)}
    out = []
    for name, (i, j) in want.items():
        L = np.zeros(6); L[i] = 1
        if j is not None: L[j] = -1
        est = float(L @ fit.params)
        se = float(np.sqrt(L @ fit.cov_params() @ L))
        se2 = float(np.sqrt(max(L @ V @ L, 0)))
        z = est / se if se else np.nan
        out.append(dict(sample=sample, metric=metric, contrast=name, est_bp=est, se_ev_bp=se, se_2way_bp=se2,
                        p=float(2 * (1 - math.erf(abs(z) / math.sqrt(2)))) if se else np.nan,
                        n=int(len(d)), n_clusters=int(d.ev_cluster.nunique())))
    # equivalence for the Crypto small residual, one-way event clusters
    est, se = float(fit.params[3]), float(np.sqrt(fit.cov_params()[3, 3]))
    for m in (50.0, 25.0):
        z1, z2 = (m - est) / se, (m + est) / se
        p = max(1 - math.erf(z1 / math.sqrt(2)), 1 - math.erf(z2 / math.sqrt(2)))  # one-sided tail, normal
        out.append(dict(sample=sample, metric=metric, contrast=f"equiv Crypto small {m:.0f}bp", est_bp=est, se_ev_bp=se,
                        se_2way_bp=np.nan, p=float(p), n=int(((d.cat == "Crypto") & (d.size3 == "s1 <$100")).sum()),
                        n_clusters=int(d[(d.cat == "Crypto") & (d.size3 == "s1 <$100")].ev_cluster.nunique())))
    return out

con = connect(threads=32, mem="80GB")
log("horizon sample")
con.execute(f"""CREATE OR REPLACE TABLE m AS SELECT cat, size3, ev_cluster, cal_week,
        1e4*D*(mid_after_100-Y) AS over, 1e4*D*(Y-mid_before) AS info, 1e4*D*(mid_after_100-mid_before) AS pi
    FROM '{WORK}/mid_YesNo.parquet' WHERE {LATE} AND {BASE} AND mid_after_100 IS NOT NULL AND ts_lead_100 IS NOT NULL
      AND cat IN ('Politics','Crypto')""")
d = con.sql("SELECT * FROM m").df()
rows = []
for metric in ["over", "info", "pi"]:
    dd = d.rename(columns={metric: "y"})
    rows += contrasts(dd, "h100", metric)
    log(f"h100 {metric}: theta2 {rows[-5]['est_bp']:.1f} ({rows[-5]['se_ev_bp']:.1f})")

# one-hour clock residual on the common sample of cutoffs R and S
log("clock midpoints")
con.execute(f"""CREATE OR REPLACE TABLE ev AS SELECT condition_id, rn, ts, D, p, Y, cat, size3, ev_cluster, cal_week,
        res_ts::DOUBLE res_ts, tau::DOUBLE tau, tau_close, mid_before, ts_lead_100, kind
    FROM '{WORK}/mid_YesNo.parquet' WHERE cat IN ('Politics','Crypto')""")
con.execute("""CREATE OR REPLACE TABLE mids AS SELECT condition_id, ts_mid, arg_max(mid, rn_mid) AS mid FROM (
    SELECT a.condition_id, b.ts AS ts_mid, b.rn AS rn_mid, (a.p + b.p)/2.0 AS mid
    FROM ev a JOIN ev b ON a.condition_id = b.condition_id AND b.rn = a.rn + 1 WHERE a.kind = 'switch') GROUP BY 1, 2""")
con.execute("""CREATE OR REPLACE TABLE cut AS SELECT condition_id, any_value(res_ts) R,
    least(any_value(res_ts), coalesce(any_value(ts + tau_close), any_value(res_ts))) S FROM ev GROUP BY 1""")
con.execute(f"""CREATE OR REPLACE TABLE mm AS SELECT e.*, x.mid AS m1, c.R, c.S
    FROM (SELECT * FROM ev WHERE {LATE} AND {BASE}) e
    ASOF LEFT JOIN mids x ON x.condition_id = e.condition_id AND x.ts_mid <= e.ts + 3600
    JOIN cut c ON c.condition_id = e.condition_id""")
for k in ["R", "S"]:
    dd = con.sql(f"""SELECT cat, size3, ev_cluster, cal_week, 1e4*D*(m1-Y) AS y
        FROM mm WHERE ts + 7200 <= {k} AND ts_lead_100 IS NOT NULL AND ts_lead_100 <= {k} AND m1 IS NOT NULL""").df()
    rows += contrasts(dd, f"clock1h_{k}", "over")
    log(f"clock {k}: theta1 {rows[-8]['est_bp']:.1f} theta2 {rows[-7]['est_bp']:.1f}")
save(pd.DataFrame(rows), "ctr_A_contrasts.csv")
log("done")
