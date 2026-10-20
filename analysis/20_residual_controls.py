"""Politics-versus-Crypto small-order residuals within common market-state cells.

On the common sample of each cutoff (as in 15_clock_horizon.py), regress the one-hour clock residual
D*(m_{t+1h} - Y) on topic-by-size indicators (Politics and Crypto, three size bins) with fixed effects for
midpoint decile, hours-to-resolution bin and resolution quarter, and log market trade count. The reported
coefficient is the Politics-small minus Crypto-small contrast; SEs cluster by topic x resolution date.

Output: results/ctl_A_residual_controls.csv
"""
from study_common import *
import statsmodels.formula.api as smf
del C

con = connect()
con.execute(f"""CREATE OR REPLACE TABLE ev AS SELECT condition_id, rn, ts, D, p, Y, cat, res_ts::DOUBLE res_ts, tau::DOUBLE tau, tau_close,
    mid_before, ev_cluster, size3, ts_lead_100, kind, mn FROM '{WORK}/mid_YesNo.parquet' WHERE cat IN ('Politics','Crypto')""")
con.execute("""CREATE OR REPLACE TABLE mids AS SELECT condition_id, ts_mid, arg_max(rn_mid, rn_mid) AS rn_mid, arg_max(mid, rn_mid) AS mid FROM (
    SELECT a.condition_id, b.ts AS ts_mid, b.rn AS rn_mid, (a.p + b.p)/2.0 AS mid
    FROM ev a JOIN ev b ON a.condition_id = b.condition_id AND b.rn = a.rn + 1 WHERE a.kind = 'switch') GROUP BY 1, 2""")
con.execute("""CREATE OR REPLACE TABLE cut AS SELECT condition_id, any_value(res_ts) R,
    least(any_value(res_ts), coalesce(any_value(ts + tau_close), any_value(res_ts))) S,
    coalesce(max(ts) FILTER (WHERE (Y*p + (1-Y)*(1-p)) < 0.95), any_value(res_ts)) Cq FROM ev GROUP BY 1""")
con.execute(f"""CREATE OR REPLACE TABLE m AS SELECT e.*, x.mid AS mid_c3600 FROM (SELECT * FROM ev WHERE {LATE} AND {BASE}) e
    ASOF LEFT JOIN mids x ON x.condition_id = e.condition_id AND x.ts_mid <= e.ts + 3600""")
con.execute("CREATE OR REPLACE TABLE m2 AS SELECT m.*, c.R, c.S, least(c.S, c.Cq) AS C FROM m JOIN cut c USING(condition_id)")
rows = []
for k in ["R", "S", "C"]:
    d = con.sql(f"""SELECT 1e4*D*(mid_c3600 - Y) AS y, cat || '|' || size3 AS cs, floor(10*mid_before)::INT AS mdec,
            floor(tau/3600)::INT AS hbin, strftime(date_trunc('quarter', to_timestamp(res_ts)), '%Y-%m') AS qtr, ln(mn) AS lmn, ev_cluster
        FROM m2 WHERE ts + 7200 <= {k} AND ts_lead_100 IS NOT NULL AND ts_lead_100 <= {k} AND mid_c3600 IS NOT NULL""").df()
    g = pd.factorize(d.ev_cluster)[0]
    for spec, f in [("topic x size only", "y ~ 0 + C(cs)"),
                    ("+ midpoint decile, hours-to-resolution, quarter FE, log trades", "y ~ 0 + C(cs) + C(mdec) + C(hbin) + C(qtr) + lmn")]:
        fit = smf.ols(f, data=d).fit(cov_type="cluster", cov_kwds={"groups": g})
        a, b = "C(cs)[Crypto|s1 <$100]", "C(cs)[Politics|s1 <$100]"
        if a not in fit.params: a = [t for t in fit.params.index if "Crypto|s1" in t][0]; b = [t for t in fit.params.index if "Politics|s1" in t][0]
        cvec = pd.Series(0.0, index=fit.params.index); cvec[b] = 1; cvec[a] = -1
        t = fit.t_test(cvec.values)
        rows.append(dict(cutoff=k, spec=spec, n=int(fit.nobs), contrast_bp=float(t.effect[0]), se_ev_bp=float(t.sd[0][0]), p=float(t.pvalue)))
        log(f"{k} | {spec}: {rows[-1]['contrast_bp']:.1f} ({rows[-1]['se_ev_bp']:.1f})")
save(pd.DataFrame(rows), "ctl_A_residual_controls.csv")
log("done")
