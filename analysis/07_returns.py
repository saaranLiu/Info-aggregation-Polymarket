"""realized spread as gross settlement margin, weighting, fees, risk proxies, and market-day competition .

  A  RS = D(p - Y) under count, contract (share), USD and equal-market weights, by topic / size / time-to-resolution; total gross dollars
  B  taker fees from the fill tape (fee_usdc): fee per contract and fee-adjusted taker cost, by topic and year
  C  RS by risk proxies: time to resolution, |p - 0.5| layer, order size relative to the market's average order
  D  market-day competition panel: active makers, maker HHI, top-maker share; association with RS within topic x resolution day
     and within market (market fixed effects); share of market-hours with >= 2 distinct makers hit
Outputs: results/ret_A_weights_{grp}.csv, ret_A_dollars_{grp}.csv, ret_B_fees_{grp}.csv, ret_C_risk_{grp}.csv, ret_D_marketday_{grp}.csv,
         ret_D_competition_reg_{grp}.csv, ret_D_hours_{grp}.csv
"""
from study_common import *
import statsmodels.formula.api as smf
FP = C; del C          # patsy resolves C(...) from the caller namespace; keep the full-pipeline module under another name
con = connect()
BR = 300 if SMOKE else 500
for grp in ["YesNo", "SportsML"]:
    log(f"[{grp}] A weights")
    mid_table(con, grp)
    con.execute("CREATE OR REPLACE TABLE t AS SELECT *, usd/p AS shares, count(*) OVER (PARTITION BY condition_id) AS n_mkt_all FROM e WHERE mn >= 100 AND tau >= 3600")
    cells = {"tau_b": "tau_b"} if grp != "YesNo" else {"tau_b": "tau_b", "cat": "CASE WHEN cat IN ('Politics','Crypto') THEN cat ELSE 'other topics' END"}
    out = []
    for wname, wexpr in [("count", None), ("shares", "shares"), ("usd", "usd"), ("equal_market", "1.0/n_mkt_all")]:
        d = boot(con, "t", {"weight": f"'{wname}'", **cells}, {"RS": "(D*(p-Y))", "eff": "(D*(p-mid_before))", "info": "(D*(Y-mid_before))"}, f"{MID} AND mid_before IS NOT NULL", weight=wexpr, B=BR); out.append(d)
        d = boot(con, "t", {"weight": f"'{wname}'", **cells, "size3": "size3"}, {"RS": "(D*(p-Y))"}, f"{MID} AND mid_before IS NOT NULL AND {LATE}", weight=wexpr, B=BR); out.append(d)
    out = pd.concat(out); out.insert(0, "grp", grp); save(out, f"ret_A_weights_{grp}.csv")
    dol = con.sql(f"""SELECT '{grp}' grp, {cells.get('cat', "'all'")} cat, tau_b, count(*) n_trades, sum(usd) usd, sum(shares) contracts,
        sum(shares*D*(p-Y)) maker_gross_usd, sum(shares*D*(p-Y))/sum(shares)*1e4 RS_share_weighted_bp, sum(shares*D*(p-Y))/sum(usd)*1e4 maker_gross_per_usd_bp
        FROM t WHERE {MID} GROUP BY ALL ORDER BY 2,3""").df()
    save(dol, f"ret_A_dollars_{grp}.csv")
    # ---- B fees from fills
    log(f"[{grp}] B fees (fill tape)")
    con.execute(f"CREATE OR REPLACE TABLE mk AS SELECT condition_id, Y, cat FROM '{FRES}/markets.parquet' WHERE eligible AND grp='{grp}'")
    con.execute(f"""CREATE OR REPLACE TABLE fills AS
        SELECT f.condition_id, f.block_timestamp ts, f.taker, f.maker, f.D, f.p_event p, f.usdc_amount usd, f.fee_usdc fee, f.usdc_amount/f.p_event shares, m.Y, m.cat
        FROM read_parquet('{FP.DA}') f JOIN mk m USING(condition_id) JOIN (SELECT DISTINCT condition_id FROM e) k USING(condition_id)
        WHERE f.p_event IS NOT NULL AND f.D IS NOT NULL AND f.usdc_amount > 0 AND f.p_event > 0""")
    fees = con.sql(f"""SELECT '{grp}' grp, {"CASE WHEN cat IN ('Politics','Crypto') THEN cat ELSE 'other topics' END" if grp=='YesNo' else "'all'"} cat, year(to_timestamp(ts)) yr,
        count(*) n_fills, avg((fee > 0)::int) share_fee_positive, sum(fee)/sum(usd)*1e4 fee_per_usd_bp, sum(fee)/sum(shares)*1e4 fee_per_contract_bp,
        avg(D*(p-Y))*1e4 RS_fill_bp, (sum(shares*D*(p-Y)) + sum(fee))/sum(shares)*1e4 taker_cost_incl_fee_share_weighted_bp
        FROM fills WHERE abs(p-0.5) < 0.3 GROUP BY ALL ORDER BY 2,3""").df()
    save(fees, f"ret_B_fees_{grp}.csv")
    # ---- C risk proxies
    risk = boot(con, "t", {"tau_b": "tau_b", "layer": FP.PRICE_LAYER}, {"RS": "(D*(p-Y))"}, "abs(p-0.5) < 0.45", B=BR); risk.insert(0, "grp", grp)
    con.execute("CREATE OR REPLACE TABLE t2 AS SELECT *, usd / avg(usd) OVER (PARTITION BY condition_id) AS rel_size FROM t")
    rel = boot(con, "t2", {"rel_size_b": "CASE WHEN rel_size < 0.25 THEN 'a <0.25x' WHEN rel_size < 1 THEN 'b 0.25-1x' WHEN rel_size < 4 THEN 'c 1-4x' ELSE 'd >=4x' END"},
               {"RS": "(D*(p-Y))", "info": "(D*(Y-mid_before))", "eff": "(D*(p-mid_before))"}, f"{MID} AND mid_before IS NOT NULL AND {LATE}", B=BR); rel.insert(0, "grp", grp)
    save(pd.concat([risk, rel]), f"ret_C_risk_{grp}.csv")
    # ---- D market-day competition
    log(f"[{grp}] D market-day panel")
    md = con.sql(f"""WITH pm AS (SELECT condition_id, cat, to_timestamp(ts)::DATE dday, maker, sum(usd) v, count(*) k FROM fills WHERE abs(p-0.5) < 0.3 GROUP BY 1,2,3,4),
        h AS (SELECT condition_id, cat, dday, count(*) n_makers, sum(v*v)/power(sum(v),2) maker_hhi, max(v)/sum(v) top1_share, sum(k) n_fills, sum(v) usd FROM pm GROUP BY 1,2,3),
        r AS (SELECT condition_id, to_timestamp(ts)::DATE dday, avg(D*(p-Y)) RS, sum(usd*D*(p-Y))/sum(usd) RS_usdw, count(*) n_trades, count(distinct taker) n_takers,
                     avg(abs(p-0.5)) mean_abs_p, avg(tau) mean_tau, any_value(Y) Y, any_value(ev_cluster) ev_cluster
              FROM t WHERE {MID} GROUP BY 1,2)
        SELECT r.*, h.cat, h.n_makers, h.maker_hhi, h.top1_share, h.n_fills, h.usd FROM r JOIN h USING(condition_id, dday) WHERE r.n_trades >= 20""").df()
    desc = md.groupby("cat" if grp == "YesNo" else lambda i: "all").agg(n_market_days=("RS", "size"), n_markets=("condition_id", "nunique"), makers_med=("n_makers", "median"), makers_q10=("n_makers", lambda s: s.quantile(.1)),
                  makers_q90=("n_makers", lambda s: s.quantile(.9)), hhi_med=("maker_hhi", "median"), top1_med=("top1_share", "median"), share_single_maker=("n_makers", lambda s: (s == 1).mean()),
                  RS_bp=("RS", lambda s: 1e4 * s.mean())).reset_index()
    save(desc, f"ret_D_marketday_{grp}.csv")
    md["RS_bp"] = 1e4 * md.RS; md["log_tau"] = np.log(md.mean_tau.clip(lower=60)); md["log_n"] = np.log(md.n_trades); md["cl"] = md.ev_cluster.astype("category").cat.codes; md["log_makers"] = np.log(md.n_makers)
    rows = []
    import statsmodels.api as sm
    def fe_ols(dd, yv, xv, fe):
        """OLS of yv on xv with fixed effects fe absorbed by within-group demeaning; cluster-robust SEs by event cluster."""
        cols = [yv] + xv; Z = dd[cols] - dd.groupby(fe)[cols].transform("mean")
        r = sm.OLS(Z[yv].to_numpy(), Z[xv].to_numpy()).fit(cov_type="cluster", cov_kwds={"groups": dd.cl.to_numpy()})
        return r
    for cname, dd in ([("pooled", md)] + ([(c, md[md.cat == c]) for c in ["Politics", "Crypto"]] if grp == "YesNo" else [])):
        if len(dd) < (100 if SMOKE else 500): continue
        for fname, v, fe in [("event FE", "maker_hhi", "ev_cluster"), ("market FE", "maker_hhi", "condition_id"), ("event FE, top1", "top1_share", "ev_cluster"), ("event FE, log makers", "log_makers", "ev_cluster")]:
            try:
                r = fe_ols(dd, "RS_bp", [v, "log_tau", "log_n", "mean_abs_p"], fe)
            except Exception as ex:
                log("reg failed", cname, fname, ex); continue
            rows.append(dict(grp=grp, cell=cname, spec=fname, var=v, coef_bp=r.params[0], se_bp=r.bse[0], p=r.pvalues[0], n=int(r.nobs), sd_var=dd[v].std()))
    save(pd.DataFrame(rows), f"ret_D_competition_reg_{grp}.csv")
    hrs = con.sql(f"""WITH ph AS (SELECT condition_id, cat, date_trunc('hour', to_timestamp(ts)) hr, count(distinct maker) n_makers, count(*) k FROM fills WHERE abs(p-0.5) < 0.3 GROUP BY 1,2,3)
        SELECT '{grp}' grp, {"cat" if grp=='YesNo' else "'all'"} cat, count(*) n_market_hours, avg((n_makers >= 2)::int) share_ge2_makers, avg((n_makers >= 2)::int) FILTER (k >= 10) share_ge2_makers_active_hours,
               quantile_cont(n_makers, 0.5) med_makers, quantile_cont(n_makers, 0.5) FILTER (k >= 10) med_makers_active_hours FROM ph GROUP BY ALL ORDER BY 2""").df()
    save(hrs, f"ret_D_hours_{grp}.csv")
    manifest(f"07_returns_{grp}")
log("done")
