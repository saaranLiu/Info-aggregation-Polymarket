"""Stage 05 (E4): quote channel. Without an order book we use (i) consecutive-pair decomposition (same-side pairs ~ one-sided
quote revision; direction switches ~ bounce + midpoint move) and (ii) a midpoint proxy m = (p_i + p_{i+1})/2 at direction switches.
Questions: where does the drift toward the outcome occur? does the midpoint follow aggressive flow?"""
from common import *
con = connect()
for grp in [g for g in GROUPS if g != 'UpDown']:
    log(f"[{grp}] load")
    con.execute(f"CREATE OR REPLACE TABLE ev AS SELECT * FROM '{WORK}/ev_{grp}.parquet' WHERE mn >= {MAIN_MIN_TRADES}")
    # ---- (i) pair decomposition
    con.execute("ALTER TABLE ev ADD COLUMN kind VARCHAR; UPDATE ev SET kind = CASE WHEN D_next IS NULL THEN NULL WHEN D = D_next THEN 'same' ELSE 'switch' END")
    for tag, where in [("mid", MID), ("all", "TRUE")]:
        df = boot_from_sql(con, "ev", {"tau_b": TAU_COARSE, "kind": "kind"},
                           {"R_1": R("p_lead_1"), "dq_1": dq("p_lead_1"), "absdp_1": "abs(p_lead_1 - p)"}, f"{where} AND kind IS NOT NULL")
        df.insert(0,'grp',grp); save(df, f"quote_{grp}_pairs_{tag}.csv")
        tot = con.sql(f"""SELECT {TAU_COARSE} tau_b, kind, count(*) n, sum({dq('p_lead_1')}) sum_dq, sum({dq('p_lead_1')}) / sum(sum({dq('p_lead_1')})) OVER (PARTITION BY {TAU_COARSE}) share_of_drift
                          FROM ev WHERE {where} AND kind IS NOT NULL GROUP BY 1,2 ORDER BY 1,2""").df()
        tot.insert(0,'grp',grp); save(tot, f"quote_{grp}_driftshare_{tag}.csv")
    # ---- (ii) midpoint proxy at switches
    con.execute("""CREATE OR REPLACE TABLE mids AS SELECT condition_id, rn + 1 AS rn_mid, ts, (p + p_lead_1)/2.0 AS mid, Y FROM ev WHERE kind = 'switch'""")
    con.execute("""CREATE OR REPLACE TABLE e1 AS SELECT e.*, m.mid AS mid_before FROM ev e ASOF LEFT JOIN mids m ON m.condition_id = e.condition_id AND m.rn_mid <= e.rn - 1""")
    for h in (1, 5, 20, 100):
        con.execute(f"""CREATE OR REPLACE TABLE e2 AS SELECT e.*, m.mid AS mid_after_{h} FROM e1 e ASOF LEFT JOIN mids m ON m.condition_id = e.condition_id AND m.rn_mid <= e.rn + {h}""")
        con.execute("DROP TABLE e1; ALTER TABLE e2 RENAME TO e1")
    mets = {}
    for h in (1, 5, 20, 100):
        mets[f"Rmid_{h}"] = f"(D*(mid_after_{h} - mid_before))"; mets[f"dqmid_{h}"] = f"((2*Y-1)*(mid_after_{h} - mid_before))"
        mets[f"Rtrade_{h}"] = R(f"p_lead_{h}"); mets[f"dqtrade_{h}"] = dq(f"p_lead_{h}")
    df = boot_from_sql(con, "e1", {"tau_b": TAU_COARSE}, mets, f"{MID} AND mid_before IS NOT NULL"); df.insert(0,'grp',grp); save(df, f"quote_{grp}_midpoint_response.csv")
    df = boot_from_sql(con, "e1", {"tau_b": TAU_COARSE, "size_b": SIZE_B}, {k: v for k, v in mets.items() if k.startswith('Rmid')}, f"{MID} AND mid_before IS NOT NULL"); df.insert(0,'grp',grp); save(df, f"quote_{grp}_midpoint_by_size.csv")
    # ---- (iii) does net aggressive flow between midpoint updates predict the midpoint change?
    con.execute("""CREATE OR REPLACE TABLE seg AS
      SELECT e.condition_id, m.rn_mid, sum(e.D*e.usd) F_usd, sum(e.D) F_cnt, count(*) n_tr, sum(e.usd) usd, any_value(e.tau) tau, any_value(e.Y) Y, any_value(e.p) p_first_in_seg
      FROM ev e ASOF JOIN mids m ON m.condition_id = e.condition_id AND m.rn_mid >= e.rn + 1
      GROUP BY 1,2""")
    con.execute("""CREATE OR REPLACE TABLE segm AS
      SELECT s.*, m.mid AS mid_next, lag(m.mid) OVER (PARTITION BY s.condition_id ORDER BY s.rn_mid) AS mid_prev
      FROM seg s JOIN mids m USING(condition_id, rn_mid)""")
    df = boot_from_sql(con, "segm", {"tau_b": TAU_COARSE},
        {"Rmid_flow_usd": "(sign(F_usd)*(mid_next - mid_prev))", "Rmid_flow_cnt": "(sign(F_cnt)*(mid_next - mid_prev))",
         "drift_mid": "((2*Y-1)*(mid_next - mid_prev))", "abs_dmid": "abs(mid_next - mid_prev)",
         "flow_toward_outcome_usd": "((2*Y-1)*sign(F_usd) = 1)::int", "n_trades_per_update": "n_tr"},
        f"mid_prev IS NOT NULL AND F_usd <> 0 AND abs(p_first_in_seg-0.5) < 0.3")
    df.insert(0,'grp',grp); save(df, f"quote_{grp}_flow_to_midpoint.csv")
    # OLS slope of mid change on standardized net flow, per tau bucket, with market-cluster bootstrap
    seg = con.sql(f"""SELECT condition_id, {TAU_COARSE} tau_b, (mid_next - mid_prev) dm, F_usd / nullif(avg(usd) OVER (PARTITION BY condition_id),0) f_std, Y
                      FROM segm WHERE mid_prev IS NOT NULL AND abs(p_first_in_seg-0.5) < 0.3""").df()
    rows = []; rng = np.random.default_rng(SEED)
    for tb, g in seg.groupby('tau_b'):
        g = g.dropna(); mk = g.condition_id.astype('category').cat.codes.to_numpy(); M = mk.max()+1
        x = g.f_std.to_numpy(); y = g.dm.to_numpy()
        sxx = np.bincount(mk, x*x, M); sxy = np.bincount(mk, x*y, M); sx = np.bincount(mk, x, M); sy = np.bincount(mk, y, M); nn = np.bincount(mk, None, M).astype(float)
        def slope(w): 
            n = (w*nn).sum(); mx = (w*sx).sum()/n; my = (w*sy).sum()/n
            return ((w*sxy).sum() - n*mx*my) / ((w*sxx).sum() - n*mx*mx)
        b = slope(np.ones(M)); bs = np.array([slope(rng.multinomial(M, np.full(M,1/M)).astype(float)) for _ in range(B_BOOT)])
        rows.append(dict(grp=grp, tau_b=tb, n_updates=len(g), n_markets=M, slope_bp_per_avgtrade=1e4*b, se=1e4*bs.std(ddof=1), lo=1e4*np.percentile(bs,2.5), hi=1e4*np.percentile(bs,97.5), corr=np.corrcoef(x,y)[0,1]))
    save(pd.DataFrame(rows), f"quote_{grp}_flow_to_midpoint_slope.csv")
    manifest(f"05_quote_{grp}")
log("done")
