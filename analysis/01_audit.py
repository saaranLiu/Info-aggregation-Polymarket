"""sample, weight, merge-rule, horizon, time-axis and price-filter audit .

Outputs (results/):
  audit_A_sample_flow.csv          database -> eligible -> group -> >=100 trades -> final day -> mid price -> proxy -> fresh -> horizon
  audit_B_merge_{grp}.csv          fills-per-trade, shared-block ordering ambiguity, price dispersion inside merged trades (YesNo from fills)
  audit_B_merge_metrics_YesNo.csv  info/eff/RS at fill level vs trade level, final day
  audit_C_weights_{grp}.csv        info/eff/RS/PI/over under count, share, USD and equal-market weights on one common sample; identity closure
  audit_D_horizon_{grp}.csv        clock length of the 100-trade horizon; share of trades without it; metrics on all vs horizon-complete trades
  audit_D_midafter_{grp}.csv       share of trades whose "mid_after_100" is not after the trade (no switch within 100 trades)
  audit_E_lastmid_{grp}.csv        PI_clk = info "by construction" check: last midpoint vs Y
  audit_F_timeaxis_{grp}.csv       on-chain resolution vs scheduled close vs metadata resolved_at; final-day metrics under each
  audit_G_pricefilter_{grp}.csv    trade-price vs midpoint filter; wider ranges by |m-0.5| layer; buy share under each
"""
from study_common import *
con = connect()
BA = 300 if not SMOKE else 100

# ---------------- A. sample flow ----------------
rows = []
mk = con.sql(f"SELECT grp, eligible, count(*) n_markets, sum(n_fills) n_fills, sum(vol) usd FROM '{FRES}/markets.parquet' GROUP BY 1,2").df()
rows.append(dict(grp="ALL", stage="0 all markets in database", n_markets=int(mk.n_markets.sum()), n_fills=float(mk.n_fills.sum()), usd=float(mk.usd.sum())))
el = mk[mk.eligible == True]
rows.append(dict(grp="ALL", stage="1 resolved, binary payout, metadata = on-chain winner", n_markets=int(el.n_markets.sum()), n_fills=float(el.n_fills.sum()), usd=float(el.usd.sum())))
for _, r in el.iterrows():
    rows.append(dict(grp=r.grp, stage="2 eligible, by contract type", n_markets=int(r.n_markets), n_fills=float(r.n_fills), usd=float(r.usd)))
for grp in GROUPS3:
    mid_table(con, grp)
    steps = [("3 >=100 trades (trade = market x block x taker x direction)", "TRUE"),
             ("4 final day (1h <= tau < 24h)", LATE),
             ("5 mid price |p-0.5|<0.3", f"{LATE} AND {MID}"),
             ("6 proxy midpoint defined", f"{LATE} AND {BASE}"),
             ("7 fresh proxy (stale<=1)", f"{LATE} AND {BASE} AND stale <= 1"),
             ("8 100-trade horizon observed", f"{LATE} AND {BASE} AND p_lead_100 IS NOT NULL"),
             ("9 horizon observed and a switch occurs within it", f"{LATE} AND {BASE} AND p_lead_100 IS NOT NULL AND rn_mid_after_100 > rn")]
    for name, w in steps:
        r = con.sql(f"SELECT count(*) n, count(distinct condition_id) m, count(distinct ev_cluster) ev, sum(usd) usd, sum(nfills) nf FROM e WHERE {w}").fetchone()
        rows.append(dict(grp=grp, stage=name, n_markets=r[1], n_trades=r[0], n_ev_clusters=r[2], n_fills=r[4], usd=r[3]))
    for c in ["Politics", "Crypto"]:
        if grp != "YesNo": continue
        for name, w in steps[3:]:
            r = con.sql(f"SELECT count(*) n, count(distinct condition_id) m, count(distinct ev_cluster) ev, sum(usd) usd, sum(nfills) nf FROM e WHERE cat='{c}' AND {w}").fetchone()
            rows.append(dict(grp=f"YesNo:{c}", stage=name, n_markets=r[1], n_trades=r[0], n_ev_clusters=r[2], n_fills=r[4], usd=r[3]))
save(pd.DataFrame(rows), "audit_A_sample_flow.csv")

for grp in GROUPS3:
    log(f"[{grp}] B-G")
    mid_table(con, grp)
    W = f"{LATE} AND {BASE}"
    # ---------------- B. merge rule ----------------
    b = con.sql(f"""WITH blk AS (SELECT condition_id, ts, count(*) k, count(distinct taker) ktak FROM e GROUP BY 1,2)
        SELECT '{grp}' grp, count(*) n_trades, avg((e.nfills > 1)::int) share_multifill, avg(e.nfills) mean_fills, quantile_cont(e.nfills, 0.99) p99_fills,
               avg((b.k > 1)::int) share_in_shared_block, avg((b.ktak > 1)::int) share_block_multi_taker,
               avg((b.k > 1 AND {LATE})::int)/nullif(avg(({LATE})::int),0) share_in_shared_block_lateday
        FROM e JOIN blk b USING(condition_id, ts)""").df()
    # same taker, same block, both directions (cannot be merged; two independent orders)
    b2 = con.sql("""WITH t AS (SELECT condition_id, ts, taker, count(distinct D) nd, count(*) k FROM e GROUP BY 1,2,3)
        SELECT avg((nd = 2)::int) share_taker_block_bothdir FROM t""").df()
    b["share_taker_block_bothdir"] = b2.iloc[0, 0]
    save(b, f"audit_B_merge_{grp}.csv")
    if grp == "YesNo" and not SMOKE:
        log("  fills (YesNo) for merge dispersion and fill-level metrics")
        con.execute(f"CREATE OR REPLACE TABLE mkts AS SELECT condition_id, Y FROM '{FRES}/markets.parquet' WHERE eligible AND grp='YesNo'")
        con.execute(f"""CREATE OR REPLACE TABLE fills AS
            SELECT t.condition_id, t.block_timestamp ts, t.taker, t.D, t.p_event p, t.usdc_amount usd, t.fee_usdc fee
            FROM read_parquet('{C.DA}') t JOIN mkts m USING(condition_id)
            WHERE t.p_event IS NOT NULL AND t.D IS NOT NULL AND t.usdc_amount > 0""")
        disp = con.sql("""WITH g AS (SELECT condition_id, ts, taker, D, count(*) k, count(distinct p) np, max(p)-min(p) prange FROM fills GROUP BY 1,2,3,4)
            SELECT count(*) n_trades, avg((k>1)::int) share_multifill, avg((k>1 AND np>1)::int) share_multifill_multiprice,
                   avg((k>1 AND prange >= 0.01)::int) share_multifill_range_ge_1tick, avg((k>1 AND prange >= 0.03)::int) share_multifill_range_ge_3ticks,
                   quantile_cont(prange, 0.5) FILTER (k>1) med_range_multifill, quantile_cont(prange, 0.9) FILTER (k>1) p90_range_multifill FROM g""").df()
        save(disp, "audit_B_merge_dispersion_YesNo.csv")
        # fill-level vs trade-level metrics (final day, mid, proxy defined): join fills to trades on the merge key
        con.execute(f"""CREATE OR REPLACE TABLE fj AS SELECT f.*, e.mid_before, e.Y, e.mid_after_100, e.tau, e.cat, e.ev_cluster
            FROM fills f JOIN (SELECT condition_id, ts, taker, D, mid_before, Y, mid_after_100, tau, cat, ev_cluster FROM e WHERE {W}) e USING(condition_id, ts, taker, D)
            WHERE abs(f.p-0.5) < 0.3""")
        mets = {"info": "(D*(Y-mid_before))", "eff": "(D*(p-mid_before))", "RS": "(D*(p-Y))", "PI_100": "(D*(mid_after_100-mid_before))", "over_100": "(D*(mid_after_100-Y))"}
        a1 = boot(con, "fj", {"level": "'fill'"}, mets, "TRUE", B=BA)
        a2 = boot(con, "e", {"level": "'trade'"}, METS, W, B=BA)
        a3 = boot(con, "e", {"level": "'trade, usd-weighted'"}, METS, W, weight="usd", B=BA)
        out = pd.concat([a1, a2, a3]); out.insert(0, "grp", grp); save(out, "audit_B_merge_metrics_YesNo.csv")
    # ---------------- C. weights on one common sample ----------------
    COMMON = f"{W} AND p_lead_100 IS NOT NULL AND mid_after_100 IS NOT NULL"
    con.execute(f"CREATE OR REPLACE TABLE cs AS SELECT *, count(*) OVER (PARTITION BY condition_id) AS n_in_cell, usd/p AS shares FROM e WHERE {COMMON}")
    mets = dict(METS); mets["R_100"] = "(D*(p_lead_100-p))"; mets["resid_100"] = "(D*(p_lead_100-mid_after_100))"; mets["s"] = "(I=1)::int"
    out = []
    for wname, wexpr in [("count", None), ("shares", "shares"), ("usd", "usd"), ("equal_market", "1.0/n_in_cell")]:
        d = boot(con, "cs", {"weight": f"'{wname}'"}, mets, "TRUE", weight=wexpr, B=BA); out.append(d)
        if grp == "YesNo":
            d = boot(con, "cs", {"weight": f"'{wname}'", "cat": "cat"}, mets, "cat IN ('Politics','Crypto')", weight=wexpr, B=BA); out.append(d)
    out = pd.concat(out); out.insert(0, "grp", grp)
    # identity closure per weight (pooled)
    pooled = out[out["cat"].isna()] if "cat" in out else out
    piv = pooled.pivot_table(index="weight", columns="metric", values="est")
    piv["closure_eff"] = (piv["eff"] - piv["info"] - piv["RS"]) * 1e4; piv["closure_PI"] = (piv["PI_100"] - piv["info"] - piv["over_100"]) * 1e4
    piv["closure_R"] = (piv["R_100"] - piv["over_100"] + piv["RS"] - piv["resid_100"]) * 1e4
    out = out.merge(piv[["closure_eff", "closure_PI", "closure_R"]].reset_index(), on="weight", how="left")
    save(out, f"audit_C_weights_{grp}.csv")
    # ---------------- D. horizon ----------------
    d1 = con.sql(f"""SELECT '{grp}' grp, cat, count(*) n, avg((p_lead_100 IS NULL)::int) share_no_h100,
            quantile_cont(ts_lead_100 - ts, [0.1, 0.25, 0.5, 0.75, 0.9]) FILTER (ts_lead_100 IS NOT NULL) q_h100_seconds,
            avg((ts_lead_100 - ts > 86400)::int) FILTER (ts_lead_100 IS NOT NULL) share_h100_longer_than_1d,
            quantile_cont(ts_lead_20 - ts, [0.1, 0.5, 0.9]) FILTER (ts_lead_20 IS NOT NULL) q_h20_seconds
        FROM e WHERE {W} GROUP BY 1,2 ORDER BY n DESC""").df()
    d2 = con.sql(f"""SELECT '{grp}' grp, size3, tau_b, count(*) n, avg((p_lead_100 IS NULL)::int) share_no_h100, avg(usd) mean_usd
        FROM e WHERE {BASE} GROUP BY 1,2,3 ORDER BY 2,3""").df()
    m_all = boot(con, "e", {"sample": "'all trades'"}, {k: METS[k] for k in ["info", "eff", "RS"]}, W, B=BA)
    m_h = boot(con, "e", {"sample": "'horizon-complete'"}, {k: METS[k] for k in ["info", "eff", "RS"]}, f"{W} AND p_lead_100 IS NOT NULL", B=BA)
    m_x = boot(con, "e", {"sample": "'excluded (no 100-trade horizon)'"}, {k: METS[k] for k in ["info", "eff", "RS"]}, f"{W} AND p_lead_100 IS NULL", B=BA)
    d3 = pd.concat([m_all, m_h, m_x]); d3.insert(0, "grp", grp)
    save(d1, f"audit_D_horizon_{grp}.csv"); save(d2, f"audit_D_horizon_selection_{grp}.csv"); save(d3, f"audit_D_horizon_metrics_{grp}.csv")
    d4 = con.sql(f"""SELECT '{grp}' grp, size3, count(*) n, avg((rn_mid_after_100 <= rn)::int) share_midafter_not_after,
            avg((rn_mid_after_20 <= rn)::int) share_midafter20_not_after,
            avg(D*(mid_after_100-Y))*1e4 over100_bp_all, avg(D*(mid_after_100-Y)) FILTER (rn_mid_after_100 > rn)*1e4 over100_bp_switch_within,
            avg(D*(Y-mid_before))*1e4 info_bp_all, avg(D*(Y-mid_before)) FILTER (rn_mid_after_100 > rn)*1e4 info_bp_switch_within
        FROM e WHERE {W} AND p_lead_100 IS NOT NULL GROUP BY 1,2 ORDER BY 2""").df()
    save(d4, f"audit_D_midafter_{grp}.csv")
    # ---------------- E. last midpoint vs Y ----------------
    H = C.CLOCK_H[grp][-1]
    e5 = con.sql(f"""SELECT '{grp}' grp, count(*) n, avg((ts + {H} >= res_ts)::int) share_clk_horizon_past_resolution,
            avg(abs(mid_last - Y)) mean_abs_lastmid_minus_Y, quantile_cont(abs(mid_last - Y), 0.5) med_abs_lastmid_minus_Y,
            avg(D*(Y - mid_last))*1e4 info_vs_lastmid_bp, avg(D*(mid_last - Y))*1e4 over_last_bp, avg(D*(Y - mid_before))*1e4 info_bp
        FROM e WHERE {W}""").df()
    save(e5, f"audit_E_lastmid_{grp}.csv")
    # ---------------- F. time axis ----------------
    f1 = con.sql(f"""SELECT '{grp}' grp, count(distinct condition_id) n_markets,
            quantile_cont(res_ts - epoch(close_at), [0.1, 0.5, 0.9]) q_res_minus_close_s,
            quantile_cont(res_ts - epoch(resolved_at), [0.1, 0.5, 0.9]) q_res_minus_resolvedat_s,
            avg((ts > epoch(close_at))::int) share_trades_after_close, avg((ts > epoch(resolved_at))::int) share_trades_after_resolvedat
        FROM (SELECT e.*, mk.close_at, mk.resolved_at FROM e JOIN (SELECT condition_id, close_at, resolved_at FROM '{FRES}/markets.parquet') mk USING(condition_id))""").df()
    save(f1, f"audit_F_timeaxis_dist_{grp}.csv")
    out = []
    for tname, texpr in [("on-chain resolution (main)", "tau"), ("scheduled close_at", "tau_close"), ("metadata resolved_at", "tau_resolved")]:
        w = f"{BASE} AND {texpr} >= 3600 AND {texpr} < 86400"
        d = boot(con, "e", {"time_axis": f"'{tname}'"}, METS, w, B=BA); out.append(d)
        if grp == "YesNo":
            d = boot(con, "e", {"time_axis": f"'{tname}'", "cat": "cat"}, METS, f"{w} AND cat IN ('Politics','Crypto')", B=BA); out.append(d)
    out = pd.concat(out); out.insert(0, "grp", grp); save(out, f"audit_F_timeaxis_{grp}.csv")
    # ---------------- G. price filter ----------------
    out = []
    filt = [("trade price (0.2,0.8) [main]", "abs(p-0.5) < 0.3"), ("midpoint (0.2,0.8)", "abs(mid_before-0.5) < 0.3"),
            ("both", "abs(p-0.5) < 0.3 AND abs(mid_before-0.5) < 0.3"), ("midpoint (0.1,0.9)", "abs(mid_before-0.5) < 0.4"),
            ("midpoint (0.05,0.95)", "abs(mid_before-0.5) < 0.45"), ("all prices", "TRUE")]
    mets = dict(METS); mets["buy_share"] = "(D=1)::int"; mets["s"] = "(I=1)::int"
    for fname, fexpr in filt:
        d = boot(con, "e", {"filter": f"'{fname}'"}, mets, f"{LATE} AND mid_before IS NOT NULL AND {fexpr}", B=BA); out.append(d)
    layer = """CASE WHEN abs(mid_before-0.5) < 0.1 THEN 'L1 |m-.5|<.1' WHEN abs(mid_before-0.5) < 0.2 THEN 'L2 <.2' WHEN abs(mid_before-0.5) < 0.3 THEN 'L3 <.3'
               WHEN abs(mid_before-0.5) < 0.4 THEN 'L4 <.4' WHEN abs(mid_before-0.5) < 0.45 THEN 'L5 <.45' ELSE 'L6 extreme' END"""
    d = boot(con, "e", {"filter": "'by midpoint layer'", "layer": layer}, mets, f"{LATE} AND mid_before IS NOT NULL", B=BA); out.append(d)
    out = pd.concat(out); out.insert(0, "grp", grp); save(out, f"audit_G_pricefilter_{grp}.csv")
    manifest(f"01_audit_{grp}")
log("done")
