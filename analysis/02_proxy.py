"""midpoint-proxy diagnostics .

Without quotes we cannot compute e_i = m_proxy - m_true.  We report what the trade tape itself can say about the proxy:
  A  internal-consistency statistics: sign violations D(p - m) < 0 (a buy below / sell above the proxy midpoint), crossed switch pairs
     (buy price < sell price inside the pair, impossible in one book state), pair half-spread, pair duration, proxy age
  B  sensitivity of info / eff / RS / PI_100 / over_100 to proxy construction: full, fresh, time-tight pairs, uncrossed pairs, sign-consistent
  C  pre-trade momentum relative to the proxy by staleness (sign of the expected proxy bias in info and eff)
  D  bound on construction error: |m_proxy - m_true| <= half pair spread if the book did not move inside the pair
Outputs: results/proxy_A_consistency_{grp}.csv, proxy_B_variants_{grp}.csv, proxy_C_momentum_{grp}.csv, proxy_D_bound_{grp}.csv
"""
from study_common import *
con = connect()
BP = 500 if not SMOKE else 100
VARIANTS = [("full (main)", "TRUE"), ("fresh (stale<=1)", "stale <= 1"),
            ("pair within 60s", "dt_pair_before <= 60"), ("pair within 60s and proxy age <= 10 min", "dt_pair_before <= 60 AND dt_since_mid <= 600"),
            ("uncrossed pair", "NOT crossed_before"), ("uncrossed and fresh", "NOT crossed_before AND stale <= 1"),
            ("sign-consistent D(p-m)>=0", "D*(p-mid_before) >= 0"), ("stale<=1 and pair within 10s", "stale <= 1 AND dt_pair_before <= 10")]
for grp in GROUPS3:
    log(f"[{grp}]")
    mid_table(con, grp)
    W = f"{LATE} AND {BASE}"
    cells = [("pooled", "'all'", "TRUE")] + ([("cat", "cat", "cat IN ('Politics','Crypto')"), ("cat_size", "cat || ' | ' || size3", "cat IN ('Politics','Crypto')")] if grp == "YesNo" else [("size", "size3", "TRUE")])
    # ---- A consistency
    out = []
    for cname, cexpr, cw in cells:
        d = con.sql(f"""SELECT '{grp}' grp, '{cname}' cell_type, {cexpr} cell, count(*) n, count(distinct condition_id) n_markets,
            avg((D*(p-mid_before) < 0)::int) share_sign_violation, avg((D*(p-mid_before) < -0.005)::int) share_sign_violation_gt_halftick,
            avg(greatest(0, -D*(p-mid_before)))*1e4 mean_violation_bp, avg(crossed_before::int) share_crossed_pair,
            avg(hs_pair_before)*1e4 mean_pair_halfspread_bp, quantile_cont(hs_pair_before, 0.5)*1e4 med_pair_halfspread_bp,
            quantile_cont(dt_pair_before, [0.5, 0.9]) q_pair_seconds, avg((dt_pair_before <= 60)::int) share_pair_within_60s,
            quantile_cont(dt_since_mid, [0.5, 0.9]) q_proxy_age_seconds, quantile_cont(stale, [0.5, 0.9]) q_stale,
            avg((stale <= 1)::int) share_fresh, avg(abs(p - mid_before))*1e4 mean_abs_p_minus_m_bp
            FROM e WHERE {W} AND {cw} GROUP BY ALL ORDER BY n DESC""").df()
        out.append(d)
    save(pd.concat(out), f"proxy_A_consistency_{grp}.csv")
    # ---- B variants
    out = []
    for vname, vexpr in VARIANTS:
        for cname, cexpr, cw in cells:
            d = boot(con, "e", {"variant": f"'{vname}'", "cell_type": f"'{cname}'", "cell": cexpr}, METS, f"{W} AND {cw} AND {vexpr}", B=BP)
            out.append(d)
    out = pd.concat(out); out.insert(0, "grp", grp); save(out, f"proxy_B_variants_{grp}.csv")
    # ---- C pre-trade momentum relative to the proxy, by staleness
    stale_b = "CASE WHEN stale<=1 THEN 'a fresh(1)' WHEN stale<=5 THEN 'b 2-5' WHEN stale<=20 THEN 'c 6-20' ELSE 'd >20' END"
    d = con.sql(f"""SELECT '{grp}' grp, {stale_b} stale_b, count(*) n, avg(D*(p_lag_1 - mid_before))*1e4 D_x_prevprice_minus_m_bp,
        avg(D*(p - p_lag_1))*1e4 pre1_bp, avg(D*(p - p_lag_5))*1e4 pre5_bp, avg(D*(p - p_lag_20))*1e4 pre20_bp,
        avg(D*(Y-mid_before))*1e4 info_bp, avg(D*(p-mid_before))*1e4 eff_bp, avg(-edge)*1e4 RS_bp, avg(D*(mid_after_100-Y))*1e4 over100_bp,
        avg(abs(p - mid_before))*1e4 mean_abs_p_minus_m_bp
        FROM e WHERE {W} GROUP BY ALL ORDER BY 2""").df()
    save(d, f"proxy_C_momentum_{grp}.csv")
    # ---- D construction-error bound (pair half-spread) and its relation to the effect sizes
    d = con.sql(f"""SELECT '{grp}' grp, {"cat" if grp == "YesNo" else "'all'"} cell, size3, count(*) n,
        avg(hs_pair_before)*1e4 bound_mean_bp, quantile_cont(hs_pair_before, 0.5)*1e4 bound_q50_bp, quantile_cont(hs_pair_before, 0.9)*1e4 bound_q90_bp,
        avg(D*(Y-mid_before))*1e4 info_bp, avg(D*(mid_after_100-Y))*1e4 over100_bp, avg(D*(p-mid_before))*1e4 eff_bp,
        avg(hs_pair_before) FILTER (crossed_before)*1e4 crossed_depth_bp, avg(crossed_before::int) share_crossed
        FROM e WHERE {W} GROUP BY ALL ORDER BY 2,3""").df()
    save(d, f"proxy_D_bound_{grp}.csv")
    manifest(f"02_proxy_{grp}")
log("done")
