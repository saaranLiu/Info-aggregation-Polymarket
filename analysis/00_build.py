"""build the midpoint-proxy event table used by every analysis stage.

Per trade (markets with >= 100 trades), on top of the full-pipeline event table:
  mid_before, stale, dt_since_mid       proxy midpoint = mean price of the most recent opposite-direction pair strictly before the trade
  pair fields of that proxy             dt_pair (seconds between the two trades of the pair), crossed (buy price < sell price), hs_pair
  mid_after_20/100, rn_mid_after_100    last switch midpoint at or before rn+h (NB: equals mid_before when no switch occurs in (rn, rn+h])
  ts_lead_100                           clock duration of the 100-trade horizon
  tau_close, tau_resolved               alternative time axes (metadata close_at / resolved_at instead of on-chain resolution)
  ev_cluster, res_week, cal_week        higher-level cluster ids: (topic, on-chain resolution date), resolution week, calendar week of the trade
Output: work/mid_{grp}.parquet for YesNo, SportsML, OverUnder.
"""
from study_common import *

con = connect()
con.execute(f"CREATE TABLE mk AS SELECT condition_id, slug, close_at, resolved_at, vol AS mkt_vol, n_fills AS mkt_fills FROM '{FRES}/markets.parquet' WHERE eligible")
for grp in GROUPS3:
    log(f"[{grp}] load events")
    con.execute(f"""CREATE OR REPLACE TABLE ev AS
        SELECT e.*, lead(ts, 100) OVER (PARTITION BY condition_id ORDER BY rn) AS ts_lead_100,
               lead(ts, 20) OVER (PARTITION BY condition_id ORDER BY rn) AS ts_lead_20
        FROM '{FWORK}/ev_{grp}.parquet' e WHERE mn >= {MIN_TR} AND {smoke_filter(grp)}""")
    con.execute("ALTER TABLE ev ADD COLUMN kind VARCHAR; UPDATE ev SET kind = CASE WHEN D_next IS NULL THEN NULL WHEN D = D_next THEN 'same' ELSE 'switch' END")
    # switch pairs -> midpoint proxies, with timing and consistency fields
    con.execute("""CREATE OR REPLACE TABLE mids AS
        SELECT a.condition_id, a.rn + 1 AS rn_mid, b.ts AS ts_mid, a.ts AS ts_a, b.ts - a.ts AS dt_pair,
               (a.p + b.p)/2.0 AS mid,
               CASE WHEN a.D = 1 THEN a.p ELSE b.p END AS ask_pair, CASE WHEN a.D = 1 THEN b.p ELSE a.p END AS bid_pair
        FROM ev a JOIN ev b ON a.condition_id = b.condition_id AND b.rn = a.rn + 1
        WHERE a.kind = 'switch'""")
    con.execute("ALTER TABLE mids ADD COLUMN crossed BOOLEAN; UPDATE mids SET crossed = ask_pair < bid_pair")
    con.execute("ALTER TABLE mids ADD COLUMN hs_pair DOUBLE; UPDATE mids SET hs_pair = (ask_pair - bid_pair)/2.0")
    log(f"[{grp}] asof joins")
    con.execute("""CREATE OR REPLACE TABLE e1 AS
        SELECT e.*, m.mid AS mid_before, e.rn - m.rn_mid AS stale, e.ts - m.ts_mid AS dt_since_mid,
               m.dt_pair AS dt_pair_before, m.crossed AS crossed_before, m.hs_pair AS hs_pair_before
        FROM ev e ASOF LEFT JOIN mids m ON m.condition_id = e.condition_id AND m.rn_mid <= e.rn - 1""")
    con.execute("""CREATE OR REPLACE TABLE e2 AS SELECT e.*, m.mid AS mid_after_20, m.rn_mid AS rn_mid_after_20 FROM e1 e
        ASOF LEFT JOIN mids m ON m.condition_id = e.condition_id AND m.rn_mid <= e.rn + 20""")
    con.execute("""CREATE OR REPLACE TABLE e3 AS SELECT e.*, m.mid AS mid_after_100, m.rn_mid AS rn_mid_after_100, m.ts_mid AS ts_mid_after_100 FROM e2 e
        ASOF LEFT JOIN mids m ON m.condition_id = e.condition_id AND m.rn_mid <= e.rn + 100""")
    # last midpoint of the market (for the "PI_clk = info by construction" audit)
    con.execute("""CREATE OR REPLACE TABLE lastmid AS SELECT condition_id, arg_max(mid, rn_mid) mid_last, max(rn_mid) rn_mid_last FROM mids GROUP BY 1""")
    con.execute(f"""CREATE OR REPLACE TABLE e4 AS
        SELECT e.*, l.mid_last, l.rn_mid_last,
               epoch(mk.close_at) - e.ts AS tau_close, epoch(mk.resolved_at) - e.ts AS tau_resolved, mk.slug, mk.mkt_vol, mk.mkt_fills,
               e.cat || '|' || strftime(to_timestamp(e.res_ts), '%Y-%m-%d') AS ev_cluster,
               strftime(date_trunc('week', to_timestamp(e.res_ts)), '%Y-%m-%d') AS res_week,
               strftime(date_trunc('week', to_timestamp(e.ts)), '%Y-%m-%d') AS cal_week,
               {SIZE3} AS size3, {SIZE_B} AS size_b, {TAU_COARSE} AS tau_b
        FROM e3 e LEFT JOIN lastmid l USING(condition_id) LEFT JOIN mk USING(condition_id)""")
    con.execute(f"COPY e4 TO '{WORK}/mid_{grp}.parquet'")
    n = con.sql(f"SELECT count(*), count(distinct condition_id), avg((mid_before IS NOT NULL)::int), avg(crossed_before::int), count(distinct ev_cluster) FROM e4").fetchone()
    log(f"[{grp}] rows={n[0]:,} markets={n[1]:,} mid_coverage={n[2]:.3f} crossed_share={n[3]:.3f} ev_clusters={n[4]:,}")
    manifest(f"00_build_{grp}", rows=n[0], markets=n[1], mid_coverage=n[2], crossed_share=n[3], ev_clusters=n[4])
log("done")
