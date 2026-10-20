"""Stage 02: event tables (per trade: forward/backward prices at trade-count and clock horizons) + market features."""
from common import *
con = connect()
con.execute(f"CREATE TABLE markets AS SELECT * FROM '{RES}/markets.parquet' WHERE eligible")
for grp in GROUPS:
    log(f"[{grp}] event table")
    hs = TRADE_H if grp != 'UpDown' else [1]
    pre = PRE_H if grp != 'UpDown' else [1]
    leads = ", ".join([f"lead(p,{h}) OVER w AS p_lead_{h}" for h in hs] + [f"lag(p,{h}) OVER w AS p_lag_{h}" for h in pre])
    con.execute(f"""CREATE OR REPLACE TABLE ev AS
      SELECT * , {leads}, lag(D,1) OVER w AS D_lag, lead(D,1) OVER w AS D_next, count(*) OVER (PARTITION BY condition_id) AS mn,
             row_number() OVER w AS rn
      FROM (
        SELECT t.condition_id, t.ts, t.taker, t.D, t.p, t.usd, t.nfills, t.self_fill, t.fee_pos,
               m.Y, m.cat, m.yr_first, m.res_ts, m.t_first,
               m.res_ts - t.ts AS tau, (m.res_ts - t.ts)/(m.res_ts - m.t_first) AS theta,
               m.Y*t.p + (1-m.Y)*(1-t.p) AS q, (2*m.Y-1)*t.D AS I, t.D*(m.Y - t.p) AS edge,
               year(to_timestamp(t.ts)) AS yr
        FROM '{WORK}/trades_{grp}.parquet' t JOIN markets m USING(condition_id)
        WHERE t.ts < m.res_ts)
      WINDOW w AS (PARTITION BY condition_id ORDER BY ts, hash(taker, p, usd))""")
    # clock horizons via block-level VWAP price
    con.execute("CREATE OR REPLACE TABLE px AS SELECT condition_id, ts, sum(p*usd)/sum(usd) pb FROM ev GROUP BY 1,2")
    for H in CLOCK_H[grp]:
        con.execute(f"""CREATE OR REPLACE TABLE ev2 AS SELECT e.*, x.pb AS p_clk_{H}
          FROM ev e ASOF LEFT JOIN px x ON x.condition_id = e.condition_id AND x.ts <= e.ts + {H}""")
        con.execute("DROP TABLE ev; ALTER TABLE ev2 RENAME TO ev")
    con.execute(f"""CREATE OR REPLACE TABLE ev2 AS SELECT e.*, x.pb AS p_pre_clk
          FROM ev e ASOF LEFT JOIN px x ON x.condition_id = e.condition_id AND x.ts <= e.ts - {PRE_CLOCK[grp]}""")
    con.execute("DROP TABLE ev; ALTER TABLE ev2 RENAME TO ev")
    con.execute(f"COPY ev TO '{WORK}/ev_{grp}.parquet'")
    n = con.sql("SELECT count(*), count(distinct condition_id), avg((mn>=100)::int) FROM ev").fetchone()
    log(f"[{grp}] events={n[0]:,} markets={n[1]:,}")
    # market features
    log(f"[{grp}] market features")
    con.execute(f"""CREATE OR REPLACE TABLE mf AS
      SELECT e.condition_id, any_value(cat) cat, any_value(yr_first) yr_first, any_value(Y) Y,
        count(*) n, sum(usd) vol, count(distinct taker) ntak, (max(ts)-min(ts))/3600.0 dur_h,
        count(*)/greatest((max(ts)-min(ts))/3600.0, 1) tph,
        avg(abs(p_lead_1 - p)) absdp1,
        2*sqrt(greatest(-covar_pop(p_lead_1 - p, p - p_lag_1), 0)) roll,
        corr(D, D_lag) dac,
        avg((I=1)::int) inf_share, sum((I=1)::int*usd)/sum(usd) inf_share_usd,
        1e4*avg(edge) edge_bp, 1e4*sum(edge*usd)/sum(usd) edge_usdw_bp,
        1e4*avg(D*(p_lead_1-p)) R1_bp,
        {"1e4*avg(D*(p_lead_20-p)) R20_bp, 1e4*avg(D*(p_lead_100-p)) R100_bp," if grp!='UpDown' else "NULL R20_bp, NULL R100_bp,"}
        1e4*avg(D*(p_clk_{CLOCK_H[grp][-1]}-p)) Rclk_last_bp,
        sum(D*usd)/sum(usd) imb_usd, avg(D) imb_cnt,
        arg_min(q, ts) q_first, arg_max(q, ts) q_last,
        arg_max(q, ts) FILTER (theta > 0.5) q_mid, arg_max(q, ts) FILTER (tau > 86400) q_24h, arg_max(q, ts) FILTER (tau > 3600) q_1h,
        arg_max(p, ts) FILTER (theta > 0.5) p_mid, arg_max(p, ts) FILTER (tau > 86400) p_24h,
        sum(D*usd) FILTER (theta > 0.5)/sum(usd) FILTER (theta > 0.5) imb_usd_firsthalf, avg(D) FILTER (theta > 0.5) imb_cnt_firsthalf,
        sum(D*usd) FILTER (theta > 0.5 AND usd >= 1000)/nullif(sum(usd) FILTER (theta > 0.5 AND usd >= 1000),0) imb_large_firsthalf,
        sum(D*usd) FILTER (theta > 0.5 AND usd < 100)/nullif(sum(usd) FILTER (theta > 0.5 AND usd < 100),0) imb_small_firsthalf,
        sum(D*usd) FILTER (tau > 86400)/nullif(sum(usd) FILTER (tau > 86400),0) imb_usd_pre24h,
        sum(D*usd) FILTER (tau > 86400 AND usd >= 1000)/nullif(sum(usd) FILTER (tau > 86400 AND usd >= 1000),0) imb_large_pre24h,
        sum(D*usd) FILTER (tau > 86400 AND usd < 100)/nullif(sum(usd) FILTER (tau > 86400 AND usd < 100),0) imb_small_pre24h,
        count(distinct taker) FILTER (theta > 0.5) ntak_firsthalf, count(distinct taker) FILTER (theta <= 0.5 AND tau > 86400) ntak_secondhalf_pre24h,
        avg(self_fill::int) self_fill_share, avg(fee_pos::int) fee_pos_share
      FROM ev e GROUP BY 1""")
    con.execute("""CREATE OR REPLACE TABLE hhi AS SELECT condition_id, sum(tv*tv)/power(sum(tv),2) taker_hhi FROM (SELECT condition_id, taker, sum(usd) tv FROM ev GROUP BY 1,2) GROUP BY 1""")
    con.execute(f"""COPY (SELECT mf.*, h.taker_hhi, mm.maker_hhi, mm.n_makers, mm.maker_edge, mm.maker_edge_usdw, mm.self_share_usd, mm.fee_share_usd
                          FROM mf JOIN hhi h USING(condition_id) LEFT JOIN '{RES}/maker_market_stats_{grp}.parquet' mm USING(condition_id))
                    TO '{RES}/market_features_{grp}.parquet'""")
    manifest(f"02_events_{grp}", events=n[0], markets=n[1], share_markets_ge100=n[2])
log("done")
