"""Stage 01: taker-level trade tables per mechanism group (all eligible markets, no sampling) + maker/market fill statistics."""
from common import *
con = connect()
con.execute(f"CREATE TABLE markets AS SELECT condition_id, grp, Y, res_ts, t_first, cat, n_fills FROM '{RES}/markets.parquet' WHERE eligible")
for grp in GROUPS:
    log(f"[{grp}] fills -> trades")
    con.execute(f"""CREATE OR REPLACE TABLE fills AS
      SELECT t.condition_id, t.block_timestamp ts, t.taker, t.maker, t.D, t.p_event p, t.usdc_amount usd, t.fee_usdc fee, m.Y, m.res_ts
      FROM read_parquet('{DA}') t JOIN markets m USING(condition_id)
      WHERE m.grp='{grp}' AND t.p_event IS NOT NULL AND t.D IS NOT NULL AND t.usdc_amount > 0""")
    con.execute(f"""CREATE OR REPLACE TABLE trades AS
      SELECT condition_id, ts, taker, D, sum(p*usd)/sum(usd) p, sum(usd) usd, count(*) nfills,
             max((maker = taker)::int)::boolean self_fill, max((fee > 0)::int)::boolean fee_pos, count(distinct maker) n_makers_hit
      FROM fills GROUP BY 1,2,3,4""")
    con.execute(f"COPY trades TO '{WORK}/trades_{grp}.parquet'")
    # maker-side market statistics (fill level): maker edge = -(D*(Y-p)), concentration, self-trade share, maker count
    con.execute(f"""CREATE OR REPLACE TABLE mm AS
      WITH ms AS (SELECT condition_id, maker, sum(usd) mv FROM fills GROUP BY 1,2),
           mh AS (SELECT condition_id, sum(mv*mv)/power(sum(mv),2) maker_hhi, count(*) n_makers FROM ms GROUP BY 1),
           fs AS (SELECT condition_id, count(*) n_fills, count(distinct taker) n_takers,
                    sum(-D*(Y-p)*usd)/sum(usd) maker_edge_usdw, avg(-D*(Y-p)) maker_edge,
                    sum((maker=taker)::int*usd)/sum(usd) self_share_usd, sum((fee>0)::int*usd)/sum(usd) fee_share_usd
                  FROM fills GROUP BY 1)
      SELECT fs.*, mh.maker_hhi, mh.n_makers FROM fs JOIN mh USING(condition_id)""")
    con.execute(f"COPY mm TO '{RES}/maker_market_stats_{grp}.parquet'")
    n = con.sql("SELECT count(*), sum(nfills), count(distinct condition_id) FROM trades").fetchone()
    log(f"[{grp}] trades={n[0]:,} fills={n[1]:,} markets={n[2]:,}")
    manifest(f"01_trades_{grp}", trades=n[0], fills=n[1], markets=n[2])
log("done")
