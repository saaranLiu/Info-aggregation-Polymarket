"""Stage 00: full market table (all markets), on-chain resolution timestamps, outcome Y, mechanism group."""
from common import *
con = connect()
log("market aggregates from daily_aligned (full scan)")
con.execute(f"""
CREATE TABLE mkt AS
SELECT condition_id,
  any_value(category_refined) cat, any_value(market_slug) slug, any_value(neg_risk) neg_risk,
  any_value(winning_outcome_label) win_label, any_value(resolution_status) status,
  any_value(CASE WHEN outcome_seq=1 THEN outcome_label END) ref_label,
  any_value(CASE WHEN outcome_seq=2 THEN outcome_label END) other_label,
  any_value(opens_at) opens_at, any_value(close_at) close_at, any_value(resolved_at) resolved_at,
  count(*) n_fills, sum(usdc_amount) vol, count(distinct taker) n_takers, count(distinct maker) n_makers,
  min(block_timestamp) t_first, max(block_timestamp) t_last,
  arg_min(p_event, block_timestamp) p_first, arg_max(p_event, block_timestamp) p_last,
  avg((taker_base_fee>0 OR maker_base_fee>0)::int) fee_market
FROM read_parquet('{DA}') GROUP BY condition_id""")
log("block -> timestamp map from OrderFilled (full scan)")
con.execute(f"CREATE TABLE blk AS SELECT split_part(id,'_',2)::BIGINT blk, min(block_timestamp) ts FROM read_parquet('{OF}') GROUP BY 1")
con.execute(f"COPY blk TO '{RES}/block_timestamps.parquet'")
log("CTF resolutions -> timestamps")
con.execute(f"""CREATE TABLE res AS
 SELECT r.condition_id, r.blk rblk, r.payout_numerators, r.outcome_slot_count, b.ts + LEAST(r.blk - b.blk, 100000)*2.1 AS res_ts, r.blk - b.blk AS blk_gap
 FROM (SELECT condition_id, split_part(id,'_',2)::BIGINT blk, payout_numerators, outcome_slot_count FROM '{CTF}/resolutions.parquet') r
 ASOF JOIN blk b ON r.blk >= b.blk""")
con.execute("""CREATE TABLE markets AS
SELECT m.*, r.res_ts, r.blk_gap, r.payout_numerators,
  CASE WHEN m.ref_label IN ('Yes','No') AND m.other_label IN ('Yes','No') THEN 'YesNo'
       WHEN m.ref_label IN ('Up','Down') AND m.other_label IN ('Up','Down') THEN 'UpDown'
       WHEN m.ref_label IN ('Over','Under') AND m.other_label IN ('Over','Under') THEN 'OverUnder'
       WHEN m.cat='Sports' AND m.ref_label IS NOT NULL AND m.other_label IS NOT NULL
            AND m.ref_label NOT IN ('Yes','No','Over','Under','Up','Down') THEN 'SportsML'
       ELSE 'Other' END AS grp,
  (m.win_label = m.ref_label)::int AS Y_meta,
  CASE WHEN r.payout_numerators[1]='1' AND r.payout_numerators[2]='0' THEN 1
       WHEN r.payout_numerators[1]='0' AND r.payout_numerators[2]='1' THEN 0 END AS Y_ctf,
  year(to_timestamp(m.t_first)) AS yr_first
FROM mkt m LEFT JOIN res r USING(condition_id)""")
# analysis-eligible flag: resolved, binary payout, metadata winner is one of the two legs and agrees with CTF, resolution after first trade
con.execute("""ALTER TABLE markets ADD COLUMN eligible BOOLEAN;
UPDATE markets SET eligible = status='resolved' AND res_ts IS NOT NULL AND Y_ctf IS NOT NULL
   AND (win_label=ref_label OR win_label=other_label) AND Y_meta = Y_ctf AND res_ts > t_first""")
con.execute("""ALTER TABLE markets ADD COLUMN Y INTEGER; UPDATE markets SET Y = Y_ctf""")
con.execute(f"COPY markets TO '{RES}/markets.parquet'")
summ = con.sql("""SELECT grp, count(*) n_markets, sum(eligible::int) eligible, sum(n_fills) fills, round(sum(vol)/1e6) vol_musd,
   avg((Y_meta=Y_ctf)::int) FILTER (Y_ctf IS NOT NULL) meta_ctf_agree, quantile_cont(blk_gap,0.5) med_blk_gap, quantile_cont(blk_gap,0.9) p90_blk_gap
   FROM markets GROUP BY 1 ORDER BY fills DESC""").df()
print(summ.to_string(index=False)); save(summ, "00_market_summary.csv")
manifest("00_markets", n_markets=int(con.sql("SELECT count(*) FROM markets").fetchone()[0]))
log("done")
