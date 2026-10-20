"""Trade price aggregation: USD-weighted fill price versus share-weighted (quantity) execution price.

A trade merges fills sharing market, block, taker and direction. The baseline price is
p_usd = sum(u_j p_j) / sum(u_j) on the event axis. The quantity-weighted execution price uses the
actual number of outcome tokens in each fill, s_j = u_j / price_j (price_j is the raw token price),
p_sh = sum(s_j p_j) / sum(s_j). Both are recomputed from fills for final-day Yes/No trades.

Outputs (results/):
  fp_A_price_gap.csv     distribution of p_sh - p_usd (all trades; multi-fill trades) by topic and size
  fp_B_metrics.csv       eff / RS on the main sample and on the observed-horizon common sample, under
                         p_usd and p_sh, for count, USD, actual-share and constructed u/p weights
"""
from study_common import *

con = connect()
DA = C.DA
log("markets and fills")
con.execute(f"""CREATE OR REPLACE TABLE mk AS SELECT condition_id, any_value(res_ts)::DOUBLE res_ts FROM '{WORK}/mid_YesNo.parquet'
    WHERE {LATE} AND {BASE} GROUP BY 1""")
con.execute(f"""CREATE OR REPLACE TABLE f AS
    SELECT t.condition_id, t.block_timestamp ts, t.taker, t.D, t.p_event pe, t.price praw, t.usdc_amount u
    FROM read_parquet('{DA}') t JOIN mk USING(condition_id)
    WHERE t.p_event IS NOT NULL AND t.D IS NOT NULL AND t.usdc_amount > 0 AND t.price > 0
      AND t.block_timestamp >= mk.res_ts - 86400 AND t.block_timestamp < mk.res_ts - 3600""")
con.execute("""CREATE OR REPLACE TABLE tr AS SELECT condition_id, ts, taker, D, sum(u*pe)/sum(u) p_usd, sum(u/praw*pe)/sum(u/praw) p_sh,
    sum(u/praw) shares_actual, sum(u) usd_f, count(*) k FROM f GROUP BY 1,2,3,4""")
con.execute(f"""CREATE OR REPLACE TABLE j AS SELECT e.condition_id, e.cat, e.size3, e.ev_cluster, e.D, e.Y, e.p, e.usd, e.mid_before, e.mid_after_100,
    e.ts_lead_100, t.p_usd, t.p_sh, t.shares_actual, t.k
    FROM '{WORK}/mid_YesNo.parquet' e JOIN tr t USING(condition_id, ts, taker, D) WHERE {LATE} AND {BASE}""")
chk = con.sql("SELECT count(*), max(abs(p - p_usd)), avg((k>1)::int), (SELECT count(*) FROM '" + WORK + f"/mid_YesNo.parquet' WHERE {LATE} AND {BASE}) FROM j").fetchone()
log(f"joined {chk[0]:,} of {chk[3]:,} main trades; max |p - p_usd| = {chk[1]:.2e}; multi-fill share {chk[2]:.3f}")

gap = con.sql("""SELECT coalesce(cat,'all') cat, coalesce(size3,'all') size3, count(*) n, avg((k>1)::int) share_multifill,
        1e4*avg(D*(p_sh-p_usd)) mean_signed_gap_bp, 1e4*avg(abs(p_sh-p_usd)) mean_abs_gap_bp,
        1e4*avg(abs(p_sh-p_usd)) FILTER (WHERE k>1) mean_abs_gap_multifill_bp,
        1e4*quantile_cont(abs(p_sh-p_usd), 0.99) p99_abs_gap_bp, avg((abs(p_sh-p_usd) >= 0.005)::int) share_gap_ge_half_cent
    FROM j WHERE cat IN ('Politics','Crypto') OR TRUE GROUP BY GROUPING SETS ((), (cat), (cat, size3)) ORDER BY 1,2""").df()
gap = gap[gap.cat.isin(["all", "Politics", "Crypto"])]
save(gap, "fp_A_price_gap.csv")

rows = []
for sname, sw in [("main", "TRUE"), ("observed_h100", "ts_lead_100 IS NOT NULL AND mid_after_100 IS NOT NULL")]:
    for cname, cw in [("pooled", "TRUE"), ("Politics", "cat='Politics'"), ("Crypto", "cat='Crypto'"),
                      ("Politics | s1 <$100", "cat='Politics' AND size3='s1 <$100'"), ("Politics | s3 >=$1k", "cat='Politics' AND size3='s3 >=$1k'")]:
        for wname, wexpr in [("trade", "1.0"), ("usd", "usd"), ("shares_actual", "shares_actual"), ("u_over_p", "usd/p")]:
            for pname in ["p_usd", "p_sh"]:
                r = con.sql(f"""SELECT 1e4*sum({wexpr}*D*({pname}-mid_before))/sum({wexpr}) eff, 1e4*sum({wexpr}*D*({pname}-Y))/sum({wexpr}) RS,
                        1e4*sum({wexpr}*D*(Y-mid_before))/sum({wexpr}) info, count(*) n FROM j WHERE {sw} AND {cw}""").fetchone()
                rows.append(dict(sample=sname, cell=cname, weight=wname, price=pname, eff_bp=r[0], RS_bp=r[1], info_bp=r[2], n=r[3]))
B = pd.DataFrame(rows)
# event-cluster SE of the per-trade RS difference (p_sh - p_usd) in each cell
for sname, sw in [("main", "TRUE"), ("observed_h100", "ts_lead_100 IS NOT NULL AND mid_after_100 IS NOT NULL")]:
    for cname, cw in [("pooled", "TRUE"), ("Politics", "cat='Politics'"), ("Crypto", "cat='Crypto'"),
                      ("Politics | s1 <$100", "cat='Politics' AND size3='s1 <$100'"), ("Politics | s3 >=$1k", "cat='Politics' AND size3='s3 >=$1k'")]:
        d = con.sql(f"SELECT ev_cluster, D*(p_sh-p_usd) x FROM j WHERE {sw} AND {cw}").df()
        se, G = cr_se_mean(d.x.to_numpy(), d.ev_cluster.to_numpy())
        B.loc[(B["sample"] == sname) & (B.cell == cname), "diff_trade_bp"] = 1e4 * d.x.mean()
        B.loc[(B["sample"] == sname) & (B.cell == cname), "diff_trade_se_ev_bp"] = 1e4 * se
save(B, "fp_B_metrics.csv")

# ---- C: replace the constructed u/p weight in the common-sample weight audit by actual fill quantities
import os
src = f"{RES}/audit_C_weights_YesNo.csv"; keep = f"{RES}/audit_C_weights_YesNo_constructed_u_over_p.csv"
if not os.path.exists(keep): os.replace(src, keep)
old = pd.read_csv(keep)
con.execute(f"""CREATE OR REPLACE TABLE cs AS SELECT e.*, t.shares_actual AS shares FROM '{WORK}/mid_YesNo.parquet' e JOIN tr t USING(condition_id, ts, taker, D)
    WHERE {LATE} AND {BASE} AND e.p_lead_100 IS NOT NULL AND e.mid_after_100 IS NOT NULL""")
mets = dict(METS); mets["R_100"] = "(D*(p_lead_100-p))"; mets["resid_100"] = "(D*(p_lead_100-mid_after_100))"; mets["s"] = "(I=1)::int"
new = [boot(con, "cs", {"weight": "'shares'"}, mets, "TRUE", weight="shares", B=300),
       boot(con, "cs", {"weight": "'shares'", "cat": "cat"}, mets, "cat IN ('Politics','Crypto')", weight="shares", B=300)]
new = pd.concat(new); new.insert(0, "grp", "YesNo")
out = pd.concat([old[old.weight != "shares"], new], ignore_index=True)
pooled = out[out["cat"].isna()]
piv = pooled.pivot_table(index="weight", columns="metric", values="est")
piv["closure_eff"] = (piv["eff"] - piv["info"] - piv["RS"]) * 1e4; piv["closure_PI"] = (piv["PI_100"] - piv["info"] - piv["over_100"]) * 1e4
piv["closure_R"] = (piv["R_100"] - piv["over_100"] + piv["RS"] - piv["resid_100"]) * 1e4
out = out.drop(columns=["closure_eff", "closure_PI", "closure_R"]).merge(piv[["closure_eff", "closure_PI", "closure_R"]].reset_index(), on="weight", how="left")
out["weight_definition"] = out.weight.map({"count": "one per trade", "equal_market": "1/n_market", "usd": "USD notional",
                                           "shares": "actual outcome-token quantity, sum_j u_j/price_j over fills"})
save(out[old.columns.tolist() + ["weight_definition"]], "audit_C_weights_YesNo.csv")
manifest("14_fill_prices", joined=chk[0], main=chk[3], max_abs_p_check=chk[1])
log("done")
