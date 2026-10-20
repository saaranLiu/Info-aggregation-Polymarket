"""Validate the transaction-derived midpoint against observed CLOB quotes.

Quotes: exchange-emitted best bid/ask from the pmxt order-book archive (fetched by 13a_fetch_quotes.py),
mapped to the reference-outcome axis (outcome_seq = 1; the complementary token's book is mirrored as
bid = 1 - ask, ask = 1 - bid). Quotes are collapsed to the last event in each second.

Alignment: on-chain block timestamps follow the off-chain CLOB match. For lead L seconds we take the
latest quote at or before ts - L and record how often the execution price equals the touched side
(buy at the ask, sell at the bid). The lead maximising this match rate defines the prevailing quote.

Validation sample: main-sample Yes/No trades (final day, interior price, proxy defined) in fetched markets,
with a two-sided, uncrossed prevailing quote no older than 15 minutes (pmxt v1 emits a full book
snapshot about every 10 minutes per token, so older quotes indicate an archive gap).

Outputs (results/):
  qv_A_alignment.csv      match rates by lead L
  qv_B_error.csv          proxy error e = m_proxy - m_quote: distribution and mean D*e by topic, size, freshness
  qv_C_reg.csv            regression of D*e on size, topic, freshness and spread (event-cluster SEs)
  qv_D_metrics.csv        eff/info/RS/PI/over computed with proxy and with quote midpoints on identical trades,
                          their differences, and event-cluster SEs; trade-count and clock horizons; cutoffs R and S
  qv_E_wb.csv             within/between decomposition with quote midpoints
  qv_F_coverage.csv       coverage of target markets and trades
"""
import glob
from study_common import *
import math
del C

QDIR = f"{PROJECT}/work/quotes"
LEADS = [0, 1, 2, 3, 4, 5, 6, 8, 10, 15, 20, 30, 60]
HS = [600, 1800, 3600, 7200]
MAX_AGE = 900
con = connect()
import datetime as dt
D0 = dt.date(2026, 2, 22); DAYS = [(D0 + dt.timedelta(days=k)).isoformat() for k in range(66)]
SAMPLE = os.environ.get("QUOTE_SAMPLE", "every3")       # systematic sample: every third archive day from 22 Feb 2026
keep = set(DAYS[0::3]) if SAMPLE == "every3" else set(DAYS)
files = sorted(f for f in glob.glob(f"{QDIR}/raw/date=*.parquet") if f.split("date=")[1][:10] in keep)
log(f"{len(files)} quote files")
save(pd.DataFrame({"day": [f.split("date=")[1][:10] for f in files]}), "qv_G_days.csv")
flist = ",".join(f"'{f}'" for f in files)
con.execute(f"""CREATE OR REPLACE TABLE q0 AS
    SELECT r.condition_id, r.event_ts::BIGINT ts, r.rg, r.row, r.event_type, t.outcome_seq,
           CASE WHEN t.outcome_seq = 1 THEN r.best_bid ELSE 1 - r.best_ask END AS bid,
           CASE WHEN t.outcome_seq = 1 THEN r.best_ask ELSE 1 - r.best_bid END AS ask,
           regexp_extract(r.filename, 'date=([0-9-]+)', 1) AS fday
    FROM read_parquet([{flist}], filename=true, union_by_name=true) r
    JOIN '{QDIR}/token_map.parquet' t ON t.condition_id = r.condition_id AND t.asset_id = r.asset_id
    WHERE r.best_bid IS NOT NULL AND r.best_ask IS NOT NULL""")
con.execute("""CREATE OR REPLACE TABLE q AS
    SELECT condition_id, ts, arg_max(bid, (fday, rg, row)) bid, arg_max(ask, (fday, rg, row)) ask
    FROM q0 GROUP BY 1, 2""")
con.execute("ALTER TABLE q ADD COLUMN mid DOUBLE; UPDATE q SET mid = (bid + ask)/2")
con.execute("ALTER TABLE q ADD COLUMN ok BOOLEAN; UPDATE q SET ok = bid > 0 AND ask < 1 AND bid < ask")
nq = con.sql("SELECT count(*), count(DISTINCT condition_id), avg(ok::int) FROM q").fetchone()
log(f"quotes: {nq[0]:,} seconds, {nq[1]} markets, valid share {nq[2]:.3f}")

con.execute(f"""CREATE OR REPLACE TABLE ev AS SELECT e.condition_id, e.rn, e.ts, e.D, e.p, e.usd, e.Y, e.cat, e.res_ts::DOUBLE res_ts,
        e.tau::DOUBLE tau, e.tau_close, e.mid_before, e.stale, e.dt_since_mid, e.crossed_before, e.ev_cluster, e.size3,
        e.ts_lead_100, e.mid_after_100, e.kind
    FROM '{WORK}/mid_YesNo.parquet' e WHERE e.condition_id IN (SELECT DISTINCT condition_id FROM q)""")
DSEC = ",".join(str(int(dt.datetime.fromisoformat(f.split("date=")[1][:10]).replace(tzinfo=dt.timezone.utc).timestamp())) for f in files)
# a trade is validated only on a downloaded UTC day, and every quote-based horizon must end before that day ends
DAYSQL = f"(floor(ts / 86400) * 86400)::BIGINT IN ({DSEC})"
con.execute(f"""CREATE OR REPLACE TABLE m AS SELECT *, (floor(ts / 86400) + 1) * 86400 AS dhi FROM ev WHERE {LATE} AND {BASE} AND {DAYSQL}""")

# ---- A: alignment
rows = []
for L in LEADS:
    r = con.sql(f"""WITH j AS (SELECT m.D, m.p, x.bid, x.ask, x.ok, m.ts - {L} - x.ts AS age FROM m
            ASOF LEFT JOIN q x ON x.condition_id = m.condition_id AND x.ts <= m.ts - {L})
        SELECT {L} lead_s, count(*) n, avg((ok AND age <= {MAX_AGE})::int) share_valid,
            avg((abs(p - CASE WHEN D = 1 THEN ask ELSE bid END) < 0.0005)::int) FILTER (WHERE ok AND age <= {MAX_AGE}) share_exact_side,
            avg((D*(p - (bid+ask)/2) > 0)::int) FILTER (WHERE ok AND age <= {MAX_AGE}) share_cost_positive,
            avg((p > bid + 1e-9 AND p < ask - 1e-9)::int) FILTER (WHERE ok AND age <= {MAX_AGE}) share_inside,
            avg(abs(p - CASE WHEN D = 1 THEN ask ELSE bid END)) FILTER (WHERE ok AND age <= {MAX_AGE}) mean_abs_side_gap
        FROM j""").df()
    rows.append(r)
A = pd.concat(rows); save(A, "qv_A_alignment.csv")
LSTAR = int(A.sort_values(["share_exact_side", "lead_s"], ascending=[False, True]).iloc[0].lead_s)
log(f"lead with maximum side match: {LSTAR}s")

# ---- prevailing and post-trade quote midpoints
con.execute(f"""CREATE OR REPLACE TABLE v0 AS SELECT m.*, x.mid AS mq, x.bid AS bq, x.ask AS aq, x.ok AS okq, m.ts - {LSTAR} - x.ts AS age_q FROM m
    ASOF LEFT JOIN q x ON x.condition_id = m.condition_id AND x.ts <= m.ts - {LSTAR}""")
for H in HS:
    con.execute(f"""CREATE OR REPLACE TABLE v1 AS SELECT v.*, x.mid AS mq_c{H}, (x.ok AND v.ts + {H} - x.ts <= {MAX_AGE} AND v.ts + {H} < v.dhi) AS okq_c{H} FROM v0 v
        ASOF LEFT JOIN q x ON x.condition_id = v.condition_id AND x.ts <= v.ts + {H}""")
    con.execute("DROP TABLE v0; ALTER TABLE v1 RENAME TO v0")
con.execute(f"""CREATE OR REPLACE TABLE v1 AS SELECT v.*, x.mid AS mq_h100, (x.ok AND v.ts_lead_100 - x.ts <= {MAX_AGE} AND v.ts_lead_100 < v.dhi) AS okq_h100 FROM v0 v
    ASOF LEFT JOIN q x ON x.condition_id = v.condition_id AND x.ts <= v.ts_lead_100""")
# proxy midpoints at clock horizons on the same trades
con.execute("""CREATE OR REPLACE TABLE mids AS SELECT condition_id, ts_mid, arg_max(rn_mid, rn_mid) AS rn_mid, arg_max(mid, rn_mid) AS mid FROM (
    SELECT a.condition_id, b.ts AS ts_mid, b.rn AS rn_mid, (a.p + b.p)/2.0 AS mid
    FROM ev a JOIN ev b ON a.condition_id = b.condition_id AND b.rn = a.rn + 1 WHERE a.kind = 'switch') GROUP BY 1, 2""")
con.execute("DROP TABLE v0; ALTER TABLE v1 RENAME TO v0")
for H in HS:
    con.execute(f"""CREATE OR REPLACE TABLE v1 AS SELECT v.*, x.mid AS mp_c{H} FROM v0 v
        ASOF LEFT JOIN mids x ON x.condition_id = v.condition_id AND x.ts_mid <= v.ts + {H}""")
    con.execute("DROP TABLE v0; ALTER TABLE v1 RENAME TO v0")
con.execute(f"""CREATE OR REPLACE TABLE v AS SELECT v0.*, least(res_ts, coalesce(ts + tau_close, res_ts)) AS S,
        mid_before - mq AS e, (aq - bq) AS spread_q,
        CASE WHEN stale <= 1 THEN 'fresh' WHEN stale <= 5 THEN 'stale 2-5' ELSE 'stale >5' END AS fresh_b
    FROM v0 WHERE okq AND age_q <= {MAX_AGE}""")

# ---- F: coverage
tg = pd.read_parquet(f"{QDIR}/targets.parquet")
cov = con.sql(f"""SELECT cat, count(DISTINCT condition_id) markets_with_quotes, count(*) trades_validation FROM v GROUP BY 1""").df()
mm = con.sql("SELECT cat, count(DISTINCT condition_id) markets_fetched, count(*) trades_main_fetched FROM m GROUP BY 1").df()
alld = con.sql(f"""SELECT cat, count(*) trades_main_on_days FROM '{WORK}/mid_YesNo.parquet' WHERE {LATE} AND {BASE}
    AND cat IN ('Politics', 'Crypto') AND {DAYSQL} GROUP BY 1""").df()
mm = mm.merge(alld, how="outer")
cov = tg.groupby("cat").agg(target_markets=("condition_id", "size"), target_trades=("n_main", "sum")).reset_index().merge(mm, how="left").merge(cov, how="left")
save(cov, "qv_F_coverage.csv")

# ---- H: representativeness: proxy estimates on sampled days, markets the archive records versus markets it omits
con.execute(f"""CREATE OR REPLACE TABLE allday AS SELECT condition_id, ev_cluster, cat, size3, D*(Y - mid_before) AS info, D*(p - mid_before) AS eff,
        D*(mid_after_100 - Y) AS over_100, condition_id IN (SELECT DISTINCT condition_id FROM q) AS recorded
    FROM '{WORK}/mid_YesNo.parquet' WHERE {LATE} AND {BASE} AND cat IN ('Politics', 'Crypto') AND {DAYSQL} AND mid_after_100 IS NOT NULL""")
hrows = []
for cname, w in [("Politics", "cat='Politics'"), ("Crypto", "cat='Crypto'"), ("Politics | s1 <$100", "cat='Politics' AND size3='s1 <$100'"),
                 ("Crypto | s1 <$100", "cat='Crypto' AND size3='s1 <$100'")]:
    for rec in (True, False):
        d = con.sql(f"SELECT * FROM allday WHERE {w} AND recorded = {rec}").df()
        if not len(d): continue
        r = dict(cell=cname, recorded=rec, n=len(d), n_markets=d.condition_id.nunique())
        for k in ["info", "eff", "over_100"]:
            x = 1e4 * d[k].to_numpy(float); r[f"{k}_bp"] = x.mean(); r[f"{k}_se_ev_bp"] = cr_se_mean(x, d.ev_cluster.to_numpy())[0]
        hrows.append(r)
save(pd.DataFrame(hrows), "qv_H_recorded.csv")

CELLS = [("pooled", "TRUE"), ("Politics", "cat='Politics'"), ("Crypto", "cat='Crypto'")] + \
        [(f"{c} | {s}", f"cat='{c}' AND size3='{s}'") for c in ["Politics", "Crypto"] for s in ["s1 <$100", "s2 $100-1k", "s3 >=$1k"]] + \
        [(f"{c} | >=$100", f"cat='{c}' AND size3<>'s1 <$100'") for c in ["Politics", "Crypto"]]

def cstats(where, exprs):
    sel = ", ".join(f"{e} AS x{i}" for i, e in enumerate(exprs))
    d = con.sql(f"SELECT condition_id, ev_cluster, {sel} FROM v WHERE {where}").df()
    out = {}
    for i, e in enumerate(exprs):
        x = d[f"x{i}"].to_numpy(float)
        if len(x) < 30: out[e] = (np.nan,) * 6; continue
        se_m, Gm = cr_se_mean(x, d.condition_id.to_numpy()); se_e, Ge = cr_se_mean(x, d.ev_cluster.to_numpy())
        out[e] = (x.mean(), se_m, se_e, Gm, Ge, len(x))
    return out

# ---- B: error distribution
rows = []
groups = CELLS + [("pooled | fresh", "fresh_b='fresh'"), ("pooled | stale 2-5", "fresh_b='stale 2-5'"), ("pooled | stale >5", "fresh_b='stale >5'"),
                  ("pooled | crossed pair", "crossed_before"), ("pooled | uncrossed pair", "NOT crossed_before"),
                  ("pooled | quoted spread <= 2c", "spread_q <= 0.02"), ("pooled | quoted spread > 2c", "spread_q > 0.02")]
for g, w in groups:
    r = con.sql(f"""SELECT count(*) n, 1e4*avg(e) mean_e_bp, 1e4*median(e) median_e_bp, 1e4*avg(abs(e)) mae_bp, 1e4*sqrt(avg(e*e)) rmse_bp,
            corr(mid_before, mq) corr_levels, 1e4*avg((aq-bq)/2) quoted_half_spread_bp,
            avg((D*(p - mid_before) < 0)::int) share_cost_negative_proxy, avg((D*(p - mq) < 0)::int) share_cost_negative_quote,
            avg((mid_before < bq OR mid_before > aq)::int) share_proxy_outside_quotes, 1e4*avg(age_q) mean_quote_age_s_x1e4
        FROM v WHERE {w}""").df()
    st = cstats(w, ["D*e"])["D*e"]
    r.insert(0, "group", g); r["mean_De_bp"] = 1e4 * st[0]; r["mean_De_se_ev_bp"] = 1e4 * st[2]; r["mean_De_se_mkt_bp"] = 1e4 * st[1]
    r["mean_quote_age_s"] = r.pop("mean_quote_age_s_x1e4") / 1e4
    rows.append(r)
save(pd.concat(rows), "qv_B_error.csv")

# ---- C: regression of D*e (bp) on characteristics, event-cluster SEs
import statsmodels.formula.api as smf
d = con.sql("""SELECT 1e4*D*e AS De, cat, size3, fresh_b, crossed_before::INT crossed, least(spread_q, 0.2)*1e4 AS spread_bp,
        ln(1 + greatest(dt_since_mid, 0)) AS lage, ev_cluster FROM v WHERE cat IN ('Politics','Crypto','Sports','Finance')""").df()
fit = smf.ols("De ~ C(cat) + C(size3) + C(fresh_b) + crossed + spread_bp + lage", data=d).fit(cov_type="cluster", cov_kwds={"groups": pd.factorize(d.ev_cluster)[0]})
save(pd.DataFrame(dict(term=fit.params.index, coef_bp=fit.params.values, se_ev_bp=fit.bse.values, p=fit.pvalues.values, n=int(fit.nobs))), "qv_C_reg.csv")

# ---- D: metrics with proxy vs quote midpoints on identical trades
pairs = [("eff", "D*(p-mid_before)", "D*(p-mq)", "TRUE"), ("info", "D*(Y-mid_before)", "D*(Y-mq)", "TRUE"),
         ("PI_100", "D*(mid_after_100-mid_before)", "D*(mq_h100-mq)", "ts_lead_100 IS NOT NULL AND okq_h100"),
         ("over_100", "D*(mid_after_100-Y)", "D*(mq_h100-Y)", "ts_lead_100 IS NOT NULL AND okq_h100")]
for k, K in [("R", "res_ts"), ("S", "S")]:
    common = f"ts + 7200 <= {K} AND ts_lead_100 IS NOT NULL AND ts_lead_100 <= {K}"   # common sample of 15_clock_horizon.py
    pairs += [(f"info_{k}", "D*(Y-mid_before)", "D*(Y-mq)", common), (f"eff_{k}", "D*(p-mid_before)", "D*(p-mq)", common)]
    for H in HS:
        pairs += [(f"over_c{H}_{k}", f"D*(mp_c{H}-Y)", f"D*(mq_c{H}-Y)", f"{common} AND okq_c{H}"),
                  (f"PI_c{H}_{k}", f"D*(mp_c{H}-mid_before)", f"D*(mq_c{H}-mq)", f"{common} AND okq_c{H}")]
rows = []
for cname, cw in CELLS:
    for mname, ep, eq, sw in pairs:
        st = cstats(f"{cw} AND {sw}", [ep, eq, f"({ep})-({eq})"])
        a, b, c = st[ep], st[eq], st[f"({ep})-({eq})"]
        rows.append(dict(cell=cname, metric=mname, n=a[5], n_markets=a[3], n_ev_clusters=a[4],
                         proxy_bp=1e4 * a[0], proxy_se_ev_bp=1e4 * a[2], quote_bp=1e4 * b[0], quote_se_ev_bp=1e4 * b[2], quote_se_mkt_bp=1e4 * b[1],
                         diff_bp=1e4 * c[0], diff_se_ev_bp=1e4 * c[2]))
    st = cstats(cw, ["D*(p-Y)"])["D*(p-Y)"]
    rows.append(dict(cell=cname, metric="RS", n=st[5], n_markets=st[3], n_ev_clusters=st[4], proxy_bp=1e4 * st[0], proxy_se_ev_bp=1e4 * st[2],
                     quote_bp=1e4 * st[0], quote_se_ev_bp=1e4 * st[2], quote_se_mkt_bp=1e4 * st[1], diff_bp=0.0, diff_se_ev_bp=0.0))
    st = cstats(cw, ["(aq-bq)/2"])["(aq-bq)/2"]
    rows.append(dict(cell=cname, metric="quoted_half_spread", n=st[5], quote_bp=1e4 * st[0], quote_se_ev_bp=1e4 * st[2]))
save(pd.DataFrame(rows), "qv_D_metrics.csv")

# ---- E: within/between with quote midpoints (and proxy, same trades)
rows = []
for mcol, label in [("mid_before", "proxy"), ("mq", "quote")]:
    g = con.sql(f"""SELECT condition_id, any_value(ev_cluster) ev, any_value(cat) cat, size3, count(*) n, avg(D) Dbar, any_value(Y) Y,
            avg({mcol}) mbar, avg(D*(Y-{mcol})) total FROM v GROUP BY condition_id, size3""").df()
    ga = con.sql(f"""SELECT condition_id, any_value(ev_cluster) ev, any_value(cat) cat, 'all' size3, count(*) n, avg(D) Dbar, any_value(Y) Y,
            avg({mcol}) mbar, avg(D*(Y-{mcol})) total FROM v GROUP BY condition_id""").df()
    g = pd.concat([g, ga]); g["between"] = g.Dbar * (g.Y - g.mbar); g["within"] = g.total - g.between
    for cname in ["pooled", "Politics", "Crypto"]:
        for size in ["all", "s1 <$100", "s2 $100-1k", "s3 >=$1k"]:
            d = g[(g.size3 == size) & ((g.cat == cname) if cname != "pooled" else True)]
            N = d.n.sum(); row = dict(midpoint=label, cell=cname, size=size, n=int(N))
            for comp in ["total", "between", "within"]:
                x = d[comp] * d.n; est = x.sum() / N; s = (x - est * d.n).groupby(d.ev).sum().to_numpy(); G = len(s)
                row[f"{comp}_bp"] = 1e4 * est; row[f"{comp}_se_ev_bp"] = 1e4 * np.sqrt((s ** 2).sum() * G / max(G - 1, 1)) / N
            rows.append(row)
save(pd.DataFrame(rows), "qv_E_wb.csv")
manifest("13b_quote_validation", quote_files=len(files), lead_s=LSTAR, max_age_s=MAX_AGE)
log("done")
