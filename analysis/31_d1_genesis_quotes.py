"""Dome L2 quotes (D1) against the transaction midpoint, Politics and Crypto.

Source: vpcapitano/polymarket-genesis-liquidity, revision c073aed6, 14 Oct 2025
through 31 Mar 2026. Empty sides were dropped by the publisher, so a match is
conditional on a two-sided book. The quote is the reference token's own book
(outcome_seq = 1). The lead is chosen on a fixed grid by the share of trades
that print on the touched side, before any residual is computed. A snapshot
older than 15 minutes is not used, and none is taken from after the trade.

A second block compares these mids with the pmxt extract already on disk
(Joseph3222/polymarket-orderbook). archive.pmxt.dev did not answer from this
host, so that direct hourly archive is not a completed check.
"""
import datetime as dt
import glob
import os
from study_common import *

REV = "c073aed6af9f7d3e2dc5f17c41c4d48366861d2e"
EXT = os.path.join(PROJECT, "external")
SRC = os.path.join(EXT, "d1", "dome_polymarket_2025-10_2026-03.parquet")
if os.path.isfile(SRC + ".aria2") or not os.path.isfile(SRC):
    raise SystemExit("D1 parquet is not complete yet")
DA = os.environ.get("DA_DIR", os.path.join(C.ROOT, "database", "daily_aligned"))
LEADS = [0, 1, 2, 3, 4, 5, 6, 8, 10, 15, 20, 30, 60]
MAX_AGE = 900
T0 = int(dt.datetime(2025, 10, 14, tzinfo=dt.timezone.utc).timestamp())
T1 = int(dt.datetime(2026, 4, 1, tzinfo=dt.timezone.utc).timestamp())

def se_mean(x, g):
    x = np.asarray(x, float)
    n = len(x)
    if n == 0:
        return np.nan, np.nan, 0
    s = pd.Series(x - x.mean()).groupby(np.asarray(g)).sum().to_numpy()
    G = len(s)
    return float(x.mean()), float(np.sqrt((s ** 2).sum()) / n * np.sqrt(G / max(G - 1, 1))), G

con = connect(threads=16, mem="120GB")
days = []
d = dt.date(2025, 10, 14)
while d <= dt.date(2026, 3, 31):
    f = os.path.join(DA, d.isoformat() + ".parquet")
    if os.path.isfile(f):
        days.append(f)
    d += dt.timedelta(days=1)
log(f"{len(days)} daily files in the Dome window")
daylist = ",".join(f"'{f}'" for f in days)
con.execute(f"""CREATE OR REPLACE TABLE ids AS
    SELECT DISTINCT condition_id FROM '{WORK}/mid_YesNo.parquet'
    WHERE cat IN ('Politics','Crypto') AND {LATE} AND ts >= {T0} AND ts < {T1}""")
con.execute(f"""CREATE OR REPLACE TABLE ref AS
    SELECT condition_id, any_value(asset_id) asset_id, count(DISTINCT asset_id) n_id
    FROM read_parquet([{daylist}])
    WHERE outcome_seq = 1 AND asset_id IS NOT NULL AND condition_id IN (SELECT condition_id FROM ids)
    GROUP BY 1""")
nref = con.sql("SELECT count(*) n, sum((n_id>1)::int) multi FROM ref").fetchone()
log(f"reference tokens {nref[0]:,}, markets with more than one {nref[1]}")
con.execute("DELETE FROM ref WHERE n_id <> 1")
out_q = os.path.join(EXT, "d1", "quotes_ref.parquet")
log("filtering Dome snapshots to reference tokens")
con.execute(f"""COPY (
    SELECT r.condition_id, q.timestamp_ms // 1000 AS ts,
           arg_max(q.best_bid, q.timestamp_ms) bid, arg_max(q.best_ask, q.timestamp_ms) ask
    FROM read_parquet('{SRC}') q
    JOIN ref r ON r.asset_id = q.token_id::VARCHAR
    WHERE q.best_bid > 0 AND q.best_ask < 1 AND q.best_bid < q.best_ask
      AND q.timestamp_ms >= {T0 * 1000} AND q.timestamp_ms < {T1 * 1000}
    GROUP BY 1, 2
) TO '{out_q}' (FORMAT PARQUET)""")
nq = con.sql(f"SELECT count(*), count(DISTINCT condition_id) FROM '{out_q}'").fetchone()
log(f"quote seconds {nq[0]:,} across {nq[1]:,} markets")

con.execute(f"""CREATE OR REPLACE TABLE ev AS
    SELECT condition_id, ts::BIGINT ts, D::DOUBLE D, p, usd, Y, cat, ev_cluster, size3,
           mid_before, mid_after_100, ts_lead_100::BIGINT ts_lead_100
    FROM '{WORK}/mid_YesNo.parquet'
    WHERE cat IN ('Politics','Crypto') AND {LATE} AND {BASE} AND ts >= {T0} AND ts < {T1}""")
n_ev = con.sql("SELECT count(*), count(DISTINCT condition_id) FROM ev").fetchone()
log(f"window trades {n_ev[0]:,} markets {n_ev[1]:,}")
con.execute(f"CREATE OR REPLACE TABLE q AS SELECT * FROM '{out_q}'")

rows = []
for L in LEADS:
    r = con.sql(f"""WITH j AS (
            SELECT e.D, e.p, x.bid, x.ask, e.ts - {L} - x.ts AS age
            FROM ev e ASOF LEFT JOIN q x ON x.condition_id = e.condition_id AND x.ts <= e.ts - {L})
        SELECT {L} lead_s, count(*) n,
            avg((age BETWEEN 0 AND {MAX_AGE})::int) share_valid,
            avg((abs(p - CASE WHEN D = 1 THEN ask ELSE bid END) < 5e-4)::int)
                FILTER (WHERE age BETWEEN 0 AND {MAX_AGE}) share_exact_side
        FROM j""").df()
    rows.append(r)
    log(f"lead {L}: valid {r.share_valid.iloc[0]:.3f}, touch {r.share_exact_side.iloc[0]}")
A = pd.concat(rows, ignore_index=True)
save(A, "d1_A_alignment.csv")
LSTAR = int(A.sort_values(["share_exact_side", "lead_s"], ascending=[False, True]).iloc[0].lead_s)
log(f"lead {LSTAR}s")

con.execute(f"""CREATE OR REPLACE TABLE m AS
    SELECT e.*, x.bid bq, x.ask aq, (x.bid + x.ask) / 2 mq, e.ts - {LSTAR} - x.ts AS age,
           (e.ts_lead_100 - y.ts BETWEEN 0 AND {MAX_AGE}) AS post_ok, (y.bid + y.ask) / 2 mq_h
    FROM ev e
    ASOF LEFT JOIN q x ON x.condition_id = e.condition_id AND x.ts <= e.ts - {LSTAR}
    ASOF LEFT JOIN q y ON y.condition_id = e.condition_id AND y.ts <= e.ts_lead_100""")

base = con.sql(f"""SELECT cat, size3, ev_cluster, condition_id,
        D*(Y - mid_before) info_p, D*(Y - mq) info_q,
        D*(mid_after_100 - Y) over_p, D*(mq_h - Y) over_q,
        age BETWEEN 0 AND {MAX_AGE} pre_ok, post_ok AND mid_after_100 IS NOT NULL h_ok
    FROM m""").df()
log(f"scored {len(base):,}")

def pack(df, col, name, sample):
    out = []
    cells = [("pooled", df)] + [(c, df[df.cat == c]) for c in ["Politics", "Crypto"]]
    cells += [(f"{c} | {s}", df[(df.cat == c) & (df.size3 == s)])
              for c in ["Politics", "Crypto"] for s in ["s1 <$100", "s3 >=$1k"]]
    for lab, g in cells:
        if len(g) == 0:
            continue
        est, se, G = se_mean(g[col].to_numpy(), g.ev_cluster.to_numpy())
        out.append(dict(sample=sample, metric=name, cell=lab, n=len(g),
                        n_markets=int(g.condition_id.nunique()), n_clusters=G,
                        est_bp=est * 1e4, se_bp=se * 1e4))
    return out

recs = []
win = base.dropna(subset=["info_p"])
recs += pack(win, "info_p", "info", "window proxy")
matched = base[base.pre_ok].dropna(subset=["info_p", "info_q"])
recs += pack(matched, "info_p", "info", "matched proxy")
recs += pack(matched, "info_q", "info", "matched quote")
recs += pack(matched.assign(d=matched.info_q - matched.info_p), "d", "info", "quote minus proxy")
hov = base[base.h_ok.fillna(False)].dropna(subset=["over_p", "over_q"])
recs += pack(base.dropna(subset=["over_p"]), "over_p", "over_100", "window proxy")
recs += pack(hov, "over_p", "over_100", "matched proxy")
recs += pack(hov, "over_q", "over_100", "matched quote")
recs += pack(hov.assign(d=hov.over_q - hov.over_p), "d", "over_100", "quote minus proxy")
save(pd.DataFrame(recs), "d1_B_metrics.csv")
cov = pd.DataFrame([
    dict(trades=int(len(base)), markets=int(base.condition_id.nunique()),
         pre_ok=int(base.pre_ok.sum()), post_ok=int(base.h_ok.fillna(False).sum()),
         lead_s=LSTAR, revision=REV)])
save(cov, "d1_C_coverage.csv")

# D1 versus the pmxt extract on overlapping sampled days. Same second, reference axis.
qdir = os.path.join(PROJECT, "work", "quotes")
raws = sorted(glob.glob(os.path.join(qdir, "raw", "date=2026-0[23]*.parquet")))
raws = [f for f in raws if "2026-02-22" <= os.path.basename(f)[5:15] <= "2026-03-31"]
tmap = os.path.join(qdir, "token_map.parquet")
if raws and os.path.isfile(tmap):
    fl = ",".join(f"'{f}'" for f in raws)
    both = con.sql(f"""
        WITH px AS (
            SELECT r.condition_id, r.event_ts::BIGINT ts,
                   arg_max(CASE WHEN t.outcome_seq = 1 THEN r.best_bid ELSE 1 - r.best_ask END, r.row) bid,
                   arg_max(CASE WHEN t.outcome_seq = 1 THEN r.best_ask ELSE 1 - r.best_bid END, r.row) ask
            FROM read_parquet([{fl}]) r
            JOIN '{tmap}' t ON t.condition_id = r.condition_id AND t.asset_id = r.asset_id
            WHERE r.best_bid IS NOT NULL AND r.best_ask IS NOT NULL
            GROUP BY 1, 2),
        pm AS (SELECT condition_id, ts, (bid+ask)/2 mid FROM px WHERE bid > 0 AND ask < 1 AND bid < ask),
        dm AS (SELECT condition_id, ts, (bid+ask)/2 mid FROM q)
        SELECT count(*) n, count(DISTINCT pm.condition_id) markets,
               1e4*avg(dm.mid - pm.mid) mean_bp, 1e4*avg(abs(dm.mid - pm.mid)) mean_abs_bp,
               avg((abs(dm.mid - pm.mid) < 0.01)::int) share_within_tick
        FROM pm JOIN dm USING (condition_id, ts)""").df()
    both["days"] = len(raws)
    save(both, "d1_D_pmxt_overlap.csv")
    log(both.to_string(index=False))
else:
    log("no pmxt extract to compare")
log("done")
