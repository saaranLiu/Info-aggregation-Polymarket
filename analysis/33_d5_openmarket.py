"""OpenMarket BTC tape (D5) against v1. Conclusions stay inside BTC.

unified/ v0.4.3, revision 74502466. Publisher lead-lag is reported as a
synchronisation diagnostic and is not used as a feature. A tick is usable
only when both its source time and its ingest time are already past. Markets
shorter than the paper's 1-24 hour window are estimated on a separate
five-minute design and are not compared with the main residual.
"""
import glob
import os
from study_common import *
from scipy.special import expit
from scipy.optimize import minimize

EXT = os.path.join(PROJECT, "external", "d5", "unified")
META = os.path.join(EXT, "market_meta", "unpartitioned", "part-000001.parquet")
CAND = os.path.join(EXT, "binance_candles_1m")
TICK = os.path.join(EXT, "binance_ticks_ms")
LAG = os.path.join(EXT, "lag_pairs_ms")
need = [META, CAND, TICK, LAG]
if not all(os.path.isdir(p) or os.path.isfile(p) for p in need):
    raise SystemExit("D5 unified files are not all present")
if glob.glob(os.path.join(EXT, "**", "*.aria2"), recursive=True):
    raise SystemExit("D5 download still in progress")
MK = os.environ.get("MARKETS_PARQUET", os.path.join(FRES, "markets.parquet"))
DA = os.environ.get("DA_DIR", os.path.join(C.ROOT, "database", "daily_aligned"))

def fit(X, y):
    pen = np.full(X.shape[1], 1e-6); pen[0] = 0.0
    def obj(b):
        z = X @ b
        return np.mean(np.logaddexp(0, z) - y * z) + 0.5 * np.sum(pen * b * b)
    def grad(b):
        return X.T @ (expit(X @ b) - y) / len(y) + pen * b
    return minimize(obj, np.zeros(X.shape[1]), jac=grad, method="BFGS", options={"gtol": 1e-8, "maxiter": 200}).x

con = connect(threads=16, mem="80GB")
con.execute(f"""CREATE OR REPLACE TABLE meta AS
    SELECT src.market_slug AS slug, src.question, src.up_token_id, src.down_token_id,
           src.first_seen_ms, src.last_seen_ms
    FROM read_parquet('{META}') src""")
con.execute(f"""CREATE OR REPLACE TABLE ours AS
    SELECT lower(src.condition_id) condition_id, src.grp, src.cat, src.Y, src.res_ts, src.t_first, src.t_last,
           src.slug ours_slug, (src.t_last - src.t_first) / 3600.0 life_h
    FROM '{MK}' src WHERE src.eligible""")
ov = con.sql("""SELECT coalesce(o.grp, 'not in v1') grp, count(*) n_meta,
        count(o.condition_id) n_in_v1, median(o.life_h) med_life_h,
        quantile_cont(o.life_h, 0.9) p90_life_h
    FROM meta m LEFT JOIN ours o ON o.ours_slug = m.slug GROUP BY 1 ORDER BY n_meta DESC""").df()
save(ov, "d5_A_coverage.csv")
log(ov.to_string(index=False))

ticks = con.sql(f"""SELECT
        count(*) n, count(DISTINCT date) n_days, 1e-3*median(ingest_ts_ms - source_ts_ms) med_ingest_lag_s,
        1e-3*quantile_cont(ingest_ts_ms - source_ts_ms, 0.9) p90_ingest_lag_s,
        avg((ingest_ts_ms < source_ts_ms)::int) share_ingest_before_source
    FROM read_parquet('{TICK}/*/*.parquet')""").df()
lags = con.sql(f"""SELECT count(*) n, median(lead_lag_ms) med_lead_lag_ms,
        quantile_cont(lead_lag_ms, 0.1) p10, quantile_cont(lead_lag_ms, 0.9) p90,
        avg((quality_flag = 'tight')::int) share_tight,
        avg((quality_flag = 'wide')::int) share_wide
    FROM read_parquet('{LAG}/*/*.parquet')""").df()
days = con.sql(f"""SELECT date, count(*) n FROM read_parquet('{CAND}/*/*.parquet') GROUP BY 1 ORDER BY 1""").df()
save(ticks, "d5_B_clocks.csv")
save(lags, "d5_B_leadlag.csv")
save(days, "d5_B_candle_days.csv")
log("ingest lag s", ticks.med_ingest_lag_s.iloc[0], "lead-lag ms", lags.med_lead_lag_ms.iloc[0], "candle days", len(days))

# BTC price-threshold markets in the paper window. OpenMarket's Binance candles are
# sparse (about two hours of minutes per day), so a candle is used only if it closed
# and was ingested within 15 minutes of the target time. This is not the official
# monthly archive: data.binance.vision did not serve the 2024-2026 files.
con.execute(f"""CREATE OR REPLACE TABLE candles AS
    SELECT close_price AS close, candle_end AS close_time, created_at
    FROM read_parquet('{CAND}/*/*.parquet') WHERE close_price > 0""")
w = pd.read_csv(os.path.join(RES, "mkt_A_windows_YesNo.csv"))
w = w[(w.win_name == "W_paper (life before 24h)") & w.p_post.between(0.02, 0.98)
      & w.imb_small.notna() & w.imb_large.notna()].copy()
w["condition_id"] = w.condition_id.str.lower()
btc = con.sql(f"""SELECT lower(src.condition_id) condition_id, src.res_ts FROM '{MK}' src
    WHERE eligible AND grp='YesNo' AND cat='Crypto'
      AND regexp_matches(lower(slug), '(^|-)(bitcoin|btc)(-|$)')
      AND regexp_matches(lower(slug), '(above|below|reach|between|price|dip)')
      AND NOT regexp_matches(lower(slug), '(say|mention|tweet|speech)')
      AND NOT regexp_matches(lower(slug), '(^|-)(ethereum|eth)(-|$)')""").df()
extm = w.merge(btc, on="condition_id")
extm["pred_ms"] = (extm.res_ts - 86400) * 1000
con.register("extm", extm[["condition_id", "pred_ms", "Y", "p_post", "imb_small", "imb_large", "ev_cluster"]])
feat = con.sql("""
    WITH a AS (
        SELECT m.condition_id, arg_max(k.close, k.close_time) c1, max(k.close_time) t1, arg_max(k.created_at, k.close_time) ing
        FROM extm m JOIN candles k ON k.close_time < m.pred_ms AND k.created_at < m.pred_ms
        GROUP BY 1),
    b AS (
        SELECT m.condition_id, arg_max(k.close, k.close_time) c0, max(k.close_time) t0
        FROM extm m JOIN candles k ON k.close_time < m.pred_ms - 86400000 AND k.created_at < m.pred_ms - 86400000
        GROUP BY 1)
    SELECT m.*, ln(a.c1/b.c0) ret24, (m.pred_ms - a.t1)/1000.0 age_s
    FROM extm m JOIN a USING (condition_id) JOIN b USING (condition_id)
    WHERE a.c1 > 0 AND b.c0 > 0 AND m.pred_ms - a.t1 <= 900000 AND (m.pred_ms - 86400000) - b.t0 <= 900000
      AND a.ing < m.pred_ms""").df()
log(f"BTC threshold markets {len(extm):,}; with a Binance candle within 15 min {len(feat):,}")
save(pd.DataFrame([dict(n_window=len(extm), n_candle=len(feat),
                        med_age_s=float(feat.age_s.median()) if len(feat) else None)]), "d5_D_coverage.csv")
if len(feat) >= 200:
    feat["res"] = pd.to_datetime(feat.ev_cluster.str.split("|").str[-1], errors="coerce")
    feat = feat[feat.res.notna()].copy()
    feat["q"] = feat.res.dt.to_period("Q").astype(str)
    y = feat.Y.to_numpy(float)
    p = np.clip(feat.p_post.to_numpy(float), 1e-4, 1 - 1e-4)
    Z = np.column_stack([np.ones(len(feat)), np.log(p / (1 - p))])
    specs = {"price": Z, "price + BTC return": np.column_stack([Z, feat.ret24.to_numpy(float) * 100]),
             "price + return + flow": np.column_stack([Z, feat.ret24.to_numpy(float) * 100, feat.imb_small, feat.imb_large])}
    fold = feat.q.to_numpy()
    pred = {}
    for lab, X in specs.items():
        P = np.full(len(feat), np.nan)
        for f in sorted(feat.q.unique()):
            te = fold == f
            qstart = feat.loc[te, "res"].min().to_period("Q").start_time
            tr = (feat.res < qstart).to_numpy()
            if tr.sum() < 80 or te.sum() == 0:
                continue
            P[np.where(te)[0]] = expit(X[te] @ fit(X[tr], y[tr]))
        pred[lab] = P
    ok = np.ones(len(feat), bool)
    for P in pred.values():
        ok &= ~np.isnan(P)
    rows = []
    if ok.sum():
        g = feat.ev_cluster.to_numpy()[ok]
        codes = pd.factorize(g)[0]; G = int(codes.max()) + 1
        rng = np.random.default_rng(2718)
        base = (pred["price"][ok] - y[ok]) ** 2
        for lab, P in pred.items():
            diff = base - (P[ok] - y[ok]) ** 2
            sums = np.bincount(codes, weights=diff); cnt = np.bincount(codes)
            draws = [sums[ix].sum() / cnt[ix].sum() for ix in (rng.integers(0, G, G) for _ in range(1000))]
            lo, hi = np.quantile(draws, [0.025, 0.975])
            rows.append(dict(model=lab, n=int(ok.sum()), brier=float(np.mean((P[ok] - y[ok]) ** 2)),
                             dbrier_vs_price=float(diff.mean()), lo=float(lo), hi=float(hi)))
    save(pd.DataFrame(rows), "d5_D_external_pred.csv")
    log(pd.DataFrame(rows).to_string(index=False) if rows else "no forward test rows")

# Five-minute design for markets that are in v1 and live under 24 hours.
# Prediction time is five minutes before res_ts. Flow is v1 signed notional in the
# preceding five minutes. The BTC return uses only 1-minute candles that have closed.
short = con.sql("""SELECT o.condition_id, m.slug, o.Y, o.res_ts, o.life_h, o.grp
    FROM meta m JOIN ours o ON o.ours_slug = m.slug
    WHERE o.life_h < 24 AND o.res_ts IS NOT NULL AND o.Y IN (0, 1)""").df()
log(f"short v1 markets {len(short):,}")
if len(short) < 200:
    log("too few short markets to score")
    raise SystemExit(0)
ids = short.condition_id.tolist()
con.register("short_ids", short[["condition_id", "res_ts", "Y"]])
files = []
d0, d1 = dt_start, dt_end = __import__("datetime").date(2026, 3, 14), __import__("datetime").date(2026, 4, 30)
d = d0
while d <= d1:
    f = os.path.join(DA, d.isoformat() + ".parquet")
    if os.path.isfile(f):
        files.append(f)
    d += __import__("datetime").timedelta(days=1)
log(f"scanning {len(files)} v1 days for the short markets")
fl = ",".join(f"'{f}'" for f in files)
con.execute(f"""CREATE OR REPLACE TABLE tr AS
    SELECT t.condition_id, t.block_timestamp::BIGINT ts, t.D::DOUBLE D, t.p_event p, t.usdc_amount usd,
           s.res_ts, s.Y
    FROM read_parquet([{fl}]) t
    JOIN short_ids s ON s.condition_id = lower(t.condition_id)
    WHERE t.D IS NOT NULL AND t.p_event IS NOT NULL AND t.usdc_amount > 0""")
# one trade per condition, second, direction: the file is fills. Collapse like the paper.
con.execute("""CREATE OR REPLACE TABLE tr1 AS
    SELECT condition_id, ts, D, sum(p*usd)/sum(usd) p, sum(usd) usd, any_value(res_ts) res_ts, any_value(Y) Y
    FROM tr GROUP BY 1, 2, 3""")
con.execute("""CREATE OR REPLACE TABLE pred AS
    SELECT condition_id, any_value(res_ts) res_ts, any_value(Y) Y,
           arg_max(p, ts) FILTER (WHERE ts < res_ts - 300) p0,
           sum(D*usd) FILTER (WHERE ts >= res_ts - 600 AND ts < res_ts - 300)
             / nullif(sum(usd) FILTER (WHERE ts >= res_ts - 600 AND ts < res_ts - 300), 0) flow
    FROM tr1 GROUP BY 1""")
con.execute(f"""CREATE OR REPLACE TABLE candles AS
    SELECT candle_start, close_price AS close, candle_end AS close_time, created_at
    FROM read_parquet('{CAND}/*/*.parquet', union_by_name=true)
    WHERE close_price > 0""")
pred = con.sql("SELECT * FROM pred WHERE p0 BETWEEN 0.02 AND 0.98 AND flow IS NOT NULL").df()
cand = con.sql("SELECT candle_start, close, close_time, created_at FROM candles ORDER BY close_time").df()
pred["t1"] = ((pred.res_ts - 300) * 1000).round().astype("int64")
pred["t0"] = ((pred.res_ts - 600) * 1000).round().astype("int64")
cand["close_time"] = cand.close_time.round().astype("int64")
cand["created_at"] = pd.to_numeric(cand.created_at, errors="coerce").round().astype("int64")
pred = pred.sort_values("t1")
a = pd.merge_asof(pred, cand.rename(columns={"close": "c1", "created_at": "ing1"}), left_on="t1", right_on="close_time", direction="backward")
a = a.sort_values("t0")
a = pd.merge_asof(a, cand.rename(columns={"close": "c0", "candle_start": "cs0", "close_time": "ct0", "created_at": "ing0"}),
                  left_on="t0", right_on="ct0", direction="backward")
# both candles must have closed, and been ingested, within 10 minutes of the target time
scored = a[(a.c1 > 0) & (a.c0 > 0) & (a.t1 - a.close_time <= 600_000) & (a.t0 - a.ct0 <= 600_000)
           & (a.ing1 < a.t1) & (a.ing0 < a.t0)].copy()
scored["ret5"] = np.log(scored.c1 / scored.c0)
log(f"scored short markets {len(scored):,}")
save(scored[["condition_id", "Y", "p0", "flow", "ret5"]], "d5_C_short_rows.csv")
if len(scored) < 400:
    log("short scored sample is below 400; prediction not fit")
else:
    scored["bucket"] = (scored.res_ts // 300).astype(int)
    y = scored.Y.to_numpy(float)
    p = np.clip(scored.p0.to_numpy(float), 1e-4, 1 - 1e-4)
    X0 = np.column_stack([np.ones(len(scored)), np.log(p / (1 - p))])
    cut = np.sort(scored.bucket.unique())[len(scored.bucket.unique()) // 2]
    tr = (scored.bucket < cut).to_numpy()
    te = ~tr
    rows = []
    preds = {}
    extras = {
        "price": X0,
        "price + BTC return": np.column_stack([X0, scored.ret5.to_numpy(float)]),
        "price + return + flow": np.column_stack([X0, scored.ret5.to_numpy(float), scored.flow.to_numpy(float)]),
    }
    for lab, X in extras.items():
        b = fit(X[tr], y[tr])
        pr = expit(X[te] @ b)
        preds[lab] = pr
        rows.append(dict(model=lab, n_train=int(tr.sum()), n_test=int(te.sum()), brier=float(np.mean((pr - y[te]) ** 2))))
    yte = y[te]
    g = scored.bucket.to_numpy()[te]
    base = (preds["price"] - yte) ** 2
    rng = np.random.default_rng(2718)
    codes = pd.factorize(g)[0]
    G = int(codes.max()) + 1
    for r in rows:
        diff = base - (preds[r["model"]] - yte) ** 2
        sums = np.bincount(codes, weights=diff)
        cnt = np.bincount(codes)
        draws = [sums[ix].sum() / cnt[ix].sum() for ix in (rng.integers(0, G, G) for _ in range(1000))]
        lo, hi = np.quantile(draws, [0.025, 0.975])
        r["dbrier_vs_price"] = float(diff.mean())
        r["lo"] = float(lo)
        r["hi"] = float(hi)
    save(pd.DataFrame(rows), "d5_C_short_pred.csv")
    log(pd.DataFrame(rows).to_string(index=False))
log("done")
