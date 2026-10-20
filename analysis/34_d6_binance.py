"""Official Binance 1-minute spot klines as an external control (D6).

data.binance.vision is only partly reachable from this host, so the candle
panel is the months whose zip tested clean, not a claim of a complete archive.
A candle is used only after its close time. Markets are Yes/No Crypto contracts
whose slug names a bitcoin or ether price threshold and does not name speech
or a mention. The three forward models are the end-of-window logit price, that
price plus the completed 24-hour return and the completed one-hour volatility,
and those plus the small- and large-order imbalances. Training labels are
markets resolved before the test quarter.
"""
import os
import zipfile
from study_common import *
from scipy.special import expit
from scipy.optimize import minimize

EXT = os.path.join(PROJECT, "external", "d6")
PARQ = os.path.join(EXT, "klines.parquet")
MK = os.environ.get("MARKETS_PARQUET", os.path.join(FRES, "markets.parquet"))

def zip_ok(path):
    try:
        with zipfile.ZipFile(path) as z:
            bad = z.testzip()
        return bad is None
    except zipfile.BadZipFile:
        return False

zips = sorted(f for f in os.listdir(EXT) if f.endswith(".zip") and zip_ok(os.path.join(EXT, f)))
log(f"{len(zips)} valid monthly zips")
if len(zips) < 6:
    raise SystemExit("not enough Binance months yet")
if not os.path.isfile(PARQ):
    import duckdb
    con = duckdb.connect()
    parts = []
    for name in zips:
        sym = name.split("-")[0]
        # duckdb reads a zip member through a temporary view built in python below
        parts.append((sym, os.path.join(EXT, name)))
    frames = []
    for sym, path in parts:
        with zipfile.ZipFile(path) as z:
            member = z.namelist()[0]
            z.extract(member, EXT)
        csv = os.path.join(EXT, member)
        frames.append(
            f"SELECT '{sym}' sym, column00::BIGINT open_ms, column04::DOUBLE px, "
            f"column06::BIGINT close_ms, column07::DOUBLE quote_volume FROM read_csv('{csv}', header=false)")
    con.execute("COPY (" + " UNION ALL ".join(frames) + f") TO '{PARQ}' (FORMAT PARQUET)")
    for sym, path in parts:
        with zipfile.ZipFile(path) as z:
            csv = os.path.join(EXT, z.namelist()[0])
        if os.path.isfile(csv):
            os.remove(csv)
    log("wrote", PARQ)

def fit(X, y):
    pen = np.full(X.shape[1], 1e-6); pen[0] = 0.0
    def obj(b):
        z = X @ b
        return np.mean(np.logaddexp(0, z) - y * z) + 0.5 * np.sum(pen * b * b)
    def grad(b):
        return X.T @ (expit(X @ b) - y) / len(y) + pen * b
    return minimize(obj, np.zeros(X.shape[1]), jac=grad, method="BFGS", options={"gtol": 1e-8, "maxiter": 250}).x

con = connect(threads=8, mem="40GB")
con.execute(f"""CREATE OR REPLACE TABLE px AS
    SELECT lower(condition_id) condition_id, res_ts, Y, slug,
           CASE WHEN regexp_matches(lower(slug), '(^|-)(bitcoin|btc)(-|$)') THEN 'BTC'
                WHEN regexp_matches(lower(slug), '(^|-)(ethereum|eth)(-|$)') THEN 'ETH' END asset
    FROM '{MK}'
    WHERE eligible AND grp = 'YesNo' AND cat = 'Crypto'
      AND regexp_matches(lower(slug), '(above|below|reach|between|price|dip)')
      AND NOT regexp_matches(lower(slug), '(say|mention|tweet|speech)')
      AND (regexp_matches(lower(slug), '(^|-)(bitcoin|btc)(-|$)')
           OR regexp_matches(lower(slug), '(^|-)(ethereum|eth)(-|$)'))""")
# a slug that names both assets is ambiguous
con.execute("""DELETE FROM px WHERE regexp_matches(lower(slug), '(^|-)(bitcoin|btc)(-|$)')
    AND regexp_matches(lower(slug), '(^|-)(ethereum|eth)(-|$)')""")
w = pd.read_csv(os.path.join(RES, "mkt_A_windows_YesNo.csv"))
w = w[w.win_name == "W_paper (life before 24h)"].copy()
w["condition_id"] = w.condition_id.str.lower()
px = con.sql("SELECT * FROM px").df()
d = w.merge(px, on="condition_id", how="inner", suffixes=("", "_m"))
d = d[d.p_post.between(0.02, 0.98) & d.imb_small.notna() & d.imb_large.notna()].copy()
d["pred_ms"] = (d.res_ts - 86400) * 1000
log(f"price-threshold markets in the paper window with both flows: {len(d):,}")
con.register("mk", d[["condition_id", "asset", "pred_ms", "Y", "p_post", "imb_small", "imb_large", "ev_cluster"]])
con.execute(f"CREATE OR REPLACE TABLE k AS SELECT * FROM '{PARQ}'")
# last completed close at the prediction time, and 24 hours earlier
feat = con.sql("""
    WITH a AS (
        SELECT m.condition_id, arg_max(k.px, k.close_ms) c1, max(k.close_ms) t1
        FROM mk m JOIN k ON k.sym = CASE WHEN m.asset = 'BTC' THEN 'BTCUSDT' ELSE 'ETHUSDT' END
        WHERE k.close_ms < m.pred_ms
        GROUP BY 1),
    b AS (
        SELECT m.condition_id, arg_max(k.px, k.close_ms) c0
        FROM mk m JOIN k ON k.sym = CASE WHEN m.asset = 'BTC' THEN 'BTCUSDT' ELSE 'ETHUSDT' END
        WHERE k.close_ms < m.pred_ms - 86400000
        GROUP BY 1),
    v AS (
        SELECT condition_id, stddev_samp(r) vol FROM (
            SELECT m.condition_id, ln(k.px / lag(k.px) OVER (PARTITION BY m.condition_id ORDER BY k.close_ms)) r
            FROM mk m JOIN k ON k.sym = CASE WHEN m.asset = 'BTC' THEN 'BTCUSDT' ELSE 'ETHUSDT' END
            WHERE k.close_ms < m.pred_ms AND k.close_ms >= m.pred_ms - 3600000)
        GROUP BY 1)
    SELECT m.*, ln(a.c1 / b.c0) ret24, v.vol
    FROM mk m JOIN a USING (condition_id) JOIN b USING (condition_id) JOIN v USING (condition_id)
    WHERE a.c1 > 0 AND b.c0 > 0 AND a.t1 > m.pred_ms - 120000
""").df()
log(f"with a completed Binance candle: {len(feat):,} of {len(d):,}")
months = sorted({z.split("-")[2] + "-" + z.split("-")[3][:2] for z in zips})
save(pd.DataFrame({"months": [",".join(months)], "n_markets_window": [len(d)], "n_with_candle": [len(feat)]}), "d6_A_coverage.csv")
feat["res"] = pd.to_datetime(feat.ev_cluster.str.split("|").str[-1], errors="coerce")
feat = feat[feat.res.notna() & feat.ret24.notna() & feat.vol.notna()].copy()
feat["q"] = feat.res.dt.to_period("Q").astype(str)
feat["qstart"] = feat.res.dt.to_period("Q").dt.start_time

def score(df, asset):
    sub = df if asset == "both" else df[df.asset == asset]
    sub = sub.copy()
    if len(sub) < 500:
        log(f"skip {asset}: {len(sub)}")
        return []
    y = sub.Y.to_numpy(float)
    p = np.clip(sub.p_post.to_numpy(float), 1e-4, 1 - 1e-4)
    Z = np.column_stack([np.ones(len(sub)), np.log(p / (1 - p))])
    specs = {
        "price": Z,
        "price + external": np.column_stack([Z, sub.ret24.to_numpy(float) * 100, sub.vol.to_numpy(float) * 100]),
        "price + external + flow": np.column_stack([Z, sub.ret24.to_numpy(float) * 100, sub.vol.to_numpy(float) * 100,
                                                    sub.imb_small.to_numpy(float), sub.imb_large.to_numpy(float)]),
    }
    fold = sub.q.to_numpy()
    out = []
    pred = {}
    for lab, X in specs.items():
        P = np.full(len(sub), np.nan)
        for f in sorted(sub.q.unique()):
            te = fold == f
            tr = (sub.res < sub.loc[te, "qstart"].min()).to_numpy()
            if tr.sum() < 200 or te.sum() == 0:
                continue
            P[np.where(te)[0]] = expit(X[te] @ fit(X[tr], y[tr]))
        pred[lab] = P
    ok = ~np.isnan(pred["price"]) & ~np.isnan(pred["price + external"]) & ~np.isnan(pred["price + external + flow"])
    if ok.sum() == 0:
        return []
    g = sub.ev_cluster.to_numpy()[ok]
    codes = pd.factorize(g)[0]
    G = int(codes.max()) + 1
    rng = np.random.default_rng(2718)
    base = (pred["price"][ok] - y[ok]) ** 2
    for lab, P in pred.items():
        diff = base - (P[ok] - y[ok]) ** 2
        sums = np.bincount(codes, weights=diff)
        cnt = np.bincount(codes)
        draws = [sums[ix].sum() / max(cnt[ix].sum(), 1) for ix in (rng.integers(0, G, G) for _ in range(1000))]
        lo, hi = np.quantile(draws, [0.025, 0.975])
        out.append(dict(asset=asset, model=lab, n=int(ok.sum()), n_clusters=int(pd.Series(g).nunique()),
                        brier=float(np.mean((P[ok] - y[ok]) ** 2)),
                        dbrier_vs_price=float(diff.mean()), lo=float(lo), hi=float(hi)))
        log(asset, lab, out[-1]["dbrier_vs_price"])
    return out

rows = []
for asset in ["BTC", "ETH", "both"]:
    rows += score(feat, asset)
save(pd.DataFrame(rows), "d6_B_prediction.csv")
log("done")
