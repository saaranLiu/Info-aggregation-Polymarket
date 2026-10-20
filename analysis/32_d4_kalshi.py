"""Kalshi settled binary markets (D4). Not pooled with Polymarket.

Trades and markets: TrevorJS/kalshi-trades, revision 5c622baffd. taker_side yes/no
maps to D = +1/-1 on the YES axis, p = yes_price/100, and the order's cash is
count times the price of the leg the taker bought. The settlement margin
RS = D(p - Y) does not use a midpoint. yes_bid/yes_ask are fetch-time snapshots
and are not used as pre-trade quotes. The window is the exchange close minus
24 hours through the close minus 1 hour, which is not an on-chain resolution time.

Dollar bins use the same $100 and $1,000 cuts as the main text. Rank bins are
thirds of taker cash within the market, on this window. Both are reported.
Standard errors cluster by event_ticker. The forward logit trains only on
events whose close is before the test quarter.
"""
import glob
import os
from study_common import *
from scipy.special import expit
from scipy.optimize import minimize

EXT = os.path.join(PROJECT, "external", "d4")
files = sorted(glob.glob(os.path.join(EXT, "*.parquet")))
if any(os.path.isfile(f + ".aria2") for f in files) or len(files) < 20:
    raise SystemExit(f"Kalshi download incomplete ({len(files)} parquet files)")
mk = [f for f in files if os.path.basename(f).startswith("markets")]
tr = [f for f in files if os.path.basename(f).startswith("trades")]
log(f"{len(mk)} market files, {len(tr)} trade files")

def se_of(sum_x, n, mean):
    """Cluster-robust SE of a mean from per-cluster sums and counts."""
    s = sum_x - mean * n
    G = len(s)
    N = n.sum()
    if N == 0 or G < 2:
        return np.nan, G
    return float(np.sqrt((s ** 2).sum()) / N * np.sqrt(G / (G - 1))), G

def contrast_se(a, b):
    """SE of mean(a) - mean(b) from frames with columns sum_x, n, cluster."""
    m = a.merge(b, on="cluster", how="outer", suffixes=("_a", "_b")).fillna(0.0)
    Na, Nb = m.n_a.sum(), m.n_b.sum()
    ma, mb = m.sum_x_a.sum() / Na, m.sum_x_b.sum() / Nb
    u = np.where(m.n_a > 0, (m.sum_x_a - ma * m.n_a) / Na, 0) - np.where(m.n_b > 0, (m.sum_x_b - mb * m.n_b) / Nb, 0)
    G = ((m.n_a + m.n_b) > 0).sum()
    se = np.sqrt((u ** 2).sum()) * np.sqrt(G / max(G - 1, 1))
    return (ma - mb) * 1e4, se * 1e4, int(G)

con = connect(threads=16, mem="120GB")
mlist = ",".join(f"'{f}'" for f in mk)
tlist = ",".join(f"'{f}'" for f in tr)
con.execute(f"""CREATE OR REPLACE TABLE mk AS
    SELECT ticker, event_ticker, market_type, close_time, lower(result) result
    FROM read_parquet([{mlist}])
    WHERE lower(result) IN ('yes','no') AND close_time IS NOT NULL AND market_type = 'binary'""")
log("markets", con.sql("SELECT count(*), count(DISTINCT event_ticker) FROM mk").fetchone())
con.execute(f"""CREATE OR REPLACE TABLE win AS
    SELECT t.ticker, m.event_ticker, m.close_time, m.result,
           CASE WHEN t.taker_side = 'yes' THEN 1 WHEN t.taker_side = 'no' THEN -1 END AS D,
           t.yes_price / 100.0 AS p,
           CASE WHEN m.result = 'yes' THEN 1.0 ELSE 0.0 END AS Y,
           t.count * (CASE WHEN t.taker_side = 'yes' THEN t.yes_price ELSE t.no_price END) / 100.0 AS cash,
           t.yes_price + t.no_price AS cents
    FROM read_parquet([{tlist}]) t
    JOIN mk m ON m.ticker = t.ticker
    WHERE t.taker_side IN ('yes','no') AND t.yes_price > 20 AND t.yes_price < 80
      AND epoch(t.created_time) >= epoch(m.close_time) - 86400
      AND epoch(t.created_time) < epoch(m.close_time) - 3600
      AND t.count > 0""")
con.execute("ALTER TABLE win ADD COLUMN rs DOUBLE; UPDATE win SET rs = D * (p - Y)")
con.execute("""CREATE OR REPLACE TABLE win2 AS
    SELECT *, percent_rank() OVER (PARTITION BY ticker ORDER BY cash) AS rnk FROM win""")
con.execute("DROP TABLE win; ALTER TABLE win2 RENAME TO win")
n = con.sql("""SELECT count(*) trades, count(DISTINCT ticker) markets, count(DISTINCT event_ticker) events,
        min(close_time) t0, max(close_time) t1, avg((cents = 100)::int) cents_100,
        avg((cash < 100)::int) sh_small, avg((cash >= 1000)::int) sh_large FROM win""").df()
log(n.to_string(index=False))
save(n, "k_A_sample.csv")

def margins(label_sql, name):
    g = con.sql(f"""SELECT event_ticker AS cid, {label_sql} AS bin, sum(rs) sum_x, count(*) n, sum(rs*cash) sum_xw, sum(cash) w
        FROM win GROUP BY 1, 2""").df()
    g = g.rename(columns={"cid": "cluster"})
    rows = []
    for b, sub in g.groupby("bin"):
        mean = sub.sum_x.sum() / sub.n.sum()
        se, G = se_of(sub.sum_x.to_numpy(), sub.n.to_numpy(), mean)
        wmean = sub.sum_xw.sum() / sub.w.sum()
        # cash-weighted cluster SE reuses the same clusters with weights
        s = sub.sum_xw.to_numpy() - wmean * sub.w.to_numpy()
        wse = float(np.sqrt((s ** 2).sum()) / sub.w.sum() * np.sqrt(len(sub) / max(len(sub) - 1, 1)))
        rows.append(dict(scheme=name, bin=b, n=int(sub.n.sum()), n_clusters=G,
                         trade_bp=mean * 1e4, trade_se=se * 1e4, cash_bp=wmean * 1e4, cash_se=wse * 1e4))
    return g, rows

g_usd, rows = margins("CASE WHEN cash < 100 THEN 's1 <$100' WHEN cash < 1000 THEN 's2 $100-1k' ELSE 's3 >=$1k' END", "dollars")
g_rnk, rows2 = margins("CASE WHEN rnk < 1.0/3 THEN 'r1 low' WHEN rnk < 2.0/3 THEN 'r2 mid' ELSE 'r3 high' END", "within-market rank")
diff_rows = []
for scheme, g, lo, hi in [("dollars", g_usd, "s1 <$100", "s3 >=$1k"), ("within-market rank", g_rnk, "r1 low", "r3 high")]:
    a = g[g.bin == hi].rename(columns={"sum_x": "sum_x"})
    est, se, G = contrast_se(g[g.bin == hi][["cluster", "sum_x", "n"]], g[g.bin == lo][["cluster", "sum_x", "n"]])
    diff_rows.append(dict(scheme=scheme, bin=f"{hi} minus {lo}", n=np.nan, n_clusters=G,
                          trade_bp=est, trade_se=se, cash_bp=np.nan, cash_se=np.nan))
save(pd.DataFrame(rows + rows2 + diff_rows), "k_B_margin.csv")

# Market-level price and flow. The price is the last YES price before the window, not a quote.
con.execute(f"""CREATE OR REPLACE TABLE pre AS
    SELECT m.ticker, arg_max(t.yes_price, t.created_time) / 100.0 AS p0
    FROM read_parquet([{tlist}]) t JOIN mk m ON m.ticker = t.ticker
    WHERE epoch(t.created_time) < epoch(m.close_time) - 86400 AND t.yes_price > 2 AND t.yes_price < 98
    GROUP BY 1""")
flow = con.sql("""SELECT ticker, any_value(event_ticker) event_ticker, any_value(close_time) close_time,
        any_value(Y) Y,
        sum(D*cash) FILTER (WHERE cash < 100) / nullif(sum(cash) FILTER (WHERE cash < 100), 0) imb_small,
        sum(D*cash) FILTER (WHERE cash >= 1000) / nullif(sum(cash) FILTER (WHERE cash >= 1000), 0) imb_large
    FROM win GROUP BY ticker""").df()
pre = con.sql("SELECT * FROM pre").df()
d = flow.merge(pre, on="ticker", how="inner")
d = d[d.p0.between(0.02, 0.98)].copy()
d["close"] = pd.to_datetime(d.close_time, utc=True, errors="coerce")
d["close"] = d["close"].dt.tz_convert(None)
log(f"prediction markets with a pre-window price {len(d):,}; both flows {int((d.imb_small.notna() & d.imb_large.notna()).sum()):,}")

def fit(X, y):
    pen = np.full(X.shape[1], 1e-6); pen[0] = 0.0
    def obj(b):
        z = X @ b
        return np.mean(np.logaddexp(0, z) - y * z) + 0.5 * np.sum(pen * b * b)
    def grad(b):
        return X.T @ (expit(X @ b) - y) / len(y) + pen * b
    return minimize(obj, np.zeros(X.shape[1]), jac=grad, method="BFGS", options={"gtol": 1e-8, "maxiter": 200}).x

def design(df, cols):
    p = np.clip(df.p0.to_numpy(float), 1e-4, 1 - 1e-4)
    X = [np.ones(len(df)), np.log(p / (1 - p))]
    for c in cols:
        X.append(df[c].to_numpy(float))
    return np.column_stack(X)

def score(df, cols, label):
    y = df.Y.to_numpy(float)
    P = np.full(len(df), np.nan)
    base = np.full(len(df), np.nan)
    fold = df.q.to_numpy()
    for f in sorted(df.q.unique()):
        te = fold == f
        tr = df.close < df.loc[te, "qstart"].min()
        if int(tr.sum()) < 400 or int(te.sum()) == 0:
            continue
        b0 = fit(design(df[tr], []), y[tr.to_numpy()])
        b1 = fit(design(df[tr], cols), y[tr.to_numpy()])
        base[np.where(te)[0]] = expit(design(df[te], []) @ b0)
        P[np.where(te)[0]] = expit(design(df[te], cols) @ b1)
    ok = ~np.isnan(P)
    if ok.sum() == 0:
        return dict(sample=label, n=0)
    diff = (base[ok] - y[ok]) ** 2 - (P[ok] - y[ok]) ** 2
    g = df.event_ticker.to_numpy()[ok]
    codes = pd.factorize(g)[0]
    G = int(codes.max()) + 1
    sums = np.bincount(codes, weights=diff); cnt = np.bincount(codes)
    rng = np.random.default_rng(2718)
    draws = [sums[ix].sum() / cnt[ix].sum() for ix in (rng.integers(0, G, G) for _ in range(1000))]
    lo, hi = np.quantile(draws, [0.025, 0.975])
    return dict(sample=label, n=int(ok.sum()), n_clusters=int(pd.Series(g).nunique()),
                brier_price=float(((base[ok] - y[ok]) ** 2).mean()),
                dbrier=float(diff.mean()), lo=float(lo), hi=float(hi))

out = []
for need, cols, lab in [
        (["imb_small"], ["imb_small"], "small flow, forward"),
        (["imb_small", "imb_large"], ["imb_small", "imb_large"], "both flows, forward")]:
    sub = d.dropna(subset=need).copy()
    sub["q"] = sub.close.dt.to_period("Q").astype(str)
    sub["qstart"] = sub.close.dt.to_period("Q").dt.start_time
    if len(sub) < 500:
        log(f"skip {lab}: {len(sub)}")
        continue
    out.append(score(sub, cols, lab))
    log(out[-1])
save(pd.DataFrame(out), "k_C_prediction.csv")
log("done")
