"""Fetch observed CLOB top-of-book quotes for the quote-validation subsample.

Source: the public pmxt Polymarket order-book archive (CC BY 4.0), normalised by
Joseph3222/polymarket-orderbook on the Hugging Face Hub (config `orderbook`).
Every `price_change` event carries the exchange-emitted best bid and best ask.

Targets: Yes/No markets with at least one main-sample trade (final day, interior
price, proxy defined) whose on-chain resolution falls inside the archive window
that overlaps the v1 contracts (2026-02-22 to 2026-04-28). For each target we keep
row groups that intersect [ts - 15 min, ts + 2 h] or [ts_lead_100 - 15 min, ts_lead_100]
for some main-sample trade, and all target-market events inside those row groups.

Only the Parquet footer, and the byte ranges of the needed columns in row groups
whose condition_id/time statistics intersect a target window, are downloaded. They
are written into a local sparse file at their original offsets, so pyarrow reads
the selected row groups locally.

Output: work/quotes/raw/date=YYYY-MM-DD.parquet, targets.parquet, token_map.parquet.
Environment: HF_ENDPOINT (default https://huggingface.co), QUOTE_DAYS, FETCH_THREADS,
QUOTE_CATS (topics, default Politics,Crypto; 'all' for every topic) and QUOTE_TAG (file suffix).
"""
import os, time, json, datetime as dt, struct, threading
from concurrent.futures import ThreadPoolExecutor
import numpy as np, pandas as pd, pyarrow as pa, pyarrow.parquet as pq, pyarrow.compute as pc, requests
from study_common import connect, WORK, PROJECT, log, LATE, BASE, C

ENDPOINT = os.environ.get("HF_ENDPOINT", "https://huggingface.co")
REPO = "Joseph3222/polymarket-orderbook"
QDIR = f"{PROJECT}/work/quotes"; RAW = f"{QDIR}/raw"; TMP = f"{QDIR}/tmp"
for d in (RAW, TMP): os.makedirs(d, exist_ok=True)
FIRST_DAY, LAST_DAY = dt.date(2026, 2, 22), dt.date(2026, 4, 28)
PRE_WINDOW = 25 * 3600
COLS = ["event_ts", "condition_id", "asset_id", "event_type", "best_bid", "best_ask"]
THREADS = int(os.environ.get("FETCH_THREADS", 48))
CATS = [c for c in os.environ.get("QUOTE_CATS", "Politics,Crypto").split(",") if c]
TAG = os.environ.get("QUOTE_TAG", "PC")
MERGE_GAP = 128 * 1024
PRE_AGE, POST_H = 900, 7200
DA = os.environ.get("DA_DIR", os.path.join(C.ROOT, "database", "daily_aligned"))


def build_targets():
    con = connect(threads=32, mem="100GB")
    lo = int(dt.datetime(2026, 2, 22, tzinfo=dt.timezone.utc).timestamp())
    hi = int(dt.datetime(2026, 4, 29, tzinfo=dt.timezone.utc).timestamp())
    t = con.sql(f"""SELECT condition_id, any_value(cat) cat, any_value(res_ts)::BIGINT res_ts, count(*) n_main
        FROM '{WORK}/mid_YesNo.parquet' WHERE {LATE} AND {BASE} GROUP BY 1
        HAVING any_value(res_ts) >= {lo} AND any_value(res_ts) < {hi}""").df()
    t["w_lo"] = t.res_ts - PRE_WINDOW; t["w_hi"] = t.res_ts
    t.to_parquet(f"{QDIR}/targets.parquet", index=False)
    ids = "','".join(t.condition_id)
    days = sorted({(dt.datetime.fromtimestamp(x, dt.timezone.utc) + dt.timedelta(days=k)).strftime("%Y-%m-%d") for x in t.res_ts for k in (-2, -1, 0)})
    files = ",".join(f"'{DA}/{d}.parquet'" for d in days if os.path.exists(f"{DA}/{d}.parquet"))
    tm = con.sql(f"SELECT DISTINCT condition_id, asset_id, outcome_seq FROM read_parquet([{files}]) WHERE condition_id IN ('{ids}')").df()
    tm.to_parquet(f"{QDIR}/token_map.parquet", index=False)
    log(f"targets: {len(t)} markets ({t.groupby('cat').size().to_dict()}), {int(t.n_main.sum()):,} main trades; token map {len(tm)} rows")
    return t


def trade_intervals(t):
    """Per market, the union of [ts - 15 min, ts + 2 h] and [ts_lead_100 - 15 min, ts_lead_100] over main-sample trades."""
    con = connect(threads=32, mem="100GB"); con.register("tg", t[["condition_id"]])
    iv = con.sql(f"""SELECT condition_id c, ts - {PRE_AGE} a, ts + {POST_H} b FROM '{WORK}/mid_YesNo.parquet' JOIN tg USING(condition_id) WHERE {LATE} AND {BASE}
        UNION ALL SELECT condition_id, ts_lead_100 - {PRE_AGE}, ts_lead_100 FROM '{WORK}/mid_YesNo.parquet' JOIN tg USING(condition_id)
        WHERE {LATE} AND {BASE} AND ts_lead_100 IS NOT NULL ORDER BY 1, 2""").df()
    out = {}
    for c, g in iv.groupby("c", sort=False):
        m = []
        for a, b in zip(g.a.to_numpy(), g.b.to_numpy()):
            if m and a <= m[-1][1]: m[-1][1] = max(m[-1][1], b)
            else: m.append([a, b])
        out[c] = np.array(m, dtype=np.int64)
    return out


def hits(iv, lo, hi):
    return iv is not None and bool(np.any((iv[:, 0] <= hi) & (iv[:, 1] >= lo)))


def signed_url(day):
    src = f"{ENDPOINT}/datasets/{REPO}/resolve/main/orderbook/date={day}/data_0.parquet"
    for attempt in range(30):
        try:
            r = requests.get(src, allow_redirects=False, timeout=30, headers={"Range": "bytes=0-0"})
        except requests.RequestException as e:
            log(f"  {day}: mirror connection error {type(e).__name__}; retrying"); time.sleep(20); continue
        if r.status_code in (301, 302, 307) and r.headers.get("location"): return r.headers["location"]
        wait = int(r.headers.get("retry-after", 30)) + 5
        log(f"  {day}: mirror returned {r.status_code}; waiting {wait}s"); time.sleep(wait)
    raise RuntimeError(f"{day}: no signed URL")


def get_range(sess, url, a, b):
    for attempt in range(12):
        try:
            r = sess.get(url, headers={"Range": f"bytes={a}-{b - 1}"}, timeout=120)
            if r.status_code == 206 and len(r.content) == b - a: return r.content
            raise IOError(f"status {r.status_code} len {len(r.content)}")
        except Exception as e:
            if attempt == 11: raise
            time.sleep(2 * (attempt + 1))


def fetch_day(day, t, ivs):
    out = f"{RAW}/date={day}__{TAG}.parquet"
    if os.path.exists(out): return dict(day=day, status="cached")
    t0 = time.time()
    url = signed_url(day)
    sess = requests.Session()
    for attempt in range(10):
        try:
            size = int(sess.get(url, headers={"Range": "bytes=0-0"}, timeout=60).headers["content-range"].split("/")[1]); break
        except Exception as e:
            log(f"  {day}: size request failed ({type(e).__name__}); retrying"); time.sleep(10 * (attempt + 1))
    else:
        raise RuntimeError(f"{day}: size request failed")
    tail = get_range(sess, url, size - 8, size)
    flen = struct.unpack("<I", tail[:4])[0]
    sparse = f"{TMP}/{day}__{TAG}.parquet"
    fd = os.open(sparse, os.O_RDWR | os.O_CREAT | os.O_TRUNC, 0o644); os.ftruncate(fd, size)
    os.pwrite(fd, get_range(sess, url, size - 8 - flen, size), size - 8 - flen)
    md = pq.ParquetFile(sparse).metadata
    names = [md.schema.column(j).name for j in range(md.num_columns)]
    ic, it = names.index("condition_id"), names.index("event_ts")
    d0 = int(dt.datetime.strptime(day, "%Y-%m-%d").replace(tzinfo=dt.timezone.utc).timestamp())
    tt = t[(t.w_hi >= d0) & (t.w_lo < d0 + 86400)].sort_values("condition_id")
    ids = tt.condition_id.to_numpy(); wlo = tt.w_lo.to_numpy(); whi = tt.w_hi.to_numpy()
    sel, ranges = [], []
    for i in range(md.num_row_groups):
        rg = md.row_group(i); sc, st = rg.column(ic).statistics, rg.column(it).statistics
        a, b = np.searchsorted(ids, sc.min, "left"), np.searchsorted(ids, sc.max, "right")
        if a < b and any(hits(ivs.get(c), st.min, st.max) for c in ids[a:b]):
            sel.append(i)
            for c in COLS:
                cc = rg.column(names.index(c))
                start = cc.dictionary_page_offset if cc.has_dictionary_page and cc.dictionary_page_offset else cc.data_page_offset
                ranges.append((start, start + cc.total_compressed_size))
    ranges.sort(); merged = []
    for a, b in ranges:
        if merged and a - merged[-1][1] <= MERGE_GAP: merged[-1][1] = max(merged[-1][1], b)
        else: merged.append([a, b])
    pieces = []
    for a, b in merged:  # cap request size at 16 MB
        while b - a > 16 * 2**20: pieces.append((a, a + 16 * 2**20)); a += 16 * 2**20
        pieces.append((a, b))
    nbytes = sum(b - a for a, b in pieces)

    local = threading.local()

    def job(ab):
        if not hasattr(local, "s"):
            local.s = requests.Session(); local.s.mount("https://", requests.adapters.HTTPAdapter(pool_connections=1, pool_maxsize=1))
        data = get_range(local.s, url, *ab); os.pwrite(fd, data, ab[0]); return len(data)
    with ThreadPoolExecutor(THREADS) as ex: got = sum(ex.map(job, pieces))
    os.close(fd)
    pf = pq.ParquetFile(sparse); idset = pa.array(ids.tolist())
    tabs = []
    for k, i in enumerate(sel):
        tb = pf.read_row_group(i, columns=COLS)
        tb = tb.filter(pc.is_in(tb["condition_id"], value_set=idset))
        if tb.num_rows:
            tabs.append(tb.append_column("rg", pa.array(np.full(tb.num_rows, i, np.int32))).append_column("row", pa.array(np.arange(tb.num_rows, dtype=np.int32))))
    tb = pa.concat_tables(tabs) if tabs else pa.table({c: pa.array([], pa.string()) for c in COLS})
    pq.write_table(tb, out + ".tmp", compression="zstd"); os.replace(out + ".tmp", out); os.remove(sparse)
    rec = dict(day=day, status="ok", row_groups=len(sel), of=md.num_row_groups, mb=round(nbytes / 1e6, 1), rows=tb.num_rows,
               markets=len(ids), seconds=round(time.time() - t0, 1))
    log(f"{day}: {len(sel)}/{md.num_row_groups} row groups, {nbytes/1e6:.0f} MB in {len(pieces)} requests, {tb.num_rows:,} rows, {rec['seconds']:.0f}s")
    return rec


if __name__ == "__main__":
    if not (os.path.exists(f"{QDIR}/targets.parquet") and os.path.exists(f"{QDIR}/token_map.parquet")): build_targets()
    t = pd.read_parquet(f"{QDIR}/targets.parquet")
    if CATS != ["all"]: t = t[t.cat.isin(CATS)]
    ivs = trade_intervals(t)
    days = os.environ.get("QUOTE_DAYS")
    alld = [(FIRST_DAY + dt.timedelta(days=k)).isoformat() for k in range((LAST_DAY - FIRST_DAY).days + 1)]
    days = days.split(",") if days else [d for r in range(3) for d in alld[r::3]]  # interleaved: every third day first
    sp = f"{QDIR}/fetch_status.json"
    status = json.load(open(sp)) if os.path.exists(sp) else {}
    for day in days:
        try:
            status[f"{day}__{TAG}"] = fetch_day(day, t, ivs)
        except Exception as e:
            log(f"{day}: FAILED {type(e).__name__}: {e}"); status[f"{day}__{TAG}"] = dict(day=day, status="failed", error=str(e)[:300])
        json.dump(status, open(sp, "w"), indent=1)
    log("done")
