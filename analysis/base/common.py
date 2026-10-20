"""Shared definitions for the full-population pipeline (see SPEC_v1.md)."""
import duckdb, os, sys, time, json
import numpy as np, pandas as pd

ROOT = os.environ.get("DATA_ROOT", os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "data"))
DA   = f"{ROOT}/database/daily_aligned/*.parquet"
OF   = f"{ROOT}/database/OrderFilled/*.parquet"
CTF  = f"{ROOT}/database/CTF"
PROJ = os.path.dirname(os.path.abspath(__file__))
SMOKE = os.environ.get('SMOKE') == '1'          # SMOKE=1: smallest group only, fewer bootstrap reps, outputs under smoke/
SUB = "smoke/" if SMOKE else ""
PROJECT = os.path.dirname(os.path.dirname(PROJ))
WORK = f"{PROJECT}/work/{SUB}base"; RES = f"{WORK}/results"; LOGS = f"{WORK}/logs"; FIGS = f"{WORK}/figures"
for d in (WORK, RES, LOGS, FIGS): os.makedirs(d, exist_ok=True)

GROUPS   = ['OverUnder'] if SMOKE else ['YesNo', 'SportsML', 'OverUnder', 'UpDown']
TRADE_H  = [1, 2, 5, 10, 20, 50, 100]
PRE_H    = [1, 5, 20, 100]
CLOCK_H  = {'YesNo': [600, 3600, 21600, 86400], 'SportsML': [60, 600, 3600, 21600],
            'OverUnder': [60, 600, 3600, 21600], 'UpDown': [10, 30, 60, 120, 300, 600, 3600]}
PRE_CLOCK = {'YesNo': 3600, 'SportsML': 600, 'OverUnder': 600, 'UpDown': 30}
MAIN_MIN_TRADES = 100
SEED = 0
B_BOOT = 50 if SMOKE else 1000

# SQL snippets ---------------------------------------------------------------
TAU_COARSE = """CASE WHEN tau < 3600 THEN 'a<1h' WHEN tau < 86400 THEN 'b1-24h' WHEN tau < 7*86400 THEN 'c1-7d'
                     WHEN tau < 30*86400 THEN 'd7-30d' ELSE 'e>30d' END"""
TAU_FINE = """CASE WHEN tau < 600 THEN 'a<10m' WHEN tau < 3600 THEN 'b10m-1h' WHEN tau < 6*3600 THEN 'c1-6h' WHEN tau < 86400 THEN 'd6-24h'
                   WHEN tau < 3*86400 THEN 'e1-3d' WHEN tau < 7*86400 THEN 'f3-7d' WHEN tau < 30*86400 THEN 'g7-30d' ELSE 'h>30d' END"""
TAU_UPDOWN = """CASE WHEN tau < 30 THEN 'a<30s' WHEN tau < 60 THEN 'b30-60s' WHEN tau < 120 THEN 'c1-2m' WHEN tau < 300 THEN 'd2-5m'
                     WHEN tau < 900 THEN 'e5-15m' WHEN tau < 3600 THEN 'f15-60m' ELSE 'g>1h' END"""
PRICE_LAYER = """CASE WHEN abs(p-0.5) < 0.1 THEN 'L1 |p-.5|<.1' WHEN abs(p-0.5) < 0.2 THEN 'L2 <.2' WHEN abs(p-0.5) < 0.3 THEN 'L3 <.3'
                      WHEN abs(p-0.5) < 0.4 THEN 'L4 <.4' WHEN abs(p-0.5) < 0.48 THEN 'L5 <.48' ELSE 'L6 extreme' END"""
SIZE_B = """CASE WHEN usd < 10 THEN 's1 <$10' WHEN usd < 100 THEN 's2 $10-100' WHEN usd < 1000 THEN 's3 $100-1k'
                 WHEN usd < 10000 THEN 's4 $1k-10k' ELSE 's5 >$10k' END"""
MID = "abs(p-0.5) < 0.3"
LOGIT = lambda x: f"ln(least(greatest({x},0.005),0.995)/(1-least(greatest({x},0.005),0.995)))"

def connect(threads=64, mem='400GB'):
    threads = int(os.environ.get("ANALYSIS_THREADS", threads))
    mem = os.environ.get("ANALYSIS_MEMORY", mem)
    con = duckdb.connect()
    con.execute(f"SET threads={threads}; SET memory_limit='{mem}'; SET preserve_insertion_order=false; SET temp_directory='{WORK}/duckdb_tmp';")
    con.execute(f"SELECT setseed({SEED/100.0})")
    return con

_T0 = time.time()
def log(*a):
    print(f"[{time.strftime('%H:%M:%S')} +{time.time()-_T0:7.0f}s]", *a, flush=True)

def save(df, name):
    path = f"{RES}/{name}"
    if name.endswith('.csv'): df.to_csv(path, index=False)
    else: df.to_parquet(path, index=False)
    log("saved", name, df.shape)

# Metric expressions on an event table ----------------------------------------
# event table columns: p, D, Y, I, q, usd, tau, theta, p_lead_{h}, p_lag_{h}, p_clk_{H}, p_pre_clk
def dq(col):   return f"((2*Y-1)*({col} - p))"            # winner-axis change
def R(col):    return f"(D*({col} - p))"                   # signed response (Y-free)
def RLO(col):  return f"(D*({LOGIT(col)} - {LOGIT('p')}))"
def PRE(col):  return f"(D*(p - {col}))"                   # pre-trade momentum in trade direction
def horizon_cols(grp):
    cols = [(f"h{h}", f"p_lead_{h}") for h in TRADE_H] if grp != 'UpDown' else []
    cols += [(f"c{H}s", f"p_clk_{H}") for H in CLOCK_H[grp]]
    return cols

# Market-cluster bootstrap -------------------------------------------------------
def cluster_boot(agg, cell_cols, pairs, B=B_BOOT, seed=SEED, market_col='condition_id'):
    """agg: per-market-per-cell aggregates with columns cell_cols + [num_k, den_k for each (name, num, den) in pairs].
    Returns tidy DataFrame: cell cols, metric, est (pooled ratio), se, lo, hi, est_mktmean, n_markets, n_obs."""
    rng = np.random.default_rng(seed)
    out = []
    for key, g in agg.groupby(cell_cols, dropna=False, observed=True):
        key = key if isinstance(key, tuple) else (key,)
        M = len(g)
        if M == 0: continue
        NUM = np.nan_to_num(g[[p[1] for p in pairs]].to_numpy(float))
        DEN = np.nan_to_num(g[[p[2] for p in pairs]].to_numpy(float))
        boots = np.full((B, len(pairs)), np.nan)
        if M >= 2:
            for b0 in range(0, B, 100):
                nb = min(100, B - b0)
                Wb = rng.multinomial(M, np.full(M, 1.0 / M), size=nb).astype(float)   # same market resample for all metrics
                with np.errstate(invalid='ignore', divide='ignore'):
                    boots[b0:b0+nb] = (Wb @ NUM) / (Wb @ DEN)
        for j, (name, num, den) in enumerate(pairs):
            nv, dv = NUM[:, j], DEN[:, j]; ok = dv > 0
            if ok.sum() == 0: continue
            est = nv.sum() / dv.sum(); mm = float(np.mean(nv[ok] / dv[ok]))
            bj = boots[:, j]; bj = bj[np.isfinite(bj)]
            se = bj.std(ddof=1) if len(bj) > 2 else np.nan
            lo, hi = (np.percentile(bj, [2.5, 97.5]) if len(bj) > 2 else (np.nan, np.nan))
            row = dict(zip(cell_cols, key)); row.update(metric=name, est=est, se=se, lo=lo, hi=hi, est_mktmean=mm, n_markets=int(ok.sum()), n_obs=dv.sum())
            out.append(row)
    return pd.DataFrame(out)

def agg_sql(table, cell_exprs, metric_exprs, where="TRUE", weight=None):
    """Build per-market-per-cell aggregate SQL. cell_exprs: dict name->sql; metric_exprs: dict name->sql expression (per row).
    For each metric: num = sum(expr [*weight]), den = count/sum(weight) over non-null expr."""
    cells = ", ".join(f"{e} AS {n}" for n, e in cell_exprs.items())
    mets = []
    for n, e in metric_exprs.items():
        if weight is None:
            mets.append(f"sum({e}) AS num_{n}, count({e}) AS den_{n}")
        else:
            mets.append(f"sum(({e})*{weight}) AS num_{n}, sum(CASE WHEN ({e}) IS NOT NULL THEN {weight} END) AS den_{n}")
    return f"SELECT condition_id, {cells}, {', '.join(mets)} FROM {table} WHERE {where} GROUP BY ALL"

def boot_from_sql(con, table, cell_exprs, metric_exprs, where="TRUE", weight=None, B=B_BOOT):
    agg = con.sql(agg_sql(table, cell_exprs, metric_exprs, where, weight)).df()
    pairs = [(n, f"num_{n}", f"den_{n}") for n in metric_exprs]
    return cluster_boot(agg, list(cell_exprs), pairs, B=B)

def manifest(stage, **kw):
    path = f"{RES}/RUN_MANIFEST.json"
    m = json.load(open(path)) if os.path.exists(path) else {}
    m[stage] = dict(time=time.strftime('%Y-%m-%d %H:%M:%S'), **kw)
    json.dump(m, open(path, 'w'), indent=1, default=str)
