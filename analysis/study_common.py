"""Shared definitions for the prediction-market study."""
import os, sys, time, json
import numpy as np, pandas as pd
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "base"))
import common as C                                    # noqa: E402  (full-pipeline definitions)
from common import log, MID, SIZE_B, TAU_COARSE, LOGIT, cluster_boot  # noqa: E402

REV   = os.path.dirname(os.path.abspath(__file__))
PROJECT = os.path.dirname(REV)
FULL = os.path.join(PROJECT, "work", "base")
FRES  = f"{FULL}/results"; FWORK = FULL
SMOKE = os.environ.get("STUDY_SMOKE") == "1"
SUB   = "smoke/" if SMOKE else ""
WORK = f"{PROJECT}/work/{SUB}events"; RES = f"{PROJECT}/results/{SUB}"; FIGS = f"{PROJECT}/paper/figures"; LOGS = f"{PROJECT}/work/{SUB}logs"
for d in (WORK, RES, FIGS, LOGS): os.makedirs(d, exist_ok=True)
SEED = 0
B_BOOT = 200 if SMOKE else 1000
MIN_TR = 100
GROUPS3 = ["YesNo", "SportsML", "OverUnder"]

# Fixed analysis cells ------------------------------------------------------------------------------
LATE   = "tau >= 3600 AND tau < 86400"                 # final day (main cell of the paper)
SIZE3  = "CASE WHEN usd < 100 THEN 's1 <$100' WHEN usd < 1000 THEN 's2 $100-1k' ELSE 's3 >=$1k' END"
BASE   = f"{MID} AND mid_before IS NOT NULL"           # mid-price trade with a defined proxy midpoint
METS   = {"info": "(D*(Y-mid_before))", "eff": "(D*(p-mid_before))", "RS": "(-edge)",
          "PI_100": "(D*(mid_after_100-mid_before))", "over_100": "(D*(mid_after_100-Y))"}
# Fixed probability-bp equivalence margins; these are analysis choices, not verified market tick sizes.
EQUIV_MARGINS_BP = (50.0, 25.0)

def connect(threads=64, mem="400GB"):
    threads = int(os.environ.get("ANALYSIS_THREADS", threads))
    mem = os.environ.get("ANALYSIS_MEMORY", mem)
    import duckdb
    con = duckdb.connect()
    con.execute(f"SET threads={threads}; SET memory_limit='{mem}'; SET preserve_insertion_order=false; SET temp_directory='{WORK}/duckdb_tmp';")
    return con

def save(df, name, sub=RES):
    path = f"{sub}/{name}"
    if name.endswith(".csv"): df.to_csv(path, index=False)
    elif name.endswith(".md"): open(path, "w").write(df if isinstance(df, str) else md_table(df))
    elif name.endswith(".parquet"): df.to_parquet(path, index=False)
    else: df.to_csv(path + ".csv", index=False)
    log("saved", name, getattr(df, "shape", ""))

def md_table(df, floatfmt=".2f"):
    """Minimal markdown table (no tabulate dependency)."""
    def f(v):
        if isinstance(v, float): return "" if np.isnan(v) else format(v, floatfmt)
        return str(v)
    cols = list(df.columns)
    lines = ["| " + " | ".join(map(str, cols)) + " |", "|" + "---|" * len(cols)]
    for _, r in df.iterrows(): lines.append("| " + " | ".join(f(r[c]) for c in cols) + " |")
    return "\n".join(lines)

def manifest(stage, **kw):
    path = f"{RES}/EXECUTION_MANIFEST.json"
    m = json.load(open(path)) if os.path.exists(path) else {}
    m[stage] = dict(time=time.strftime("%Y-%m-%d %H:%M:%S"), smoke=SMOKE, **kw)
    json.dump(m, open(path, "w"), indent=1, default=str)

def smoke_filter(grp):
    """Market subsample for smoke runs: deterministic hash of condition_id."""
    return "hash(condition_id) % 50 = 0" if SMOKE else "TRUE"

def mid_table(con, grp, name="e"):
    """Load work/events/mid_{grp}.parquet (built by 00_build.py) as a DuckDB table."""
    con.execute(f"CREATE OR REPLACE TABLE {name} AS SELECT * FROM '{WORK}/mid_{grp}.parquet'")
    return name

# Inference helpers -----------------------------------------------------------------------------------
def agg_sql(table, cell_exprs, metric_exprs, where="TRUE", weight=None, cluster="condition_id"):
    cells = ", ".join(f"{e} AS {n}" for n, e in cell_exprs.items())
    mets = []
    for n, e in metric_exprs.items():
        if weight is None: mets.append(f"sum({e}) AS num_{n}, count({e}) AS den_{n}")
        else: mets.append(f"sum(({e})*{weight}) AS num_{n}, sum(CASE WHEN ({e}) IS NOT NULL THEN {weight} END) AS den_{n}")
    return f"SELECT {cluster} AS condition_id, {cells}, {', '.join(mets)} FROM {table} WHERE {where} GROUP BY ALL"

def boot(con, table, cell_exprs, metric_exprs, where="TRUE", weight=None, B=B_BOOT, cluster="condition_id", seed=SEED):
    """Pooled ratio estimates with a cluster bootstrap over `cluster` (market by default; pass an event-cluster column)."""
    agg = con.sql(agg_sql(table, cell_exprs, metric_exprs, where, weight, cluster)).df()
    pairs = [(n, f"num_{n}", f"den_{n}") for n in metric_exprs]
    out = cluster_boot(agg, list(cell_exprs), pairs, B=B, seed=seed)
    out["cluster"] = cluster
    return out

def cr_se_mean(x, g):
    """Cluster-robust SE of a plain mean with clusters g (one-way).  x, g arrays."""
    x = np.asarray(x, float); n = len(x); xb = x.mean()
    s = pd.Series(x - xb).groupby(np.asarray(g)).sum().to_numpy()
    G = len(s)
    return np.sqrt((s**2).sum()) / n * np.sqrt(G / max(G - 1, 1)), G

def twoway_se_mean(x, g1, g2):
    """Cameron-Gelbach-Miller two-way cluster-robust SE of a mean: V = V1 + V2 - V12."""
    x = np.asarray(x, float); n = len(x); xb = x.mean()
    def V(g):
        s = pd.Series(x - xb).groupby(g).sum().to_numpy(); G = len(s)
        return (s**2).sum() / n**2 * G / max(G - 1, 1), G
    v1, G1 = V(np.asarray(g1)); v2, G2 = V(np.asarray(g2))
    g12 = pd.Series(np.asarray(g1)).astype(str) + "|" + pd.Series(np.asarray(g2)).astype(str)
    v12, _ = V(g12.to_numpy())
    v = max(v1 + v2 - v12, min(v1, v2) * 0.0)   # guard against negative variance
    return np.sqrt(max(v, 0.0)), G1, G2

def bh(p):
    p = np.asarray(p, float); n = len(p); o = np.argsort(p); ranked = p[o] * n / np.arange(1, n + 1)
    q = np.minimum.accumulate(ranked[::-1])[::-1]; out = np.empty(n); out[o] = np.minimum(q, 1); return out

def tost(est, se, margin):
    """Two one-sided tests of |est| < margin.  Returns p-value (max of the two one-sided p-values)."""
    from scipy.stats import norm
    p1 = 1 - norm.cdf((est + margin) / se)      # H0: est <= -margin
    p2 = 1 - norm.cdf((margin - est) / se)      # H0: est >=  margin
    return max(p1, p2)
