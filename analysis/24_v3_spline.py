"""Market-level prediction with a pre-specified flexible price baseline.

On markets with both a small-order and a large-order imbalance before 24 hours to resolution, compare

  M1   logit of the end-of-window price
  M1s  a linear spline in that price, knots at 0.2, 0.4, 0.6 and 0.8, fixed before seeing the scores
  M2   M1s plus the small-order imbalance
  M4   M1s plus both imbalances

Missing imbalances are not filled in as zero. Five event-grouped folds share the markets. A forward
design trains only on markets resolved at least one day before the test quarter starts, so every training
label is known before any test market's prediction time. Paired intervals resample event clusters.

Output: spl_A_prediction.csv
"""
import math
from study_common import *
from scipy.special import expit
from scipy.optimize import minimize

KNOTS = (0.2, 0.4, 0.6, 0.8)
RIDGE = 1e-6

def basis(p, kind):
    p = np.clip(p, 1e-4, 1 - 1e-4)
    if kind == "M1":
        z = np.log(p / (1 - p))
        return np.column_stack([np.ones(len(p)), z])
    cols = [np.ones(len(p)), p] + [np.maximum(p - k, 0) for k in KNOTS]
    return np.column_stack(cols)

def fit(X, y):
    pen = np.full(X.shape[1], RIDGE); pen[0] = 0
    def obj(b):
        z = X @ b
        return np.mean(np.logaddexp(0, z) - y * z) + 0.5 * np.sum(pen * b * b)
    def grad(b):
        return X.T @ (expit(X @ b) - y) / len(y) + pen * b
    b = np.zeros(X.shape[1])
    f = minimize(obj, b, jac=grad, method="BFGS", options={"gtol": 1e-8, "maxiter": 200})
    return f.x

def folds_of(cluster):
    names, counts = np.unique(cluster, return_counts=True)
    loads = np.zeros(5, int); alloc = {}
    for j in np.argsort(-counts, kind="stable"):
        f = int(np.argmin(loads)); loads[f] += counts[j]; alloc[names[j]] = f
    return np.array([alloc[g] for g in cluster])

def score(df, design):
    y = df.Y.to_numpy(float)
    rows = []
    preds = {}
    for kind, cols in [("M1", []), ("M1s", []), ("M2", ["imb_small"]), ("M4", ["imb_small", "imb_large"])]:
        P = np.full(len(df), np.nan)
        for f in sorted(df.fold.unique()):
            te, tr = df.fold.to_numpy() == f, df.fold.to_numpy() != f
            if design == "forward":
                tr = (df.res < df.loc[te, "qstart"].min()).to_numpy()
            if tr.sum() < 400 or te.sum() == 0: continue
            Xtr = basis(df.p.to_numpy()[tr], "M1" if kind == "M1" else "M1s")
            Xte = basis(df.p.to_numpy()[te], "M1" if kind == "M1" else "M1s")
            if cols:
                Xtr = np.column_stack([Xtr, df.loc[tr, cols].to_numpy(float)])
                Xte = np.column_stack([Xte, df.loc[te, cols].to_numpy(float)])
            P[np.where(te)[0]] = expit(Xte @ fit(Xtr, y[tr]))
        preds[kind] = P
    for kind, P in preds.items():
        ok = ~np.isnan(P) & ~np.isnan(preds["M1"]) & ~np.isnan(preds["M1s"])
        def bloss(q):
            return (q - y[ok]) ** 2
        b, b_lin, b_spl = bloss(P[ok]), bloss(preds["M1"][ok]), bloss(preds["M1s"][ok])
        g = df.ev_cluster.to_numpy()[ok]
        def paired(diff):
            codes = pd.factorize(g)[0]; G = codes.max() + 1
            sums = np.bincount(codes, weights=diff); cnt = np.bincount(codes)
            rng = np.random.default_rng(2718)
            draws = [sums[t].sum() / cnt[t].sum() for t in (rng.integers(0, G, G) for _ in range(1000))]
            lo, hi = np.quantile(draws, [0.025, 0.975])
            return float(diff.mean()), float(lo), float(hi)
        vs_s, slo, shi = paired(b_spl - b)
        vs_l, llo, lhi = paired(b_lin - b)
        rows.append(dict(design=design, model=kind, n=int(ok.sum()), n_clusters=int(pd.Series(g).nunique()),
                         brier=float(b.mean()), dbrier_vs_spline=vs_s, spline_lo=slo, spline_hi=shi,
                         dbrier_vs_linear=vs_l, linear_lo=llo, linear_hi=lhi))
        log(f"{design} {kind}: Brier {b.mean():.5f}, vs linear {vs_l:.2e}")
    return rows

d = pd.read_csv(RES + "/mkt_A_windows_YesNo.csv")
d = d[d.win_name == "W_paper (life before 24h)"].copy()
d["res"] = pd.to_datetime(d.ev_cluster.str.split("|").str[-1], errors="coerce")
n_both_missing = int((d.imb_small.notna() & d.imb_large.isna()).sum())
d = d[(d.p_post.between(0.02, 0.98)) & d.imb_small.notna() & d.imb_large.notna() & d.res.notna()].copy()
d["p"] = d.p_post; d["qstart"] = d.res.dt.to_period("Q").dt.start_time
log(f"common sample {len(d):,} markets; small-only flow excluded {n_both_missing:,}")
d["fold"] = folds_of(d.ev_cluster.to_numpy())
rows = score(d, "cv")
# forward: a test fold is a resolution quarter; training markets resolved before that quarter
d["fold"] = d.res.dt.to_period("Q").astype(str)
rows += score(d, "forward")
save(pd.DataFrame(rows), "spl_A_prediction.csv")
log("done")
