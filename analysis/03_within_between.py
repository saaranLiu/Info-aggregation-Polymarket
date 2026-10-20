"""audit of the within/between decomposition of info .

  A  verify the implemented decomposition against the explicit formula
       E_g[D(Y_g - m)] = Dbar_g (Y_g - mbar_g) - Cov_g(D, m);   pooled = sum_g (n_g/N) [.] ; equal-market version also reported
  B  definition diagnostic: hold prices and directions fixed, replace the terminal outcome -> within is invariant, between flips
  C  numerical example written to results/wb_C_example.md
  D  out-of-sample outcome prediction increment of direction and size, conditional on the pre-trade proxy midpoint (immediate price)
     or on the post-flow midpoint (100 trades later); GroupKFold by event cluster; paired cluster bootstrap over event clusters
  E  serial dependence of directions inside a market (why per-trade info cannot be summed as independent news)
Outputs: results/wb_A_formula_{grp}.csv, wb_B_yreplace_{grp}.csv, wb_C_example.md, wb_D_oos_{grp}.csv, wb_E_serial_{grp}.csv
"""
from study_common import *
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
con = connect()
rng = np.random.default_rng(SEED)

def decomp(df):
    """df columns: condition_id, D, Y, m, n_g. Returns pooled impl / formula / equal-market components in bp."""
    g = df.groupby("condition_id")
    Dbar = g.D.transform("mean"); mbar = g.m.transform("mean")
    impl_total = (df.D * (df.Y - df.m)).mean(); impl_between = (Dbar * (df.Y - df.m)).mean(); impl_within = ((df.D - Dbar) * (df.Y - df.m)).mean()
    per = g.apply(lambda x: pd.Series(dict(n=len(x), between=x.D.mean() * (x.Y.iloc[0] - x.m.mean()), within=-np.cov(x.D, x.m, bias=True)[0, 1] if len(x) > 1 else 0.0)), include_groups=False)
    w = per.n / per.n.sum()
    return dict(impl_total_bp=1e4 * impl_total, impl_between_bp=1e4 * impl_between, impl_within_bp=1e4 * impl_within,
                formula_between_bp=1e4 * (w * per.between).sum(), formula_within_bp=1e4 * (w * per.within).sum(),
                eqmkt_between_bp=1e4 * per.between.mean(), eqmkt_within_bp=1e4 * per.within.mean(), n_trades=len(df), n_markets=len(per))

def logit(x): x = np.clip(np.asarray(x, float), 0.005, 0.995); return np.log(x / (1 - x))

for grp in GROUPS3:
    log(f"[{grp}]")
    mid_table(con, grp)
    W = f"{LATE} AND {BASE}"
    df = con.sql(f"SELECT condition_id, ev_cluster, cat, D::DOUBLE D, Y::DOUBLE Y, mid_before m, p, usd, mid_after_100, size3, D_lag FROM e WHERE {W}").df()
    cells = [("pooled", df)] + ([(c, df[df.cat == c]) for c in ["Politics", "Crypto", "Sports", "Finance"]] if grp == "YesNo" else [])
    # ---- A
    rows = []
    for name, d in cells:
        r = decomp(d); r.update(grp=grp, cell=name); rows.append(r)
    save(pd.DataFrame(rows), f"wb_A_formula_{grp}.csv")
    # ---- B: Y replacement (prices and directions fixed)
    rows = []
    for name, d in cells:
        mk = d.groupby("condition_id").agg(Y=("Y", "first"), mbar=("m", "mean"))
        reps = {"observed Y": mk.Y, "1 - Y": 1 - mk.Y, "Y = 1 for all": pd.Series(1.0, index=mk.index), "Y = 0 for all": pd.Series(0.0, index=mk.index),
                "Y ~ Bernoulli(market mean midpoint)": pd.Series(rng.binomial(1, mk.mbar.clip(0, 1)).astype(float), index=mk.index),
                "Y shuffled across markets": pd.Series(rng.permutation(mk.Y.to_numpy()), index=mk.index)}
        for rname, ynew in reps.items():
            dd = d.copy(); dd["Y"] = dd.condition_id.map(ynew).astype(float)
            r = decomp(dd); r.update(grp=grp, cell=name, replacement=rname); rows.append(r)
    save(pd.DataFrame(rows), f"wb_B_yreplace_{grp}.csv")
    # ---- D: out-of-sample prediction increment (GroupKFold over event clusters)
    d = df.dropna(subset=["mid_after_100"]).copy()
    if grp == "YesNo" and len(d) > 3_000_000: d = d.sample(3_000_000, random_state=SEED)
    d["lm"] = logit(d.m); d["lma"] = logit(d.mid_after_100); d["lp"] = logit(d.p); d["lusd"] = np.log1p(d.usd)
    d["Dl"] = d.D * d.lusd; d["Ds2"] = d.D * (d.size3 == "s2 $100-1k"); d["Ds3"] = d.D * (d.size3 == "s3 >=$1k")
    specs = {"M0: logit(m_before)": ["lm"], "M1: M0 + D": ["lm", "D"], "M2: M0 + D + D*log(size) + D*size bins": ["lm", "D", "Dl", "Ds2", "Ds3"],
             "P0: logit(trade price)": ["lp"], "P1: P0 + D (bounce)": ["lp", "D"],
             "A0: logit(m_after_100)": ["lma"], "A1: A0 + D": ["lma", "D"], "A2: A0 + D + D*log(size) + D*size bins": ["lma", "D", "Dl", "Ds2", "Ds3"]}
    rows = []
    for name, dd in [("pooled", d)] + ([(c, d[d.cat == c]) for c in ["Politics", "Crypto"]] if grp == "YesNo" else []):
        if len(dd) < 5000: continue
        gkf = GroupKFold(n_splits=5); groups = dd.ev_cluster.to_numpy(); y = dd.Y.to_numpy()
        preds = {s: np.zeros(len(dd)) for s in specs}
        for tr, te in gkf.split(dd, y, groups):
            for s, cols in specs.items():
                X = dd[cols].to_numpy(); lr = LogisticRegression(C=1e6, max_iter=500).fit(X[tr], y[tr]); preds[s][te] = lr.predict_proba(X[te])[:, 1]
        base_of = {"M1: M0 + D": "M0: logit(m_before)", "M2: M0 + D + D*log(size) + D*size bins": "M0: logit(m_before)", "P1: P0 + D (bounce)": "P0: logit(trade price)",
                   "A1: A0 + D": "A0: logit(m_after_100)", "A2: A0 + D + D*log(size) + D*size bins": "A0: logit(m_after_100)"}
        cl = pd.Series(groups).astype("category").cat.codes.to_numpy(); G = cl.max() + 1
        for s in specs:
            pr = np.clip(preds[s], 1e-6, 1 - 1e-6); ll = -(y * np.log(pr) + (1 - y) * np.log(1 - pr)); br = (pr - y) ** 2
            row = dict(grp=grp, cell=name, spec=s, n=len(dd), n_clusters=G, logloss=ll.mean(), brier=br.mean())
            if s in base_of:
                pb = np.clip(preds[base_of[s]], 1e-6, 1 - 1e-6); llb = -(y * np.log(pb) + (1 - y) * np.log(1 - pb)); brb = (pb - y) ** 2
                dl = llb - ll; db = brb - br                   # positive = improvement over the base
                sl = np.bincount(cl, dl, G); sb = np.bincount(cl, db, G); nn = np.bincount(cl, None, G).astype(float)
                Wb = rng.multinomial(G, np.full(G, 1 / G), size=1000).astype(float)
                bl = (Wb @ sl) / (Wb @ nn); bb = (Wb @ sb) / (Wb @ nn)
                row.update(d_logloss=dl.mean(), d_logloss_lo=np.percentile(bl, 2.5), d_logloss_hi=np.percentile(bl, 97.5),
                           d_brier=db.mean(), d_brier_lo=np.percentile(bb, 2.5), d_brier_hi=np.percentile(bb, 97.5), base=base_of[s])
            rows.append(row)
    save(pd.DataFrame(rows), f"wb_D_oos_{grp}.csv")
    # ---- E: serial dependence of D within market (final-day cell)
    rows = []
    for name, dd in cells:
        dd = dd.dropna(subset=["D_lag"])
        ac = np.corrcoef(dd.D, dd.D_lag)[0, 1]; same = (dd.D == dd.D_lag).mean()
        first_of_run = dd[dd.D != dd.D_lag]
        rows.append(dict(grp=grp, cell=name, n=len(dd), D_autocorr=ac, share_same_as_previous=same, mean_run_length=1 / max(1 - same, 1e-9),
                         info_bp_all=1e4 * (dd.D * (dd.Y - dd.m)).mean(), info_bp_first_of_run=1e4 * (first_of_run.D * (first_of_run.Y - first_of_run.m)).mean(),
                         info_bp_continuation=1e4 * (dd[dd.D == dd.D_lag].D * (dd[dd.D == dd.D_lag].Y - dd[dd.D == dd.D_lag].m)).mean()))
    save(pd.DataFrame(rows), f"wb_E_serial_{grp}.csv")
    manifest(f"03_wb_{grp}")

# ---- C: numerical example
ex = pd.DataFrame({"trade": [1, 2, 3, 4], "D": [+1, -1, +1, -1], "m": [0.40, 0.50, 0.42, 0.52]})
lines = ["# Numerical example: within-market direction-price covariance does not depend on Y", "",
         "One market, four trades, proxy midpoints m and taker directions D (buys at low m, sells at high m):", "",
         "| trade | D | m |", "|---|---|---|"] + [f"| {r.trade} | {r.D:+d} | {r.m:.2f} |" for r in ex.itertuples()] + [""]
for Y in (1, 0):
    info = (ex.D * (Y - ex.m)); Dbar = ex.D.mean(); between = Dbar * (Y - ex.m.mean()); within = -np.cov(ex.D, ex.m, bias=True)[0, 1]
    lines += [f"Y = {Y}: info per trade = {list(np.round(info, 3))}; mean info = {info.mean():.4f} = between {between:.4f} + within {within:.4f}"]
lines += ["", "The within term -Cov_g(D, m) = +0.05 is identical under both outcomes: it measures that buys occur at lower midpoints than sells.",
          "The within component is outcome-invariant. Separate out-of-sample models test predictive increments conditional on the proxy, not a causal interpretation of this identity."]
save("\n".join(lines), "wb_C_example.md")
log("done")
