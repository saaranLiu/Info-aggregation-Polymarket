"""Market-level: market-level order-imbalance regressions, clarified .

For each market (>= 200 trades) and window:
  W_paper   all trades more than 24h before resolution; price = last switch midpoint inside the window (= post-window price)
  W_48_24   trades between 48h and 24h before resolution; p_pre = last switch midpoint before 48h, p_post = last before 24h
  W_7d_24   trades between 7d and 24h before resolution; p_pre at 7d, p_post at 24h
Imbalance = signed USD share of small (<$100) / large (>=$1k) / all orders in the window.
Regressions logit(Y) ~ f(price) + imbalance with (i) linear logit(price), (ii) decile dummies of the price (flexible calibration);
event-cluster (topic x resolution day) robust SEs; cross-validated Brier improvement with GroupKFold by event cluster vs plain KFold.
Outputs: results/mkt_A_windows_{grp}.csv (market table), results/mkt_B_regressions.csv
"""
from study_common import *
import statsmodels.api as sm
from sklearn.model_selection import KFold, GroupKFold
from sklearn.linear_model import LogisticRegression
con = connect()
WINDOWS = {"W_paper (life before 24h)": (86400, None), "W_48h_24h": (86400, 172800), "W_7d_24h": (86400, 7 * 86400)}
def logit(p): p = np.clip(np.asarray(p, float), 0.005, 0.995); return np.log(p / (1 - p))

rows = []
for grp in ["YesNo", "SportsML"]:
    log(f"[{grp}] market windows")
    mid_table(con, grp)
    con.execute("CREATE OR REPLACE TABLE sw AS SELECT condition_id, rn, tau, (p + p_lag_1)/2.0 AS smid FROM e WHERE D_lag IS NOT NULL AND D <> D_lag AND mn >= 200")
    parts = []
    for wname, (lo, hi) in WINDOWS.items():
        win = f"tau > {lo}" + (f" AND tau <= {hi}" if hi else "")
        pre = f"(SELECT condition_id, arg_max(smid, rn) p_pre FROM sw WHERE tau > {hi} GROUP BY 1)" if hi else "(SELECT condition_id, NULL::DOUBLE p_pre FROM sw GROUP BY 1)"
        q = f"""SELECT a.condition_id, '{wname}' AS win_name, a.cat, a.Y, a.ev_cluster, a.n_win, a.imb_small, a.imb_large, a.imb_all, a.usd_win, po.p_post, pr.p_pre
            FROM (SELECT condition_id, any_value(cat) cat, any_value(Y) Y, any_value(ev_cluster) ev_cluster, count(*) n_win, sum(usd) usd_win,
                         sum(D*usd) FILTER (usd < 100)/nullif(sum(usd) FILTER (usd < 100),0) imb_small,
                         sum(D*usd) FILTER (usd >= 1000)/nullif(sum(usd) FILTER (usd >= 1000),0) imb_large,
                         sum(D*usd)/sum(usd) imb_all
                  FROM e WHERE mn >= 200 AND {win} GROUP BY 1) a
            LEFT JOIN (SELECT condition_id, arg_max(smid, rn) p_post FROM sw WHERE tau > {lo} GROUP BY 1) po USING(condition_id)
            LEFT JOIN {pre} pr USING(condition_id)"""
        parts.append(con.sql(q).df())
    M = pd.concat(parts); save(M, f"mkt_A_windows_{grp}.csv")
    cells = [("pooled", M)] + ([(c, M[M.cat == c]) for c in ["Politics", "Crypto", "Sports"]] if grp == "YesNo" else [])
    for cname, Mc in cells:
        for wname in WINDOWS:
            d0 = Mc[(Mc.win_name == wname) & (Mc.n_win >= 20)]
            for pcol in (["p_post"] if WINDOWS[wname][1] is None else ["p_pre", "p_post"]):
                for imb in ["imb_small", "imb_large", "imb_all"]:
                    d = d0.dropna(subset=[pcol, imb]).copy(); d = d[(d[pcol] > 0.02) & (d[pcol] < 0.98)]
                    if len(d) < (50 if SMOKE else 300) or d.Y.nunique() < 2: continue
                    d["lo"] = logit(d[pcol]); d["Y"] = d.Y.astype(int)
                    for base_name, Xb in [("linear logit(p)", d[["lo"]].to_numpy()), ("price decile dummies", pd.get_dummies(pd.qcut(d[pcol].rank(method="first"), 10, labels=False), drop_first=True).to_numpy(float))]:
                        X = np.column_stack([np.ones(len(d)), Xb, d[imb].to_numpy()])
                        try:
                            r = sm.Logit(d.Y.to_numpy(), X).fit(disp=0, maxiter=300, cov_type="cluster", cov_kwds={"groups": d.ev_cluster.astype("category").cat.codes.to_numpy()})
                            r_m = sm.Logit(d.Y.to_numpy(), X).fit(disp=0, maxiter=300, cov_type="cluster", cov_kwds={"groups": d.condition_id.astype("category").cat.codes.to_numpy()})
                        except Exception as ex:
                            log("fit failed", grp, cname, wname, pcol, imb, base_name, ex); continue
                        b = r.params[-1]; se_ev = r.bse[-1]; se_mk = r_m.bse[-1]
                        # CV Brier improvement: plain KFold vs GroupKFold(event)
                        y = d.Y.to_numpy(); Xb0 = X[:, :-1]
                        def cvb(splitter, groups=None):
                            o0 = np.zeros(len(d)); o1 = np.zeros(len(d))
                            for tr, te in (splitter.split(X, y, groups) if groups is not None else splitter.split(X, y)):
                                o0[te] = LogisticRegression(C=1e6, max_iter=500).fit(Xb0[tr, 1:], y[tr]).predict_proba(Xb0[te, 1:])[:, 1]
                                o1[te] = LogisticRegression(C=1e6, max_iter=500).fit(X[tr, 1:], y[tr]).predict_proba(X[te, 1:])[:, 1]
                            return np.mean((o0 - y) ** 2) - np.mean((o1 - y) ** 2)
                        gain_plain = cvb(KFold(5, shuffle=True, random_state=SEED)); gain_group = cvb(GroupKFold(5), d.ev_cluster.to_numpy())
                        rows.append(dict(grp=grp, cell=cname, window=wname, price=pcol, imbalance=imb, baseline=base_name, n=len(d), n_events=d.ev_cluster.nunique(),
                                         b=b, se_event=se_ev, p_event=2 * (1 - __import__("scipy").stats.norm.cdf(abs(b / se_ev))), se_market=se_mk, sd_imb=d[imb].std(),
                                         brier_gain_cv_plain=gain_plain, brier_gain_cv_groupkfold=gain_group))
    manifest(f"06_market_{grp}")
save(pd.DataFrame(rows), "mkt_B_regressions.csv")
log("done")
