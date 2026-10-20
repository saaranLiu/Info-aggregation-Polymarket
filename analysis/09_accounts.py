"""account stability with a price-adjusted measure .

Taker accounts with >= 50 final-day mid-price trades in each half (split by time within account, or by market-hash parity).
For winner share s, information info = D(Y - m) and realized spread RS we report: slope of half B on half A, the between-half
correlation (reliability), the implied variance of true account effects (Cov(A,B)), and the out-of-sample second-half mean
of accounts in the top decile of the first half with an event-cluster SE.
Outputs: results/acct_A_splithalf_{grp}.csv, acct_B_topdecile_{grp}.csv
"""
from study_common import *
con = connect()
MINH = 20 if SMOKE else 50
for grp in ["YesNo", "SportsML"]:
    log(f"[{grp}]")
    mid_table(con, grp)
    d = con.sql(f"""SELECT taker, condition_id, ev_cluster, ts, (I=1)::int s, D*(Y-mid_before) info, -edge RS, hash(condition_id || 'parity') % 2 AS par,
                           row_number() OVER (PARTITION BY taker ORDER BY ts, rn) AS k, count(*) OVER (PARTITION BY taker) AS n
                    FROM e WHERE {LATE} AND {BASE} QUALIFY n >= {2 * MINH}""").df()
    d["half_time"] = np.where(d.k <= d.n / 2, "A", "B"); d["half_par"] = np.where(d.par == 0, "A", "B")
    rows, rows2 = [], []
    for split in ["half_time", "half_par"]:
        g = d.groupby(["taker", split]).agg(n=("s", "size"), s=("s", "mean"), info=("info", "mean"), RS=("RS", "mean")).reset_index()
        w = g.pivot(index="taker", columns=split, values=["n", "s", "info", "RS"]).dropna()
        if ("n", "B") not in w.columns: continue
        w = w[(w[("n", "A")] >= MINH) & (w[("n", "B")] >= MINH)]
        if len(w) < 30: continue
        for met in ["s", "info", "RS"]:
            a, b = w[(met, "A")].to_numpy(), w[(met, "B")].to_numpy()
            slope = np.cov(a, b)[0, 1] / np.var(a, ddof=1); corr = np.corrcoef(a, b)[0, 1]
            rows.append(dict(grp=grp, split=split, metric=met, n_accounts=len(w), mean_A=a.mean(), mean_B=b.mean(), slope_B_on_A=slope, corr_AB=corr,
                             var_A=np.var(a, ddof=1), cov_AB=np.cov(a, b)[0, 1], implied_sd_true_effect=np.sqrt(max(np.cov(a, b)[0, 1], 0)),
                             top_decile_A_mean_A=a[a >= np.quantile(a, .9)].mean(), top_decile_A_mean_B=b[a >= np.quantile(a, .9)].mean(),
                             bottom_decile_A_mean_B=b[a <= np.quantile(a, .1)].mean()))
            # out-of-sample second-half mean of the top decile with event-cluster SE on trades
            top = w.index[a >= np.quantile(a, .9)]
            tb = d[(d.taker.isin(top)) & (d[split] == "B")]
            se, G = cr_se_mean(tb[met].to_numpy(), tb.ev_cluster.to_numpy())
            rows2.append(dict(grp=grp, split=split, metric=met, n_top_accounts=len(top), n_trades_B=len(tb), mean_B=tb[met].mean(), se_event=se, G_event=G, population_mean_B=d[d[split] == "B"][met].mean()))
    save(pd.DataFrame(rows), f"acct_A_splithalf_{grp}.csv"); save(pd.DataFrame(rows2), f"acct_B_topdecile_{grp}.csv")
    manifest(f"09_accounts_{grp}")
log("done")
