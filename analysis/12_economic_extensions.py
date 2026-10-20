"""Economic extensions using the retained market-window and summary tables.

The joint-flow comparison holds the market sample fixed across four models.
Five folds keep topic-by-resolution-day clusters intact. Coefficient inference
and paired loss intervals use the same cluster unit. This stage reads saved
market-level observations; it does not rebuild the transaction database.
"""
from pathlib import Path
import csv
import json
import math
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import expit

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results"
PAPER = ROOT / "paper"
SEED = 2718
RIDGE = 1e-8
WINDOWS = ["W_paper (life before 24h)", "W_48h_24h", "W_7d_24h"]
CELLS = [("YesNo", "pooled"), ("YesNo", "Politics"),
         ("YesNo", "Crypto"), ("SportsML", "pooled")]
NAMES = ["price", "small", "large", "joint"]


def fit_logit(X, y):
    penalty = np.ones(X.shape[1]) * RIDGE
    penalty[0] = 0
    def objective(b):
        z = X @ b
        return np.mean(np.logaddexp(0, z)-y*z) + .5*np.sum(penalty*b*b)
    def gradient(b):
        return X.T @ (expit(X @ b)-y)/len(y) + penalty*b
    def hessian(b):
        p = expit(X @ b)
        return X.T @ ((p*(1-p))[:,None]*X)/len(y) + np.diag(penalty)
    b0 = np.zeros(X.shape[1]); b0[1] = 1
    fitted = minimize(objective, b0, jac=gradient, hess=hessian,
                      method="trust-exact", options={"gtol":1e-8, "maxiter":150})
    if np.max(np.abs(gradient(fitted.x))) > 1e-6:
        raise RuntimeError((fitted.message, gradient(fitted.x)))
    return fitted.x


def clusters_se(X, y, b, cluster):
    p = expit(X @ b)
    H = X.T @ ((p*(1-p))[:,None]*X)
    penalty = np.ones(X.shape[1])*RIDGE*len(y); penalty[0] = 0
    bread = np.linalg.inv(H+np.diag(penalty))
    _, codes = np.unique(cluster, return_inverse=True)
    G = codes.max()+1
    scores = np.zeros((G, X.shape[1]))
    np.add.at(scores, codes, X*(y-p)[:,None])
    V = bread @ (scores.T@scores) @ bread
    V *= G/(G-1)*(len(y)-1)/(len(y)-X.shape[1])
    return np.sqrt(np.maximum(np.diag(V),0))


def group_folds(cluster):
    names, counts = np.unique(cluster, return_counts=True)
    order = np.argsort(-counts, kind="stable")
    loads = np.zeros(5, int); allocation = {}
    for j in order:
        fold = int(np.argmin(loads)); loads[fold] += counts[j]
        allocation[names[j]] = fold
    folds = np.array([allocation[g] for g in cluster])
    assert set(folds) == set(range(5))
    return folds


def paired_interval(loss_gain, cluster):
    _, codes = np.unique(cluster, return_inverse=True)
    G = codes.max()+1
    sums = np.bincount(codes, weights=loss_gain)
    counts = np.bincount(codes)
    rng = np.random.default_rng(SEED)
    draws=[]
    for _ in range(10):
        weights=rng.multinomial(G,np.full(G,1/G),size=100)
        draws.extend((weights@sums)/(weights@counts))
    return np.percentile(draws,[2.5,97.5])


def bh(frame, pcol="p", qcol="q_bh"):
    order = np.argsort(frame[pcol].to_numpy())
    raw=frame[pcol].to_numpy()[order]*len(frame)/np.arange(1,len(frame)+1)
    q=np.minimum.accumulate(raw[::-1])[::-1]
    out=np.empty(len(frame)); out[order]=np.minimum(q,1)
    frame[qcol]=out


def main():
    data={g:pd.read_csv(RES/f"mkt_A_windows_{g}.csv") for g in ["YesNo","SportsML"]}
    rows=[]; predictions=[]
    for grp,cell in CELLS:
        for window in WINDOWS:
            d=data[grp]
            d=d[(d.win_name==window)&(d.n_win>=20)]
            if cell!="pooled": d=d[d.cat==cell]
            d=d.dropna(subset=["p_post","imb_small","imb_large","Y","ev_cluster"])
            d=d[(d.p_post>.02)&(d.p_post<.98)].sort_values("condition_id").reset_index(drop=True)
            if len(d)<300: raise RuntimeError((grp,cell,window,len(d)))
            y=d.Y.to_numpy(float); cluster=d.ev_cluster.to_numpy()
            assert len(np.unique(d.condition_id))==len(d)
            p=np.clip(d.p_post.to_numpy(),.005,.995)
            xb=np.column_stack([np.ones(len(d)),np.log(p/(1-p))])
            if grp=="YesNo" and cell=="pooled":
                topics=pd.get_dummies(d.cat,drop_first=True,dtype=float).to_numpy()
                xb=np.column_stack([xb,topics])
            small=d.imb_small.to_numpy();large=d.imb_large.to_numpy()
            designs=[xb,np.column_stack([xb,small]),np.column_stack([xb,large]),
                     np.column_stack([xb,small,large])]
            folds=group_folds(cluster)
            oof=[]
            for X in designs:
                pr=np.zeros(len(d))
                for k in range(5):
                    train=folds!=k;test=~train
                    b=fit_logit(X[train],y[train]);pr[test]=expit(X[test]@b)
                oof.append(np.clip(pr,1e-12,1-1e-12))
            b=fit_logit(designs[-1],y)
            se=clusters_se(designs[-1],y,b,cluster)
            base_brier=(oof[0]-y)**2
            base_ll=-(y*np.log(oof[0])+(1-y)*np.log(1-oof[0]))
            for j,name in enumerate(NAMES[1:],1):
                gain=base_brier-(oof[j]-y)**2
                ll=base_ll+(y*np.log(oof[j])+(1-y)*np.log(1-oof[j]))
                lo,hi=paired_interval(gain,cluster)
                rows.append(dict(grp=grp,cell=cell,window=window,model=name,n=len(d),
                                 n_events=len(np.unique(cluster)),base_brier=base_brier.mean(),
                                 brier_gain=gain.mean(),brier_gain_lo=lo,brier_gain_hi=hi,
                                 logloss_gain=ll.mean(),small_b=b[-2],small_se=se[-2],
                                 large_b=b[-1],large_se=se[-1]))
            out=d[["condition_id","win_name","cat","Y","ev_cluster","p_post","imb_small","imb_large"]].copy()
            out.insert(0,"analysis_cell",cell);out.insert(0,"grp",grp);out["fold"]=folds
            for name,pr in zip(NAMES,oof):out["pred_"+name]=pr
            predictions.append(out)
            print(grp,cell,window,"n",len(d),"small",round(b[-2],3),"large",round(b[-1],3),
                  "joint Brier",round(rows[-1]["brier_gain"],6),flush=True)
    result=pd.DataFrame(rows);result.to_csv(RES/"joint_flow_models.csv",index=False)
    pd.concat(predictions).to_csv(RES/"joint_flow_predictions.csv",index=False)
    coefficient=[]
    for _,r in result[result.model=="joint"].iterrows():
        for term in ["small","large"]:
            est=r[term+"_b"];se=r[term+"_se"]
            coefficient.append(dict(grp=r.grp,cell=r.cell,window=r.window,term=term,
                                    b=est,se=se,p=math.erfc(abs(est/se)/math.sqrt(2))))
    coef=pd.DataFrame(coefficient);bh(coef);coef.to_csv(RES/"joint_flow_coefficients.csv",index=False)

    tex=[r"\begin{tabular}{lrrrr}",r"\toprule",
         r"Segment & Markets & Small-flow $\beta$ & Large-flow $\beta$ & $10^4\Delta$Brier [95\% CI] \\",r"\midrule"]
    main_rows=result[(result.window==WINDOWS[0])&(result.model=="joint")]
    def scaled(value):
        value=float(value)*1e4
        return f"{value:+.2f}" if abs(value)<.05 else f"{value:+.1f}"
    for _,r in main_rows.iterrows():
        label=r.cell if r.cell!="pooled" else ("Yes/No" if r.grp=="YesNo" else "Sports moneyline")
        tex.append(f"{label} & {r.n:,} & {r.small_b:+.2f} ({r.small_se:.2f}) & {r.large_b:+.2f} ({r.large_se:.2f}) & "
                   f"{scaled(r.brier_gain)} [{scaled(r.brier_gain_lo)}, {scaled(r.brier_gain_hi)}]"+r" \\")
    tex += [r"\bottomrule",r"\end{tabular}"]
    (PAPER/"tables/joint_flow.tex").write_text("\n".join(tex)+"\n")

    weights=pd.read_csv(RES/"audit_C_weights_YesNo.csv"); weights=weights[weights.cat.isna()]
    cov=[]
    for metric in ["eff","info","RS"]:
        sub=weights[weights.metric==metric].set_index("weight")
        e=sub.loc["count","est"]*1e4
        for weight in ["usd","shares"]:
            ew=sub.loc[weight,"est"]*1e4
            cov.append(dict(metric=metric,weight=weight,unweighted_bp=e,weighted_bp=ew,
                            normalized_covariance_bp=ew-e))
    cov=pd.DataFrame(cov);cov.to_csv(RES/"weight_covariance.csv",index=False)
    assert abs(cov[(cov.weight=="usd")&(cov.metric=="eff")].normalized_covariance_bp.iloc[0]
               -cov[(cov.weight=="usd")&(cov.metric=="info")].normalized_covariance_bp.iloc[0]
               -cov[(cov.weight=="usd")&(cov.metric=="RS")].normalized_covariance_bp.iloc[0])<1e-8
    tex=[r"\begin{tabular}{lrrr}",r"\toprule",
         r"Component & Per trade & Per USD & Covariance shift \\",r"\midrule"]
    for metric,label in [("eff","Execution cost"),("info","Information"),("RS","Settlement margin")]:
        r=cov[(cov.weight=="usd")&(cov.metric==metric)].iloc[0]
        tex.append(f"{label} & {r.unweighted_bp:.0f} & {r.weighted_bp:.0f} & {r.normalized_covariance_bp:+.0f}"+r" \\")
    tex += [r"\bottomrule",r"\end{tabular}"]
    (PAPER/"tables/weight_covariance.tex").write_text("\n".join(tex)+"\n")
    shifts=[]
    tex=[r"\begin{tabular}{lrr}",r"\toprule",
         r"Component & Market shift & Order shift \\",r"\midrule"]
    for metric,label in [("eff","Execution cost"),("info","Information"),("RS","Settlement margin")]:
        z=weights[weights.metric==metric].set_index("weight")
        market=(z.loc["count","est"]-z.loc["equal_market","est"])*1e4
        order=(z.loc["usd","est"]-z.loc["count","est"])*1e4
        shifts.append(dict(metric=metric,market_activity_shift_bp=market,order_notional_shift_bp=order))
        tex.append(f"{label} & {market:+.0f} & {order:+.0f}"+r" \\")
    tex += [r"\bottomrule",r"\end{tabular}"]
    pd.DataFrame(shifts).to_csv(RES/"weight_activity_shifts.csv",index=False)
    (PAPER/"tables/weight_shifts.tex").write_text("\n".join(tex)+"\n")

    h=pd.read_csv(RES/"inf_A_horizon_samples.csv")
    h=h[(h.grp=="YesNo")&(h["sample"]=="horizon")]
    gradients=[]
    tex=[r"\begin{tabular}{lrrr}",r"\toprule",
         r"Topic & $\Delta\info$ & $\Delta\PI_{100}$ & $\Delta\ovr_{100}$ \\",r"\midrule"]
    for topic in ["Politics","Crypto"]:
        small=h[h.cell==topic+" | s1 <$100"].set_index("metric")
        large=h[h.cell==topic+" | s3 >=$1k"].set_index("metric")
        values={m:float(large.loc[m,"est_bp"]-small.loc[m,"est_bp"])
                for m in ["info","PI_100","over_100"]}
        assert abs(values["PI_100"]-values["info"]-values["over_100"])<1e-8
        gradients.append(dict(topic=topic,**values))
        tex.append(topic+" & "+" & ".join(f"{values[m]:+.0f}" for m in ["info","PI_100","over_100"])+r" \\")
    tex += [r"\bottomrule",r"\end{tabular}"]
    pd.DataFrame(gradients).to_csv(RES/"size_response_gradients.csv",index=False)
    (PAPER/"tables/size_gradients.tex").write_text("\n".join(tex)+"\n")

    a=pd.read_csv(RES/"acct_A_splithalf_YesNo.csv");b=pd.read_csv(RES/"acct_B_topdecile_YesNo.csv")
    a=a[a.metric=="info"];b=b[b.metric=="info"]
    accounts=a.merge(b,on=["grp","split","metric"],suffixes=("_account","_trade"))
    accounts.to_csv(RES/"account_persistence_summary.csv",index=False)
    tex=[r"\begin{tabular}{lrrrrr}",r"\toprule",
         r"Split & Accounts & Slope & Correlation & Top-decile B (bp) & Eligible B (bp) \\",r"\midrule"]
    for _,r in accounts.iterrows():
        label="Within-account time" if r.split=="half_time" else "Disjoint market sets"
        tex.append(f"{label} & {r.n_accounts:,} & {r.slope_B_on_A:.2f} & {r.corr_AB:.2f} & "
                   f"{r.mean_B_trade*1e4:.0f} ({r.se_event*1e4:.0f}) & {r.population_mean_B*1e4:.0f}"+r" \\")
    tex += [r"\bottomrule",r"\end{tabular}"]
    (PAPER/"tables/account_persistence.tex").write_text("\n".join(tex)+"\n")
    config=dict(seed=SEED,windows=WINDOWS,cells=CELLS,models=NAMES,folds=5,
                cluster="topic by resolution date",bootstrap_draws=1000,ridge_average_loss=RIDGE,
                sample="n_win>=20; .02<p_post<.98; both size imbalances observed; common sample for all models",
                pooled_controls="logit(p_post), topic indicators",cell_controls="logit(p_post)",
                new_coefficient_family=24,source="saved market-window observations and summary CSVs")
    (RES/"ECONOMIC_EXTENSIONS.json").write_text(json.dumps(config,indent=2)+"\n")
    report=["# Economic extensions", "", "The analysis uses the retained market-window observations and summary estimates.",
            "", "## Joint-flow comparison", "",
            "Four nested outcome-logit models use exactly the same markets in each topic/window: price, price plus small flow, price plus large flow, and price plus both. The pooled Yes/No baseline includes topic indicators. All models use five folds grouped by topic and resolution date. Paired cluster bootstrap intervals resample held-out loss differences. Coefficient tests form a separate, exploratory 24-test family.", "",
            "| Segment | Window | Markets | Small coefficient (SE) | Large coefficient (SE) | Joint Brier gain [95% CI] |",
            "|---|---|---:|---:|---:|---:|"]
    for _,r in result[result.model=="joint"].iterrows():
        report.append(f"| {r.grp}/{r.cell} | {r.window} | {r.n} | {r.small_b:.4f} ({r.small_se:.4f}) | {r.large_b:.4f} ({r.large_se:.4f}) | {r.brier_gain:.6f} [{r.brier_gain_lo:.6f}, {r.brier_gain_hi:.6f}] |")
    report += ["", "## Weighting identities", "",
               "For any positive weight w, E_w[x]-E[x]=Cov(w,x)/E[w]. The USD weighting shift of RS equals the shift of execution cost minus the shift of information. Across markets, the per-trade minus equal-market mean equals Cov_market(n_market, mean_x_market)/E_market[n_market]. The source tables identify these normalized covariances exactly on the common horizon sample.", "",
               "## Account and size summaries", "",
               "The account summary retains separate equal-account slopes and trade-weighted held-out means, under within-account time and disjoint-market hash splits. Size gradients are differences between the large and small bin means on the observed horizon. They are descriptive; within-topic covariance is required for inference on those differences."]
    (RES/"ECONOMIC_EXTENSIONS.md").write_text("\n".join(report)+"\n")
    print("Wrote extensions, tables, and prediction figure.",flush=True)


if __name__=="__main__":main()
