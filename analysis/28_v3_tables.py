"""Tables and macros for the third-round checks. Reads results/; writes paper/tables and paper/numbers_v3.tex."""
import math
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RES, TABLES = ROOT / "results", ROOT / "paper" / "tables"
from manuscript_format import tex_decimal, cell, probability
M = {}

f0 = tex_decimal
def pct(x): return f"{100 * float(x):.1f}"

C = pd.read_csv(RES / "ctr_A_contrasts.csv")
def crow(sample, metric, name):
    return C[(C["sample"] == sample) & (C.metric == metric) & C.contrast.str.startswith(name)].iloc[0]
t1, t2, t3 = crow("h100", "over", "theta1"), crow("h100", "over", "theta2"), crow("h100", "over", "theta3")
for pre, r in [("ThOne", t1), ("ThTwo", t2), ("ThThree", t3)]:
    M[pre], M[pre + "SE"], M[pre + "P"] = f0(r.est_bp), f"{r.se_ev_bp:.1f}", probability(r.p)
    M[pre + "W"] = f"{r.se_2way_bp:.1f}"
i2 = crow("h100", "info", "theta2"); p2 = crow("h100", "pi", "theta2")
M["ThTwoInfo"], M["ThTwoInfoSE"] = f0(i2.est_bp), f"{i2.se_ev_bp:.1f}"
M["ThTwoPi"], M["ThTwoPiSE"] = f0(p2.est_bp), f"{p2.se_ev_bp:.1f}"
for k, pre in [("clock1h_R", "R"), ("clock1h_S", "S")]:
    r = crow(k, "over", "theta2")
    M[f"ThTwo{pre}"], M[f"ThTwo{pre}SE"] = f0(r.est_bp), f"{r.se_ev_bp:.1f}"

def tost(est, se, m):
    z1, z2 = (m - est) / se, (m + est) / se
    def tail(z): return 0.5 * (1 - math.erf(z / math.sqrt(2)))
    return max(tail(z1), tail(z2))
e50 = crow("h100", "over", "equiv Crypto small 50")
M["EqFiftyP"] = f"{100*tost(e50.est_bp, e50.se_ev_bp, 50):.1f}"
M["EqTwentyP"] = f"{100*tost(e50.est_bp, e50.se_ev_bp, 25):.1f}"

lines = [r"\begin{tabular}{lrr}", r"\toprule", r"Contrast & Estimate & Two-way SE \\", r"\midrule"]
for lab, r in [(r"Politics, below \$100", t1), (r"Politics, large minus small", t2), (r"Politics minus Crypto, small", t3)]:
    lines.append(f"{lab} & {cell(r.est_bp, r.se_ev_bp)} & {r.se_2way_bp:.1f}" + r" \\")
lines += [r"\addlinespace", f"Alignment, large minus small & {cell(i2.est_bp, i2.se_ev_bp)} & {i2.se_2way_bp:.1f}" + r" \\",
          f"Response, large minus small & {cell(p2.est_bp, p2.se_ev_bp)} & {p2.se_2way_bp:.1f}" + r" \\",
          r"\bottomrule", r"\end{tabular}"]
(TABLES / "contrasts.tex").write_text("\n".join(lines) + "\n")

A, B = pd.read_csv(RES / "dir_A_agreement.csv"), pd.read_csv(RES / "dir_B_metrics.csv")
M["DirAgr"], M["DirAgrUsd"] = pct(A[A.cell == "pooled"].share_agree.iloc[0]), pct(A[A.cell == "pooled"].share_agree_usd.iloc[0])
M["DirZero"] = pct(A[A.cell == "pooled"].share_zero_tick.iloc[0])
def bget(cell, metric):
    return B[(B.step == "fixed midpoint") & (B.cell == cell) & (B.metric == metric)].iloc[0]
ps, pl = bget("Politics | s1 <$100", "over"), bget("Politics | s3 >=$1k", "over")
M["DirPolS"], M["DirPolSSE"], M["DirPolSTk"] = f0(ps.diff_bp), f"{ps.se_ev_bp:.1f}", f0(ps.tick_bp)
M["DirPolL"], M["DirPolLSE"], M["DirPolLTk"] = f0(pl.diff_bp), f"{pl.se_ev_bp:.1f}", f0(pl.tick_bp)
rb = B[(B.step == "rebuilt midpoint") & (B.cell == "Politics | s1 <$100") & (B.metric == "over")].iloc[0]
M["DirReb"] = f0(rb.diff_bp)
lines = [r"\begin{tabular}{lrrrr}", r"\toprule",
         r"Cell & Agree & On-chain $\ovr_{100}$ & Tick rule & Difference \\", r"\midrule"]
for cname, lab in [("pooled", "Yes/No"), ("Politics | s1 <$100", r"Politics $<\$100$"),
                    ("Politics | s3 >=$1k", r"Politics $\ge\$1$k"), ("Crypto | s1 <$100", r"Crypto $<\$100$")]:
    a = A[A.cell == cname].iloc[0]; r = bget(cname, "over")
    lines.append(f"{lab} & {pct(a.share_agree)}\\% & {f0(r.onchain_bp)} & {f0(r.tick_bp)} & {cell(r.diff_bp, r.se_ev_bp)}" + r" \\")
lines += [r"\bottomrule", r"\end{tabular}"]
(TABLES / "direction.tex").write_text("\n".join(lines) + "\n")

O = pd.read_csv(RES / "dir_C_order.csv")
o = O[O.cell == "Politics+Crypto"].iloc[0]
M["OrdDiff"], M["OrdSE"] = f"{o.diff_bp:.1f}".replace("-", "\\ensuremath{-}"), f"{o.se_ev_bp:.1f}"

R = pd.read_csv(RES / "ret_A_retention.csv")
rp = R[(R.cat == "Politics") & (R.size3 == "s1 <$100")].iloc[0]
rc = R[(R.cat == "Crypto") & (R.size3 == "s1 <$100")].iloc[0]
M["RetPol"], M["RetCry"] = pct(rp.share_kept), pct(rc.share_kept)
M["RetPolH"], M["RetCryH"] = f"{rp.med_hours:.1f}", f"{rc.med_hours * 60:.1f}"
lines = [r"\begin{tabular}{lrrrrr}", r"\toprule",
         r"Cell & Kept & Median & $p_{90}$ & RS kept & RS dropped \\", r"\midrule"]
for _, r in R.iterrows():
    lab = ("Politics" if r["cat"] == "Politics" else "Crypto") + " " + {"s1 <$100": r"$<\$100$", "s2 $100-1k": r"\$100--1k", "s3 >=$1k": r"$\ge\$1$k"}[r["size3"]]
    lines.append(f"{lab} & {pct(r.share_kept)}\\% & {r.med_hours:.1f} h & {r.p90_hours:.1f} h & {cell(r.rs_kept_bp, r.rs_kept_se)} & {cell(r.rs_dropped_bp, r.rs_dropped_se)}" + r" \\")
lines += [r"\bottomrule", r"\end{tabular}"]
(TABLES / "retention.tex").write_text("\n".join(lines) + "\n")

K = pd.read_csv(RES / "con_A_concentration.csv")
def kget(cat, w): return K[(K.cat == cat) & (K.weight == w)].iloc[0]
u, p = kget("Yes/No", "usd"), kget("Politics", "usd")
M["ConUsd"], M["ConUsdSE"] = f0(u.mean_bp), f"{u.mean_se_bp:.1f}"
M["ConDrop"], M["ConDropSE"] = f0(u.drop5_bp), f"{u.drop5_se_bp:.1f}"
M["ConTopFive"] = pct(u.top5_abs_share)
M["ConPol"], M["ConPolSE"] = f0(p.mean_bp), f"{p.mean_se_bp:.1f}"
M["ConPolDrop"], M["ConPolDSE"] = f0(p.drop5_bp), f"{p.drop5_se_bp:.1f}"
M["ConPolTop"] = pct(p.top5_abs_share)
D = pd.read_csv(RES / "con_C_politics_dates.csv")
M["ConDateSh"], M["ConDateBp"] = pct(D.usd_share.iloc[0]), f0(D.contrib_bp.iloc[0])
lines = [r"\begin{tabular}{lrrrr}", r"\toprule",
         r"Sample & Full sample & Drop 5 dates & Drop largest & Top 5 share \\", r"\midrule"]
for cat, w, lab in [("Yes/No", "trade", "Yes/No, per trade"), ("Yes/No", "usd", "Yes/No, per dollar"),
                    ("Politics", "usd", "Politics, per dollar"), ("Crypto", "usd", "Crypto, per dollar")]:
    r = kget(cat, w)
    lines.append(f"{lab} & {cell(r.mean_bp, r.mean_se_bp)} & {cell(r.drop5_bp, r.drop5_se_bp)} & {f0(r.drop1_bp)} & {pct(r.top5_abs_share)}\\%" + r" \\")
lines += [r"\bottomrule", r"\end{tabular}"]
(TABLES / "concentration.tex").write_text("\n".join(lines) + "\n")
lines = [r"\begin{tabular}{lrr}", r"\toprule", r"Resolution date & Dollar share & Contribution \\", r"\midrule"]
for _, r in D.head(5).iterrows():
    lines.append(f"{r.ev_cluster.split('|')[1]} & {pct(r.usd_share)}\\% & {f0(r.contrib_bp)}" + r" \\")
lines += [r"\bottomrule", r"\end{tabular}"]
(TABLES / "concentration_dates.tex").write_text("\n".join(lines) + "\n")

S = pd.read_csv(RES / "spl_A_prediction.csv")
def sget(design, model): return S[(S.design == design) & (S.model == model)].iloc[0]
for design, pre in [("cv", "Cv"), ("forward", "Fw")]:
    r = sget(design, "M2")
    M[f"Spl{pre}"] = f"{1e4 * r.dbrier_vs_linear:.1f}".replace("-", "\\ensuremath{-}")
    M[f"Spl{pre}Lo"] = f"{1e4 * r.linear_lo:.1f}".replace("-", "\\ensuremath{-}")
    M[f"Spl{pre}Hi"] = f"{1e4 * r.linear_hi:.1f}".replace("-", "\\ensuremath{-}")
lines = [r"\begin{tabular}{llrr}", r"\toprule",
         r"Design & Model & $10^4$Brier & \thead{Gain over\\linear price} \\", r"\midrule"]
for design, lab in [("cv", "Cross-validation"), ("forward", "Forward by quarter")]:
    for model, mlab in [("M1", "linear price"), ("M1s", "price spline"), ("M2", "spline + small flow"), ("M4", "spline + both flows")]:
        r = sget(design, model)
        gain = "0.0" if model == "M1" else f"{1e4 * r.dbrier_vs_linear:.1f} [{1e4 * r.linear_lo:.1f}, {1e4 * r.linear_hi:.1f}]".replace("-", "\\ensuremath{-}")
        lines.append(f"{lab} & {mlab} & {1e4*r.brier:.1f} & {gain}" + r" \\")
    lines.append(r"\addlinespace")
lines += [r"\bottomrule", r"\end{tabular}"]
(TABLES / "spline.tex").write_text("\n".join(lines) + "\n")

bad = [k for k in M if not k.replace("Eq", "E").isalpha() and not k[0].isalpha()]
assert not [k for k in M if not k.isalnum()], list(M)
(ROOT / "paper" / "numbers_v3.tex").write_text("% Generated by analysis/28_v3_tables.py.\n" +
    "\n".join(f"\\newcommand{{\\{k}}}{{{v}}}" for k, v in sorted(M.items())) + "\n")
print(f"{len(M)} macros")
