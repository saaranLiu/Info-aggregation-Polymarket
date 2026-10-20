"""Paper tables and macros for the fixed-clock residuals, forward periods, within/between by size, fill-price
aggregation and observed-quote validation. Reads results written by 13b, 14, 15 and 16; writes paper/tables/*.tex
and paper/numbers_r2.tex. Quote-validation outputs are optional: if 13b has not been run the quote macros are
written as `--` and the quote table is omitted.
"""
import math, os
from pathlib import Path
import numpy as np, pandas as pd

ROOT = Path(__file__).resolve().parents[1]; RES = ROOT / "results"; PAPER = ROOT / "paper"; TABLES = PAPER / "tables"
rd = lambda n: pd.read_csv(RES / n)
from manuscript_format import tex_decimal, cell, probability
M = {}

f0 = tex_decimal
def sgn(x): return tex_decimal(x, signed=True)

# ---------------- fixed-clock residuals under event cutoffs
A = rd("clk_A_cells.csv"); E = rd("clk_E_diff.csv"); B = rd("clk_B_sample.csv")
def g(cut, c, m):
    r = A[(A.cutoff == cut) & (A.cell == c) & (A.metric == m)].iloc[0]; return r
PS, PL, CS, CL = "Politics | s1 <$100", "Politics | s3 >=$1k", "Crypto | s1 <$100", "Crypto | s3 >=$1k"
LAB = {"R": "R: on-chain resolution", "S": "S: scheduled close", "S1": "S1: close $-$ 1 h", "C": "C: price convergence"}
tex = [r"\begin{tabular}{lrrrrrrrr}", r"\toprule",
       r" & \multicolumn{4}{c}{Politics $<\$100$} & Politics & \multicolumn{2}{c}{Crypto $<\$100$} & Difference \\",
       r"\cmidrule(lr){2-5}\cmidrule(lr){7-8}",
       r"Cutoff & 10 min & 1 h & 2 h & 100 trades & $\ge\$1$k, 1 h & 1 h & 100 trades & small, 1 h \\", r"\midrule"]
for k in ["R", "S", "S1", "C"]:
    v = [g(k, PS, m) for m in ("over_c600", "over_c3600", "over_c7200", "over_100")] + [g(k, PL, "over_c3600")] + \
        [g(k, CS, m) for m in ("over_c3600", "over_100")]
    d = E[(E.cutoff == k) & (E.metric == "over_c3600")].iloc[0]
    tex.append(LAB[k] + " & " + " & ".join(cell(r.est_bp, r.se_ev_bp) for r in v) + f" & {cell(d.diff_bp, d.se_bp)}" + r" \\")
tex += [r"\midrule"]
row = []
for k in ["R", "S", "S1", "C"]:
    p = g(k, "Politics", "over_c3600"); c = g(k, "Crypto", "over_c3600")
    row.append(f"{k}: {p.n_markets:,}/{c.n_markets:,}")
tex += [r"\multicolumn{9}{l}{Politics/Crypto markets in each common sample: " + "; ".join(row) + r".} \\", r"\bottomrule", r"\end{tabular}"]
(TABLES / "clock_residuals.tex").write_text("\n".join(tex) + "\n")
for k in ["R", "S", "S1", "C"]:
    kk = {"R": "R", "S": "S", "S1": "Sone", "C": "C"}[k]
    for c, pre in [(PS, "PolS"), (PL, "PolL"), (CS, "CryS"), ("Politics", "Pol"), ("Crypto", "Cry"), ("pooled", "All")]:
        for m, mm in [("over_c3600", "Hour"), ("over_c7200", "Two"), ("over_c600", "Ten"), ("over_100", "Hund"), ("info", "Info"), ("RS", "RS"), ("PI_c3600", "PIHour")]:
            r = g(k, c, m); M[f"{pre}{mm}{kk}"] = f0(r.est_bp); M[f"{pre}{mm}{kk}SE"] = f"{r.se_ev_bp:.1f}"
    for m, mm in [("over_c3600", "Hour"), ("over_100", "Hund"), ("over_c600", "Ten"), ("over_c7200", "Two")]:
        d = E[(E.cutoff == k) & (E.metric == m)].iloc[0]
        M[f"Diff{mm}{kk}"] = f0(d.diff_bp); M[f"Diff{mm}{kk}SE"] = f"{d.se_bp:.1f}"; M[f"Diff{mm}{kk}P"] = probability(d.p)
M["DiffMin"] = f0(E.diff_bp.min()); M["DiffMax"] = f0(E.diff_bp.max()); M["DiffPMax"] = probability(E.p.max()); M["DiffCells"] = str(len(E))
crs = A[(A.cell == CS) & A.metric.str.startswith("over")]
M["CrySmallMin"] = f0(crs.est_bp.min()); M["CrySmallMax"] = f0(crs.est_bp.max())
for k in ["R", "S", "C"]:
    kk = {"R": "R", "S": "S", "C": "C"}[k]
    for cat, pre in [("Politics", "Pol"), ("Crypto", "Cry")]:
        r = B[(B.cutoff == k) & (B.cat == cat)].iloc[0]
        M[f"{pre}N{kk}"] = f"{int(r.n_common):,}"; M[f"{pre}Mkts{kk}"] = f"{int(r.m_common):,}"
M["PolAfterClose"] = f"{100 * B[(B.cat == 'Politics')].share_after_close_main.iloc[0]:.1f}"
M["CryAfterClose"] = f"{100 * B[(B.cat == 'Crypto')].share_after_close_main.iloc[0]:.1f}"
CT = rd("ctl_A_residual_controls.csv")
for k in ["R", "S", "C"]:
    r = CT[(CT.cutoff == k) & CT.spec.str.startswith("+")].iloc[0]
    M[f"Ctl{k}"] = f0(r.contrast_bp); M[f"Ctl{k}SE"] = f"{r.se_ev_bp:.1f}"

# ---------------- forward periods and expanding-window prediction
F = rd("clk_C_forward.csv")
per = ["P1 <2025", "P2 2025H1", "P3 2025H2", "P4 2026"]; plab = {"P1 <2025": "$<2025$", "P2 2025H1": "2025 H1", "P3 2025H2": "2025 H2", "P4 2026": "2026"}
tex = [r"\begin{tabular}{lrrr}", r"\toprule",
       r"Period & R & S & C \\", r"\midrule"]
for c, topic in [(PS, "Politics"), (CS, "Crypto")]:
    tex.append(r"\multicolumn{4}{l}{" + topic + r" $<\$100$} \\")
    for p in per:
        vals = []
        for k in ["R", "S", "C"]:
            r = F[(F.cutoff == k) & (F.cell == c) & (F.period == p) & (F.metric == "over_c3600")]
            vals.append(cell(r.est_bp.iloc[0], r.se_ev_bp.iloc[0]) if len(r) else "--")
        tex.append(plab[p] + " & " + " & ".join(vals) + r" \\")
    if topic == "Politics": tex.append(r"\addlinespace")
tex += [r"\bottomrule", r"\end{tabular}"]
(TABLES / "forward_periods.tex").write_text("\n".join(tex) + "\n")
for k in ["R", "S", "C"]:
    for i, p in enumerate(per):
        r = F[(F.cutoff == k) & (F.cell == PS) & (F.period == p) & (F.metric == "over_c3600")].iloc[0]
        M[f"FwdPol{k}P{'ABCD'[i]}"] = f0(r.est_bp); M[f"FwdPol{k}P{'ABCD'[i]}SE"] = f"{r.se_ev_bp:.1f}"
D = rd("clk_D_forward_pred.csv")
for k in ["S", "C"]:
    q = D[(D.cutoff == k) & (D.test_quarter != "pooled forward")].drop_duplicates("test_quarter")
    M[f"FwdCoefMin{k}"] = f"{q.coef_pol_s1.min():.1f}".replace("-", "\\ensuremath{-}"); M[f"FwdCoefMax{k}"] = f"{q.coef_pol_s1.max():.1f}".replace("-", "\\ensuremath{-}")
    M[f"FwdQuarters{k}"] = str(q.test_quarter.nunique()); M[f"FwdNeg{k}"] = str(int((q.coef_pol_s1 < 0).sum()))
    for c, pre in [("Politics", "Pol"), ("Politics | s1 <$100", "PolS"), ("Crypto", "Cry"), ("Politics+Crypto", "All")]:
        r = D[(D.cutoff == k) & (D.test_quarter == "pooled forward") & (D.cell == c)].iloc[0]
        sc = lambda x: f"{1e4 * x:.1f}".replace("-", "\\ensuremath{-}")
        M[f"FwdBrier{pre}{k}"] = sc(r.dbrier); M[f"FwdBrier{pre}{k}Lo"] = sc(r.dbrier_lo); M[f"FwdBrier{pre}{k}Hi"] = sc(r.dbrier_hi)
        M[f"FwdLL{pre}{k}"] = sc(r.dlogloss); M[f"FwdLL{pre}{k}Lo"] = sc(r.dlogloss_lo); M[f"FwdLL{pre}{k}Hi"] = sc(r.dlogloss_hi)

tex = [r"\begin{tabular}{lrrrr}", r"\toprule", r"Test quarter & \thead{$10\beta_{\mathrm{Pol}}$\\$<\$100$} & \thead{$10\beta_{\mathrm{Cry}}$\\$<\$100$} & \thead{Politics\\$\Delta$Brier} & \thead{Crypto\\$\Delta$Brier} \\", r"\midrule"]
q = D[D.cutoff == "S"]
for tq in [x for x in q.test_quarter.unique() if x != "pooled forward"] + ["pooled forward"]:
    z = q[q.test_quarter == tq].set_index("cell")
    br = lambda c: (f"{1e4 * z.loc[c, 'dbrier']:.1f}".replace("-", "\\ensuremath{-}") +
                    (f" [{1e4 * z.loc[c, 'dbrier_lo']:.1f}, {1e4 * z.loc[c, 'dbrier_hi']:.1f}]".replace("-", "\\ensuremath{-}") if not pd.isna(z.loc[c, "dbrier_lo"]) else ""))
    co = lambda c: "" if pd.isna(z[c].iloc[0]) else tex_decimal(10*z[c].iloc[0])
    lab = tq.replace("-", " ") if tq != "pooled forward" else "pooled"
    if tq == "pooled forward": tex.append(r"\midrule")
    tex.append(f"{lab} & {co('coef_pol_s1')} & {co('coef_cry_s1')} & {br('Politics')} & {br('Crypto')}" + r" \\")
tex += [r"\bottomrule", r"\end{tabular}"]
(TABLES / "forward_pred.tex").write_text("\n".join(tex) + "\n")

# ---------------- within/between by size
W = rd("wb_F_size.csv")
tex = [r"\begin{tabular}{llrrr}", r"\toprule", r"Topic & Size & $\info$ & Between & Within \\", r"\midrule"]
for c in ["Politics", "Crypto", "pooled"]:
    for s, sl in [("s1 <$100", r"$<\$100$"), ("s2 $100-1k", r"\$100--1k"), ("s3 >=$1k", r"$\ge\$1$k"), ("all", "all")]:
        r = W[(W["sample"] == "main") & (W.cell == c) & (W["size"] == s)]
        if c == "pooled" and s != "all": continue
        r = r.iloc[0]
        tex.append(f"{'Yes/No' if c == 'pooled' else c} & {sl} & {cell(r.total_bp, r.total_se_ev_bp)} & {cell(r.between_bp, r.between_se_ev_bp)} & {cell(r.within_bp, r.within_se_ev_bp)}" + r" \\")
    if c != "pooled": tex.append(r"\addlinespace")
tex += [r"\bottomrule", r"\end{tabular}"]
(TABLES / "wb_size.tex").write_text("\n".join(tex) + "\n")
for _, r in W.iterrows():
    pre = {"pooled": "All", "Politics": "Pol", "Crypto": "Cry"}[r.cell] + {"all": "", "s1 <$100": "S", "s2 $100-1k": "M", "s3 >=$1k": "L"}[r["size"]] + \
          {"main": "", "clock_common_S": "ClkS"}[r["sample"]]
    M[f"WB{pre}Tot"] = f0(r.total_bp); M[f"WB{pre}Btw"] = f0(r.between_bp); M[f"WB{pre}BtwSE"] = f"{r.between_se_ev_bp:.1f}"
    M[f"WB{pre}Wtn"] = f0(r.within_bp); M[f"WB{pre}WtnSE"] = f"{r.within_se_ev_bp:.1f}"

# ---------------- fill-price aggregation
G = rd("fp_A_price_gap.csv"); P = rd("fp_B_metrics.csv")
allr = G[(G.cat == "all") & (G["size3"] == "all")].iloc[0] if ((G.cat == "all") & (G["size3"] == "all")).any() else None
if allr is not None:
    M["FPAbsGap"] = f"{allr.mean_abs_gap_bp:.1f}"; M["FPSignedGap"] = f"{allr.mean_signed_gap_bp:.1f}"; M["FPMulti"] = f"{100 * allr.share_multifill:.1f}"
    M["FPPNinetyNine"] = f"{allr.p99_abs_gap_bp:.1f}"
pl = G[(G.cat == "Politics") & (G["size3"] == "s3 >=$1k")].iloc[0]; M["FPAbsGapPolL"] = f"{pl.mean_abs_gap_bp:.1f}"
def pm(sample, c, w, price, col): return P[(P["sample"] == sample) & (P.cell == c) & (P.weight == w) & (P.price == price)].iloc[0][col]
M["FPRSDiff"] = f"{pm('main', 'pooled', 'trade', 'p_usd', 'diff_trade_bp'):.1f}"; M["FPRSDiffSE"] = f"{pm('main', 'pooled', 'trade', 'p_usd', 'diff_trade_se_ev_bp'):.1f}"
M["FPRSDiffPolL"] = f"{pm('main', PL, 'trade', 'p_usd', 'diff_trade_bp'):.1f}"
tex = [r"\begin{tabular}{lrrrr}", r"\toprule", r"Weight & $\eff$ & $\info$ & $\RS$ & $\RS$, share price \\", r"\midrule"]
for w, lab in [("trade", "per trade"), ("shares_actual", "per share, observed"), ("u_over_p", r"per share, $u/p$"), ("usd", "per dollar")]:
    tex.append(f"{lab} & {f0(pm('observed_h100', 'pooled', w, 'p_usd', 'eff_bp'))} & {f0(pm('observed_h100', 'pooled', w, 'p_usd', 'info_bp'))} & "
               f"{f0(pm('observed_h100', 'pooled', w, 'p_usd', 'RS_bp'))} & {f0(pm('observed_h100', 'pooled', w, 'p_sh', 'RS_bp'))}" + r" \\")
tex += [r"\bottomrule", r"\end{tabular}"]
(TABLES / "fill_prices.tex").write_text("\n".join(tex) + "\n")
for w, pre in [("shares_actual", "Sh"), ("u_over_p", "Up"), ("usd", "Usd"), ("trade", "Tr")]:
    M[f"FPRS{pre}"] = f0(pm("observed_h100", "pooled", w, "p_usd", "RS_bp")); M[f"FPRS{pre}Sh"] = f0(pm("observed_h100", "pooled", w, "p_sh", "RS_bp"))
    M[f"FPInfo{pre}"] = f0(pm("observed_h100", "pooled", w, "p_usd", "info_bp")); M[f"FPEff{pre}"] = f0(pm("observed_h100", "pooled", w, "p_usd", "eff_bp"))

# ---------------- observed-quote validation (optional)
QV = {n: RES / f"qv_{n}.csv" for n in ["A_alignment", "B_error", "C_reg", "D_metrics", "E_wb", "F_coverage", "G_days", "H_recorded"]}
have_q = all(p.exists() for p in QV.values())
if have_q:
    exec((Path(__file__).with_name("18b_quote_tables.py")).read_text(), {"rd": rd, "M": M, "TABLES": TABLES, "f0": f0, "cell": cell, "sgn": sgn, "pd": pd, "np": np, "math": math})
else:
    for k in ["QMktsAll", "QMktsPol", "QMktsCry", "QTrades", "QLead", "QMatchPct"]: M[k] = "--"
    M["QuoteParagraph"] = ""; M["QuoteAppendix"] = ""
    (TABLES / "quote_validation.tex").write_text("\\begin{tabular}{l}\\toprule quote validation not yet run \\\\ \\bottomrule\\end{tabular}\n")
M["HaveQuotes"] = "1" if have_q else "0"

(PAPER / "numbers_r2.tex").write_text("% Generated by analysis/18_r2_outputs.py.\n" +
                                      "\n".join(f"\\newcommand{{\\{k}}}{{{v}}}" for k, v in sorted(M.items())) + "\n")
bad = [k for k in M if not k.isalpha()]
assert not bad, bad

def v(k): return M[k].replace("\\ensuremath{-}", "-")
md = ["", "## Clock horizons, event cutoffs, fill prices and observed quotes", "",
      "Generated by `analysis/18_r2_outputs.py` from `clk_*`, `ctl_A`, `fp_*`, `audit_C_weights_YesNo.csv`, `wb_F_size.csv` and `qv_*`. Probability bp; parentheses are event-cluster SEs.", "",
      "Cutoffs: R = on-chain resolution; S = min(scheduled close, R); S1 = S - 1 h; C = min(S, last trade with winner-token price below 0.95). "
      "Common sample of a cutoff: the trade, its 2-hour clock horizon and its 100th later trade all precede the cutoff.", "",
      f"- Politics <$100, 1-hour clock residual: R {v('PolSHourR')} ({v('PolSHourRSE')}), S {v('PolSHourS')} ({v('PolSHourSSE')}), "
      f"S1 {v('PolSHourSone')} ({v('PolSHourSoneSE')}), C {v('PolSHourC')} ({v('PolSHourCSE')}). Crypto <$100 lies in [{v('CrySmallMin')}, {v('CrySmallMax')}] across all cells.",
      f"- Politics-minus-Crypto small-order difference: {v('DiffMin')} to {v('DiffMax')} bp, p <= {v('DiffPMax')} in {v('DiffCells')} cutoff-by-horizon cells. "
      f"Within midpoint-decile, hour-to-resolution and quarter cells with a log-trade control: R {v('CtlR')} ({v('CtlRSE')}), S {v('CtlS')} ({v('CtlSSE')}), C {v('CtlC')} ({v('CtlCSE')}).",
      f"- Forward periods (cutoff S): the Politics small-order direction coefficient is negative in {v('FwdNegS')} of {v('FwdQuartersS')} test quarters; "
      f"the pooled Politics Brier gain (x1e4) is {v('FwdBrierPolS')} [{v('FwdBrierPolSLo')}, {v('FwdBrierPolSHi')}], i.e. not distinguishable from zero.",
      f"- Fill prices: share- and dollar-weighted fill prices differ by {v('FPRSDiff')} bp on average in the per-trade margin. "
      f"Observed-horizon margin: {v('FPRSTr')} per trade, {v('FPRSSh')} per actual share ({v('FPRSShSh')} with the share-weighted price), {v('FPRSUp')} per usd/p share, {v('FPRSUsd')} per dollar.",
      f"- Quote validation ({'available' if have_q else 'not yet run'}): {v('QMktsAll')} markets ({v('QMktsPol')} Politics, {v('QMktsCry')} Crypto), {v('QTrades')} trades, "
      f"alignment lead {v('QLead')} s with {v('QMatchPct')}% exact-side matches. Sample: every third archive day from 22 Feb 2026 (`QUOTE_SAMPLE=every3`).", ""]
rpath = ROOT / "RESULTS.md"
txt = rpath.read_text()
head = txt.split("\n## Clock horizons, event cutoffs")[0].rstrip("\n")
rpath.write_text(head + "\n" + "\n".join(md))
print(f"{len(M)} macros; quotes={'yes' if have_q else 'no'}")
