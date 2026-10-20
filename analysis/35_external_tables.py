"""Appendix tables and macros for the external-data checks. Reads results/; writes paper/tables and paper/numbers_ext.tex."""
import math
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RES, TABLES = ROOT / "results", ROOT / "paper" / "tables"
M = {}

from manuscript_format import tex_decimal, cell
f0 = f1 = tex_decimal

def row(df, sample, metric, cellname):
    hit = df[(df["sample"] == sample) & (df.metric == metric) & (df.cell == cellname)]
    if len(hit) != 1:
        raise SystemExit(f"missing {sample} {metric} {cellname}")
    return hit.iloc[0]

B = pd.read_csv(RES / "d1_B_metrics.csv")
A = pd.read_csv(RES / "d1_A_alignment.csv")
C = pd.read_csv(RES / "d1_C_coverage.csv").iloc[0]
D = pd.read_csv(RES / "d1_D_pmxt_overlap.csv").iloc[0]
star = A.sort_values(["share_exact_side", "lead_s"], ascending=[False, True]).iloc[0]
M["DOneLead"] = str(int(C.lead_s))
M["DOneTouch"] = f"{100 * float(star.share_exact_side):.1f}"
M["DOneWin"] = f"{int(C.trades):,}"
M["DOnePre"] = f"{int(C.pre_ok):,}"
info = row(B, "quote minus proxy", "info", "pooled")
res = row(B, "quote minus proxy", "over_100", "pooled")
pols = row(B, "quote minus proxy", "over_100", "Politics | s1 <$100")
M["DOneInfo"], M["DOneInfoSE"] = f0(info.est_bp), f0(info.se_bp)
M["DOneRes"], M["DOneResSE"] = f0(res.est_bp), f0(res.se_bp)
M["DOnePolS"], M["DOnePolSSE"] = f0(pols.est_bp), f0(pols.se_bp)
polq = row(B, "matched quote", "over_100", "Politics | s1 <$100")
M["DOnePolQ"], M["DOnePolQSE"] = f0(polq.est_bp), f0(polq.se_bp)
M["DOnePmDays"] = str(int(D.days))
M["DOnePmMkt"] = str(int(D.markets))
M["DOnePmMean"] = f1(D.mean_bp)
M["DOnePmAbs"] = f0(D.mean_abs_bp)
M["DOnePmTick"] = f"{100 * float(D.share_within_tick):.1f}"

cells = [
    ("pooled", "Pooled"),
    ("Politics | s1 <$100", r"Politics $<\$100$"),
    ("Politics | s3 >=$1k", r"Politics $\ge\$1$k"),
    ("Crypto | s1 <$100", r"Crypto $<\$100$"),
    ("Crypto | s3 >=$1k", r"Crypto $\ge\$1$k"),
]
lines = [r"\begin{tabular}{lrrrr}", r"\toprule",
         r"Cell & $n$ & Alignment & $n$ & Residual \\", r"\midrule"]
for key, lab in cells:
    a = row(B, "quote minus proxy", "info", key)
    r = row(B, "quote minus proxy", "over_100", key)
    lines.append(f"{lab} & {int(a.n):,} & {cell(a.est_bp, a.se_bp)} & {int(r.n):,} & {cell(r.est_bp, r.se_bp)}" + r" \\")
lines += [r"\bottomrule", r"\end{tabular}"]
(TABLES / "external_dome.tex").write_text("\n".join(lines) + "\n")

cov = pd.read_csv(RES / "d5_A_coverage.csv")
clk = pd.read_csv(RES / "d5_B_clocks.csv").iloc[0]
lag = pd.read_csv(RES / "d5_B_leadlag.csv").iloc[0]
dcov = pd.read_csv(RES / "d5_D_coverage.csv").iloc[0]
pred = pd.read_csv(RES / "d5_C_short_pred.csv")
d6 = pd.read_csv(RES / "d6_A_coverage.csv").iloc[0]
up = cov[cov.grp == "UpDown"].iloc[0]
M["ExtMeta"] = f"{int(cov.n_meta.sum()):,}"
M["ExtInV"] = f"{int(up.n_in_v1):,}"
M["ExtLife"] = f"{float(up.med_life_h):.1f}"
M["ExtTickDays"] = str(int(clk.n_days))
M["ExtLag"] = f"{float(clk.med_ingest_lag_s):.1f}"
M["ExtLead"] = f0(lag.med_lead_lag_ms)
M["ExtTight"] = f"{100 * float(lag.share_tight):.1f}"
M["ExtBtcN"] = f"{int(dcov.n_window):,}"
M["ExtBtcHit"] = str(int(dcov.n_candle))
ret = pred[pred.model == "price + BTC return"].iloc[0]
flo = pred[pred.model == "price + return + flow"].iloc[0]
M["ExtShort"] = str(int(ret.n_train + ret.n_test))
M["ExtTest"] = str(int(ret.n_test))
for pre, r in [("ExtRet", ret), ("ExtFlow", flo)]:
    M[pre + "G"] = f1(1e4 * float(r.dbrier_vs_price))
    M[pre + "Lo"] = f1(1e4 * float(r.lo))
    M[pre + "Hi"] = f1(1e4 * float(r.hi))
M["ExtDsixN"] = f"{int(d6.n_markets_window):,}"
M["ExtDsixHit"] = str(int(d6.n_with_candle))

kpath = RES / "k_B_margin.csv"
if kpath.exists() and (RES / "k_C_prediction.csv").exists() and (RES / "k_A_sample.csv").exists():
    kb = pd.read_csv(kpath)
    ka = pd.read_csv(RES / "k_A_sample.csv").iloc[0]
    kc = pd.read_csv(RES / "k_C_prediction.csv")
    M["KalTrades"] = f"{int(ka.trades):,}"
    M["KalMkts"] = f"{int(ka.markets):,}"
    M["KalEvents"] = f"{int(ka.events):,}"
    M["KalCents"] = f"{100 * float(ka.cents_100):.1f}"
    def kget(scheme, b):
        hit = kb[(kb.scheme == scheme) & (kb.bin == b)]
        if len(hit) != 1:
            raise SystemExit(f"missing kalshi {scheme} {b}")
        return hit.iloc[0]
    sml, lrg = kget("dollars", "s1 <$100"), kget("dollars", "s3 >=$1k")
    diff, rank = kget("dollars", "s3 >=$1k minus s1 <$100"), kget("within-market rank", "r3 high minus r1 low")
    M["KalSml"], M["KalSmlSE"] = f0(sml.trade_bp), f0(sml.trade_se)
    M["KalLrg"], M["KalLrgSE"] = f0(lrg.trade_bp), f0(lrg.trade_se)
    M["KalDiff"], M["KalDiffSE"] = f0(diff.trade_bp), f0(diff.trade_se)
    M["KalRank"], M["KalRankSE"] = f0(rank.trade_bp), f0(rank.trade_se)
    lines = [r"\begin{tabular}{lrrrr}", r"\toprule",
             r"Bin & Trades & Clusters & Per trade & Per dollar \\", r"\midrule"]
    order = ["s1 <$100", "s2 $100-1k", "s3 >=$1k", "s3 >=$1k minus s1 <$100"]
    labels = {"s1 <$100": r"$<\$100$", "s2 $100-1k": r"\$100--\$1k", "s3 >=$1k": r"$\ge\$1$k",
              "s3 >=$1k minus s1 <$100": r"Large minus small"}
    sub = kb[kb.scheme == "dollars"]
    for b in order:
        r = sub[sub.bin == b].iloc[0]
        cash = "---" if math.isnan(float(r.cash_bp)) else cell(r.cash_bp, r.cash_se)
        n = "---" if math.isnan(float(r.n)) else f"{int(r.n):,}"
        lines.append(f"{labels[b]} & {n} & {int(r.n_clusters):,} & {cell(r.trade_bp, r.trade_se)} & {cash}" + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    (TABLES / "external_kalshi.tex").write_text("\n".join(lines) + "\n")
    lines = [r"\begin{tabular}{lrrrr}", r"\toprule",
             r"Bin & Trades & Clusters & Per trade & Per dollar \\", r"\midrule"]
    order = ["r1 low", "r2 mid", "r3 high", "r3 high minus r1 low"]
    labels = {"r1 low": "Low third", "r2 mid": "Middle third", "r3 high": "High third",
              "r3 high minus r1 low": "High minus low"}
    sub = kb[kb.scheme == "within-market rank"]
    for b in order:
        r = sub[sub.bin == b].iloc[0]
        cash = "---" if math.isnan(float(r.cash_bp)) else cell(r.cash_bp, r.cash_se)
        n = "---" if math.isnan(float(r.n)) else f"{int(r.n):,}"
        lines.append(f"{labels[b]} & {n} & {int(r.n_clusters):,} & {cell(r.trade_bp, r.trade_se)} & {cash}" + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    (TABLES / "external_kalshi_rank.tex").write_text("\n".join(lines) + "\n")
    both = kc[kc["sample"] == "both flows, forward"]
    small = kc[kc["sample"] == "small flow, forward"]
    if len(both) and int(both.iloc[0].n) > 0:
        r = both.iloc[0]
        M["KalBothN"] = f"{int(r.n):,}"
        M["KalBothG"] = f1(1e4 * float(r.dbrier))
        M["KalBothLo"] = f1(1e4 * float(r.lo))
        M["KalBothHi"] = f1(1e4 * float(r.hi))
    else:
        M["KalBothN"] = "0"
        M["KalBothG"] = M["KalBothLo"] = M["KalBothHi"] = "---"
    if len(small) and int(small.iloc[0].n) > 0:
        r = small.iloc[0]
        M["KalSmallN"] = f"{int(r.n):,}"
        M["KalSmallG"] = f1(1e4 * float(r.dbrier))
        M["KalSmallLo"] = f1(1e4 * float(r.lo))
        M["KalSmallHi"] = f1(1e4 * float(r.hi))
    else:
        M["KalSmallN"] = "0"
        M["KalSmallG"] = M["KalSmallLo"] = M["KalSmallHi"] = "---"

out = ["% Generated by analysis/35_external_tables.py."]
for k in sorted(M):
    out.append(r"\newcommand{\%s}{%s}" % (k, M[k]))
(ROOT / "paper" / "numbers_ext.tex").write_text("\n".join(out) + "\n")
print(f"wrote {len(M)} macros")
