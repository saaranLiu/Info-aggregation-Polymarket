# Executed by 18_r2_outputs.py with rd, M, TABLES, f0, cell, sgn, pd, np, math in scope.
# Builds tables/quote_validation.tex and the quote macros, including the two quote paragraphs.
A = rd("qv_A_alignment.csv"); Bq = rd("qv_B_error.csv"); Cq = rd("qv_C_reg.csv"); Dq = rd("qv_D_metrics.csv")
Eq = rd("qv_E_wb.csv"); Fq = rd("qv_F_coverage.csv")
LSTAR = int(A.sort_values(["share_exact_side", "lead_s"], ascending=[False, True]).iloc[0].lead_s)
a = A[A.lead_s == LSTAR].iloc[0]; a0 = A[A.lead_s == 0].iloc[0]
M["QLead"] = str(LSTAR); M["QMatchPct"] = f"{100 * a.share_exact_side:.1f}"; M["QMatchZeroPct"] = f"{100 * a0.share_exact_side:.1f}"
M["QCostPosPct"] = f"{100 * a.share_cost_positive:.1f}"
pc = Fq[Fq.cat.isin(["Politics", "Crypto"])].fillna(0)
M["QMktsPol"] = f"{int(pc[pc.cat == 'Politics'].markets_with_quotes.iloc[0]):,}"; M["QMktsCry"] = f"{int(pc[pc.cat == 'Crypto'].markets_with_quotes.iloc[0]):,}"
M["QMktsAll"] = f"{int(pc.markets_with_quotes.sum()):,}"; M["QTrades"] = f"{int(pc.trades_validation.sum()):,}"
M["QTargetMkts"] = f"{int(pc.target_markets.sum()):,}"
M["QCoverPct"] = f"{100 * pc.trades_validation.sum() / pc.trades_main_on_days.sum():.1f}"
M["QDays"] = str(len(rd("qv_G_days.csv")))

def b(g, col): return Bq[Bq.group == g].iloc[0][col]
def d(c, m): r = Dq[(Dq.cell == c) & (Dq.metric == m)]; return r.iloc[0] if len(r) else None
for g, pre in [("pooled", "All"), ("Politics", "Pol"), ("Crypto", "Cry"), ("Politics | s1 <$100", "PolS"), ("Crypto | s1 <$100", "CryS"),
               ("Politics | >=$100", "PolL"), ("Crypto | >=$100", "CryL"), ("pooled | fresh", "Fresh"), ("pooled | stale >5", "Stale"),
               ("pooled | crossed pair", "Crossed")]:
    if not (Bq.group == g).any(): continue
    M[f"QDe{pre}"] = f0(b(g, "mean_De_bp")); M[f"QDe{pre}SE"] = f"{b(g, 'mean_De_se_ev_bp'):.1f}"
    M[f"QMAE{pre}"] = f0(b(g, "mae_bp")); M[f"QHS{pre}"] = f0(b(g, "quoted_half_spread_bp"))
    M[f"QNegProxy{pre}"] = f"{100 * b(g, 'share_cost_negative_proxy'):.1f}"; M[f"QNegQuote{pre}"] = f"{100 * b(g, 'share_cost_negative_quote'):.1f}"
    M[f"QOutside{pre}"] = f"{100 * b(g, 'share_proxy_outside_quotes'):.1f}"
for c, pre in [("pooled", "All"), ("Politics", "Pol"), ("Crypto", "Cry"), ("Politics | s1 <$100", "PolS"), ("Crypto | s1 <$100", "CryS"),
               ("Politics | >=$100", "PolL"), ("Crypto | >=$100", "CryL")]:
    for m, mm in [("info", "Info"), ("eff", "Eff"), ("over_100", "Hund"), ("over_c3600_R", "HourR"), ("over_c3600_S", "HourS"),
                  ("over_c7200_R", "TwoR"), ("info_S", "InfoS"), ("info_R", "InfoR")]:
        r = d(c, m)
        if r is None or pd.isna(r.n): continue
        M[f"Q{pre}{mm}P"] = f0(r.proxy_bp); M[f"Q{pre}{mm}Q"] = f0(r.quote_bp); M[f"Q{pre}{mm}QSE"] = f"{r.quote_se_ev_bp:.1f}"
        M[f"Q{pre}{mm}D"] = f0(r.diff_bp); M[f"Q{pre}{mm}DSE"] = f"{r.diff_se_ev_bp:.1f}"; M[f"Q{pre}{mm}N"] = f"{int(r.n):,}"
        M[f"Q{pre}{mm}Mk"] = f"{int(r.n_markets):,}"
reg = Cq.set_index("term")
def rc(t): return reg.loc[t] if t in reg.index else None
for t, pre in [("C(fresh_b)[T.stale >5]", "Stale"), ("C(fresh_b)[T.stale 2-5]", "StaleMid"), ("crossed", "Crossed"), ("spread_bp", "Spread"),
               ("C(cat)[T.Politics]", "Pol"), ("C(size3)[T.s2 $100-1k]", "Mid"), ("C(size3)[T.s3 >=$1k]", "Large")]:
    r = rc(t)
    if r is None: continue
    M[f"QReg{pre}"] = f0(r.coef_bp) if pre != "Spread" else f"{r.coef_bp:.1f}"; M[f"QReg{pre}SE"] = f"{r.se_ev_bp:.1f}" if pre != "Spread" else f"{r.se_ev_bp:.1f}"
    M[f"QReg{pre}P"] = f"{r.p:.1f}"
def wb(mid, c, s, comp): r = Eq[(Eq.midpoint == mid) & (Eq.cell == c) & (Eq["size"] == s)].iloc[0]; return r[f"{comp}_bp"], r[f"{comp}_se_ev_bp"]
for c, pre in [("Politics", "Pol"), ("Crypto", "Cry"), ("pooled", "All")]:
    for s, sp in [("all", ""), ("s1 <$100", "S")]:
        for comp, cp in [("between", "Btw"), ("within", "Wtn"), ("total", "Tot")]:
            for mid, mp in [("proxy", "P"), ("quote", "Q")]:
                e, se = wb(mid, c, s, comp); M[f"QWB{pre}{sp}{cp}{mp}"] = f0(e); M[f"QWB{pre}{sp}{cp}{mp}SE"] = f"{se:.1f}"

# --- table
rows = [("Yes/No", "pooled"), ("Politics", "Politics"), ("\\quad $<\\$100$", "Politics | s1 <$100"), ("\\quad $\\ge\\$100$", "Politics | >=$100"),
        ("Crypto", "Crypto"), ("\\quad $<\\$100$", "Crypto | s1 <$100"), ("\\quad $\\ge\\$100$", "Crypto | >=$100")]
mets = [("info", "$\\info$"), ("over_c3600_R", "$\\ovr_{1\\mathrm h}$, R"), ("over_c3600_S", "$\\ovr_{1\\mathrm h}$, S"), ("over_100", "$\\ovr_{100}$")]
tex = [r"\begin{tabular}{lrr" + "rr" * len(mets) + "}", r"\toprule",
       " & & & " + " & ".join(f"\\multicolumn{{2}}{{c}}{{{ml}}}" for _, ml in mets) + r" \\",
       "".join(f"\\cmidrule(lr){{{4 + 2 * i}-{5 + 2 * i}}}" for i in range(len(mets))),
       r"Cell & Markets & Trades & " + " & ".join(["Quote & Proxy$-$quote"] * len(mets)) + r" \\", r"\midrule"]
for lab, c in rows:
    g = Bq[Bq.group == c]
    if not len(g) or g.iloc[0].n < 30: continue
    g = g.iloc[0]; r0 = d(c, "info")
    vals = [f"{int(r0.n_markets):,}", f"{int(g.n):,}"]
    for m, _ in mets:
        r = d(c, m)
        vals += ["--", "--"] if r is None or pd.isna(r.n) or r.n < 30 else [cell(r.quote_bp, r.quote_se_ev_bp), cell(r.diff_bp, r.diff_se_ev_bp)]
    tex.append(lab + " & " + " & ".join(vals) + r" \\")
tex += [r"\bottomrule", r"\end{tabular}"]
(TABLES / "quote_validation.tex").write_text("\n".join(tex) + "\n")

groups = [("All trades", "pooled"), ("Politics $<\\$100$", "Politics | s1 <$100"), ("Politics $\\ge\\$100$", "Politics | >=$100"),
          ("Crypto $<\\$100$", "Crypto | s1 <$100"), ("Crypto $\\ge\\$100$", "Crypto | >=$100"),
          ("Fresh pair", "pooled | fresh"), ("2--5 trades old", "pooled | stale 2-5"), ("$>5$ trades old", "pooled | stale >5"),
          ("Crossed pair", "pooled | crossed pair"), ("Uncrossed pair", "pooled | uncrossed pair"),
          ("Quoted spread $\\le$2c", "pooled | quoted spread <= 2c"), ("Quoted spread $>$2c", "pooled | quoted spread > 2c")]
tex = [r"\begin{tabular}{lrrrrr}", r"\toprule", r"Trades & $n$ & $De$ & $|e|$ & Outside & $D(p-m)<0$ \\", r"\midrule"]
for lab, gname in groups:
    if gname is None: continue
    g = Bq[Bq.group == gname]
    if not len(g): continue
    g = g.iloc[0]
    tex.append(f"{lab} & {int(g.n):,} & {cell(g.mean_De_bp, g.mean_De_se_ev_bp)} & {f0(g.mae_bp)} & {100 * g.share_proxy_outside_quotes:.1f}\\% & "
               f"{100 * g.share_cost_negative_proxy:.1f} / {100 * g.share_cost_negative_quote:.1f}\\%" + r" \\")
    if gname in ("pooled", "Crypto | >=$100", "pooled | stale >5", "pooled | uncrossed pair"): tex.append(r"\addlinespace")
tex += [r"\bottomrule", r"\end{tabular}"]
(TABLES / "quote_error.tex").write_text("\n".join(tex) + "\n")

_np, _nq = float(M["QNegProxyAll"]), float(M["QNegQuoteAll"])
_fr = (_np - _nq) / _np
M["QNegNoisePct"] = f"{100 * _fr:.1f}"
_noise = "most" if _fr > 0.6 else ("about half" if _fr >= 0.4 else "a minority")
_rc = Dq[Dq.metric.str.startswith("over_") & Dq.cell.isin(["pooled", "Politics", "Crypto", "Politics | s1 <$100", "Crypto | s1 <$100",
                                                           "Politics | >=$100", "Crypto | >=$100"])].diff_bp.dropna()
M["QResidMaxAbs"] = f"{_rc.abs().max():.1f}"
Hq = rd("qv_H_recorded.csv"); Fpc = Fq[Fq.cat.isin(["Politics", "Crypto"])]
M["QRecPct"] = f"{100 * Fpc.trades_main_fetched.sum() / Fpc.trades_main_on_days.sum():.1f}"
_z = []
for cname, pre in [("Politics | s1 <$100", "PolS"), ("Crypto | s1 <$100", "CryS")]:
    a_, b_ = Hq[(Hq.cell == cname) & Hq.recorded].iloc[0], Hq[(Hq.cell == cname) & ~Hq.recorded].iloc[0]
    M[f"QRec{pre}"] = f0(a_.over_100_bp); M[f"QRec{pre}SE"] = f"{a_.over_100_se_ev_bp:.1f}"
    M[f"QOmit{pre}"] = f0(b_.over_100_bp); M[f"QOmit{pre}SE"] = f"{b_.over_100_se_ev_bp:.1f}"
    _z.append(abs(a_.over_100_bp - b_.over_100_bp) / math.hypot(a_.over_100_se_ev_bp, b_.over_100_se_ev_bp))
_repr = ("not distinguishable from the omitted ones on this measure" if max(_z) < 1.96
         else "not representative of the omitted ones on this measure, and quote-based levels are reported only as proxy-minus-quote differences")
M["QuoteParagraph"] = (
    f"Table~\\ref{{tab:quotes}} compares the two midpoints on identical trades. The signed proxy error is small: $De$ averages "
    f"$\\QDeAll$ bp (SE \\QDeAllSE) pooled, $\\QDePolS$ (\\QDePolSSE) for small Politics orders and $\\QDeCryS$ (\\QDeCrySSE) for small Crypto orders, "
    f"against simulated S1 errors of 61 bp. It grows with staleness: relative to fresh pairs, pairs more than five trades old shift $De$ by $\\QRegStale$ bp (SE \\QRegStaleSE) "
    f"after controlling for topic, size, crossing and the quoted spread. The share of trades with negative execution cost falls from \\QNegProxyAll\\% with the proxy "
    f"to \\QNegQuoteAll\\% with quotes, so {_noise} of the proxy's negative-cost trades reflect proxy error rather than trades inside the spread. "
    f"Across the topic and size cells of the table, proxy and quote residuals differ by at most \\QResidMaxAbs{{}} bp at any horizon. "
    f"Replacing the proxy with quote midpoints leaves the one-hour residual of small Politics orders at $\\QPolSHourRQ$ bp (SE \\QPolSHourRQSE; proxy $\\QPolSHourRP$) under R "
    f"and the small Crypto residual at $\\QCrySHourRQ$ (\\QCrySHourRQSE). With quote midpoints, the between component of small Politics orders is "
    f"$\\QWBPolSBtwQ$ bp (\\QWBPolSBtwQSE) and their within component $\\QWBPolSWtnQ$ (\\QWBPolSWtnQSE).")
M["QuoteAppendix"] = (
    f"We use the public pmxt order-book archive in the Hugging Face normalisation \\nolinkurl{{Joseph3222/}}\\allowbreak\\nolinkurl{{polymarket-orderbook}}, which stores each \\texttt{{price\\_change}} and \\texttt{{book}} event with the exchange's best bid and ask. "
    f"We download only the row groups and columns needed for the target markets and the intervals from 15 minutes before to two hours after each main-sample trade, "
    f"plus the 100-trade endpoint (script 13a). Token books are mapped to the reference outcome with the on-chain token map; the complementary book is mirrored. "
    f"Block timestamps follow the off-chain match: with no lead, \\QMatchZeroPct\\% of trades execute at the touched side; the share peaks at \\QMatchPct\\% at a \\QLead-second lead, "
    f"where \\QCostPosPct\\% of trades have positive quoted execution cost. Multi-fill trades walk the book, so exact matches are not expected for all trades. "
    f"The archive records \\QRecPct\\% of the main-sample Politics and Crypto trades on the sampled days; the omitted markets do not appear in the archive rows that cover their trading windows, so the gap is in the recorder's market list rather than in our selection. "
    f"On the sampled days, the proxy residual $\\ovr_{{100}}$ of small Politics orders is $\\QRecPolS$ bp (SE \\QRecPolSSE) in recorded and $\\QOmitPolS$ (\\QOmitPolSSE) in omitted markets; "
    f"for small Crypto orders it is $\\QRecCryS$ (\\QRecCrySSE) and $\\QOmitCryS$ (\\QOmitCrySSE), so the recorded markets are {_repr}. "
    f"After requiring a fresh two-sided quote and same-day horizons, the validation keeps \\QCoverPct\\% of all main-sample Politics and Crypto trades on these days. The quoted half-spread averages \\QHSPolS{{}} bp for small Politics orders and \\QHSCryS{{}} bp for small Crypto orders. "
    f"Mean absolute proxy error is \\QMAEFresh{{}} bp for fresh pairs and \\QMAEStale{{}} bp for pairs older than five trades. Topic and size coefficients in the error regression are "
    f"$\\QRegPol$ (\\QRegPolSE) bp for Politics, $\\QRegMid$ (\\QRegMidSE) for \\$100--1k and $\\QRegLarge$ (\\QRegLargeSE) for $\\ge$\\$1k.")
