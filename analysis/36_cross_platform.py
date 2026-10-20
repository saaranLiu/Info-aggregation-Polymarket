"""Compare saved venue summaries without pooling their events or inference.

Polymarket: main Yes/No sample, 1--24 h before on-chain resolution.
Kalshi: execution-price sample, 1--24 h before exchange close.
Trade shares are descriptive; margin SEs retain each source's event clusters.
No quote-residual or causal platform comparison is inferred from these summaries.
"""
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from paper_style import *
from manuscript_format import cell, tex_decimal


def main():
    counts = pd.read_csv(RES / 'wb_F_size.csv')
    counts = counts[(counts['sample'] == 'main') & (counts.cell == 'pooled')]
    kal = pd.read_csv(RES / 'k_B_margin.csv')
    kabs = kal[(kal.scheme == 'dollars') & kal.n.notna()].copy()
    bins = ['s1 <$100', 's2 $100-1k', 's3 >=$1k']
    shares = []
    for venue, data, field in [('Polymarket', counts, 'size'), ('Kalshi', kabs, 'bin')]:
        data = data.set_index(field).loc[bins]
        for b, r in data.iterrows():
            shares.append(dict(venue=venue, bin=b, n=int(r.n), share_pct=100*r.n/data.n.sum()))
    shares = pd.DataFrame(shares)
    shares.to_csv(RES / 'k_D_participation_comparison.csv', index=False)

    poly = pd.read_csv(RES / 'inf_A_horizon_samples.csv')
    poly = poly[(poly.grp == 'YesNo') & (poly['sample'] == 'all') & (poly.metric == 'RS')]
    margins = []
    for topic in ['Politics', 'Crypto']:
        for b in bins:
            r = poly[poly.cell == f'{topic} | {b}'].iloc[0]
            margins.append(dict(venue='Polymarket', topic=topic, bin=b, n=int(r.n),
                                estimate_bp=r.est_bp, se_bp=r.se_ev, clusters=int(r.G_ev),
                                cluster_definition='topic x on-chain resolution date'))
    for _, r in kabs.iterrows():
        margins.append(dict(venue='Kalshi', topic='All eligible', bin=r['bin'], n=int(r.n),
                            estimate_bp=r.trade_bp, se_bp=r.trade_se, clusters=int(r.n_clusters),
                            cluster_definition='event_ticker'))
    margins = pd.DataFrame(margins)
    margins.to_csv(RES / 'k_E_margin_comparison.csv', index=False)

    fig = plt.figure(figsize=(WIDE, 2.40))
    a = fig.add_axes([.12, .22, .28, .57])
    b = fig.add_axes([.65, .22, .335, .57])
    for venue, offset, color, hatch in [('Polymarket', -.19, NAVY, ''), ('Kalshi', .19, '#BDBDBD', '//')]:
        r = shares[shares.venue == venue].set_index('bin').loc[bins]
        a.barh(np.arange(3)+offset, r.share_pct, height=.33, color=color,
               hatch=hatch, edgecolor=INK, linewidth=.45, label=venue, zorder=3)
        for y, value in enumerate(r.share_pct):
            a.text(value+1.5, y+offset, f'{value:.1f}', va='center', fontsize=8)
    a.set_yticks(range(3), ['<$100', '$100–1,000', '≥$1,000'])
    a.set_ylim(2.65, -.65); a.set_xlim(0, 103)
    a.set_xlabel('Share of trades (%)'); style(a, 'x', zero=False, bins=4)
    panel(a, 'a', 'Small orders dominate activity')
    a.legend(loc='upper left', bbox_to_anchor=(-.01, 1.37), ncol=2, fontsize=8, columnspacing=.9)
    for venue, topic, offset, color, marker, face in [
        ('Polymarket', 'Politics', -.17, POL, 'D', POL),
        ('Polymarket', 'Crypto', 0, CRY, 'o', CRY),
        ('Kalshi', 'All eligible', .17, INK, 's', 'white')]:
        r = margins[(margins.venue == venue) & (margins.topic == topic)].set_index('bin').loc[bins]
        b.errorbar(r.estimate_bp, np.arange(3)+offset, xerr=1.96*r.se_bp,
                   fmt=marker, color=color, mfc=face, ms=3.5, capsize=1.7,
                   elinewidth=.75, markeredgewidth=.7, zorder=3)
    b.set_yticks(range(3), ['<$100', '$100–1,000', '≥$1,000'])
    b.set_ylim(2.65, -.65); b.set_xlim(-1070, 440)
    b.set_xlabel('Settlement margin per trade (bp)'); style(b, 'x', bins=3)
    panel(b, 'b', 'Shared small-order margin sign')
    handles = [Line2D([], [], color=c, marker=m, mfc=f, ls='', label=t, ms=3.5)
               for t,c,m,f in [('PM Politics',POL,'D',POL),('PM Crypto',CRY,'o',CRY),('Kalshi',INK,'s','white')]]
    fig.legend(handles=handles, loc='upper center', bbox_to_anchor=(.78, 1.0),
               ncol=3, fontsize=8, columnspacing=.8, handletextpad=.3)
    finish(fig, 'cross_platform')

    # One appendix table retains both absolute and within-market relative size.
    lines = [r'\begin{tabular}{lrrr}', r'\toprule',
             r'Size definition & Trades & Per trade & Per dollar \\', r'\midrule']
    for scheme, labels in [
        ('dollars', [r'$<\$100$', r'\$100--\$1k', r'$\ge\$1$k', 'Large minus small']),
        ('within-market rank', ['Low third', 'Middle third', 'High third', 'High minus low'])]:
        lines.append(r'\multicolumn{4}{l}{'+('Cash bins' if scheme == 'dollars' else 'Within-market cash ranks')+r'} \\')
        for label, (_, r) in zip(labels, kal[kal.scheme == scheme].iterrows()):
            n = '---' if pd.isna(r.n) else f'{int(r.n):,}'
            cash = '---' if pd.isna(r.cash_bp) else cell(r.cash_bp, r.cash_se)
            lines.append(f'{label} & {n} & {cell(r.trade_bp, r.trade_se)} & {cash}'+r' \\')
        if scheme == 'dollars': lines.append(r'\midrule')
    lines += [r'\bottomrule', r'\end{tabular}']
    (ROOT/'paper/tables/external_kalshi_detail.tex').write_text('\n'.join(lines)+'\n')

    pred = pd.read_csv(RES/'k_C_prediction.csv')
    lines = [r'\begin{tabular}{lrrr}', r'\toprule',
             r'Cohort/model & Markets & Price loss & Gain [95.0\% CI] \\', r'\midrule']
    for label, (_, r) in zip(['Small flow', 'Both flows'], pred.iterrows()):
        gain = f'{tex_decimal(1e4*r.dbrier)} [{tex_decimal(1e4*r.lo)}, {tex_decimal(1e4*r.hi)}]'
        lines.append(f'{label} & {int(r.n):,} & {tex_decimal(1e4*r.brier_price)} & {gain}'+r' \\')
    lines += [r'\bottomrule', r'\end{tabular}']
    (ROOT/'paper/tables/external_kalshi_prediction.tex').write_text('\n'.join(lines)+'\n')


if __name__ == '__main__':
    main()
