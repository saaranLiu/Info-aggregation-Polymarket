"""Source-backed categorical comparisons and sensitivity plots for the paper."""
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
from paper_style import *


def topic_handles():
    return [Line2D([], [], color=TOPICS[t], marker=MARKERS[t], ls='', ms=4,
                   label='Yes/No' if t == 'pooled' else t) for t in ['pooled', 'Politics', 'Crypto']]


def fig_weights():
    d = pd.read_csv(RES / 'audit_C_weights_YesNo.csv').fillna({'cat': 'pooled'})
    d = d[d.metric == 'RS']
    fig = plt.figure(figsize=(WIDE, 2.75))
    a = fig.add_axes([.095, .21, .345, .60])
    b = fig.add_axes([.652, .21, .335, .60])
    order = ['equal_market', 'count', 'shares', 'usd']
    for topic, off, fill in [('pooled', -.22, '#B6B6B6'), ('Politics', 0, '#EAA4C5'), ('Crypto', .22, '#9DBFE0')]:
        r = d[d.cat == topic].set_index('weight').loc[order]
        a.barh(np.arange(4)+off, r.est*1e4, height=.19, color=fill, edgecolor=TOPICS[topic], linewidth=.65, zorder=3)
    a.set_yticks(range(4), ['Per market', 'Per trade', 'Per share', 'Per dollar'])
    a.set_ylim(3.6, -.6); a.set_xlim(-700, 350)
    a.set_xlabel('Settlement margin (bp)'); style(a, 'x'); panel(a, 'a', 'Margins by weighting')
    fig.legend(handles=[Patch(facecolor=fill, edgecolor=TOPICS[t], linewidth=.65, label=label)
                        for t,fill,label in [('pooled','#B6B6B6','Yes/No'),('Politics','#EAA4C5','Politics'),('Crypto','#9DBFE0','Crypto')]],
               loc='upper center', bbox_to_anchor=(.26, .995), ncol=3, columnspacing=.85, handlelength=1.1)
    c = pd.read_csv(RES / 'con_A_concentration.csv')
    rows = [('Yes/No', 'trade', 'Yes/No, per trade', NAVY), ('Yes/No', 'usd', 'Yes/No, per dollar', NAVY),
            ('Politics', 'usd', 'Politics, per dollar', POL), ('Crypto', 'usd', 'Crypto, per dollar', CRY)]
    for y, (cat, w, _, color) in enumerate(rows):
        r = c[(c.cat == cat) & (c.weight == w)].iloc[0]
        b.plot([r.mean_bp, r.drop5_bp], [y-.11, y+.11], color='#B5B5B5', lw=.7, zorder=2)
        for off, value, se, face in [(-.11, r.mean_bp, r.mean_se_bp, color), (.11, r.drop5_bp, r.drop5_se_bp, 'white')]:
            b.errorbar(value, y+off, xerr=1.96*se, color=color, fmt='o', mfc=face,
                       ms=3.8, capsize=2, elinewidth=.85, markeredgewidth=.8, zorder=3)
    b.set_yticks(range(4), [r[2] for r in rows]); b.set_ylim(3.6, -.6)
    b.set_xlim(-1250, 230); b.set_xlabel('Settlement margin (bp)')
    style(b, 'x'); panel(b, 'b', 'Resolution-date sensitivity')
    h = [Line2D([], [], marker='o', color=INK, ls='', ms=3.8, label='Full sample'),
         Line2D([], [], marker='o', color=INK, mfc='white', ls='', ms=3.8, label='Five dates removed')]
    fig.legend(handles=h, loc='upper center', bbox_to_anchor=(.78, .995), ncol=2, columnspacing=.95)
    finish(fig, 'weights')


def fig_size():
    d = pd.read_csv(RES / 'inf_A_horizon_samples.csv')
    d = d[(d.grp == 'YesNo') & (d['sample'] == 'horizon')]
    fig, axes = plt.subplots(1, 3, figsize=(WIDE, 2.35), sharey=True)
    metrics = [('info', 'Outcome alignment'), ('RS', 'Settlement margin'), ('over_100', 'H100 residual')]
    for ax, (metric, title), letter in zip(axes, metrics, 'abc'):
        for topic, off in [('Politics', -.10), ('Crypto', .10)]:
            cells = [f'{topic} | s1 <$100', f'{topic} | s2 $100-1k', f'{topic} | s3 >=$1k']
            r = d[d.metric == metric].set_index('cell').loc[cells]
            ax.errorbar(r.est_bp, np.arange(3)+off, xerr=1.96*r.se_ev, color=TOPICS[topic],
                        fmt=MARKERS[topic], ms=4, capsize=2, elinewidth=.85, markeredgewidth=.7, label=topic)
        if metric == 'over_100': ax.axvspan(-50, 50, color=PALE, zorder=0)
        ax.set_yticks(range(3), ['<$100', '$100–1,000', '≥$1,000']); ax.set_ylim(2.55, -.55)
        style(ax, 'x', bins=3); panel(ax, letter, title)
    axes[0].set_ylabel('Order notional')
    handles = [Line2D([], [], marker=MARKERS[t], color=TOPICS[t], ls='', label=t) for t in ['Politics','Crypto']]
    fig.legend(handles=handles, loc='upper center', ncol=2, bbox_to_anchor=(.55, 1.0), columnspacing=1.6)
    fig.text(.55, .015, 'Probability bp', ha='center', fontsize=9)
    fig.subplots_adjust(left=.13, right=.985, top=.76, bottom=.24, wspace=.32)
    finish(fig, 'size_pricing')


VARIANTS = ['full (main)', 'fresh (stale<=1)', 'pair within 60s', 'pair within 60s and proxy age <= 10 min',
            'uncrossed pair', 'uncrossed and fresh', 'stale<=1 and pair within 10s', 'sign-consistent D(p-m)>=0']
NAMES = ['Main', 'Fresh', 'Pair ≤60 s', 'Pair ≤60 s; age ≤10 min', 'Uncrossed', 'Uncrossed + fresh', 'Fresh; pair ≤10 s', 'Sign-consistent selection']


def fig_proxy_full():
    d = pd.read_csv(RES / 'proxy_B_variants_YesNo.csv')
    cells = [('Politics | s1 <$100', 'Politics, <$100', POL), ('Politics | s3 >=$1k', 'Politics, ≥$1,000', POL),
             ('Crypto | s1 <$100', 'Crypto, <$100', CRY), ('Crypto | s3 >=$1k', 'Crypto, ≥$1,000', CRY)]
    fig, axes = plt.subplots(3, 4, figsize=(WIDE, 4.7), sharey=True)
    for row, (metric, label) in enumerate([('info', 'Alignment'), ('RS', 'Margin'), ('over_100', 'H100 residual')]):
        for col, (cell, title, color) in enumerate(cells):
            ax = axes[row, col]
            sub = d[(d.cell == cell) & (d.metric == metric)].set_index('variant').loc[VARIANTS]
            for y, (_, r) in enumerate(sub.iterrows()):
                ax.errorbar(r.est*1e4, y, xerr=[[max(0,r.est-r.lo)*1e4], [max(0,r.hi-r.est)*1e4]],
                            fmt='s' if y == 7 else 'o', color=color, mfc=color if y == 0 else 'white',
                            ms=2.8, capsize=1.3, elinewidth=.65)
            ax.set_yticks(range(8), NAMES, fontsize=7); ax.set_ylim(7.6, -.6)
            style(ax, 'x', bins=3); ax.tick_params(axis='x', labelsize=7)
            if row == 0: ax.set_title(f'({"abcd"[col]}) {title}', fontsize=8.5, pad=6)
            if col == 0: ax.set_ylabel(label, fontsize=9)
    fig.text(.63, .015, 'Probability bp', ha='center', fontsize=9)
    fig.subplots_adjust(left=.265, right=.985, top=.92, bottom=.10, wspace=.30, hspace=.40)
    finish(fig, 'proxy_sensitivity_full', ROOT / 'paper/supplementary_figures')


def fig_proxy():
    d = pd.read_csv(RES / 'proxy_B_variants_YesNo.csv')
    names = ['Main', 'Fresh', 'Pair ≤60 s', 'Pair ≤60 s;\nage ≤10 min', 'Uncrossed', 'Uncrossed + fresh', 'Fresh; pair ≤10 s', 'Sign-consistent\nselection']
    fig, ax = plt.subplots(figsize=(COL, 3.05))
    ax.axvspan(-50, 50, color=PALE, zorder=0)
    for topic, off in [('Politics', -.12), ('Crypto', .12)]:
        sub = d[(d.cell == f'{topic} | s1 <$100') & (d.metric == 'over_100')].set_index('variant').loc[VARIANTS]
        for y, (_, r) in enumerate(sub.iterrows()):
            ax.errorbar(r.est*1e4, y+off, xerr=[[max(0,r.est-r.lo)*1e4], [max(0,r.hi-r.est)*1e4]],
                        fmt=MARKERS[topic], color=TOPICS[topic], mfc=TOPICS[topic] if y == 0 else 'white',
                        ms=3.3, capsize=1.5, elinewidth=.8, markeredgewidth=.75)
    ax.set_yticks(range(8), names, fontsize=8); ax.set_ylim(7.6, -.6)
    ax.set_xlabel('Small-order H100 residual (bp)', fontsize=8.5)
    style(ax, 'x', bins=4); ax.spines['left'].set_visible(False); ax.tick_params(axis='y', length=0)
    handles = [Line2D([], [], marker=MARKERS[t], color=TOPICS[t], ls='', label=t) for t in ['Politics','Crypto']]
    fig.legend(handles=handles, loc='upper center', ncol=2, bbox_to_anchor=(.65, 1.0), columnspacing=.9)
    fig.subplots_adjust(left=.43, right=.98, top=.88, bottom=.17)
    finish(fig, 'proxy_sensitivity')


def fig_robustness():
    d = pd.read_csv(RES / 'robustness_matrix_pooled.csv')
    variants = list(dict.fromkeys(d.variant)); metrics = ['info','eff','RS','PI_100','over_100']
    vals = d.pivot(index='variant', columns='metric', values='est_bp').loc[variants,metrics]
    lo = d.pivot(index='variant', columns='metric', values='lo_bp').loc[variants,metrics]
    hi = d.pivot(index='variant', columns='metric', values='hi_bp').loc[variants,metrics]
    labels = ['Main', 'Per trade (H100)', 'Per market (H100)', 'Per share (H100)', 'Per dollar (H100)',
              'Scheduled close', 'Metadata resolution', 'Midpoint filter', 'Price + midpoint filter', 'Wide midpoint filter',
              'Fresh midpoint', 'Short pair + fresh age', 'Uncrossed pair', 'Event-cluster SE', 'Two-way SE']
    fig, ax = plt.subplots(figsize=(COL, 3.45))
    cmap = LinearSegmentedColormap.from_list('pink_blue', ['#9CBFDF','#FCFCFC','#E9A2C3'])
    im = ax.imshow(vals, cmap=cmap, norm=TwoSlopeNorm(vcenter=0, vmin=-400, vmax=400), aspect='auto')
    for i in range(len(variants)):
        for j in range(len(metrics)):
            star = '*' if lo.iloc[i,j] > 0 or hi.iloc[i,j] < 0 else ''
            ax.text(j, i, f'{vals.iloc[i,j]:.1f}{star}', ha='center', va='center', fontsize=7.5)
    ax.set_xticks(range(5), ['Align.','Cost','Margin','Resp.','Resid.'], fontsize=8)
    ax.xaxis.tick_top(); ax.set_yticks(range(len(labels)), labels, fontsize=7.5)
    ax.set_xticks(np.arange(-.5,5,1), minor=True); ax.set_yticks(np.arange(-.5,len(labels),1), minor=True)
    ax.grid(which='minor', color='white', linewidth=.8); ax.tick_params(which='both', length=0)
    for y in [.5,4.5,6.5,9.5,12.5]: ax.axhline(y, color='#BBBBBB', lw=.45)
    for s in ax.spines.values(): s.set_visible(False)
    fig.subplots_adjust(left=.46, right=.985, top=.93, bottom=.16)
    cax = fig.add_axes([.49,.085,.46,.025]); cb=fig.colorbar(im,cax=cax,orientation='horizontal',ticks=[-400,0,400])
    cb.ax.xaxis.set_major_formatter(StrMethodFormatter('{x:,.1f}'))
    cb.ax.tick_params(labelsize=7, length=2, width=.5, pad=1); cb.outline.set_linewidth(.4)
    cb.set_label('Probability bp', fontsize=8, labelpad=1)
    finish(fig, 'robustness_matrix', ROOT / 'paper/supplementary_figures')


if __name__ == '__main__':
    fig_weights(); fig_size(); fig_proxy_full(); fig_proxy(); fig_robustness()
