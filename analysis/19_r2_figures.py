"""Clock and quote figures drawn exclusively from saved result tables."""
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from paper_style import *


def clock_profile():
    d = pd.read_csv(RES / 'clk_A_cells.csv')
    fig, axes = plt.subplots(1, 4, figsize=(WIDE, 2.25), sharey=True)
    metrics = ['over_c600','over_c1800','over_c3600','over_c7200']; minutes=np.array([10,30,60,120])
    titles=['Resolution (R)','Scheduled close (S)','Close −1 h (S1)','Convergence (C)']
    for ax, cut, title, letter in zip(axes, ['R','S','S1','C'], titles, 'abcd'):
        ax.axhspan(-50,50,color=PALE,zorder=0)
        for topic, off in [('Politics',-.025),('Crypto',.025)]:
            r=d[(d.cutoff==cut)&(d.cell==f'{topic} | s1 <$100')].set_index('metric').loc[metrics]
            x=minutes*np.exp(off)
            ax.plot(x,r.est_bp, color=TOPICS[topic], lw=.9, zorder=3)
            ax.errorbar(x,r.est_bp,yerr=1.96*r.se_ev_bp,fmt=MARKERS[topic], color=TOPICS[topic],
                        ms=3.5,capsize=1.7,elinewidth=.75,markeredgewidth=.6,zorder=4)
        style(ax); ax.set_xscale('log',base=2)
        ax.set_xticks(minutes,['10.0','30.0','60.0','120.0']); ax.set_xlim(8,145)
        ax.minorticks_off(); panel(ax,letter,title)
    axes[0].set_ylabel('Small-order residual (bp)')
    handles=[Line2D([],[],color=TOPICS[t],marker=MARKERS[t],label=t) for t in ['Politics','Crypto']]
    fig.legend(handles=handles,loc='upper center',ncol=2,bbox_to_anchor=(.54,1.0),columnspacing=1.5)
    fig.text(.54,.015,'Time after trade (min; log scale)',ha='center',fontsize=9)
    fig.subplots_adjust(left=.08,right=.985,top=.76,bottom=.24,wspace=.23)
    finish(fig,'clock_profile')


def quote_validation():
    a=pd.read_csv(RES/'qv_A_alignment.csv').sort_values('lead_s')
    q=pd.read_csv(RES/'qv_D_metrics.csv')
    fig=plt.figure(figsize=(COL,3.35))
    top=fig.add_axes([.31,.64,.66,.23]); bottom=fig.add_axes([.31,.17,.66,.28])
    best=a.loc[a.share_exact_side.idxmax()]
    top.plot(a.lead_s,a.share_exact_side*100,'o-',color=CRY,ms=3,lw=.9)
    top.axvline(best.lead_s,color=POL,lw=.7,ls=(0,(3,2)),zorder=1)
    top.plot(best.lead_s,best.share_exact_side*100,'D',color=POL,ms=4,zorder=4)
    top.set_xscale('symlog',linthresh=10)
    top.set_xticks([0,5,10,30,60],['0.0','5.0','10.0','30.0','60.0'])
    top.set_xlabel('Quote lead (s)',labelpad=2); top.set_ylabel('Touched-side match (%)',fontsize=8.5)
    style(top,zero=False,bins=3);panel(top,'a',f'Quote lead (selected: {best.lead_s:.1f} s)')
    cells=[('Politics | s1 <$100','Politics, <$100',POL),('Politics | >=$100','Politics, ≥$100',POL),
           ('Crypto | s1 <$100','Crypto, <$100',CRY),('Crypto | >=$100','Crypto, ≥$100',CRY)]
    for y,(cell,label,color) in enumerate(cells):
        for metric,off,marker,face in [('info',-.12,'o',color),('over_100',.12,'s','white')]:
            r=q[(q.cell==cell)&(q.metric==metric)].iloc[0]
            bottom.errorbar(r.diff_bp,y+off,xerr=1.96*r.diff_se_ev_bp,color=color,fmt=marker,mfc=face,
                            ms=3.2,capsize=1.7,elinewidth=.8,markeredgewidth=.7)
    bottom.set_yticks(range(4),[c[1] for c in cells],fontsize=8);bottom.set_ylim(3.55,-.55)
    bottom.set_xlabel('Proxy minus quote (bp)',labelpad=3,fontsize=8.5)
    style(bottom,'x',bins=3);panel(bottom,'b','Paired midpoint differences')
    handles=[Line2D([],[],marker='o',color=INK,ls='',label='Alignment'),
             Line2D([],[],marker='s',color=INK,mfc='white',ls='',label='H100 residual')]
    fig.legend(handles=handles,loc='lower center',ncol=2,bbox_to_anchor=(.61,.005),columnspacing=.9,fontsize=8)
    finish(fig,'quote_validation')


if __name__=='__main__':
    clock_profile();quote_validation()
