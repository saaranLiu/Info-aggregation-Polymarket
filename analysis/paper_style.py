"""Times-based publication figures, with fixed physical dimensions and embedded fonts."""
from pathlib import Path
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.ticker import MaxNLocator, StrMethodFormatter

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / 'results'
FIGS = ROOT / 'paper' / 'figures'
COL, WIDE = 3.34, 7.0
POL, CRY, NAVY = '#D94D92', '#2F73B5', '#686868'
GREY, INK, GRID, PALE = '#888888', '#171717', '#E4E4E4', '#F1F1F1'
TOPICS = {'Politics': POL, 'Crypto': CRY, 'pooled': NAVY}
MARKERS = {'Politics': 'D', 'Crypto': 'o', 'pooled': 's'}

# Keep the manuscript's Times default; an explicit serif override supports Linux.
FONT_FAMILY = os.environ.get('FIGURE_FONT_FAMILY', 'Times New Roman')
font_dirs = [Path('/System/Library/Fonts/Supplemental'), Path('/Library/Fonts'),
             Path('/usr/share/fonts/truetype/msttcorefonts'), Path('C:/Windows/Fonts')]
if os.environ.get('FIGURE_FONT_DIR'):
    font_dirs.insert(0, Path(os.environ['FIGURE_FONT_DIR']))
for folder in font_dirs:
    if folder.is_dir():
        for p in folder.glob('*.ttf'):
            if 'times' in p.name.lower(): font_manager.fontManager.addfont(str(p))
font_manager.findfont(font_manager.FontProperties(family=FONT_FAMILY), fallback_to_default=False)
plt.rcParams.update({
    'font.family': FONT_FAMILY, 'font.size': 9,
    'mathtext.fontset': 'custom', 'mathtext.rm': FONT_FAMILY,
    'mathtext.it': FONT_FAMILY+':italic', 'mathtext.bf': FONT_FAMILY+':bold',
    'mathtext.bfit': FONT_FAMILY+':bold:italic', 'mathtext.sf': FONT_FAMILY,
    'mathtext.tt': FONT_FAMILY, 'mathtext.cal': FONT_FAMILY+':italic',
    'mathtext.fallback': None,
    'axes.titlesize': 9, 'axes.titleweight': 'bold', 'axes.titlecolor': INK,
    'axes.labelsize': 9, 'xtick.labelsize': 8, 'ytick.labelsize': 8,
    'legend.fontsize': 8.5, 'legend.title_fontsize': 8.5,
    'text.color': INK, 'axes.labelcolor': INK, 'xtick.color': INK, 'ytick.color': INK,
    'axes.spines.top': False, 'axes.spines.right': False,
    'axes.linewidth': .65, 'axes.edgecolor': '#444444',
    'grid.color': GRID, 'grid.linewidth': .45, 'axes.axisbelow': True,
    'legend.frameon': False, 'legend.fancybox': False,
    'legend.handlelength': 1.5, 'legend.handletextpad': .45,
    'lines.linewidth': 1, 'lines.markersize': 4,
    'pdf.fonttype': 42, 'ps.fonttype': 42,
    'savefig.facecolor': 'white', 'figure.facecolor': 'white',
})


def finish(fig, name, folder=FIGS):
    folder.mkdir(parents=True, exist_ok=True)
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    labels = list(fig.texts)
    for ax in fig.axes:
        labels += list(ax.texts) + [ax.title, ax._left_title, ax._right_title]
        if ax.axison:
            labels += [ax.xaxis.label, ax.yaxis.label]
            for axis in [ax.xaxis, ax.yaxis]:
                low, high = sorted(axis.get_view_interval())
                for tick in axis.get_major_ticks():
                    if low <= tick.get_loc() <= high:
                        labels += [tick.label1, tick.label2]
    for legend in fig.legends:
        labels += legend.get_texts()
    outside = []
    for label in labels:
        if label.get_visible() and label.get_text():
            box = label.get_window_extent(renderer)
            if box.x0 < 0 or box.y0 < 0 or box.x1 > fig.bbox.width or box.y1 > fig.bbox.height:
                outside.append(label.get_text())
    if outside:
        raise ValueError(f'{name}: text exceeds fixed canvas: {outside}')
    # Preserve the designed font sizes at the manuscript's actual column width.
    # Every label belongs within the fixed canvas, with no crop-dependent scaling.
    for ext in ['pdf', 'png']:
        fig.savefig(folder / f'{name}.{ext}', dpi=360)
    plt.close(fig)
    print(f'Figure: {name}', flush=True)


def style(ax, axis='y', zero=True, bins=4):
    ax.grid(axis=axis)
    target = ax.yaxis if axis == 'y' else ax.xaxis
    target.set_major_locator(MaxNLocator(bins))
    target.set_major_formatter(StrMethodFormatter('{x:,.1f}'))
    ax.tick_params(direction='out', length=2.5, width=.6, pad=2)
    if zero:
        (ax.axhline if axis == 'y' else ax.axvline)(0, color='#666666', lw=.75, ls=(0,(3,2)), zorder=1)


def panel(ax, letter, title):
    ax.set_title(f'({letter}) {title}', loc='left', pad=6)
