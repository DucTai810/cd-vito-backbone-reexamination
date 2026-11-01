"""Render Fig. 4 from the 24 saved development interval cells; intervals are read, not recomputed."""
from pathlib import Path
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / 'results/intervals/development_intervals.json'
DEST = ROOT / 'outputs/figures'
WIDTH, HEIGHT, FONT = 4.8, 2.45, 8

def main():
    data = json.loads(SOURCE.read_text(encoding='utf-8'))
    assert data['status'] == 'COMPLETE' and data['configuration_domain_cells'] == 24
    rows = data['rows']
    cfgs = list(dict.fromkeys(r['configuration'] for r in rows))
    assert len(cfgs) == 8 and len(rows) == 24
    def label(c):
        if c.startswith('K'):
            return f'last {c[1:]}'
        r, h = c.removeprefix('ADAMW_r').split('_h')
        return f'AdamW {r}/{h}'
    plt.rcParams.update({'font.family':'serif', 'font.size':FONT, 'axes.labelsize':FONT,
                         'axes.titlesize':FONT, 'xtick.labelsize':FONT,
                         'ytick.labelsize':FONT, 'legend.fontsize':FONT, 'pdf.fonttype':42})
    fig, axes = plt.subplots(1, 3, figsize=(WIDTH, HEIGHT), sharey=True)
    fig.subplots_adjust(left=.275, right=.99, bottom=.18, top=.77, wspace=.15)
    ink, gray = '#202020', '#858585'
    for ax, ds, title in zip(axes, ['artaxor','clipart1k','neu-det'], ['ArTaxOr','Clipart1k','NEU-DET']):
        selected = {r['configuration']:r for r in rows if r['dataset'] == ds}
        ax.axvspan(.8, .9, color='#f0f0f0', zorder=0)
        ax.axvline(.9, color=gray, ls='--', lw=.8)
        ax.axvline(1, color=ink, lw=.7)
        ax.axhline(3.5, color='#dddddd', lw=.8)
        for i, c in enumerate(cfgs):
            row = selected[c]
            recovered = row['R_CI95_low'] >= .9
            assert recovered == row['recovered_LCB_ge_0_9']
            ax.plot([row['R_CI95_low'],row['R_CI95_high']], [i,i], color=ink, lw=1.1)
            ax.plot(row['R'], i, marker='o' if c.startswith('K') else 's', ms=4.2,
                    mfc=ink if recovered else 'white', mec=ink, ls='none')
        ax.set_title(title, pad=5)
        ax.set_xlim(.8,1.52)
        ax.set_xticks([.9,1.2,1.5])
        ax.tick_params(axis='y',length=0)
        ax.spines[['top','right']].set_visible(False)
        ax.grid(axis='x',color='#dddddd',lw=.5)
    axes[0].set_yticks(range(8),[label(c) for c in cfgs])
    axes[0].set_ylim(7.6,-.6)
    fig.text(.63,.035,'Mean per-sampling R vs. A (95% CI)',ha='center',fontsize=FONT)
    handles = [Line2D([],[],color=ink,marker='o',mfc=ink,ls='none',label='LCB ≥ 0.9'),
               Line2D([],[],color=ink,marker='o',mfc='white',ls='none',label='LCB < 0.9'),
               Line2D([],[],color=gray,ls='--',label='Recovery 0.9'),
               Line2D([],[],color=ink,lw=.7,label='Frozen A (R=1)')]
    fig.legend(handles=handles,loc='upper center',bbox_to_anchor=(.60,1.0),ncol=2,
               frameon=False,handlelength=1.1,columnspacing=1.0,labelspacing=.4)
    # Fixed canvas, not bbox_inches='tight': prevents shrinking labels below 8 pt in LaTeX.
    DEST.mkdir(parents=True,exist_ok=True)
    fig.savefig(DEST/'TN7_TN13_R_intervals.pdf')
    fig.savefig(DEST/'TN7_TN13_R_intervals.png',dpi=200)
    plt.close(fig)
    print(json.dumps({'cells':len(rows),'width_inches':WIDTH,'height_inches':HEIGHT,
                      'minimum_font_points_at_canvas_width':FONT,'statistics_recomputed':False}))

if __name__ == '__main__':
    main()
