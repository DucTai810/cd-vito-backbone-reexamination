"""Render Fig. 2 (recovery and final weight change) and Fig. 3 (probe-set AP) from saved CSVs."""
from pathlib import Path
import csv
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / 'results/figure_data'
DEST = ROOT / 'outputs/figures'
FONT = 8
COLORS = ['#2866a5', '#c56626', '#30865c']
DOMAINS = {'artaxor': 'ArTaxOr', 'clipart1k': 'Clipart1k', 'neu-det': 'NEU-DET'}

def read(name):
    with (SOURCE / name).open(encoding='utf-8-sig') as f:
        return list(csv.DictReader(f))

def by_rate(rows):
    return sorted(rows, key=lambda r: float(r['r']))

def render_recovery_and_drift():
    sweep = read('data_fig3_lr_sweep.csv')
    drift = read('data_fig4a_drift.csv')
    fig, axes = plt.subplots(1, 2, figsize=(4.8, 2.55), layout='constrained')
    for (domain, name), color in zip(DOMAINS.items(), COLORS):
        rows = by_rate(r for r in sweep if r['series'] == domain)
        axes[0].plot([float(r['r']) for r in rows], [float(r['R']) for r in rows],
                     '-o', ms=3, color=color, label=name)
        rows = by_rate(r for r in drift if r['domain'] == domain)
        change = [float(r['median_rel_change_24_blocks']) * 100 for r in rows]
        axes[1].plot([float(r['r']) for r in rows], change, '-', color=color)
        for r, y in zip(rows, change):
            collapsed = r['collapsed_R_le_0.2'] == '1'
            axes[1].plot(float(r['r']), y, 'o', ms=4, color=color, mfc='white' if collapsed else color)
    rows = by_rate(r for r in sweep if r['series'] == 'pooled')
    rates = [float(r['r']) for r in rows]
    axes[0].plot(rates, [float(r['R']) for r in rows], color='black', lw=1, label='pooled mean (95% CI)')
    axes[0].fill_between(rates, [float(r['ci_low']) for r in rows], [float(r['ci_high']) for r in rows],
                         color='black', alpha=.13)
    axes[0].axhline(.2, color='gray', ls=':', lw=.7)
    axes[0].axhline(.9, color='gray', ls='--', lw=.7)
    axes[0].set_ylabel('Recovery ratio R')
    axes[1].set_ylabel('Median weight change (%)')
    axes[1].set_yscale('log')
    for ax, title in zip(axes, ['(a) Recovery', '(b) Final weight change']):
        ax.set_xscale('log')
        ax.set_xlabel('Backbone/head LR ratio r')
        ax.set_title(title)
        ax.grid(alpha=.15)
    # Outside the axes: inside panel (a) the legend covers curves at every corner.
    fig.legend(*axes[0].get_legend_handles_labels(), loc='outside upper center', ncol=4,
               fontsize=FONT, frameon=False, handlelength=1.1, columnspacing=1.0)
    fig.savefig(DEST / 'lr_and_drift.pdf')
    plt.close(fig)

def render_probe_trajectories():
    trajectories = read('data_fig4b_trajectories.csv')
    fig, axes = plt.subplots(1, 2, figsize=(4.8, 1.9), layout='constrained')
    for ax, (domain, name) in zip(axes, list(DOMAINS.items())[:2]):
        for rate, color in zip(['0.00316', '0.01', '1'], COLORS):
            rows = sorted((x for x in trajectories if x['domain'] == domain and float(x['r']) == float(rate)),
                          key=lambda x: float(x['fraction']))
            ax.plot([float(x['fraction']) for x in rows], [float(x['probe_AP']) for x in rows],
                    '-o', ms=2, color=color, label='r=' + rate)
        ax.set_title(name)
        ax.set_xlabel('Training fraction')
        ax.grid(alpha=.15)
    axes[0].set_ylabel('Probe-set AP')
    axes[1].legend(fontsize=FONT)
    fig.savefig(DEST / 'probe_trajectories.pdf')
    plt.close(fig)

def main():
    plt.rcParams.update({'font.family': 'serif', 'font.size': FONT, 'axes.labelsize': FONT,
                         'xtick.labelsize': FONT, 'ytick.labelsize': FONT, 'pdf.fonttype': 42})
    DEST.mkdir(parents=True, exist_ok=True)
    render_recovery_and_drift()
    render_probe_trajectories()
    print('Rendered lr_and_drift.pdf and probe_trajectories.pdf from saved CSVs.')

if __name__ == '__main__':
    main()
