"""New figure layout only; original plotted CSV values are unchanged."""
from pathlib import Path
import csv
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
root=Path(__file__).resolve().parents[1];out=root/'outputs/figures';out.mkdir(parents=True,exist_ok=True)
def read(n):return list(csv.DictReader((root/'results/figure_data'/n).open(encoding='utf-8-sig')))
plt.rcParams.update({'font.family':'serif','font.size':8,'axes.labelsize':8,'xtick.labelsize':7,'ytick.labelsize':7,'pdf.fonttype':42})
colors=['#2866a5','#c56626','#30865c'];domains=['artaxor','clipart1k','neu-det']
a=read('data_fig3_lr_sweep.csv');b=read('data_fig4a_drift.csv')
fig,axes=plt.subplots(1,2,figsize=(4.8,2.35),layout='constrained')
for d,c in zip(domains,colors):
    v=sorted([r for r in a if r['series']==d],key=lambda r:float(r['r']))
    axes[0].plot([float(r['r']) for r in v],[float(r['R']) for r in v],'-o',ms=3,color=c,label=d)
    v=sorted([r for r in b if r['domain']==d],key=lambda r:float(r['r']))
    axes[1].plot([float(r['r']) for r in v],[float(r['median_rel_change_24_blocks'])*100 for r in v],'-',color=c)
    for r in v:axes[1].plot(float(r['r']),float(r['median_rel_change_24_blocks'])*100,'o',ms=4,color=c,mfc='white' if r['collapsed_R_le_0.2']=='1' else c)
v=sorted([r for r in a if r['series']=='pooled'],key=lambda r:float(r['r']))
xx=[float(r['r']) for r in v];axes[0].plot(xx,[float(r['R']) for r in v],color='black',lw=1);axes[0].fill_between(xx,[float(r['ci_low']) for r in v],[float(r['ci_high']) for r in v],color='black',alpha=.13)
axes[0].axhline(.2,color='gray',ls=':',lw=.7);axes[0].axhline(.9,color='gray',ls='--',lw=.7)
axes[0].set_ylabel('Recovery ratio R');axes[1].set_ylabel('Median weight change (%)');axes[1].set_yscale('log')
for ax,title in zip(axes,['(a) Recovery','(b) Final weight change']):ax.set_xscale('log');ax.set_xlabel('Backbone/head LR ratio r');ax.set_title(title);ax.grid(alpha=.15)
axes[0].legend(fontsize=6,loc='upper right');fig.savefig(out/'lr_and_drift.pdf');plt.close(fig)
t=read('data_fig4b_trajectories.csv');fig,axes=plt.subplots(1,2,figsize=(4.8,1.9),layout='constrained')
for ax,d in zip(axes,domains[:2]):
    for r,c in zip(['0.00316','0.01','1'],colors):
        v=sorted([x for x in t if x['domain']==d and float(x['r'])==float(r)],key=lambda x:float(x['fraction']))
        ax.plot([float(x['fraction']) for x in v],[float(x['probe_AP']) for x in v],'-o',ms=2,color=c,label='r='+r)
    ax.set_title('ArTaxOr' if d=='artaxor' else 'Clipart1k');ax.set_xlabel('Training fraction');ax.grid(alpha=.15)
axes[0].set_ylabel('Probe-set AP');axes[1].legend(fontsize=6);fig.savefig(out/'probe_trajectories.pdf');plt.close(fig)
print('New panels rendered from original CSVs; no source data changed.')
