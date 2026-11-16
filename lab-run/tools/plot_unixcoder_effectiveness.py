"""Plot measured attack/filter outcomes and detector ROC from saved evidence."""
import argparse,json
from pathlib import Path


def main():
    p=argparse.ArgumentParser(description=__doc__); p.add_argument('--run',type=Path,required=True)
    args=p.parse_args(); report=json.loads((args.run/'unixcoder-effectiveness.json').read_text())
    if report.get('completed') is not True: raise ValueError('Completed evidence required')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from sklearn.metrics import roc_curve
    methods=list(report['detectors']); short={'spectral_signature':'Spectral signature','shrinkage_mahalanobis':'OAS Mahalanobis'}
    plt.rcParams.update({'font.size':11,'axes.titlesize':13,'axes.spines.top':False,'axes.spines.right':False})
    fig,axes=plt.subplots(1,3,figsize=(14,4.8),layout='constrained')
    labels=['Clean victim','Poisoned victim']+[short[m]+'\nwith rejection' for m in methods]
    values=[report['models']['clean']['asr_percent'],report['models']['poisoned_20pct']['asr_percent']]+[report['detectors'][m]['after_filter']['asr_percent'] for m in methods]
    bars=axes[0].barh(labels,values,color=['#5d748b','#a4493d','#2d6b6e','#244d79'])
    axes[0].bar_label(bars,labels=[f'{v:.2f}%' for v in values],padding=4,fontsize=10)
    axes[0].invert_yaxis(); axes[0].set(xlim=(0,116),xticks=[0,25,50,75,100],xlabel='Target hits / original 416 attacks (%)',title='Attack success before and after rejection')
    for i,m in enumerate(methods):
        metrics=report['detectors'][m]['metrics']
        bars=axes[1].bar([i*3,i*3+1],[100*metrics['recall'],100*metrics['clean_false_positive_rate']],width=.8,color=['#2d6b6e','#a4493d'])
        axes[1].bar_label(bars,labels=[f"{100*metrics['recall']:.2f}%",f"{100*metrics['clean_false_positive_rate']:.2f}%"],padding=3,fontsize=10)
    axes[1].set(ylim=(0,112),yticks=[0,25,50,75,100],xticks=[.5,3.5],xticklabels=[short[m] for m in methods],ylabel='Held-out rate (%)',title='Trigger recall and clean false positives')
    from matplotlib.patches import Patch
    axes[1].legend(handles=[Patch(color='#2d6b6e',label='Trigger recall'),Patch(color='#a4493d',label='Clean false positives')],loc='upper center',bbox_to_anchor=(.5,-.1),frameon=False)
    values=[report['detectors'][m]['after_filter']['clean_retention_percent'] for m in methods]
    bars=axes[2].bar([short[m] for m in methods],values,color=['#2d6b6e','#244d79'],width=.6)
    axes[2].bar_label(bars,labels=[f'{v:.2f}%' for v in values],padding=4)
    axes[2].set(ylim=(0,112),yticks=[0,25,50,75,100],ylabel='Clean inputs accepted / 992 (%)',title='Clean-data cost of rejection')
    for ax in axes: ax.grid(axis='x' if ax is axes[0] else 'y',alpha=.15); ax.set_axisbelow(True)
    fig.suptitle('UniXcoder: fixed 20% poison budget, seed 42, validation-calibrated detectors',fontsize=14)
    fig.savefig(args.run/'effectiveness-summary.png',dpi=180); plt.close(fig)
    fig,ax=plt.subplots(figsize=(6,5.2),layout='constrained')
    for m,color in zip(methods,['#2d6b6e','#244d79']):
        rows=[json.loads(s) for s in (args.run/(m+'-test-scores.jsonl')).read_text().splitlines()]
        labels=[int(r['kind']=='triggered') for r in rows]; scores=[r['score'] for r in rows]
        x,y,_=roc_curve(labels,scores); metric=report['detectors'][m]['metrics']
        ax.plot(x,y,color=color,linewidth=2,label=f"{short[m]} (AUC {metric['roc_auc']:.4f})")
        ax.scatter([metric['clean_false_positive_rate']],[metric['recall']],color=color,s=55,zorder=4)
    ax.plot([0,1],[0,1],':',color='#777777',linewidth=1)
    ax.set(xlim=(0,1),ylim=(0,1.02),xlabel='Clean false-positive rate (992 inputs)',ylabel='Trigger recall (416 inputs)',title='UniXcoder detector ROC')
    ax.plot([],[],marker='o',linestyle='',color='#555555',label='Validation-calibrated cutoffs')
    ax.legend(loc='lower right',frameon=False,fontsize=9); ax.grid(alpha=.15)
    fig.savefig(args.run/'detector-roc.png',dpi=180); plt.close(fig)

if __name__=='__main__': main()
