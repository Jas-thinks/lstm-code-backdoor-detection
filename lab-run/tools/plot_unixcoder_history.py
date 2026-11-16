"""Render the recorded clean-training/validation history for presentation evidence."""
import argparse
import json
import math
from pathlib import Path


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--history',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    args=p.parse_args()
    rows=json.loads(args.history.read_text())
    if not rows or len({r['epoch'] for r in rows})!=len(rows): raise ValueError('Empty or duplicate epochs')
    if any(not math.isfinite(r[key]) for r in rows for key in ('train_nll','validation_bleu4')):
        raise ValueError('Non-finite metrics')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    epochs=[r['epoch'] for r in rows]
    selected=max(rows,key=lambda r:r['validation_bleu4'])
    fig,axes=plt.subplots(1,2,figsize=(10,3.7),layout='constrained')
    axes[0].plot(epochs,[r['train_nll'] for r in rows],'-o',color='#244d79',linewidth=2)
    axes[0].set(title='Training loss',ylabel='Mean token negative log-likelihood')
    axes[1].plot(epochs,[r['validation_bleu4'] for r in rows],'-o',color='#244d79',linewidth=2)
    axes[1].scatter([selected['epoch']],[selected['validation_bleu4']],color='#26834a',s=100,zorder=3,label='Selected checkpoint')
    axes[1].set(title='Clean validation quality',ylabel='Smoothed BLEU-4 (0–100)')
    axes[1].legend(frameon=False)
    for axis in axes:
        axis.set(xlabel='Epoch',xticks=epochs)
        axis.grid(axis='y',alpha=.2)
        axis.spines[['top','right']].set_visible(False)
    fig.suptitle('UniXcoder: seed 42; checkpoint selected using validation only',fontsize=12)
    fig.savefig(args.out,dpi=180)
    plt.close(fig)

if __name__=='__main__': main()
