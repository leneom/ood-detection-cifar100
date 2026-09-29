"""Render the saved class diagnostic arrays with an unobstructed legend."""
from pathlib import Path
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'report/tinyimagenet-class-heterogeneity'
with np.load(OUT/'class_values.npz',allow_pickle=False) as z:
    auroc=z['auroc']
    accepted=z['accepted']
labels=('MSP','Energy','MaxLogit','Entropy','KNN-50','Pooled Mahalanobis')

def render(values,title,xlabel,stem):
    fig,ax=plt.subplots(figsize=(9.5,5.5),constrained_layout=True)
    for vi,(variant,color,offset) in enumerate((('T1','#247ba0',-.14),
                                                  ('calibrated','#dc5f3b',.14))):
        stats=[]
        for mi,label in enumerate(labels):
            q=np.quantile(values[vi,mi]*100,[0,.25,.5,.75,1],method='linear')
            stats.append({'label':label,'whislo':q[0],'q1':q[1],'med':q[2],
                          'q3':q[3],'whishi':q[4],'fliers':[]})
        ax.bxp(stats,positions=np.arange(6)+offset,vert=False,widths=.23,
               showfliers=False,patch_artist=True,
               boxprops={'facecolor':color,'edgecolor':color,'alpha':.65},
               medianprops={'color':'black','linewidth':1.25},
               whiskerprops={'color':color},capprops={'color':color})
        ax.plot([],[],color=color,linewidth=7,alpha=.65,label=variant)
    ax.set_yticks(np.arange(6),labels)
    ax.invert_yaxis()
    ax.set_xlim(0,100)
    ax.set_xlabel(xlabel)
    ax.set_title(title)
    ax.grid(axis='x',alpha=.25)
    ax.legend(loc='upper left',frameon=False)
    fig.savefig(OUT/f'{stem}.png',dpi=180)
    fig.savefig(OUT/f'{stem}.svg')
    plt.close(fig)

render(auroc.mean(axis=0),
       'Per-class AUROC, 200 Tiny ImageNet synsets (3-seed class means)',
       'Class AUROC (%)','class-auroc')
render(accepted[:,:,:,2].mean(axis=0)/50,
       'Per-class OOD acceptance at frozen 95% ID target (3-seed class means)',
       'Class OOD acceptance (%)','class-fixed95-acceptance')
print(json.dumps({'state':'rerendered_saved_arrays','legend':'upper left'}))
