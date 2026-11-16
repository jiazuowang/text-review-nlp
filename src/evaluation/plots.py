"""Six static figures computed from frozen test scores."""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix,roc_curve,roc_auc_score

def create_plots(predictions,pairs,out):
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    main=predictions.loc[predictions.main_task]
    def save(fig,name):
        fig.tight_layout();fig.savefig(out/name,dpi=180);plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(9,4))
    for ax,key in zip(axes,['B','C']):
        cm=confusion_matrix(main.label,main[key+'_score']>=.5,labels=[0,1])
        ax.imshow(cm,cmap='Blues');ax.set(xticks=[0,1],yticks=[0,1],xticklabels=['Human','AI'],yticklabels=['Human','AI'],xlabel='Predicted',ylabel='True',title=f'{key}: threshold 0.5, n={len(main)}')
        for (i,j),v in np.ndenumerate(cm): ax.text(j,i,str(v),ha='center',va='center',color='white' if v>cm.max()/2 else 'black',fontsize=16)
    save(fig,'01_confusion_matrices.png')
    edited=pd.read_csv(out/'edited_metrics.csv');e=edited.loc[(edited.weighting=='per_text')&(edited.setting=='default')&(edited.metric=='edited_TPR')]
    fig,ax=plt.subplots(figsize=(10,5));conditions=sorted(e.condition.unique());x=np.arange(len(conditions))
    for key,offset,color in [('B',-.17,'#245a81'),('C',.17,'#c27036')]:
        f=e.loc[e.detector==key].set_index('condition').loc[conditions]
        ax.bar(x+offset,f.value,.32,label=key,color=color,yerr=[f.value-f.ci_low,f.ci_high-f.value],capsize=3)
    counts=e.loc[e.detector=='B'].set_index('condition').pairs
    ax.set(xticks=x,xticklabels=[c.replace('llama-3.1-70b','Llama').replace('gpt-4o','GPT')+f' (n={counts[c]})' for c in conditions],ylim=(0,1),ylabel='AI detection rate',title='Eligible edited test texts: 95% group-bootstrap intervals')
    ax.tick_params(axis='x',rotation=25);ax.legend();save(fig,'02_edited_TPR.png')
    table=pd.read_csv(out/'metrics_summary.csv');t=table.loc[table.weighting=='per_text']
    fig,ax=plt.subplots(figsize=(7,5))
    for key,color in [('B','#245a81'),('C','#c27036')]:
        for setting,marker in [('default','o'),('low_FPR','s')]:
            f=t.loc[(t.detector==key)&(t.setting==setting)].set_index('metric')
            a,b=f.loc['FPR'],f.loc['recall']
            ax.errorbar(a.value,b.value,xerr=[[a.value-a.ci_low],[a.ci_high-a.value]],yerr=[[b.value-b.ci_low],[b.ci_high-b.value]],fmt=marker,color=color,capsize=3,label=f'{key} {setting}')
    ax.set(xlim=(0,.55),ylim=(0,1),xlabel='Human false-positive rate',ylabel='AI detection rate',title=f'Test operating points (n={len(main)})');ax.legend();save(fig,'03_threshold_tradeoff.png')
    fig,axes=plt.subplots(1,2,figsize=(13,4),sharex=True,sharey=True)
    for ax,key in zip(axes,['B','C']):
        for condition,f in pairs.groupby('condition'):
            ax.scatter(f.normalized_word_edit_distance,f[key+'_edited_score']-f[key+'_original_score'],s=7,alpha=.3,label=condition)
        ax.axhline(0,color='gray',lw=.8);ax.set(xlabel='Normalized word edit distance',title=f'{key}: {len(pairs)} pairs')
    axes[1].legend(loc='upper left',bbox_to_anchor=(1,1),fontsize=8,markerscale=2)
    axes[0].set_ylabel('Edited score - original score');save(fig,'04_paired_score_change.png')
    slices=pd.read_csv(out/'slice_metrics.csv');f=slices.loc[(slices.slice_by=='length_bin')&(slices.setting=='default')]
    fig,axes=plt.subplots(1,2,figsize=(10,4))
    for ax,metric in zip(axes,['FPR','recall']):
        for key in ['B','C']:
            part=f.loc[f.detector==key];ax.plot(part.slice_value,part[metric],marker='o',label=key)
        counts=f.loc[f.detector=='B']
        ax.set(xticks=range(len(counts)),xticklabels=[f'{r.slice_value}\nH={r.human}, AI={r.AI}' for r in counts.itertuples()],ylim=(0,1),xlabel='English word count',ylabel=metric,title='Test length slices');ax.legend()
    save(fig,'05_length_slices.png')
    fig,ax=plt.subplots(figsize=(6,5))
    for key in ['B','C']:
        x,y,_=roc_curve(main.label,main[key+'_score']);ax.plot(x,y,label=f'{key}: AUROC {roc_auc_score(main.label,main[key+"_score"]):.3f}')
    ax.plot([0,1],[0,1],'--',color='gray');ax.set(xlabel='False-positive rate',ylabel='True-positive rate',title=f'Frozen test ROC (n={len(main)})');ax.legend();save(fig,'06_ROC.png')
