"""Weighted classification metrics and paired cluster bootstrap."""
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

NAMES = ['accuracy','precision','recall','FPR','F1','macro_F1','AUROC']

def metrics(y, score, threshold=.5, weights=None):
    y,score=np.asarray(y),np.asarray(score)
    w=np.ones(len(y)) if weights is None else np.asarray(weights,dtype=float)
    if len(y)!=len(score) or len(y)!=len(w) or not np.isfinite(score).all() or not set(y)<= {0,1} or (w<0).any():
        raise ValueError('Invalid predictions or weights')
    pred=score>=threshold
    tn=w[(y==0)&~pred].sum(); fp=w[(y==0)&pred].sum()
    fn=w[(y==1)&~pred].sum(); tp=w[(y==1)&pred].sum()
    div=lambda a,b: float(a/b) if b else float('nan')
    # Undefined precision/F1 (no positive predictions) is reported as zero.
    precision=float(tp/(tp+fp)) if tp+fp else 0.
    f1=float(2*tp/(2*tp+fp+fn)) if 2*tp+fp+fn else 0.
    f0=float(2*tn/(2*tn+fp+fn)) if 2*tn+fp+fn else 0.
    both=(tp+fn)>0 and (tn+fp)>0
    return dict(accuracy=div(tp+tn,w.sum()),precision=precision,recall=div(tp,tp+fn),
                FPR=div(fp,fp+tn),F1=f1,macro_F1=(f1+f0)/2 if both else float('nan'),
                AUROC=float(roc_auc_score(y,score,sample_weight=w)) if both else float('nan'),
                TN=float(tn),FP=float(fp),FN=float(fn),TP=float(tp))

def group_weights(groups, equal=False):
    _,inverse,counts=np.unique(np.asarray(groups),return_inverse=True,return_counts=True)
    return inverse, 1/counts[inverse] if equal else np.ones(len(inverse))

def bootstrap_main(frame, thresholds, repeats=1000, seed=42):
    rows,differences=[],[]
    inv,_=group_weights(frame.group_id)
    n=int(inv.max()+1)
    draws=np.random.default_rng(seed).multinomial(n,np.ones(n)/n,size=repeats)
    for mode in ['per_text','equal_group_weight']:
        _,base=group_weights(frame.group_id,mode=='equal_group_weight')
        for setting in ['default','low_FPR']:
            sampled={}
            for key in ['B','C']:
                threshold=thresholds[key][setting]
                point=metrics(frame.label,frame[key+'_score'],threshold,base)
                estimates=[metrics(frame.label,frame[key+'_score'],threshold,base*d[inv]) for d in draws]
                sample=np.array([[result[name] for name in NAMES] for result in estimates])
                sampled[key]=sample
                for j,name in enumerate(NAMES):
                    values=sample[:,j]; finite=values[np.isfinite(values)]
                    lo,hi=np.quantile(finite,[.025,.975]) if len(finite) else [np.nan,np.nan]
                    rows.append(dict(detector=key,setting=setting,threshold=threshold,weighting=mode,metric=name,
                                     value=point[name],ci_low=lo,ci_high=hi,valid_bootstraps=len(finite),
                                     texts=len(frame),groups=n,human=int((frame.label==0).sum()),AI=int((frame.label==1).sum())))
            for j,name in enumerate(NAMES):
                values=sampled['B'][:,j]-sampled['C'][:,j]; values=values[np.isfinite(values)]
                point=metrics(frame.label,frame.B_score,thresholds['B'][setting],base)[name]-metrics(frame.label,frame.C_score,thresholds['C'][setting],base)[name]
                lo,hi=np.quantile(values,[.025,.975]) if len(values) else [np.nan,np.nan]
                differences.append(dict(comparison='B_minus_C',setting=setting,weighting=mode,metric=name,value=point,ci_low=lo,ci_high=hi,valid_bootstraps=len(values)))
    return pd.DataFrame(rows),pd.DataFrame(differences)

def paired_edit_summary(pairs,thresholds,repeats=1000,seed=42):
    rows=[]
    for condition,f in pairs.groupby('condition',sort=True):
        inv,_=group_weights(f.group_id); n=int(inv.max()+1)
        draws=np.random.default_rng(seed).multinomial(n,np.ones(n)/n,size=repeats)
        for weighting in ['per_text','equal_group_weight']:
            _,base=group_weights(f.group_id,weighting=='equal_group_weight')
            w=draws[:,inv]*base; den=w.sum(axis=1)
            for setting in ['default','low_FPR']:
                for key in ['B','C']:
                    a=f[key+'_original_score'].to_numpy(); b=f[key+'_edited_score'].to_numpy()
                    before=a>=thresholds[key][setting]; after=b>=thresholds[key][setting]
                    values={'original_TPR':before,'edited_TPR':after,'miss_rate':~after,
                            'TPR_change':after.astype(float)-before,'mean_score_change':b-a,
                            'AI_to_human_flip':before&~after,'human_to_AI_flip':~before&after}
                    for name,v in values.items():
                        estimate=float(np.average(v,weights=base)); sample=w@v/den
                        lo,hi=np.quantile(sample,[.025,.975])
                        rows.append(dict(condition=condition,detector=key,setting=setting,weighting=weighting,metric=name,
                                         value=estimate,ci_low=lo,ci_high=hi,pairs=len(f),groups=n,parents=f.parent_id.nunique()))
    return pd.DataFrame(rows)
