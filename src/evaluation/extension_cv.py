"""Supplementary grouped cross-validation on the original training partition."""
import argparse
import json
from pathlib import Path
import time
import joblib
import pandas as pd
from sklearn.base import clone
from sklearn.model_selection import StratifiedGroupKFold
from src.common import load_development_data,metrics

ROOT=Path(__file__).resolve().parents[2]

def run(model_dir,output,root=ROOT):
    if output.exists():raise FileExistsError('Use a fresh output directory')
    train,_,_=load_development_data(root);output.mkdir(parents=True)
    models={'B':joblib.load(root/'resources/detectors/B/pipeline.joblib'),'X':joblib.load(model_dir/'pipeline.joblib')}
    rows=[];predictions=[]
    for fold,(tr,va) in enumerate(StratifiedGroupKFold(n_splits=5,shuffle=True,random_state=42).split(train.text,train.label,train.group_id)):
        assert not set(train.iloc[tr].group_id)&set(train.iloc[va].group_id)
        for key,template in models.items():
            tick=time.perf_counter();model=clone(template);model.fit(train.iloc[tr].text.tolist(),train.iloc[tr].label)
            scores=model.predict_proba(train.iloc[va].text.tolist())[:,list(model.classes_).index(1)]
            rows.append(dict(fold=fold,detector=key,texts=len(va),groups=train.iloc[va].group_id.nunique(),seconds=time.perf_counter()-tick,**metrics(train.iloc[va].label,scores)))
            f=train.iloc[va][['sample_id','group_id','label']].copy();f['fold']=fold;f['detector']=key;f['ai_score']=scores;predictions.append(f)
            print(f'Fold {fold} {key}: macro F1={rows[-1]["macro_f1"]:.4f}',flush=True)
    pd.DataFrame(rows).to_csv(output/'fold_metrics.csv',index=False);pd.concat(predictions).to_csv(output/'out_of_fold_predictions.csv',index=False)
    summary=pd.DataFrame(rows).groupby('detector')[['accuracy','macro_f1','human_false_positive_rate','ai_recall','roc_auc']].agg(['mean','std'])
    summary.to_csv(output/'summary.csv')
    (output/'protocol.json').write_text(json.dumps({'folds':5,'seed':42,'partition':'train','group':'group_id','role':'Supplementary stability at hyperparameters already selected on validation; model choice remains frozen.'},indent=2),encoding='utf-8')
    print(summary.to_string())

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--model-dir',type=Path,default=ROOT/'artifacts/extension_x');ap.add_argument('--output',type=Path,default=ROOT/'artifacts/extension_x_cv');a=ap.parse_args();run(a.model_dir,a.output)
