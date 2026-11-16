"""Expanded candidates selected using the frozen development partitions."""
import argparse
import copy
from datetime import datetime,timezone
import itertools
import json
from pathlib import Path
import time
import warnings
import joblib
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline,FeatureUnion
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler,Normalizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import ExtraTreesClassifier,HistGradientBoostingClassifier
from sklearn.exceptions import ConvergenceWarning
from src.features.style_features import StyleTransformer
from src.common import load_development_data,metrics,sha256
from scripts.freeze_module_e import select_threshold

ROOT=Path(__file__).resolve().parents[2]

def representation(name,cfg):
    char=TfidfVectorizer(analyzer='char',ngram_range=tuple(cfg['char_ngram_range']),min_df=cfg['min_df'],max_features=cfg['char_max_features'],sublinear_tf=True)
    if name=='char':return char
    word=TfidfVectorizer(ngram_range=tuple(cfg['word_ngram_range']),min_df=cfg['min_df'],max_features=cfg['word_max_features'],sublinear_tf=True)
    steps=[('word',word),('char',char)]
    if name=='word_char_style':
        steps.append(('style',Pipeline([('extract',StyleTransformer('full')),('scale',StandardScaler()),('norm',Normalizer())])))
    return FeatureUnion(steps)

def train(output,root=ROOT):
    cfg=json.loads((root/'configs/extension_x.json').read_text())
    train,val,hashes=load_development_data(root)
    if output.exists():raise FileExistsError('Choose a new output directory')
    output.mkdir(parents=True)
    rows=[];best=None;best_score=-1;best_scores=None
    def consider(model,scores,description,seconds=0):
        nonlocal best,best_score,best_scores
        row={'candidate_id':len(rows),**description,**metrics(val.label,scores),'seconds':seconds}
        rows.append(row)
        if row['macro_f1']>best_score:
            best_score=row['macro_f1'];best=copy.deepcopy(model);best_scores=scores.copy()
        print(f"{row['candidate_id']:02d} {row['family']} val F1={row['macro_f1']:.4f} accuracy={row['accuracy']:.4f}",flush=True)
    for key in ['B','C']:
        model=joblib.load(root/f'resources/detectors/{key}/pipeline.joblib')
        consider(model,model.predict_proba(val.text.tolist())[:,list(model.classes_).index(1)],{'family':'baseline_'+key})
    for name in cfg['representations']:
        transformer=representation(name,cfg);a=transformer.fit_transform(train.text,train.label);b=transformer.transform(val.text)
        for c,weight in itertools.product(cfg['C'],cfg['class_weight']):
            t=time.perf_counter();clf=LogisticRegression(C=c,class_weight=weight,solver='liblinear',max_iter=3000,random_state=cfg['seed'])
            with warnings.catch_warnings():
                warnings.simplefilter('error',ConvergenceWarning);clf.fit(a,train.label)
            model=Pipeline([('features',transformer),('classifier',clf)])
            consider(model,clf.predict_proba(b)[:,1],{'family':name+'_LR','C':c,'class_weight':weight},time.perf_counter()-t)
    transformer=StyleTransformer('full');a=transformer.fit_transform(train.text,train.label);b=transformer.transform(val.text)
    for leaf,max_features in itertools.product(cfg['extra_trees_min_samples_leaf'],cfg['extra_trees_max_features']):
        t=time.perf_counter();clf=ExtraTreesClassifier(n_estimators=400,min_samples_leaf=leaf,max_features=max_features,class_weight='balanced',random_state=cfg['seed'],n_jobs=2).fit(a,train.label)
        consider(Pipeline([('features',transformer),('classifier',clf)]),clf.predict_proba(b)[:,1],{'family':'style_ExtraTrees','leaf':leaf,'max_features':max_features},time.perf_counter()-t)
    for leaf,l2 in itertools.product(cfg['boosting_max_leaf_nodes'],cfg['boosting_l2_regularization']):
        t=time.perf_counter();clf=HistGradientBoostingClassifier(max_iter=200,max_leaf_nodes=leaf,l2_regularization=l2,learning_rate=.05,early_stopping=False,random_state=cfg['seed']).fit(a,train.label)
        consider(Pipeline([('features',transformer),('classifier',clf)]),clf.predict_proba(b)[:,1],{'family':'style_HistGradientBoosting','max_leaf_nodes':leaf,'l2':l2},time.perf_counter()-t)
    pd.DataFrame(rows).to_csv(output/'validation_search.csv',index=False)
    chosen=max(rows,key=lambda x:(x['macro_f1'],-x['candidate_id']))
    joblib.dump(best,output/'pipeline.joblib',compress=3)
    np.testing.assert_allclose(joblib.load(output/'pipeline.joblib').predict_proba(val.text.tolist())[:,1],best_scores,atol=1e-12,rtol=0)
    pred=val[['sample_id','group_id','label']].copy();pred['ai_score']=best_scores;pred.to_csv(output/'validation_predictions.csv',index=False)
    frozen={'status':'validation_selected_before_extension_test','created_utc':datetime.now(timezone.utc).isoformat(),
            'selected':chosen,'candidates':len(rows),'default_threshold':.5,'low_FPR':select_threshold(val.label.to_numpy(),best_scores),
            'pipeline_sha256':sha256(output/'pipeline.joblib'),'config_sha256':sha256(root/'configs/extension_x.json'),
            'source_sha256':sha256(Path(__file__)),'input_sha256':hashes,'test_status':cfg['test_status']}
    (output/'selection_frozen.json').write_text(json.dumps(frozen,indent=2),encoding='utf-8')
    print(json.dumps(frozen,indent=2))

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--output',type=Path,default=ROOT/'artifacts/extension_x');train(ap.parse_args().output)
