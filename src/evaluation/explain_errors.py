"""Exact additive score decompositions for fixed error examples."""
import argparse
import html
import json
from pathlib import Path
import re
import joblib
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[2]

def forest_contributions(forest, values):
    """Average child-minus-parent class proportions along actual tree paths."""
    column=list(forest.classes_).index(1)
    baseline=0.;contrib=np.zeros(values.shape[1])
    for estimator in forest.estimators_:
        tree=estimator.tree_
        p=tree.value[:,0,column]/tree.value[:,0,:].sum(axis=1)
        baseline+=p[0]
        path=estimator.decision_path(values).indices
        for parent,child in zip(path[:-1],path[1:]):
            contrib[tree.feature[parent]]+=p[child]-p[parent]
    return baseline/len(forest.estimators_),contrib/len(forest.estimators_)

def explain(out):
    f=pd.read_csv(out/'error_cases.csv');models={k:joblib.load(ROOT/f'resources/detectors/{k}/pipeline.joblib') for k in ['B','C']}
    records=[];sections=[]
    for row in f.itertuples():
        model=models[row.detector];x=[row.text]
        for _,transformer in model.steps[:-1]: x=transformer.transform(x)
        if row.detector=='B':
            features=model[0].get_feature_names_out();v=x.toarray()[0]*model[-1].coef_[0];base=float(model[-1].intercept_[0]);unit='log_odds'
            score=float(1/(1+np.exp(-(base+v.sum()))))
        else:
            features=model[0].get_feature_names_out();base,v=forest_contributions(model[-1],x);unit='AI_score';score=float(base+v.sum())
        actual=getattr(row,row.detector+'_score')
        if not np.isclose(score,actual,atol=1e-10): raise ValueError('Score decomposition mismatch')
        def top(direction):
            ix=np.flatnonzero(v>0) if direction=='positive' else np.flatnonzero(v<0)
            ix=sorted(ix,key=lambda i:abs(v[i]),reverse=True)[:5]
            return [{'feature':str(features[i]),'contribution':float(v[i]),'value':float(x[0,i])} for i in ix]
        tags=[]
        if re.search(r'^\s*(?:[-*•]|\d+[.)])\s+',row.text,re.M):tags.append('list_form')
        if re.search(r'\b(?:I|my|me|we|our)\b',row.text,re.I):tags.append('first_person')
        if '?' in row.text:tags.append('question_mark')
        r={'sample_id':row.sample_id,'detector':row.detector,'error_type':row.error_type,'category':row.category,
           'word_count':int(row.word_count),'surface_tags':tags,'score':actual,'baseline':base,'contribution_units':unit,
           'positive_contributions':top('positive'),'negative_contributions':top('negative'),'reconstructed_score':score,
           'edit_magnitude':'not_applicable_raw_main_test','fact_changes':'not_assessed_raw_main_test'}
        records.append(r)
        pos='; '.join(f'{a["feature"]} ({a["contribution"]:+.4f})' for a in r['positive_contributions'])
        neg='; '.join(f'{a["feature"]} ({a["contribution"]:+.4f})' for a in r['negative_contributions'])
        sections.append(f'<section><h2>{row.detector} {row.error_type}: {row.sample_id}</h2><p>Category: {html.escape(row.category)}; words: {row.word_count}; AI score: {actual:.4f}</p><p>Positive {unit} contributions: {html.escape(pos)}</p><p>Negative {unit} contributions: {html.escape(neg)}</p><pre>{html.escape(row.text)}</pre></section>')
    with (out/'error_explanations.jsonl').open('w',encoding='utf-8') as stream:
        for r in records:stream.write(json.dumps(r,ensure_ascii=False)+'\n')
    (out/'error_analysis.html').write_text('<!doctype html><meta charset="utf-8"><title>Error analysis</title><style>body{max-width:1100px;margin:30px auto;font:12pt "Times New Roman",SimSun}pre{font:inherit;white-space:pre-wrap;line-height:1.5}section{border-bottom:1px solid #aaa;padding:20px}</style><h1>Fixed test errors and model contributions</h1><p>B: TF-IDF times logistic-regression coefficient, in log-odds. C: average class-proportion change along decision-tree paths, in score units. All contributions plus baseline reconstruct the saved model score.</p>'+''.join(sections),encoding='utf-8')
    f.groupby(['detector','error_type','category']).agg(examples=('sample_id','size'),mean_words=('word_count','mean')).to_csv(out/'error_category_counts.csv')
    print(f'Explained {len(records)} fixed errors; all reconstructed scores agree within 1e-10.')

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--analysis',type=Path,default=ROOT/'artifacts/module_e')
    explain(ap.parse_args().analysis)
