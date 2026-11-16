"""Follow-up test evaluation and synthetic local-source diagnostics."""
import argparse
import hashlib
import json
from pathlib import Path
import time
import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from src.common import sha256,load_development_data
from src.evaluation.run import load_inputs,build_pairs,benchmark
from src.evaluation.metrics import metrics,bootstrap_main,paired_edit_summary
from src.evaluation.local_analysis import analyze,render
from src.features.style_features import WORD

ROOT=Path(__file__).resolve().parents[2]

def stable(value):return hashlib.sha256(('42|'+value).encode()).hexdigest()

def mixed_diagnostic(val,model,out):
    rows=[];documents=[];column=list(model.classes_).index(1)
    grouped=[]
    for gid,f in val.groupby('group_id'):
        h=f.loc[f.label.eq(0)&f.word_count.ge(120)].sort_values('sample_id')
        a=f.loc[f.label.eq(1)&f.word_count.ge(120)].sort_values('sample_id')
        if len(h) and len(a):grouped.append((gid,h.iloc[0],a.iloc[0]))
    grouped=sorted(grouped,key=lambda row:stable(row[0]))[:30]
    for i,(gid,h,a) in enumerate(grouped):
        chunks={}
        for label,row in [(0,h),(1,a)]:
            words=list(WORD.finditer(row.text));chunks[label]=[row.text[words[s].start():words[s+59].end()] for s in [0,60]]
        labels=[0,1,0,1] if i%2==0 else [1,0,1,0]
        blocks=[chunks[label][j//2] for j,label in enumerate(labels)]
        text='\n\n'.join(blocks);whole=float(model.predict_proba([text])[0,column])
        block_scores=model.predict_proba(blocks)[:,column]
        removed=['\n\n'.join(blocks[:j]+blocks[j+1:]) for j in range(4)]
        impact=whole-model.predict_proba(removed)[:,column]
        for j,label in enumerate(labels):
            rows.append(dict(group_id=gid,block=j,label=label,source_id=h.sample_id if label==0 else a.sample_id,
                             text=blocks[j],window_score=float(block_scores[j]),deletion_impact=float(impact[j])))
        documents.append(dict(group_id=gid,text=text,labels=labels,source_ids=[h.sample_id,a.sample_id]))
    frame=pd.DataFrame(rows);frame.to_csv(out/'mixed_blocks.csv',index=False,encoding='utf-8-sig')
    f=frame.rename(columns={'window_score':'B_score','deletion_impact':'C_score'})
    thresholds={'B':{'default':.5,'low_FPR':.5},'C':{'default':0.,'low_FPR':0.}}
    summary,_=bootstrap_main(f,thresholds,1000,42)
    summary=summary.loc[summary.setting.eq('default')];summary.detector=summary.detector.map({'B':'60_word_score','C':'block_deletion_impact'})
    summary.to_csv(out/'localization_metrics.csv',index=False)
    result={'documents':len(documents),'blocks':len(frame),'words_per_block':60,'split':'validation',
            'construction':'First 120 words of same-group human and AI originals; alternating 60-word blocks; order alternates by document.',
            'scope':'Artificial concatenation diagnostic on the validation partition used for model selection. Authentic mixed-authorship sentence labels remain unavailable.',
            'window_metrics':metrics(frame.label,frame.window_score,.5),
            'deletion_impact_AUROC':float(roc_auc_score(frame.label,frame.deletion_impact)),
            'human_false_positive_example_count':int(((frame.label==0)&(frame.window_score>=.5)).sum())}
    (out/'localization_validation.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    if documents:
        first=documents[0];local=analyze(first['text'],model)
        (out/'local_demo.txt').write_text(first['text'],encoding='utf-8')
        (out/'local_demo_sources.json').write_text(json.dumps(first,ensure_ascii=False,indent=2),encoding='utf-8')
        (out/'local_demo.json').write_text(json.dumps(local,ensure_ascii=False,indent=2),encoding='utf-8')
        render(first['text'],local,out/'local_demo.html',title='Validation mixed text passage impact example')
    return result

def evaluate(model_dir,output,root=ROOT):
    if output.exists():raise FileExistsError('Choose a new evaluation output')
    frozen=json.loads((model_dir/'selection_frozen.json').read_text())
    if sha256(model_dir/'pipeline.joblib')!=frozen['pipeline_sha256']:raise ValueError('Model changed after selection')
    if sha256(root/'configs/extension_x.json')!=frozen['config_sha256']:raise ValueError('Config changed')
    f,protocol,provenance=load_inputs(root)
    pairs,audit=build_pairs(f);ids=set(f.loc[f.main_task,'sample_id'])|set(pairs.sample_id)|set(pairs.parent_id)
    pred=f.loc[f.sample_id.isin(ids)].sort_values('sample_id').copy();output.mkdir(parents=True)
    t=time.perf_counter();model=joblib.load(model_dir/'pipeline.joblib');load_seconds=time.perf_counter()-t
    t=time.perf_counter();pred['X_score']=model.predict_proba(pred.text.tolist())[:,list(model.classes_).index(1)];inference_seconds=time.perf_counter()-t
    baseline=pd.read_csv(root/'artifacts/module_e/predictions.csv').set_index('sample_id')
    for key in ['B','C']:pred[key+'_score']=pred.sample_id.map(baseline[key+'_score'])
    if pred[['X_score','B_score','C_score']].isna().any().any():raise ValueError('Prediction ID mismatch')
    pred.drop(columns='text').to_csv(output/'predictions.csv',index=False)
    main=pred.loc[pred.main_task].copy()
    thresholds={'B':{'default':.5,'low_FPR':frozen['low_FPR']['threshold']},'C':{'default':.5,'low_FPR':protocol['low_FPR_thresholds']['B']['threshold']}}
    compare=main.drop(columns=['B_score','C_score']).assign(B_score=main.X_score,C_score=main.B_score)
    table,diff=bootstrap_main(compare,thresholds,1000,42)
    table.detector=table.detector.map({'B':'X','C':'B'});diff.comparison='X_minus_B'
    table.to_csv(output/'metrics_summary.csv',index=False);diff.to_csv(output/'model_differences.csv',index=False)
    for key,column in [('B','X_score'),('C','B_score')]:
        indexed=pred.set_index('sample_id')[column]
        pairs[key+'_original_score']=pairs.parent_id.map(indexed);pairs[key+'_edited_score']=pairs.sample_id.map(indexed)
    edits=paired_edit_summary(pairs,thresholds,1000,42);edits.detector=edits.detector.map({'B':'X','C':'B'})
    edits.to_csv(output/'edited_metrics.csv',index=False)
    pairs=pairs.rename(columns={c:c.replace('B_','X_').replace('C_','B_') for c in pairs if c.startswith(('B_','C_'))})
    pairs.to_csv(output/'paired_scores.csv',index=False)
    audit.to_csv(output/'pair_audit.csv',index=False)
    order=main.assign(order=main.sample_id.map(stable)).sort_values('order')
    runtime={'load_seconds':load_seconds,'full_batch_seconds':inference_seconds,'full_batch_texts':len(pred),**benchmark(model,order.text)}
    (output/'runtime.json').write_text(json.dumps(runtime,indent=2),encoding='utf-8')
    _,val,_=load_development_data(root);diagnostic=mixed_diagnostic(val,model,output)
    result={'status':'follow_up_evaluation_complete','test_role':frozen['test_status'],'selected':frozen['selected'],
            'selection_frozen_sha256':sha256(model_dir/'selection_frozen.json'),'pipeline_sha256':frozen['pipeline_sha256'],
            'test_texts':len(main),'edited_pairs':len(pairs),'local_diagnostic':diagnostic,
            'default_test':metrics(main.label,main.X_score,.5),'low_FPR_test':metrics(main.label,main.X_score,frozen['low_FPR']['threshold']),
            'baseline_prediction_sha256':sha256(root/'artifacts/module_e/predictions.csv'),
            'artifact_sha256':{p.name:sha256(p) for p in output.iterdir() if p.is_file()}}
    (output/'run.json').write_text(json.dumps(result,indent=2),encoding='utf-8');print(json.dumps(result,indent=2))

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--model-dir',type=Path,default=ROOT/'artifacts/extension_x');ap.add_argument('--output',type=Path,default=ROOT/'artifacts/extension_x_evaluation')
    a=ap.parse_args();evaluate(a.model_dir,a.output)
