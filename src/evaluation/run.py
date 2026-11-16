"""Evaluate frozen detectors on the exported test partition; never fit models."""
import argparse
from datetime import datetime,timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import time
import html
import joblib
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_info
from src.edits.analyze import quality
from src.evaluation.metrics import metrics,bootstrap_main,paired_edit_summary

ROOT=Path(__file__).resolve().parents[2]
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def stable_id(x): return hashlib.sha256(('42|'+x).encode()).hexdigest()

def load_inputs(root=ROOT):
    protocol=json.loads((root/'configs/module_e_frozen.json').read_text())
    folder=root/'data/module_e'; provenance=json.loads((folder/'provenance.json').read_text())
    if sha(root/'configs/module_e_frozen.json')!=provenance['protocol_sha256']: raise ValueError('Protocol changed')
    for name,digest in provenance['export_files'].items():
        if sha(folder/name)!=digest: raise ValueError('Export changed: '+name)
    f=pd.read_parquet(folder/'test_texts.parquet'); m=pd.read_csv(folder/'split_manifest.csv')
    if not f.sample_id.is_unique or not f.split.eq('test').all(): raise ValueError('Invalid test membership')
    test=m.loc[m.split.eq('test')].set_index('sample_id')
    if set(f.sample_id)!=set(test.index): raise ValueError('Test IDs changed')
    for col in ['group_id','split','variant','label','text_sha256','eligible','pair_eligible','main_task']:
        if not (f[col].to_numpy()==test.loc[f.sample_id,col].to_numpy()).all(): raise ValueError('Metadata changed: '+col)
    if m.groupby('group_id').split.nunique().max()!=1 or m.groupby('normalized_sha256').split.nunique().max()!=1:
        raise ValueError('Cross-partition leakage')
    for row in f.itertuples():
        if hashlib.sha256(row.text.encode()).hexdigest()!=row.text_sha256: raise ValueError('Text changed')
    return f,protocol,provenance

def build_pairs(f):
    byid=f.set_index('sample_id'); pairs=[]; audit=[]
    for edit in f.loc[f.variant.isin(['human_edited','llm_edited'])].itertuples():
        parent=byid.loc[edit.parent_id]
        if parent.split!=edit.split or parent.group_id!=edit.group_id or parent.variant!='ai_raw' or edit.label!=1:
            raise ValueError('Invalid pair')
        include=bool(parent.eligible and edit.eligible)
        if include!=bool(edit.pair_eligible): raise ValueError('Pair inclusion changed')
        condition='expert' if edit.variant=='human_edited' else f'{edit.editor}:{edit.edit_prompt_id}'
        audit.append(dict(sample_id=edit.sample_id,parent_id=edit.parent_id,condition=condition,included=include,
                          original_exclusion=parent.exclusion_reason,edited_exclusion=edit.exclusion_reason))
        if include:
            pairs.append(dict(sample_id=edit.sample_id,parent_id=edit.parent_id,group_id=edit.group_id,condition=condition,
                              **quality(parent.text,edit.text,[.5,1.5])))
    return pd.DataFrame(pairs),pd.DataFrame(audit)

def benchmark(pipeline,texts):
    # Stable first 30 hashed main-test IDs; three timed passes after warm-up.
    texts=list(texts)[:30]
    pipeline.predict_proba(texts[:2])
    timings=[]
    for _ in range(3):
        for text in texts:
            t=time.perf_counter();pipeline.predict_proba([text]);timings.append(1000*(time.perf_counter()-t))
    t=time.perf_counter();pipeline.predict_proba(texts);seconds=time.perf_counter()-t
    return dict(unique_texts=len(texts),single_repeats=3,single_median_ms=float(np.median(timings)),
                single_P95_ms=float(np.quantile(timings,.95)),batch_seconds=seconds,batch_texts_per_second=len(texts)/seconds,
                batch_size=len(texts),words_min=min(len(t.split()) for t in texts),words_median=float(np.median([len(t.split()) for t in texts])),
                words_max=max(len(t.split()) for t in texts),word_measure='whitespace tokens for runtime sample',
                estimator_n_jobs=getattr(pipeline[-1],'n_jobs',None))

def slices(main,thresholds):
    rows=[]; main=main.copy()
    main['length_bin']=pd.cut(main.word_count,[49,100,200,400,1000],labels=['50-100','101-200','201-400','401-1000'])
    for column in ['length_bin','category','generator']:
        source=main.loc[main.label.eq(1)] if column=='generator' else main
        for value,f in source.groupby(column,observed=True,dropna=False):
            for key in ['B','C']:
                for setting,t in thresholds[key].items():
                    rows.append(dict(slice_by=column,slice_value=str(value),detector=key,setting=setting,texts=len(f),groups=f.group_id.nunique(),
                                     human=int(f.label.eq(0).sum()),AI=int(f.label.eq(1).sum()),**metrics(f.label,f[key+'_score'],t)))
    return pd.DataFrame(rows)

def errors(main,out):
    chosen=[]
    for key in ['B','C']:
        for kind,mask in [('FP',main.label.eq(0)&main[key+'_score'].ge(.5)),('FN',main.label.eq(1)&main[key+'_score'].lt(.5))]:
            part=main.loc[mask].copy();part['selection_hash']=part.sample_id.map(stable_id)
            part=part.sort_values('selection_hash').head(10);part['detector']=key;part['error_type']=kind
            chosen.append(part)
    result=pd.concat(chosen,ignore_index=True)
    result.to_csv(out/'error_cases.csv',index=False,encoding='utf-8-sig')
    sections=[]
    for r in result.itertuples():
        sections.append(f'<section><h2>{r.detector} {r.error_type}: {r.sample_id}</h2><p>{html.escape(r.category)} | {r.word_count} words | B {r.B_score:.4f} | C {r.C_score:.4f}</p><pre>{html.escape(r.text)}</pre></section>')
    (out/'error_cases.html').write_text('<!doctype html><meta charset="utf-8"><title>Test error cases</title><style>body{max-width:1100px;margin:30px auto;font-family:Arial}pre{white-space:pre-wrap;line-height:1.5}section{border-bottom:1px solid #aaa;padding:20px}</style><h1>Deterministically selected test errors</h1>'+''.join(sections),encoding='utf-8')

def summarize(predictions,pairs,protocol,out):
    thresholds={k:{'default':.5,'low_FPR':protocol['low_FPR_thresholds'][k]['threshold']} for k in ['B','C']}
    main=predictions.loc[predictions.main_task].copy()
    table,diff=bootstrap_main(main,thresholds,protocol['bootstrap']['replicates'],protocol['bootstrap']['seed'])
    table.to_csv(out/'metrics_summary.csv',index=False);diff.to_csv(out/'model_differences.csv',index=False)
    print('Main metrics and paired bootstrap complete',flush=True)
    paired_edit_summary(pairs,thresholds,protocol['bootstrap']['replicates'],protocol['bootstrap']['seed']).to_csv(out/'edited_metrics.csv',index=False)
    common=set(pairs.groupby('parent_id').condition.nunique().loc[lambda s:s==7].index)
    if common:
        paired_edit_summary(pairs.loc[pairs.parent_id.isin(common)],thresholds,protocol['bootstrap']['replicates'],protocol['bootstrap']['seed']).to_csv(out/'edited_common_metrics.csv',index=False)
    slices(main,thresholds).to_csv(out/'slice_metrics.csv',index=False)
    errors(main,out)
    main.groupby('label').word_count.describe().to_csv(out/'main_length_distribution.csv')
    result={'main_texts':len(main),'main_groups':main.group_id.nunique(),'human':int(main.label.eq(0).sum()),'AI':int(main.label.eq(1).sum()),
            'edited_pairs':len(pairs),'common_parents_all_conditions':len(common),'thresholds':thresholds,
            'bootstrap':protocol['bootstrap'],'undefined_metrics':'Precision and F1 use zero for empty prediction denominator; FPR/recall/AUROC and macro_F1 requiring missing classes are NaN.',
            'edited_quality':'All eligible published attempts retained. Automated flags provided. Test edits have no semantic review.',
            'weighting':'per_text: weight 1; equal_group_weight: each group sums to 1 within the evaluated subset. Bootstrap resamples whole groups, identical draws for B and C.'}
    (out/'metrics.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    return result

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--output',type=Path,default=ROOT/'artifacts/module_e');args=ap.parse_args()
    if args.output.exists(): raise FileExistsError('Use a fresh output directory')
    f,protocol,provenance=load_inputs();pairs,audit=build_pairs(f)
    ids=set(f.loc[f.main_task,'sample_id'])|set(pairs.sample_id)|set(pairs.parent_id)
    selected=f.loc[f.sample_id.isin(ids)].sort_values('sample_id').copy()
    args.output.mkdir(parents=True)
    audit.to_csv(args.output/'pair_audit.csv',index=False)
    runtime={'platform':platform.platform(),'CPU':platform.processor(),'logical_CPUs':os.cpu_count(),
             'threadpools':threadpool_info(),'timing_scope':'Loaded in current Python process; feature extraction included; OS caches uncontrolled.'}
    try:
        import psutil
        runtime['RAM_bytes']=psutil.virtual_memory().total
        runtime['process_RSS_before_bytes']=psutil.Process().memory_info().rss
    except ImportError: runtime['RAM_bytes']='unavailable'
    for key in ['B','C']:
        model=ROOT/f'resources/detectors/{key}/pipeline.joblib'
        expected=protocol['input_sha256'][f'artifacts/module_{key.lower()}/pipeline.joblib']
        if sha(model)!=expected: raise ValueError('Model changed: '+key)
        t=time.perf_counter();pipeline=joblib.load(model);load_seconds=time.perf_counter()-t
        t=time.perf_counter();selected[key+'_score']=pipeline.predict_proba(selected.text.tolist())[:,list(pipeline.classes_).index(1)];elapsed=time.perf_counter()-t
        order=selected.loc[selected.main_task].copy();order['order']=order.sample_id.map(stable_id);order=order.sort_values('order')
        runtime[key]={'load_seconds':load_seconds,'full_batch_seconds':elapsed,'full_batch_texts':len(selected),**benchmark(pipeline,order.text)}
        if 'psutil' in locals(): runtime[key]['process_RSS_after_bytes']=psutil.Process().memory_info().rss
        print(f'{key}: scored {len(selected)} frozen test texts',flush=True)
    (args.output/'runtime.json').write_text(json.dumps(runtime,indent=2),encoding='utf-8')
    selected.drop(columns='text').to_csv(args.output/'predictions.csv',index=False)
    byid=selected.set_index('sample_id')
    for key in ['B','C']:
        pairs[key+'_original_score']=pairs.parent_id.map(byid[key+'_score'])
        pairs[key+'_edited_score']=pairs.sample_id.map(byid[key+'_score'])
    pairs.to_csv(args.output/'paired_scores.csv',index=False)
    result=summarize(selected,pairs,protocol,args.output)
    from src.evaluation.plots import create_plots
    create_plots(selected,pairs,args.output)
    from src.evaluation.explain_errors import explain
    explain(args.output)
    result.update(module='E',status='frozen_test_evaluation_complete',created_utc=datetime.now(timezone.utc).isoformat(),
                  protocol_sha256=sha(ROOT/'configs/module_e_frozen.json'),source_sha256={p.name:sha(p) for p in Path(__file__).parent.glob('*.py')},
                  artifact_sha256={p.name:sha(p) for p in args.output.iterdir() if p.is_file()})
    (args.output/'run.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps(result,indent=2))

if __name__=='__main__':main()
