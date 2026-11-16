"""Validation-only paired editing analysis with exact token edit distance."""
import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import html
import json
import math
from pathlib import Path
import re

import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
WORDS = re.compile(r"[A-Za-z]+(?:['’\-][A-Za-z]+)*")
NUMBERS = re.compile(r'(?<!\w)[+-]?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?%?')
DATES = re.compile(r'\b(?:\d{1,4}[-/]\d{1,2}[-/]\d{1,4}|(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2}(?:,?\s+\d{4})?)\b', re.I)
NAMES = re.compile(r'\b(?:[A-Z][a-z]+(?:\s+[A-Z][a-z]+)+|[A-Z]{2,})\b')
CITATIONS = re.compile(r'https?://[^\s<>"\)]+|\[(?:\d+|\d+(?:\s*[,–-]\s*\d+)+)\]')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def text_sha(text):
    return hashlib.sha256(text.encode('utf-8')).hexdigest()


def word_distance(a, b):
    """Exact unit-cost Levenshtein distance using Myers bit vectors."""
    if not a: return len(b)
    if not b: return len(a)
    positions = {}
    for i, token in enumerate(a): positions[token] = positions.get(token,0) | (1 << i)
    mask = (1 << len(a)) - 1
    pv, mv, score, high = mask, 0, len(a), 1 << (len(a)-1)
    for token in b:
        eq = positions.get(token,0)
        xv = eq | mv
        xh = (((eq & pv) + pv) ^ pv) | eq
        ph = mv | ~(xh | pv)
        mh = pv & xh
        score += int(bool(ph & high)) - int(bool(mh & high))
        ph = (ph << 1) | 1
        mh <<= 1
        pv = (mh | ~(xv | ph)) & mask
        mv = ph & xv & mask
    return score


def changes(pattern, original, edited):
    a, b = Counter(pattern.findall(original)), Counter(pattern.findall(edited))
    return {'removed': sorted((a-b).elements()), 'added': sorted((b-a).elements())}


def quality(original, edited, bounds):
    a, b = [w.lower() for w in WORDS.findall(original)], [w.lower() for w in WORDS.findall(edited)]
    ratio = len(b)/len(a) if a else None
    dist = word_distance(a,b)
    flags = []
    if ratio is None or ratio < bounds[0] or ratio > bounds[1]: flags.append('length_ratio')
    details = {}
    for name, pattern in [('numbers',NUMBERS),('dates',DATES),('name_candidates',NAMES),('citations',CITATIONS)]:
        detail = changes(pattern,original,edited)
        details[name+'_changes'] = json.dumps(detail,ensure_ascii=False,sort_keys=True)
        if detail['removed'] or detail['added']: flags.append(name)
    return {'original_words':len(a),'edited_words':len(b),'length_ratio':ratio,
            'word_edit_distance':dist,'normalized_word_edit_distance':dist/max(len(a),len(b),1),
            'identical_text':original==edited,'identical_word_sequence':a==b,
            'review_flags':';'.join(flags),'automatic_screen':'flagged' if flags else 'unflagged',**details}


def load_input(root=ROOT):
    folder = root/'data/module_d'
    provenance = json.loads((folder/'provenance.json').read_text())
    if provenance['split'] != 'val': raise ValueError('Module D requires validation data')
    for name, expected in provenance['export_files'].items():
        if sha(folder/name) != expected: raise ValueError('Input checksum mismatch: '+name)
    f = pd.read_parquet(folder/'validation_texts.parquet')
    m = pd.read_csv(folder/'validation_manifest.csv').set_index('sample_id')
    if not f.sample_id.is_unique or set(f.sample_id)!=set(m.index): raise ValueError('Sample membership mismatch')
    if not f.split.eq('val').all(): raise ValueError('Non-validation input')
    for row in f.itertuples():
        entry=m.loc[row.sample_id]
        for col in ['split','group_id','variant','label','text_sha256','eligible','pair_eligible']:
            if getattr(row,col)!=entry[col]: raise ValueError('Metadata mismatch: '+col)
        if text_sha(row.text)!=row.text_sha256: raise ValueError('Text hash mismatch')
    return f,provenance


def pair_data(frame, config):
    if not frame.sample_id.is_unique: raise ValueError('Duplicate sample IDs')
    if not frame.split.eq('val').all(): raise ValueError('Non-validation pair source')
    byid=frame.set_index('sample_id')
    pairs,audit=[],[]
    for edit in frame.loc[frame.variant.isin(['human_edited','llm_edited'])].sort_values('sample_id').itertuples():
        condition='expert' if edit.variant=='human_edited' else f'{edit.editor}:{edit.edit_prompt_id}'
        if condition not in config['conditions']: raise ValueError('Unknown condition')
        reasons=[]
        if edit.parent_id not in byid.index:
            reasons.append('missing_parent');parent=None
        else:
            parent=byid.loc[edit.parent_id]
            if parent.variant!='ai_raw' or parent.label!=1 or edit.label!=1: raise ValueError('Invalid source labels')
            if parent.group_id!=edit.group_id or parent.split!=edit.split: raise ValueError('Parent split/group mismatch')
            if not parent.eligible: reasons.append('parent:'+str(parent.exclusion_reason))
        if not edit.eligible: reasons.append('edit:'+str(edit.exclusion_reason))
        expected=parent is not None and bool(parent.eligible) and bool(edit.eligible)
        if bool(edit.pair_eligible)!=expected: raise ValueError('Pair eligibility mismatch')
        audit.append({'sample_id':edit.sample_id,'parent_id':edit.parent_id,'condition':condition,
            'included':expected,'exclusion_reasons':';'.join(reasons)})
        if not expected: continue
        pairs.append({'sample_id':edit.sample_id,'parent_id':edit.parent_id,'group_id':edit.group_id,
            'prompt_id':edit.prompt_id,'category':edit.category,'generator':edit.generator,
            'condition':condition,'editor':edit.editor,'edit_prompt_id':edit.edit_prompt_id,
            'split':edit.split,'label':1,'original_text':parent.text,'edited_text':edit.text,
            'original_sha256':parent.text_sha256,'edited_sha256':edit.text_sha256,
            **quality(parent.text,edit.text,config['length_ratio_review_bounds'])})
    return pd.DataFrame(pairs),pd.DataFrame(audit)


def review_queue(pairs,config):
    selected=set()
    for condition,group in pairs.groupby('condition'):
        ids=sorted(group.sample_id,key=lambda x:text_sha(f"{config['seed']}|{x}"))
        selected.update(ids[:math.ceil(len(ids)*config['review_fraction'])])
    q=pairs.copy()
    q['random_audit_sample']=q.sample_id.isin(selected)
    q['review_required']=q.random_audit_sample | q.review_flags.ne('')
    q['human_review_status']='pending'
    q['human_reviewer']=''
    q['human_review_notes']=''
    return q


def score_pairs(pairs,root=ROOT):
    unique={}
    for row in pairs.itertuples():
        unique[row.parent_id]=row.original_text
        unique[row.sample_id]=row.edited_text
    ids=sorted(unique)
    result=pairs.copy()
    records=[]
    for key in ['B','C']:
        folder=root/f'resources/detectors/{key}'
        run=json.loads((folder/'run.json').read_text())
        if sha(folder/'pipeline.joblib') != run['artifact_sha256']['pipeline.joblib']:
            raise ValueError('Detector checksum mismatch')
        pipeline=joblib.load(folder/'pipeline.joblib')
        values=pipeline.predict_proba([unique[x] for x in ids])[:,list(pipeline.classes_).index(1)]
        scores=dict(zip(ids,values))
        result[key+'_original_score']=result.parent_id.map(scores)
        result[key+'_edited_score']=result.sample_id.map(scores)
        result[key+'_score_change']=result[key+'_edited_score']-result[key+'_original_score']
        records.append({'detector':key,'pipeline_sha256':sha(folder/'pipeline.joblib'),
            'training_run_sha256':sha(folder/'run.json'),'unique_texts_scored':len(ids)})
        print(f'Detector {key}: {len(ids)} validation texts scored',flush=True)
    return result,records


def summarize(scores,threshold,scope):
    rows=[]
    for condition,part in scores.groupby('condition',sort=True):
        for key in ['B','C']:
            before=part[key+'_original_score'].ge(threshold)
            after=part[key+'_edited_score'].ge(threshold)
            flips=before & ~after
            rows.append({'scope':scope,'condition':condition,'detector':key,'pairs':len(part),
                'unique_parents':part.parent_id.nunique(),'original_detection_rate':before.mean(),
                'edited_detection_rate':after.mean(),'mean_score_change':part[key+'_score_change'].mean(),
                'ai_to_human_count':int(flips.sum()),'ai_to_human_fraction_all_pairs':flips.mean(),
                'ai_to_human_fraction_detected_originals':float(flips.sum()/before.sum()) if before.sum() else None,
                'human_to_ai_count':int((~before & after).sum()),
                'mean_word_edit_distance':part.normalized_word_edit_distance.mean(),
                'mean_length_ratio':part.length_ratio.mean()})
    return pd.DataFrame(rows)


def save_examples(pairs,output,config):
    ids=sorted(pairs.sample_id,key=lambda x:text_sha(f"examples|{config['seed']}|{x}"))[:config['example_count']]
    chosen=pairs.set_index('sample_id').loc[ids].reset_index()
    chosen.to_json(output/'examples.jsonl',orient='records',lines=True,force_ascii=False)
    sections=[]
    for row in chosen.itertuples():
        sections.append(f'<section><h2>{html.escape(row.sample_id)} — {html.escape(row.condition)}</h2>'
            f'<p>Length ratio {row.length_ratio:.3f} · Normalized word edit distance {row.normalized_word_edit_distance:.3f}</p>'
            f'<div class="pair"><article><h3>Original</h3><pre>{html.escape(row.original_text)}</pre></article>'
            f'<article><h3>Edited</h3><pre>{html.escape(row.edited_text)}</pre></article></div></section>')
    page='<!doctype html><html lang="en"><meta charset="utf-8"><title>Editing pairs</title><style>body{font-family:Arial,sans-serif;margin:32px;color:#17202a}.pair{display:grid;grid-template-columns:1fr 1fr;gap:20px}pre{white-space:pre-wrap;line-height:1.6}article{background:#f3f5f7;padding:20px}section{margin-bottom:40px}@media(max-width:800px){.pair{grid-template-columns:1fr}}</style><h1>Validation originals and edits</h1><p>Five examples selected by fixed hash order. Human quality review remains pending.</p>'+''.join(sections)+'</html>'
    (output/'examples.html').write_text(page,encoding='utf-8')


def make_plot(scores,output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,2,figsize=(11,4),sharex=True,sharey=True)
    for ax,key in zip(axes,['B','C']):
        for condition,part in scores.groupby('condition'):
            ax.scatter(part.normalized_word_edit_distance,part[key+'_score_change'],s=7,alpha=.35,label=condition)
        ax.axhline(0,color='black',lw=.8)
        ax.set_title('Detector '+key);ax.set_xlabel('Normalized word edit distance')
    axes[0].set_ylabel('Edited score - original score')
    handles,labels=axes[1].get_legend_handles_labels()
    fig.legend(handles,labels,loc='lower center',ncol=4,fontsize=7)
    fig.subplots_adjust(bottom=.24,wspace=.13)
    fig.savefig(output/'edit_distance_score_change.png',dpi=180)
    plt.close(fig)


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--output',type=Path,default=ROOT/'artifacts/module_d')
    args=ap.parse_args()
    if args.output.exists() and any(args.output.iterdir()): raise FileExistsError('Choose a fresh output directory')
    config=json.loads((ROOT/'configs/module_d.json').read_text())
    if config['split']!='val' or config['threshold']!=.5: raise ValueError('Unexpected development protocol')
    frame,provenance=load_input()
    pairs,audit=pair_data(frame,config)
    args.output.mkdir(parents=True,exist_ok=True)
    pairs.to_parquet(args.output/'paired_edits.parquet',index=False)
    audit.to_csv(args.output/'pair_audit.csv',index=False)
    queue=review_queue(pairs,config)
    queue.drop(columns=['original_text','edited_text']).to_csv(args.output/'quality_review.csv',index=False)
    queue.loc[queue.review_required].to_csv(args.output/'review_required_pairs.csv',index=False,encoding='utf-8-sig')
    scores,detectors=score_pairs(pairs)
    scores.drop(columns=['original_text','edited_text']).to_csv(args.output/'paired_scores.csv',index=False)
    summary=summarize(scores,config['threshold'],'available_pairs')
    summary.to_csv(args.output/'condition_summary.csv',index=False)
    common_ids=set(scores.groupby('parent_id').condition.nunique().loc[lambda x:x==len(config['conditions'])].index)
    common=scores.loc[scores.parent_id.isin(common_ids)]
    summarize(common,config['threshold'],'common_parent_cohort').to_csv(args.output/'common_cohort_summary.csv',index=False)
    scores[['parent_id']].drop_duplicates().sort_values('parent_id').assign(common_all_conditions=lambda x:x.parent_id.isin(common_ids)).to_csv(args.output/'parent_cohorts.csv',index=False)
    audit.groupby('condition').agg(candidates=('sample_id','size'),included=('included','sum')).to_csv(args.output/'condition_coverage.csv')
    records=[]
    for row in pairs.itertuples():
        records.append({'sample_id':row.sample_id,'parent_id':row.parent_id,'condition':row.condition,
            'origin':'published_Beemo_edit','editor':row.editor,'prompt_id':row.edit_prompt_id,
            'source_revision':str(frame.dataset_revision.iloc[0]),'temperature':'unknown',
            'generation_timestamp':'unknown','original_sha256':row.original_sha256,'edited_sha256':row.edited_sha256})
    with (args.output/'generation_manifest.jsonl').open('w',encoding='utf-8') as stream:
        for row in records: stream.write(json.dumps(row,ensure_ascii=False)+'\n')
    save_examples(pairs,args.output,config)
    make_plot(scores,args.output)
    result={'module':'D','status':'automated_analysis_complete_human_review_pending',
        'created_utc':datetime.now(timezone.utc).isoformat(),'config':config,'input_provenance':provenance,
        'candidate_pairs':len(audit),'included_pairs':len(pairs),'excluded_pairs':len(audit)-len(pairs),
        'unique_parents':pairs.parent_id.nunique(),'common_cohort_parents':len(common_ids),
        'random_review_sample':int(queue.random_audit_sample.sum()),
        'flagged_pairs':int(queue.review_flags.ne('').sum()),'review_required':int(queue.review_required.sum()),
        'human_review_completed':0,'custom_generation':'not_run','detectors':detectors,
        'final_test_status':'reserved_for_module_E',
        'source_sha256':{name:sha(ROOT/name) for name in ['src/edits/analyze.py','configs/module_d.json']},
        'artifact_sha256':{p.name:sha(p) for p in args.output.iterdir() if p.is_file()}}
    (args.output/'run.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps({k:v for k,v in result.items() if k not in ['artifact_sha256','input_provenance']},indent=2))


if __name__=='__main__':main()
