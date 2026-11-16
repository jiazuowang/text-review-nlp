"""Validate saved AI review decisions and reproduce conditional quality summaries."""
import argparse
import hashlib
import html
import json
from pathlib import Path

import pandas as pd


def summarize(scores, decisions):
    if decisions.sample_id.duplicated().any():
        raise ValueError('Duplicate review decision')
    if not set(decisions.status) <= {'pass', 'fail', 'uncertain'}:
        raise ValueError('Invalid review status')
    joined = scores.merge(decisions, on='sample_id', how='left', validate='one_to_one', suffixes=('', '_review'))
    for key in ['parent_id', 'condition', 'original_sha256', 'edited_sha256']:
        selected = joined.status.notna()
        if not joined.loc[selected, key].equals(joined.loc[selected, key+'_review']):
            raise ValueError('Review provenance mismatch: '+key)
    joined['status'] = joined.status.fillna('not_selected_for_review')
    rows = []
    for condition, part in joined.groupby('condition', sort=True):
        accepted = part.loc[part.status.eq('pass')]
        for detector in ['B', 'C']:
            original = accepted[detector+'_original_score']
            edited = accepted[detector+'_edited_score']
            rows.append(dict(condition=condition, detector=detector, reviewed_pairs=int(part.status.ne('not_selected_for_review').sum()),
                passed_pairs=len(accepted), original_detection_rate=(original >= .5).mean() if len(accepted) else None,
                edited_detection_rate=(edited >= .5).mean() if len(accepted) else None,
                mean_score_change=(edited-original).mean(),
                ai_to_human_count=int(((original >= .5)&(edited < .5)).sum()),
                human_to_ai_count=int(((original < .5)&(edited >= .5)).sum())))
    return joined, pd.DataFrame(rows)


def run(analysis, review_file, output):
    output.mkdir(parents=True, exist_ok=False)
    scores = pd.read_csv(analysis/'paired_scores.csv')
    decisions = pd.read_json(review_file, lines=True)
    required = pd.read_csv(analysis/'review_required_pairs.csv')
    if set(decisions.sample_id) != set(required.sample_id):
        raise ValueError('Required review coverage mismatch')
    if decisions.human_review_completed.any():
        raise ValueError('AI review must retain human-review provenance')
    joined, summary = summarize(scores, decisions)
    joined.to_csv(output/'reviewed_pairs.csv', index=False)
    summary.to_csv(output/'quality_pass_summary.csv', index=False)
    counts = joined.groupby(['condition','status']).size().unstack(fill_value=0)
    counts.to_csv(output/'quality_counts.csv')
    accepted = joined.loc[joined.status.eq('pass')]
    cohort_sets = [set(accepted.loc[accepted.condition.eq(c),'parent_id']) for c in sorted(joined.condition.unique())]
    common = sorted(set.intersection(*cohort_sets))
    result = dict(status='required_AI_review_complete', required_pairs=len(required), completed_AI_reviews=len(decisions),
        human_reviews_completed=0, total_eligible_pairs=len(scores), decisions=decisions.status.value_counts().to_dict(),
        reason_codes=decisions.reason_code.value_counts().to_dict(), reviewer_types=decisions.reviewer_type.value_counts().to_dict(),
        unreviewed_pairs=int(joined.status.eq('not_selected_for_review').sum()), quality_pass_common_parents=len(common),
        review_sha256=hashlib.sha256(review_file.read_bytes()).hexdigest(),
        external_fact_verification='not_performed',
        interpretation='Conditional descriptive results on AI-reviewed passes; condition-specific parent sets and targeted review selection.',
        regeneration='Saved review judgments are fixed inputs; statistics and evidence exports are reproduced deterministically.')
    (output/'run.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    (output/'common_pass_parents.json').write_text(json.dumps(common),encoding='utf-8')
    texts = pd.read_parquet(analysis/'paired_edits.parquet').set_index('sample_id')
    examples=[]
    for code in ['P','T','M','O','C']:
        row=decisions.loc[decisions.reason_code.eq(code)].sort_values('sample_id').iloc[0].to_dict()
        pair=texts.loc[row['sample_id']]
        row.update(original_text=pair.original_text, edited_text=pair.edited_text)
        examples.append(row)
    (output/'review_examples.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in examples),encoding='utf-8')
    body=['<!doctype html><meta charset="utf-8"><title>D review evidence</title><style>body{max-width:1100px;margin:30px auto;font-family:Arial}pre{white-space:pre-wrap;line-height:1.5}article{border-top:1px solid #bbb}</style><h1>D: AI review evidence</h1>']
    for r in examples:
        body.append('<article><h2>'+html.escape(r['sample_id'])+'</h2><p>'+html.escape(r['status']+' / '+r['reason_code']+' / '+r['review_input'])+'</p><p>'+html.escape(r['pair_context_note'])+'</p><h3>Original</h3><pre>'+html.escape(r['original_text'])+'</pre><h3>Edited</h3><pre>'+html.escape(r['edited_text'])+'</pre></article>')
    (output/'review_examples.html').write_text('\n'.join(body),encoding='utf-8')
    print(json.dumps(result,indent=2))


if __name__ == '__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--analysis',type=Path,default=Path('artifacts/module_d'))
    ap.add_argument('--reviews',type=Path,default=Path('data/module_d/ai_review_decisions.jsonl'))
    ap.add_argument('--output',type=Path,default=Path('artifacts/module_d_quality'))
    args=ap.parse_args();run(args.analysis,args.reviews,args.output)
