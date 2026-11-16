"""Context scores, sentence deletion impacts and explicit surface revision checks."""
import argparse
import hashlib
import html
import json
from pathlib import Path
import re
import joblib
import numpy as np
from src.features.style_features import WORD

ROOT=Path(__file__).resolve().parents[2]

def sentence_spans(text):
    spans=[];start=0
    for boundary in re.finditer(r'''[.!?]+["')\]]*(?=\s|$)|\n+''',text):
        end=boundary.end()
        if end>start:spans.append((start,end));start=end
    if start<len(text):spans.append((start,len(text)))
    return spans

def suggestions(text,spans):
    result=[];seen={}
    for start,end in spans:
        sentence=text[start:end];words=WORD.findall(sentence);norm=' '.join(w.lower() for w in words)
        if len(words)>=8 and norm in seen:
            result.append(dict(start=start,end=end,kind='repeated_sentence',evidence=sentence.strip(),
                               suggestion='Check the repeated sentence. Merge duplicate content or add information specific to this sentence.',earlier_start=seen[norm]))
        seen[norm]=start
        if len(words)>40:
            result.append(dict(start=start,end=end,kind='long_sentence',evidence=f'{len(words)} English words',
                               suggestion='Split the sentence by claim, evidence and outcome, with an explicit subject for each clause.'))
        for match in re.finditer(r'\b(?:very|really|extremely)\s+(?:important|good|bad|useful|effective)\b',sentence,re.I):
            result.append(dict(start=start+match.start(),end=start+match.end(),kind='vague_evaluation',evidence=match.group(),
                               suggestion='Specify what is being evaluated and add conditions or a checkable example supported by the available evidence.'))
    return result

def analyze(text,model,threshold=.5):
    tokens=list(WORD.finditer(text));n=len(tokens)
    base={'word_count':n,'threshold':threshold,'score_meaning':'Document AI-origin classification score',
          'local_meaning':'Context-window classification scores and sentence deletion impacts',
          'editing_goal':'Clarity, specificity and removal of redundant content',
          'local_validation_status':'Synthetic validation diagnostics; authentic sentence authorship labels unavailable'}
    if n<50 or n>1000:return {**base,'status':'outside_50_1000_word_scope','score':None,'sentences':[],'windows':[],'suggestions':[]}
    column=list(model.classes_).index(1)
    score=lambda texts:model.predict_proba(texts)[:,column]
    original=float(score([text])[0]);spans=sentence_spans(text)
    starts=sorted(set([*range(0,max(n-60,0)+1,30),max(n-60,0)]))
    windows=[(tokens[s].start(),tokens[min(s+60,n)-1].end()) for s in starts]
    ws=score([text[a:b] for a,b in windows])
    records=[];delete_texts=[];delete_indices=[]
    for a,b in spans:
        words=len(WORD.findall(text[a:b]));overlap=np.array([max(0,min(b,v)-max(a,u)) for u,v in windows])
        local=float(np.average(ws,weights=overlap)) if overlap.sum() else None
        record={'start':a,'end':b,'text':text[a:b],'word_count':words,'context_score':local,'deletion_impact':None,
                'impact_status':'remaining_text_under_50_words' if n-words<50 else 'pending'}
        if words==0:record['impact_status']='whitespace'
        elif n-words>=50 and len(delete_indices)<80:
            delete_indices.append(len(records));delete_texts.append(text[:a]+' '+text[b:])
        elif n-words>=50:record['impact_status']='80_sentence_runtime_limit'
        records.append(record)
    if delete_texts:
        for index,value in zip(delete_indices,score(delete_texts)):
            records[index].update(deletion_impact=original-float(value),impact_status='measured')
    return {**base,'status':'ok','score':original,'prediction':'AI origin' if original>=threshold else 'human',
            'sentences':records,'windows':[{'start':a,'end':b,'score':float(v)} for (a,b),v in zip(windows,ws)],
            'suggestions':suggestions(text,spans)}

def render(text,result,path,title='Local Text Analysis'):
    spans=[]
    for s in result['sentences']:
        impact=s['deletion_impact'];strength=min(abs(impact or 0)*5,.65)
        color=f'rgba(214,113,49,{strength:.3f})' if (impact or 0)>0 else f'rgba(42,124,165,{strength:.3f})'
        tip=f'Context score: {s["context_score"]}; Deletion impact: {impact}; {s["impact_status"]}'
        spans.append(f'<span style="background:{color}" title="{html.escape(tip)}">{html.escape(s["text"])}</span>')
    rows=[]
    for s in sorted(result['sentences'],key=lambda s:abs(s['deletion_impact'] or 0),reverse=True)[:10]:
        if s['deletion_impact'] is not None:
            rows.append(f'<tr><td>{html.escape(s["text"].strip())}</td><td>{s["deletion_impact"]:+.4f}</td><td>{s["context_score"]:.4f}</td></tr>')
    tips=''.join(f'<li><q>{html.escape(x["evidence"])}</q>: {html.escape(x["suggestion"])}</li>' for x in result['suggestions']) or '<li>The text triggered zero checks under the three implemented rules.</li>'
    page=f'''<!doctype html><html lang="en"><meta charset="utf-8"><title>{html.escape(title)}</title>
<style>body{{max-width:1100px;margin:32px auto;font:12pt "Times New Roman",SimSun;line-height:1.7}}h1,h2{{font:inherit}}.text{{white-space:pre-wrap;background:#fafafa;padding:20px}}table{{border-collapse:collapse;width:100%}}td,th{{border:1px solid #ccc;padding:8px;text-align:left}}</style>
<h1>{html.escape(title)}</h1><p>Document score: {result['score']}; threshold: {result['threshold']}; words: {result['word_count']}.</p>
<p>Orange indicates a decrease in the AI score after deletion; blue indicates an increase. Color intensity follows the change magnitude. Hover for values. Context scores use overlapping 60-word windows.</p>
<p>Passage results describe model sensitivity. Natural sentence-origin labels require dedicated validation. See localization_validation.json for the constructed mixed-text diagnostic.</p>
<div class="text">{''.join(spans) or html.escape(text)}</div><h2>Passages with larger impacts</h2><table><tr><th>Passage</th><th>Original score minus score after deletion</th><th>Context score</th></tr>{''.join(rows)}</table>
<h2>Writing suggestions</h2><ul>{tips}</ul><p>Checks cover repeated sentences, sentences over 40 words and specific vague evaluations. Preserve facts, sources and author intent when revising.</p></html>'''
    path.write_text(page,encoding='utf-8')

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--input',type=Path,required=True);ap.add_argument('--output',type=Path,required=True)
    ap.add_argument('--model-dir',type=Path,default=ROOT/'artifacts/extension_x');args=ap.parse_args()
    frozen=json.loads((args.model_dir/'selection_frozen.json').read_text())
    if hashlib.sha256((args.model_dir/'pipeline.joblib').read_bytes()).hexdigest()!=frozen['pipeline_sha256']:raise ValueError('Model hash changed')
    text=args.input.read_text(encoding='utf-8-sig');model=joblib.load(args.model_dir/'pipeline.joblib');result=analyze(text,model)
    args.output.mkdir(parents=True,exist_ok=False)
    (args.output/'analysis.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8');render(text,result,args.output/'analysis.html')
