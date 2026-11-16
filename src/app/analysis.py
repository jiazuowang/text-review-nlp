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
                               suggestion='Check whether this sentence repeats an earlier point. Merge overlapping statements or add specific evidence.',earlier_start=seen[norm]))
        seen[norm]=start
        if len(words)>40:
            result.append(dict(start=start,end=end,kind='long_sentence',evidence=f'{len(words)} English words',
                               suggestion='Split this long sentence at a change of subject, evidence or conclusion.'))
        for match in re.finditer(r'\b(?:very|really|extremely)\s+(?:important|good|bad|useful|effective)\b',sentence,re.I):
            result.append(dict(start=start+match.start(),end=start+match.end(),kind='vague_evaluation',evidence=match.group(),
                               suggestion='Specify the evaluation criteria and provide a concrete example or measurable result.'))
    return result

def analyze(text,model,threshold=.5,cancel=None):
    tokens=list(WORD.finditer(text));n=len(tokens)
    base={'word_count':n,'threshold':threshold,'score_meaning':'Document AI-origin classification score',
          'local_meaning':'Context-window classification scores and sentence deletion impacts',
          'editing_goal':'Clarity, specificity and removal of redundant content',
          'local_validation_status':'Synthetic validation diagnostics; authentic sentence authorship labels unavailable'}
    if n == 0:
        raise ValueError('Enter non-empty English text containing words.')
    column=list(model.classes_).index(1)
    def score(texts):
        from src.app.service import check_cancel
        chunks=[]
        for start in range(0,len(texts),8):
            check_cancel(cancel)
            chunks.extend(model.predict_proba(texts[start:start+8])[:,column])
        return np.array(chunks)
    original=float(score([text])[0]);spans=sentence_spans(text)
    starts=sorted(set([*range(0,max(n-60,0)+1,30),max(n-60,0)]))
    windows=[(tokens[s].start(),tokens[min(s+60,n)-1].end()) for s in starts]
    ws=score([text[a:b] for a,b in windows])
    records=[];delete_texts=[];delete_indices=[]
    for a,b in spans:
        words=len(WORD.findall(text[a:b]));overlap=np.array([max(0,min(b,v)-max(a,u)) for u,v in windows])
        local=float(np.average(ws,weights=overlap)) if overlap.sum() else None
        record={'start':a,'end':b,'text':text[a:b],'word_count':words,'context_score':local,'deletion_impact':None,
                'impact_status':'no_remaining_words' if n-words == 0 else 'pending'}
        if words==0:record['impact_status']='whitespace'
        elif n-words>0 and len(delete_indices)<80:
            delete_indices.append(len(records));delete_texts.append(text[:a]+' '+text[b:])
        elif n-words>0:record['impact_status']='80_sentence_runtime_limit'
        records.append(record)
    if delete_texts:
        for index,value in zip(delete_indices,score(delete_texts)):
            records[index].update(deletion_impact=original-float(value),impact_status='measured')
    return {**base,'status':'ok','score':original,'prediction':'AI origin' if original>=threshold else 'Human origin',
            'sentences':records,'windows':[{'start':a,'end':b,'score':float(v)} for (a,b),v in zip(windows,ws)],
            'suggestions':suggestions(text,spans)}
