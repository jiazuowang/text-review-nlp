"""Freeze final-evaluation protocol using existing validation predictions."""
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]


def select_threshold(labels, scores, fpr_limit=.05):
    y=np.asarray(labels);s=np.asarray(scores,dtype=float)
    if set(np.unique(y))!={0,1} or not np.isfinite(s).all():raise ValueError('Need finite scores and both classes')
    human=y==0;ai=y==1;max_fp=int(np.floor(fpr_limit*human.sum()))
    candidates=np.unique(np.r_[0.,.5,s,np.nextafter(s,np.inf),np.nextafter(1.,np.inf)])
    choices=[]
    for t in candidates:
        p=s>=t;fp=int(p[human].sum());tp=int(p[ai].sum())
        if fp<=max_fp:choices.append((-tp,fp,-float(t),float(t)))
    t=min(choices)[3];p=s>=t
    return dict(threshold=t,human_count=int(human.sum()),AI_count=int(ai.sum()),
        false_positives=int(p[human].sum()),true_positives=int(p[ai].sum()),
        validation_FPR=float(p[human].mean()),validation_TPR=float(p[ai].mean()))


def main():
    target=ROOT/'configs/module_e_frozen.json'
    if target.exists():raise FileExistsError(target)
    paths={'B':'artifacts/module_b/validation_predictions.csv','C':'artifacts/module_c/full_validation_predictions.csv'}
    inputs={};thresholds={}
    for key,path in paths.items():
        f=pd.read_csv(ROOT/path)
        thresholds[key]=select_threshold(f.label,f.ai_score)
        for name in [path,f'artifacts/module_{key.lower()}/pipeline.joblib',f'artifacts/module_{key.lower()}/run.json']:
            inputs[name]=hashlib.sha256((ROOT/name).read_bytes()).hexdigest()
    for name in ['CHECKSUMS.json','data/processed/split_manifest.csv','data/processed/texts.parquet','data/module_d/ai_review_decisions.jsonl','scripts/freeze_module_e.py']:
        inputs[name]=hashlib.sha256((ROOT/name).read_bytes()).hexdigest()
    protocol=dict(module='E',status='protocol_frozen_before_test_inference',created_utc=datetime.now(timezone.utc).isoformat(),
        positive_label=1,default_threshold=.5,low_FPR_target=.05,low_FPR_thresholds=thresholds,
        threshold_selection='Maximize validation AI true positives subject to human FP <= floor(0.05*n_human); ties minimize FP then maximize threshold; predict AI iff score >= threshold.',
        primary_models=['B','C'],main_test='Frozen eligible human and ai_raw test partition',
        edited_test='All frozen pair-eligible test edits, all seven published conditions; retain all attempts and source quality flags.',
        main_metrics=['accuracy','precision','recall','FPR','F1','macro_F1','AUROC'],
        edited_metrics=['TPR','miss_rate','mean_score_change','AI_to_human_flip','human_to_AI_flip'],
        weighting=['per_text','equal_group_weight'],bootstrap=dict(unit='group_id',replicates=1000,seed=42,CI=[.025,.975],paired_model_differences=True),
        runtime='Record CPU/RAM/threads and text lengths; separate load time, warmed single-text median/P95, batch throughput.',
        error_examples='Deterministic SHA256(42|sample_id) order; up to 10 false positives and 10 false negatives per detector at 0.5.',
        optional_extensions=['C feature ablations','external domain evaluation','custom edits','augmentation'],
        test_tuning='Frozen protocol; implementation corrections and reruns must be logged.',input_sha256=inputs)
    target.write_text(json.dumps(protocol,indent=2),encoding='utf-8')
    print(json.dumps(thresholds,indent=2))


if __name__=='__main__':main()
