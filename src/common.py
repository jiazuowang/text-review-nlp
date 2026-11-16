"""Shared frozen-data validation and binary development metrics."""
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, average_precision_score, confusion_matrix, f1_score, roc_auc_score

ROOT = Path(__file__).resolve().parents[1]


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_development_data(root):
    """Verify frozen provenance before reading the two development partitions."""
    root = Path(root)
    frozen = json.loads((root / 'CHECKSUMS.json').read_text(encoding='utf-8'))
    names = ['data/processed/split_manifest.csv', 'data/processed/audit.json',
             'data/processed/train_main.parquet', 'data/processed/val_main.parquet']
    hashes = {}
    for name in names:
        hashes[name] = sha256(root / name)
        if hashes[name] != frozen[name]:
            raise ValueError(f'Frozen module A checksum mismatch: {name}')
    manifest = pd.read_csv(root / names[0])
    audit = json.loads((root / names[1]).read_text(encoding='utf-8'))
    if audit['manifest_sha256'] != hashes[names[0]]:
        raise ValueError('Audit manifest checksum mismatch')
    frames = []
    for split in ('train', 'val'):
        frame = pd.read_parquet(root / f'data/processed/{split}_main.parquet')
        expected = manifest.loc[manifest.main_task & manifest.split.eq(split)].set_index('sample_id')
        if not frame.sample_id.is_unique or set(frame.sample_id) != set(expected.index):
            raise ValueError(f'{split}: sample membership mismatch')
        if set(frame.label) != {0, 1} or not frame.split.eq(split).all():
            raise ValueError(f'{split}: invalid labels or split')
        if not frame.main_task.all() or not frame.eligible.all():
            raise ValueError(f'{split}: ineligible rows')
        for row in frame.itertuples():
            record = expected.loc[row.sample_id]
            for col in ('label', 'group_id', 'variant', 'text_sha256'):
                if getattr(row, col) != record[col]:
                    raise ValueError(f'{split}: manifest metadata mismatch: {col}')
            if hashlib.sha256(row.text.encode('utf-8')).hexdigest() != row.text_sha256:
                raise ValueError(f'{split}: text mismatch')
        frames.append(frame)
    for col in ('sample_id', 'group_id', 'prompt_id', 'normalized_sha256'):
        if set(frames[0][col]) & set(frames[1][col]):
            raise ValueError(f'Train/validation overlap: {col}')
    return *frames, hashes


def metrics(labels, scores, threshold=0.5):
    predicted = (np.asarray(scores) >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(labels, predicted, labels=[0, 1]).ravel()
    return {'macro_f1': float(f1_score(labels, predicted, average='macro', zero_division=0)),
            'accuracy': float(accuracy_score(labels, predicted)),
            'ai_f1': float(f1_score(labels, predicted, zero_division=0)),
            'roc_auc': float(roc_auc_score(labels, scores)),
            'average_precision': float(average_precision_score(labels, scores)),
            'human_false_positive_rate': float(fp / (tn + fp)),
            'ai_recall': float(tp / (tp + fn)),
            'tn': int(tn), 'fp': int(fp), 'fn': int(fn), 'tp': int(tp)}


def select_candidate(rows):
    # Candidate order is the predeclared deterministic tie breaker.
    return max(range(len(rows)), key=lambda i: (rows[i]['macro_f1'], -i))
