"""Module B: train on frozen training data and select using validation macro F1."""
import argparse
from datetime import datetime, timezone
import hashlib
import itertools
import json
from pathlib import Path
import platform
import time
import warnings

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.exceptions import ConvergenceWarning
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, average_precision_score,
                             confusion_matrix, f1_score, roc_auc_score)
from sklearn.pipeline import Pipeline

ROOT = Path(__file__).resolve().parents[2]


from src.common import sha256, load_development_data, metrics, select_candidate


def train(root=ROOT, config_path=None, output=None):
    root = Path(root)
    config_path = Path(config_path or root / 'configs/module_b.json')
    output = Path(output or root / 'artifacts/module_b')
    cfg = json.loads(config_path.read_text(encoding='utf-8'))
    if cfg['selection_metric'] != 'macro_f1' or cfg['threshold'] != 0.5:
        raise ValueError('Module B protocol requires macro_f1 selection and threshold 0.5')
    train_frame, val_frame, hashes = load_development_data(root)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError('Choose a fresh --output directory to preserve experiment records')
    output.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    rows, best_model, best_score = [], None, -1.0
    for ngram, min_df in itertools.product(cfg['ngram_ranges'], cfg['min_df']):
        vectorizer = TfidfVectorizer(ngram_range=tuple(ngram), min_df=min_df,
            max_features=cfg['max_features'], sublinear_tf=True, lowercase=True,
            dtype=np.float64)
        x_train = vectorizer.fit_transform(train_frame.text)
        x_val = vectorizer.transform(val_frame.text)
        for c, weight in itertools.product(cfg['C'], cfg['class_weight']):
            tick = time.perf_counter()
            model = LogisticRegression(C=c, class_weight=weight, solver='liblinear',
                max_iter=cfg['max_iter'], random_state=cfg['seed'])
            with warnings.catch_warnings():
                warnings.simplefilter('error', ConvergenceWarning)
                model.fit(x_train, train_frame.label)
            score = model.predict_proba(x_val)[:, list(model.classes_).index(1)]
            row = {'candidate_id': len(rows), 'ngram_range': list(ngram),
                   'min_df': min_df, 'C': c, 'class_weight': weight,
                   'features': x_train.shape[1], 'iterations': int(model.n_iter_.max()),
                   'fit_predict_seconds': time.perf_counter() - tick,
                   **metrics(val_frame.label, score, cfg['threshold'])}
            rows.append(row)
            if row['macro_f1'] > best_score:
                best_score = row['macro_f1']
                best_model = Pipeline([('tfidf', vectorizer), ('classifier', model)])
            print(f"Candidate {row['candidate_id']:02d}: validation macro F1={row['macro_f1']:.6f}", flush=True)
    selected = rows[select_candidate(rows)]
    pd.DataFrame(rows).to_csv(output / 'validation_search.csv', index=False)
    joblib.dump(best_model, output / 'pipeline.joblib', compress=3)
    restored = joblib.load(output / 'pipeline.joblib')
    scores = best_model.predict_proba(val_frame.text)[:, 1]
    np.testing.assert_allclose(restored.predict_proba(val_frame.text)[:, 1], scores, rtol=0, atol=1e-12)
    predictions = val_frame[['sample_id', 'group_id', 'label']].copy()
    predictions['ai_score'] = scores
    predictions['prediction'] = (scores >= cfg['threshold']).astype(int)
    predictions.to_csv(output / 'validation_predictions.csv', index=False)
    feature_names = best_model.named_steps['tfidf'].get_feature_names_out()
    coefficients = best_model.named_steps['classifier'].coef_[0]
    order = np.argsort(coefficients)
    indices = np.concatenate([order[:30], order[-30:][::-1]])
    pd.DataFrame({'term': feature_names[indices], 'coefficient': coefficients[indices]}).to_csv(
        output / 'top_coefficients.csv', index=False)
    result = {'module': 'B', 'status': 'validation_selected',
        'created_utc': datetime.now(timezone.utc).isoformat(), 'config': cfg,
        'config_sha256': sha256(config_path), 'input_sha256': hashes,
        'source_sha256': sha256(Path(__file__)),
        'common_sha256': sha256(ROOT / 'src/common.py'),
        'training_rows': len(train_frame), 'validation_rows': len(val_frame),
        'candidates': len(rows), 'selected': selected,
        'threshold': cfg['threshold'], 'fit_split': 'train', 'selection_split': 'val',
        'final_test_status': 'reserved_for_module_E',
        'tie_break': 'earliest candidate in declared grid order',
        'validation_role': 'hyperparameter selection; development estimates',
        'serialization_verified': True, 'duration_seconds': time.perf_counter() - started,
        'environment': {'python': platform.python_version(), 'sklearn': sklearn.__version__,
                        'pandas': pd.__version__, 'numpy': np.__version__, 'joblib': joblib.__version__},
        'artifact_sha256': {p.name: sha256(p) for p in output.iterdir() if p.is_file()}}
    (output / 'run.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(json.dumps(result, indent=2))
    return result


def predict_texts(texts, artifact=ROOT / 'artifacts/module_b/pipeline.joblib'):
    """Load a trusted local artifact and return AI-origin scores in input order."""
    if isinstance(texts, str):
        raise ValueError("Pass a list of texts")
    values = list(texts)
    if any(not isinstance(t, str) or not t.strip() for t in values):
        raise ValueError('Each input must be nonempty text')
    if not values:
        return np.array([], dtype=float)
    model = joblib.load(artifact)
    return model.predict_proba(values)[:, list(model.classes_).index(1)]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=ROOT / 'configs/module_b.json')
    parser.add_argument('--output', type=Path, default=ROOT / 'artifacts/module_b')
    args = parser.parse_args()
    train(config_path=args.config, output=args.output)


if __name__ == '__main__':
    main()
