"""Module C random forests, validation selection and feature ablations."""
import argparse
from datetime import datetime, timezone
import itertools
import json
from pathlib import Path
import platform
import time

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.pipeline import Pipeline
from src.common import ROOT, load_development_data, metrics, sha256
from src.features.style_features import GROUPS, StyleTransformer, schema


def train(output=None, with_pos=False, config_path=None):
    output = Path(output or ROOT / 'artifacts/module_c')
    cfg = json.loads(Path(config_path or ROOT / 'configs/module_c.json').read_text(encoding='utf-8'))
    tr, va, hashes = load_development_data(ROOT)
    groups = cfg['with_pos_groups'] if with_pos else cfg['available_groups']
    if cfg['selection_metric'] != 'macro_f1' or cfg['threshold'] != 0.5:
        raise ValueError('Expected macro_f1 selection and 0.5 threshold')
    if output.exists() and any(output.iterdir()):
        raise FileExistsError('Choose a fresh output directory')
    if with_pos:
        from src.features.style_features import load_tagger
        load_tagger()
    output.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    rows, winners, group_models = [], [], {}
    selected_pipeline, selected_row = None, None
    for group in groups:
        transformer = StyleTransformer(group).fit(tr.text)
        xtr, xva = transformer.transform(tr.text), transformer.transform(va.text)
        names = transformer.get_feature_names_out()
        for split, frame, matrix in [('train', tr, xtr), ('val', va, xva)]:
            features = pd.DataFrame(matrix, columns=names)
            features.insert(0, 'sample_id', frame.sample_id.to_numpy())
            features.to_csv(output / f'{group}_{split}_features.csv', index=False)
        best_row, best_model = None, None
        for depth, leaf in itertools.product(cfg['max_depth'], cfg['min_samples_leaf']):
            tick = time.perf_counter()
            forest = RandomForestClassifier(n_estimators=cfg['n_estimators'], random_state=cfg['seed'],
                n_jobs=cfg['n_jobs'], max_depth=depth, min_samples_leaf=leaf, class_weight=cfg['class_weight'])
            forest.fit(xtr, tr.label)
            scores = forest.predict_proba(xva)[:, 1]
            row = {'candidate_id': len(rows), 'group': group, 'features': len(names),
                'max_depth': depth, 'min_samples_leaf': leaf, 'class_weight': cfg['class_weight'],
                'fit_predict_seconds': time.perf_counter() - tick, **metrics(va.label, scores)}
            rows.append(row)
            if best_row is None or row['macro_f1'] > best_row['macro_f1']:
                best_row, best_model = row, forest
            print(f"{group} depth={depth} leaf={leaf}: macro F1={row['macro_f1']:.6f}", flush=True)
        winners.append(best_row)
        pipeline = Pipeline([('features', transformer), ('classifier', best_model)])
        group_models[group] = pipeline
        joblib.dump(pipeline, output / f'{group}.joblib', compress=3)
        p = va[['sample_id', 'group_id', 'label']].copy()
        p['ai_score'] = best_model.predict_proba(xva)[:, 1]
        p['prediction'] = (p.ai_score >= cfg['threshold']).astype(int)
        p.to_csv(output / f'{group}_validation_predictions.csv', index=False)
        if selected_row is None or best_row['macro_f1'] > selected_row['macro_f1']:
            selected_row, selected_pipeline = best_row, pipeline
    joblib.dump(selected_pipeline, output / 'pipeline.joblib', compress=3)
    restored = joblib.load(output / 'pipeline.joblib')
    np.testing.assert_allclose(restored.predict_proba(va.text), selected_pipeline.predict_proba(va.text), atol=1e-12, rtol=0)
    selected_features = selected_pipeline.named_steps['features']
    importance = permutation_importance(selected_pipeline.named_steps['classifier'],
        selected_features.transform(va.text), va.label, scoring='f1_macro',
        n_repeats=cfg['permutation_repeats'], random_state=cfg['seed'], n_jobs=cfg['n_jobs'])
    pd.DataFrame({'feature': selected_features.get_feature_names_out(), 'mean_f1_decrease': importance.importances_mean,
        'std_f1_decrease': importance.importances_std}).sort_values('mean_f1_decrease', ascending=False).to_csv(
            output / 'validation_permutation_importance.csv', index=False)
    pd.DataFrame(rows).to_csv(output / 'validation_search.csv', index=False)
    pd.DataFrame(winners).to_csv(output / 'ablation_summary.csv', index=False)
    lengths = []
    for split, frame in [('train', tr), ('val', va)]:
        for label, part in frame.groupby('label'):
            lengths.append({'split': split, 'label': int(label), 'n': len(part),
                **{k: float(v) for k, v in part.word_count.describe().to_dict().items()}})
    pd.DataFrame(lengths).to_csv(output / 'length_summary.csv', index=False)
    (output / 'feature_schema.json').write_text(json.dumps(schema(), ensure_ascii=False, indent=2), encoding='utf-8')
    status = 'complete' if with_pos else 'partial_POS_resource_pending'
    result = {'module': 'C', 'status': status, 'created_utc': datetime.now(timezone.utc).isoformat(),
        'config': cfg, 'with_pos': with_pos, 'completed_groups': groups,
        'pending_groups': [] if with_pos else ['base_pos', 'full', 'full_no_length'],
        'input_sha256': hashes, 'source_sha256': {p: sha256(ROOT / p) for p in
            ['src/common.py', 'src/models/train_style.py', 'src/features/style_features.py', 'configs/module_c.json']},
        'training_rows': len(tr), 'validation_rows': len(va), 'selected': selected_row,
        'selection_rule': 'highest validation macro_f1; earliest candidate breaks ties',
        'fit_split': 'train', 'selection_split': 'val', 'final_test_status': 'reserved_for_module_E',
        'serialization_verified': True, 'duration_seconds': time.perf_counter() - started,
        'environment': {'python': platform.python_version(), 'sklearn': sklearn.__version__,
                        'numpy': np.__version__, 'pandas': pd.__version__, 'joblib': joblib.__version__}}
    if with_pos:
        result['pos_manifest_sha256'] = sha256(ROOT / 'resources/pos/manifest.json')
    result['artifact_sha256'] = {p.name: sha256(p) for p in output.iterdir() if p.is_file()}
    (output / 'run.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(json.dumps(result, indent=2))
    return result


def predict_texts(texts, artifact=ROOT / 'artifacts/module_c/pipeline.joblib'):
    if isinstance(texts, str):
        raise ValueError('Pass a list of texts')
    values = list(texts)
    if any(not isinstance(t, str) or not t.strip() for t in values):
        raise ValueError('Each input must be nonempty text')
    if not values:
        return np.array([], dtype=float)
    return joblib.load(artifact).predict_proba(values)[:, 1]


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--output', type=Path, default=ROOT / 'artifacts/module_c')
    ap.add_argument('--with-pos', action='store_true')
    ap.add_argument('--config', type=Path, default=ROOT / 'configs/module_c.json')
    args = ap.parse_args()
    train(args.output, args.with_pos, args.config)
