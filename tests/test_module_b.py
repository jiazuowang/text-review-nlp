"""Module B leakage guards, frozen membership and persisted inference checks."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import joblib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from src.models.tfidf_lr import ROOT, load_development_data, metrics, predict_texts, select_candidate, train


class ModuleBTests(unittest.TestCase):
    def test_development_loader_never_opens_test_text(self):
        original = pd.read_parquet
        seen = []
        def guarded(path, *args, **kwargs):
            seen.append(Path(path).name)
            self.assertIn(Path(path).name, ('train_main.parquet', 'val_main.parquet'))
            return original(path, *args, **kwargs)
        with patch('src.common.pd.read_parquet', side_effect=guarded):
            tr, va, _ = load_development_data(ROOT)
        self.assertEqual((len(tr), len(va)), (2318, 492))
        self.assertEqual(seen, ['train_main.parquet', 'val_main.parquet'])

    def test_frozen_checksum_tampering_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            (root / 'data/processed').mkdir(parents=True)
            (root / 'data/processed/split_manifest.csv').write_text('tampered')
            (root / 'CHECKSUMS.json').write_text(json.dumps({'data/processed/split_manifest.csv': 'original'}))
            with self.assertRaisesRegex(ValueError, 'checksum mismatch'):
                load_development_data(root)

    def test_training_vocabulary_excludes_validation_tokens(self):
        tr = pd.DataFrame({'text': ['human apple', 'human pear', 'machine red', 'machine blue'], 'label': [0, 0, 1, 1]})
        va = pd.DataFrame({'text': ['human unseenvalidationtoken', 'machine unseenvalidationtoken'],
                           'label': [0, 1], 'sample_id': ['v0', 'v1'], 'group_id': ['g0', 'g1']})
        cfg = {'seed': 42, 'selection_metric': 'macro_f1', 'threshold': 0.5,
               'ngram_ranges': [[1, 1]], 'min_df': [1], 'C': [1],
               'class_weight': [None], 'max_features': 100, 'max_iter': 200}
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            config = root / 'config.json'
            config.write_text(json.dumps(cfg))
            with patch('src.models.tfidf_lr.load_development_data', return_value=(tr, va, {})):
                result = train(root, config, root / 'output')
            model = joblib.load(root / 'output/pipeline.joblib')
            self.assertNotIn('unseenvalidationtoken', model.named_steps['tfidf'].vocabulary_)
            self.assertEqual(result['fit_split'], 'train')
            self.assertEqual(result['final_test_status'], 'reserved_for_module_E')
            self.assertTrue(result['serialization_verified'])

    def test_selection_uses_macro_f1_and_declared_tie_order(self):
        self.assertEqual(select_candidate([{'macro_f1': .7, 'accuracy': .99},
                                           {'macro_f1': .8, 'accuracy': .8},
                                           {'macro_f1': .8, 'accuracy': .95}]), 1)

    def test_metric_label_orientation(self):
        result = metrics([0, 0, 1, 1], [.1, .6, .4, .9])
        self.assertEqual([result[x] for x in ('tn', 'fp', 'fn', 'tp')], [1, 1, 1, 1])
        self.assertEqual(result['human_false_positive_rate'], .5)

    def test_inference_roundtrip_and_input_validation(self):
        texts = ['human apple', 'human pear', 'machine red', 'machine blue']
        model = Pipeline([('tfidf', TfidfVectorizer()), ('classifier', LogisticRegression())]).fit(texts, [0, 0, 1, 1])
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / 'model.joblib'
            joblib.dump(model, path)
            np.testing.assert_allclose(predict_texts(texts, path), model.predict_proba(texts)[:, 1])
            self.assertEqual(len(predict_texts([], path)), 0)
            with self.assertRaises(ValueError):
                predict_texts([' '], path)


if __name__ == '__main__':
    unittest.main()
