import tempfile
from pathlib import Path
import unittest
from unittest.mock import patch
import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from src.features.style_features import extract, StyleTransformer, GROUPS, LENGTH, ROOT, load_tagger
from src.models.train_style import predict_texts


class StyleTests(unittest.TestCase):
    @unittest.skipUnless((ROOT / 'resources/pos/manifest.json').exists(), 'Official POS model required')
    def test_official_POS_resource_and_full_transform(self):
        tagger = load_tagger()
        tagged = tagger.tag('The quick brown fox jumps over the lazy dog'.split())
        self.assertEqual(tagged[0], ('The', 'DT'))
        self.assertEqual(tagged[-1], ('dog', 'NN'))
        matrix = StyleTransformer('full').fit_transform(['The cat sleeps, and the dog runs.', ''])
        self.assertEqual(matrix.shape, (2, len(GROUPS['full'])))
        self.assertTrue(np.isfinite(matrix).all())
        self.assertTrue(np.all(matrix[1] == 0))

    def test_empty_and_punctuation(self):
        self.assertTrue(all(v == 0 for v in extract('').values()))
        self.assertTrue(np.isfinite(list(extract('... ,!?').values())).all())

    def test_hand_computed_example(self):
        f = extract('Cat, cat sleeps. Dogs run!')
        self.assertEqual(f['word_count'], 5)
        self.assertEqual(f['sentence_count'], 2)
        self.assertEqual(f['sentence_words_mean'], 2.5)
        self.assertEqual(f['sentence_words_std'], .5)
        self.assertEqual(f['type_token_ratio'], .8)
        self.assertEqual(f['hapax_ratio'], .6)
        self.assertEqual(f['comma_per_100_words'], 20)
        self.assertEqual(f['comma_sentence_ratio'], .5)
        self.assertAlmostEqual(f['comma_sentence_rate_mean'], 1/6)
        self.assertAlmostEqual(f['comma_relative_position_mean'], 1/3)

    def test_paragraphs_and_lists(self):
        f = extract('- One item.\n- Another item.\n\nLast paragraph.')
        self.assertEqual(f['paragraph_count'], 2)
        self.assertAlmostEqual(f['list_line_ratio'], 2/3)

    def test_no_cross_sentence_bigrams(self):
        self.assertEqual(extract('Cat. Cat.')['repeated_bigram_ratio'], 0)
        self.assertAlmostEqual(extract('Red cat red cat.')['repeated_bigram_ratio'], 1/3)

    def test_length_ablation(self):
        self.assertFalse(set(GROUPS['base_comma_no_length']) & LENGTH)
        self.assertFalse(set(GROUPS['full_no_length']) & LENGTH)

    def test_POS_formulas_using_tagged_fixture(self):
        class Tagger:
            def tag(self, words):
                tags = {'Cat': 'NN', 'cat': 'NN', 'sleeps': 'VBZ', ',': ','}
                return [(w, tags[w]) for w in words]
        f = extract('Cat, cat sleeps.', True, Tagger())
        self.assertEqual(f['pos_diversity_1'], 2/3)
        self.assertEqual(f['pos_diversity_2'], 1)
        self.assertEqual(f['pos_diversity_5'], 0)
        self.assertEqual(f['comma_pos_pair_diversity'], 1)

    def test_no_POS_fallback_when_missing(self):
        with patch('src.features.style_features.load_tagger', side_effect=FileNotFoundError('missing')):
            with self.assertRaises(FileNotFoundError):
                StyleTransformer('full').fit(['Hello world.'])

    def test_serialization_and_inference(self):
        texts = ['Cat sleeps.', 'Dogs run.', 'Firstly, we consider this result.', 'Moreover, this suggests a conclusion.']
        pipe = Pipeline([('features', StyleTransformer('base_comma')), ('classifier', RandomForestClassifier(n_estimators=5, random_state=42))])
        pipe.fit(texts, [0, 0, 1, 1])
        with tempfile.TemporaryDirectory() as d:
            p = Path(d)/'model.joblib'
            joblib.dump(pipe, p)
            np.testing.assert_allclose(predict_texts(texts, p), pipe.predict_proba(texts)[:, 1])
            with self.assertRaises(ValueError): predict_texts('a string', p)
            with self.assertRaises(ValueError): predict_texts([''], p)


if __name__ == '__main__':
    unittest.main()
