import unittest
import numpy as np
from src.evaluation.local_analysis import analyze,sentence_spans,suggestions
from src.models.extension import representation

class Dummy:
    classes_=np.array([0,1])
    def predict_proba(self,texts):
        scores=np.array([.8 if 'trigger' in t else .2 for t in texts]);return np.c_[1-scores,scores]

class ExtensionTests(unittest.TestCase):
    def test_offsets_preserve_original(self):
        text='First sentence!\n\nNext one. Final tail'
        self.assertEqual(''.join(text[a:b] for a,b in sentence_spans(text)),text)
    def test_deletion_direction_and_window_bounds(self):
        text='trigger '+('word '*59)+'. '+('other '*60)+'.'
        result=analyze(text,Dummy())
        self.assertAlmostEqual(result['sentences'][0]['deletion_impact'],.6)
        self.assertTrue(all(0<=s['start']<s['end']<=len(text) for s in result['windows']))
    def test_short_input_returns_empty_scores(self):
        self.assertIsNone(analyze('A short text.',Dummy())['score'])
    def test_suggestions_point_to_evidence(self):
        text='This is a very important result for every person. This is a very important result for every person.'
        tips=suggestions(text,sentence_spans(text))
        self.assertTrue(any(t['kind']=='repeated_sentence' for t in tips))
        for tip in tips:
            if tip['kind']=='vague_evaluation':self.assertEqual(text[tip['start']:tip['end']],tip['evidence'])
    def test_validation_words_never_fitted(self):
        cfg={'char_ngram_range':[3,5],'min_df':1,'char_max_features':100,'word_ngram_range':[1,2],'word_max_features':100}
        v=representation('word_char',cfg);v.fit(['training alpha beta','training beta gamma'])
        before=dict(v.transformer_list[0][1].vocabulary_);v.transform(['unseenwordzebra'])
        self.assertEqual(before,v.transformer_list[0][1].vocabulary_)

if __name__=='__main__':unittest.main()
