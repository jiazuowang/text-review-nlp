import math
import random
import unittest
import pandas as pd
from src.edits.analyze import word_distance, quality, pair_data, review_queue, summarize


class EditingTests(unittest.TestCase):
    def test_distance_against_dynamic_programming(self):
        rng=random.Random(42)
        for _ in range(250):
            a=[rng.choice('abcd') for _ in range(rng.randrange(100))]
            b=[rng.choice('abcd') for _ in range(rng.randrange(100))]
            previous=list(range(len(b)+1))
            for i,x in enumerate(a,1):
                row=[i]
                for j,y in enumerate(b,1):
                    row.append(min(row[-1]+1,previous[j]+1,previous[j-1]+(x!=y)))
                previous=row
            self.assertEqual(word_distance(a,b),previous[-1])

    def test_quality_flags_and_boundaries(self):
        q=quality('New York paid 10 on January 1, 2024 [1].','Boston paid 12 on January 2, 2024 [2].',[.5,1.5])
        for flag in ['numbers','dates','name_candidates','citations']:self.assertIn(flag,q['review_flags'])
        self.assertNotIn('length_ratio',quality('a b','a',[.5,1.5])['review_flags'])
        self.assertIn('length_ratio',quality('a b c','a',[.5,1.5])['review_flags'])
        self.assertEqual(quality('Hello, world!','hello world',[.5,1.5])['word_edit_distance'],0)
        self.assertIn('length_ratio',quality('','a',[.5,1.5])['review_flags'])

    def fixture(self):
        common=dict(split='val',group_id='g',label=1,eligible=True,exclusion_reason='',prompt_id='p',category='x',generator='g',text_sha256='h',editor='',edit_prompt_id='')
        return pd.DataFrame([dict(common,sample_id='a',variant='ai_raw',text='one two',parent_id='',pair_eligible=False),dict(common,sample_id='b',variant='human_edited',text='one three',parent_id='a',pair_eligible=True),dict(common,sample_id='c',variant='human_edited',text='x',parent_id='a',eligible=False,exclusion_reason='outside_word_range',pair_eligible=False)])

    def test_pairs_keep_exclusion_audit(self):
        pairs,audit=pair_data(self.fixture(),{'conditions':['expert'],'length_ratio_review_bounds':[.5,1.5]})
        self.assertEqual(len(pairs),1);self.assertEqual(len(audit),2)
        self.assertEqual(audit.included.sum(),1)
        self.assertIn('outside_word_range',audit.iloc[1].exclusion_reasons)

    def test_pair_integrity_guards(self):
        config={'conditions':['expert'],'length_ratio_review_bounds':[.5,1.5]}
        for col,value in [('split','test'),('group_id','other'),('pair_eligible',False),('label',0)]:
            f=self.fixture();f.loc[1,col]=value
            with self.assertRaises(ValueError):pair_data(f,config)

    def test_review_sampling(self):
        pairs=pd.DataFrame([dict(sample_id=str(i),condition='a' if i<11 else 'b',review_flags='numbers' if i%3==0 else '') for i in range(25)])
        config={'seed':42,'review_fraction':.2}
        a=review_queue(pairs,config);b=review_queue(pairs.iloc[::-1],config)
        self.assertEqual(set(a.loc[a.random_audit_sample].sample_id),set(b.loc[b.random_audit_sample].sample_id))
        self.assertEqual(a.random_audit_sample.sum(),math.ceil(11*.2)+math.ceil(14*.2))
        self.assertTrue(a.loc[a.review_flags.ne('')].review_required.all())

    def test_flip_denominators(self):
        f=pd.DataFrame(dict(condition=['x']*3,parent_id=['a','b','c'],normalized_word_edit_distance=[.1]*3,length_ratio=[1]*3))
        for key in ['B','C']:
            f[key+'_original_score']=[.8,.6,.1];f[key+'_edited_score']=[.3,.7,.6]
            f[key+'_score_change']=f[key+'_edited_score']-f[key+'_original_score']
        row=summarize(f,.5,'test').iloc[0]
        self.assertEqual(row.ai_to_human_count,1)
        self.assertAlmostEqual(row.ai_to_human_fraction_all_pairs,1/3)
        self.assertEqual(row.ai_to_human_fraction_detected_originals,.5)
        self.assertEqual(row.human_to_ai_count,1)


if __name__=='__main__':unittest.main()
