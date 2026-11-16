import unittest
import pandas as pd
from src.edits.review import summarize


class ReviewTests(unittest.TestCase):
    def fixture(self):
        rows=[]
        for i in range(4):
            r=dict(sample_id=str(i),parent_id='p'+str(i),condition='expert',original_sha256='o',edited_sha256='e')
            for d in ['B','C']:r.update({d+'_original_score':.8,d+'_edited_score':.2 if i==0 else .9})
            rows.append(r)
        scores=pd.DataFrame(rows)
        review=scores[['sample_id','parent_id','condition','original_sha256','edited_sha256']].iloc[:3].copy()
        review['status']=['pass','fail','uncertain']
        return scores,review

    def test_unreviewed_and_uncertain_excluded_from_pass_rates(self):
        s,r=self.fixture();joined,summary=summarize(s,r)
        self.assertEqual(joined.status.iloc[3],'not_selected_for_review')
        self.assertTrue((summary.passed_pairs==1).all())
        self.assertTrue((summary.edited_detection_rate==0).all())
        self.assertTrue((summary.ai_to_human_count==1).all())

    def test_changed_text_and_duplicate_decisions_rejected(self):
        s,r=self.fixture();r.loc[0,'edited_sha256']='changed'
        with self.assertRaises(ValueError):summarize(s,r)
        s,r=self.fixture()
        with self.assertRaises(ValueError):summarize(s,pd.concat([r,r.iloc[:1]]))

    def test_empty_pass_subset_has_missing_rate(self):
        s,r=self.fixture();r['status']='uncertain'
        _,summary=summarize(s,r)
        self.assertTrue(summary.edited_detection_rate.isna().all())


if __name__=='__main__':unittest.main()
