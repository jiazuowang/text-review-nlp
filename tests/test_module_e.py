import unittest
import numpy as np
import pandas as pd
from src.evaluation.metrics import metrics,bootstrap_main,group_weights,paired_edit_summary
from src.evaluation.explain_errors import forest_contributions
from sklearn.ensemble import RandomForestClassifier

class EvaluationTests(unittest.TestCase):
    def test_tree_path_contributions_reconstruct_score(self):
        x=np.array([[0.,0.],[0.,1.],[1.,0.],[1.,1.]])
        forest=RandomForestClassifier(n_estimators=7,random_state=42).fit(x,[0,1,1,0])
        for row in x:
            base,parts=forest_contributions(forest,row.reshape(1,-1))
            self.assertAlmostEqual(base+parts.sum(),forest.predict_proba([row])[0,1])
    def test_hand_calculated_confusion(self):
        m=metrics([0,0,0,1,1],[.1,.2,.8,.4,.9])
        self.assertEqual([m[k] for k in ['TN','FP','FN','TP']],[2,1,1,1])
        self.assertAlmostEqual(m['accuracy'],.6);self.assertAlmostEqual(m['FPR'],1/3)
        self.assertAlmostEqual(m['F1'],.5);self.assertAlmostEqual(m['macro_F1'],(2/3+.5)/2)
    def test_equal_group_weight_and_ties(self):
        inv,w=group_weights(['a','a','b'],True)
        self.assertTrue(np.allclose(np.bincount(inv,weights=w),[1,1]))
        self.assertEqual(metrics([0,1],[.5,.5])['TP'],1)
        self.assertEqual(metrics([0,1],[.5,.5])['FP'],1)
    def test_degenerate_and_invalid_inputs(self):
        self.assertTrue(np.isnan(metrics([1,1],[.1,.2])['AUROC']))
        self.assertEqual(metrics([0,1],[.1,.2])['precision'],0)
        with self.assertRaises(ValueError):metrics([0,1],[np.nan,.2])
    def test_paired_identical_models_zero_difference(self):
        f=pd.DataFrame({'group_id':['a','a','b','b'],'label':[0,1,0,1],'B_score':[.1,.9,.3,.4],'C_score':[.1,.9,.3,.4]})
        thresholds={k:{'default':.5,'low_FPR':.7} for k in ['B','C']}
        _,d=bootstrap_main(f,thresholds,20)
        self.assertTrue(np.allclose(d[['value','ci_low','ci_high']],0))
    def test_paired_flip_denominator_and_direction(self):
        f=pd.DataFrame({'condition':['x','x'],'group_id':['a','b'],'parent_id':['p','q'],
                        'B_original_score':[.9,.1],'B_edited_score':[.1,.9],
                        'C_original_score':[.9,.1],'C_edited_score':[.1,.9]})
        thresholds={k:{'default':.5,'low_FPR':.7} for k in ['B','C']}
        s=paired_edit_summary(f,thresholds,20)
        self.assertTrue((s.loc[s.metric=='AI_to_human_flip','value']==.5).all())
        self.assertTrue((s.loc[s.metric=='TPR_change','value']==0).all())

if __name__=='__main__': unittest.main()
