import unittest
from scripts.freeze_module_e import select_threshold


class ThresholdTests(unittest.TestCase):
    def test_tied_human_scores_respect_integer_false_positive_budget(self):
        r=select_threshold([0]*20+[1]*3,[.4]*19+[.8]+[.4,.7,.9])
        self.assertEqual(r['false_positives'],1)
        self.assertEqual(r['true_positives'],2)
        self.assertGreater(r['threshold'],.4)

    def test_no_feasible_true_positive_uses_zero_false_positive_threshold(self):
        r=select_threshold([0,0,1],[1.,1.,.5])
        self.assertEqual(r['false_positives'],0)
        self.assertEqual(r['true_positives'],0)


if __name__=='__main__':unittest.main()
