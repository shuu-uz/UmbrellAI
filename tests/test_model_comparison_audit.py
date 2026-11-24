import unittest
from model_comparison_audit import paired_brier_interval


class PairedBootstrapChecks(unittest.TestCase):
    def test_identical_predictions_have_zero_difference(self):
        r = paired_brier_interval([0,1,0,1], [.2,.8,.3,.7], [.2,.8,.3,.7], [0,0,1,1])
        self.assertEqual((r['delta'],r['low'],r['high']), (0,0,0))

    def test_two_scenarios_are_resampled_as_one_respondent(self):
        # Each respondent's two errors cancel in the paired difference.
        r = paired_brier_interval([0,1,0,1], [0,0,0,0], [1,1,1,1], [0,0,1,1])
        self.assertEqual((r['delta'],r['low'],r['high']), (0,0,0))
        self.assertEqual(r['respondents'],2)

    def test_unpaired_or_incomplete_groups_are_rejected(self):
        with self.assertRaises(ValueError):
            paired_brier_interval([0,1], [0], [0,1], [0,0])
        with self.assertRaises(ValueError):
            paired_brier_interval([0,1,0], [0,1,0], [0,1,0], [0,0,1])


if __name__ == '__main__':
    unittest.main()
