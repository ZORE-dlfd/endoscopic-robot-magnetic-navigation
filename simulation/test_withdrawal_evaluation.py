import unittest

import pandas as pd

from evaluation_core import CoverageAccumulator
from evaluate_withdrawal import build_summary
from build_camera_viewer import roundtrip_coverage_rows


class WithdrawalEvaluationTests(unittest.TestCase):
    def test_existing_effective_dwell_threshold_qualifies_at_point_four_seconds(self):
        counter = CoverageAccumulator([3.0], dwell=.4, gap=.8)
        counter.update([0], 0.0, 0.2)
        result = counter.update([0], 0.4, 0.2)
        self.assertAlmostEqual(result['coverage_pct'], 100.0)

    def test_fold_hidden_faces_remain_uncovered_in_full_area_statistics(self):
        masks = {'recoverable': [True, False, True],
                 'fold_hidden': [False, True, False],
                 'insertion_covered': [True, False, False],
                 'insertion_missed': [False, False, True]}
        result = build_summary([2., 5., 7.], masks, [1., 1., 2.], 5.0)
        self.assertEqual(result['insertion_covered_area_mm2'], 2.0)
        self.assertEqual(result['withdrawal_added_area_mm2'], 7.0)
        self.assertEqual(result['fold_hidden_area_mm2'], 5.0)
        self.assertEqual(result['total_lumen_area_mm2'], 14.0)
        self.assertAlmostEqual(result['insertion_coverage_pct'], 100.0 * 2 / 14)
        self.assertAlmostEqual(result['final_coverage_pct'], 100.0 * 9 / 14)
        self.assertAlmostEqual(result['uncovered_area_mm2'], 5.0)
        self.assertTrue(result['coverage_increased'])

    def test_roundtrip_coverage_uses_full_wall_area_and_starts_with_withdrawal_at_handoff(self):
        rows, insertion_area = roundtrip_coverage_rows(
            times=pd.Series([0.0, 1.0, 2.0, 3.0]),
            arc=pd.Series([0.0, 10.0, 10.0, 0.0]),
            areas=[2.0, 5.0, 7.0],
            insertion_first_seen=[0.0, -1.0, -1.0],
            recoverable=[True, False, True],
            withdrawal_first_seen=[-1.0, 0.0, 0.5],
            join_time=2.0)

        self.assertEqual(insertion_area, 2.0)
        self.assertEqual(rows[1]['withdrawal_added_area_mm2'], 0.0)
        self.assertEqual(rows[2]['withdrawal_added_area_mm2'], 0.0)
        self.assertEqual(rows[3]['withdrawal_added_area_mm2'], 7.0)
        self.assertAlmostEqual(rows[3]['coverage_pct'], 100.0 * 9 / 14)


if __name__ == '__main__':
    unittest.main()
