import unittest

import numpy as np

from evaluation_core import CoverageAccumulator, evaluation_methods, evaluation_times


class EvaluationTests(unittest.TestCase):
    def test_all_retimed_endoscopy_baselines_are_evaluated(self):
        methods = evaluation_methods()
        self.assertEqual(len(methods), 8)
        for mode in ['fixed', 'slowdown', 'wait', 'coordinated']:
            self.assertIn(mode + '_new', methods)
            self.assertIn(mode + '_endoscopy', methods)

    def test_common_budget_is_an_exact_sampling_instant(self):
        times = evaluation_times(411.9, 286.99953, .4)
        self.assertIn(286.99953, times)
        self.assertEqual(times[-1], 411.9)
        self.assertTrue(np.all(np.diff(times) > 0))

    def test_area_weight_and_unseen_complement(self):
        counter = CoverageAccumulator([1, 3], dwell=.4, gap=.8)
        self.assertIsNone(counter.update([], 0, .2)['repeat_pct'])
        counter.update([0], .2, .2)
        covered = counter.update([0], .4, .2)
        self.assertAlmostEqual(covered['coverage_pct'], 25)
        self.assertAlmostEqual(covered['unseen_pct'], 75)
        self.assertAlmostEqual(covered['coverage_pct']+covered['unseen_pct'],100)

    def test_repeat_requires_distinct_observation_segment(self):
        counter = CoverageAccumulator([2], dwell=.2, gap=.8)
        self.assertEqual(counter.update([0], 0, .2)['repeat_pct'], 0)
        self.assertEqual(counter.update([0], .4, .2)['repeat_pct'], 0)
        counter.update([], .8, .2)
        repeated = counter.update([0], 1., .2)
        self.assertEqual(repeated['repeat_pct'], 0)
        counter.update([], 2., .2)
        counter.update([0], 2.2, .2)
        self.assertEqual(counter.update([0], 2.4, .2)['repeat_pct'], 50)


if __name__ == '__main__':
    unittest.main()
