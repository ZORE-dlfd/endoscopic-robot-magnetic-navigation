import unittest

import numpy as np

from withdrawal_trajectory import reverse_reference_arrays, retimed_clock, smooth_time_dilation


class WithdrawalTrajectoryTests(unittest.TestCase):
    def test_reverse_reorders_position_without_flipping_body_orientation(self):
        times = np.array([0.0, 1.0, 2.0])
        positions = np.array([[0., 0., 0.], [1., 0., 0.], [2., 0., 0.]])
        quaternions = np.array([[0., 0., 0., 1.], [0., 0., .0871557, .9961947],
                                [0., 0., .1736482, .9848078]])
        arc = np.array([0.0, 1.0, 2.0])
        result = reverse_reference_arrays(times, positions, quaternions, arc)
        np.testing.assert_allclose(result['time_s'], [0., 1., 2.])
        np.testing.assert_allclose(result['position_mm'], positions[::-1])
        np.testing.assert_allclose(result['arc_s_mm'], arc[::-1])
        self.assertTrue(np.all(np.diff(result['time_s']) > 0))
        dots = np.sum(result['quaternion_xyzw'][1:] * result['quaternion_xyzw'][:-1], axis=1)
        self.assertTrue(np.all(dots > 0))
        np.testing.assert_allclose(result['quaternion_xyzw'][0], quaternions[-1], atol=1e-6)
        np.testing.assert_allclose(result['quaternion_xyzw'][-1], quaternions[0], atol=1e-6)

    def test_slowdown_clock_is_strictly_increasing_and_has_no_holds(self):
        times = np.array([0.0, 0.2, 0.4, 0.6])
        factors = smooth_time_dilation(times, [1.0, 1.2, 1.1, 1.0])
        retimed = retimed_clock(times, factors)
        self.assertTrue(np.all(np.diff(retimed) > 0))
        self.assertLessEqual(float(factors.max()), 1.25)
        self.assertTrue(np.all(factors >= 1.0))


if __name__ == '__main__':
    unittest.main()
