import unittest

import numpy as np
from scipy.spatial.transform import Rotation

from coordination import build_schedule, classify_orientation_error, estimate_settle_time, prune_unsettled_waits, speed_dilation


class CoordinationTests(unittest.TestCase):
    def test_replay_rejects_wait_that_does_not_actually_settle(self):
        camera = {'time_s': [0., 1., 2., 3., 4., 5.],
                  'goal_joint_angle_deg': [[30., 0., 0.]] * 6,
                  'joint_angle_deg': [[0., 0., 0.], [0., 0., 0.], [15., 0., 0.],
                                      [0., 0., 0.], [29., 0., 0.], [29., 0., 0.]],
                  'joint_velocity_deg_s': [[0., 0., 0.]] * 6}
        waits = [{'start_s': 1., 'duration_s': 1.}, {'start_s': 3., 'duration_s': 1.}]
        times, source, kept, rejected = prune_unsettled_waits(
            [0., 1., 2., 3., 4., 5.], [0., 1., 1., 2., 2., 3.], waits, camera)
        np.testing.assert_allclose(times, [0., 1., 2., 3., 4.])
        np.testing.assert_allclose(source, [0., 1., 2., 2., 3.])
        self.assertEqual(len(kept), 1)
        self.assertEqual(kept[0]['start_s'], 2.)
        self.assertEqual(len(rejected), 1)
        self.assertGreater(rejected[0]['replay_end_axis_error_deg'], 3.)

    def test_replay_rejects_wait_when_no_actual_lag_needs_it(self):
        camera = {'time_s': [0., 1., 2., 3.],
                  'goal_joint_angle_deg': [[0., 0., 0.]] * 4,
                  'joint_angle_deg': [[0., 0., 0.]] * 4,
                  'joint_velocity_deg_s': [[0., 0., 0.]] * 4}
        _, _, kept, rejected = prune_unsettled_waits(
            [0., 1., 2., 3.], [0., 1., 1., 2.], [{'start_s': 1., 'duration_s': 1.}], camera)
        self.assertEqual(kept, [])
        self.assertEqual(len(rejected), 1)

    def test_replay_cannot_accept_an_interpolated_wait_endpoint(self):
        camera = {'time_s': [0., 1., 2., 3.],
                  'goal_joint_angle_deg': [[30., 0., 0.]] * 4,
                  'joint_angle_deg': [[0., 0., 0.], [0., 0., 0.], [29., 0., 0.], [29., 0., 0.]],
                  'joint_velocity_deg_s': [[0., 0., 0.]] * 4}
        with self.assertRaises(ValueError):
            prune_unsettled_waits([0., 1., 1.9, 3.], [0., 1., 1., 2.],
                                 [{'start_s': 1., 'duration_s': .9}], camera)

    def test_unreachable_target_error_does_not_trigger_slowdown(self):
        desired = Rotation.identity().as_quat()
        actual = Rotation.identity().as_quat()
        components = classify_orientation_error(
            target_direction=[0, 1, 0],
            feasible_direction=[1, 0, 0],
            desired_quaternion=desired,
            actual_quaternion=actual,
        )

        self.assertAlmostEqual(components['cone_error_deg'], 90.0)
        self.assertAlmostEqual(components['tracking_error_deg'], 0.0)
        self.assertEqual(speed_dilation(components['tracking_error_deg'], 0.0), 1.0)

    def test_resolvable_tracking_lag_triggers_bounded_slowdown(self):
        factor = speed_dilation(tracking_error_deg=30.0, desired_rate_deg_s=20.0)

        self.assertGreater(factor, 1.0)
        self.assertLessEqual(factor, 1.25)
        self.assertEqual(speed_dilation(tracking_error_deg=3.0, desired_rate_deg_s=20.0), 1.0)

    def test_moderate_lag_does_not_slow_the_entire_path(self):
        self.assertEqual(speed_dilation(12.0, 30.0), 1.0)
        self.assertLessEqual(speed_dilation(120.0, 300.0), 1.25)

    def test_wait_budget_prevents_repeated_long_holds(self):
        times = np.arange(30, dtype=float)
        goals = np.zeros((len(times), 3))
        goals[:, 0] = np.where(np.arange(len(times)) % 4 < 2, -65.0, 65.0)
        _, _, waits = build_schedule(times, goals, np.ones(len(times)),
                                      add_feedback_waits=True, max_total_wait_s=0.9,
                                      lag_trigger_deg=12.0, cooldown_s=0.0)
        self.assertTrue(waits)
        self.assertLessEqual(sum(wait['duration_s'] for wait in waits), 0.9 + 1e-9)
        self.assertTrue(all(wait['lag_after_deg'] <= 6.0 for wait in waits))

    def test_zero_wait_budget_keeps_progress(self):
        times = np.arange(6, dtype=float)
        goals = np.tile([70.0, 0.0, 0.0], (len(times), 1))
        goals[0] = 0.0
        new_times, _, waits = build_schedule(times, goals, np.ones(len(times)),
                                            add_feedback_waits=True, max_total_wait_s=0.0)
        self.assertEqual(waits, [])
        self.assertAlmostEqual(new_times[-1], times[-1])

    def test_wait_is_state_based_and_only_for_reachable_pose(self):
        settled = estimate_settle_time([0, 0, 0], [0, 0, 0], [0, 0, 0])
        reachable = estimate_settle_time([20, 0, 0], [0, 0, 0], [0, 0, 0])
        unreachable = estimate_settle_time([0, 0, 0], [0, 0, 0], [95, 0, 0])

        self.assertEqual(settled, 0.0)
        self.assertGreater(reachable, 0.0)
        self.assertLessEqual(reachable, 1.2)
        self.assertIsNone(unreachable)

    def test_wait_requires_pose_settling_not_only_small_angle_error(self):
        moving = estimate_settle_time([0, 0, 0], [15, 0, 0], [0, 0, 0])

        self.assertGreater(moving, 0.0)

    def test_feedback_wait_is_added_only_when_measured_joint_state_lags(self):
        times = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
        goals = np.array([[0, 0, 0], [70, 0, 0], [70, 0, 0], [70, 0, 0], [70, 0, 0]], dtype=float)
        _, _, waits = build_schedule(times, goals, np.ones(len(times)), add_feedback_waits=True,
                                      lag_trigger_deg=12.0, max_total_wait_s=1.2)

        self.assertTrue(waits)
        self.assertLessEqual(waits[0]['lag_after_deg'], 3.0)

    def test_no_wait_is_added_when_camera_is_already_tracking(self):
        times = np.arange(5, dtype=float)
        goals = np.tile([0.0, 0.0, 0.0], (len(times), 1))
        _, _, waits = build_schedule(times, goals, np.ones(len(times)), add_feedback_waits=True)

        self.assertEqual(waits, [])


if __name__ == '__main__':
    unittest.main()
