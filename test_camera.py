import unittest

import pandas as pd

import numpy as np
from scipy.spatial.transform import Rotation

from camera_model import BASE, SETTINGS, RetimedReference, bounded_gimbal, forward_frame, target_frames
from evaluation_core import CoverageAccumulator


class CameraTests(unittest.TestCase):
    def test_forward_target_rejection_keeps_centerline_lookahead(self):
        source = pd.read_csv(BASE / 'source_trajectory_state.csv')
        replay = pd.read_csv(BASE / 'capsule_replay.csv')
        selected = replay.iloc[[np.argmin(abs(replay.time_s.to_numpy() - time)) for time in (71.4, 71.6)]]

        result = target_frames(selected, source)

        self.assertAlmostEqual(result['preview_mm'][0], 10.8, places=1)
        self.assertGreater(result['preview_mm'][1], 10.0)
        for index, (_, row) in enumerate(selected.iterrows()):
            target_arc = row.arc_s_mm + result['preview_mm'][index]
            expected = np.array([np.interp(target_arc, source.arc_s_mm, source[f'{axis}_mm']) for axis in 'xyz'])
            np.testing.assert_allclose(result['target'][index], expected, atol=1e-6)
    def test_initial_rejected_target_stays_on_centerline(self):
        source = pd.read_csv(BASE / 'source_trajectory_state.csv')
        replay = pd.read_csv(BASE / 'capsule_replay.csv')
        selected = replay.iloc[[np.argmin(abs(replay.time_s.to_numpy() - 71.6))]]

        result = target_frames(selected, source)
        target_arc = selected.iloc[0].arc_s_mm + result['preview_mm'][0]
        expected = np.array([np.interp(target_arc, source.arc_s_mm, source[f'{axis}_mm']) for axis in 'xyz'])

        np.testing.assert_allclose(result['target'][0], expected, atol=1e-6)

    def test_lookahead_target_persists_when_alignment_falls_below_hold_threshold(self):
        source = pd.read_csv(BASE / 'source_trajectory_state.csv')
        replay = pd.read_csv(BASE / 'capsule_replay.csv')
        selected = replay.iloc[[np.argmin(abs(replay.time_s.to_numpy() - time)) for time in (71.4, 71.6)]].copy()
        quaternion_columns = [f'actual_q{axis}' for axis in 'xyzw']
        second_rotation = Rotation.from_quat(selected.iloc[1][quaternion_columns].to_numpy(float))
        selected.loc[selected.index[1], quaternion_columns] = (second_rotation * Rotation.from_euler('y', 8, degrees=True)).as_quat()

        result = target_frames(selected, source)

        self.assertAlmostEqual(result['preview_mm'][1], result['preview_mm'][0], places=1)
        target_arc = selected.iloc[1].arc_s_mm + result['preview_mm'][1]
        expected = np.array([np.interp(target_arc, source.arc_s_mm, source[f'{axis}_mm']) for axis in 'xyz'])
        np.testing.assert_allclose(result['target'][1], expected, atol=1e-6)
    def test_gimbal_joint_limits_cover_forward_quadrant_travel(self):
        self.assertEqual(SETTINGS['max_joint_angle_deg'], 89.)
        times = np.arange(0, 10, .02)
        goals = np.tile([80., 75., 0.], (len(times), 1))
        angles, velocities, accelerations = bounded_gimbal(times, goals)
        self.assertGreater(np.max(angles[:, 0]), 60.)
        self.assertGreater(np.max(angles[:, 1]), 60.)
        self.assertLessEqual(np.max(np.abs(angles[:, :2])), SETTINGS['max_joint_angle_deg'] + 1e-9)
        self.assertLessEqual(np.max(np.abs(velocities)), SETTINGS['max_joint_speed_deg_s'] + 1e-9)
        self.assertLessEqual(np.max(np.abs(accelerations)), SETTINGS['max_joint_acceleration_deg_s2'] + 1e-9)
        camera_forward = Rotation.from_euler('ZYX', angles, degrees=True).as_matrix()[:, :, 0]
        self.assertTrue(np.all(camera_forward[:, 0] >= -1e-8))
    def test_camera_tracks_turning_centerline_without_fov_crossing_rear(self):
        source = pd.read_csv(BASE / 'source_trajectory_state.csv')
        replay = pd.read_csv(BASE / 'capsule_replay.csv')
        result = target_frames(replay, source)
        index = int(np.argmin(abs(replay.time_s.to_numpy() - 71.6)))

        self.assertLess(result['aim_error_deg'][index], 20.)
        camera = Rotation.from_quat(result['quaternion'][index]).as_matrix()
        body = Rotation.from_quat(replay.iloc[index][[f'actual_q{axis}' for axis in 'xyzw']].to_numpy(float)).as_matrix()
        half_vertical = np.tan(np.deg2rad(SETTINGS['fov_vertical_deg'] / 2))
        half_horizontal = half_vertical * SETTINGS['aspect']
        corner_cosines = []
        for side in (-1, 1):
            for height in (-1, 1):
                ray = camera[:, 0] + side * half_horizontal * camera[:, 1] + height * half_vertical * camera[:, 2]
                corner_cosines.append(np.dot(ray, body[:, 0]) / np.linalg.norm(ray))
        self.assertGreater(min(corner_cosines), 0.)
    def test_retiming_preserves_geometry_and_hold(self):
        class Original:
            def at(self, time):
                return {'position':np.array([time,time*time,0]),'velocity':np.array([1,2*time,0]),'acceleration':np.array([0,2,0]),'omega_body':np.ones(3),'alpha_body':np.zeros(3),'rotation':Rotation.identity()}
        reference=RetimedReference(Original(),[0,1,2,3,4],[0,1,1,2,3])
        held=reference.at(1.5)
        np.testing.assert_allclose(held['position'],[1,1,0])
        np.testing.assert_allclose(held['velocity'],[0,0,0])
        for time in np.linspace(0,4,30):
            state=reference.at(time)
            self.assertAlmostEqual(state['position'][1],state['position'][0]**2)

    def test_rotation_frame(self):
        frame = forward_frame([1, 0, 0], [0, 0, 1])
        np.testing.assert_allclose(frame.T @ frame, np.eye(3), atol=1e-12)
        self.assertAlmostEqual(np.linalg.det(frame), 1)

    def test_gimbal_limits(self):
        times = np.arange(0, 4, .02)
        goals = np.tile([70., -60., 120.], (len(times), 1))
        angles, velocities, accelerations = bounded_gimbal(times, goals)
        self.assertLessEqual(np.max(np.abs(angles[:, :2])), SETTINGS['max_joint_angle_deg'] + 1e-9)
        self.assertLessEqual(np.max(np.abs(velocities)), 60 + 1e-9)
        self.assertLessEqual(np.max(np.abs(accelerations)), 150 + 1e-9)
        forward = Rotation.from_euler('ZYX', angles, degrees=True).as_matrix()[:, :, 0]
        self.assertTrue(np.all(forward[:, 0] >= -1e-8))

    def test_area_and_repeat(self):
        counter = CoverageAccumulator(np.array([1., 3.]), dwell=.4, gap=.5)
        self.assertIsNone(counter.update([], 0., .2)['repeat_pct'])
        counter.update([0], .2, .2)
        result = counter.update([0], .4, .2)
        self.assertEqual(result['coverage_pct'], 25)
        self.assertEqual(result['unseen_pct'], 75)
        self.assertEqual(result['repeat_pct'], 0)
        counter.update([], 1.1, .2)
        counter.update([0], 1.3, .2)
        result = counter.update([0], 1.5, .2)
        self.assertEqual(result['repeat_pct'], 50)


if __name__ == '__main__':
    unittest.main()
