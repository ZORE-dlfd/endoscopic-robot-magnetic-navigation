import unittest

import numpy as np

from withdrawal_visibility import camera_candidate_directions, surface_masks, unique_new_area


class WithdrawalVisibilityTests(unittest.TestCase):
    def test_hidden_faces_are_excluded_but_recoverable_insert_misses_remain(self):
        visibility = np.array([[True, False, True], [False, False, True]])
        masks = surface_masks([2.0, 5.0, 7.0], visibility, [0.0, -1.0, -1.0])
        np.testing.assert_array_equal(masks['recoverable'], [True, False, True])
        np.testing.assert_array_equal(masks['fold_hidden'], [False, True, False])
        np.testing.assert_array_equal(masks['insertion_missed'], [False, False, True])

    def test_withdrawal_adds_only_unique_recoverable_missed_area(self):
        result = unique_new_area([2.0, 5.0, 7.0], [True, False, True],
                                 [True, False, False], [True, True, True])
        self.assertAlmostEqual(result['insertion_covered_area_mm2'], 2.0)
        self.assertAlmostEqual(result['withdrawal_added_area_mm2'], 7.0)
        self.assertAlmostEqual(result['final_covered_area_mm2'], 9.0)
        np.testing.assert_array_equal(result['withdrawal_new_mask'], [False, False, True])

    def test_all_camera_candidates_stay_inside_forward_cone(self):
        candidates = camera_candidate_directions(np.eye(3), max_angle_deg=45.0)
        dots = candidates @ np.array([1.0, 0.0, 0.0])
        self.assertEqual(len(candidates), 17)
        self.assertTrue(np.all(dots >= np.cos(np.deg2rad(45.0)) - 1e-12))

    def test_recoverability_grid_samples_forward_cone_rings(self):
        candidates = camera_candidate_directions(np.eye(3), max_angle_deg=45.0,
                                                 ring_angles_deg=(15., 30., 45.), azimuth_count=12)
        self.assertEqual(len(candidates), 37)
        self.assertAlmostEqual(float(np.max(np.rad2deg(np.arccos(candidates[:, 0])))), 45.0)


if __name__ == '__main__':
    unittest.main()
