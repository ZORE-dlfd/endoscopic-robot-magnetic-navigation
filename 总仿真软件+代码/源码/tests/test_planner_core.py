import numpy as np
from scipy.spatial.transform import Rotation

from retreat_planner.coverage import Coverage, choose_near_best
from retreat_planner.geometry import cone_angle, viewport_mask
from retreat_planner.controller import State, _advance_gimbal, advance


def test_original_pd_single_step_and_roll_wrap():
    angle, velocity = _advance_gimbal(np.array([0., 0., 179.]), np.zeros(3),
                                       np.array([10., -20., -179.]), .01)
    np.testing.assert_allclose(velocity, [1.2, -1.5, .24], atol=1e-12)
    np.testing.assert_allclose(angle, [.006, -.0075, 179.0012], atol=1e-12)
    angle, velocity = _advance_gimbal(np.array([0., 0., 179.999]), np.array([0., 0., 10.]),
                                       np.array([0., 0., -170.]), .01)
    np.testing.assert_allclose(velocity, [0., 0., 10.60012], atol=1e-12)
    np.testing.assert_allclose(angle, [0., 0., -179.8979994], atol=1e-12)


def test_viewport_uses_true_aspect_and_fov():
    points = np.array([[10, 0, 0], [10, 7, 0], [10, 8, 0], [10, 0, 6], [-10, 0, 0]])
    np.testing.assert_array_equal(viewport_mask(points, np.eye(3), 60, 4 / 3), [1, 1, 0, 0, 0])


def test_coverage_requires_two_visible_endpoints_and_dwell():
    coverage = Coverage(np.array([10., 20.]), np.array([False, True]), .4, .8)
    coverage.observe(np.array([0]), 0, 0)
    for time in [.1, .2, .3]:
        coverage.observe(np.array([0]), time, .1)
    assert coverage.added_area == 0
    coverage.observe(np.array([0]), .4, .1)
    assert coverage.added_area == 10
    coverage.observe(np.array([0, 1]), .5, .1)
    assert coverage.added_area == 10
    assert coverage.total_area == 30


def test_coverage_does_not_credit_first_appearance_interval():
    coverage = Coverage(np.array([10.]), np.array([False]), .1, .8)
    coverage.observe(np.array([], dtype=int), 0, 0)
    coverage.observe(np.array([0]), .1, .1)
    assert coverage.added_area == 0
    coverage.observe(np.array([0]), .2, .1)
    assert coverage.added_area == 10


def test_prediction_clone_cannot_change_real_coverage_or_reward_twice():
    actual = Coverage(np.array([10.]), np.array([False]), .1, .8)
    actual.observe(np.array([0]), 0, 0)
    predicted = actual.clone()
    predicted.observe(np.array([0, 0]), .1, .1)
    predicted.observe(np.array([0]), .2, .1)
    assert predicted.added_area == 10
    assert actual.added_area == 0
    assert actual.first_seen[0] == -1


def test_observation_gap_resets_unqualified_dwell():
    coverage = Coverage(np.array([10.]), np.array([False]), .4, .8)
    coverage.observe(np.array([0]), 0, 0)
    coverage.observe(np.array([0]), .2, .2)
    coverage.observe(np.array([], dtype=int), .3, .1)
    coverage.observe(np.array([0]), 1.2, .1)
    coverage.observe(np.array([0]), 1.4, .2)
    assert coverage.added_area == 0
    coverage.observe(np.array([0]), 1.6, .2)
    assert coverage.added_area == 10


def test_near_best_prefers_smooth_and_rejects_large_loss():
    assert choose_near_best(np.array([100., 96., 90.]), np.array([8., 2., 0.]), .05) == 1
    assert choose_near_best(np.zeros(3), np.array([8., 2., 0.]), .05) == 2


def test_actual_cone_is_checked_between_decisions():
    state = State(np.array([44., 0., 0.]), np.array([50., 0., 0.]), np.array([44., 0., 0.]))
    result = advance(state, np.zeros(3), .4)
    assert result.max_cone_deg > 45
    assert not result.feasible


def test_euler_axis_cone_matches_rotation():
    angles = np.array([[20., 30., 90.], [-10., -20., 0.]])
    forward = Rotation.from_euler('ZYX', angles, degrees=True).as_matrix()[:, :, 0]
    np.testing.assert_allclose(cone_angle(forward), np.rad2deg(np.arccos(np.cos(np.deg2rad(angles[:, 0])) * np.cos(np.deg2rad(angles[:, 1])))))
