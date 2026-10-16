from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from magnetic_app.domain.replay import Replay, multiply, conjugate, rotation, slerp


@pytest.fixture(scope='module')
def replay():
    return Replay()


def test_antipodal_quaternions():
    quaternion = np.array([0., 0., 0., 1.])
    np.testing.assert_allclose(slerp(quaternion, -quaternion, .5), quaternion)


def test_rotation_and_inverse():
    quaternion = np.array([.2, -.3, .4, .5])
    quaternion /= np.linalg.norm(quaternion)
    matrix = rotation(quaternion)
    np.testing.assert_allclose(matrix.T @ matrix, np.eye(3), atol=1e-12)
    np.testing.assert_allclose(multiply(quaternion, conjugate(quaternion)), [0, 0, 0, 1], atol=1e-12)


def test_continuous_robot_and_camera_mount(replay):
    for time in [32.123, 300.01, 404.123, 633.999]:
        pose = replay.frame(time)
        next_pose = replay.frame(time + .001)
        assert np.linalg.norm(pose.position - next_pose.position) < .05
        np.testing.assert_allclose(pose.camera_position, pose.position + pose.body_rotation[:, 0] * 5.8, atol=1e-9)
        np.testing.assert_allclose(pose.camera_rotation.T @ pose.camera_rotation, np.eye(3), atol=1e-10)
        assert np.linalg.norm(pose.camera_rotation - next_pose.camera_rotation) < .03


def test_recorded_pose_knots_preserved(replay):
    for index in [0, 100, 1500, 2000, len(replay.times) - 1]:
        frame = replay.frame(replay.times[index])
        np.testing.assert_allclose(frame.position, replay.positions[index], atol=1e-10)
    for index in [0, 50, 1600, 2000, len(replay.camera_times) - 1]:
        frame = replay.frame(replay.camera_times[index])
        np.testing.assert_allclose(frame.camera_rotation, rotation(replay.camera_quaternions[index]), atol=1e-9)


def test_disjoint_coverage_exact_area(replay):
    insertion, withdrawal = replay.coverage(replay.duration)
    assert not np.any(insertion & withdrawal)
    assert insertion.sum() > 0 and withdrawal.sum() > 0
    assert replay.areas[insertion].sum() == pytest.approx(177090.4177645445, abs=.01)
    assert replay.areas[withdrawal].sum() == pytest.approx(20909.910963773727, abs=.01)
    assert replay.areas.sum() == pytest.approx(246417.473972328, abs=.01)
    assert not replay.coverage(-1)[0].any()
    assert not replay.coverage(replay.join_time - 1e-7)[1].any()


def test_clamping_and_metadata(replay):
    assert replay.frame(-100).time == 0
    assert replay.frame(10000).time == replay.duration
    assert 'Colon001' in replay.payload['geometry']['surface_notice']
    assert len(replay.faces) == 33696


def test_same_time_field_formula(replay):
    frame = replay.frame(400.123)
    points = np.array([frame.position, frame.position + [10, 20, 30]])
    field = replay.field(frame, points, 'total')
    np.testing.assert_allclose(field[0], replay.vector(frame, 'applied_field', 'mT'))
    np.testing.assert_allclose(field[1] - field[0], replay.gradient(frame) @ np.array([10, 20, 30]))


def test_all_recorded_columns_readable(replay):
    frame = replay.frame(410.123)
    for key in replay.data:
        value = replay.value(frame, key)
        assert isinstance(value, (str, float, int))
    assert isinstance(replay.value(frame, 'phase'), str)
