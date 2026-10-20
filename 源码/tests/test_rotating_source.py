from types import SimpleNamespace
from pathlib import Path

import numpy as np

from magnetic_app.domain.coil_sources import CoilSources
from magnetic_app.domain.replay import rotation
from magnetic_app.domain.replay import Replay


def test_source_pose_interpolates_rigid_rotation_not_chords():
    offsets = np.tile([1., 0., 0.], (8, 1))
    axes = np.tile([0., 0., 1.], (8, 1))
    payload = dict(positions_mm=np.stack([offsets, -offsets]), currents_a=np.ones((2, 8)),
                   axes=axes, offsets_m=offsets/1000., current_limit_a=30.,
                   array_quaternion=np.array([[0., 0., 0., 1.], [0., 0., 1., 0.]]),
                   anchor_mm=np.array([[0., 0., 0.], [2., 0., 0.]]))
    sources = CoilSources(payload)
    frame = SimpleNamespace(index=0, amount=.5)
    positions, currents, moments = sources.at(frame)
    np.testing.assert_allclose(positions, np.tile([1., 1., 0.], (8, 1)), atol=1e-12)
    np.testing.assert_allclose(sources.axes_at(frame), axes, atol=1e-12)


def test_source_axes_follow_quaternion():
    axes = np.tile([1., 0., 0.], (8, 1))
    payload = dict(positions_mm=np.zeros((2, 8, 3)), currents_a=np.ones((2, 8)),
                   axes=axes, offsets_m=np.zeros((8, 3)), current_limit_a=30.,
                   array_quaternion=np.array([[0., 0., 0., 1.], [0., 0., 1., 0.]]),
                   anchor_mm=np.zeros((2, 3)))
    sources = CoilSources(payload)
    np.testing.assert_allclose(sources.axes_at(SimpleNamespace(index=0, amount=.5)),
                               np.tile([0., 1., 0.], (8, 1)), atol=1e-12)


def test_imported_states_camera_currents_and_coverage_are_exact():
    replay = Replay()
    root = Path(__file__).resolve().parents[2]/'数据'/'20mm转动验证'
    for phase, start in [('insertion', 0), ('withdrawal', replay.payload['roundTrip']['join_index'])]:
        with np.load(root/phase/'trajectory.npz') as saved:
            selected = slice(start, start+len(saved['time_s']))
            np.testing.assert_array_equal(replay.positions[selected], saved['state'][:, :3]*1000.)
            np.testing.assert_array_equal(replay.coils.currents[selected], saved['currents'])
            np.testing.assert_array_equal(replay.coils.quaternions[selected], saved['array_quaternion'])
            np.testing.assert_array_equal(replay.camera['joint_angle_deg'][selected], saved['camera_angles'])
            np.testing.assert_array_equal(replay.camera['joint_velocity_deg_s'][selected], saved['camera_velocity'])
            for local_index in np.linspace(0, len(saved['time_s'])-1, 25, dtype=int):
                frame = replay.frame(replay.times[start+local_index])
                positions, currents, moments = replay.coils.at(frame)
                np.testing.assert_allclose(positions, saved['positions'][local_index]*1000., atol=1e-10)
                np.testing.assert_allclose(replay.coils.axes_at(frame), saved['source_axes'][local_index], atol=1e-12)
                np.testing.assert_allclose(replay.vector(frame, 'applied_force', 'mN'), saved['force'][local_index]*1000., atol=1e-12)
    with np.load(root/'insertion_coverage.npz') as saved:
        np.testing.assert_array_equal(replay.insertion_seen, saved['first_seen'])
    with np.load(root/'withdrawal_coverage.npz') as saved:
        expected = np.where(replay.insertion_seen>=0., -1., saved['first_seen'])
        np.testing.assert_array_equal(replay.withdrawal_seen, expected)
    assert replay.payload['magneticSimulation']['clearance_certificate']['full_route_success']
    assert replay.prediction['method'] == 'dense-short-horizon-pd-v1'
    assert replay.robot['magnetic_moment_A_m2'] == replay.robot['config']['magnetic_moment_A_m2']
    assert replay.robot['sample_count'] == len(replay.times)
