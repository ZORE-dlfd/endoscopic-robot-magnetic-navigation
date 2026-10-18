from pathlib import Path
import sys

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT/'源码'),str(ROOT/'构建'/'依赖')]

from magnetic_app.domain.replay import Replay


@pytest.fixture(scope='module')
def replay():
    return Replay()


def test_complete_route_has_independent_endpoint_checks(replay):
    report = replay.payload['magneticSimulation']
    assert report['coil_count'] == 8 and report['no_position_forcing']
    for phase in report['reports']:
        assert phase['complete'] and phase['route_arc_unchanged']
        assert phase['endpoint_distance_mm'] < 1.
        assert phase['route_length_mm']-phase['progress_mm'] < .151
        assert phase['peak_current_a'] <= 10.+1e-7


def test_recorded_motion_uses_source_field_force_and_torque(replay):
    config = replay.robot['config']
    for index in np.linspace(0,len(replay.times)-1,50,dtype=int):
        frame = replay.frame(replay.times[index])
        field = replay.vector(frame, 'applied_field', 'mT')/1000.
        gradient = np.array([[replay.value(frame, f'applied_G_{first}{second}_T_m')
                              for second in 'xyz'] for first in 'xyz'])
        moment = config['magnetic_moment_A_m2']*frame.body_rotation[:,0]
        force = replay.vector(frame,'applied_force','mN')/1000.
        torque = replay.vector(frame,'applied_torque','uN_m')/1e6
        np.testing.assert_allclose(force,gradient.T@moment,atol=1e-12)
        np.testing.assert_allclose(torque,np.cross(moment,field),atol=1e-12)
        resistance = replay.vector(frame,'applied_resistance','mN')/1000.
        acceleration = replay.vector(frame,'actual_acceleration','mm_s2')/1000.
        np.testing.assert_allclose(acceleration,(force+resistance)/config['mass_kg']+config['gravity_m_s2'],atol=1e-10)


def test_source_motion_and_reversal_are_continuous(replay):
    anchor_velocity = np.diff(replay.coils.positions[:,0],axis=0)/np.diff(replay.times)[:,None]/1000.
    assert np.max(np.linalg.norm(anchor_velocity,axis=1)) <= .2+1e-8
    before,after = replay.frame(replay.join_time-1e-6),replay.frame(replay.join_time+1e-6)
    assert np.linalg.norm(after.position-before.position) < .001
    assert np.linalg.norm(after.camera_rotation-before.camera_rotation) < .001
    assert np.max(np.abs(np.diff(replay.positions,axis=0))) < 5.
    assert not np.any(replay.data['camera_wait_planned'])


def test_withdrawal_really_replans_dense_short_horizon(replay):
    prediction = replay.prediction
    assert prediction['method'] == 'dense-short-horizon-pd-v1'
    assert prediction['config']['decision_s'] == .4
    assert prediction['config']['horizon_s'] == 1.2
    assert prediction['config']['gain_tolerance'] == .05
    assert prediction['config']['integration_s'] == .01
    decisions = prediction['decisions']
    assert len(decisions) > 100
    assert min(row['searched_direction_count'] for row in decisions) > 37
    assert prediction['metrics']['constraint_violating_steps'] == 0
    assert prediction['metrics']['prediction_fallback_count'] == 0


def test_supine_posture_keeps_world_gravity_and_array_above_capsule(replay):
    posture = replay.payload['patientPosture']
    assert posture['posture'] == 'supine' and not posture['contact_model']
    np.testing.assert_allclose(replay.robot['config']['gravity_m_s2'], [0., 0., -9.81])
    np.testing.assert_allclose(posture['gravity_m_s2'], [0., 0., -9.81])
    span = np.ptp(replay.points, axis=0)
    assert span[1] > span[2] and replay.points[:, 2].min() == pytest.approx(20.)
    assert np.all(replay.coils.positions[:, :, 2]-replay.positions[:, None, 2] > 40.)
    assert np.min(replay.coils.offsets[:, 2])*1000. == pytest.approx(48.75)
    assert np.all(replay.payload['coilSources']['axes'][:, 2] < 0.)
    assert replay.payload['magneticSimulation']['patient_posture'] == posture


def test_supine_geometry_is_rigidly_transformed_reference(replay):
    original = Replay(ROOT/'数据'/'参考原版')
    posture = replay.payload['patientPosture']
    rotation = np.asarray(posture['rotation_from_original'])
    translation = np.asarray(posture['translation_mm'])
    np.testing.assert_allclose(replay.points, original.points@rotation.T+translation)
    np.testing.assert_array_equal(replay.areas, original.areas)
    np.testing.assert_array_equal(replay.faces, original.faces)
    np.testing.assert_array_equal(replay.source['risk_norm'], original.source['risk_norm'])


def test_roundtrip_starts_at_anus_reaches_deep_end_and_returns(replay):
    original = Replay(ROOT/'数据'/'参考原版')
    posture = replay.payload['patientPosture']
    rotation = np.asarray(posture['rotation_from_original'])
    translation = np.asarray(posture['translation_mm'])
    anus = original.centerline[0]@rotation.T+translation
    deep = original.centerline[-1]@rotation.T+translation
    start = replay.frame(0.)
    turn = replay.frame(replay.join_time)
    finish = replay.frame(replay.duration)
    assert np.linalg.norm(start.position-anus) < 1., 'Insertion must start at the original anal entry'
    assert np.linalg.norm(turn.position-deep) < 1., 'Withdrawal must start at the deep endpoint'
    assert np.linalg.norm(finish.position-anus) < 1., 'Withdrawal must finish at the anal entry'
    tangent = replay.centerline[1]-replay.centerline[0]
    tangent /= np.linalg.norm(tangent)
    assert np.dot(start.body_rotation[:, 0], tangent) > .99
    direction = replay.payload['routeDirection']
    assert direction['insertion_start_label'] == '肛门入口'
    assert direction['withdrawal_end_label'] == '肛门入口'
    assert direction['verification']['passed']
