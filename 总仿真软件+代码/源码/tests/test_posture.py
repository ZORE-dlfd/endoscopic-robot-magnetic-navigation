from pathlib import Path
import sys

import numpy as np
import pandas as pd
import pytest
from scipy.spatial.transform import Rotation

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT/'源码'), str(ROOT/'构建'/'依赖')]

from magnetic_app.domain.replay import Replay


@pytest.fixture(scope='module')
def reference():
    return Replay(ROOT/'数据'/'参考原版')


def test_supine_transform_lays_long_axis_on_horizontal_plane(reference):
    from coil_simulation.posture import SupineTransform
    transform = SupineTransform.from_points(reference.points)
    points = transform.positions(reference.points)
    np.testing.assert_allclose(transform.rotation.T@transform.rotation, np.eye(3), atol=1e-12)
    assert np.linalg.det(transform.rotation) == pytest.approx(1.)
    np.testing.assert_allclose(np.ptp(points, axis=0), np.ptp(reference.points, axis=0)[[0, 2, 1]])
    assert points[:, 2].min() == pytest.approx(20.)
    np.testing.assert_allclose(transform.vectors([0., 0., 1.]), [0., -1., 0.])
    np.testing.assert_allclose(transform.metadata()['gravity_m_s2'], [0., 0., -9.81])
    assert not np.allclose(transform.vectors([0., 0., -9.81]), [0., 0., -9.81])


def test_reference_geometry_route_and_camera_stay_aligned(reference):
    from coil_simulation.posture import SupineTransform
    transform = SupineTransform.from_points(reference.points)
    payload = transform.payload(reference.payload)
    geometry = payload['geometry']
    np.testing.assert_allclose(geometry['points'], transform.positions(reference.points))
    np.testing.assert_array_equal(geometry['areas_mm2'], reference.areas)
    np.testing.assert_array_equal(geometry['faces'], reference.faces)
    np.testing.assert_allclose(geometry['normals'], transform.vectors(reference.payload['geometry']['normals']))
    for prefix in ['', 't', 'n', 'b']:
        keys = [axis+'_mm' if not prefix else prefix+axis for axis in 'xyz']
        before = np.column_stack([reference.source[key] for key in keys])
        after = np.column_stack([payload['source'][key] for key in keys])
        np.testing.assert_allclose(after, transform.positions(before) if not prefix else transform.vectors(before))
    camera = payload['cameras']['roundtrip_new']
    np.testing.assert_allclose(camera['position'], transform.positions(reference.camera['position']))
    np.testing.assert_array_equal(camera['joint_angle_deg'], reference.camera['joint_angle_deg'])
    rotations = Rotation.from_quat(camera['quaternion']).as_matrix()
    np.testing.assert_allclose(rotations, transform.rotation@Rotation.from_quat(reference.camera['quaternion']).as_matrix(), atol=1e-12)
    np.testing.assert_array_equal(reference.points, reference.payload['geometry']['points'])
    repeated = transform.payload(reference.payload)
    np.testing.assert_array_equal(repeated['geometry']['points'], geometry['points'])
    with pytest.raises(ValueError, match='already'):
        transform.payload(payload)


def test_source_csv_transform_preserves_input_route(reference):
    from coil_simulation.posture import SupineTransform
    transform = SupineTransform.from_points(reference.points)
    source = pd.read_csv(ROOT/'数据'/'仿真输入'/'trajectory_state.csv')
    result = transform.source(source)
    payload = transform.payload(reference.payload)
    for key in ['arc_s_mm', 'risk_norm', 'curvature_inv_mm']:
        np.testing.assert_array_equal(result[key], source[key])
    np.testing.assert_allclose(result[['x_mm', 'y_mm', 'z_mm']], transform.positions(source[['x_mm', 'y_mm', 'z_mm']]))
    np.testing.assert_allclose(result[['qx', 'qy', 'qz', 'qw']], transform.quaternions(source[['qx', 'qy', 'qz', 'qw']]))
    np.testing.assert_allclose(result[['x_mm', 'y_mm', 'z_mm']], np.column_stack([payload['source'][axis+'_mm'] for axis in 'xyz'])[::-1], atol=1e-8)
    np.testing.assert_allclose(result[['vx_mm_s', 'vy_mm_s', 'vz_mm_s']], transform.vectors(source[['vx_mm_s', 'vy_mm_s', 'vz_mm_s']]))


def test_world_gradient_rotates_but_body_angular_rates_do_not(reference):
    from coil_simulation.posture import SupineTransform
    transform = SupineTransform.from_points(reference.points)
    result = transform.payload(reference.payload)['datasets']['capsule_roundtrip']
    indices = [0, 10, 100]
    before = np.array([[reference.data[f'applied_G_{first}{second}_T_m'][indices] for second in 'xyz'] for first in 'xyz']).transpose(2, 0, 1)
    after = np.array([[result[f'applied_G_{first}{second}_T_m'][indices] for second in 'xyz'] for first in 'xyz']).transpose(2, 0, 1)
    np.testing.assert_allclose(after, transform.rotation@before@transform.rotation.T, atol=1e-12)
    for axis in 'xyz':
        np.testing.assert_array_equal(result[f'actual_omega_body_{axis}_rad_s'], reference.data[f'actual_omega_body_{axis}_rad_s'])


def test_transform_rejects_nonfinite_or_empty_geometry():
    from coil_simulation.posture import SupineTransform
    for points in [np.empty((0, 3)), np.array([[np.nan, 1., 2.]])]:
        with pytest.raises(ValueError):
            SupineTransform.from_points(points)


def test_supine_dynamics_remain_stable_when_halving_step(reference, tmp_path):
    import json
    from coil_simulation.posture import SupineTransform
    from coil_simulation.export import save_payload
    from coil_simulation.simulation import Settings, simulate_phase
    transform = SupineTransform.from_points(reference.points)
    save_payload(transform.payload(reference.payload), tmp_path)
    replay = Replay(tmp_path, load_predictive=False)
    inputs = ROOT/'数据'/'仿真输入'
    config = json.loads((inputs/'config.json').read_text('utf-8'))
    config['bias_field_T'] = .01
    config['gravity_m_s2'] = [0., 0., -9.81]
    source = transform.source(pd.read_csv(inputs/'trajectory_state.csv'))
    risk = json.loads((inputs/'simulation_summary.json').read_text('utf-8'))['assumptions']
    coarse, final_coarse, report = simulate_phase(source, config, risk, replay, settings=Settings(step_s=.02), stop_after_s=4.)
    fine, final_fine, fine_report = simulate_phase(source, config, risk, replay, settings=Settings(step_s=.01), stop_after_s=4.)
    assert np.linalg.norm(final_coarse['state'][:3]-final_fine['state'][:3]) < .00005
    assert np.linalg.norm(final_coarse['state'][3:6]-final_fine['state'][3:6]) < .00005
    assert report['peak_current_a'] <= 10. and fine_report['peak_current_a'] <= 10.
    assert coarse[-1]['arc'] > 0. and fine[-1]['arc'] > 0.
