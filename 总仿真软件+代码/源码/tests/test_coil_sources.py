from pathlib import Path
import importlib.util
import sys

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / '源码'))
sys.path.insert(0, str(ROOT / '构建' / '依赖'))


def test_coil_source_module_exists():
    assert importlib.util.find_spec('magnetic_app.domain.coil_sources') is not None


def test_field_comes_from_sources_and_not_local_affine_extrapolation():
    from magnetic_app.domain.coil_sources import source_field, source_gradient
    positions = np.array([[0., 0., 150.]])
    moments = np.array([[0., 0., 20.]])
    points = np.array([[0., 0., 0.], [40., 0., 0.]])
    fields = source_field(points, positions, moments)
    assert fields.shape == (2, 3)
    assert np.isclose(fields[0, 2], 1e-7 * 40 / .15**3 * 1000)
    gradient = source_gradient(points[0], positions, moments)
    affine = fields[0] + (points[1] - points[0]) @ gradient.T
    assert np.linalg.norm(fields[1] - affine) > .01


def test_source_gradient_matches_finite_difference_and_maxwell_constraints():
    from magnetic_app.domain.coil_sources import source_field, source_gradient
    positions = np.array([[30., -50., 150.], [-70., 0., 100.]])
    moments = np.array([[4., 2., 20.], [-3., 6., -12.]])
    point = np.array([2., 5., -4.])
    step = .001
    finite = np.column_stack([
        (source_field(point + np.eye(3)[axis] * step, positions, moments)
         - source_field(point - np.eye(3)[axis] * step, positions, moments)) / (2 * step)
        for axis in range(3)
    ])
    gradient = source_gradient(point, positions, moments)
    np.testing.assert_allclose(gradient, finite, rtol=1e-7, atol=1e-10)
    np.testing.assert_allclose(gradient, gradient.T, atol=1e-12)
    assert abs(np.trace(gradient)) < 1e-12


def test_simulation_module_exists():
    assert importlib.util.find_spec('coil_simulation.simulation') is not None


def test_coupled_camera_uses_original_dense_predictive_implementation():
    assert importlib.util.find_spec('coil_simulation.coupled_camera') is not None
    from coil_simulation.coupled_camera import Config, dense_goals, select_predictive
    from retreat_planner.search import dense_goals as original_dense
    from retreat_planner.planning import select_predictive as original_predictive
    assert dense_goals is original_dense
    assert select_predictive is original_predictive
    assert Config().decision_s == .4
    assert Config().horizon_s == 1.2
    assert Config().gain_tolerance == .05
    assert Config().integration_s == .01


def test_route_projection_cannot_escape_requested_window():
    import pandas as pd
    from coil_simulation.path_guidance import SpatialRoute
    points = pd.DataFrame(dict(arc_s_mm=[0., 100., 200., 300.],
        x_mm=[0., 100., 200., 300.], y_mm=[0.]*4, z_mm=[0.]*4,
        qx=[0.]*4, qy=[0.]*4, qz=[0.]*4, qw=[1.]*4,
        speed_risk_limit_mm_s=[2.]*4, speed_clearance_limit_mm_s=[3.]*4,
        speed_curvature_limit_mm_s=[4.]*4, risk_norm=[0.]*4,
        clearance_mm=[10.]*4, curvature_inv_mm=[0.]*4))
    route = SpatialRoute(points)
    original = route.arc_m.copy()
    result = route.project_arc([.25, 0., 0.], anchor_arc_m=.15, window_m=.01)
    assert .14 <= result <= .16 + 1e-12
    np.testing.assert_array_equal(route.arc_m, original)


def test_short_magnetic_integration_agrees_within_point_zero_five_mm():
    import json
    import pandas as pd
    from magnetic_app.domain.replay import Replay
    from coil_simulation.simulation import Settings, simulate_phase
    original = Replay(ROOT/'数据'/'参考原版')
    inputs = ROOT/'数据'/'仿真输入'
    config = json.loads((inputs/'config.json').read_text('utf-8'))
    config['bias_field_T'] = .01
    source = pd.read_csv(inputs/'trajectory_state.csv')
    risk = json.loads((inputs/'simulation_summary.json').read_text('utf-8'))['assumptions']
    coarse,final_coarse,report_coarse = simulate_phase(source,config,risk,original,settings=Settings(step_s=.02),stop_after_s=4.)
    fine,final_fine,report_fine = simulate_phase(source,config,risk,original,settings=Settings(step_s=.01),stop_after_s=4.)
    assert np.linalg.norm(final_coarse['state'][:3]-final_fine['state'][:3]) < .00005
    assert np.linalg.norm(final_coarse['state'][3:6]-final_fine['state'][3:6]) < .00005
    assert report_coarse['peak_current_a'] <= 10.
