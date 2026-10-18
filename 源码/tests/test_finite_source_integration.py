from pathlib import Path
import sys

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT/'源码'), str(ROOT/'构建'/'依赖')]


def test_finite_source_design_matches_validated_assumptions():
    from coil_simulation.finite_sources import CoilDesign, FiniteArray
    design = CoilDesign(turns=1200)
    array = FiniteArray(design, layout_scale=.75)
    assert design.core_length_m == .052 and design.core_radius_m == .016
    assert design.winding_outer_radius_m == .028
    assert design.current_limit_a == 10.
    assert abs(design.resistance_ohm-12.3102) < .0001
    assert np.linalg.norm(array.magnetization_matrix-np.diag(np.diag(array.magnetization_matrix))) > 0


def test_display_kernel_recovers_exact_finite_sources_away_from_coils():
    from magnetic_app.domain.finite_field import FiniteField
    from coil_simulation.finite_sources import CoilDesign, FiniteArray
    array = FiniteArray(CoilDesign(turns=1200), layout_scale=.75)
    kernel = FiniteField(ROOT/'数据'/'clon001'/'finite_field.npz')
    exterior = (kernel.radial[:, None] > .029) | (kernel.axial[None, :] > .027)
    assert np.all(np.max(np.abs(kernel.values), axis=(2, 3))[exterior] > 0)
    random = np.random.default_rng(20261018)
    currents = random.uniform(-8., 8., 8)
    points = random.uniform(-.006, .006, (32, 3))
    expected = np.array([array.responses(point)[0]@currents for point in points])
    actual = kernel.field(points, array.offsets, array.axes, currents, array.magnetization(currents))
    np.testing.assert_allclose(actual, expected, rtol=.002, atol=2e-6)
    expected_gradient = np.einsum('i,ijk->jk', currents, array.responses(points[0])[1])
    gradient = kernel.gradient(points[0], array.offsets, array.axes, currents, array.magnetization(currents))
    np.testing.assert_allclose(gradient, expected_gradient, rtol=.005, atol=5e-5)


def test_compact_payload_and_each_coil_parameters_are_present():
    from magnetic_app.domain.replay import Replay
    replay = Replay()
    sources = replay.payload['coilSources']
    assert sources['model'] == 'finite-winding-core-coupled-v1'
    assert sources['design']['turns'] == 1200
    assert 'moment_per_amp_a_m2' not in sources
    frame = replay.frame(replay.duration*.6)
    rows = replay.coils.parameter_rows(frame)
    assert len(rows) == 8
    for row in rows:
        assert row['core_length_mm'] == 52 and row['core_diameter_mm'] == 32
        assert row['winding_outer_diameter_mm'] == 56
        assert abs(row['current_a']) <= 10.+1e-7
        assert abs(row['power_w']-row['current_a']**2*row['resistance_ohm']) < 1e-8
        assert 0 <= row['saturation_fraction'] <= 1
    assert replay.prediction['method'] == 'dense-short-horizon-pd-v1'
    assert replay.robot['config']['bias_field_T'] == .008


def test_lookup_axis_symmetry_and_out_of_bounds():
    from magnetic_app.domain.finite_field import FiniteField
    from coil_simulation.finite_sources import CoilDesign, cylinder_field_gradient, winding_field_gradient
    kernel = FiniteField(ROOT/'数据'/'clon001'/'finite_field.npz')
    positions = np.zeros((1, 3))
    axes = np.array([[0., 0., 1.]])
    point = np.array([0., 0., .1])
    design = CoilDesign(turns=1200)
    exact_field, exact_gradient = winding_field_gradient(point, design)
    core_field, core_gradient = cylinder_field_gradient(point, design)
    exact_field += core_field*200000.
    exact_gradient += core_gradient*200000.
    field = kernel.field(point, positions, axes, [1.], [200000.])
    gradient = kernel.gradient(point, positions, axes, [1.], [200000.])
    np.testing.assert_allclose(field, exact_field, rtol=1e-9, atol=1e-12)
    np.testing.assert_allclose(gradient, exact_gradient, rtol=1e-9, atol=1e-12)
    np.testing.assert_allclose(kernel.field(-point, positions, axes, [1.], [200000.]), field, atol=1e-12)
    np.testing.assert_allclose(kernel.gradient(-point, positions, axes, [1.], [200000.]), -gradient, atol=1e-12)
    with pytest.raises(ValueError, match='exceeds'):
        kernel.field([4., 0., 0.], positions, axes, [1.], [200000.])


def test_source_interior_is_excluded_even_with_neighboring_coils():
    from magnetic_app.domain.finite_field import FiniteField
    kernel = FiniteField(ROOT/'数据'/'clon001'/'finite_field.npz')
    positions = np.array([[0., 0., 0.], [.1, 0., 0.]])
    axes = np.array([[0., 0., 1.], [0., 0., 1.]])
    np.testing.assert_array_equal(kernel.field([0., 0., 0.], positions, axes, [1., 1.], [200000., 200000.]), np.zeros(3))
    np.testing.assert_array_equal(kernel.gradient([0., 0., 0.], positions, axes, [1., 1.], [200000., 200000.]), np.zeros((3, 3)))
