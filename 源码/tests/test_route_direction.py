from copy import deepcopy
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT/'源码'), str(ROOT/'构建'/'依赖')]

from magnetic_app.domain.replay import Replay


@pytest.fixture(scope='module')
def reference():
    return Replay(ROOT/'数据'/'参考原版')


@pytest.fixture
def contract():
    return dict(entry_original_mm=[.970171, -145.567, -506.25],
        deep_original_mm=[35.5332, -257.172, -342.2], landmark_tolerance_mm=.001)


def test_insertion_uses_confirmed_reference_not_reversed_csv(reference, contract):
    from coil_simulation.route_direction import insertion_source
    source = insertion_source(reference, contract)
    raw = pd.read_csv(ROOT/'数据'/'仿真输入'/'trajectory_state.csv')
    np.testing.assert_allclose(source[['x_mm', 'y_mm', 'z_mm']].iloc[0], contract['entry_original_mm'])
    np.testing.assert_allclose(source[['x_mm', 'y_mm', 'z_mm']].iloc[-1], contract['deep_original_mm'])
    assert np.linalg.norm(source[['x_mm', 'y_mm', 'z_mm']].iloc[0]-raw[['x_mm', 'y_mm', 'z_mm']].iloc[0]) > 100.
    for key in source:
        np.testing.assert_array_equal(source[key], reference.source[key])


def test_direction_contract_rejects_reversed_or_wrong_endpoints(reference, contract):
    from coil_simulation.route_direction import insertion_source
    reversed_reference = deepcopy(reference)
    reversed_reference.source = {key: np.asarray(values)[::-1] for key, values in reference.source.items()}
    with pytest.raises(ValueError, match='entry|endpoint'):
        insertion_source(reversed_reference, contract)
    contract['entry_original_mm'][0] += 10.
    with pytest.raises(ValueError, match='entry|endpoint'):
        insertion_source(reference, contract)


def test_phase_verifier_rejects_playback_that_starts_at_deep_end(reference, contract):
    from coil_simulation.route_direction import insertion_source, verify_phases
    source = insertion_source(reference, contract)
    entry = source[['x_mm', 'y_mm', 'z_mm']].iloc[0].to_numpy()/1000.
    deep = source[['x_mm', 'y_mm', 'z_mm']].iloc[-1].to_numpy()/1000.
    entry_state = np.r_[entry, np.zeros(10)]
    deep_state = np.r_[deep, np.zeros(10)]
    insertion = [dict(state=entry_state), dict(state=deep_state)]
    withdrawal = [dict(state=deep_state.copy()), dict(state=entry_state.copy())]
    result = verify_phases(source, insertion, withdrawal)
    assert result['passed'] and result['insertion_start_error_mm'] == 0.
    with pytest.raises(ValueError, match='direction'):
        verify_phases(source, withdrawal, insertion)


def test_short_magnetic_motion_is_inward_from_anal_entry(reference, contract, tmp_path):
    from coil_simulation.route_direction import insertion_source
    from coil_simulation.posture import SupineTransform
    from coil_simulation.export import save_payload
    from coil_simulation.simulation import Settings, simulate_phase
    transform = SupineTransform.from_points(reference.points)
    save_payload(transform.payload(reference.payload), tmp_path)
    replay = Replay(tmp_path, load_predictive=False)
    source = transform.source(insertion_source(reference, contract))
    config = json.loads((ROOT/'数据'/'仿真输入'/'config.json').read_text('utf-8'))
    config['bias_field_T'] = .01
    assumptions = json.loads((ROOT/'数据'/'仿真输入'/'simulation_summary.json').read_text('utf-8'))['assumptions']
    rows, state, report = simulate_phase(source, config, assumptions, replay, settings=Settings(), stop_after_s=4.)
    entry = source[['x_mm', 'y_mm', 'z_mm']].iloc[0].to_numpy()
    np.testing.assert_allclose(rows[0]['state'][:3]*1000., entry)
    assert rows[-1]['source_arc'] > .005
    assert report['peak_current_a'] <= 10.


def test_replay_rejects_wrong_actual_entry_even_with_valid_serialized_data(contract, tmp_path):
    from coil_simulation.export import save_payload
    current = Replay()
    payload = deepcopy(current.payload)
    source = pd.DataFrame(current.source)
    entry = source[['x_mm', 'y_mm', 'z_mm']].iloc[0].to_numpy()
    deep = source[['x_mm', 'y_mm', 'z_mm']].iloc[-1].to_numpy()
    payload['routeDirection'] = dict(entry_position_mm=entry.tolist(),deep_position_mm=deep.tolist(),
        verification=dict(passed=True,tolerance_mm=1.))
    for index, axis in enumerate('xyz'):
        payload['datasets']['capsule_roundtrip'][f'actual_position_{axis}_mm'][0] = deep[index]
    save_payload(payload, tmp_path)
    import shutil
    shutil.copy2(ROOT/'数据'/'clon001'/'finite_field.npz', tmp_path/'finite_field.npz')
    with pytest.raises(ValueError, match='入口|方向'):
        Replay(tmp_path, load_predictive=False)
