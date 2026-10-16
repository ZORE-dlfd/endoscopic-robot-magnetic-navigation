import argparse
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from withdrawal_trajectory import reverse_reference_arrays, retimed_clock


ROOT = Path(__file__).resolve().parent
BASE = ROOT.parent / 'Magnetic 1 - Field scopes'
COORDINATION = ROOT.parent / 'Magnetic 1 - Capsule Camera - Coordination v2'


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def reverse_source(frame):
    position = frame[[f'ref_position_{axis}_mm' for axis in 'xyz']].to_numpy(dtype=float)
    quaternion = frame[[f'ref_q{axis}' for axis in 'xyzw']].to_numpy(dtype=float)
    reversed_data = reverse_reference_arrays(frame.time_s.to_numpy(dtype=float), position, quaternion,
                                             frame.arc_s_mm.to_numpy(dtype=float))
    source = frame.iloc[::-1].copy().reset_index(drop=True)
    source['time_s'] = reversed_data['time_s']
    source['arc_s_mm'] = reversed_data['arc_s_mm']
    for component, axis in enumerate('xyz'):
        source[f'{axis}_mm'] = reversed_data['position_mm'][:, component]
    for component, axis in enumerate('xyzw'):
        source[f'q{axis}'] = reversed_data['quaternion_xyzw'][:, component]
    return source


def apply_dilation(source, camera_plan_path):
    plan = json.loads(camera_plan_path.read_text(encoding='utf-8'))
    plan_arc = np.asarray(plan['arc_s_mm'], dtype=float)
    plan_factors = np.asarray(plan['time_dilation'], dtype=float)
    order = np.argsort(plan_arc)
    factors = np.interp(source.arc_s_mm.to_numpy(dtype=float), plan_arc[order], plan_factors[order])
    new_times = retimed_clock(source.time_s.to_numpy(dtype=float), factors)
    mapped = np.asarray(source.time_s, dtype=float)
    indices = np.searchsorted(mapped, mapped, side='left')
    scheduled = source.iloc[indices].copy().reset_index(drop=True)
    scheduled['time_s'] = new_times
    scheduled['time_dilation'] = factors
    return scheduled, new_times, mapped, factors


def run(retime=False):
    source_frame = pd.read_csv(COORDINATION / 'capsule_slowdown_replay.csv')
    reverse = reverse_source(source_frame)
    if retime:
        plan_path = ROOT / 'withdrawal_camera_plan.json'
        if not plan_path.exists():
            raise FileNotFoundError('Run plan_withdrawal_camera.py once before --retime')
        scheduled, new_times, original_times, factors = apply_dilation(reverse, plan_path)
        output_name = 'capsule_withdrawal_replay.csv'
    else:
        factors = np.ones(len(reverse), dtype=float)
        scheduled = reverse.copy()
        new_times = np.asarray(reverse.time_s, dtype=float)
        original_times = new_times.copy()
        scheduled['time_dilation'] = factors
        output_name = 'capsule_withdrawal_base_replay.csv'

    physics = load_module('magnetic_model', ROOT.parent / 'Magnetic_shape_comparison' / 'magnetic_model.py')
    builder = load_module('withdrawal_simulation_builder', ROOT.parent / 'Magnetic 1' / 'build_simulation.py')
    builder.ellipsoid_inertia = physics.capsule_inertia
    config = json.loads((BASE / 'capsule_config.json').read_text(encoding='utf-8'))
    base_reference = physics.ReferenceTrajectory(reverse)
    if retime:
        from scipy.interpolate import PchipInterpolator

        clock = PchipInterpolator(new_times, original_times)

        class RetimeReference:
            def __init__(self):
                self.times = new_times

            def at(self, time):
                original_time = float(clock(np.clip(time, new_times[0], new_times[-1])))
                scale = float(clock(np.clip(time, new_times[0], new_times[-1]), 1))
                acceleration_scale = float(clock(np.clip(time, new_times[0], new_times[-1]), 2))
                state = base_reference.at(original_time)
                return {**state,
                        'velocity': state['velocity'] * scale,
                        'acceleration': state['acceleration'] * scale ** 2 + state['velocity'] * acceleration_scale,
                        'omega_body': state['omega_body'] * scale,
                        'alpha_body': state['alpha_body'] * scale ** 2 + state['omega_body'] * acceleration_scale}

        reference = RetimeReference()
    else:
        reference = base_reference

    solution = builder.solve(reference, config)
    output_times = np.unique(np.r_[np.arange(0.0, new_times[-1], 0.2), new_times[-1]])
    replay = builder.state_table(reference, scheduled, output_times, solution, config)
    if retime:
        from scipy.interpolate import PchipInterpolator

        mapped_clock = PchipInterpolator(new_times, original_times)
        replay['arc_s_mm'] = np.interp(mapped_clock(output_times), reverse.time_s, reverse.arc_s_mm)
        replay['time_dilation'] = np.interp(output_times, new_times, factors)
    replay['camera_wait_planned'] = 0.0
    replay.to_csv(ROOT / output_name, index=False)
    summary = {'mode': 'slowdown_only' if retime else 'base_reverse',
               'duration_s': float(output_times[-1]),
               'source_duration_s': float(reverse.time_s.iloc[-1]),
               'max_time_dilation': float(factors.max()),
               'wait_count': 0,
               'start_arc_mm': float(replay.arc_s_mm.iloc[0]),
               'end_arc_mm': float(replay.arc_s_mm.iloc[-1]),
               'replay_csv': output_name}
    (ROOT / f'{output_name[:-4]}_summary.json').write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--retime', action='store_true')
    run(parser.parse_args().retime)
