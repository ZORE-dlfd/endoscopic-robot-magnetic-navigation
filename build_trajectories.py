import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter1d
from scipy.spatial.transform import Rotation

from camera_model import BASE, ROOT, SETTINGS, RetimedReference, baseline_frames, target_frames
from coordination import build_schedule, prune_unsettled_waits, speed_dilation

sys.dont_write_bytecode = True


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def visible_camera(data, source):
    queries = []

    def collect(origin, target):
        queries.append([np.asarray(origin).tolist(), np.asarray(target).tolist()])
        return True

    target_frames(data, source, collect)
    (ROOT / 'ray_queries.json').write_text(json.dumps(queries), encoding='utf-8')
    subprocess.run([r'D:\3dslicer\3D Slicer 5.10.0\bin\PythonSlicer.exe', 'ray_check.py'], cwd=ROOT, check=True)
    results = json.loads((ROOT / 'ray_results.json').read_text(encoding='utf-8'))
    cache = {tuple(np.round(np.r_[origin, target], 5)): result for (origin, target), result in zip(queries, results)}
    return target_frames(data, source, lambda origin, target: cache[tuple(np.round(np.r_[origin, target], 5))])


def coordination_dilation(camera, replay):
    camera_times = np.asarray(camera['time_s'], dtype=float)
    body = Rotation.from_quat(replay[[f'actual_q{axis}' for axis in 'xyzw']].to_numpy())
    desired_world = Rotation.from_quat(np.asarray(camera['goal_quaternion'], dtype=float))
    desired_relative = body.inv() * desired_world
    segments = (desired_relative[:-1].inv() * desired_relative[1:]).as_rotvec()
    rates = np.linalg.norm(segments, axis=1) * 180.0 / np.pi / np.diff(camera_times)
    rate_samples = np.r_[rates[0], (rates[:-1] + rates[1:]) / 2.0, rates[-1]]
    factor = speed_dilation(np.asarray(camera['tracking_error_deg']), rate_samples)
    return np.clip(gaussian_filter1d(factor, sigma=6.0), 1.0, 1.25), rate_samples


def make_variant(name, source, fixed_camera, factors, add_feedback_waits, settings, builder, physics, baseline):
    source_times = source.time_s.to_numpy(dtype=float)
    replay_times = np.asarray(fixed_camera['time_s'], dtype=float)
    source_factors = np.interp(source_times, replay_times, factors)
    camera_goals = np.asarray(fixed_camera['goal_joint_angle_deg'], dtype=float)
    source_goals = np.column_stack([
        np.interp(source_times, replay_times, np.unwrap(np.deg2rad(camera_goals[:, axis])) * 180.0 / np.pi)
        for axis in range(3)])
    source_goals[:, 2] = (source_goals[:, 2] + 180.0) % 360.0 - 180.0
    new_times, original_times, waits = build_schedule(source_times, source_goals, source_factors,
                                                       add_feedback_waits=add_feedback_waits)
    schedule_path = ROOT / f'{name}_reference.csv'
    rejected_waits = []
    feedback_rounds = 0
    while True:
        indices = np.searchsorted(source_times, original_times, side='left').clip(0, len(source_times) - 1)
        scheduled = source.iloc[indices].copy().reset_index(drop=True)
        scheduled['time_s'] = new_times
        reference = RetimedReference(physics.ReferenceTrajectory(source), new_times, original_times)
        solution = builder.solve(reference, settings)
        wait_boundaries = [boundary for wait in waits
                           for boundary in [wait['start_s'], wait['start_s'] + wait['duration_s']]]
        output_times = np.unique(np.r_[np.arange(0, reference.times[-1], .2), wait_boundaries, reference.times[-1]])
        replay = builder.state_table(reference, scheduled, output_times, solution, settings)
        replay['arc_s_mm'] = np.interp(reference.clock(output_times), source_times, source.arc_s_mm)
        replay['camera_wait_planned'] = np.asarray([
            any(wait['start_s'] <= time < wait['start_s'] + wait['duration_s'] for wait in waits)
            for time in output_times], dtype=float)
        camera = visible_camera(replay, source)
        checked_times, checked_source, checked_waits, rejected = prune_unsettled_waits(
            new_times, original_times, waits, camera)
        feedback_rounds += 1
        waits = checked_waits
        if not rejected:
            break
        rejected_waits.extend({**wait, 'feedback_round': feedback_rounds} for wait in rejected)
        new_times, original_times = checked_times, checked_source
        print(f'{name}: replay feedback round {feedback_rounds}, rejected {len(rejected)}, retained {len(waits)}', flush=True)
    scheduled.to_csv(schedule_path, index=False)
    replay_path = ROOT / f'capsule_{name}_replay.csv'
    replay.to_csv(replay_path, index=False)
    metadata = {'duration_s': float(output_times[-1]), 'slowdown_max': float(source_factors.max()),
                'slowdown_mean': float(np.average(source_factors, weights=np.gradient(source_times))),
                'wait_count': len(waits), 'wait_duration_s': float(sum(wait['duration_s'] for wait in waits)),
                'waits': waits, 'reference_csv': schedule_path.name, 'replay_csv': replay_path.name,
                'wait_budget_s': float(0.03 * (source_times[-1] - source_times[0])),
                'rejected_waits': rejected_waits, 'replay_feedback_rounds': feedback_rounds,
                'all_retained_waits_replay_verified': True}
    return {f'{name}_new': camera, f'{name}_endoscopy': baseline_frames(replay, baseline)}, metadata, replay


def build():
    source = pd.read_csv(BASE / 'source_trajectory_state.csv')
    original = pd.read_csv(BASE / 'capsule_replay.csv')
    settings = json.loads((BASE / 'capsule_config.json').read_text(encoding='utf-8'))
    baseline = json.loads((ROOT / 'endoscopy_baseline.json').read_text(encoding='utf-8'))
    fixed_camera = visible_camera(original, source)
    physics = load_module('magnetic_model', ROOT.parent / 'Magnetic_shape_comparison' / 'magnetic_model.py')
    builder = load_module('camera_simulation_builder', ROOT.parent / 'Magnetic 1' / 'build_simulation.py')
    builder.ellipsoid_inertia = physics.capsule_inertia
    factors, rates = coordination_dilation(fixed_camera, original)

    result = {'fixed_new': fixed_camera, 'fixed_endoscopy': baseline_frames(original, baseline)}
    ablations = {'original': {'duration_s': float(original.time_s.iloc[-1]), 'slowdown_max': 1.0,
                              'slowdown_mean': 1.0, 'wait_count': 0, 'wait_duration_s': 0.0}}
    replay_outputs = {}
    for name, profile, add_waits in [('slowdown', factors, False),
                                     ('wait', np.ones(len(factors)), True),
                                     ('coordinated', factors, True)]:
        tracks, metadata, replay = make_variant(name, source, fixed_camera, profile, add_waits,
                                                settings, builder, physics, baseline)
        result.update(tracks)
        ablations[name] = metadata
        replay_outputs[name] = replay

    replay_outputs['coordinated'].to_csv(ROOT / 'capsule_cooperative_replay.csv', index=False)
    result['settings'] = SETTINGS
    result['original_duration_s'] = float(original.time_s.iloc[-1])
    result['cooperative_duration_s'] = ablations['coordinated']['duration_s']
    result['slowdown_max'] = float(factors.max())
    result['planned_waits'] = ablations['coordinated']['waits']
    result['ablation_modes'] = ablations
    result['coordination_profile'] = {'tracking_error_deg': fixed_camera['tracking_error_deg'],
                                      'cone_error_deg': fixed_camera['cone_residual_deg'],
                                      'desired_rate_deg_s': rates.tolist(), 'time_dilation': factors.tolist()}
    result['method_notice'] = ('Quaternion camera tracking lag is separated from residual target error caused by the forward-view cone. '
                               'Slowdown follows predicted feasible tracking lag and relative angular demand. Holds are simulated from '
                               'current joint state and added only if the feasible goal settles within 1.2 s; total holds are capped '
                               'at 3 percent of original duration, with 8 s cooldown and 24 deg entry threshold. Slowdown is capped '
                               'at 1.25 time dilation and ignores tracking lag below 15 deg. This is offline state '
                               'simulation, not live image feedback. All candidate holds are additionally replay-verified: '
                               'unnecessary or unsettled events are removed and dynamics recomputed until every retained '
                               'event passes exact-boundary joint-state checks. Four ablation modes share geometry and evaluation thresholds.')
    (ROOT / 'camera_trajectories.json').write_text(json.dumps(result, separators=(',', ':'), allow_nan=False), encoding='utf-8')
    print(json.dumps({'original_s': result['original_duration_s'], 'ablations': ablations,
                      'slowdown_max': result['slowdown_max'], 'profile_rate_p95_deg_s': float(np.percentile(rates, 95))},
                     ensure_ascii=False))


if __name__ == '__main__':
    build()
