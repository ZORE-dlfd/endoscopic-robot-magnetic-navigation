import argparse
import csv
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / '源码'), str(ROOT / '构建' / '依赖')]
os.environ.setdefault('PYTHONIOENCODING', 'utf-8')

import numpy as np
from scipy.spatial.transform import Rotation

from retreat_planner.config import Config
from retreat_planner.experiment import run_method
from retreat_planner.scene import Scene, sha256
from retreat_planner.visibility import Visibility


def export_overlay(scene, output, result, config, environment):
    replay = scene.replay
    with np.load(output / 'trajectory.npz', allow_pickle=False) as stored:
        trajectory = {key: stored[key] for key in stored.files}
    times = trajectory['time_s']
    absolute = times + replay.join_time
    with (output / 'decisions.csv').open(encoding='utf-8-sig', newline='') as stream:
        decisions = list(csv.DictReader(stream))
    goals = np.empty((len(times), 3))
    goals[0] = scene.initial_angle
    previous = scene.initial_angle.copy()
    for index, decision in enumerate(decisions):
        begin = float(decision['time_s'])
        end = float(decisions[index + 1]['time_s']) if index + 1 < len(decisions) else scene.duration
        target = np.array([float(decision['target_' + axis + '_deg']) for axis in ['yaw', 'pitch', 'roll']])
        selected = (times > begin + 1e-9) & (times <= end + 1e-9)
        amount = np.clip((times[selected] - begin) / (end - begin), 0, 1)
        goals[selected] = previous + amount[:, None] * ((target - previous + 180) % 360 - 180)
        previous = target
    body = scene.rotations(times)
    actual = Rotation.from_quat(trajectory['world_camera_quaternion'])
    desired = body * Rotation.from_euler('ZYX', goals, degrees=True)
    origins = np.array([scene.pose(time)[0] for time in times])
    forward_actual = actual.as_matrix()[:, :, 0]
    forward_desired = desired.as_matrix()[:, :, 0]
    angle_error = np.rad2deg(np.arccos(np.clip(np.einsum('ij,ij->i', forward_actual, forward_desired), -1, 1)))
    camera = dict(time_s=absolute, position=origins, quaternion=actual.as_quat(),
                  target=origins + forward_desired * 20., goal_quaternion=desired.as_quat(),
                  cone_residual_deg=np.zeros(len(times)), tracking_error_deg=np.rad2deg((actual.inv() * desired).magnitude()),
                  preview_mm=np.zeros(len(times)), joint_angle_deg=trajectory['joint_angle_deg'],
                  joint_velocity_deg_s=trajectory['joint_velocity_deg_s'],
                  joint_acceleration_deg_s2=trajectory['joint_acceleration_deg_s2'], aim_error_deg=angle_error,
                  goal_joint_angle_deg=goals, arc_s_mm=np.interp(absolute, replay.times, replay.data['arc_s_mm']))
    relative = (actual[:-1].inv() * actual[1:]).as_rotvec() / np.diff(times)[:, None]
    roll = np.r_[0., np.rad2deg(relative[:, 0])]
    assets = replay.path
    destination = assets / 'predictive_camera.npz'
    np.savez_compressed(destination, **{'camera__' + key: value for key, value in camera.items()},
                        withdrawal_first_seen_s=trajectory['withdrawal_first_seen_s'], roll_rate_deg_s=roll)
    added = result['withdrawal_added_area_mm2']
    insertion = result['insertion_covered_area_mm2']
    total = result['total_area_mm2']
    summary = dict(insertion_covered_area_mm2=insertion, withdrawal_added_area_mm2=added,
                   final_covered_area_mm2=insertion + added, total_lumen_area_mm2=total,
                   uncovered_area_mm2=total - insertion - added, insertion_coverage_pct=100 * insertion / total,
                   withdrawal_added_pct_of_total_lumen=100 * added / total,
                   withdrawal_added_pct_of_insertion_covered=100 * added / insertion,
                   coverage_gain_percentage_points=100 * added / total, final_coverage_pct=100 * (insertion + added) / total,
                   coverage_increased=added > 0, duration_s=scene.duration, wait_count=0,
                   forward_cone_limit_deg=45., constraint_violating_steps=result['constraint_violating_steps'],
                   method_notice='Offline dense search + short-horizon prediction + original PD. Same Hermite body pose as player. '
                   '0.1 s endpoint visibility, 0.4 s accumulated dwell, not continuous/clinical validation. '
                   'No legacy 37-direction recoverability mask; uncovered does not mean globally invisible.')
    implementation = {str(filename.relative_to(ROOT)).replace(chr(92), '/'): sha256(filename)
                      for filename in sorted((ROOT / '源码' / 'retreat_planner').glob('*.py'))}
    for filename in [Path(__file__), ROOT / '源码' / 'magnetic_app' / 'domain' / 'replay.py']:
        implementation[str(filename.relative_to(ROOT)).replace(chr(92), '/')] = sha256(filename)
    manifest = dict(schema=1, method='dense-short-horizon-pd-v1', config=config.dictionary(),
                    source_sha256={name: sha256(assets / name) for name in ['metadata.json', 'arrays.npz']},
                    trajectory_sha256=sha256(destination), implementation_sha256=implementation,
                    join_time_s=replay.join_time, duration_s=scene.duration, summary=summary, metrics=result,
                    environment=environment, body_interpolation='same Hermite position and SLERP orientation as Replay.body_pose',
                    visibility='0.1 s endpoints of each observed interval; dwell 0.4 s; repeat gap 0.8 s',
                    runtime='Precomputed offline; desktop playback never executes this planner')
    (assets / 'predictive_camera.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2, allow_nan=False), 'utf-8')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, default=ROOT / '文档' / '验证' / '预测退镜')
    parser.add_argument('--seconds', type=float)
    args = parser.parse_args()
    scene = Scene(ROOT / '数据' / 'clon001')
    inputs = {name: sha256(scene.replay.path / name) for name in ['metadata.json', 'arrays.npz']}
    config = Config().validate()
    visibility = Visibility(scene, config.atlas_size)
    try:
        result = run_method('predictive', scene, visibility, config, args.seconds or scene.duration, args.output)
        for name, digest in inputs.items():
            if sha256(scene.replay.path / name) != digest:
                raise RuntimeError('Original input changed during planning: ' + name)
        if args.seconds is None:
            if result['constraint_violating_steps'] or result['prediction_fallback_count']:
                raise RuntimeError('Prediction contains violations/fallbacks; no production overlay exported')
            export_overlay(scene, args.output, result, config, visibility.environment)
    finally:
        visibility.close()


if __name__ == '__main__':
    main()
