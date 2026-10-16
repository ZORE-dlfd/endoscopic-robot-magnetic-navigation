import csv
import json
import time
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from .controller import State, advance
from .coverage import Coverage
from .geometry import cone_angle
from .planning import Plan, observe_transition, select_predictive
from .search import dense_goals


class EvaluationWindow:
    def __init__(self, scene, duration):
        self.scene = scene
        self.duration = duration

    def __getattr__(self, name):
        return getattr(self.scene, name)


def write_csv(path, rows):
    if not rows:
        return
    with Path(path).open('w', encoding='utf-8-sig', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def motion_metrics(times, angles, velocities, accelerations, world_quaternions):
    steps = np.diff(times)
    jerk = np.diff(accelerations, axis=0) / steps[:, None]
    world_rotations = Rotation.from_quat(world_quaternions)
    world_speed = np.rad2deg((world_rotations[1:] * world_rotations[:-1].inv()).magnitude()) / steps
    reversal_count = 0
    for axis in range(3):
        signs = np.sign(velocities[np.abs(velocities[:, axis]) >= 1., axis])
        reversal_count += int(np.count_nonzero(np.diff(signs)))
    cone = np.rad2deg(np.arccos(np.clip(np.cos(np.deg2rad(angles[:, 0])) * np.cos(np.deg2rad(angles[:, 1])), -1, 1)))
    violating = ((cone > 45 + 1e-7) | np.any(np.abs(angles[:, :2]) > 89 + 1e-7, axis=1) |
                 np.any(np.abs(velocities) > 60 + 1e-7, axis=1) | np.any(np.abs(accelerations) > 150 + 1e-5, axis=1))
    return dict(max_joint_speed_deg_s=float(np.max(np.abs(velocities))),
                max_joint_acceleration_deg_s2=float(np.max(np.abs(accelerations))),
                rms_joint_acceleration_deg_s2=float(np.sqrt(np.mean(np.sum(accelerations ** 2, axis=1)))),
                rms_joint_jerk_deg_s3=float(np.sqrt(np.mean(np.sum(jerk ** 2, axis=1)))),
                p95_joint_jerk_deg_s3=float(np.percentile(np.linalg.norm(jerk, axis=1), 95)),
                max_world_camera_speed_deg_s=float(world_speed.max()),
                p95_world_camera_speed_deg_s=float(np.percentile(world_speed, 95)),
                reversal_count=reversal_count, max_actual_cone_deg=float(cone.max()),
                constraint_violating_steps=int(np.count_nonzero(violating)),
                constraint_violating_duration_s=float(steps @ violating[1:]),
                roll_travel_deg=float(np.abs((np.diff(angles[:, 2]) + 180) % 360 - 180).sum()))


def run_method(method, scene, visibility, config, stop_time, output):
    if method != 'predictive':
        raise ValueError('Unknown method')
    config.validate()
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    scene = EvaluationWindow(scene, min(float(stop_time), scene.duration))
    if scene.duration <= 0:
        raise ValueError('Evaluation duration must be positive')
    state = State(scene.initial_angle.copy(), scene.initial_velocity.copy(), scene.initial_angle.copy())
    coverage = Coverage(scene.areas, scene.insertion_covered, scene.settings['minimum_dwell_s'], scene.settings['repeat_gap_s'])
    origin, body = scene.pose(0.)
    coverage.observe(visibility.visible(origin, body @ Rotation.from_euler('ZYX', state.angle, degrees=True).as_matrix()), 0., 0.)
    times, angles, velocities, accelerations = [0.], [state.angle.copy()], [state.velocity.copy()], [np.zeros(3)]
    decisions, curve, planning_times = [], [], []
    previous_index, time_now, fallbacks = None, 0., 0
    started = time.perf_counter()
    while time_now < scene.duration - 1e-8:
        planning_started = time.perf_counter()
        details = dict(fallback=False, predicted_gain_mm2=None, predicted_cost=None, prediction_count=0, delay_s=0.)
        candidate_count, selected_score = 37, 0.
        goals, scores, candidate_count = dense_goals(scene, visibility, state, time_now, coverage, config)
        plan, details = select_predictive(goals, state, coverage, time_now, scene, visibility, config)
        planning_elapsed = time.perf_counter() - planning_started
        planning_times.append(planning_elapsed)
        fallbacks += int(details['fallback'])
        target = plan.next_target(state)
        duration = min(config.decision_s, scene.duration - time_now)
        transition = advance(state, target, duration, config.integration_s)
        observe_transition(transition, time_now, coverage, scene, visibility, config)
        decisions.append(dict(time_s=time_now, target_yaw_deg=float(target[0]), target_pitch_deg=float(target[1]),
                              target_roll_deg=float(target[2]), searched_direction_count=candidate_count,
                              immediate_score_mm2=selected_score, planning_seconds=planning_elapsed, **details))
        times.extend(time_now + transition.offsets)
        angles.extend(transition.angles)
        velocities.extend(transition.velocities)
        accelerations.extend(transition.accelerations)
        state = transition.state
        time_now = min(scene.duration, time_now + duration)
        curve.append(dict(time_s=time_now, added_area_mm2=coverage.added_area,
                          added_percentage_points=100 * coverage.added_area / coverage.total_area))
        if len(decisions) % 50 == 0:
            print(f'{method}: {time_now:.1f}/{scene.duration:.1f}s, added={coverage.added_area:.1f} mm2, elapsed={time.perf_counter()-started:.1f}s', flush=True)
    times, angles, velocities, accelerations = map(np.asarray, (times, angles, velocities, accelerations))
    world = scene.rotations(times) * Rotation.from_euler('ZYX', angles, degrees=True)
    metrics = motion_metrics(times, angles, velocities, accelerations, world.as_quat())
    insertion_area = float(scene.areas[scene.insertion_covered].sum())
    result = dict(method=method, duration_s=scene.duration, full_withdrawal=abs(scene.duration - scene.scene.duration) < 1e-7,
                  total_area_mm2=coverage.total_area, insertion_covered_area_mm2=insertion_area,
                  withdrawal_added_area_mm2=coverage.added_area, added_percentage_points=100 * coverage.added_area / coverage.total_area,
                  final_coverage_pct=100 * (insertion_area + coverage.added_area) / coverage.total_area,
                  decision_count=len(decisions), planning_mean_ms=float(np.mean(planning_times) * 1000),
                  planning_p95_ms=float(np.percentile(planning_times, 95) * 1000), planning_max_ms=float(np.max(planning_times) * 1000),
                  experiment_wall_seconds=time.perf_counter() - started, prediction_fallback_count=fallbacks, **metrics)
    np.savez_compressed(output / 'trajectory.npz', time_s=times, joint_angle_deg=angles,
                        joint_velocity_deg_s=velocities, joint_acceleration_deg_s2=accelerations,
                        world_camera_quaternion=world.as_quat(), withdrawal_first_seen_s=coverage.first_seen)
    write_csv(output / 'decisions.csv', decisions)
    write_csv(output / 'coverage_curve.csv', curve)
    face_rows = (dict(face_id=index, area_mm2=float(area), insertion_covered=bool(scene.insertion_covered[index]),
                      withdrawal_first_seen_s=float(coverage.first_seen[index]),
                      newly_covered=bool(not scene.insertion_covered[index] and coverage.first_seen[index] >= 0))
                 for index, area in enumerate(scene.areas))
    with (output / 'face_coverage.csv').open('w', encoding='utf-8-sig', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=['face_id', 'area_mm2', 'insertion_covered', 'withdrawal_first_seen_s', 'newly_covered'])
        writer.writeheader()
        writer.writerows(face_rows)
    (output / 'summary.json').write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False), 'utf-8')
    print(f'{method} done: added={coverage.added_area:.3f} mm2, final={result["final_coverage_pct"]:.4f}%, violations={metrics["constraint_violating_steps"]}', flush=True)
    return result
