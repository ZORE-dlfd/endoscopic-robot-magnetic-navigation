import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / '源码'), str(ROOT / '构建' / '依赖')]

import numpy as np
from scipy.spatial.transform import Rotation

from magnetic_app.domain.replay import Replay
from retreat_planner.scene import Scene, sha256
from retreat_planner.visibility import Visibility


def main():
    original = Replay(load_predictive=False)
    replay = Replay()
    manifest = replay.prediction
    for filename, digest in manifest['implementation_sha256'].items():
        if sha256(ROOT / filename) != digest:
            raise AssertionError('Generation implementation changed: ' + filename)
    for key in original.data:
        np.testing.assert_array_equal(original.data[key], replay.data[key])
    insertion = original.camera_times <= original.join_time
    for key in original.camera:
        np.testing.assert_array_equal(original.camera[key][insertion], replay.camera[key][:insertion.sum()])
    np.testing.assert_array_equal(original.insertion_seen, replay.insertion_seen)
    selected = replay.camera_times >= replay.join_time
    times = replay.camera_times[selected] - replay.join_time
    angle = replay.camera['joint_angle_deg'][selected]
    velocity = replay.camera['joint_velocity_deg_s'][selected]
    acceleration = np.diff(velocity, axis=0) / np.diff(times)[:, None]
    cone = np.rad2deg(np.arccos(np.clip(np.cos(np.deg2rad(angle[:, 0])) * np.cos(np.deg2rad(angle[:, 1])), -1, 1)))
    violating = ((cone[1:] > 45 + 1e-7) | np.any(abs(angle[1:, :2]) > 89 + 1e-7, axis=1) |
                 np.any(abs(velocity[1:]) > 60 + 1e-7, axis=1) | np.any(abs(acceleration) > 150 + 1e-5, axis=1))
    assert not violating.any()
    np.testing.assert_allclose(acceleration, replay.camera['joint_acceleration_deg_s2'][selected][1:], atol=1e-6, rtol=0)
    maximum_rotation_error = 0.
    maximum_mount_error = 0.
    for index in range(0, len(times), 37):
        frame = replay.frame(times[index] + replay.join_time)
        actual = Rotation.from_matrix(frame.camera_rotation)
        expected = Rotation.from_quat(replay.camera['quaternion'][selected][index])
        maximum_rotation_error = max(maximum_rotation_error, float((actual.inv() * expected).magnitude()))
        maximum_mount_error = max(maximum_mount_error, float(np.linalg.norm(frame.camera_position - replay.camera['position'][selected][index])))
    assert maximum_rotation_error < 1e-8 and maximum_mount_error < 1e-8
    scene = Scene(replay.path)
    renderer = Visibility(scene)
    count = len(replay.faces)
    accumulated = np.zeros(count)
    last_seen = np.full(count, -np.inf)
    first_seen = np.full(count, -1.)
    previous_visible = np.zeros(count, dtype=bool)
    observations = 0
    previous_time = 0.
    config = manifest['config']
    decision = 0.
    sample_times = [0.]
    while decision < scene.duration - 1e-8:
        duration = min(config['decision_s'], scene.duration - decision)
        offsets = np.r_[np.arange(config['visibility_s'], duration - 1e-8, config['visibility_s']), duration]
        sample_times.extend(decision + offsets)
        decision = min(scene.duration, decision + duration)
    try:
        for time in sample_times:
            frame = replay.frame(float(time) + replay.join_time)
            visible = np.zeros(count, dtype=bool)
            visible[renderer.visible(frame.camera_position, frame.camera_rotation)] = True
            accumulated[visible & (time - last_seen > replay.settings['repeat_gap_s'] + 1e-9)] = 0.
            eligible = visible & previous_visible
            accumulated[eligible] += float(time - previous_time)
            qualified = eligible & (accumulated >= replay.settings['minimum_dwell_s'] - 1e-9) & (first_seen < 0)
            first_seen[qualified] = time
            last_seen[visible] = time
            previous_visible = visible
            previous_time = float(time)
            observations += 1
    finally:
        renderer.close()
    np.testing.assert_allclose(first_seen, replay.withdrawal_seen, atol=1e-7, rtol=0)
    new_faces = (first_seen >= 0) & ~(replay.insertion_seen >= 0)
    area = float(replay.areas[new_faces].sum())
    assert abs(area - replay.summary['withdrawal_added_area_mm2']) < 1e-8
    for time in [0., replay.join_time, 320., 410.123, 550., replay.duration]:
        observed, added = replay.coverage(time)
        area = float(replay.areas[observed | added].sum())
        assert abs(area - replay.metrics(time)['final_covered_area_mm2']) < 1e-7
    result = dict(passed=True, original_motion_columns=len(original.data), original_motion_samples=len(original.times),
                  original_motion_unchanged=True, insertion_camera_and_coverage_unchanged=True,
                  sample_count=len(times), visibility_observations=observations,
                  max_camera_rotation_error_rad=maximum_rotation_error, max_mount_error_mm=maximum_mount_error,
                  independent_constraint_violating_steps=int(violating.sum()),
                  max_cone_deg=float(cone.max()), max_joint_speed_deg_s=float(abs(velocity).max()),
                  max_joint_acceleration_deg_s2=float(abs(acceleration).max()),
                  independent_first_seen_matches=True, unique_added_faces=int(new_faces.sum()),
                  withdrawal_added_area_mm2=float(replay.areas[new_faces].sum()), final_coverage_pct=replay.summary['final_coverage_pct'],
                  limitation='Same face-ID renderer, separate playback pose and independent dwell accumulator; sampled geometric validation only.')
    output = ROOT / '文档' / '验证' / '预测退镜' / 'independent_audit.json'
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False), 'utf-8')
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
