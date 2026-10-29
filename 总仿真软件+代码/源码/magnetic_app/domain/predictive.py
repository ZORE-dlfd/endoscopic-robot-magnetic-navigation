import hashlib
import json

import numpy as np


def file_hash(filename):
    return hashlib.sha256(filename.read_bytes()).hexdigest()


def apply_prediction(replay):
    manifest_file = replay.path / 'predictive_camera.json'
    track_file = replay.path / 'predictive_camera.npz'
    if not manifest_file.is_file() or not track_file.is_file():
        raise ValueError('预测退镜数据缺失，请使用完整的软件文件夹，不要只复制EXE。')
    manifest = json.loads(manifest_file.read_text('utf-8'))
    if manifest.get('schema') != 1 or manifest.get('method') != 'dense-short-horizon-pd-v1':
        raise ValueError('预测退镜数据版本不匹配。')
    for name in ['metadata.json', 'arrays.npz']:
        if file_hash(replay.path / name) != manifest['source_sha256'][name]:
            raise ValueError('预测退镜与原始数据不匹配：' + name)
    if file_hash(track_file) != manifest['trajectory_sha256']:
        raise ValueError('预测退镜轨迹校验失败。')
    with np.load(track_file, allow_pickle=False) as stored:
        camera = {key[len('camera__'):]: stored[key] for key in stored.files if key.startswith('camera__')}
        first_seen = stored['withdrawal_first_seen_s']
        roll = stored['roll_rate_deg_s']
    times = camera['time_s']
    if (len(times) < 2 or not np.isfinite(times).all() or np.any(np.diff(times) <= 0) or
            abs(times[0] - replay.join_time) > 1e-8 or abs(times[-1] - replay.duration) > 1e-8 or
            abs(manifest['join_time_s'] - replay.join_time) > 1e-8 or
            abs(manifest['duration_s'] - (replay.duration - replay.join_time)) > 1e-8):
        raise ValueError('预测退镜时间范围无效。')
    if set(camera) != set(replay.camera):
        raise ValueError('预测退镜相机字段不完整。')
    for key, values in camera.items():
        if (values.shape != (len(times),) + np.asarray(replay.camera[key]).shape[1:] or
                not np.isfinite(values).all()):
            raise ValueError('预测退镜相机字段无效：' + key)
    if roll.shape != times.shape or not np.isfinite(roll).all():
        raise ValueError('预测退镜滚转速率字段无效。')
    if (first_seen.shape != replay.areas.shape or not np.isfinite(first_seen).all() or
            np.any((first_seen != -1) & ((first_seen < 0) | (first_seen > manifest['duration_s'] + 1e-8)))):
        raise ValueError('预测退镜首次覆盖时间无效。')
    if not np.allclose(np.linalg.norm(camera['quaternion'], axis=1), 1., atol=1e-8):
        raise ValueError('预测退镜四元数无效。')
    keep = replay.camera_times <= replay.join_time
    if (not np.allclose(camera['joint_angle_deg'][0], replay.camera['joint_angle_deg'][keep][-1], atol=1e-8) or
            not np.allclose(camera['joint_velocity_deg_s'][0], replay.camera['joint_velocity_deg_s'][keep][-1], atol=1e-8) or
            abs(np.dot(camera['quaternion'][0], replay.camera_quaternions[keep][-1])) < 1 - 1e-8):
        raise ValueError('进镜与预测退镜衔接不一致。')
    replay.camera = {key: np.concatenate([np.asarray(replay.camera[key])[keep], values[1:]])
                     for key, values in camera.items()}
    replay.payload['cameras']['roundtrip_new'] = replay.camera
    old_roll = np.asarray(replay.payload['cameraMotion']['roundtrip_new']['roll_rate_deg_s'])
    combined_roll = np.r_[old_roll[keep], roll[1:]]
    replay.payload['cameraMotion']['roundtrip_new'] = dict(roll_rate_deg_s=combined_roll,
        roll_rate_p95_deg_s=float(np.percentile(abs(combined_roll), 95)))
    insertion_covered = (replay.insertion_seen >= 0) & (replay.insertion_seen <= replay.join_time)
    added = ~insertion_covered & (first_seen >= 0)
    summary = manifest['summary']
    if (not np.isclose(replay.areas[added].sum(), summary['withdrawal_added_area_mm2'], rtol=0, atol=1e-7) or
            not np.isclose(replay.areas[insertion_covered | added].sum(), summary['final_covered_area_mm2'], rtol=0, atol=1e-7)):
        raise ValueError('预测退镜覆盖面积与逐面片记录不一致。')
    replay.masks = dict(insertion_covered=insertion_covered, withdrawal_first_seen_s=first_seen)
    replay.withdrawal_seen, replay.summary, replay.prediction = first_seen, summary, manifest
    replay.payload['withdrawalCoverage'] = dict(masks=replay.masks, summary=summary)
    insertion_area = float(replay.areas[insertion_covered].sum())
    total = float(replay.areas.sum())
    event_times = np.unique(np.r_[0., first_seen[added], manifest['duration_s']])
    order = np.argsort(first_seen[added])
    observed_times = first_seen[added][order]
    cumulative = np.r_[0., np.cumsum(replay.areas[added][order])]
    rows = [row for row in replay.metric_rows if row['time_s'] < replay.join_time]
    for time in event_times:
        area = float(cumulative[np.searchsorted(observed_times, time, side='right')])
        absolute = float(time + replay.join_time)
        rows.append(dict(time_s=absolute, arc_s_mm=float(np.interp(absolute, replay.times, replay.data['arc_s_mm'])),
                         coverage_pct=100 * (insertion_area + area) / total, unseen_pct=100 * (1 - (insertion_area + area) / total),
                         effective_pct=None, repeat_pct=None, effective_time_mean_pct=None,
                         completion_pct=100 * time / manifest['duration_s'], insertion_covered_area_mm2=insertion_area,
                         withdrawal_added_area_mm2=area, withdrawal_added_pct_of_total_lumen=100 * area / total,
                         coverage_gain_percentage_points=100 * area / total, final_covered_area_mm2=insertion_area + area))
    replay.metric_rows = rows
    replay.metric_times = np.array([row['time_s'] for row in rows])
    replay.payload['evaluation']['methods']['withdrawal_new'] = dict(rows=rows, final=rows[-1],
        first_seen_s=first_seen + np.where(first_seen >= 0, replay.join_time, 0), common_budget=rows[-1])
