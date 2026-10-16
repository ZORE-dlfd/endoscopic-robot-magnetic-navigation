import csv
import json
import sys
from pathlib import Path

import numpy as np
import vtk
from scipy.spatial.transform import Rotation, Slerp
from vtk.util.numpy_support import numpy_to_vtk, vtk_to_numpy

from camera_model import SETTINGS, forward_frame
from coordination import integrate_gimbal_segment, quaternion_distance_deg, speed_dilation
from evaluation_core import CoverageAccumulator
from withdrawal_visibility import camera_candidate_directions, surface_masks, unique_new_area


ROOT = Path(__file__).resolve().parent


def read_csv(path):
    with path.open(newline='', encoding='utf-8-sig') as stream:
        return list(csv.DictReader(stream))


def vectors(rows, prefix, axes, suffix):
    names = [f'{prefix}{axis}' if not suffix else f'{prefix}_{axis}_{suffix}' for axis in axes]
    return np.asarray([[float(row[name]) for name in names] for row in rows])


def make_renderer(geometry, settings):
    points = vtk.vtkPoints()
    for point in geometry['points']:
        points.InsertNextPoint(*point)
    cells = vtk.vtkCellArray()
    for face in geometry['faces']:
        triangle = vtk.vtkTriangle()
        for index, point_id in enumerate(face):
            triangle.GetPointIds().SetId(index, int(point_id))
        cells.InsertNextCell(triangle)
    mesh = vtk.vtkPolyData()
    mesh.SetPoints(points)
    mesh.SetPolys(cells)
    face_count = len(geometry['faces'])
    ids = np.arange(1, face_count + 1, dtype=np.uint32)
    colors = np.column_stack([ids & 255, (ids >> 8) & 255, (ids >> 16) & 255]).astype(np.uint8)
    scalars = numpy_to_vtk(colors, deep=True, array_type=vtk.VTK_UNSIGNED_CHAR)
    scalars.SetName('face_id')
    mesh.GetCellData().SetScalars(scalars)
    mapper = vtk.vtkPolyDataMapper()
    mapper.SetInputData(mesh)
    mapper.SetScalarModeToUseCellData()
    mapper.SetColorModeToDirectScalars()
    actor = vtk.vtkActor()
    actor.SetMapper(mapper)
    actor.GetProperty().LightingOff()
    actor.GetProperty().BackfaceCullingOff()
    renderer = vtk.vtkRenderer()
    renderer.SetBackground(0, 0, 0)
    renderer.AddActor(actor)
    window = vtk.vtkRenderWindow()
    window.SetOffScreenRendering(1)
    window.SetMultiSamples(0)
    window.SetSize(int(settings['evaluation_width']), int(settings['evaluation_height']))
    window.AddRenderer(renderer)
    capture = vtk.vtkWindowToImageFilter()
    capture.SetInput(window)
    capture.SetInputBufferTypeToRGB()
    capture.ReadFrontBufferOff()
    camera = renderer.GetActiveCamera()
    camera.SetViewAngle(float(settings['fov_vertical_deg']))
    camera.SetClippingRange(float(settings['near_mm']), float(settings['far_mm']))
    return window, capture, camera, face_count


def rendered_face_mask(window, capture, camera, origin, frame, centers, normals, settings):
    camera.SetPosition(*origin)
    camera.SetFocalPoint(*(origin + frame[:, 0]))
    camera.SetViewUp(*frame[:, 2])
    window.Render()
    capture.Modified()
    capture.Update()
    rgb = vtk_to_numpy(capture.GetOutput().GetPointData().GetScalars()).reshape(-1, 3).astype(np.int32)
    pixel_ids = rgb[:, 0] + (rgb[:, 1] << 8) + (rgb[:, 2] << 16)
    counts = np.bincount(pixel_ids, minlength=len(centers) + 1)
    if len(counts) > len(centers) + 1:
        raise ValueError('Unexpected face IDs in offscreen render')
    delta = centers - origin
    distance = np.linalg.norm(delta, axis=1)
    incidence = np.abs(np.einsum('ij,ij->i', normals, delta)) / np.maximum(distance, 1e-8)
    return ((distance >= settings['min_observation_distance_mm']) &
            (distance <= settings['max_observation_distance_mm']) &
            (incidence >= np.cos(np.deg2rad(settings['max_incidence_deg']))) &
            (counts[1:] >= settings['min_projected_pixels']))


def camera_goal(body_matrix, direction, up_reference):
    frame = forward_frame(direction, up_reference)
    relative = body_matrix.T @ frame
    angles = Rotation.from_matrix(relative).as_euler('ZYX', degrees=True)
    if np.any(np.abs(angles[:2]) > SETTINGS['max_joint_angle_deg'] + 1e-8):
        return None, None
    return frame, angles


def evaluate():
    geometry = json.loads((ROOT / 'geometry.json').read_text(encoding='utf-8'))
    evaluation = json.loads((ROOT / 'evaluation.json').read_text(encoding='utf-8'))
    camera_data = json.loads((ROOT / 'camera_trajectories.json').read_text(encoding='utf-8'))
    insertion_first_seen = np.asarray(evaluation['methods']['slowdown_new']['first_seen_s'], dtype=float)
    points = np.asarray(geometry['points'], dtype=float)
    faces = np.asarray(geometry['faces'], dtype=int)
    centers = points[faces].mean(axis=1)
    normals = np.asarray(geometry['normals'], dtype=float)
    areas = np.asarray(geometry['areas_mm2'], dtype=float)
    replay_path = ROOT / 'capsule_withdrawal_replay.csv'
    if not replay_path.exists():
        replay_path = ROOT / 'capsule_withdrawal_base_replay.csv'
    replay = read_csv(replay_path)
    replay_times = np.asarray([float(row['time_s']) for row in replay])
    eval_times = np.unique(np.r_[np.arange(0.0, replay_times[-1], SETTINGS['evaluation_step_s']), replay_times[-1]])
    position_values = vectors(replay, 'actual_position', 'xyz', 'mm')
    body_quaternions = vectors(replay, 'actual_q', 'xyzw', '')
    body_rotations = Rotation.from_quat(body_quaternions)
    body_at = Slerp(replay_times, body_rotations)(eval_times)
    position_at = np.column_stack([np.interp(eval_times, replay_times, position_values[:, axis]) for axis in range(3)])
    arc_values = np.asarray([float(row['arc_s_mm']) for row in replay])
    arc_at = np.interp(eval_times, replay_times, arc_values)
    settings = evaluation['settings']
    window, capture, camera, face_count = make_renderer(geometry, settings)
    if face_count != len(areas) or len(insertion_first_seen) != face_count:
        raise ValueError('face arrays do not match geometry')

    previous_angles = np.asarray(camera_data['slowdown_new']['joint_angle_deg'][-1], dtype=float)
    previous_velocity = np.asarray(camera_data['slowdown_new']['joint_velocity_deg_s'][-1], dtype=float)
    previous_up = (body_at[0] * Rotation.from_euler('ZYX', previous_angles, degrees=True)).as_matrix()[:, 2]
    coverage = CoverageAccumulator(areas, settings['minimum_dwell_s'], settings['repeat_gap_s'])
    candidate_visibility_union = np.zeros(face_count, dtype=bool)
    target_frames, goal_quaternions, actual_quaternions = [], [], []
    selected_goals, actual_angles, actual_velocities = [], [], []
    tracking_errors, cone_residuals, aims, previews, camera_origins = [], [], [], [], []
    time_dilations, rows = [], []

    for frame_index, at_time in enumerate(eval_times):
        body_rotation = body_at[frame_index]
        body_matrix = body_rotation.as_matrix()
        robot_origin = position_at[frame_index]
        origin = robot_origin + SETTINGS['mount_offset_mm'] * body_matrix[:, 0]
        directions = camera_candidate_directions(body_matrix, SETTINGS['max_camera_axis_angle_deg'],
                                                ring_angles_deg=(15.0, 30.0, 45.0), azimuth_count=12)
        candidate_frames, candidate_angles, candidate_masks = [], [], []
        for direction in directions:
            goal_frame, goal_angles = camera_goal(body_matrix, direction, previous_up)
            if goal_frame is None:
                continue
            candidate_frames.append(goal_frame)
            candidate_angles.append(goal_angles)
            mask = rendered_face_mask(window, capture, camera, origin, goal_frame, centers, normals, settings)
            candidate_masks.append(mask)
            candidate_visibility_union |= mask
        if not candidate_masks:
            raise RuntimeError('No admissible camera pose at withdrawal frame')

        eligible_now = candidate_visibility_union.copy()
        missed_now = eligible_now & (insertion_first_seen < 0)
        already_seen = coverage.count > 0
        target_pool = missed_now & ~already_seen
        scores = np.asarray([areas[mask & target_pool].sum() for mask in candidate_masks])
        target_candidate_count = min(25, len(scores))
        best = int(np.argmax(scores[:target_candidate_count]))
        if target_frames and best != target_frames[-1]['candidate_index']:
            old = target_frames[-1]['candidate_index']
            if old < target_candidate_count and scores[old] >= 0.78 * scores[best]:
                best = old
        if scores[best] <= 0:
            best = int(np.argmin([np.linalg.norm((goal - previous_angles + 180) % 360 - 180)
                                  for goal in candidate_angles]))
        goal_frame = candidate_frames[best]
        goal_angles = candidate_angles[best]
        if frame_index:
            previous_angles, previous_velocity = integrate_gimbal_segment(
                previous_angles, previous_velocity, selected_goals[-1], goal_angles,
                float(at_time - eval_times[frame_index - 1]), step_s=0.01)
        else:
            previous_angles = previous_angles.copy()
        desired_rotation = body_rotation * Rotation.from_euler('ZYX', goal_angles, degrees=True)
        actual_rotation = body_rotation * Rotation.from_euler('ZYX', previous_angles, degrees=True)
        actual_frame = actual_rotation.as_matrix()
        previous_up = actual_frame[:, 2]
        actual_origin = origin
        actual_visible = rendered_face_mask(window, capture, camera, actual_origin, actual_frame,
                                            centers, normals, settings)
        elapsed = float(at_time - eval_times[frame_index - 1]) if frame_index else 0.0
        coverage_row = coverage.update(np.flatnonzero(actual_visible), float(at_time), elapsed)
        newly_qualified = (insertion_first_seen < 0) & (coverage.first_time >= 0)
        new_area = float(areas[newly_qualified].sum())
        eligible_area = float(areas[eligible_now].sum())
        insertion_area = float(areas[eligible_now & (insertion_first_seen >= 0)].sum())
        total_final = insertion_area + new_area
        body_forward = body_matrix[:, 0]
        cone_error = float(np.rad2deg(np.arccos(np.clip(np.dot(body_forward, actual_frame[:, 0]), -1, 1))))
        tracking = quaternion_distance_deg(desired_rotation.as_quat(), actual_rotation.as_quat())
        selected_visible = candidate_masks[best] & target_pool
        if selected_visible.any():
            target = np.average(centers[selected_visible], axis=0, weights=areas[selected_visible])
        else:
            target = origin + goal_frame[:, 0] * 30.0
        camera_origins.append(origin.tolist())
        target_frames.append({'candidate_index': best, 'target': target.tolist()})
        selected_goals.append(goal_angles.tolist())
        actual_angles.append(previous_angles.tolist())
        actual_velocities.append(previous_velocity.tolist())
        goal_quaternions.append(desired_rotation.as_quat().tolist())
        actual_quaternions.append(actual_rotation.as_quat().tolist())
        tracking_errors.append(tracking)
        cone_residuals.append(cone_error)
        aims.append(float(np.rad2deg(np.arccos(np.clip(np.dot(goal_frame[:, 0],
                                                               (target - origin) / max(np.linalg.norm(target - origin), 1e-9)), -1, 1)))))
        previews.append(float(np.linalg.norm(target - origin)))
        rows.append({'time_s': float(at_time), 'arc_s_mm': float(arc_at[frame_index]),
                     'recoverable_area_mm2': eligible_area,
                     'fold_hidden_area_mm2': float(areas[~eligible_now].sum()),
                     'insertion_covered_area_mm2': insertion_area,
                     'insertion_missed_recoverable_area_mm2': float(areas[missed_now].sum()),
                     'withdrawal_added_area_mm2': new_area,
                     'final_covered_area_mm2': total_final,
                     'final_recoverable_coverage_pct': 100.0 * total_final / max(eligible_area, 1e-9),
                     'current_effective_view_pct': coverage_row['effective_pct'] if 'effective_pct' in coverage_row else None,
                     'cone_angle_deg': cone_error, 'tracking_error_deg': tracking})

    masks = surface_masks(areas, candidate_visibility_union[None, :], insertion_first_seen)
    area_report = unique_new_area(areas, masks['recoverable'], masks['insertion_covered'], coverage.first_time >= 0)
    recoverable_area = float(areas[masks['recoverable']].sum())
    fold_hidden_area = float(areas[masks['fold_hidden']].sum())
    insertion_missed_area = float(areas[masks['insertion_missed']].sum())
    for row in rows:
        row['recoverable_area_mm2'] = recoverable_area
        row['fold_hidden_area_mm2'] = fold_hidden_area
        row['insertion_covered_area_mm2'] = area_report['insertion_covered_area_mm2']
        row['insertion_missed_recoverable_area_mm2'] = insertion_missed_area
        row['final_covered_area_mm2'] = (area_report['insertion_covered_area_mm2'] +
                                         row['withdrawal_added_area_mm2'])
        row['final_recoverable_coverage_pct'] = (100.0 * row['final_covered_area_mm2'] /
                                                 max(recoverable_area, 1e-9))
    relative_goals = body_at.inv() * Rotation.from_quat(np.asarray(goal_quaternions))
    relative_delta = (relative_goals[:-1].inv() * relative_goals[1:]).as_rotvec()
    rates = np.linalg.norm(relative_delta, axis=1) * 180 / np.pi / np.diff(eval_times)
    rate_samples = np.r_[rates[0], (rates[:-1] + rates[1:]) * 0.5, rates[-1]] if len(rates) else np.zeros(len(eval_times))
    dilation = np.clip(np.asarray(speed_dilation(tracking_errors, rate_samples)), 1.0, 1.25)
    from withdrawal_trajectory import smooth_time_dilation
    dilation = smooth_time_dilation(eval_times, dilation)

    withdrawal_camera = {'time_s': eval_times.tolist(), 'position': camera_origins,
                         'quaternion': actual_quaternions, 'goal_quaternion': goal_quaternions,
                         'target': [item['target'] for item in target_frames],
                         'preview_mm': previews, 'aim_error_deg': aims,
                         'tracking_error_deg': tracking_errors, 'cone_residual_deg': cone_residuals,
                         'goal_joint_angle_deg': selected_goals, 'joint_angle_deg': actual_angles,
                         'joint_velocity_deg_s': actual_velocities,
                         'joint_acceleration_deg_s2': np.gradient(np.asarray(actual_velocities), eval_times, axis=0).tolist(),
                         'arc_s_mm': arc_at.tolist()}
    (ROOT / 'withdrawal_camera.json').write_text(json.dumps(withdrawal_camera, separators=(',', ':')), encoding='utf-8')
    (ROOT / 'withdrawal_camera_plan.json').write_text(json.dumps(
        {'time_s': eval_times.tolist(), 'arc_s_mm': arc_at.tolist(), 'time_dilation': dilation.tolist(),
         'candidate_pose_count': 37, 'target_pose_count': 25, 'wait_count': 0}, separators=(',', ':')), encoding='utf-8')
    (ROOT / 'withdrawal_visibility_masks.json').write_text(json.dumps(
        {'recoverable': masks['recoverable'].tolist(), 'fold_hidden': masks['fold_hidden'].tolist(),
         'insertion_covered': masks['insertion_covered'].tolist(),
         'insertion_missed': masks['insertion_missed'].tolist(),
         'withdrawal_first_seen_s': coverage.first_time.tolist()}, separators=(',', ':')), encoding='utf-8')
    import csv as csv_module
    with (ROOT / 'withdrawal_evaluation.csv').open('w', newline='', encoding='utf-8-sig') as stream:
        writer = csv_module.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    face_rows = []
    for index in range(face_count):
        face_rows.append({'face_id': index, 'area_mm2': float(areas[index]),
                          'recoverable': bool(masks['recoverable'][index]),
                          'fold_hidden': bool(masks['fold_hidden'][index]),
                          'insertion_covered': bool(masks['insertion_covered'][index]),
                          'insertion_missed': bool(masks['insertion_missed'][index]),
                          'withdrawal_first_seen_s': float(coverage.first_time[index]),
                          'withdrawal_added': bool(area_report['withdrawal_new_mask'][index])})
    with (ROOT / 'withdrawal_face_coverage.csv').open('w', newline='', encoding='utf-8-sig') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(face_rows[0]))
        writer.writeheader()
        writer.writerows(face_rows)
    summary = {key: value for key, value in area_report.items() if key != 'withdrawal_new_mask'}
    summary.update({
               'total_lumen_area_mm2': float(areas.sum()),
               'recoverable_area_mm2': recoverable_area,
               'fold_hidden_area_mm2': fold_hidden_area,
               'insertion_missed_recoverable_area_mm2': insertion_missed_area,
               'withdrawal_added_pct_of_recoverable': 100.0 * area_report['withdrawal_added_area_mm2'] /
                   max(float(areas[masks['recoverable']].sum()), 1e-9),
               'coverage_gain_percentage_points': 100.0 * area_report['withdrawal_added_area_mm2'] /
                   max(float(areas[masks['recoverable']].sum()), 1e-9),
               'duration_s': float(eval_times[-1]), 'wait_count': 0,
               'forward_cone_limit_deg': SETTINGS['max_camera_axis_angle_deg'],
               'candidate_camera_views_per_position': 37,
               'target_camera_views_per_position': 25,
               'method_notice': 'Offline geometric simulation. Fold-hidden mask is approximated by 37 sampled camera directions at each 0.4 s withdrawal pose; not live image or clinical safety validation.'})
    (ROOT / 'withdrawal_summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')
    window.Finalize()
    print(json.dumps({'candidate_poses': len(eval_times) * 17,
                      'withdrawal_added_area_mm2': area_report['withdrawal_added_area_mm2'],
                      'recoverable_area_mm2': summary['recoverable_area_mm2'],
                      'max_time_dilation': float(dilation.max()), 'cone_error_max_deg': float(max(cone_residuals)),
                      'tracking_error_p95_deg': float(np.percentile(tracking_errors, 95))}, ensure_ascii=False))


if __name__ == '__main__':
    evaluate()
    import slicer
    slicer.app.exit()
