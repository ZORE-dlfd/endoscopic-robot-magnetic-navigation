import json
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation, Slerp
from scipy.interpolate import PchipInterpolator
from coordination import angle_between, quaternion_distance_deg

ROOT = Path(__file__).resolve().parent
BASE = ROOT.parent / 'Magnetic 1 - Field scopes'
SETTINGS = {'fov_vertical_deg': 60., 'aspect': 4 / 3, 'near_mm': .2, 'far_mm': 150.,
            'max_camera_axis_angle_deg': 45., 'max_joint_angle_deg': 89., 'target_alignment_enter_cos': .5, 'target_alignment_hold_cos': .45,
            'preview_slew_mm_s': 10., 'max_joint_speed_deg_s': 60., 'max_joint_acceleration_deg_s2': 150.,
            'joint_gain_s2': 12., 'joint_damping_s': 6., 'mount_offset_mm': 5.8,
            'evaluation_step_s': .4, 'evaluation_width': 160, 'evaluation_height': 120,
            'min_observation_distance_mm': 2., 'max_observation_distance_mm': 80.,
            'max_incidence_deg': 75., 'min_projected_pixels': 2, 'minimum_dwell_s': .4,
            'repeat_gap_s': .8, 'preview_min_mm': 5., 'preview_max_mm': 45., 'visibility_step_mm': 3.}


def unit(vector):
    vector = np.asarray(vector, dtype=float)
    return vector / max(np.linalg.norm(vector), 1e-12)


def forward_frame(forward, up):
    forward = unit(forward)
    up = np.asarray(up, dtype=float) - np.dot(up, forward) * forward
    if np.linalg.norm(up) < 1e-8:
        axis = np.eye(3)[np.argmin(np.abs(forward))]
        up = axis - np.dot(axis, forward) * forward
    up = unit(up)
    return np.column_stack([forward, unit(np.cross(up, forward)), up])


class RetimedReference:
    def __init__(self, original, new_times, original_times):
        self.original = original
        self.times = np.asarray(new_times)
        self.clock = PchipInterpolator(new_times, original_times)

    def at(self, time):
        time = float(np.clip(time, self.times[0], self.times[-1]))
        original_time = float(self.clock(time))
        scale = float(self.clock(time, 1))
        derivative = float(self.clock(time, 2))
        state = self.original.at(original_time)
        return {**state, 'velocity': state['velocity'] * scale,
                'acceleration': state['acceleration'] * scale ** 2 + state['velocity'] * derivative,
                'omega_body': state['omega_body'] * scale,
                'alpha_body': state['alpha_body'] * scale ** 2 + state['omega_body'] * derivative}


def bounded_gimbal(times, goals):
    times = np.asarray(times)
    goals = np.asarray(goals)
    angles = np.zeros_like(goals)
    velocities = np.zeros_like(goals)
    accelerations = np.zeros_like(goals)
    axis_limit = SETTINGS['max_joint_angle_deg']
    angles[0, :2] = np.clip(goals[0, :2], -axis_limit, axis_limit)
    angles[0, 2] = np.clip(goals[0, 2], -180, 180)
    for index in range(1, len(times)):
        remaining = times[index] - times[index - 1]
        angle, velocity = angles[index - 1].copy(), velocities[index - 1].copy()
        last_accel = np.zeros(3)
        while remaining > 1e-10:
            step = min(.01, remaining)
            error = (goals[index] - angle + 180) % 360 - 180
            acceleration = np.clip(12 * error - 6 * velocity, -150, 150)
            candidate = np.clip(velocity + acceleration * step, -60, 60)
            for axis in range(2):
                distance = max(0, axis_limit - np.sign(candidate[axis]) * angle[axis])
                brake_speed = np.sqrt(2 * 150 * distance)
                if abs(candidate[axis]) > brake_speed:
                    acceleration[axis] = -np.sign(velocity[axis]) * 150
                    candidate[axis] = velocity[axis] + acceleration[axis] * step
            angle += (velocity + candidate) * .5 * step
            angle[:2] = np.clip(angle[:2], -axis_limit, axis_limit)
            angle[2] = (angle[2] + 180) % 360 - 180
            velocity = candidate
            last_accel = acceleration
            remaining -= step
        angles[index], velocities[index], accelerations[index] = angle, velocity, last_accel
    return angles, velocities, accelerations


def target_frames(data, source, blocked=None):
    positions = data[[f'actual_position_{axis}_mm' for axis in 'xyz']].to_numpy()
    body = Rotation.from_quat(data[[f'actual_q{axis}' for axis in 'xyzw']].to_numpy())
    matrices = body.as_matrix()
    mounted = positions + SETTINGS['mount_offset_mm'] * matrices[:, :, 0]
    source_points = source[['x_mm', 'y_mm', 'z_mm']].to_numpy()
    arc = source.arc_s_mm.to_numpy()
    goals, goal_rotations, targets, previews, occluded, cone_residuals = [], [], [], [], [], []
    previous_up = matrices[0, :, 2]
    previous_preview = None
    for index, origin in enumerate(mounted):
        current_arc = data.arc_s_mm.iloc[index]
        source_index = min(len(arc) - 1, np.searchsorted(arc, current_arc))
        curvature = source.get('curvature_inv_mm', source.get('curvature_1_mm', source.get('kappa_1_mm', None)))
        if curvature is None:
            tangent = unit(source_points[min(source_index + 1, len(arc) - 1)] - source_points[max(source_index - 1, 0)])
            future_index = min(len(arc) - 1, source_index + 4)
            future = unit(source_points[future_index] - source_points[source_index])
            bend = np.linalg.norm(future - tangent) / max(arc[future_index] - arc[source_index], 1.)
        else:
            bend = abs(curvature.iloc[source_index])
        risk = data.risk_norm.iloc[index]
        clearance_limit = max(5., min(45., 6 + 2 * source.clearance_mm.iloc[source_index])) if 'clearance_mm' in source else 45.
        desired_preview = np.clip((12 + 1.8 * data.actual_speed_mm_s.iloc[index]) / (1 + 25 * bend + .8 * risk), 5, clearance_limit)
        selected = None
        hidden = False
        alignment_limit = SETTINGS['target_alignment_hold_cos'] if previous_preview is not None else SETTINGS['target_alignment_enter_cos']
        for preview in np.arange(desired_preview, 4, -3):
            target_arc = min(arc[-1], current_arc + SETTINGS['mount_offset_mm'] + preview)
            target = np.array([np.interp(target_arc, arc, source_points[:, component]) for component in range(3)])
            difference = target - origin
            if np.linalg.norm(difference) < .5 or unit(difference) @ matrices[index, :, 0] < alignment_limit:
                continue
            if blocked is not None and blocked(origin, target):
                hidden = True
                continue
            selected = (target, target_arc - current_arc)
            break
        if selected is not None:
            proposed_preview = selected[1]
            if previous_preview is not None:
                elapsed = data.time_s.iloc[index] - data.time_s.iloc[index - 1]
                maximum_change = SETTINGS['preview_slew_mm_s'] * max(float(elapsed), 0.)
                proposed_preview = previous_preview + np.clip(proposed_preview - previous_preview, -maximum_change, maximum_change)
        elif previous_preview is not None:
            proposed_preview = previous_preview
        else:
            proposed_preview = SETTINGS['mount_offset_mm'] + SETTINGS['preview_min_mm']
        target_arc = min(arc[-1], current_arc + proposed_preview)
        target = np.array([np.interp(target_arc, arc, source_points[:, component]) for component in range(3)])
        preview = target_arc - current_arc
        previous_preview = preview
        frame = forward_frame(target - origin, previous_up)
        target_forward = unit(target - origin)
        body_forward = matrices[index, :, 0]
        alignment = float(np.clip(np.dot(frame[:, 0], body_forward), -1, 1))
        axis_angle = np.arccos(alignment)
        maximum_axis_angle = np.deg2rad(SETTINGS['max_camera_axis_angle_deg'])
        if axis_angle > maximum_axis_angle:
            lateral = frame[:, 0] - alignment * body_forward
            if np.linalg.norm(lateral) < 1e-8:
                lateral = np.cross(body_forward, previous_up)
            limited_forward = np.cos(maximum_axis_angle) * body_forward + np.sin(maximum_axis_angle) * unit(lateral)
            frame = forward_frame(limited_forward, previous_up)
        relative = matrices[index].T @ frame
        euler = Rotation.from_matrix(relative).as_euler('ZYX', degrees=True)
        joint_limit = SETTINGS['max_joint_angle_deg']
        goal = np.clip(euler, [-joint_limit, -joint_limit, -180], [joint_limit, joint_limit, 180])
        goals.append(goal)
        goal_rotations.append((Rotation.from_matrix(matrices[index]) * Rotation.from_euler('ZYX', goal, degrees=True)).as_quat())
        cone_residuals.append(angle_between(target_forward, frame[:, 0]))
        targets.append(target)
        previews.append(preview)
        occluded.append(hidden and selected is None)
        previous_up = frame[:, 2]
    angles, velocity, acceleration = bounded_gimbal(data.time_s.to_numpy(), goals)
    rotations = body * Rotation.from_euler('ZYX', angles, degrees=True)
    frames = rotations.as_matrix()
    errors = np.rad2deg(np.arccos(np.clip(np.einsum('ij,ij->i', frames[:, :, 0], np.array([unit(target-origin) for target, origin in zip(targets, mounted)])), -1, 1)))
    tracking_errors = [quaternion_distance_deg(goal, actual) for goal, actual in zip(goal_rotations, rotations.as_quat())]
    return {'position': mounted.tolist(), 'quaternion': rotations.as_quat().tolist(), 'target': np.asarray(targets).tolist(),
            'goal_quaternion': np.asarray(goal_rotations).tolist(), 'cone_residual_deg': cone_residuals,
            'tracking_error_deg': tracking_errors,
            'preview_mm': previews, 'occluded': occluded, 'joint_angle_deg': angles.tolist(),
            'joint_velocity_deg_s': velocity.tolist(), 'joint_acceleration_deg_s2': acceleration.tolist(),
            'aim_error_deg': errors.tolist(), 'goal_joint_angle_deg': np.asarray(goals).tolist(),
            'time_s': data.time_s.tolist(), 'arc_s_mm': data.arc_s_mm.tolist()}


def baseline_frames(data, baseline):
    body = Rotation.from_quat(data[[f'actual_q{axis}' for axis in 'xyzw']].to_numpy()).as_matrix()
    positions = data[[f'actual_position_{axis}_mm' for axis in 'xyz']].to_numpy() + SETTINGS['mount_offset_mm'] * body[:, :, 0]
    queries = np.clip(data.arc_s_mm.to_numpy(), baseline['arc_s_mm'][0], baseline['arc_s_mm'][-1])
    rotations = Slerp(baseline['arc_s_mm'], Rotation.from_quat(baseline['quaternion']))(queries)
    frames = rotations.as_matrix()
    return {'position': positions.tolist(), 'quaternion': rotations.as_quat().tolist(),
            'target': (positions + frames[:, :, 0] * 12).tolist(), 'preview_mm': [12.] * len(data),
            'joint_angle_deg': (Rotation.from_matrix(body).inv() * rotations).as_euler('ZYX', degrees=True).tolist(),
            'time_s': data.time_s.tolist(), 'arc_s_mm': data.arc_s_mm.tolist()}


if __name__ == '__main__':
    (ROOT / 'camera_settings.json').write_text(json.dumps(SETTINGS, indent=2), encoding='utf-8')
