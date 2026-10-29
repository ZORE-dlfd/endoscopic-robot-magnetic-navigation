from dataclasses import dataclass
import time

import numpy as np
from scipy.spatial.transform import Rotation

from retreat_planner.controller import _advance_gimbal
from . import model
from .path_guidance import SpatialRoute, effective_speed_cap, magnetic_path_demand
from .redundant_actuator import default_array_layout, allocate_field_priority
from .coil_physics import coil_responses


@dataclass
class Settings:
    step_s: float = .02
    output_s: float = .1
    control_s: float = .2
    moment_per_amp: float = 20.
    current_limit_a: float = 10.
    coil_speed_m_s: float = .2
    max_phase_s: float = 700.
    endpoint_arc_tolerance_m: float = .00015
    endpoint_position_tolerance_m: float = .001


class CameraPolicy:
    def __init__(self, replay, reverse):
        selected = replay.camera_times >= replay.join_time if reverse else replay.camera_times <= replay.join_time
        arcs = np.asarray(replay.camera['arc_s_mm'])[selected] / 1000.
        order = np.argsort(arcs, kind='stable')
        arcs, indices = np.unique(arcs[order], return_index=True)
        self.arcs = arcs
        self.goals = np.asarray(replay.camera['goal_joint_angle_deg'])[selected][order][indices]

    def goal(self, source_arc, reference_rotation, actual_rotation):
        angles = np.array([np.interp(source_arc, self.arcs, self.goals[:, axis]) for axis in range(3)])
        world = reference_rotation * Rotation.from_euler('ZYX', angles, degrees=True)
        result = (actual_rotation.inv() * world).as_euler('ZYX', degrees=True)
        forward = (actual_rotation.inv() * world).as_matrix()[:, 0]
        cone = np.rad2deg(np.arccos(np.clip(forward[0], -1., 1.)))
        if cone > 44.5:
            result[:2] *= 44.5 / cone
        result[:2] = np.clip(result[:2], -89., 89.)
        return result


def make_route(source, reverse=False):
    data = source.copy()
    if reverse:
        data = data.iloc[::-1].reset_index(drop=True)
        data['arc_s_mm'] = float(source['arc_s_mm'].iloc[-1]) - data['arc_s_mm']
    return SpatialRoute(data)


def angle_between(first, second):
    return float(np.rad2deg(np.arccos(np.clip(np.dot(first, second) /
        max(np.linalg.norm(first) * np.linalg.norm(second), 1e-18), -1., 1.))))


def evaluate(state, route, progress, source_positions, source_axes, config, settings, acceleration_limits,
             camera_angles, camera_velocity, policy, reverse, goal_override=None):
    arc = route.project_arc(state[:3], progress, .025)
    reference = route.at(arc)
    body = Rotation.from_quat(state[6:10] / np.linalg.norm(state[6:10]))
    source_arc = route.arc_m[-1] - arc if reverse else arc
    goal = policy.goal(source_arc, reference['rotation'], body) if goal_override is None else goal_override
    camera_error = float(np.max(np.abs((goal-camera_angles+180.) % 360.-180.)))
    route_cap = route.speed_ceiling(arc, acceleration_limits['brake'])
    speed_cap, dilation = effective_speed_cap(route_cap, route_cap, route_cap, camera_error,
                                              np.max(np.abs(camera_velocity)))
    demand = magnetic_path_demand(model, reference, state, config, speed_cap,
        acceleration_limits['accelerate'], acceleration_limits['brake'], acceleration_limits['lateral'])
    fields, gradients = coil_responses((source_positions - state[:3])[None],
        source_axes[None], settings.moment_per_amp)
    force_matrix = np.einsum('nji,j->in', gradients[0], demand['moment'])
    allocation = allocate_field_priority(fields[0], force_matrix, demand['field'],
        demand['force_command'], settings.current_limit_a)
    currents = allocation['currents_a']
    field = fields[0] @ currents
    gradient = np.einsum('i,ijk->jk', currents, gradients[0])
    force = gradient.T @ demand['moment']
    torque = np.cross(demand['moment'], field)
    inertia = model.ellipsoid_inertia(config['mass_kg'], config['robot_length_m'], config['robot_diameter_m'])
    derivative = np.empty(13)
    derivative[:3] = state[3:6]
    derivative[3:6] = (force + demand['resistance']) / config['mass_kg'] + config['gravity_m_s2']
    derivative[6:10] = model.quaternion_derivative(state[6:10], state[10:13])
    derivative[10:13] = (body.inv().apply(torque) - np.cross(state[10:13], inertia * state[10:13])) / inertia
    return dict(derivative=derivative, arc=arc, reference=reference, body=body, source_arc=source_arc,
        goal=goal, camera_error=camera_error, dilation=dilation, speed_cap=speed_cap, route_cap=route_cap,
        demand=demand, currents=currents, field=field, gradient=gradient, force=force, torque=torque,
        force_error=allocation['force_error_n'], field_error=allocation['field_error_t'])


def simulate_phase(source, config, risk_assumptions, replay, reverse=False, initial=None, settings=None,
                   stop_after_s=None, camera_planner=None):
    settings = settings or Settings()
    route = make_route(source, reverse)
    original_arcs = route.arc_m.copy()
    offsets, axes = default_array_layout()
    limits = dict(accelerate=float(risk_assumptions['tangent_accel_max']) / 1000.,
                  brake=float(risk_assumptions['brake_decel_max']) / 1000.,
                  lateral=float(risk_assumptions['normal_accel_max']) / 1000.)
    policy = CameraPolicy(replay, reverse)
    if initial is None:
        state = np.zeros(13)
        state[:3] = route.at(0.)['position_m']
        state[6:10] = (Rotation.from_rotvec(np.deg2rad([0., 0., 3.])) * route.at(0.)['rotation']).as_quat()
        camera_angles = policy.goal(0., route.at(0.)['rotation'], Rotation.from_quat(state[6:10]))
        camera_velocity = np.zeros(3)
        anchor = state[:3].copy()
    else:
        state = initial['state'].copy()
        camera_angles = initial['camera_angles'].copy()
        camera_velocity = initial['camera_velocity'].copy()
        anchor = initial['anchor'].copy()
    progress = route.project_arc(state[:3], 0., .025)
    rows = []
    time_s = 0.
    control_started = 0.
    anchor_start = anchor.copy()
    anchor_target = anchor.copy()
    last_camera_velocity = camera_velocity.copy()
    started = time.perf_counter()
    step_count = 0
    complete = False
    max_time = min(settings.max_phase_s, stop_after_s) if stop_after_s is not None else settings.max_phase_s
    while time_s <= max_time + 1e-9:
        if camera_planner is not None:
            camera_planner.update(time_s,state,progress,route,camera_angles,camera_velocity)
        if time_s >= control_started + settings.control_s - 1e-9 or step_count == 0:
            control_started = time_s
            anchor_start = anchor.copy()
            movement = state[:3] - anchor_start
            movement *= min(1., settings.coil_speed_m_s * settings.control_s / max(np.linalg.norm(movement), 1e-18))
            anchor_target = anchor_start + movement

        def at(candidate, offset):
            amount = np.clip((time_s + offset - control_started) / settings.control_s, 0., 1.)
            location = anchor_start + amount * (anchor_target-anchor_start)
            return evaluate(candidate, route, progress, location[None] + offsets, axes,
                config, settings, limits, camera_angles, camera_velocity, policy, reverse,
                None if camera_planner is None else camera_planner.target)

        current = at(state, 0.)
        progress = current['arc']
        amount = np.clip((time_s-control_started) / settings.control_s, 0., 1.)
        anchor = anchor_start + amount * (anchor_target-anchor_start)
        endpoint_distance = np.linalg.norm(state[:3] - route.points_m[-1])
        complete = (route.arc_m[-1]-progress <= settings.endpoint_arc_tolerance_m
                    and endpoint_distance <= settings.endpoint_position_tolerance_m)
        if step_count % round(settings.output_s/settings.step_s) == 0 or complete:
            rows.append(dict(time_s=time_s, state=state.copy(), camera_angles=camera_angles.copy(),
                camera_velocity=camera_velocity.copy(), camera_acceleration=(camera_velocity-last_camera_velocity)/settings.step_s,
                anchor=anchor.copy(), positions=anchor[None]+offsets, reverse=reverse, **current))
        if complete or time_s >= max_time - 1e-9:
            break
        step = min(settings.step_s, max_time-time_s)
        first = current['derivative']
        second = at(state + first*step/2, step/2)['derivative']
        third = at(state + second*step/2, step/2)['derivative']
        fourth = at(state + third*step, step)['derivative']
        next_state = state + step * (first+2*second+2*third+fourth)/6
        next_state[6:10] /= np.linalg.norm(next_state[6:10])
        last_camera_velocity = camera_velocity.copy()
        elapsed = 0.
        while elapsed < step-1e-10:
            camera_step = min(.01,step-elapsed)
            goal = current['goal'] if camera_planner is None else camera_planner.actuator_goal(time_s+elapsed+camera_step)
            previous_velocity = camera_velocity.copy()
            camera_angles, camera_velocity = _advance_gimbal(camera_angles,camera_velocity,goal,camera_step)
            if camera_planner is not None:
                camera_planner.check_motion(camera_angles,camera_velocity,(camera_velocity-previous_velocity)/camera_step)
            elapsed += camera_step
        state = next_state
        step_count += 1
        time_s = step_count * settings.step_s
        if step_count % round(20/settings.step_s) == 0:
            print(f'{"withdrawal" if reverse else "insertion"}: {time_s:.0f}s, '
                  f'{progress*1000:.1f}/{route.arc_m[-1]*1000:.1f}mm, '
                  f'elapsed={time.perf_counter()-started:.1f}s', flush=True)
    if not np.array_equal(route.arc_m, original_arcs) or np.any(np.diff(route.arc_m) <= 0):
        raise RuntimeError('Route arc coordinates were modified')
    if stop_after_s is None and not complete:
        raise RuntimeError(f'Magnetic motion did not reach endpoint: {progress*1000:.2f}mm, distance={endpoint_distance*1000:.2f}mm')
    final = dict(state=state.copy(), camera_angles=camera_angles.copy(),
                 camera_velocity=camera_velocity.copy(), anchor=anchor.copy())
    report = dict(complete=bool(complete), duration_s=time_s, endpoint_distance_mm=float(endpoint_distance*1000),
        progress_mm=float(progress*1000), route_length_mm=float(route.arc_m[-1]*1000),
        wall_seconds=time.perf_counter()-started, step_s=settings.step_s,
        peak_current_a=float(max(np.max(np.abs(row['currents'])) for row in rows)),
        cross_track_max_mm=float(max(np.linalg.norm(row['state'][:3]-row['reference']['position_m'])*1000 for row in rows)),
        camera_wait_count=0, route_arc_unchanged=True)
    return rows, final, report
