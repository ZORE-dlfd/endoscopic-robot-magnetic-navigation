import numpy as np


def unit(vector):
    vector = np.asarray(vector, dtype=float)
    return vector / max(float(np.linalg.norm(vector)), 1e-12)


def angle_between(first, second):
    return float(np.rad2deg(np.arccos(np.clip(np.dot(unit(first), unit(second)), -1.0, 1.0))))


def quaternion_distance_deg(first, second):
    first = np.array(first, dtype=float, copy=True)
    second = np.array(second, dtype=float, copy=True)
    first /= max(np.linalg.norm(first), 1e-12)
    second /= max(np.linalg.norm(second), 1e-12)
    dot = np.clip(abs(float(np.dot(first, second))), 0.0, 1.0)
    return float(np.rad2deg(2.0 * np.arccos(dot)))


def classify_orientation_error(target_direction, feasible_direction, desired_quaternion, actual_quaternion):
    return {
        'cone_error_deg': angle_between(target_direction, feasible_direction),
        'tracking_error_deg': quaternion_distance_deg(desired_quaternion, actual_quaternion),
    }


def speed_dilation(tracking_error_deg, desired_rate_deg_s, deadband_deg=15.0,
                   rate_soft_limit_deg_s=50.0, max_factor=1.25):
    tracking_error, desired_rate = np.broadcast_arrays(
        np.asarray(tracking_error_deg, dtype=float), np.asarray(desired_rate_deg_s, dtype=float))
    lag_term = np.maximum(tracking_error - deadband_deg, 0.0) / 80.0
    rate_term = np.maximum(desired_rate - rate_soft_limit_deg_s, 0.0) / 120.0
    factor = np.clip(1.0 + lag_term + rate_term, 1.0, max_factor)
    return float(factor) if factor.ndim == 0 else factor


def _advance_gimbal(angle, velocity, goal, step, angle_limit=89.0, speed_limit=60.0, acceleration_limit=150.0):
    error = (goal - angle + 180.0) % 360.0 - 180.0
    acceleration = np.clip(12.0 * error - 6.0 * velocity, -acceleration_limit, acceleration_limit)
    candidate = np.clip(velocity + acceleration * step, -speed_limit, speed_limit)
    for axis in range(2):
        distance = max(0.0, angle_limit - np.sign(candidate[axis]) * angle[axis])
        brake_speed = np.sqrt(2.0 * acceleration_limit * distance)
        if abs(candidate[axis]) > brake_speed:
            acceleration[axis] = -np.sign(velocity[axis]) * acceleration_limit
            candidate[axis] = velocity[axis] + acceleration[axis] * step
    next_angle = angle + (velocity + candidate) * 0.5 * step
    next_angle[:2] = np.clip(next_angle[:2], -angle_limit, angle_limit)
    next_angle[2] = (next_angle[2] + 180.0) % 360.0 - 180.0
    return next_angle, candidate


def integrate_gimbal_segment(angle, velocity, start_goal, end_goal, duration_s, step_s=0.01):
    angle = np.asarray(angle, dtype=float).copy()
    velocity = np.asarray(velocity, dtype=float).copy()
    start_goal = np.asarray(start_goal, dtype=float)
    end_goal = np.asarray(end_goal, dtype=float)
    elapsed = 0.0
    while elapsed < duration_s - 1e-12:
        step = min(step_s, duration_s - elapsed)
        ratio = min(1.0, (elapsed + step) / max(duration_s, 1e-12))
        goal = start_goal + ratio * ((end_goal - start_goal + 180.0) % 360.0 - 180.0)
        angle, velocity = _advance_gimbal(angle, velocity, goal, step)
        elapsed += step
    return angle, velocity


def prune_unsettled_waits(schedule_times, source_times, waits, camera,
                         lag_trigger_deg=24.0, lag_release_deg=3.0, speed_release_deg_s=5.0):
    schedule_times = np.asarray(schedule_times, dtype=float)
    source_times = np.asarray(source_times, dtype=float)
    camera_times = np.asarray(camera['time_s'], dtype=float)
    goals = np.asarray(camera['goal_joint_angle_deg'], dtype=float)
    angles = np.asarray(camera['joint_angle_deg'], dtype=float)
    errors = np.max(np.abs((goals - angles + 180.0) % 360.0 - 180.0), axis=1)
    speeds = np.max(np.abs(camera['joint_velocity_deg_s']), axis=1)
    keep_knots = np.ones(len(schedule_times), dtype=bool)
    adjusted_times = schedule_times.copy()
    kept, rejected = [], []
    removed_duration = 0.0
    for wait in sorted(waits, key=lambda event: event['start_s']):
        start = wait['start_s']
        end = start + wait['duration_s']
        start_matches = np.flatnonzero(np.isclose(camera_times, start, atol=1e-8, rtol=0.0))
        end_matches = np.flatnonzero(np.isclose(camera_times, end, atol=1e-8, rtol=0.0))
        if not len(start_matches) or not len(end_matches):
            raise ValueError('wait release verification requires exact boundary samples')
        start_index, end_index = start_matches[0], end_matches[0]
        verified = {**wait, 'replay_start_axis_error_deg': float(errors[start_index]),
                    'replay_end_axis_error_deg': float(errors[end_index]),
                    'replay_end_axis_speed_deg_s': float(speeds[end_index])}
        if (errors[start_index] >= lag_trigger_deg and errors[end_index] <= lag_release_deg
                and speeds[end_index] <= speed_release_deg_s):
            kept.append({**verified, 'start_s': start - removed_duration})
            continue
        end_knots = np.flatnonzero(np.isclose(schedule_times, end, atol=1e-8, rtol=0.0))
        if len(end_knots) != 1 or end_knots[0] == 0:
            raise ValueError('wait end must match a unique schedule knot')
        knot = end_knots[0]
        if not np.isclose(source_times[knot], source_times[knot - 1], atol=1e-8, rtol=0.0):
            raise ValueError('cannot remove a moving segment as a wait')
        keep_knots[knot] = False
        adjusted_times[knot:] -= wait['duration_s']
        removed_duration += wait['duration_s']
        rejected.append(verified)
    return adjusted_times[keep_knots], source_times[keep_knots], kept, rejected


def build_schedule(source_times, goals_deg, factors, *, add_feedback_waits=False,
                   lag_trigger_deg=24.0, lag_release_deg=3.0, cooldown_s=8.0,
                   max_wait_s=1.2, max_total_wait_s=None, step_s=0.01):
    source_times = np.asarray(source_times, dtype=float)
    goals = np.asarray(goals_deg, dtype=float)
    factors = np.asarray(factors, dtype=float)
    if len(source_times) != len(goals) or len(goals) != len(factors):
        raise ValueError('source times, goals, and factors must have equal lengths')
    if len(source_times) < 2 or np.any(np.diff(source_times) <= 0):
        raise ValueError('source times must be strictly increasing')
    if np.any(factors < 1.0) or not np.isfinite(factors).all():
        raise ValueError('time-dilation factors must be finite and at least one')
    if max_total_wait_s is None:
        max_total_wait_s = 0.03 * (source_times[-1] - source_times[0])
    if not np.isfinite(max_total_wait_s) or max_total_wait_s < 0.0:
        raise ValueError('total wait budget must be finite and nonnegative')

    new_times = [0.0]
    original_times = [float(source_times[0])]
    waits = []
    wait_used_s = 0.0
    angle = goals[0].copy()
    velocity = np.zeros(3)
    last_wait_end = -np.inf
    for index in range(1, len(source_times)):
        interval = (source_times[index] - source_times[index - 1]) * (factors[index] + factors[index - 1]) / 2.0
        angle, velocity = integrate_gimbal_segment(angle, velocity, goals[index - 1], goals[index], interval, step_s)
        new_time = new_times[-1] + interval
        new_times.append(new_time)
        original_times.append(float(source_times[index]))
        lag = np.max(np.abs((goals[index] - angle + 180.0) % 360.0 - 180.0))
        remaining_budget = max_total_wait_s - wait_used_s
        if (add_feedback_waits and lag > lag_trigger_deg and remaining_budget > 1e-9
                and new_time - last_wait_end >= cooldown_s
                and index < len(source_times) - 1):
            wait = estimate_settle_time(angle, velocity, goals[index], threshold_deg=lag_release_deg,
                                        max_wait_s=min(max_wait_s, remaining_budget), step_s=step_s)
            if wait is not None and wait > 0.0:
                angle, velocity = integrate_gimbal_segment(angle, velocity, goals[index], goals[index], wait, step_s)
                new_time += wait
                new_times.append(new_time)
                original_times.append(float(source_times[index]))
                waits.append({'start_s': float(new_time - wait), 'duration_s': float(wait),
                              'source_index': index, 'lag_before_deg': float(lag),
                              'lag_after_deg': float(np.max(np.abs((goals[index] - angle + 180.0) % 360.0 - 180.0)))})
                wait_used_s += wait
                last_wait_end = new_time
    return np.asarray(new_times), np.asarray(original_times), waits


def estimate_settle_time(angle, velocity, goal, *, reachable=True, threshold_deg=3.0,
                         speed_threshold_deg_s=5.0, max_wait_s=1.2, step_s=0.01):
    angle = np.asarray(angle, dtype=float).copy()
    velocity = np.asarray(velocity, dtype=float).copy()
    goal = np.asarray(goal, dtype=float)
    if not reachable or np.any(np.abs(goal[:2]) > 89.0):
        return None

    def settled():
        error = (goal - angle + 180.0) % 360.0 - 180.0
        return np.max(np.abs(error)) <= threshold_deg and np.max(np.abs(velocity)) <= speed_threshold_deg_s

    if settled():
        return 0.0
    elapsed = 0.0
    while elapsed < max_wait_s - 1e-12:
        interval = min(step_s, max_wait_s - elapsed)
        angle, velocity = _advance_gimbal(angle, velocity, goal, interval)
        elapsed += interval
        if settled():
            return elapsed
    return None
