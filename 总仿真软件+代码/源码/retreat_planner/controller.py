from dataclasses import dataclass

import numpy as np

def _advance_gimbal(angle, velocity, goal, step):
    error = (goal - angle + 180.) % 360. - 180.
    acceleration = np.clip(12. * error - 6. * velocity, -150., 150.)
    candidate = np.clip(velocity + acceleration * step, -60., 60.)
    for axis in range(2):
        distance = max(0., 89. - np.sign(candidate[axis]) * angle[axis])
        brake_speed = np.sqrt(2. * 150. * distance)
        if abs(candidate[axis]) > brake_speed:
            acceleration[axis] = -np.sign(velocity[axis]) * 150.
            candidate[axis] = velocity[axis] + acceleration[axis] * step
    next_angle = angle + (velocity + candidate) * .5 * step
    next_angle[:2] = np.clip(next_angle[:2], -89., 89.)
    next_angle[2] = (next_angle[2] + 180.) % 360. - 180.
    return next_angle, candidate


@dataclass
class State:
    angle: np.ndarray
    velocity: np.ndarray
    goal: np.ndarray


@dataclass
class Transition:
    state: State
    offsets: np.ndarray
    angles: np.ndarray
    velocities: np.ndarray
    accelerations: np.ndarray
    max_cone_deg: float
    feasible: bool


def advance(state, target, duration, step_s=.01):
    if duration <= 0 or step_s <= 0:
        raise ValueError('Duration and integration step must be positive')
    target = np.asarray(target, dtype=float)
    angle, velocity = state.angle.copy(), state.velocity.copy()
    offsets, angles, velocities, accelerations = [], [], [], []
    elapsed = 0.
    while elapsed < duration - 1e-12:
        step = min(step_s, duration - elapsed)
        ratio = min(1., (elapsed + step) / duration)
        goal = state.goal + ratio * ((target - state.goal + 180.) % 360. - 180.)
        next_angle, next_velocity = _advance_gimbal(angle, velocity, goal, step)
        accelerations.append((next_velocity - velocity) / step)
        angle, velocity = next_angle, next_velocity
        elapsed += step
        offsets.append(elapsed)
        angles.append(angle.copy())
        velocities.append(velocity.copy())
    angles, velocities, accelerations = map(np.asarray, (angles, velocities, accelerations))
    forward_cosine = np.cos(np.deg2rad(angles[:, 0])) * np.cos(np.deg2rad(angles[:, 1]))
    initial_cosine = np.cos(np.deg2rad(state.angle[0])) * np.cos(np.deg2rad(state.angle[1]))
    cone = np.rad2deg(np.arccos(np.clip(np.r_[initial_cosine, forward_cosine], -1, 1)))
    maximum = float(cone.max())
    feasible = bool(maximum <= 45 + 1e-7 and np.max(np.abs(angles[:, :2])) <= 89 + 1e-7 and
                    np.max(np.abs(velocities)) <= 60 + 1e-7 and np.max(np.abs(accelerations)) <= 150 + 1e-5)
    return Transition(State(angle, velocity, target.copy()), np.asarray(offsets), angles, velocities,
                      accelerations, maximum, feasible)
