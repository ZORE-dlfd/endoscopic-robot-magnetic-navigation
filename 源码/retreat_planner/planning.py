from dataclasses import dataclass, replace

import numpy as np
from scipy.spatial.transform import Rotation

from .controller import advance
from .coverage import choose_near_best


@dataclass
class Plan:
    target: np.ndarray
    delay_s: float = 0.

    def next_target(self, state):
        return state.goal.copy() if self.delay_s > 0 else self.target.copy()


@dataclass
class Prediction:
    gain: float
    cost: float
    feasible: bool
    max_cone: float


def observe_transition(transition, start_time, coverage, scene, visibility, config):
    duration = float(transition.offsets[-1])
    offsets = np.r_[np.arange(config.visibility_s, duration - 1e-8, config.visibility_s), duration]
    previous = 0.
    for offset in offsets:
        index = int(np.argmin(np.abs(transition.offsets - offset)))
        time = min(scene.duration, start_time + float(offset))
        origin, body = scene.pose(time)
        frame = body @ Rotation.from_euler('ZYX', transition.angles[index], degrees=True).as_matrix()
        coverage.observe(visibility.visible(origin, frame), time, float(offset - previous))
        previous = offset


def rollout(plan, state, coverage, time, scene, visibility, config):
    predicted = coverage.clone()
    before = predicted.added_area
    horizon = min(config.horizon_s, scene.duration - time)
    elapsed, cost, max_cone = 0., 0., 0.
    previous_acceleration = None
    while elapsed < horizon - 1e-9:
        duration = min(config.decision_s, horizon - elapsed)
        goal = state.goal if elapsed < plan.delay_s - 1e-8 else plan.target
        transition = advance(state, goal, duration, config.integration_s)
        max_cone = max(max_cone, transition.max_cone_deg)
        if not transition.feasible:
            return Prediction(0., float('inf'), False, max_cone)
        normalized_velocity = transition.velocities / 60.
        normalized_acceleration = transition.accelerations / 150.
        acceleration_chain = transition.accelerations
        if previous_acceleration is not None:
            acceleration_chain = np.vstack([previous_acceleration, acceleration_chain])
        normalized_jerk = np.diff(acceleration_chain, axis=0) / config.integration_s / 1500.
        cost += float((np.sum(normalized_velocity ** 2) * .05 + np.sum(normalized_acceleration ** 2) +
                       .1 * np.sum(normalized_jerk ** 2)) * config.integration_s)
        observe_transition(transition, time + elapsed, predicted, scene, visibility, config)
        previous_acceleration = transition.accelerations[-1]
        state = transition.state
        elapsed += duration
    return Prediction(predicted.added_area - before, cost, True, max_cone)


def select_predictive(goals, state, coverage, time, scene, visibility, config):
    plans = [Plan(goal) for goal in goals]
    if config.include_delayed:
        first_step = rollout(Plan(state.goal.copy()), state, coverage, time, scene, visibility,
                             replace(config, horizon_s=config.decision_s))
        if first_step.feasible and first_step.gain > 1e-9:
            plans += [Plan(goal, config.decision_s) for goal in goals if np.linalg.norm((goal - state.goal + 180) % 360 - 180) > 1e-5]
    predictions = [rollout(plan, state, coverage, time, scene, visibility, config) for plan in plans]
    feasible = [index for index, prediction in enumerate(predictions) if prediction.feasible]
    if not feasible:
        return Plan(state.goal.copy()), dict(fallback=True, predicted_gain_mm2=0., predicted_cost=None, prediction_count=len(plans))
    chosen = choose_near_best([predictions[index].gain for index in feasible], [predictions[index].cost for index in feasible], config.gain_tolerance)
    index = feasible[chosen]
    prediction = predictions[index]
    return plans[index], dict(fallback=False, predicted_gain_mm2=prediction.gain, predicted_cost=prediction.cost,
                              prediction_count=len(plans), delay_s=plans[index].delay_s)
