from __future__ import annotations

import numpy as np
from scipy.interpolate import CubicSpline
from scipy.spatial.transform import Rotation, RotationSpline


def scaled_parameters():
    ratio = 12.0 / 25.0 * (6.0 / 13.0) ** 2
    return {"volume_ratio": ratio, "mass_kg": 0.00516 * ratio,
            "magnetic_moment_A_m2": 0.253 * ratio}


def ellipsoid_inertia(mass, length, diameter):
    semiaxes = np.array([length / 2, diameter / 2, diameter / 2])
    return mass / 5 * (np.sum(semiaxes ** 2) - semiaxes ** 2)


def gradient_basis():
    return np.array([
        np.diag([1, -1, 0]) / np.sqrt(2),
        np.diag([1, 1, -2]) / np.sqrt(6),
        [[0, 1 / np.sqrt(2), 0], [1 / np.sqrt(2), 0, 0], [0, 0, 0]],
        [[0, 0, 1 / np.sqrt(2)], [0, 0, 0], [1 / np.sqrt(2), 0, 0]],
        [[0, 0, 0], [0, 0, 1 / np.sqrt(2)], [0, 1 / np.sqrt(2), 0]],
    ], dtype=float)


def minimum_gradient(moment, force):
    magnitude = np.linalg.norm(moment)
    if magnitude < 1e-12:
        raise ValueError("Magnetic moment must be nonzero")
    direction = moment / magnitude
    parallel = np.dot(force, direction) / magnitude
    transverse = (force - np.dot(force, direction) * direction) / magnitude
    projection = np.outer(direction, direction)
    return (parallel * (1.5 * projection - 0.5 * np.eye(3))
            + np.outer(transverse, direction) + np.outer(direction, transverse))


def inverse_field(moment, torque, bias):
    magnitude = np.linalg.norm(moment)
    if magnitude < 1e-12:
        raise ValueError("Magnetic moment must be nonzero")
    direction = moment / magnitude
    unavailable = np.dot(torque, direction) * direction
    supported = torque - unavailable
    field = np.cross(supported, moment) / magnitude ** 2 + bias * direction
    return field, supported, unavailable


def resistance(velocity, config):
    load = config["mass_kg"] * np.linalg.norm(config["gravity_m_s2"])
    denominator = np.sqrt(np.dot(velocity, velocity) + config["friction_smooth_speed_m_s"] ** 2)
    return -config["friction_coefficient"] * load * velocity / denominator


def quaternion_derivative(quaternion, omega_body):
    vector, scalar = quaternion[:3], quaternion[3]
    return 0.5 * np.r_[scalar * omega_body + np.cross(vector, omega_body),
                       -np.dot(vector, omega_body)]


class ReferenceTrajectory:
    def __init__(self, dataframe):
        self.times = dataframe["time_s"].to_numpy(dtype=float)
        positions = dataframe[["x_mm", "y_mm", "z_mm"]].to_numpy(dtype=float) / 1000
        quaternions = dataframe[["qx", "qy", "qz", "qw"]].to_numpy(dtype=float)
        if len(self.times) < 4 or np.any(np.diff(self.times) <= 0):
            raise ValueError("Trajectory requires at least four strictly ordered times")
        if not np.isfinite(positions).all() or not np.isfinite(quaternions).all():
            raise ValueError("Nonfinite trajectory data")
        self.position = CubicSpline(self.times, positions, axis=0,
                                    bc_type=((1, np.zeros(3)), (1, np.zeros(3))))
        self.rotation = RotationSpline(self.times, Rotation.from_quat(quaternions))

    def at(self, time):
        time = float(np.clip(time, self.times[0], self.times[-1]))
        rotation = self.rotation(time)
        return {"position": self.position(time), "velocity": self.position(time, 1),
                "acceleration": self.position(time, 2), "rotation": rotation,
                "omega_body": self.rotation(time, 1), "alpha_body": self.rotation(time, 2)}


def reference_demand(reference, time, config):
    state = reference.at(time)
    rotation = state["rotation"].as_matrix()
    inertia = ellipsoid_inertia(config["mass_kg"], config["robot_length_m"], config["robot_diameter_m"])
    gravity = np.array(config["gravity_m_s2"])
    resist = resistance(state["velocity"], config)
    force = config["mass_kg"] * (state["acceleration"] - gravity) - resist
    torque_body = inertia * state["alpha_body"] + np.cross(state["omega_body"], inertia * state["omega_body"])
    torque = rotation @ torque_body
    moment = config["magnetic_moment_A_m2"] * rotation[:, 0]
    field, supported, unavailable = inverse_field(moment, torque, config["bias_field_T"])
    gradient = minimum_gradient(moment, force)
    return {**state, "moment": moment, "field": field, "gradient": gradient,
            "force": force, "torque": torque, "supported_torque": supported,
            "unavailable_torque": unavailable, "resistance": resist}


def controlled_demand(reference, time, actual_state, config):
    target = reference.at(time)
    position, velocity = actual_state[:3], actual_state[3:6]
    quaternion = actual_state[6:10] / np.linalg.norm(actual_state[6:10])
    omega_body = actual_state[10:13]
    actual_rotation = Rotation.from_quat(quaternion).as_matrix()
    target_rotation = target["rotation"].as_matrix()
    direction = actual_rotation[:, 0]
    inertia = ellipsoid_inertia(config["mass_kg"], config["robot_length_m"], config["robot_diameter_m"])
    accel_command = (target["acceleration"]
                     + config["position_kp_s2"] * (target["position"] - position)
                     + config["position_kd_s"] * (target["velocity"] - velocity))
    resist = resistance(velocity, config)
    force_command = config["mass_kg"] * (accel_command - config["gravity_m_s2"]) - resist
    reference_torque_body = (inertia * target["alpha_body"]
                            + np.cross(target["omega_body"], inertia * target["omega_body"]))
    omega_actual_world = actual_rotation @ omega_body
    omega_target_world = target_rotation @ target["omega_body"]
    torque_command = (target_rotation @ reference_torque_body
                      + inertia[1] * config["axis_kp_s2"] * np.cross(direction, target_rotation[:, 0])
                      + inertia[1] * config["axis_kd_s"] * (omega_target_world - omega_actual_world))
    moment = config["magnetic_moment_A_m2"] * direction
    field, supported, unavailable = inverse_field(moment, torque_command, config["bias_field_T"])
    gradient = minimum_gradient(moment, force_command)
    magnetic_force = gradient.T @ moment
    magnetic_torque = np.cross(moment, field)
    acceleration = (magnetic_force + resist) / config["mass_kg"] + config["gravity_m_s2"]
    alpha_body = (actual_rotation.T @ magnetic_torque - np.cross(omega_body, inertia * omega_body)) / inertia
    derivative = np.r_[velocity, acceleration, quaternion_derivative(quaternion, omega_body), alpha_body]
    return {"derivative": derivative, "field": field, "gradient": gradient,
            "force": magnetic_force, "force_command": force_command, "torque": magnetic_torque,
            "torque_command": torque_command, "supported_torque": supported,
            "unavailable_torque": unavailable, "moment": moment, "resistance": resist,
            "acceleration": acceleration, "alpha_body": alpha_body,
            "direction": direction, "target_direction": target_rotation[:, 0], "quaternion": quaternion}
