from __future__ import annotations

import numpy as np
from scipy.interpolate import CubicSpline
from scipy.spatial.transform import Rotation, Slerp


def effective_speed_cap(risk_limit_m_s, clearance_limit_m_s,
                        curvature_limit_m_s, camera_error_deg,
                        camera_rate_deg_s):
    route_cap = max(0.0, min(float(risk_limit_m_s),
                             float(clearance_limit_m_s),
                             float(curvature_limit_m_s)))
    dilation = np.clip(
        1.0 + max(float(camera_error_deg) - 15.0, 0.0) / 80.0
        + max(float(camera_rate_deg_s) - 50.0, 0.0) / 120.0,
        1.0,
        1.25,
    )
    return float(route_cap / dilation), float(dilation)


def progress_acceleration(tangential_speed_m_s, speed_ceiling_m_s,
                          accelerate_m_s2, brake_m_s2, response_time_s=0.25):
    speed_error = float(speed_ceiling_m_s) - float(tangential_speed_m_s)
    if speed_error > 0.0:
        return min(float(accelerate_m_s2), speed_error / response_time_s)
    if speed_error < 0.0:
        return -min(float(brake_m_s2), -speed_error / response_time_s)
    return 0.0


def path_force_command(position_m, velocity_m_s, route_position_m, route_tangent,
                       speed_ceiling_m_s, mass_kg, gravity_m_s2, resistance_n,
                       accelerate_m_s2, brake_m_s2, cross_track_gain_s2,
                       lateral_accel_limit_m_s2, cross_track_damping_s=0.0):
    position = np.asarray(position_m, dtype=float)
    velocity = np.asarray(velocity_m_s, dtype=float)
    tangent = np.asarray(route_tangent, dtype=float).copy()
    tangent /= max(float(np.linalg.norm(tangent)), 1e-12)
    tangential_speed = float(np.dot(velocity, tangent))
    longitudinal_acceleration = progress_acceleration(
        tangential_speed, speed_ceiling_m_s, accelerate_m_s2, brake_m_s2
    )
    route_error = np.asarray(route_position_m, dtype=float) - position
    lateral_error = route_error - np.dot(route_error, tangent) * tangent
    lateral_velocity = velocity - np.dot(velocity, tangent) * tangent
    lateral_acceleration = (float(cross_track_gain_s2) * lateral_error
                            - float(cross_track_damping_s) * lateral_velocity)
    lateral_norm = np.linalg.norm(lateral_acceleration)
    if lateral_norm > lateral_accel_limit_m_s2:
        lateral_acceleration *= lateral_accel_limit_m_s2 / lateral_norm
    acceleration = tangent * longitudinal_acceleration + lateral_acceleration
    force = (float(mass_kg) * (acceleration - np.asarray(gravity_m_s2, dtype=float))
             - np.asarray(resistance_n, dtype=float))
    return force, acceleration


def magnetic_path_demand(model, route_state, actual_state, config,
                         speed_ceiling_m_s, accelerate_m_s2,
                         brake_m_s2, lateral_accel_limit_m_s2):
    position = actual_state[:3]
    velocity = actual_state[3:6]
    resistance = model.resistance(velocity, config)
    inertia = model.ellipsoid_inertia(
        config["mass_kg"], config["robot_length_m"], config["robot_diameter_m"]
    )
    force_command, acceleration = path_force_command(
        position, velocity, route_state["position_m"], route_state["tangent"],
        speed_ceiling_m_s, config["mass_kg"], config["gravity_m_s2"], resistance,
        accelerate_m_s2, brake_m_s2, config["position_kp_s2"],
        lateral_accel_limit_m_s2, config.get("position_kd_s", 0.0),
    )
    quaternion = actual_state[6:10] / np.linalg.norm(actual_state[6:10])
    actual_rotation = Rotation.from_quat(quaternion)
    direction = actual_rotation.as_matrix()[:, 0]
    target_direction = route_state["rotation"].as_matrix()[:, 0]
    omega_world = actual_rotation.apply(actual_state[10:13])
    torque_command = inertia[1] * (
        config["axis_kp_s2"] * np.cross(direction, target_direction)
        - config["axis_kd_s"] * omega_world
    )
    moment = config["magnetic_moment_A_m2"] * direction
    field, supported_torque, unavailable_torque = model.inverse_field(
        moment, torque_command, config["bias_field_T"]
    )
    return {
        "field": field,
        "force_command": force_command,
        "torque_command": torque_command,
        "moment": moment,
        "resistance": resistance,
        "acceleration_command": acceleration,
        "supported_torque": supported_torque,
        "unavailable_torque": unavailable_torque,
    }


class SpatialRoute:
    def __init__(self, dataframe):
        self.arc_mm = dataframe["arc_s_mm"].to_numpy(dtype=float)
        self.arc_m = self.arc_mm / 1000.0
        self.points_m = dataframe[["x_mm", "y_mm", "z_mm"]].to_numpy(dtype=float) / 1000.0
        if (len(self.arc_m) < 4 or np.any(np.diff(self.arc_m) <= 0.0)
                or not np.isfinite(self.points_m).all()):
            raise ValueError("Spatial route requires finite points and increasing arc length")
        self.position_spline = CubicSpline(self.arc_m, self.points_m, axis=0)
        self.rotation_spline = Slerp(
            self.arc_m,
            Rotation.from_quat(dataframe[["qx", "qy", "qz", "qw"]].to_numpy(dtype=float)),
        )
        self.speed_limits = {
            name: dataframe[column].to_numpy(dtype=float) / 1000.0
            for name, column in (
                ("risk", "speed_risk_limit_mm_s"),
                ("clearance", "speed_clearance_limit_mm_s"),
                ("curvature", "speed_curvature_limit_mm_s"),
            )
        }
        self.risk = dataframe["risk_norm"].to_numpy(dtype=float)
        self.clearance_mm = dataframe["clearance_mm"].to_numpy(dtype=float)
        self.curvature_inv_mm = dataframe["curvature_inv_mm"].to_numpy(dtype=float)

    def at(self, arc_m):
        arc_m = float(np.clip(arc_m, self.arc_m[0], self.arc_m[-1]))
        tangent = np.asarray(self.position_spline(arc_m, 1), dtype=float)
        tangent /= max(float(np.linalg.norm(tangent)), 1e-12)
        return {
            "arc_m": arc_m,
            "position_m": np.asarray(self.position_spline(arc_m), dtype=float),
            "tangent": tangent,
            "rotation": self.rotation_spline(arc_m),
            "risk_norm": float(np.interp(arc_m, self.arc_m, self.risk)),
            "clearance_mm": float(np.interp(arc_m, self.arc_m, self.clearance_mm)),
            "curvature_inv_mm": float(np.interp(arc_m, self.arc_m, self.curvature_inv_mm)),
            "risk_speed_limit_m_s": float(np.interp(
                arc_m, self.arc_m, self.speed_limits["risk"])),
            "clearance_speed_limit_m_s": float(np.interp(
                arc_m, self.arc_m, self.speed_limits["clearance"])),
            "curvature_speed_limit_m_s": float(np.interp(
                arc_m, self.arc_m, self.speed_limits["curvature"])),
        }

    def speed_ceiling(self, arc_m, brake_decel_m_s2=0.008):
        arc_m = float(np.clip(arc_m, self.arc_m[0], self.arc_m[-1]))
        local_limits = np.minimum.reduce(list(self.speed_limits.values()))
        local_limits = local_limits.copy()
        local_limits[-1] = 0.0
        ahead = self.arc_m >= arc_m
        braking_envelope = np.sqrt(
            local_limits[ahead] ** 2
            + 2.0 * float(brake_decel_m_s2) * (self.arc_m[ahead] - arc_m)
        )
        return float(min(np.interp(arc_m, self.arc_m, local_limits),
                         np.min(braking_envelope)))

    def project_arc(self, position_m, anchor_arc_m=None, window_m=None):
        position = np.asarray(position_m, dtype=float)
        starts = self.points_m[:-1]
        segments = self.points_m[1:] - starts
        squared_lengths = np.einsum("ij,ij->i", segments, segments)
        fractions = np.einsum("ij,ij->i", position - starts, segments)
        fractions /= np.maximum(squared_lengths, 1e-18)
        fractions = np.clip(fractions, 0.0, 1.0)
        if anchor_arc_m is not None:
            if window_m is None or window_m <= 0.0:
                raise ValueError("A positive projection window is required with an anchor")
            lower_arc = max(self.arc_m[0], float(anchor_arc_m) - float(window_m))
            upper_arc = min(self.arc_m[-1], float(anchor_arc_m) + float(window_m))
            spans = np.maximum(np.diff(self.arc_m), 1e-18)
            fractions = np.clip(fractions,
                np.clip((lower_arc - self.arc_m[:-1]) / spans, 0., 1.),
                np.clip((upper_arc - self.arc_m[:-1]) / spans, 0., 1.))
        projections = starts + fractions[:, None] * segments
        distances_squared = np.einsum("ij,ij->i", projections - position,
                                      projections - position)
        if anchor_arc_m is not None:
            if window_m is None or window_m <= 0.0:
                raise ValueError("A positive projection window is required with an anchor")
            lower_arc = max(self.arc_m[0], float(anchor_arc_m) - float(window_m))
            upper_arc = min(self.arc_m[-1], float(anchor_arc_m) + float(window_m))
            segment_lower = self.arc_m[:-1]
            segment_upper = self.arc_m[1:]
            candidates = np.flatnonzero(
                (segment_upper >= lower_arc) & (segment_lower <= upper_arc)
            )
            if not len(candidates):
                candidates = np.array([int(np.argmin(
                    np.abs((segment_lower + segment_upper) * 0.5 - anchor_arc_m)
                ))])
            index = int(candidates[np.argmin(distances_squared[candidates])])
        else:
            index = int(np.argmin(distances_squared))
        return float(self.arc_m[index] + fractions[index]
                     * (self.arc_m[index + 1] - self.arc_m[index]))


