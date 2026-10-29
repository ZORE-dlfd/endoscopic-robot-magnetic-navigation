from __future__ import annotations

import numpy as np


MU0_OVER_4PI = 1e-7


def dipole_field(moment_a_m2, displacement_m):
    moment = np.asarray(moment_a_m2, dtype=float)
    displacement = np.asarray(displacement_m, dtype=float)
    radius_squared = float(displacement @ displacement)
    if radius_squared <= 0:
        raise ValueError("Field is undefined at the dipole location")
    radius = np.sqrt(radius_squared)
    projection = float(moment @ displacement)
    return MU0_OVER_4PI * (3.0 * displacement * projection / radius**5 - moment / radius**3)


def dipole_gradient(moment_a_m2, displacement_m):
    moment = np.asarray(moment_a_m2, dtype=float)
    displacement = np.asarray(displacement_m, dtype=float)
    radius_squared = float(displacement @ displacement)
    if radius_squared <= 0:
        raise ValueError("Field gradient is undefined at the dipole location")
    radius = np.sqrt(radius_squared)
    projection = float(moment @ displacement)
    identity = np.eye(3)
    numerator = (np.outer(displacement, moment) + np.outer(moment, displacement)
                 + projection * identity)
    return MU0_OVER_4PI * (3.0 * numerator / radius**5
                           - 15.0 * projection * np.outer(displacement, displacement) / radius**7)


def dipole_wrench(moment_a_m2, source_moment_a_m2, source_position_m, robot_position_m):
    displacement = np.asarray(robot_position_m, dtype=float) - np.asarray(source_position_m, dtype=float)
    field = dipole_field(source_moment_a_m2, displacement)
    gradient = dipole_gradient(source_moment_a_m2, displacement)
    robot_moment = np.asarray(moment_a_m2, dtype=float)
    force = gradient.T @ robot_moment
    torque = np.cross(robot_moment, field)
    return field, gradient, force, torque


def coil_responses(offsets_m, axes, dipole_moment_per_amp_a_m2):
    offsets = np.asarray(offsets_m, dtype=float)
    source_axes = np.asarray(axes, dtype=float)
    if (offsets.ndim != 3 or offsets.shape[-1] != 3 or offsets.shape[1] < 1
            or source_axes.shape != offsets.shape):
        raise ValueError("Expected candidate by coils by xyz arrays")
    displacement = -offsets
    radius_squared = np.sum(displacement * displacement, axis=-1)
    if np.any(radius_squared <= 0):
        raise ValueError("Coil cannot be located at the capsule position")
    radius = np.sqrt(radius_squared)
    moment = source_axes * float(dipole_moment_per_amp_a_m2)
    projection = np.sum(moment * displacement, axis=-1)
    fields = MU0_OVER_4PI * (
        3.0 * displacement * projection[..., None] / radius[..., None]**5
        - moment / radius[..., None]**3
    )
    identity = np.eye(3)
    numerator = (displacement[..., :, None] * moment[..., None, :]
                 + moment[..., :, None] * displacement[..., None, :]
                 + projection[..., None, None] * identity)
    gradients = MU0_OVER_4PI * (
        3.0 * numerator / radius[..., None, None]**5
        - 15.0 * projection[..., None, None]
        * displacement[..., :, None] * displacement[..., None, :]
        / radius[..., None, None]**7
    )
    return np.swapaxes(fields, 1, 2), gradients
