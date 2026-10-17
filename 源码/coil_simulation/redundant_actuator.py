from __future__ import annotations

import numpy as np
from scipy.optimize import LinearConstraint, lsq_linear, minimize

from .coil_physics import coil_responses


def default_array_layout():
    indices = np.arange(8)
    azimuth = indices * np.pi / 4.0
    polar = np.deg2rad(np.where(indices % 2, 60.0, 35.0))
    radii = np.where(indices % 2, 0.13, 0.16)
    radial = np.column_stack((np.sin(polar) * np.cos(azimuth),
                              np.sin(polar) * np.sin(azimuth), np.cos(polar)))
    return radii[:, None] * radial, -radial


def array_wrench_matrices(offsets_m, axes, moment_a_m2, source_strength):
    fields, gradients = coil_responses(
        np.asarray(offsets_m)[None], np.asarray(axes)[None], source_strength,
    )
    forces = np.einsum('nji,j->in', gradients[0], np.asarray(moment_a_m2))
    return fields[0], forces


def allocate_field_priority(field_matrix, force_matrix, desired_field_t,
                            desired_force_n, current_limit_a):
    fields = np.asarray(field_matrix, dtype=float)
    forces = np.asarray(force_matrix, dtype=float)
    target_field = np.asarray(desired_field_t, dtype=float)
    target_force = np.asarray(desired_force_n, dtype=float)
    base = np.linalg.lstsq(fields, target_field, rcond=None)[0]
    if np.max(np.abs(base)) > current_limit_a:
        base = lsq_linear(fields, target_field,
                          bounds=(-current_limit_a, current_limit_a),
                          tol=1e-12, lsmr_tol=1e-12).x
    _, singular_values, right = np.linalg.svd(fields, full_matrices=True)
    rank = int(np.sum(singular_values > singular_values[0] * 1e-12))
    nullspace = right[rank:].T
    currents = base.copy()
    if nullspace.shape[1]:
        scale = max(np.linalg.norm(target_force), 1e-6)
        matrix = forces @ nullspace / scale
        target = (target_force - forces @ base) / scale
        coordinates = np.linalg.lstsq(matrix, target, rcond=None)[0]
        candidate = base + nullspace @ coordinates
        if np.max(np.abs(candidate)) <= current_limit_a:
            currents = candidate
        else:
            def objective(values):
                residual = matrix @ values - target
                return float(residual @ residual)

            def jacobian(values):
                return 2.0 * matrix.T @ (matrix @ values - target)

            result = minimize(
                objective, np.zeros(nullspace.shape[1]), jac=jacobian,
                method='SLSQP', constraints=[LinearConstraint(
                    nullspace, -current_limit_a - base, current_limit_a - base,
                )], options={'maxiter': 80, 'ftol': 1e-12},
            )
            candidate = base + nullspace @ result.x
            if (np.max(np.abs(candidate)) <= current_limit_a + 1e-9
                    and objective(result.x) < objective(np.zeros(nullspace.shape[1]))):
                currents = candidate
    return {'currents_a': currents, 'current_peak_a': float(np.max(np.abs(currents))),
            'field_error_t': float(np.linalg.norm(fields @ currents - target_field)),
            'force_error_n': float(np.linalg.norm(forces @ currents - target_force))}


def plan_array_pose(robot_position_m, desired_field_t, desired_force_n,
                    moment_a_m2, source_strength, current_limit_a,
                    previous_positions_m=None, delta_time_s=0.2,
                    max_translation_speed_m_s=0.20):
    offsets, axes = default_array_layout()
    anchor = np.asarray(robot_position_m, dtype=float).copy()
    speed = 0.0
    if previous_positions_m is not None:
        previous_anchor = np.asarray(previous_positions_m)[0] - offsets[0]
        movement = anchor - previous_anchor
        distance = np.linalg.norm(movement)
        movement *= min(1.0, max_translation_speed_m_s * delta_time_s / max(distance, 1e-12))
        anchor = previous_anchor + movement
        speed = float(np.linalg.norm(movement) / delta_time_s)
    offsets = offsets + anchor - np.asarray(robot_position_m)
    fields, forces = array_wrench_matrices(offsets, axes, moment_a_m2, source_strength)
    allocation = allocate_field_priority(fields, forces, desired_field_t,
                                         desired_force_n, current_limit_a)
    return {'parameters': None, 'offsets_m': offsets, 'axes': axes, 'feasible': True,
            'evaluations': 1, 'translation_speed_m_s': speed, **allocation}
