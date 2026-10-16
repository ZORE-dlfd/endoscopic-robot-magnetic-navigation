import numpy as np


def reverse_reference_arrays(time_s, position_mm, quaternion_xyzw, arc_s_mm):
    times = np.asarray(time_s, dtype=float)
    positions = np.asarray(position_mm, dtype=float)
    quaternions = np.asarray(quaternion_xyzw, dtype=float)
    arc = np.asarray(arc_s_mm, dtype=float)
    count = len(times)
    if count < 2 or times.ndim != 1 or np.any(np.diff(times) <= 0):
        raise ValueError('source times must be strictly increasing and contain at least two samples')
    if positions.shape != (count, 3) or quaternions.shape != (count, 4) or arc.shape != (count,):
        raise ValueError('positions, quaternions, and arc must align with source times')
    if not all(np.isfinite(values).all() for values in (times, positions, quaternions, arc)):
        raise ValueError('reference arrays must be finite')
    if np.any(np.diff(arc) <= 0):
        raise ValueError('source arc length must be strictly increasing')
    reverse_quaternions = quaternions[::-1].copy()
    norms = np.linalg.norm(reverse_quaternions, axis=1)
    if np.any(norms < 1e-12):
        raise ValueError('source quaternions must be nonzero')
    reverse_quaternions /= norms[:, None]
    for index in range(1, count):
        if np.dot(reverse_quaternions[index - 1], reverse_quaternions[index]) < 0:
            reverse_quaternions[index] *= -1.0
    reverse_times = times[-1] - times[::-1]
    reverse_times -= reverse_times[0]
    return {'time_s': reverse_times, 'position_mm': positions[::-1].copy(),
            'quaternion_xyzw': reverse_quaternions, 'arc_s_mm': arc[::-1].copy()}


def smooth_time_dilation(time_s, factors):
    from scipy.ndimage import gaussian_filter1d

    times = np.asarray(time_s, dtype=float)
    values = np.asarray(factors, dtype=float)
    if times.ndim != 1 or values.shape != times.shape or len(times) < 2:
        raise ValueError('time and dilation factors must be matching vectors')
    if np.any(np.diff(times) <= 0) or not np.isfinite(values).all() or np.any(values < 1.0):
        raise ValueError('times must increase and dilation factors must be finite and at least one')
    return np.clip(gaussian_filter1d(values, sigma=6.0), 1.0, 1.25)


def retimed_clock(time_s, factors):
    times = np.asarray(time_s, dtype=float)
    dilation = np.asarray(factors, dtype=float)
    if times.ndim != 1 or dilation.shape != times.shape or len(times) < 2:
        raise ValueError('time and dilation factors must be matching vectors')
    if np.any(np.diff(times) <= 0) or not np.isfinite(dilation).all() or np.any(dilation < 1):
        raise ValueError('source times must increase and dilation factors must be finite and at least one')
    intervals = np.diff(times) * (dilation[:-1] + dilation[1:]) * 0.5
    return np.r_[0.0, np.cumsum(intervals)]
