import numpy as np


def pick_centerline(clip, point, size, radius=18.):
    clip = np.asarray(clip, dtype=float)
    if clip.ndim != 2 or clip.shape[1] != 4 or len(clip) < 2 or min(size) <= 0:
        return -1
    start, finish = clip[:-1], clip[1:]
    lower = np.zeros(len(start))
    upper = np.ones(len(start))
    valid = np.isfinite(start).all(axis=1) & np.isfinite(finish).all(axis=1)
    for axis in range(3):
        for sign in [-1., 1.]:
            first = start[:, 3] + sign * start[:, axis]
            last = finish[:, 3] + sign * finish[:, axis]
            change = last - first
            parallel = abs(change) < 1e-12
            valid &= ~(parallel & (first < 0))
            crossing = np.divide(-first, change, out=np.zeros_like(first), where=~parallel)
            lower = np.where(change > 1e-12, np.maximum(lower, crossing), lower)
            upper = np.where(change < -1e-12, np.minimum(upper, crossing), upper)
    valid &= lower <= upper
    indices = np.flatnonzero(valid)
    if not len(indices):
        return -1
    clipped_start = start[indices] + lower[indices, None] * (finish[indices] - start[indices])
    clipped_finish = start[indices] + upper[indices, None] * (finish[indices] - start[indices])
    front = (clipped_start[:, 3] > 1e-9) & (clipped_finish[:, 3] > 1e-9)
    indices, clipped_start, clipped_finish = indices[front], clipped_start[front], clipped_finish[front]
    if not len(indices):
        return -1
    scale = np.array([size[0] / 2, -size[1] / 2])
    offset = np.array(size) / 2
    screen_start = clipped_start[:, :2] / clipped_start[:, 3, None] * scale + offset
    screen_finish = clipped_finish[:, :2] / clipped_finish[:, 3, None] * scale + offset
    delta = screen_finish - screen_start
    square = np.einsum('ij,ij->i', delta, delta)
    fraction = np.clip(np.einsum('ij,ij->i', np.asarray(point) - screen_start, delta) / np.maximum(square, 1e-12), 0, 1)
    distances = np.linalg.norm(screen_start + fraction[:, None] * delta - point, axis=1)
    best = int(np.argmin(distances))
    if distances[best] > radius:
        return -1
    amount = fraction[best]
    perspective = amount * clipped_start[best, 3] / ((1 - amount) * clipped_finish[best, 3] + amount * clipped_start[best, 3])
    parameter = lower[indices[best]] + perspective * (upper[indices[best]] - lower[indices[best]])
    return int(indices[best] + (parameter > .5))
