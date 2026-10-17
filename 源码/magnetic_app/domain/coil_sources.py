import numpy as np


def source_field(points_mm, positions_mm, moments_a_m2):
    points = np.asarray(points_mm, dtype=float)
    displacement = (points[..., None, :] - np.asarray(positions_mm)) / 1000.
    squared = np.sum(displacement * displacement, axis=-1)
    if np.any(squared < 1e-18):
        raise ValueError('Cannot evaluate the field at a point-dipole source')
    radii = np.sqrt(squared)
    projection = np.sum(displacement * np.asarray(moments_a_m2), axis=-1)
    fields = 1e-4 * (3 * displacement * projection[..., None] / radii[..., None]**5
                     - np.asarray(moments_a_m2) / radii[..., None]**3)
    return np.sum(fields, axis=-2)


def source_gradient(point_mm, positions_mm, moments_a_m2):
    displacement = (np.asarray(point_mm) - np.asarray(positions_mm)) / 1000.
    radii = np.linalg.norm(displacement, axis=-1)
    if np.any(radii < 1e-9):
        raise ValueError('Cannot evaluate a gradient at a point-dipole source')
    moments = np.asarray(moments_a_m2)
    projection = np.sum(displacement * moments, axis=-1)
    numerator = (displacement[..., :, None] * moments[..., None, :]
                 + moments[..., :, None] * displacement[..., None, :]
                 + projection[..., None, None] * np.eye(3))
    return np.sum(1e-7 * (3 * numerator / radii[..., None, None]**5
        - 15 * projection[..., None, None] * displacement[..., :, None]
        * displacement[..., None, :] / radii[..., None, None]**7), axis=0)


class CoilSources:
    def __init__(self, payload):
        self.positions = np.asarray(payload['positions_mm'])
        self.currents = np.asarray(payload['currents_a'])
        self.axes = np.asarray(payload['axes'])
        self.strength = float(payload['moment_per_amp_a_m2'])
        self.limit = float(payload['current_limit_a'])
        if (self.positions.shape != (len(self.currents), 8, 3)
                or self.currents.shape[1:] != (8,) or self.axes.shape != (8, 3)
                or not np.isfinite(self.positions).all() or not np.isfinite(self.currents).all()
                or np.max(np.abs(self.currents)) > self.limit + 1e-7):
            raise ValueError('Eight-coil source data is invalid')

    def at(self, frame):
        positions = self.positions[frame.index] * (1-frame.amount) + self.positions[frame.index+1] * frame.amount
        currents = self.currents[frame.index] * (1-frame.amount) + self.currents[frame.index+1] * frame.amount
        moments = self.axes * currents[:, None] * self.strength
        return positions, currents, moments
