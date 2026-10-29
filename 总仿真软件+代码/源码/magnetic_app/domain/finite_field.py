from pathlib import Path

import numpy as np


class FiniteField:
    def __init__(self, path):
        with np.load(Path(path), allow_pickle=False) as archive:
            self.radial = archive['radial_m']
            self.axial = archive['axial_m']
            self.values = archive['responses']
        if (self.values.shape != (len(self.radial), len(self.axial), 2, 5)
                or not np.all(np.diff(self.radial) > 0)
                or not np.all(np.diff(self.axial) > 0)
                or not np.isfinite(self.values).all()):
            raise ValueError('Finite magnetic response table is invalid')

    def components(self, points_m, positions_m, axes):
        displacement = np.asarray(points_m, dtype=float)[..., None, :] - positions_m
        axial = np.sum(displacement*axes, axis=-1)
        radial_vector = displacement-axial[..., None]*axes
        radial = np.linalg.norm(radial_vector, axis=-1)
        if np.any(radial > self.radial[-1]) or np.any(np.abs(axial) > self.axial[-1]):
            raise ValueError('Field query exceeds the finite-source display table')
        radial_index = np.clip(np.searchsorted(self.radial, radial, side='right')-1, 0, len(self.radial)-2)
        axial_index = np.clip(np.searchsorted(self.axial, np.abs(axial), side='right')-1, 0, len(self.axial)-2)
        radial_fraction = ((radial-self.radial[radial_index]) /
                           (self.radial[radial_index+1]-self.radial[radial_index]))[..., None, None]
        axial_fraction = ((np.abs(axial)-self.axial[axial_index]) /
                          (self.axial[axial_index+1]-self.axial[axial_index]))[..., None, None]
        lower = (self.values[radial_index, axial_index]*(1-radial_fraction)
                 + self.values[radial_index+1, axial_index]*radial_fraction)
        upper = (self.values[radial_index, axial_index+1]*(1-radial_fraction)
                 + self.values[radial_index+1, axial_index+1]*radial_fraction)
        values = lower*(1-axial_fraction)+upper*axial_fraction
        values[..., [0, 2, 4]] *= np.sign(axial)[..., None, None]
        direction = radial_vector/np.maximum(radial[..., None], 1e-15)
        on_axis = radial < 1e-12
        if np.any(on_axis):
            helper = np.cross(axes, [0., 0., 1.])
            vertical = np.linalg.norm(helper, axis=-1) < 1e-8
            helper[vertical] = np.cross(axes[vertical], [0., 1., 0.])
            helper /= np.linalg.norm(helper, axis=-1, keepdims=True)
            direction = np.where(on_axis[..., None], helper, direction)
        return values, direction, radial, axial

    def field(self, points_m, positions_m, axes, currents_a, magnetization_a_m):
        values, radial_direction, radial, axial = self.components(points_m, positions_m, axes)
        components = values[..., 0, :]*np.asarray(currents_a)[..., None]
        components += values[..., 1, :]*np.asarray(magnetization_a_m)[..., None]
        field = components[..., 0, None]*radial_direction+components[..., 1, None]*axes
        inside = (radial < .028) & (np.abs(axial) < .026)
        return np.where(np.any(inside, axis=-1)[..., None], 0., np.sum(field, axis=-2))

    def gradient(self, point_m, positions_m, axes, currents_a, magnetization_a_m):
        values, radial_direction, radial, axial = self.components(point_m, positions_m, axes)
        components = values[:, 0, :]*np.asarray(currents_a)[:, None]
        components += values[:, 1, :]*np.asarray(magnetization_a_m)[:, None]
        azimuthal = np.cross(axes, radial_direction)
        circumferential = np.where(radial > 1e-12, components[:, 0]/np.maximum(radial, 1e-15), components[:, 2])
        outer = lambda first, second: first[:, :, None]*second[:, None, :]
        gradient = (components[:, 2, None, None]*outer(radial_direction, radial_direction)
                    + circumferential[:, None, None]*outer(azimuthal, azimuthal)
                    + components[:, 3, None, None]*(outer(radial_direction, axes)+outer(axes, radial_direction))
                    + components[:, 4, None, None]*outer(axes, axes))
        inside = (radial < .028) & (np.abs(axial) < .026)
        return np.zeros((3, 3)) if np.any(inside) else np.sum(gradient, axis=0)
