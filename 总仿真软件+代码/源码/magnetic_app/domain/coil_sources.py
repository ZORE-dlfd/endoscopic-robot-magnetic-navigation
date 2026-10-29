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
    def __init__(self, payload, assets=None):
        self.positions = np.asarray(payload['positions_mm'])
        self.currents = np.asarray(payload['currents_a'])
        self.axes = np.asarray(payload['axes'])
        self.finite = payload.get('model') == 'finite-winding-core-coupled-v1'
        self.design = payload.get('design', {})
        self.strength = float(payload.get('moment_per_amp_a_m2', 0.))
        self.limit = float(payload['current_limit_a'])
        self.quaternions = np.asarray(payload['array_quaternion'], dtype=float) if 'array_quaternion' in payload else None
        if self.quaternions is not None:
            self.anchors = np.asarray(payload['anchor_mm'], dtype=float)
            self.offsets = np.asarray(payload['offsets_m'], dtype=float)
            if (self.quaternions.shape != (len(self.currents), 4)
                    or self.anchors.shape != (len(self.currents), 3) or self.offsets.shape != (8, 3)
                    or not np.isfinite(self.quaternions).all() or not np.isfinite(self.anchors).all()
                    or not np.isfinite(self.offsets).all()
                    or not np.allclose(np.linalg.norm(self.quaternions, axis=1), 1., atol=1e-6)):
                raise ValueError('Rotating source poses are invalid')
        if (self.positions.shape != (len(self.currents), 8, 3)
                or self.currents.shape[1:] != (8,) or self.axes.shape != (8, 3)
                or not np.isfinite(self.positions).all() or not np.isfinite(self.currents).all()
                or np.max(np.abs(self.currents)) > self.limit + 1e-7):
            raise ValueError('Eight-coil source data is invalid')
        if self.finite:
            from .finite_field import FiniteField
            self.kernel = FiniteField(assets/payload['field_table'])
            self.magnetization_matrix = np.asarray(payload['magnetization_matrix_a_m_per_a'])
            self.offsets = np.asarray(payload['offsets_m'])
            if self.magnetization_matrix.shape != (8, 8) or not np.isfinite(self.magnetization_matrix).all():
                raise ValueError('Finite-core coupling matrix is invalid')

    def pose_at(self, frame):
        from .replay import rotation, slerp
        if self.quaternions is None:
            return np.eye(3), np.zeros(3)
        orientation = rotation(slerp(self.quaternions[frame.index], self.quaternions[frame.index+1], frame.amount))
        anchor = self.anchors[frame.index]*(1-frame.amount)+self.anchors[frame.index+1]*frame.amount
        return orientation, anchor

    def axes_at(self, frame):
        return self.axes @ self.pose_at(frame)[0].T if self.quaternions is not None else self.axes

    def at(self, frame):
        if self.quaternions is None:
            positions = self.positions[frame.index] * (1-frame.amount) + self.positions[frame.index+1] * frame.amount
        else:
            orientation, anchor = self.pose_at(frame)
            positions = self.offsets @ orientation.T*1000.+anchor
        axes = self.axes_at(frame)
        currents = self.currents[frame.index] * (1-frame.amount) + self.currents[frame.index+1] * frame.amount
        if self.finite:
            design = self.design
            winding_area = np.pi*(design['winding_inner_radius_m']**2
                + design['winding_inner_radius_m']*design['winding_outer_radius_m']
                + design['winding_outer_radius_m']**2)/3
            core_volume = np.pi*design['core_radius_m']**2*design['core_length_m']
            moments = axes*(design['turns']*winding_area*currents
                + core_volume*(self.magnetization_matrix@currents))[:, None]
        else:
            moments = axes * currents[:, None] * self.strength
        return positions, currents, moments

    def field(self, frame, points_mm):
        positions, currents, moments = self.at(frame)
        if not self.finite:
            return source_field(points_mm, positions, moments)
        return self.kernel.field(np.asarray(points_mm)/1000., positions/1000., self.axes_at(frame),
                                 currents, self.magnetization_matrix@currents)*1000.

    def gradient(self, frame):
        positions, currents, moments = self.at(frame)
        if not self.finite:
            return source_gradient(frame.position, positions, moments)
        return self.kernel.gradient(frame.position/1000., positions/1000., self.axes_at(frame),
                                    currents, self.magnetization_matrix@currents)

    def parameter_rows(self, frame):
        if not self.finite:
            return []
        positions, currents, moments = self.at(frame)
        magnetization = self.magnetization_matrix@currents
        design = self.design
        resistance = design['resistance_ohm_at_assumed_temperature']
        return [dict(coil=f'C{index+1}', core_length_mm=design['core_length_m']*1000.,
            core_diameter_mm=design['core_radius_m']*2000., winding_length_mm=design['winding_length_m']*1000.,
            winding_inner_diameter_mm=design['winding_inner_radius_m']*2000.,
            winding_outer_diameter_mm=design['winding_outer_radius_m']*2000., turns=design['turns'],
            copper_diameter_mm=design['copper_diameter_m']*1000.,
            insulated_diameter_mm=design['insulated_diameter_m']*1000.,
            relative_permeability=design['relative_permeability'],
            saturation_polarization_t=design['saturation_polarization_t'],
            current_limit_a=self.limit, ideal_driver_voltage_limit_v=design['ideal_driver_voltage_limit_v'],
            resistance_ohm=resistance, current_a=float(current), voltage_v=float(current*resistance),
            power_w=float(current**2*resistance),
            saturation_fraction=float(abs(magnetization[index])/design['saturation_magnetization_a_m']),
            anchor_distance_mm=float(np.linalg.norm(self.offsets[index])*1000.),
            offset_mm=(self.offsets[index]*1000.).tolist(), axis=self.axes_at(frame)[index].tolist(),
            position_mm=positions[index].tolist()) for index, current in enumerate(currents)]
