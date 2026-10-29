from copy import deepcopy
from dataclasses import dataclass
import re

import numpy as np
from scipy.spatial.transform import Rotation


@dataclass(frozen=True)
class SupineTransform:
    rotation: np.ndarray
    translation_mm: np.ndarray
    table_clearance_mm: float = 20.

    @classmethod
    def from_points(cls, points, table_clearance_mm=20.):
        points = np.asarray(points, dtype=float)
        if (points.ndim != 2 or points.shape[1] != 3 or not len(points)
                or not np.isfinite(points).all() or not np.isfinite(table_clearance_mm)
                or table_clearance_mm < 0.):
            raise ValueError('Posture requires finite nonempty geometry and nonnegative clearance')
        lower, upper = points.min(axis=0), points.max(axis=0)
        center = (lower+upper)/2.
        rotation = np.array([[1., 0., 0.], [0., 0., -1.], [0., 1., 0.]])
        translation = np.array([-center[0], center[2], table_clearance_mm-lower[1]])
        return cls(rotation, translation, float(table_clearance_mm))

    def positions(self, values):
        return self.vectors(values)+self.translation_mm

    def vectors(self, values):
        return np.asarray(values, dtype=float)@self.rotation.T

    def quaternions(self, values):
        return (Rotation.from_matrix(self.rotation)*Rotation.from_quat(values)).as_quat()

    def metadata(self):
        return dict(schema=1, posture='supine', label='仰卧姿态假设',
            rotation_from_original=self.rotation.tolist(), translation_mm=self.translation_mm.tolist(),
            mapping='original X -> world X; original Z -> world -Y; original Y -> world Z',
            gravity_m_s2=[0., 0., -9.81], gravity_frame='fixed world -Z, not rotated with anatomy',
            table_z_mm=0., table_clearance_mm=self.table_clearance_mm,
            array_frame='fixed world +Z side; currents and motion recomputed',
            assumption='Raw mesh has no verified anatomical orientation; assumed supine placement, not patient registration',
            contact_model=False)

    def _triplet(self, table, keys, positions=False):
        if all(key in table for key in keys):
            values = np.column_stack([table[key] for key in keys])
            result = self.positions(values) if positions else self.vectors(values)
            for index, key in enumerate(keys):
                table[key] = result[:, index]

    def source(self, table):
        result = table.copy()
        self._triplet(result, [axis+'_mm' for axis in 'xyz'], positions=True)
        for prefix in ['t', 'n', 'b']:
            self._triplet(result, [prefix+axis for axis in 'xyz'])
        for prefix, unit in [('v', 'mm_s'), ('a', 'mm_s2')]:
            self._triplet(result, [prefix+axis+'_'+unit for axis in 'xyz'])
        quaternion = self.quaternions(np.column_stack([table['q'+axis] for axis in 'xyzw']))
        for index, axis in enumerate('xyzw'):
            result['q'+axis] = quaternion[:, index]
        angles = Rotation.from_quat(quaternion).as_euler('xyz', degrees=True)
        for index, angle in enumerate(['roll', 'pitch', 'yaw']):
            result[angle+'_deg'] = angles[:, index]
        return result

    def _dataset(self, table):
        groups = set()
        for key in table:
            match = re.fullmatch(r'(.+)_([xyz])_(.+)', key)
            if match and '_body' not in match[1]:
                groups.add((match[1], match[3]))
        for prefix, unit in groups:
            self._triplet(table, [f'{prefix}_{axis}_{unit}' for axis in 'xyz'],
                positions=prefix.endswith('_position'))
        for prefix in ['actual', 'ref']:
            quaternion = self.quaternions(np.column_stack([table[prefix+'_q'+axis] for axis in 'xyzw']))
            for index, axis in enumerate('xyzw'):
                table[prefix+'_q'+axis] = quaternion[:, index]
            angles = Rotation.from_quat(quaternion).as_euler('xyz', degrees=True)
            for index, angle in enumerate(['roll', 'pitch', 'yaw']):
                key = prefix+'_'+angle+'_deg'
                if key in table:
                    table[key] = angles[:, index]
        for prefix in ['required', 'applied']:
            gradient = np.stack([np.column_stack([table[f'{prefix}_G_{first}{second}_T_m']
                for second in 'xyz']) for first in 'xyz'], axis=1)
            gradient = self.rotation@gradient@self.rotation.T
            for first_index, first in enumerate('xyz'):
                for second_index, second in enumerate('xyz'):
                    table[f'{prefix}_G_{first}{second}_T_m'] = gradient[:, first_index, second_index]

    def payload(self, original):
        if 'patientPosture' in original:
            raise ValueError('Reference already has a patient posture transform')
        result = deepcopy(original)
        geometry = result['geometry']
        geometry['points'] = self.positions(geometry['points'])
        geometry['normals'] = self.vectors(geometry['normals'])
        geometry['bounds'] = [geometry['points'].min(axis=0).tolist(), geometry['points'].max(axis=0).tolist()]
        geometry['coordinate_transform'] = dict(original=geometry.get('coordinate_transform'),
            patient_posture=self.metadata())
        result['source'] = self.source(result['source'])
        for dataset in result['datasets'].values():
            self._dataset(dataset)
        for camera in result['cameras'].values():
            if not isinstance(camera, dict):
                continue
            for key in ['position', 'target']:
                if key in camera:
                    camera[key] = self.positions(camera[key])
            for key in ['quaternion', 'goal_quaternion']:
                if key in camera:
                    camera[key] = self.quaternions(camera[key])
        bounds = np.asarray(result['registry']['field_world_bounds_mm'])
        corners = np.stack(np.meshgrid(*bounds, indexing='ij'), axis=-1).reshape(-1, 3)
        corners = self.positions(corners)
        result['registry']['field_world_bounds_mm'] = np.column_stack([corners.min(axis=0), corners.max(axis=0)]).tolist()
        result['patientPosture'] = self.metadata()
        return result
