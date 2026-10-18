from dataclasses import dataclass
import json
from pathlib import Path
import sys

import numpy as np


INSERTION = '#087fd1'
WITHDRAWAL = '#f36b21'
UNSEEN = '#b9c3d1'


def assets_path():
    if getattr(sys, 'frozen', False):
        return Path(sys._MEIPASS) / 'assets'
    return Path(__file__).resolve().parents[3] / '数据' / 'clon001'


def conjugate(quaternion):
    result = np.array(quaternion, dtype=float, copy=True)
    result[..., :3] *= -1
    return result


def multiply(first, second):
    first, second = np.asarray(first), np.asarray(second)
    vector = first[..., 3, None] * second[..., :3] + second[..., 3, None] * first[..., :3] + np.cross(first[..., :3], second[..., :3])
    scalar = first[..., 3] * second[..., 3] - np.sum(first[..., :3] * second[..., :3], axis=-1)
    return np.concatenate([vector, scalar[..., None]], axis=-1)


def slerp(first, second, amount):
    first, second = np.asarray(first, dtype=float), np.asarray(second, dtype=float)
    dot = np.sum(first * second, axis=-1, keepdims=True)
    second = np.where(dot < 0, -second, second)
    angle = np.arccos(np.clip(np.abs(dot), 0, 1))
    sine = np.sin(angle)
    amount = np.asarray(amount)[..., None]
    safe = np.maximum(sine, 1e-12)
    result = np.where(sine > 1e-6, np.sin((1 - amount) * angle) / safe * first + np.sin(amount * angle) / safe * second, (1 - amount) * first + amount * second)
    return result / np.linalg.norm(result, axis=-1, keepdims=True)


def rotation(quaternion):
    qx, qy, qz, qw = np.asarray(quaternion)
    return np.array([
        [1 - 2 * (qy*qy + qz*qz), 2 * (qx*qy - qz*qw), 2 * (qx*qz + qy*qw)],
        [2 * (qx*qy + qz*qw), 1 - 2 * (qx*qx + qz*qz), 2 * (qy*qz - qx*qw)],
        [2 * (qx*qz - qy*qw), 2 * (qy*qz + qx*qw), 1 - 2 * (qx*qx + qy*qy)],
    ])


def interval(times, time):
    index = int(np.clip(np.searchsorted(times, time, side='right') - 1, 0, len(times) - 2))
    amount = float(np.clip((time - times[index]) / (times[index + 1] - times[index]), 0, 1))
    return index, amount


@dataclass
class Frame:
    time: float
    index: int
    amount: float
    camera_index: int
    camera_amount: float
    position: np.ndarray
    body_rotation: np.ndarray
    camera_position: np.ndarray
    camera_rotation: np.ndarray
    target: np.ndarray


class Replay:
    def __init__(self, path=None, load_predictive=True):
        self.path = Path(path) if path else assets_path()
        manifest_path = self.path / 'magnetic_manifest.json'
        if manifest_path.is_file():
            import hashlib
            manifest = json.loads(manifest_path.read_text('utf-8'))
            for name, expected in manifest['sha256'].items():
                filename = self.path / name
                if not filename.is_file() or hashlib.sha256(filename.read_bytes()).hexdigest() != expected:
                    raise ValueError('八线圈仿真数据校验失败，请重新完整复制软件文件夹：' + name)
        arrays = np.load(self.path / 'arrays.npz', allow_pickle=False)

        def decode(value):
            if isinstance(value, dict):
                if '__array__' in value:
                    return arrays[value['__array__']]
                return {key: decode(item) for key, item in value.items()}
            if isinstance(value, list):
                return [decode(item) for item in value]
            return value

        self.payload = decode(json.loads((self.path / 'metadata.json').read_text(encoding='utf-8')))
        arrays.close()
        self.data = self.payload['datasets']['capsule_roundtrip']
        self.camera = self.payload['cameras']['roundtrip_new']
        self.times = self.data['time_s']
        self.duration = float(self.times[-1])
        self.join_time = self.payload['roundTrip']['join_time_s']
        self.positions = np.column_stack([self.data[f'actual_position_{axis}_mm'] for axis in 'xyz'])
        self.velocities = np.column_stack([self.data[f'actual_velocity_{axis}_mm_s'] for axis in 'xyz'])
        self.body_quaternions = np.column_stack([self.data['actual_q' + axis] for axis in 'xyzw'])
        self.body_quaternions /= np.linalg.norm(self.body_quaternions, axis=1, keepdims=True)
        self.prepare_camera()
        self.points = self.payload['geometry']['points']
        self.faces = self.payload['geometry']['faces'].astype(np.uint32)
        self.areas = self.payload['geometry']['areas_mm2']
        self.source = self.payload['source']
        self.centerline = np.column_stack([self.source[axis + '_mm'] for axis in 'xyz'])
        self.insertion_seen = self.payload['evaluation']['methods']['slowdown_new']['first_seen_s']
        self.masks = self.payload['withdrawalCoverage']['masks']
        self.withdrawal_seen = self.masks['withdrawal_first_seen_s']
        self.summary = self.payload['withdrawalCoverage']['summary']
        self.metric_rows = self.payload['evaluation']['methods']['withdrawal_new']['rows']
        self.metric_times = np.array([row['time_s'] for row in self.metric_rows])
        self.settings = self.payload['evaluation']['settings']
        self.registry = self.payload['registry']
        self.robot = self.registry['robots'][0]
        self.coils = None
        if 'coilSources' in self.payload:
            from .coil_sources import CoilSources
            self.coils = CoilSources(self.payload['coilSources'], self.path)
        self.prediction = self.payload.get('cameraPrediction')
        if load_predictive and self.coils is None:
            from .predictive import apply_prediction
            apply_prediction(self)
            self.prepare_camera()
        self.validate_route_direction()

    def validate_route_direction(self):
        direction = self.payload.get('routeDirection')
        if direction is None:
            return
        expected = np.asarray([direction['entry_position_mm'],direction['deep_position_mm']],dtype=float)
        if (expected.shape != (2,3) or not np.isfinite(expected).all()
                or not np.allclose(expected,self.centerline[[0,-1]],atol=1e-6,rtol=0.)):
            raise ValueError('肛门入口与路线方向标记不一致，请重新生成完整数据。')
        actual = np.asarray([self.positions[0],self.body_pose(self.join_time)[0],self.positions[-1]])
        if np.any(np.linalg.norm(actual-expected[[0,1,0]],axis=1)>1.):
            raise ValueError('进退镜方向错误：必须从肛门入口进镜，在深部换向并退回肛门入口。')
        tangent = self.centerline[1]-self.centerline[0]
        tangent /= max(np.linalg.norm(tangent),1e-12)
        if np.dot(rotation(self.body_quaternions[0])[:,0],tangent)<.95:
            raise ValueError('肛门入口处机身朝向错误：进镜必须面向肠道内部。')

    def prepare_camera(self):
        self.camera_times = self.camera['time_s']
        self.camera_quaternions = self.camera['quaternion']
        indices = np.clip(np.searchsorted(self.times, self.camera_times, side='right') - 1, 0, len(self.times)-2)
        amounts = (self.camera_times - self.times[indices]) / (self.times[indices+1] - self.times[indices])
        body_at_camera = slerp(self.body_quaternions[indices], self.body_quaternions[indices+1], amounts)
        self.relative_quaternions = multiply(conjugate(body_at_camera), self.camera_quaternions)

    def body_pose(self, time):
        time = float(np.clip(time, 0, self.duration))
        index, amount = interval(self.times, time)
        span = self.times[index+1] - self.times[index]
        position = ((2*amount**3 - 3*amount**2 + 1)*self.positions[index] + (amount**3 - 2*amount**2 + amount)*span*self.velocities[index]
                    + (-2*amount**3 + 3*amount**2)*self.positions[index+1] + (amount**3 - amount**2)*span*self.velocities[index+1])
        body_quaternion = slerp(self.body_quaternions[index], self.body_quaternions[index+1], amount)
        return position, body_quaternion

    def frame(self, time):
        time = float(np.clip(time, 0, self.duration))
        index, amount = interval(self.times, time)
        position, body_quaternion = self.body_pose(time)
        body_rotation = rotation(body_quaternion)
        camera_index, camera_amount = interval(self.camera_times, time)
        relative = slerp(self.relative_quaternions[camera_index], self.relative_quaternions[camera_index+1], camera_amount)
        camera_rotation = rotation(multiply(body_quaternion, relative))
        camera_position = position + body_rotation[:, 0] * self.settings['mount_offset_mm']
        target = self.camera['target'][camera_index] * (1-camera_amount) + self.camera['target'][camera_index+1] * camera_amount
        return Frame(time, index, amount, camera_index, camera_amount, position, body_rotation, camera_position, camera_rotation, target)

    def value(self, frame, key, default=0):
        values = self.data.get(key)
        if values is None:
            return default
        if isinstance(values[frame.index], str):
            return values[frame.index if frame.amount < 1 else frame.index+1]
        return float(values[frame.index] * (1-frame.amount) + values[frame.index+1] * frame.amount)

    def camera_value(self, frame, key):
        values = np.asarray(self.camera[key])
        first, second = values[frame.camera_index], values[frame.camera_index + 1]
        if key in {'joint_angle_deg', 'goal_joint_angle_deg'}:
            return first + ((second - first + 180.) % 360. - 180.) * frame.camera_amount
        return first * (1 - frame.camera_amount) + second * frame.camera_amount

    def vector(self, frame, prefix, unit):
        return np.array([self.value(frame, f'{prefix}_{axis}_{unit}') for axis in 'xyz'])

    def gradient(self, frame):
        if self.coils is not None:
            return self.coils.gradient(frame)
        return np.array([[self.value(frame, f'applied_G_{first}{second}_T_m') for second in 'xyz'] for first in 'xyz'])

    def field(self, frame, points, mode='total'):
        if self.coils is not None:
            result = self.coils.field(frame, points)
            axis = frame.body_rotation[:, 0]
            if mode == 'turning':
                return (result - np.sum(result * axis, axis=-1, keepdims=True) * axis) * 1000.
            if mode == 'residual':
                result = result - self.robot['config']['bias_field_T'] * 1000. * axis
            return result
        field = self.vector(frame, 'applied_field', 'mT')
        if mode == 'turning':
            axis = frame.body_rotation[:, 0]
            turning = (field - np.dot(field, axis)*axis) * 1000
            return np.broadcast_to(turning, np.shape(points))
        result = field + (np.asarray(points) - frame.position) @ self.gradient(frame).T
        if mode == 'residual':
            result -= self.robot['config']['bias_field_T'] * 1000 * frame.body_rotation[:, 0]
        return result

    def coverage(self, time):
        insertion = (self.insertion_seen >= 0) & (self.insertion_seen <= min(time, self.join_time))
        withdrawal = ((self.withdrawal_seen >= 0) & (self.withdrawal_seen + self.join_time <= time)
                      & ~np.asarray(self.masks['insertion_covered'], dtype=bool))
        return insertion, withdrawal

    def metrics(self, time):
        index = int(np.clip(np.searchsorted(self.metric_times, time, side='right') - 1, 0, len(self.metric_rows)-1))
        return self.metric_rows[index]

    def phase(self, time):
        return '进镜观察' if time < self.join_time else '连续退镜 · 补拍'
