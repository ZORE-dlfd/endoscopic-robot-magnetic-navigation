import hashlib
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from magnetic_app.domain.replay import Replay, rotation, slerp


def sha256(filename):
    return hashlib.sha256(Path(filename).read_bytes()).hexdigest()


class Scene:
    def __init__(self, assets):
        self.replay = Replay(assets, load_predictive=False)
        replay = self.replay
        self.points, self.faces, self.areas = replay.points, replay.faces, replay.areas
        self.centers = self.points[self.faces].mean(axis=1)
        self.normals = np.asarray(replay.payload['geometry']['normals'])
        self.insertion_covered = replay.insertion_seen >= 0
        self.settings = replay.settings
        self.duration = replay.duration - replay.join_time
        join_index = int(np.argmin(abs(replay.camera_times - replay.join_time)))
        if abs(replay.camera_times[join_index] - replay.join_time) > 1e-8:
            raise ValueError('Original insertion must include the exact phase boundary')
        self.initial_angle = replay.camera['joint_angle_deg'][join_index].copy()
        self.initial_velocity = replay.camera['joint_velocity_deg_s'][join_index].copy()

    def pose(self, time):
        if time < -1e-8 or time > self.duration + 1e-8:
            raise ValueError('Requested body pose outside withdrawal')
        position, quaternion = self.replay.body_pose(self.replay.join_time + float(np.clip(time, 0, self.duration)))
        body = rotation(quaternion)
        return position + self.settings['mount_offset_mm'] * body[:, 0], body

    def rotations(self, times):
        replay = self.replay
        absolute = np.clip(np.asarray(times) + replay.join_time, 0, replay.duration)
        indices = np.clip(np.searchsorted(replay.times, absolute, side='right') - 1, 0, len(replay.times) - 2)
        amounts = (absolute - replay.times[indices]) / (replay.times[indices + 1] - replay.times[indices])
        return Rotation.from_quat(slerp(replay.body_quaternions[indices], replay.body_quaternions[indices + 1], amounts))
