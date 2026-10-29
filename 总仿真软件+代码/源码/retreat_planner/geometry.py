import numpy as np
from scipy.spatial.transform import Rotation

def forward_frame(forward, up):
    forward = np.asarray(forward, dtype=float)
    forward = forward / np.linalg.norm(forward)
    up = np.asarray(up, dtype=float) - np.dot(up, forward) * forward
    if np.linalg.norm(up) < 1e-8:
        axis = np.eye(3)[np.argmin(np.abs(forward))]
        up = axis - np.dot(axis, forward) * forward
    up /= np.linalg.norm(up)
    return np.column_stack([forward, np.cross(up, forward), up])


def cone_angle(directions):
    directions = np.asarray(directions, dtype=float)
    return np.rad2deg(np.arccos(np.clip(directions[..., 0] / np.linalg.norm(directions, axis=-1), -1, 1)))


def viewport_mask(delta, frame, fov_deg, aspect):
    local = np.asarray(delta) @ frame
    tangent = np.tan(np.deg2rad(fov_deg / 2))
    return ((local[:, 0] > 0) & (np.abs(local[:, 1]) <= local[:, 0] * tangent * aspect) &
            (np.abs(local[:, 2]) <= local[:, 0] * tangent))


def goal_for_direction(body, local_direction, up_reference, limit=89.):
    frame = forward_frame(body @ local_direction, up_reference)
    angles = Rotation.from_matrix(body.T @ frame).as_euler('ZYX', degrees=True)
    if np.any(np.abs(angles[:2]) > limit + 1e-8):
        return None
    return angles


def direction_grid(step_deg, center=None, radius=None):
    if step_deg <= 0:
        raise ValueError('Angular step must be positive')
    if center is None:
        values = np.unique(np.r_[np.arange(-45., 45. + 1e-8, step_deg), -45., 0., 45.])
        yaw, pitch = np.meshgrid(values, values)
    else:
        offsets = np.arange(-radius, radius + 1e-8, step_deg)
        yaw, pitch = np.meshgrid(center[0] + offsets, center[1] + offsets)
    yaw, pitch = yaw.ravel(), pitch.ravel()
    radians_yaw, radians_pitch = np.deg2rad(yaw), np.deg2rad(pitch)
    directions = np.column_stack([np.cos(radians_yaw) * np.cos(radians_pitch),
                                  np.sin(radians_yaw) * np.cos(radians_pitch), -np.sin(radians_pitch)])
    return directions[cone_angle(directions) <= 45 + 1e-8]


def direction_to_angles(direction):
    return np.rad2deg([np.arctan2(direction[1], direction[0]), -np.arcsin(np.clip(direction[2], -1, 1))])
