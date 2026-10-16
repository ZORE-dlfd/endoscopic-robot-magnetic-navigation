import numpy as np
from scipy.spatial.transform import Rotation

from .geometry import direction_grid, direction_to_angles, goal_for_direction


def diverse_indices(directions, scores, count, separation_deg):
    selected = []
    threshold = np.cos(np.deg2rad(separation_deg))
    for index in np.argsort(-scores, kind='stable'):
        if not selected or np.max(directions[selected] @ directions[index]) < threshold:
            selected.append(int(index))
        if len(selected) == count:
            break
    return selected


def approximate_scores(scene, origin, body, up, directions, atlas_ids, target_mask):
    ids = atlas_ids[target_mask[atlas_ids]]
    if len(ids) == 0:
        return np.zeros(len(directions))
    delta = scene.centers[ids] - origin
    area_dot = scene.areas[ids] * np.abs(np.einsum('ij,ij->i', scene.normals[ids], delta))
    tangent = np.tan(np.deg2rad(scene.settings['fov_vertical_deg'] / 2))
    aspect = scene.settings['aspect']
    focal = scene.settings['evaluation_height'] / (2 * tangent)
    scores = []
    for begin in range(0, len(directions), 32):
        forward = directions[begin:begin + 32] @ body.T
        vertical = up - (forward @ up)[:, None] * forward
        vertical /= np.maximum(np.linalg.norm(vertical, axis=1, keepdims=True), 1e-12)
        lateral = np.cross(vertical, forward)
        frames = np.stack([forward, lateral, vertical], axis=2)
        local = np.einsum('fi,cij->cfj', delta, frames)
        depth = local[:, :, 0]
        pixels = area_dot[None, :] * focal ** 2 / np.maximum(depth, 1e-8) ** 3
        mask = ((depth > 0) & (np.abs(local[:, :, 1]) <= depth * tangent * aspect) &
                (np.abs(local[:, :, 2]) <= depth * tangent) & (pixels >= scene.settings['min_projected_pixels']))
        scores.extend(mask @ scene.areas[ids])
    return np.asarray(scores)


def dense_goals(scene, visibility, state, time, coverage, config):
    origin, body = scene.pose(time)
    up = (body @ Rotation.from_euler('ZYX', state.angle, degrees=True).as_matrix())[:, 2]
    atlas_ids = visibility.atlas(origin)
    coarse = direction_grid(config.coarse_deg)
    coarse_scores = approximate_scores(scene, origin, body, up, coarse, atlas_ids, coverage.target_mask)
    peaks = diverse_indices(coarse, coarse_scores, config.peak_count, config.coarse_deg)
    grids = [coarse]
    for peak in peaks:
        grids.append(direction_grid(config.fine_deg, direction_to_angles(coarse[peak]), config.coarse_deg))
    directions = np.unique(np.round(np.vstack(grids), 10), axis=0)
    scores = approximate_scores(scene, origin, body, up, directions, atlas_ids, coverage.target_mask)
    indices = diverse_indices(directions, scores, config.candidate_count, max(4., config.fine_deg * 2))
    goals = [state.angle.copy(), state.goal.copy()]
    for index in indices:
        goal = goal_for_direction(body, directions[index], up)
        if goal is not None:
            goals.append(goal)
    unique = []
    for goal in goals:
        if not any(np.linalg.norm((goal - previous + 180) % 360 - 180) < 1e-5 for previous in unique):
            unique.append(goal)
    exact_scores = []
    for goal in unique:
        frame = body @ Rotation.from_euler('ZYX', goal, degrees=True).as_matrix()
        ids = visibility.visible(origin, frame)
        exact_scores.append(float(scene.areas[ids[coverage.target_mask[ids]]].sum()))
    return unique, np.asarray(exact_scores), len(directions)
