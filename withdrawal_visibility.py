import numpy as np


def surface_masks(areas_mm2, possible_pose_visibility, insertion_first_seen_s):
    areas = np.asarray(areas_mm2, dtype=float)
    visible = np.asarray(possible_pose_visibility, dtype=bool)
    first_seen = np.asarray(insertion_first_seen_s, dtype=float)
    if areas.ndim != 1 or not np.isfinite(areas).all() or np.any(areas < 0):
        raise ValueError('face areas must be a finite nonnegative vector')
    if visible.ndim != 2 or visible.shape[1] != len(areas) or visible.shape[0] == 0:
        raise ValueError('pose visibility must have shape (nonzero poses, faces)')
    if first_seen.shape != areas.shape:
        raise ValueError('insertion first-seen times must match face count')
    recoverable = np.any(visible, axis=0)
    fold_hidden = ~recoverable
    insertion_covered = np.isfinite(first_seen) & (first_seen >= 0.0)
    insertion_missed = recoverable & ~insertion_covered
    return {'recoverable': recoverable, 'fold_hidden': fold_hidden,
            'insertion_covered': insertion_covered, 'insertion_missed': insertion_missed}


def unique_new_area(areas_mm2, recoverable_mask, insertion_covered_mask, withdrawal_covered_mask):
    areas = np.asarray(areas_mm2, dtype=float)
    masks = [np.asarray(mask, dtype=bool) for mask in
             (recoverable_mask, insertion_covered_mask, withdrawal_covered_mask)]
    if areas.ndim != 1 or not np.isfinite(areas).all() or np.any(areas < 0):
        raise ValueError('face areas must be a finite nonnegative vector')
    if any(mask.shape != areas.shape for mask in masks):
        raise ValueError('all face masks must match face count')
    recoverable, insertion_covered, withdrawal_covered = masks
    withdrawal_new = recoverable & ~insertion_covered & withdrawal_covered
    insertion_area = float(areas[insertion_covered].sum())
    withdrawal_area = float(areas[withdrawal_new].sum())
    return {'insertion_covered_area_mm2': insertion_area,
            'withdrawal_added_area_mm2': withdrawal_area,
            'final_covered_area_mm2': insertion_area + withdrawal_area,
            'withdrawal_new_mask': withdrawal_new}


def camera_candidate_directions(body_matrix, max_angle_deg=45.0, ring_angles_deg=(30.0, 45.0), azimuth_count=8):
    body = np.asarray(body_matrix, dtype=float)
    if body.shape != (3, 3) or azimuth_count < 4:
        raise ValueError('body frame must be 3x3 and azimuth count at least four')
    if not 0 < max_angle_deg <= 90:
        raise ValueError('maximum camera cone angle must be in (0, 90] degrees')
    forward, left, up = body[:, 0], body[:, 1], body[:, 2]
    directions = [forward.copy()]
    for ring_angle in ring_angles_deg:
        if ring_angle <= 0 or ring_angle > max_angle_deg:
            continue
        theta = np.deg2rad(ring_angle)
        for index in range(azimuth_count):
            phi = 2 * np.pi * index / azimuth_count
            lateral = np.cos(phi) * left + np.sin(phi) * up
            directions.append(np.cos(theta) * forward + np.sin(theta) * lateral)
    return np.asarray(directions)
