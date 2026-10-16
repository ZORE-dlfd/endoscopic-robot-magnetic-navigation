import json
from pathlib import Path

import numpy as np

from withdrawal_visibility import unique_new_area


def build_summary(areas_mm2, masks, withdrawal_first_seen_s, duration_s, wait_count=0):
    areas = np.asarray(areas_mm2, dtype=float)
    recoverable = np.asarray(masks['recoverable'], dtype=bool)
    fold_hidden = np.asarray(masks['fold_hidden'], dtype=bool)
    insertion_covered = np.asarray(masks['insertion_covered'], dtype=bool)
    insertion_missed = np.asarray(masks['insertion_missed'], dtype=bool)
    withdrawal_seen = np.asarray(withdrawal_first_seen_s, dtype=float) >= 0.0
    if any(mask.shape != areas.shape for mask in
           (recoverable, fold_hidden, insertion_covered, insertion_missed, withdrawal_seen)):
        raise ValueError('surface masks and observations must match the area array')
    if np.any(recoverable & fold_hidden) or np.any(recoverable == fold_hidden):
        raise ValueError('recoverable and fold-hidden masks must be complements')
    if np.any(insertion_covered & insertion_missed) or np.any(insertion_missed & ~recoverable):
        raise ValueError('insertion covered/missed masks must partition recoverable faces')
    area = unique_new_area(areas, recoverable, insertion_covered, withdrawal_seen)
    recoverable_area = float(areas[recoverable].sum())
    total_area = float(areas.sum())
    if not np.isclose(area['final_covered_area_mm2'],
                      area['insertion_covered_area_mm2'] + area['withdrawal_added_area_mm2']):
        raise AssertionError('coverage area does not conserve')
    return {
        'insertion_covered_area_mm2': area['insertion_covered_area_mm2'],
        'insertion_missed_recoverable_area_mm2': float(areas[insertion_missed].sum()),
        'withdrawal_added_area_mm2': area['withdrawal_added_area_mm2'],
        'final_covered_area_mm2': area['final_covered_area_mm2'],
        'recoverable_area_mm2': recoverable_area,
        'fold_hidden_area_mm2': float(areas[fold_hidden].sum()),
        'total_lumen_area_mm2': total_area,
        'uncovered_area_mm2': max(total_area - area['final_covered_area_mm2'], 0.0),
        'insertion_coverage_pct': 100.0 * area['insertion_covered_area_mm2'] / max(total_area, 1e-12),
        'withdrawal_added_pct_of_total_lumen': 100.0 * area['withdrawal_added_area_mm2'] / max(total_area, 1e-12),
        'withdrawal_added_pct_of_insertion_covered': 100.0 * area['withdrawal_added_area_mm2'] / max(area['insertion_covered_area_mm2'], 1e-12),
        'coverage_gain_percentage_points': 100.0 * area['withdrawal_added_area_mm2'] / max(total_area, 1e-12),
        'final_coverage_pct': 100.0 * area['final_covered_area_mm2'] / max(total_area, 1e-12),
        'coverage_increased': bool(area['withdrawal_added_area_mm2'] > 0.0),
        'duration_s': float(duration_s), 'wait_count': int(wait_count),
        'withdrawal_new_mask': area['withdrawal_new_mask'].tolist(),
    }


def evaluate_files(root=None):
    root = Path(root) if root is not None else Path(__file__).resolve().parent
    geometry = json.loads((root / 'geometry.json').read_text(encoding='utf-8'))
    masks = json.loads((root / 'withdrawal_visibility_masks.json').read_text(encoding='utf-8'))
    camera = json.loads((root / 'withdrawal_camera.json').read_text(encoding='utf-8'))
    evaluation = json.loads((root / 'evaluation.json').read_text(encoding='utf-8'))
    insertion_first_seen = np.asarray(evaluation['methods']['slowdown_new']['first_seen_s'], dtype=float)
    if insertion_first_seen.shape != np.asarray(geometry['areas_mm2']).shape:
        raise ValueError('slowdown insertion first-seen times do not match mesh face count')
    masks['insertion_covered'] = (insertion_first_seen >= 0.0).tolist()
    masks['insertion_missed'] = (np.asarray(masks['recoverable'], dtype=bool) & (insertion_first_seen < 0.0)).tolist()
    summary = build_summary(geometry['areas_mm2'], masks, masks['withdrawal_first_seen_s'],
                            camera['time_s'][-1], wait_count=0)
    summary['candidate_camera_views_per_position'] = 37
    summary['target_camera_views_per_position'] = 25
    summary['forward_cone_limit_deg'] = 45.0
    summary['method_notice'] = ('Offline geometric simulation. Fold-hidden area is approximated using 37 sampled camera directions '
                                'at each 0.4-second withdrawal pose. No stationary waits; not clinical safety or image-quality validation.')
    output = {key: value for key, value in summary.items() if key != 'withdrawal_new_mask'}
    (root / 'withdrawal_summary.json').write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding='utf-8')
    return output


if __name__ == '__main__':
    print(json.dumps(evaluate_files(), ensure_ascii=False, indent=2))
