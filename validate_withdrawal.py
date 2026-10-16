import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def validate():
    trajectory = pd.read_csv(ROOT / 'capsule_withdrawal_replay.csv')
    evaluation = pd.read_csv(ROOT / 'withdrawal_evaluation.csv')
    camera = json.loads((ROOT / 'withdrawal_camera.json').read_text(encoding='utf-8'))
    summary = json.loads((ROOT / 'withdrawal_summary.json').read_text(encoding='utf-8'))
    exe_report = json.loads((ROOT / 'exe_validation.json').read_text(encoding='utf-8'))
    manifest = json.loads((ROOT / 'baseline_readonly_manifest.json').read_text(encoding='utf-8'))

    quaternion = trajectory[['actual_qx', 'actual_qy', 'actual_qz', 'actual_qw']].to_numpy()
    quaternion_adjacent_dot = np.sum(quaternion[:-1] * quaternion[1:], axis=1)
    html_hash = sha256(ROOT / 'Magnetic_Camera.html')
    exe_hash = sha256(ROOT / 'Magnetic_Camera.exe')
    source_changes = []
    missing_sources = []
    for item in manifest:
        path = Path(item['Path'])
        if not path.is_file():
            missing_sources.append(str(path))
        elif sha256(path).upper() != item['Hash'].upper():
            source_changes.append(str(path))

    checks = {
        'trajectory_time_strictly_increases': bool(np.all(np.diff(trajectory.time_s) > 0)),
        'centerline_arc_strictly_reverses': bool(np.all(np.diff(trajectory.arc_s_mm) < 0)),
        'withdrawal_starts_at_distal_end': bool(trajectory.arc_s_mm.iloc[0] > trajectory.arc_s_mm.iloc[-1]),
        'no_planned_waits': bool(np.all(trajectory.camera_wait_planned.to_numpy() == 0)),
        'body_quaternion_has_no_sign_flip': bool(np.all(quaternion_adjacent_dot >= 0)),
        'camera_stays_inside_forward_cone': bool(max(camera['cone_residual_deg']) <= summary['forward_cone_limit_deg'] + 1e-6),
        'withdrawal_new_area_is_monotonic': bool(np.all(np.diff(evaluation.withdrawal_added_area_mm2) >= -1e-8)),
        'withdrawal_addition_starts_at_zero': bool(abs(evaluation.withdrawal_added_area_mm2.iloc[0]) < 1e-8),
        'final_area_equals_insertion_plus_new': bool(np.isclose(
            summary['final_covered_area_mm2'],
            summary['insertion_covered_area_mm2'] + summary['withdrawal_added_area_mm2'])),
        'trajectory_camera_and_coverage_durations_match': bool(np.isclose(
            trajectory.time_s.iloc[-1], summary['duration_s'], atol=1e-6)),
        'magnetic_force_residual_is_small': bool(trajectory.applied_force_residual_N.abs().max() < 1e-10),
        'magnetic_torque_residual_is_small': bool(trajectory.applied_torque_residual_N_m.abs().max() < 1e-10),
        'maxwell_constraints_are_small': bool(max(
            trajectory.applied_maxwell_trace_T_m.abs().max(),
            trajectory.applied_maxwell_symmetry_T_m.abs().max()) < 1e-10),
        'embedded_html_matches_delivered_html': bool(
            html_hash == exe_report['embedded_sha256'] == exe_report['html_sha256']),
        'executable_hash_matches_build_report': bool(exe_hash == exe_report['exe_sha256']),
        'read_only_source_manifest_unchanged': not source_changes and not missing_sources,
    }

    report = {
        'checks': checks,
        'all_checks_passed': all(checks.values()),
        'source_files_verified': len(manifest),
        'changed_source_files': source_changes,
        'missing_source_files': missing_sources,
        'trajectory_duration_s': float(trajectory.time_s.iloc[-1]),
        'withdrawal_added_area_mm2': float(summary['withdrawal_added_area_mm2']),
        'coverage_gain_percentage_points': float(summary['coverage_gain_percentage_points']),
        'forward_cone_max_deg': float(max(camera['cone_residual_deg'])),
        'max_applied_field_mT': float(trajectory.applied_field_norm_mT.max()),
        'max_applied_gradient_T_m': float(trajectory.applied_gradient_norm_T_m.max()),
        'minimum_centerline_clearance_mm': float(trajectory.clearance_mm.min()),
        'capsule_wall_clearance_verified': False,
        'html_sha256': html_hash,
        'exe_sha256': exe_hash,
        'limitations': [
            'Rigid-geometry offline simulation; not clinical safety or image-quality validation.',
            'Fold-hidden mask is an approximation based on 37 sampled forward camera directions.',
            'Zero minimum centerline clearance does not demonstrate capsule-to-wall safety.',
            'Required field values are model inverse-solver outputs, not proof of real actuator feasibility.'
        ]
    }
    output = ROOT / 'withdrawal_validation.json'
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if not report['all_checks_passed']:
        raise SystemExit(1)


if __name__ == '__main__':
    validate()
