"""Independent external-package evaluation; never updates weights/scalers."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation

from field_data import decode, external_validation, load_packages, write_json
from field_model import FieldPredictor, physical_derivative


def error_metrics(predicted, expected, floor):
    error = np.asarray(predicted, dtype=np.float64)-expected
    return dict(rmse=float(np.sqrt(np.mean(error**2))), mae=float(np.mean(np.abs(error))),
        max_abs=float(np.max(np.abs(error))),
        relative_l2=float(np.linalg.norm(error)/max(np.linalg.norm(expected), floor*np.sqrt(expected.size))),
        relative_denominator_floor_per_component=floor)


def response_metrics(predicted, expected, field_floor, gradient_floor):
    pf, pg = decode(predicted)
    ef, eg = decode(expected)
    return dict(field_T_per_A=error_metrics(pf, ef, field_floor),
        gradient_T_per_m_per_A=error_metrics(pg, eg, gradient_floor),
        per_coil=[dict(coil=k, field=error_metrics(pf[:, :, k], ef[:, :, k], field_floor),
            gradient=error_metrics(pg[:, k], eg[:, k], gradient_floor)) for k in range(8)])


def trajectory_metrics(predictor, records, field_floor, gradient_floor, training_x, overlap_radius_m):
    reports = []
    tree = cKDTree(training_x)
    for record in records:
        for phase in ('insertion', 'withdrawal'):
            path = Path(record['run_folder'])/'simulation'/phase/'trajectory.npz'
            if not path.exists():
                continue
            with np.load(path, allow_pickle=False) as archive:
                d = {key: archive[key] for key in ('q_local_m', 'array_quaternion', 'currents',
                    'moment_world_A_m2', 'field', 'gradient', 'force', 'torque')}
            keep = tree.query(d['q_local_m'])[0] > overlap_radius_m
            excluded = int((~keep).sum())
            if not keep.any():
                reports.append(dict(case_id=record['case_id'], phase=phase, samples=0,
                                    overlap_excluded=excluded, skipped='all positions overlap training'))
                continue
            d = {key: value[keep] for key, value in d.items()}
            values, outside = predictor.predict(d['q_local_m'], allow_outside=True)
            fields, gradients = decode(values)
            matrix = Rotation.from_quat(d['array_quaternion']).as_matrix()
            local_field = np.einsum('nik,nk->ni', fields, d['currents'])
            local_gradient = np.einsum('nk,nkij->nij', d['currents'], gradients)
            world_field = np.einsum('nij,nj->ni', matrix, local_field)
            world_gradient = matrix @ local_gradient @ matrix.transpose(0, 2, 1)
            force = np.einsum('nji,nj->ni', world_gradient, d['moment_world_A_m2'])
            torque = np.cross(d['moment_world_A_m2'], world_field)
            reports.append(dict(case_id=record['case_id'], phase=phase, samples=len(values), overlap_excluded=excluded,
                navigation_status=record['navigation_status'], outside_training_box=int(outside.sum()),
                current_max_abs_A=float(np.abs(d['currents']).max()),
                field_T=error_metrics(world_field, d['field'], field_floor),
                gradient_T_per_m=error_metrics(world_gradient, d['gradient'], gradient_floor),
                force_N=error_metrics(force, d['force'], 1e-12),
                torque_N_m=error_metrics(torque, d['torque'], 1e-12)))
    return reports


def evaluate(checkpoint, data_root, output, device='cuda', run_name='baseline_30A_long_4h',
             overlap_radius_m=1e-6, field_floor=1e-10, gradient_floor=1e-9):
    if min(field_floor, gradient_floor) <= 0:
        raise ValueError('Relative-error denominator floors must be positive')
    predictor = FieldPredictor(checkpoint, device)
    metadata = predictor.metadata
    reference_path = Path(checkpoint).parent/'training_reference.npz'
    with np.load(reference_path, allow_pickle=False) as d:
        reference_x, reference_y = d['q_local_m'], d['targets_96']
    training = dict(x=reference_x, audit={key: metadata[key] for key in ('physics_signature', 'solver_signature')})
    training['audit']['case_ids'] = metadata['training_case_ids']
    data = external_validation(training, load_packages(data_root, run_name), overlap_radius_m)
    predicted, outside = predictor.predict(data['x'], allow_outside=True)
    distance, nearest = cKDTree(reference_x).query(data['x'])
    field_mean = np.broadcast_to(np.asarray(metadata['normalizer']['y_mean']), data['y'].shape)
    metrics = response_metrics(predicted, data['y'], field_floor, gradient_floor)
    spatial = []
    for lo, hi in ((0., .001), (.001, .003), (.003, .01), (.01, float('inf'))):
        ids = (distance >= lo) & (distance < hi)
        if ids.any():
            spatial.append(dict(nearest_train_distance_m=[lo, hi if np.isfinite(hi) else None],
                samples=int(ids.sum()), metrics=response_metrics(predicted[ids], data['y'][ids], field_floor, gradient_floor)))
    derivative_ids = np.linspace(0, len(data['x'])-1, min(128, len(data['x'])), dtype=int)
    q = torch.as_tensor(data['x'][derivative_ids], dtype=torch.float32, device=predictor.device)
    derivative, direct = physical_derivative(predictor.model, q)
    derivative, direct = derivative.detach().cpu().numpy(), direct.detach().cpu().numpy()
    label_gradient = decode(data['y'][derivative_ids])[1]
    report = dict(external_data_audit=data['audit'], metrics=metrics, samples=len(predicted),
        per_case={case: response_metrics(predicted[data['cases'] == case], data['y'][data['cases'] == case],
            field_floor, gradient_floor) for case in np.unique(data['cases'])},
        nearest_neighbor_baseline=response_metrics(reference_y[nearest], data['y'], field_floor, gradient_floor),
        training_mean_baseline=response_metrics(field_mean, data['y'], field_floor, gradient_floor),
        spatial_distance_groups=spatial, outside_training_box=int(outside.sum()),
        derivative_consistency=dict(samples=len(q), direct_vs_dB_dq=error_metrics(direct, derivative, gradient_floor),
            dB_dq_vs_label=error_metrics(derivative, label_gradient, gradient_floor)),
        recorded_current_reconstruction=trajectory_metrics(predictor, data['audit']['records'], field_floor, gradient_floor,
            reference_x, overlap_radius_m),
        note='Recorded-current reconstruction reuses saved trajectories/currents; it is not a new closed-loop navigation run. External cases may share the same field-function domain.',
        model_accepted=False, acceptance_thresholds_frozen=False, closed_loop_validated=False,
        hardware_validated=False, speedup_measured=False)
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    write_json(output/'evaluation.json', report)
    np.savez_compressed(output/'predictions.npz', q_local_m=data['x'], expected=data['y'], predicted=predicted,
        case_id=data['cases'], outside_training_box=outside, nearest_training_distance_m=distance)
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='External validation and physical response evaluation; no training')
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--data-root', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--run-name', default='baseline_30A_long_4h')
    parser.add_argument('--overlap-radius-m', type=float, default=1e-6)
    parser.add_argument('--field-floor', type=float, default=1e-10)
    parser.add_argument('--gradient-floor', type=float, default=1e-9)
    args = parser.parse_args()
    result = evaluate(args.checkpoint, args.data_root, args.output, args.device, args.run_name,
        args.overlap_radius_m, args.field_floor, args.gradient_floor)
    print(json.dumps(result['metrics'], ensure_ascii=False, indent=2))
