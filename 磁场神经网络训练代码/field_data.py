"""Load frozen response packages without splitting local training cases."""
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

TARGET_ORDER = 'coil-major fields[8,3], then gradients[8,3,3]; G[component,coordinate]'


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')


def file_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def signature(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def encode(fields, gradients):
    return np.concatenate([fields.transpose(0, 2, 1).reshape(-1, 24), gradients.reshape(-1, 72)], axis=1)


def decode(values):
    return values[:, :24].reshape(-1, 8, 3).transpose(0, 2, 1), values[:, 24:].reshape(-1, 8, 3, 3)


def load_packages(root, run_name='baseline_30A_long_4h', include_incomplete=True):
    root = Path(root).resolve()
    if not root.is_dir():
        raise ValueError(f'Data root does not exist: {root}')
    xs, ys, cases, kinds, records, skipped = [], [], [], [], [], []
    physics_signature = solver_signature = None
    for run in sorted(root.glob(f'colon_*/{run_name}')):
        case = run.parent.name
        status_path, data_path = run/'status.json', run/'dataset'/'field_responses.npz'
        status = read_json(status_path) if status_path.exists() else {}
        if status.get('status') not in ('complete', 'incomplete') or not status.get('dataset_generated'):
            skipped.append(dict(case_id=case, reason='not finalized with a dataset', status=status.get('status')))
            continue
        if status['status'] == 'incomplete' and not include_incomplete:
            skipped.append(dict(case_id=case, reason='incomplete navigation excluded by config'))
            continue
        settings = read_json(run/'config'/'settings_resolved.json')
        baseline = read_json(run/'config'/'baseline_assumptions.json')
        manifest = read_json(run/'config'/'source_manifest.json')
        sources = {k.replace('\\', '/'): v for k, v in manifest['source_sha256'].items()}
        # Aggregate hashes differ across Windows/Linux if path separators differ.
        normalized_solver = signature(sources)
        physics = {key: baseline[key] for key in ('design', 'layout_scale', 'offsets_m', 'axes')}
        physics_hash = signature(physics)
        if physics_signature not in (None, physics_hash) or solver_signature not in (None, normalized_solver):
            raise ValueError(f'{case}: physical config or normalized source manifest differs')
        physics_signature, solver_signature = physics_hash, normalized_solver
        consistency = read_json(run/'validation'/'physical_consistency.json')
        if consistency.get('reconstruction_passed') is not True:
            raise ValueError(f'{case}: saved physical reconstruction did not pass')
        validation_path = run/'validation'/'package_validation.json'
        if validation_path.exists() and read_json(validation_path).get('data_integrity_passed') is not True:
            raise ValueError(f'{case}: existing package validation failed')
        with np.load(data_path, allow_pickle=False) as archive:
            data = {key: archive[key] for key in archive.files}
        count = len(data['q_local_m'])
        shapes = dict(q_local_m=(count, 3), fields_T_per_A=(count, 3, 8),
                      gradients_T_per_m_per_A=(count, 8, 3, 3), targets_96=(count, 96))
        for name, shape in shapes.items():
            if data[name].shape != shape or not np.isfinite(data[name]).all():
                raise ValueError(f'{case}: invalid shape or nonfinite {name}')
        if count == 0 or status.get('case_id') != case or settings['case_id'] != case:
            raise ValueError(f'{case}: empty dataset or mismatched case identity')
        for name in ('case_id', 'sample_id', 'parent_id', 'sampling_kind', 'config_hash', 'solver_hash'):
            if data[name].shape != (count,):
                raise ValueError(f'{case}: invalid metadata shape {name}')
        if not np.all(data['case_id'] == case):
            raise ValueError(f'{case}: sample case identity mismatch')
        for name in ('config_hash', 'solver_hash'):
            if not np.all(data[name] == settings[name]):
                raise ValueError(f'{case}: {name} provenance mismatch')
        if len(np.unique(data['sample_id'])) != count or np.any(data['parent_id'] < 0):
            raise ValueError(f'{case}: invalid sample/parent IDs')
        expected = encode(data['fields_T_per_A'], data['gradients_T_per_m_per_A'])
        if not np.array_equal(expected, data['targets_96']):
            raise ValueError(f'{case}: 96-dimensional output ordering mismatch')
        q = data['q_local_m']
        offsets, axes = np.array(baseline['offsets_m']), np.array(baseline['axes'])
        displacement = q[:, None] - offsets[None]
        axial = np.einsum('nki,ki->nk', displacement, axes)
        radial = np.linalg.norm(displacement - axial[..., None]*axes[None], axis=-1)
        design = baseline['design']
        half_length = max(design['core_length_m'], design['winding_length_m'])/2 + .001
        radius = max(design['core_radius_m'], design['winding_outer_radius_m']) + .001
        if np.any((np.abs(axial) < half_length) & (radial < radius)):
            raise ValueError(f'{case}: query inside excluded magnetic source region')
        xs.append(q)
        ys.append(expected)
        cases.append(np.full(count, case))
        kinds.append(data['sampling_kind'].astype(str))
        records.append(dict(case_id=case, run_name=run_name, run_folder=str(run), samples=count,
            navigation_status=status['status'], phases=status.get('phases'),
            sha256=file_hash(data_path), config_hash=settings['config_hash'],
            solver_hash=settings['solver_hash'], package_validation_present=validation_path.exists(),
            internal_reconstruction_passed=True))
    if not xs:
        raise ValueError(f'No finalized response packages in {root}')
    x, y, case_ids, sample_kinds = np.concatenate(xs), np.concatenate(ys), np.concatenate(cases), np.concatenate(kinds)
    # Deduplicate exact coordinates across cases, but reject conflicting labels.
    _, first, inverse = np.unique(x, axis=0, return_index=True, return_inverse=True)
    if not np.allclose(y, y[first[inverse]], rtol=1e-8, atol=1e-12):
        raise ValueError('Identical query coordinates have conflicting labels')
    keep = np.sort(first)
    kind_counts = {str(kind): int(np.sum(sample_kinds[keep] == kind)) for kind in np.unique(sample_kinds[keep])}
    audit = dict(root=str(root), records=records, skipped=skipped, raw_samples=len(x),
        unique_samples=len(keep), exact_duplicates_removed=len(x)-len(keep),
        samples_by_kind=kind_counts,
        case_ids=[row['case_id'] for row in records], physics_signature=physics_signature,
        solver_signature=solver_signature, target_order=TARGET_ORDER,
        scope='All local packages are training data; no internal holdout. Incomplete navigation is not a successful trial.',
        validation_scope='Structural/provenance audit plus saved internal reconstruction; not independent solver or hardware validation.')
    return dict(x=x[keep], y=y[keep], cases=case_ids[keep], kinds=sample_kinds[keep], audit=audit)


def external_validation(training, validation, overlap_radius_m=1e-6):
    a, b = training['audit'], validation['audit']
    if set(a['case_ids']) & set(b['case_ids']):
        raise ValueError('Training and validation contain overlapping case IDs')
    if any(a[key] != b[key] for key in ('physics_signature', 'solver_signature')):
        raise ValueError('External validation physics/source signatures differ')
    if overlap_radius_m < 0:
        raise ValueError('overlap_radius_m must be nonnegative')
    distance = cKDTree(training['x']).query(validation['x'])[0]
    keep = distance > overlap_radius_m
    if not keep.any():
        raise ValueError('All external validation positions overlap training positions')
    validation = dict(validation)
    validation.update({key: validation[key][keep] for key in ('x', 'y', 'cases', 'kinds') if key in validation})
    validation['audit'] = dict(b, overlap_removed=int((~keep).sum()), retained_samples=int(keep.sum()),
        overlap_radius_m=overlap_radius_m, nearest_training_distance_m_percentiles=np.percentile(distance[keep], [0, 50, 95, 100]).tolist(),
        scope='External cases; near-duplicate local positions removed. This is not a spatial-block holdout or proof of new-field-function generalization.')
    return validation
