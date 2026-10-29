"""Diagnose the frozen baseline using existing training data; never train."""
import os
os.environ['NUMBA_CACHE_DIR'] = 'D:/mag/训练结果/field_mlp_v1_analysis/numba_cache'
os.environ['MPLCONFIGDIR'] = 'D:/mag/训练结果/field_mlp_v1_analysis/mpl_cache'
import sys
sys.dont_write_bytecode = True
import json
import time
from pathlib import Path
import numpy as np
import torch
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation
from field_data import read_json, write_json, file_hash, encode, decode
from field_model import FieldPredictor, physical_derivative
from evaluate_field import error_metrics

ROOT = Path('D:/mag/训练结果/field_mlp_v1')
OUT = Path('D:/mag/训练结果/field_mlp_v1_analysis/error_sources')
OUT.mkdir(parents=True, exist_ok=True)
sys.stdout.reconfigure(encoding='utf-8')
torch.set_num_threads(2)
rng = np.random.default_rng(20261023)
started = time.perf_counter()
before = file_hash(ROOT/'last.pt')
predictor = FieldPredictor(ROOT/'last.pt', 'cuda')
with np.load(ROOT/'training_reference.npz', allow_pickle=False) as d:
    x, y, cases = d['q_local_m'], d['targets_96'], d['case_id']
pred, _ = predictor.predict(x)
scale = np.asarray(predictor.metadata['normalizer']['y_scale'])
tree = cKDTree(x)
audit = read_json(ROOT/'data_audit.json')['training']
kinds = np.full(len(x), '', dtype='<U10')
nav = np.full(len(x), '', dtype='<U12')
package_checks = []
for record in audit['records']:
    folder = Path(record['run_folder'])
    with np.load(folder/'dataset/field_responses.npz', allow_pickle=False) as d:
        distance, ids = tree.query(d['q_local_m'])
        assert np.all(distance == 0)
        assert np.array_equal(y[ids], d['targets_96'])
        owner = cases[ids] == record['case_id']
        kinds[ids[owner]] = d['sampling_kind'][owner]
        nav[ids[owner]] = record['navigation_status']
    package_checks.append(dict(case=record['case_id'], sha256_matches=file_hash(folder/'dataset/field_responses.npz') == record['sha256']))
assert np.all(kinds != '') and all(a['sha256_matches'] for a in package_checks)
baseline = read_json(Path(audit['records'][0]['run_folder'])/'config/baseline_assumptions.json')
offsets, axes = np.asarray(baseline['offsets_m']), np.asarray(baseline['axes'])
displacement = x[:, None]-offsets[None]
axial = np.einsum('nki,ki->nk', displacement, axes)
radial = np.linalg.norm(displacement-axial[..., None]*axes[None], axis=-1)
design = baseline['design']
half = max(design['core_length_m'], design['winding_length_m'])/2
radius = max(design['core_radius_m'], design['winding_outer_radius_m'])
# Distance to a conservative closed cylindrical envelope, not actual material.
clearance = np.sqrt(np.maximum(np.abs(axial)-half, 0)**2+np.maximum(radial-radius, 0)**2).min(1)
nn_distance = tree.query(x, k=2)[0][:, 1]
energy = {name: np.sum(y[:, s]**2, axis=1) for name, s in [('field', slice(0,24)), ('gradient', slice(24,96))]}
loss = {name: np.mean(((pred[:, s]-y[:, s])/scale[s])**2, axis=1) for name, s in [('field', slice(0,24)), ('gradient', slice(24,96))]}
total_loss = loss['field']+loss['gradient']

def group_metrics(ids):
    result = dict(samples=int(np.sum(ids)), sample_fraction=float(np.mean(ids)),
                  total_loss_share=float(total_loss[ids].sum()/total_loss.sum()))
    for name, s in [('field', slice(0,24)), ('gradient', slice(24,96))]:
        a, b = pred[ids, s], y[ids, s]
        rel = np.linalg.norm(a-b, axis=1)/np.maximum(np.linalg.norm(b, axis=1), 1e-12)
        result[name] = dict(error_metrics(a, b, 1e-12), standardized_mse=float(loss[name][ids].mean()),
            loss_share=float(loss[name][ids].sum()/loss[name].sum()),
            truth_energy_share=float(energy[name][ids].sum()/energy[name].sum()),
            sample_relative_l2_percentiles=np.percentile(rel, [50,90,95,99]).tolist(),
            fraction_samples_relative_error_below_10pct=float(np.mean(rel < .1)))
    return result

def quantile_groups(values):
    cuts = np.unique(np.quantile(values, [0,.25,.5,.75,.9,.95,.99,1]))
    return [dict(range=[float(lo), float(hi)], metrics=group_metrics((values >= lo)&((values < hi) if hi < cuts[-1] else (values <= hi))))
            for lo, hi in zip(cuts[:-1], cuts[1:]) if np.any((values >= lo)&(values <= hi))]

report = dict(scope='Existing training data diagnostics only; no holdout, no optimizer updates, no new navigation simulations',
    samples=len(x), gpu=torch.cuda.get_device_name(), checkpoint_sha256=before,
    package_hash_checks=package_checks, overall=group_metrics(np.ones(len(x), dtype=bool)),
    by_sampling_kind={k:group_metrics(kinds == k) for k in np.unique(kinds)},
    by_navigation_status={k:group_metrics(nav == k) for k in np.unique(nav)},
    by_case={k:group_metrics(cases == k) for k in np.unique(cases)},
    by_source_envelope_clearance_m=quantile_groups(clearance),
    by_nearest_other_sample_distance_m=quantile_groups(nn_distance),
    by_field_rms_T_per_A=quantile_groups(np.sqrt(energy['field']/24)),
    by_gradient_rms_T_per_m_A=quantile_groups(np.sqrt(energy['gradient']/72)),
    coordinates_m=dict(min=x.min(0).tolist(), max=x.max(0).tolist()),
    exclusions='Distance uses an enclosing cylinder; does not certify hardware clearance or accuracy.')
report['loss_concentration'] = {}
for name in ('field','gradient'):
    sorted_loss = np.sort(loss[name])[::-1]
    report['loss_concentration'][name] = {str(f):float(sorted_loss[:max(1,int(np.ceil(len(x)*f)))].sum()/sorted_loss.sum()) for f in (.001,.01,.05,.1)}
report['top_gradient_loss_samples'] = [dict(index=int(i), case=str(cases[i]), kind=str(kinds[i]),
    q_local_m=x[i].tolist(), source_envelope_clearance_mm=float(clearance[i]*1000),
    gradient_loss=float(loss['gradient'][i])) for i in np.argsort(loss['gradient'])[-20:][::-1]]
print('Grouped all training samples', flush=True)
write_json(OUT/'diagnostics_partial.json', report)

# Independently recompute existing labels, and differentiate the original field.
sys.path.insert(0, 'D:/mag/代码')
from bootstrap import load_physics
from compact_coils.physics import FiniteArray
array = load_physics()
np.testing.assert_allclose(array.offsets, offsets, rtol=0, atol=1e-15)
np.testing.assert_allclose(array.axes, axes, rtol=0, atol=1e-15)
manifest = read_json(Path(audit['records'][0]['run_folder'])/'config/source_manifest.json')
physics_key = next(k for k in manifest['source_sha256'] if k.replace('\\','/').endswith('compact_coils/physics.py'))
assert file_hash(Path('D:/mag/代码')/physics_key) == manifest['source_sha256'][physics_key]
check_ids = np.unique(np.concatenate([rng.choice(len(x),128,replace=False), np.argsort(total_loss)[-32:], np.argsort(clearance)[:16],
    *[rng.choice(np.flatnonzero(cases == c), 8, replace=False) for c in np.unique(cases)]]))
original = np.array([encode(f[None],g[None])[0] for f,g in (array.responses(x[i]) for i in check_ids)])
np.testing.assert_allclose(original, y[check_ids], rtol=1e-8, atol=1e-12)
recomputed = dict(samples=len(check_ids), field=error_metrics(original[:,:24],y[check_ids,:24],1e-12),
    gradient=error_metrics(original[:,24:],y[check_ids,24:],1e-12), allclose_passed=True)
finite_difference = {}
for step in (1e-6, 5e-7):
    derivatives=[]
    for i in check_ids:
        derivatives.append(np.stack([(array.responses(x[i]+axis*step)[0]-array.responses(x[i]-axis*step)[0])/(2*step) for axis in np.eye(3)],axis=-1).transpose(1,0,2).reshape(72))
    derivatives=np.asarray(derivatives)
    finite_difference[str(step)] = error_metrics(derivatives,y[check_ids,24:],1e-12)
report['label_checks'] = dict(original_solver_recalculation=recomputed, central_finite_difference=finite_difference,
    note='Diagnostic finite differences at existing points; these probes are not added to training data. Same physical assumptions, not independent hardware truth.')
print('Original solver and finite differences checked', flush=True)

# Probe whether original quadrature is converged near difficult points.
refine_ids = np.unique(np.concatenate([rng.choice(check_ids,16,replace=False), np.argsort(clearance)[:16], np.argsort(total_loss)[-16:]]))
refined = FiniteArray(array.design,array.layout_scale,quadrature_multiplier=2)
refined_y = np.array([encode(f[None],g[None])[0] for f,g in (refined.responses(x[i]) for i in refine_ids)])
report['quadrature_refinement'] = dict(samples=len(refine_ids), multiplier=2,
    field=error_metrics(refined_y[:,:24],y[refine_ids,:24],1e-12),
    gradient=error_metrics(refined_y[:,24:],y[refine_ids,24:],1e-12),
    per_point=[dict(index=int(i),kind=str(kinds[i]),clearance_mm=float(clearance[i]*1000),
        field_relative_l2=float(np.linalg.norm(a[:24]-b[:24])/np.linalg.norm(b[:24])),
        gradient_relative_l2=float(np.linalg.norm(a[24:]-b[24:])/np.linalg.norm(b[24:]))) for i,a,b in zip(refine_ids,refined_y,y[refine_ids])],
    note='One quadrature refinement is a numerical sensitivity check, not a proof of convergence.')
print('Quadrature sensitivity checked', flush=True)

derivative_ids = np.unique(np.concatenate([rng.choice(len(x),768,replace=False),np.argsort(total_loss)[-128:],np.argsort(clearance)[:64]]))
q=torch.as_tensor(x[derivative_ids],dtype=torch.float32,device='cuda')
derivative,direct=physical_derivative(predictor.model,q)
derivative,direct=derivative.detach().cpu().numpy(),direct.detach().cpu().numpy()
truth_g=y[derivative_ids,24:].reshape(-1,8,3,3)
def physical_residual(g):
    norm=max(float(np.linalg.norm(g)),1e-30)
    return dict(trace_over_tensor_norm=float(np.linalg.norm(np.trace(g,axis1=-2,axis2=-1))/norm),
        antisymmetric_over_tensor_norm=float(np.linalg.norm(g-g.swapaxes(-1,-2))/norm))
report['network_derivatives']=dict(samples=len(q),direct_vs_label=error_metrics(direct,truth_g,1e-12),
    derivative_vs_label=error_metrics(derivative,truth_g,1e-12), direct_vs_derivative=error_metrics(direct,derivative,1e-12),
    label_physical_residual=physical_residual(truth_g),direct_physical_residual=physical_residual(direct),
    derivative_physical_residual=physical_residual(derivative),
    note='Autograd of predicted B is compared with the separately predicted gradient head. Trace/curl are meaningful in a source-free, uniform-permeability query region.')

trajectory_reports=[]
aggregate={k:[[],[]] for k in ('field','gradient','force','torque')}
for record in audit['records']:
    for phase in ('insertion','withdrawal'):
        path=Path(record['run_folder'])/'simulation'/phase/'trajectory.npz'
        if not path.exists(): continue
        with np.load(path,allow_pickle=False) as d:
            a={k:d[k] for k in ('q_local_m','array_quaternion','currents','moment_world_A_m2','field','gradient','force','torque')}
        values,outside=predictor.predict(a['q_local_m'],allow_outside=True)
        f,g=decode(values)
        rot=Rotation.from_quat(a['array_quaternion']).as_matrix()
        field=np.einsum('nij,njk,nk->ni',rot,f,a['currents'])
        gradient=rot@np.einsum('nk,nkij->nij',a['currents'],g)@rot.transpose(0,2,1)
        force=np.einsum('nji,nj->ni',gradient,a['moment_world_A_m2'])
        torque=np.cross(a['moment_world_A_m2'],field)
        result=dict(case=record['case_id'],navigation_status=record['navigation_status'],phase=phase,samples=len(values),outside_box=int(outside.sum()))
        for k,estimate in [('field',field),('gradient',gradient),('force',force),('torque',torque)]:
            result[k]=error_metrics(estimate,a[k],1e-12)
            aggregate[k][0].append(estimate); aggregate[k][1].append(a[k])
        trajectory_reports.append(result)
report['recorded_current_reconstruction']=dict(per_case_phase=trajectory_reports,
    overall={k:error_metrics(np.concatenate(v[0]),np.concatenate(v[1]),1e-12) for k,v in aggregate.items()},
    note='Frozen recorded positions, currents and orientations; training trajectory reconstruction, not an external evaluation or new closed-loop navigation run.')
print('Recorded-current force reconstruction checked', flush=True)

np.savez_compressed(OUT/'per_sample_diagnostics.npz',q_local_m=x,case_id=cases,sampling_kind=kinds,
    source_envelope_clearance_m=clearance,nearest_other_sample_distance_m=nn_distance,
    field_loss=loss['field'],gradient_loss=loss['gradient'],
    field_rms=np.sqrt(energy['field']/24),gradient_rms=np.sqrt(energy['gradient']/72),
    field_relative_l2=np.linalg.norm(pred[:,:24]-y[:,:24],axis=1)/np.maximum(np.linalg.norm(y[:,:24],axis=1),1e-12),
    gradient_relative_l2=np.linalg.norm(pred[:,24:]-y[:,24:],axis=1)/np.maximum(np.linalg.norm(y[:,24:],axis=1),1e-12))
report['checkpoint_unchanged']=file_hash(ROOT/'last.pt') == before
assert report['checkpoint_unchanged']
report['elapsed_seconds']=time.perf_counter()-started
write_json(OUT/'diagnostics.json',report)
print(json.dumps({k:report[k] for k in ('loss_concentration','label_checks','quadrature_refinement','network_derivatives','checkpoint_unchanged','elapsed_seconds')},ensure_ascii=False),flush=True)
print('REPORT '+str(OUT/'diagnostics.json'),flush=True)
