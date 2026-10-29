"""Follow up frozen-model diagnostics; no optimization or dataset changes."""
import os
os.environ['NUMBA_CACHE_DIR']='D:/mag/训练结果/field_mlp_v1_analysis/numba_cache'
import sys
sys.dont_write_bytecode=True
sys.path.insert(0,'D:/mag/代码')
import numpy as np
import torch
from pathlib import Path
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation
from bootstrap import load_physics
from compact_coils.physics import FiniteArray
from field_model import FieldPredictor, physical_derivative
from field_data import read_json,write_json,encode,decode,file_hash
from evaluate_field import error_metrics
sys.stdout.reconfigure(encoding='utf-8')
torch.set_num_threads(2)
root=Path('D:/mag/训练结果/field_mlp_v1')
out=Path('D:/mag/训练结果/field_mlp_v1_analysis/error_sources')
before=file_hash(root/'last.pt')
predictor=FieldPredictor(root/'last.pt','cuda')
rng=np.random.default_rng(20261024)
with np.load(root/'training_reference.npz') as d: x,y,cases=d['q_local_m'],d['targets_96'],d['case_id']
with np.load(out/'per_sample_diagnostics.npz') as d: kinds,clearance,loss=d['sampling_kind'],d['source_envelope_clearance_m'],d['field_loss']+d['gradient_loss']
array=load_physics()
ids=np.unique(np.concatenate([np.argsort(clearance)[:5],np.argsort(loss)[-5:],rng.choice(np.flatnonzero(clearance<.01),5,replace=False),rng.choice(np.flatnonzero(kinds=='trajectory'),5,replace=False)]))
values={}
for multiplier in (1,2,4,6):
 a=array if multiplier==1 else FiniteArray(array.design,array.layout_scale,quadrature_multiplier=multiplier)
 values[multiplier]=np.array([encode(f[None],g[None])[0] for f,g in (a.responses(x[i]) for i in ids)])
 print('quadrature multiplier',multiplier,'checked',len(ids),'existing points',flush=True)
refinement=[]
for j,i in enumerate(ids):
 refinement.append(dict(index=int(i),kind=str(kinds[i]),clearance_mm=float(clearance[i]*1000),
  changes=[dict(from_multiplier=m,to_multiplier=n,field_relative_l2=float(np.linalg.norm(values[n][j,:24]-values[m][j,:24])/np.linalg.norm(values[n][j,:24])),gradient_relative_l2=float(np.linalg.norm(values[n][j,24:]-values[m][j,24:])/np.linalg.norm(values[n][j,24:]))) for m,n in [(1,2),(2,4),(4,6)]],
  base_vs_m6_field_relative_l2=float(np.linalg.norm(values[1][j,:24]-values[6][j,:24])/np.linalg.norm(values[6][j,:24])),
  base_vs_m6_gradient_relative_l2=float(np.linalg.norm(values[1][j,24:]-values[6][j,24:])/np.linalg.norm(values[6][j,24:]))))
report=dict(quadrature_refinement=refinement,scope='Diagnostic only; selected existing positions, no new training data, no weight updates')
write_json(out/'verification_partial.json',report)

checks=[]
ratios=[]
force_truth=[]
torque_truth=[]
for rec in read_json(root/'data_audit.json')['training']['records']:
 for phase in ('insertion','withdrawal'):
  path=Path(rec['run_folder'])/'simulation'/phase/'trajectory.npz'
  if not path.exists(): continue
  with np.load(path) as d:
   selected=np.unique(np.linspace(0,len(d['q_local_m'])-1,8).astype(int))
   b={k:d[k][selected] for k in ('q_local_m','currents','array_quaternion','moment_world_A_m2','field','gradient','force','torque')}
   force_truth.append(d['force']);torque_truth.append(d['torque'])
  rot=Rotation.from_quat(b['array_quaternion']).as_matrix()
  f,g=zip(*(array.responses(q) for q in b['q_local_m']));f,g=np.array(f),np.array(g)
  local=np.einsum('nik,nk->ni',f,b['currents'])
  field=np.einsum('nij,nj->ni',rot,local)
  gradient=rot@np.einsum('nk,nkij->nij',b['currents'],g)@rot.transpose(0,2,1)
  force=np.einsum('nji,nj->ni',gradient,b['moment_world_A_m2'])
  torque=np.cross(b['moment_world_A_m2'],field)
  checks.append(dict(case=rec['case_id'],phase=phase,samples=len(selected),**{k:error_metrics(v,b[k],1e-12) for k,v in [('field',field),('gradient',gradient),('force',force),('torque',torque)]}))
  contributions=f*b['currents'][:,None,:]
  ratios.extend(np.linalg.norm(contributions,axis=1).sum(1)/np.maximum(np.linalg.norm(local,axis=1),1e-12))
report['original_solver_recorded_current_checks']=checks
report['field_cancellation_factor_percentiles']=np.percentile(ratios,[0,25,50,75,90,99,100]).tolist()
report['field_cancellation_factor_definition']='sum of norms of eight coil contributions / norm of their vector sum; selected recorded states'
for name,arrs in [('force',force_truth),('torque',torque_truth)]:
 a=np.concatenate(arrs);report[name+'_truth']=dict(component_rms=float(np.sqrt(np.mean(a*a))),vector_norm_percentiles=np.percentile(np.linalg.norm(a,axis=1),[0,50,90,99,100]).tolist())
print('Recorded solver reconstruction verified',flush=True)

# Same-data nearest-neighbor diagnostic: exclude the queried coordinate itself.
# This is not a validation estimate: nearby points can share trajectory parents.
idx=cKDTree(x).query(x,k=2)[1][:,1]
report['other_point_interpolation']={}
for kind in np.unique(kinds):
 s=kinds==kind
 neighbor=y[idx[s]];truth=y[s]
 g=neighbor[:,24:].reshape(-1,8,3,3)
 dx=x[s]-x[idx[s]]
 corrected=neighbor[:,:24]+np.einsum('nkij,nj->nki',g,dx).reshape(-1,24)
 report['other_point_interpolation'][str(kind)]=dict(samples=int(s.sum()),nearest_field=error_metrics(neighbor[:,:24],truth[:,:24],1e-12),nearest_gradient=error_metrics(neighbor[:,24:],truth[:,24:],1e-12),first_order_field=error_metrics(corrected,truth[:,:24],1e-12),note='Other training points; trajectory neighbors may share parent or path. Not independent validation.')

derivative_checks={}
for kind in np.unique(kinds):
 ids=rng.choice(np.flatnonzero(kinds==kind),min(2048,int(np.sum(kinds==kind))),replace=False)
 q=torch.as_tensor(x[ids],dtype=torch.float32,device='cuda')
 dg,direct=physical_derivative(predictor.model,q)
 dg,direct=dg.detach().cpu().numpy(),direct.detach().cpu().numpy()
 label=y[ids,24:].reshape(-1,8,3,3)
 derivative_checks[str(kind)]=dict(samples=len(ids),direct_vs_label=error_metrics(direct,label,1e-12),dB_dq_vs_label=error_metrics(dg,label,1e-12),direct_vs_dB_dq=error_metrics(direct,dg,1e-12))
report['representative_derivative_checks']=derivative_checks
assert file_hash(root/'last.pt')==before
report['checkpoint_unchanged']=True
write_json(out/'verification.json',report)
print('REPORT '+str(out/'verification.json'),flush=True)
