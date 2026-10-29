"""Read-only model diagnostics on training data; not external validation."""
import json
from pathlib import Path
import numpy as np
from field_model import FieldPredictor
from evaluate_field import response_metrics
from field_data import write_json

root = Path('D:/mag/训练结果/field_mlp_v1')
predictor = FieldPredictor(root/'last.pt', 'cuda')
with np.load(root/'training_reference.npz', allow_pickle=False) as d:
    x, y, cases = d['q_local_m'], d['targets_96'], d['case_id']
pred, _ = predictor.predict(x)
mean = np.asarray(predictor.metadata['normalizer']['y_mean'])
scale = np.asarray(predictor.metadata['normalizer']['y_scale'])
groups = {}
for name, s in [('field', slice(0,24)), ('gradient',slice(24,96))]:
    error = pred[:,s]-y[:,s]
    standardized_mse = float(np.mean((error/scale[s])**2))
    baseline_mse = float(np.mean(((y[:,s]-mean[s])/scale[s])**2))
    mse_channel = np.mean(error**2, axis=0)
    var_channel = np.var(y[:,s],axis=0)
    rms_truth = np.sqrt(np.mean(y[:,s]**2,axis=0))
    groups[name] = dict(standardized_mse=standardized_mse,
        standardized_rmse=float(np.sqrt(standardized_mse)),
        mean_baseline_standardized_mse=baseline_mse,
        improvement_vs_mean_mse=1-standardized_mse/baseline_mse,
        per_channel_r2=(1-mse_channel/np.maximum(var_channel,1e-30)).tolist(),
        per_sample_relative_l2_percentiles=np.percentile(np.linalg.norm(error,axis=1)/np.maximum(np.linalg.norm(y[:,s],axis=1),1e-12),[50,90,95,99]).tolist(),
        per_channel_rmse=np.sqrt(mse_channel).tolist(), reference_component_rms=rms_truth.tolist())
rows=[json.loads(t) for t in (root/'history.jsonl').read_text().splitlines()]
report=dict(scope='Training-set diagnostics only; no parameter updates, no external validation',
    samples=len(x), target_metrics=response_metrics(pred,y,1e-10,1e-9), groups=groups,
    total_standardized_mse=sum(v['standardized_mse'] for v in groups.values()),
    per_case={c:response_metrics(pred[cases==c],y[cases==c],1e-10,1e-9) for c in np.unique(cases)},
    curve={str(i):float(np.mean([r['training_loss'] for r in rows[i-50:i]])) for i in (50,100,200,300,400,500)})
out=Path('D:/mag/训练结果/field_mlp_v1_analysis')
out.mkdir(exist_ok=True)
write_json(out/'training_fit_diagnostics.json',report)
print(json.dumps({k:v for k,v in report.items() if k!='per_case'},ensure_ascii=False,indent=2))
