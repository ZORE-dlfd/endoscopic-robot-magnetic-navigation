"""Read-only diagnostics for the centerline-weighted training run."""
import json
from pathlib import Path

import numpy as np

from evaluate_field import response_metrics
from field_data import write_json
from field_model import FieldPredictor

ROOT = Path('D:/mag/训练结果/field_mlp_centerline_v1')
predictor = FieldPredictor(ROOT/'last.pt', 'cuda')
with np.load(ROOT/'training_reference.npz', allow_pickle=False) as data:
    x, y, cases, kinds = (data[key] for key in ('q_local_m', 'targets_96', 'case_id', 'sampling_kind'))
predicted, _ = predictor.predict(x)
scale = np.asarray(predictor.metadata['normalizer']['y_scale'])
weights = predictor.metadata['config']['sampling_weights']

def group_metrics(mask):
    result = dict(samples=int(mask.sum()), response=response_metrics(predicted[mask], y[mask], 1e-10, 1e-9))
    result['standardized_mse'] = {
        'field': float(np.mean(((predicted[mask, :24]-y[mask, :24])/scale[:24])**2)),
        'gradient': float(np.mean(((predicted[mask, 24:]-y[mask, 24:])/scale[24:])**2))}
    return result

by_kind = {str(kind): group_metrics(kinds == kind) for kind in np.unique(kinds)}
weighted = np.asarray([weights[kind] for kind in kinds], dtype=np.float64)
normalized_error = ((predicted-y)/scale)**2
weighted_field = np.average(normalized_error[:, :24].mean(axis=1), weights=weighted)
weighted_gradient = np.average(normalized_error[:, 24:].mean(axis=1), weights=weighted)
report = dict(scope='Training-fit diagnostics only; no independent validation data was configured',
    samples=len(x), by_sampling_kind=by_kind, by_case={str(case): group_metrics(cases == case) for case in np.unique(cases)},
    unweighted_standardized_mse={'field': float(np.mean(((predicted[:,:24]-y[:,:24])/scale[:24])**2)),
        'gradient': float(np.mean(((predicted[:,24:]-y[:,24:])/scale[24:])**2))},
    weighted_objective=dict(field=float(weighted_field), gradient=float(weighted_gradient),
        total=float(weighted_field+weighted_gradient), sample_weights=weights))
write_json(ROOT/'fit_diagnostics.json', report)
print(json.dumps(report, ensure_ascii=False, indent=2))
