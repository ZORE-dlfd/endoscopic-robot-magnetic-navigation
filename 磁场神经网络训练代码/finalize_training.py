"""Save a finished run's curve and verify checkpoint loading; no training."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch

from field_data import read_json, write_json
from field_model import FieldPredictor


def finalize(root):
    status = read_json(root/'status.json')
    config = read_json(root/'config.json')
    assert status['status'] == 'training_finished'
    predictor = FieldPredictor(root/'last.pt', 'cuda')
    assert predictor.metadata['epoch'] == status['epochs']
    assert all(torch.isfinite(v).all().item() for v in predictor.model.state_dict().values())
    with np.load(root/'training_reference.npz', allow_pickle=False) as data:
        q = data['q_local_m'][np.linspace(0, len(data['q_local_m'])-1, 128, dtype=int)]
    values, outside = predictor.predict(q)
    assert values.shape == (128, 96) and np.isfinite(values).all() and not outside.any()
    rows = [json.loads(line) for line in (root/'history.jsonl').read_text(encoding='utf-8').splitlines()]
    assert [r['epoch'] for r in rows] == list(range(1, status['epochs']+1))
    assert all(np.isfinite(r['training_loss']) for r in rows)
    audit = read_json(root/'data_audit.json')['training']
    for kind in ('stdout', 'stderr'):
        source = root.parent/(root.name+'_'+kind+'.log')
        if source.exists():
            shutil.copy2(source, root/(kind+'.log'))
    fig, ax = plt.subplots(figsize=(9, 4.8))
    ax.plot([r['epoch'] for r in rows], [r['training_loss'] for r in rows], color='#2474d6', lw=1.8)
    ax.set(xlabel='Epoch', ylabel='Standardized training loss (field + gradient)',
           title='Field surrogate | RTX 2080 Ti | no external validation')
    ax.grid(alpha=.25)
    fig.tight_layout()
    fig.savefig(root/'training_loss.png', dpi=180)
    plt.close(fig)
    report = dict(status=status, training_cases=audit['case_ids'], training_samples=audit['unique_samples'],
        first_training_loss=rows[0]['training_loss'], last_training_loss=rows[-1]['training_loss'],
        checkpoint_reload_passed=True, checkpoint_epoch=predictor.metadata['epoch'],
        gpu_prediction_shape=list(values.shape), gpu_prediction_finite=True, validation_available=False,
        note='128 training positions checked for finite inference only. No independent accuracy or closed-loop validation performed.')
    write_json(root/'training_summary.json', report)
    lines = ['GPU training finished: '+str(status['epochs'])+' epochs.',
        'Elapsed seconds: '+str(status['elapsed_s']), 'Model: last.pt',
        'Samples: '+str(audit['unique_samples']), 'Cases: '+', '.join(audit['case_ids']),
        'Final standardized training loss: '+str(rows[-1]['training_loss']),
        'Checkpoint reload and 128-position CUDA finite-inference checks passed.',
        'External validation has not been performed.',
        'For validation on another device, keep this entire folder, especially last.pt, training_reference.npz, source_snapshot, config.json and data_audit.json.']
    (root/'RESULTS.txt').write_text('\n'.join(lines), encoding='utf-8')
    manifest = {p.relative_to(root).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in sorted(root.rglob('*')) if p.is_file() and p.name != 'artifacts_sha256.json'}
    write_json(root/'artifacts_sha256.json', manifest)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print('Saved artifacts:', len(manifest))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--folder', required=True)
    finalize(Path(parser.parse_args().folder))
