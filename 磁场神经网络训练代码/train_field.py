import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')

import argparse
import json
import random
import shutil
import time
from pathlib import Path

import numpy as np
import torch

from field_data import external_validation, file_hash, load_packages, read_json, write_json
from field_model import FieldMLP, checkpoint_payload, fit_normalizer, response_loss, select_device


def load_config(path):
    config = read_json(path)
    if config['device'] != 'cuda' and not config['device'].startswith('cuda:'):
        raise ValueError('Training configuration must explicitly request CUDA')
    if config['epochs'] < 1 or config['batch_size'] < 1 or not config['hidden'] or min(config['hidden']) < 1:
        raise ValueError('Invalid epochs, batch size or hidden layers')
    if config['learning_rate'] <= 0 or min(config['field_weight'], config['gradient_weight']) <= 0:
        raise ValueError('Learning rate and loss weights must be positive')
    if 'sampling_weights' in config and any(value <= 0 for value in config['sampling_weights'].values()):
        raise ValueError('Sampling weights must be positive')
    return config


def load_inputs(config):
    data = load_packages(config['train_root'], config['run_name'], config['include_incomplete'])
    val = None
    if config.get('validation_root'):
        val = load_packages(config['validation_root'], config['run_name'], config['include_incomplete'])
        val = external_validation(data, val, config['validation_overlap_radius_m'])
    return data, val


def train(config):
    device = select_device(config['device'])
    torch.set_num_threads(config['cpu_threads'])
    random.seed(config['seed'])
    np.random.seed(config['seed'])
    torch.manual_seed(config['seed'])
    torch.cuda.manual_seed_all(config['seed'])
    torch.use_deterministic_algorithms(True)
    data, val = load_inputs(config)
    output = Path(config['output_dir']).resolve()
    output.mkdir(parents=True, exist_ok=False)
    write_json(output/'config.json', config)
    write_json(output/'data_audit.json', dict(training=data['audit'], validation=val['audit'] if val else None))
    write_json(output/'source_sha256.json', {p.name: file_hash(p) for p in Path(__file__).parent.glob('*.py')})
    snapshot = output/'source_snapshot'
    snapshot.mkdir()
    for path in Path(__file__).parent.iterdir():
        if path.is_file() and path.suffix in ('.py', '.json', '.md'):
            shutil.copy2(path, snapshot/path.name)
    np.savez_compressed(output/'training_reference.npz', q_local_m=data['x'], targets_96=data['y'],
                        case_id=data['cases'], sampling_kind=data['kinds'])
    normalizer = fit_normalizer(data['x'], data['y'])
    write_json(output/'normalizer.json', normalizer)
    model = FieldMLP(config['hidden'], normalizer).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config['learning_rate'], weight_decay=config['weight_decay'])
    # Dataset remains in CPU memory; only current minibatches are transferred.
    x = torch.from_numpy(data['x'].astype(np.float32))
    y = torch.from_numpy(((data['y']-np.array(normalizer['y_mean']))/np.array(normalizer['y_scale'])).astype(np.float32))
    sample_weights = config.get('sampling_weights', {})
    unknown_kinds = set(np.unique(data['kinds'])) - set(sample_weights)
    if sample_weights and unknown_kinds:
        raise ValueError(f'Missing sampling weights for kinds: {sorted(unknown_kinds)}')
    weights = torch.as_tensor([sample_weights.get(kind, 1.) for kind in data['kinds']], dtype=torch.float32)
    vx = torch.from_numpy(val['x'].astype(np.float32)) if val else None
    vy = torch.from_numpy(((val['y']-np.array(normalizer['y_mean']))/np.array(normalizer['y_scale'])).astype(np.float32)) if val else None
    vw = torch.as_tensor([sample_weights.get(kind, 1.) for kind in val['kinds']], dtype=torch.float32) if val and sample_weights else None
    bounds = [data['x'].min(0).tolist(), data['x'].max(0).tolist()]
    generator = torch.Generator().manual_seed(config['seed'])
    best, best_epoch, stale = float('inf'), None, 0
    started = time.perf_counter()

    def save(name, epoch):
        payload = checkpoint_payload(model, config['hidden'], normalizer, data['audit'], bounds, epoch, config)
        payload['optimizer_state'] = optimizer.state_dict()
        temporary = output/(name+'.tmp')
        torch.save(payload, temporary)
        temporary.replace(output/name)

    write_json(output/'status.json', dict(status='training', device=str(device), validation_available=val is not None))
    print(json.dumps(dict(device=str(device), gpu=torch.cuda.get_device_name(device), training_samples=len(x),
        validation_samples=len(vx) if val else 0, output=str(output)), ensure_ascii=False), flush=True)
    try:
        with (output/'history.jsonl').open('w', encoding='utf-8') as log:
            for epoch in range(1, config['epochs']+1):
                model.train()
                total = 0.
                order = torch.randperm(len(x), generator=generator)
                for ids in order.split(config['batch_size']):
                    bx, by = x[ids].to(device), y[ids].to(device)
                    optimizer.zero_grad(set_to_none=True)
                    loss = response_loss(model(bx), by, config['field_weight'], config['gradient_weight'], weights[ids].to(device))
                    if not torch.isfinite(loss):
                        raise FloatingPointError('Nonfinite training loss')
                    loss.backward()
                    optimizer.step()
                    total += loss.item()*len(ids)
                validation_loss = None
                if val:
                    model.eval()
                    value = 0.
                    with torch.inference_mode():
                        for start in range(0, len(vx), config['batch_size']):
                            bx, by = vx[start:start+config['batch_size']].to(device), vy[start:start+config['batch_size']].to(device)
                            weight_batch = vw[start:start+config['batch_size']].to(device) if vw is not None else None
                            value += response_loss(model(bx), by, config['field_weight'], config['gradient_weight'], weight_batch).item()*len(bx)
                    validation_loss = value/len(vx)
                    if not np.isfinite(validation_loss):
                        raise FloatingPointError('Nonfinite validation loss')
                    if validation_loss < best:
                        best, best_epoch, stale = validation_loss, epoch, 0
                        save('best.pt', epoch)
                    else:
                        stale += 1
                row = dict(epoch=epoch, training_loss=total/len(x), validation_loss=validation_loss,
                           elapsed_s=time.perf_counter()-started)
                log.write(json.dumps(row)+'\n')
                log.flush()
                print(json.dumps(row), flush=True)
                save('last.pt', epoch)
                write_json(output/'status.json', dict(status='training', **row, validation_available=val is not None))
                if val and config['patience'] > 0 and stale >= config['patience']:
                    break
        final = dict(status='training_finished', epochs=epoch, best_epoch=best_epoch,
            validation_available=val is not None, selected_checkpoint='best.pt' if val else 'last.pt',
            elapsed_s=time.perf_counter()-started, gpu=torch.cuda.get_device_name(device),
            model_validated=False, closed_loop_validated=False, hardware_validated=False,
            note='Training completed; accuracy and navigation acceptance require separate evaluation.')
        write_json(output/'status.json', final)
        return final
    except BaseException as exc:
        write_json(output/'status.json', dict(status='failed', error=str(exc), elapsed_s=time.perf_counter()-started))
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='CUDA field surrogate training; all local cases are training data')
    parser.add_argument('--config', required=True)
    parser.add_argument('--audit-only', action='store_true', help='Read/audit inputs only; no model or optimizer')
    args = parser.parse_args()
    cfg = load_config(args.config)
    if args.audit_only:
        training, validation = load_inputs(cfg)
        print(json.dumps(dict(training=training['audit'], validation=validation['audit'] if validation else None), ensure_ascii=False, indent=2))
    else:
        print(json.dumps(train(cfg), ensure_ascii=False))
