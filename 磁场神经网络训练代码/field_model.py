"""MLP response model; public predictions use metres and physical response units."""
import numpy as np
import torch
from torch import nn

from field_data import TARGET_ORDER


class FieldMLP(nn.Module):
    def __init__(self, hidden, normalizer):
        super().__init__()
        layers = []
        sizes = [3, *hidden, 96]
        for i, (left, right) in enumerate(zip(sizes[:-1], sizes[1:])):
            layers.append(nn.Linear(left, right))
            if i < len(sizes)-2:
                layers.append(nn.SiLU())
        self.network = nn.Sequential(*layers)
        for name, value in normalizer.items():
            self.register_buffer(name, torch.as_tensor(value, dtype=torch.float32))

    def forward(self, q_m):
        return self.network((q_m-self.x_mean)/self.x_scale)

    def physical(self, q_m):
        return self(q_m)*self.y_scale+self.y_mean


def fit_normalizer(x, y):
    scale = y.std(0)
    for group in (slice(0, 24), slice(24, 96)):
        floor = max(float(np.sqrt(np.mean(scale[group]**2)))*.01, 1e-15)
        scale[group] = np.maximum(scale[group], floor)
    return dict(x_mean=x.mean(0).tolist(), x_scale=np.maximum(x.std(0), 1e-6).tolist(),
                y_mean=y.mean(0).tolist(), y_scale=scale.tolist())


def select_device(name):
    device = torch.device(name)
    if device.type == 'cuda':
        if not torch.cuda.is_available():
            raise RuntimeError('CUDA requested but unavailable; no CPU fallback')
        torch.cuda.get_device_properties(device)
    return device


def response_loss(prediction, target, field_weight=1., gradient_weight=1., sample_weight=None):
    field_error = (prediction[:, :24]-target[:, :24]).square().mean(dim=1)
    gradient_error = (prediction[:, 24:]-target[:, 24:]).square().mean(dim=1)
    if sample_weight is None:
        field, gradient = field_error.mean(), gradient_error.mean()
    else:
        weights = sample_weight / sample_weight.mean()
        field = (field_error*weights).mean()
        gradient = (gradient_error*weights).mean()
    return field_weight*field+gradient_weight*gradient


def physical_derivative(model, q_m):
    """dB/dq in physical units, including normalization's chain rule."""
    q = q_m.detach().clone().requires_grad_(True)
    values = model.physical(q)
    derivatives = [torch.autograd.grad(values[:, i].sum(), q, retain_graph=True)[0]
                   for i in range(24)]
    return torch.stack(derivatives, dim=1).reshape(-1, 8, 3, 3), values[:, 24:].reshape(-1, 8, 3, 3)


def checkpoint_payload(model, hidden, normalizer, audit, bounds, epoch, config):
    return dict(format_version=1, target_order=TARGET_ORDER, hidden=hidden, normalizer=normalizer,
        model_state={k: v.detach().cpu().clone() for k, v in model.state_dict().items()},
        physics_signature=audit['physics_signature'], solver_signature=audit['solver_signature'],
        training_case_ids=audit['case_ids'], training_bounds_m=bounds,
        epoch=epoch, config=config)


class FieldPredictor:
    def __init__(self, checkpoint, device='cuda'):
        self.device = select_device(device)
        saved = torch.load(checkpoint, map_location='cpu', weights_only=True)
        if saved['format_version'] != 1 or saved['target_order'] != TARGET_ORDER:
            raise ValueError('Unsupported checkpoint format or target order')
        self.metadata = saved
        self.model = FieldMLP(saved['hidden'], saved['normalizer']).to(self.device)
        self.model.load_state_dict(saved['model_state'])
        self.model.eval()
        self.lower, self.upper = np.asarray(saved['training_bounds_m'])

    def predict(self, q_local_m, batch_size=4096, allow_outside=False):
        q = np.asarray(q_local_m, dtype=np.float64)
        if q.ndim != 2 or q.shape[1] != 3 or not len(q) or not np.isfinite(q).all():
            raise ValueError('Expected a nonempty finite Nx3 array in metres')
        if batch_size < 1:
            raise ValueError('batch_size must be positive')
        outside = np.any((q < self.lower) | (q > self.upper), axis=1)
        if outside.any() and not allow_outside:
            raise ValueError('Outside training bounding box; use original solver fallback')
        parts = []
        with torch.inference_mode():
            for start in range(0, len(q), batch_size):
                values = torch.as_tensor(q[start:start+batch_size], dtype=torch.float32, device=self.device)
                parts.append(self.model.physical(values).cpu().numpy())
        result = np.concatenate(parts)
        if not np.isfinite(result).all():
            raise FloatingPointError('Nonfinite network prediction')
        # Inside this box does not certify accuracy or source exclusion.
        return result, outside
