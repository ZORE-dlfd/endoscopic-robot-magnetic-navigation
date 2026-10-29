"""CPU-only interface tests. No optimizer, backward pass or training is run."""
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import sys

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from field_data import decode, encode, external_validation, load_packages, write_json
from field_model import FieldMLP, FieldPredictor, checkpoint_payload, fit_normalizer, physical_derivative, response_loss, select_device
from evaluate_field import error_metrics, evaluate


def make_package(root, case, x, incomplete=False, corrupt_order=False, windows=False):
    run = Path(root)/case/'baseline_30A_long_4h'
    for name in ('dataset', 'config', 'validation'):
        (run/name).mkdir(parents=True)
    count = len(x)
    # Smooth analytical field fixture; all points are far from source volumes.
    coefficients = np.arange(1, 73, dtype=float).reshape(8, 3, 3)*1e-5
    fields = np.einsum('kij,nj->nik', coefficients, x)
    gradients = np.broadcast_to(coefficients, (count, 8, 3, 3)).copy()
    targets = encode(fields, gradients)
    if corrupt_order:
        targets[:, 0] += 1.
    np.savez_compressed(run/'dataset'/'field_responses.npz', q_local_m=x,
        fields_T_per_A=fields, gradients_T_per_m_per_A=gradients, targets_96=targets,
        sample_id=np.arange(count), parent_id=np.arange(count), sampling_kind=np.full(count, 'workspace'),
        case_id=np.full(count, case), config_hash=np.full(count, 'cfg'), solver_hash=np.full(count, 'raw-hash'))
    write_json(run/'status.json', dict(case_id=case, status='incomplete' if incomplete else 'complete', dataset_generated=True))
    write_json(run/'config'/'settings_resolved.json', dict(case_id=case, config_hash='cfg', solver_hash='raw-hash'))
    write_json(run/'config'/'baseline_assumptions.json', dict(design=dict(core_length_m=.05,
        winding_length_m=.04, core_radius_m=.01, winding_outer_radius_m=.02), layout_scale=1.,
        offsets_m=[[1.+k, 1., 1.] for k in range(8)], axes=[[0., 0., 1.]]*8))
    key = 'vendor\\physics.py' if windows else 'vendor/physics.py'
    write_json(run/'config'/'source_manifest.json', dict(source_sha256={key: 'same-source'}))
    write_json(run/'validation'/'physical_consistency.json', dict(reconstruction_passed=True))
    return run


class Interfaces(unittest.TestCase):
    def setUp(self):
        # Explicit D-drive test location; no system temporary directory is used.
        parent = Path(__file__).resolve().parents[1]/'test_artifacts'
        parent.mkdir(exist_ok=True)
        self.temp = tempfile.TemporaryDirectory(dir=parent)
        self.root = Path(self.temp.name)

    def tearDown(self):
        self.temp.cleanup()

    def test_encode_decode_and_force_axes(self):
        f = np.arange(48).reshape(2, 3, 8)
        g = np.arange(144).reshape(2, 8, 3, 3)
        y = encode(f, g)
        np.testing.assert_array_equal(y[0, :6], f[0, :, :2].T.reshape(-1))
        ff, gg = decode(y)
        np.testing.assert_array_equal(ff, f)
        np.testing.assert_array_equal(gg, g)
        moment = np.array([[1., 2., 3.], [3., 4., 5.]])
        matrix = g.sum(1)
        force = np.einsum('nji,nj->ni', matrix, moment)
        np.testing.assert_array_equal(force, np.stack([m.T@v for m, v in zip(matrix, moment)]))

    def test_all_local_cases_included_and_cross_platform_hashes(self):
        x = np.array([[0., 0., 0.], [.01, .02, .03]])
        make_package(self.root, 'colon_001', x, windows=True)
        make_package(self.root, 'colon_002', x+.1, incomplete=True)
        data = load_packages(self.root)
        self.assertEqual(set(data['audit']['case_ids']), {'colon_001', 'colon_002'})
        self.assertEqual(len(data['x']), 4)
        self.assertEqual(data['audit']['records'][1]['navigation_status'], 'incomplete')
        restricted = load_packages(self.root, include_incomplete=False)
        self.assertEqual(restricted['audit']['case_ids'], ['colon_001'])

    def test_corrupt_order_rejected(self):
        make_package(self.root, 'colon_001', np.zeros((2, 3)), corrupt_order=True)
        with self.assertRaisesRegex(ValueError, 'ordering'):
            load_packages(self.root)

    def test_external_overlap_removed_and_case_leakage_rejected(self):
        make_package(self.root/'train', 'colon_001', np.array([[0., 0., 0.], [.1, .1, .1]]))
        make_package(self.root/'val', 'colon_002', np.array([[0., 0., 0.], [.05, .05, .05]]))
        train = load_packages(self.root/'train')
        val = external_validation(train, load_packages(self.root/'val'))
        self.assertEqual(len(val['x']), 1)
        self.assertEqual(val['audit']['overlap_removed'], 1)
        with self.assertRaisesRegex(ValueError, 'case IDs'):
            external_validation(train, train)

    def test_normalization_chain_rule_and_checkpoint_roundtrip(self):
        norm = dict(x_mean=[0.]*3, x_scale=[2., 3., 4.], y_mean=[0.]*96, y_scale=[5.]*96)
        model = FieldMLP([], norm)
        with torch.no_grad():
            model.network[0].weight.fill_(1.)
            model.network[0].bias.zero_()
        q = torch.tensor([[.1, .2, .3]])
        derivative, _ = physical_derivative(model, q)
        np.testing.assert_allclose(derivative.detach().numpy(), np.broadcast_to([2.5, 5/3, 1.25], (1, 8, 3, 3)), rtol=1e-6)
        audit = dict(physics_signature='p', solver_signature='s', case_ids=['colon_001'])
        path = self.root/'model.pt'
        torch.save(checkpoint_payload(model, [], norm, audit, [[-1.]*3, [1.]*3], 0, {}), path)
        predictor = FieldPredictor(path, 'cpu')
        pred, _ = predictor.predict(q.numpy())
        np.testing.assert_allclose(pred, model.physical(q).detach().numpy())
        with self.assertRaisesRegex(ValueError, 'Outside'):
            predictor.predict([[2., 0., 0.]])

    def test_scaler_fits_only_input_arrays_and_constant_outputs_are_finite(self):
        x = np.array([[0., 0., 0.], [2., 4., 8.]])
        norm = fit_normalizer(x, np.zeros((2, 96)))
        np.testing.assert_allclose(norm['x_mean'], [1., 2., 4.])
        self.assertTrue(np.all(np.asarray(norm['y_scale']) > 0))

    def test_sample_weighted_loss_is_normalized_and_changes_sample_priority(self):
        target = torch.zeros((2, 96))
        prediction = torch.zeros_like(target)
        prediction[0] = 1.
        equal = response_loss(prediction, target)
        uniform = response_loss(prediction, target, sample_weight=torch.tensor([2., 2.]))
        prioritized = response_loss(prediction, target, sample_weight=torch.tensor([3., 1.]))
        torch.testing.assert_close(equal, uniform)
        self.assertGreater(prioritized.item(), equal.item())

    def test_cuda_no_silent_fallback(self):
        with patch('torch.cuda.is_available', return_value=False):
            with self.assertRaisesRegex(RuntimeError, 'no CPU fallback'):
                select_device('cuda')

    def test_evaluation_runs_without_optimizer_or_weight_updates(self):
        x = np.array([[0., 0., 0.], [.1, .1, .1]])
        make_package(self.root/'train', 'colon_001', x)
        make_package(self.root/'val', 'colon_002', np.array([[.02, .03, .04], [.05, .06, .07]]))
        data = load_packages(self.root/'train')
        norm = fit_normalizer(data['x'], data['y'])
        model = FieldMLP([4], norm)
        before = {k: v.clone() for k, v in model.state_dict().items()}
        checkpoint = self.root/'untrained_interface_fixture.pt'
        torch.save(checkpoint_payload(model, [4], norm, data['audit'], [[0.]*3, [.1]*3], 0, {}), checkpoint)
        np.savez_compressed(self.root/'training_reference.npz', q_local_m=data['x'], targets_96=data['y'])
        with patch('torch.optim.AdamW', side_effect=AssertionError('Training must not run')):
            report = evaluate(checkpoint, self.root/'val', self.root/'report', device='cpu')
        self.assertEqual(report['samples'], 2)
        self.assertFalse(report['model_accepted'])
        self.assertTrue((self.root/'report'/'evaluation.json').exists())
        for key, value in model.state_dict().items():
            torch.testing.assert_close(value, before[key])
        self.assertEqual(error_metrics(data['y'], data['y'], 1e-10)['rmse'], 0.)


if __name__ == '__main__':
    unittest.main(verbosity=2)
