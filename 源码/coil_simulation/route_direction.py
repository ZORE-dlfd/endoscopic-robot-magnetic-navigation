import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation


def insertion_source(reference, contract):
    source = pd.DataFrame(reference.source).copy()
    positions = source[['x_mm', 'y_mm', 'z_mm']].to_numpy(dtype=float)
    tolerance = float(contract['landmark_tolerance_mm'])
    expected = np.asarray([contract['entry_original_mm'], contract['deep_original_mm']], dtype=float)
    if (len(positions) < 4 or not np.isfinite(positions).all() or not np.isfinite(expected).all()
            or not np.isfinite(tolerance) or tolerance <= 0.):
        raise ValueError('Invalid insertion entry or endpoint contract')
    if np.any(np.linalg.norm(positions[[0, -1]]-expected, axis=1) > tolerance):
        raise ValueError('Insertion entry or deep endpoint differs from the confirmed original route')
    arc = source['arc_s_mm'].to_numpy(dtype=float)
    if not np.isfinite(arc).all() or abs(arc[0]) > 1e-8 or np.any(np.diff(arc) <= 0.):
        raise ValueError('Insertion arc must increase from the anal entry')
    if np.linalg.norm(reference.positions[0]-positions[0]) > 1.:
        raise ValueError('Reference playback does not start at the confirmed anal entry')
    tangent = np.gradient(positions, axis=0)
    tangent /= np.maximum(np.linalg.norm(tangent, axis=1, keepdims=True), 1e-12)
    axis = Rotation.from_quat(source[['qx', 'qy', 'qz', 'qw']]).as_matrix()[:, :, 0]
    if np.min(np.einsum('ij,ij->i', tangent, axis)) < .95:
        raise ValueError('Insertion body orientation must face inward along the anal-entry route')
    return source


def verify_phases(source, insertion, withdrawal, tolerance_mm=1.):
    positions = source[['x_mm', 'y_mm', 'z_mm']].to_numpy(dtype=float)
    entry, deep = positions[0], positions[-1]
    expected = [('insertion_start', insertion[0], entry), ('insertion_end', insertion[-1], deep),
                ('withdrawal_start', withdrawal[0], deep), ('withdrawal_end', withdrawal[-1], entry)]
    errors = {name+'_error_mm': float(np.linalg.norm(np.asarray(row['state'][:3])*1000.-target))
              for name, row, target in expected}
    if not all(np.isfinite(value) and value <= tolerance_mm for value in errors.values()):
        raise ValueError('Wrong insertion/withdrawal direction or endpoint: '+str(errors))
    if not np.array_equal(insertion[-1]['state'], withdrawal[0]['state']):
        raise ValueError('Capsule state discontinuity at insertion/withdrawal reversal')
    return dict(passed=True, tolerance_mm=float(tolerance_mm), **errors)


def direction_metadata(source, contract, verification):
    positions = source[['x_mm', 'y_mm', 'z_mm']].to_numpy(dtype=float)
    return dict(schema=1, insertion_start_label='肛门入口', insertion_end_label='肠道深部终点',
        withdrawal_start_label='肠道深部终点', withdrawal_end_label='肛门入口',
        entry_position_mm=positions[0].tolist(), deep_position_mm=positions[-1].tolist(),
        reference_entry_original_mm=contract['entry_original_mm'],
        reference_deep_original_mm=contract['deep_original_mm'],
        source='Original confirmed insertion source; legacy reverse CSV is not used for magnetic generation',
        arc_definition='0 at anal entry; increases to deep endpoint; decreases during withdrawal',
        verification=verification)
