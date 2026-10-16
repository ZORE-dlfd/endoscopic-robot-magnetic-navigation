from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from magnetic_app.domain.replay import Replay
from magnetic_app.rendering.geometry import Domains, arrows, capsule, risk_color


@pytest.fixture(scope='module')
def context():
    replay = Replay()
    return replay, Domains(replay), replay.frame(410.123)


@pytest.mark.parametrize('mode', ['total', 'residual', 'turning'])
@pytest.mark.parametrize('scope', ['local', 'centerline', 'whole'])
def test_streamline_buffers_finite(context, mode, scope):
    replay, domains, frame = context
    data = domains.streamline_vertices(frame, scope, mode, replay.registry['field_scope_ranges'][scope][mode][1])
    assert data.shape[1] == 11
    assert data.shape[0] % 2 == 0
    assert np.isfinite(data).all()


def test_robot_glyphs_and_risk_colors():
    data = capsule()
    assert np.isfinite(data).all()
    assert data[:, 0].min() == -6 and data[:, 0].max() == 6
    np.testing.assert_allclose(np.linalg.norm(data[:, 3:6], axis=1), 1, atol=1e-6)
    glyph = arrows([[0, 0, 0]], [[0, 0, 1]])
    assert np.isfinite(glyph).all()
    np.testing.assert_allclose(risk_color([0, 1]), [[16/255, 0, 122/255], [197/255, 0, 0]])
def test_original_field_display_scales():
    from magnetic_app.rendering.geometry import field_lengths
    np.testing.assert_allclose(field_lengths([10., 15.], 'total', 'local'), [4.6, 6.9])
    assert field_lengths([10., 100.], 'total', 'centerline').max() == pytest.approx(11.5)
    np.testing.assert_allclose(field_lengths([.02, .02], 'turning', 'local'), [7.475, 7.475])
