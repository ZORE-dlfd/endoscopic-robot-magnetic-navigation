from pathlib import Path
import sys
import shutil
from types import SimpleNamespace

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from magnetic_app.domain.replay import Replay
from magnetic_app.rendering.picking import pick_centerline
from magnetic_app.domain.algorithms import algorithm_sections


def test_sparse_segment_midpoint_and_clipping():
    clip = np.array([[-.8, 0., 0., 1.], [.8, 0., 0., 1.]])
    assert pick_centerline(clip, (500, 400), (1000, 800)) in (0, 1)
    assert pick_centerline(clip, (500, 450), (1000, 800)) == -1
    clip[:, 3] = -1
    assert pick_centerline(clip, (500, 400), (1000, 800)) == -1
    clip[:, 3] = 1
    clip[:, 2] = 2
    assert pick_centerline(clip, (500, 400), (1000, 800)) == -1


def test_four_plain_language_algorithm_sections():
    sections = algorithm_sections()
    assert len(sections) == 4
    assert list(sections) == ['机器人运动', '磁场解算', '进镜摄像头', '退镜摄像头']
    for text in sections.values():
        assert len(text) > 400
        assert '$' not in text
        assert chr(92) not in text
    assert '1.2' in sections['退镜摄像头'] and '95%' in sections['退镜摄像头']


def test_predictive_preserves_original_motion_and_insertion():
    original, current = Replay(load_predictive=False), Replay()
    assert current.prediction['method'] == 'dense-short-horizon-pd-v1'
    assert original.data.keys() == current.data.keys()
    for key in original.data:
        np.testing.assert_array_equal(current.data[key], original.data[key])
    before = original.camera_times <= original.join_time
    for key in original.camera:
        np.testing.assert_allclose(np.asarray(current.camera[key])[:before.sum()], np.asarray(original.camera[key])[before], atol=1e-10)
    np.testing.assert_array_equal(current.insertion_seen, original.insertion_seen)
    assert current.duration == original.duration
    assert len(current.payload['cameraMotion']['roundtrip_new']['roll_rate_deg_s']) == len(current.camera_times)
    assert current.prediction['metrics']['constraint_violating_steps'] == 0
    assert current.prediction['metrics']['prediction_fallback_count'] == 0
    insertion, added = current.coverage(current.duration)
    assert not (insertion & added).any()
    assert current.areas[added].sum() == pytest.approx(current.summary['withdrawal_added_area_mm2'], abs=1e-7)
    assert current.summary['final_covered_area_mm2'] == pytest.approx(current.areas[insertion | added].sum())


@pytest.mark.parametrize("problem", ["missing", "corrupted"])
def test_prediction_never_silently_falls_back(tmp_path, problem):
    source = Replay(load_predictive=False).path
    for filename in ["metadata.json", "arrays.npz", "predictive_camera.json", "predictive_camera.npz"]:
        shutil.copy2(source / filename, tmp_path / filename)
    track = tmp_path / "predictive_camera.npz"
    if problem == "missing":
        track.unlink()
    else:
        with track.open("ab") as stream:
            stream.write(b"corrupted")
    with pytest.raises(ValueError, match="缺失|校验失败"):
        Replay(tmp_path)


def test_joint_angle_readout_takes_short_path_at_roll_boundary():
    source = SimpleNamespace(camera={"joint_angle_deg": np.array([[0., 0., 179.], [0., 0., -179.]])})
    frame = SimpleNamespace(camera_index=0, camera_amount=.5)
    result = Replay.camera_value(source, frame, "joint_angle_deg")
    np.testing.assert_allclose(result, [0., 0., 180.])
