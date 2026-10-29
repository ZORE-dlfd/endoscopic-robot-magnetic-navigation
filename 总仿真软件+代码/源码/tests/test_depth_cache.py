from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "源码"))

from magnetic_app.depth.cache import DepthMapCache, resolve_cache_directory


def test_resolve_cache_directory_uses_portable_bundle_when_frozen(tmp_path):
    executable_path = tmp_path / "bundle" / "Magnetic_Camera_8Coil.exe"

    cache_directory = resolve_cache_directory(
        "route-a",
        app_data_directory=tmp_path / "appdata",
        executable_path=executable_path,
    )

    assert cache_directory == tmp_path / "bundle" / "缓存数据" / "route-a"


def test_resolve_cache_directory_uses_appdata_for_source_runs(tmp_path):
    cache_directory = resolve_cache_directory(
        "route-a",
        app_data_directory=tmp_path / "appdata",
    )

    assert cache_directory == tmp_path / "appdata" / "depth-cache" / "route-a"


def test_depth_cache_round_trips_maps_and_interpolates(tmp_path):
    cache = DepthMapCache(tmp_path, signature="route-a", frame_count=3)
    cache.begin((2, 2))
    cache.write(0, np.full((2, 2), 1, dtype=np.float32))
    cache.write(1, np.full((2, 2), 3, dtype=np.float32))
    cache.write(2, np.full((2, 2), 5, dtype=np.float32))
    cache.finalize()

    reopened = DepthMapCache(tmp_path, signature="route-a", frame_count=3)
    assert reopened.is_valid
    np.testing.assert_allclose(reopened.get_interpolated(0, 0.5), 2.0)
    np.testing.assert_allclose(reopened.get_interpolated(1, 0.25), 3.5)


def test_depth_cache_rejects_a_different_signature(tmp_path):
    cache = DepthMapCache(tmp_path, signature="route-a", frame_count=1)
    cache.begin((1, 1))
    cache.write(0, np.ones((1, 1), dtype=np.float32))
    cache.finalize()

    reopened = DepthMapCache(tmp_path, signature="route-b", frame_count=1)
    assert not reopened.is_valid


def test_depth_cache_does_not_accept_incomplete_generation(tmp_path):
    cache = DepthMapCache(tmp_path, signature="route-a", frame_count=2)
    cache.begin((1, 1))
    cache.write(0, np.ones((1, 1), dtype=np.float32))

    reopened = DepthMapCache(tmp_path, signature="route-a", frame_count=2)
    assert not reopened.is_valid


def test_depth_cache_resumes_partial_generation_using_npy_metadata(tmp_path):
    cache = DepthMapCache(tmp_path, signature="route-a", frame_count=2)
    cache.begin((3, 4))
    cache.write(0, np.full((3, 4), 7, dtype=np.float32))
    cache.close_partial()

    resumed = DepthMapCache(tmp_path, signature="route-a", frame_count=2)
    assert resumed.begin((3, 4)) == 1
    resumed.write(1, np.full((3, 4), 9, dtype=np.float32))
    resumed.finalize()

    reopened = DepthMapCache(tmp_path, signature="route-a", frame_count=2)
    assert reopened.is_valid
    np.testing.assert_allclose(reopened.get(0), 7)
    np.testing.assert_allclose(reopened.get(1), 9)


def test_depth_cache_streams_frames_when_large_memmap_is_unavailable(tmp_path, monkeypatch):
    def reject_memmap(*args, **kwargs):
        raise OSError(22, "Invalid argument", str(args[0]))

    monkeypatch.setattr(np.lib.format, "open_memmap", reject_memmap)

    cache = DepthMapCache(tmp_path, signature="route-large", frame_count=3)
    assert cache.begin((2, 2)) == 0
    cache.write(0, np.full((2, 2), 7, dtype=np.float32))
    cache.close_partial()

    resumed = DepthMapCache(tmp_path, signature="route-large", frame_count=3)
    assert resumed.begin((2, 2)) == 1
    resumed.write(1, np.full((2, 2), 9, dtype=np.float32))
    resumed.write(2, np.full((2, 2), 11, dtype=np.float32))
    resumed.finalize()

    reopened = DepthMapCache(tmp_path, signature="route-large", frame_count=3)
    assert reopened.is_valid
    np.testing.assert_allclose(reopened.get(0), 7)
    np.testing.assert_allclose(reopened.get(1), 9)
    np.testing.assert_allclose(reopened.get(2), 11)
