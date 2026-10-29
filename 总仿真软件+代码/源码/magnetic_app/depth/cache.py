"""Persistent depth maps for deterministic route playback."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import numpy as np


CACHE_VERSION = 1
CACHE_FILENAME = "network_depth.npy"
MANIFEST_FILENAME = "network_depth.json"
PARTIAL_FILENAME = "network_depth.partial.npy"
PROGRESS_FILENAME = "network_depth.progress.json"


def resolve_cache_directory(
    signature: str,
    app_data_directory: str | Path,
    executable_path: str | Path | None = None,
) -> Path:
    if executable_path is not None:
        cache_root = Path(executable_path).resolve().parent / "缓存数据"
    else:
        cache_root = Path(app_data_directory) / "depth-cache"
    return cache_root / str(signature)


def route_signature(replay_path: str | Path) -> str:
    """Hash the route inputs and cache format without depending on install path."""
    root = Path(replay_path)
    digest = hashlib.sha256(f"depth-cache:{CACHE_VERSION}\n".encode("ascii"))
    for filename in ("metadata.json", "arrays.npz", "predictive_camera.npz"):
        path = root / filename
        digest.update(filename.encode("utf-8"))
        if not path.is_file():
            digest.update(b"missing")
            continue
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
    return digest.hexdigest()


class DepthMapCache:
    def __init__(self, directory: str | Path, signature: str, frame_count: int):
        self.directory = Path(directory)
        self.signature = str(signature)
        self.frame_count = int(frame_count)
        self.data_path = self.directory / CACHE_FILENAME
        self.manifest_path = self.directory / MANIFEST_FILENAME
        self.partial_path = self.directory / PARTIAL_FILENAME
        self.progress_path = self.directory / PROGRESS_FILENAME
        self._stream = None
        self._data_offset = 0
        self.shape = None
        self.completed_count = 0
        self.is_valid = self._open_complete()

    @staticmethod
    def _read_header(stream):
        version = np.lib.format.read_magic(stream)
        if version == (1, 0):
            shape, fortran_order, dtype = np.lib.format.read_array_header_1_0(stream)
        elif version == (2, 0):
            shape, fortran_order, dtype = np.lib.format.read_array_header_2_0(stream)
        else:
            raise ValueError(f"unsupported depth cache format version: {version}")
        return tuple(shape), bool(fortran_order), np.dtype(dtype), stream.tell()

    @staticmethod
    def _write_header(stream, shape):
        np.lib.format.write_array_header_1_0(stream, {
            "descr": np.lib.format.dtype_to_descr(np.dtype(np.float16)),
            "fortran_order": False,
            "shape": shape,
        })

    def _open_complete(self) -> bool:
        try:
            manifest = json.loads(self.manifest_path.read_text(encoding="utf-8"))
            shape = tuple(int(value) for value in manifest["shape"])
            if (manifest.get("version") != CACHE_VERSION
                    or manifest.get("signature") != self.signature
                    or int(manifest.get("frame_count", -1)) != self.frame_count
                    or manifest.get("complete") is not True
                    or len(shape) != 2):
                return False
            stream = self.data_path.open("rb")
            data_shape, fortran_order, dtype, data_offset = self._read_header(stream)
            expected_shape = (self.frame_count, *shape)
            expected_size = data_offset + int(np.prod(expected_shape)) * dtype.itemsize
            if (data_shape != expected_shape or fortran_order or dtype != np.float16
                    or stream.seek(0, os.SEEK_END) != expected_size):
                stream.close()
                return False
            stream.seek(data_offset)
            self._stream = stream
            self._data_offset = data_offset
            self.shape = shape
            self.completed_count = self.frame_count
            return True
        except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError):
            if self._stream is not None:
                self._stream.close()
                self._stream = None
            return False

    def begin(self, shape: tuple[int, int]) -> int:
        self.directory.mkdir(parents=True, exist_ok=True)
        shape = tuple(int(value) for value in shape)
        self.shape = shape
        progress = self._read_progress()
        resumable = (
            progress is not None
            and progress.get("signature") == self.signature
            and int(progress.get("frame_count", -1)) == self.frame_count
            and tuple(progress.get("shape", ())) == shape
            and self.partial_path.is_file()
        )
        if resumable:
            stream = None
            try:
                stream = self.partial_path.open("r+b")
                data_shape, fortran_order, dtype, data_offset = self._read_header(stream)
                completed_count = min(
                    max(int(progress.get("completed_count", 0)), 0), self.frame_count)
                required_size = data_offset + completed_count * int(np.prod(shape)) * dtype.itemsize
                if (data_shape != (self.frame_count, *shape) or fortran_order
                        or dtype != np.float16 or stream.seek(0, os.SEEK_END) < required_size):
                    raise ValueError("partial cache shape or dtype does not match")
                stream.seek(data_offset)
                self._stream = stream
                self._data_offset = data_offset
                self.completed_count = completed_count
            except (OSError, ValueError, TypeError):
                if stream is not None:
                    stream.close()
                resumable = False
        if not resumable:
            self._stream = self.partial_path.open("w+b")
            self._write_header(self._stream, (self.frame_count, *shape))
            self._data_offset = self._stream.tell()
            self.completed_count = 0
            self._write_progress()
        self.is_valid = False
        return self.completed_count

    def _read_progress(self):
        try:
            return json.loads(self.progress_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return None

    def _write_progress(self):
        payload = {
            "signature": self.signature,
            "frame_count": self.frame_count,
            "shape": self.shape,
            "completed_count": self.completed_count,
        }
        temp = self.progress_path.with_suffix(self.progress_path.suffix + ".tmp")
        temp.write_text(json.dumps(payload), encoding="utf-8")
        os.replace(temp, self.progress_path)

    def close_partial(self):
        if self._stream is not None:
            self._stream.flush()
            self._stream.close()
        self._stream = None

    def write(self, index: int, depth: np.ndarray) -> None:
        if self._stream is None or self.shape is None:
            raise RuntimeError("depth cache generation has not started")
        index = int(index)
        if index != self.completed_count:
            raise ValueError(f"expected depth frame {self.completed_count}, got {index}")
        depth = np.asarray(depth, dtype=np.float32)
        if depth.size == int(np.prod(self.shape)):
            depth = depth.reshape(self.shape)
        if depth.shape != self.shape:
            raise ValueError(f"expected depth shape {self.shape}, got {depth.shape}")
        frame_bytes = depth.astype(np.float16).tobytes(order="C")
        frame_offset = self._data_offset + index * len(frame_bytes)
        self._stream.seek(frame_offset)
        self._stream.write(frame_bytes)
        self._stream.flush()
        self.completed_count += 1
        self._write_progress()

    def finalize(self) -> None:
        if self._stream is None or self.completed_count != self.frame_count:
            raise RuntimeError("cannot finalize an incomplete depth cache")
        self._stream.flush()
        self._stream.close()
        self._stream = None
        os.replace(self.partial_path, self.data_path)
        manifest = {
            "version": CACHE_VERSION,
            "signature": self.signature,
            "frame_count": self.frame_count,
            "shape": self.shape,
            "dtype": "float16",
            "complete": True,
        }
        temp = self.manifest_path.with_suffix(self.manifest_path.suffix + ".tmp")
        temp.write_text(json.dumps(manifest), encoding="utf-8")
        os.replace(temp, self.manifest_path)
        self.progress_path.unlink(missing_ok=True)
        self.is_valid = self._open_complete()
        if not self.is_valid:
            raise RuntimeError("completed depth cache could not be reopened")

    def get(self, index: int) -> np.ndarray:
        if not self.is_valid or self._stream is None:
            raise RuntimeError("depth cache is not complete")
        index = int(np.clip(index, 0, self.frame_count - 1))
        frame_bytes = int(np.prod(self.shape)) * np.dtype(np.float16).itemsize
        self._stream.seek(self._data_offset + index * frame_bytes)
        payload = self._stream.read(frame_bytes)
        if len(payload) != frame_bytes:
            raise ValueError(f"depth cache frame {index} is truncated")
        return np.frombuffer(payload, dtype=np.float16).reshape(self.shape).astype(np.float32)

    def get_interpolated(self, index: int, amount: float) -> np.ndarray:
        index = int(np.clip(index, 0, self.frame_count - 1))
        first = self.get(index)
        if index >= self.frame_count - 1:
            return first
        amount = float(np.clip(amount, 0.0, 1.0))
        if amount == 0.0:
            return first
        second = self.get(index + 1)
        return first * (1.0 - amount) + second * amount
