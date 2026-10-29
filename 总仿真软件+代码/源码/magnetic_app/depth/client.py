"""Depth client: capture a frame, call the depth service, colourise, export.

The heavy model lives in a separate process (深度服务/server.py) so the
PyInstaller bundle of the simulator stays small.
"""
import io
import urllib.request

import numpy as np

DEPTH_URL = "http://127.0.0.1:8765/estimate"

# compact turbo-ish LUT, interpolated to 256 entries
_LUT = np.array([
    [48, 18, 59], [68, 58, 131], [70, 107, 227], [62, 155, 254],
    [49, 196, 209], [46, 225, 147], [82, 246, 84], [150, 249, 54],
    [204, 231, 51], [238, 195, 57], [251, 148, 47], [247, 99, 26],
    [219, 52, 12], [163, 19, 6], [98, 4, 2], [48, 18, 59],
], dtype=np.float32)
_X = np.linspace(0, 1, len(_LUT))
_T = np.linspace(0, 1, 256)
TURBO = np.stack([np.interp(_T, _X, _LUT[:, c]) for c in range(3)], axis=1).astype(np.uint8)


def qimage_to_png(image) -> bytes:
    from PySide6.QtCore import QBuffer, QIODevice
    buf = QBuffer()
    buf.open(QIODevice.OpenModeFlag.WriteOnly)
    image.save(buf, "PNG")
    return bytes(buf.data())


def qimage_to_rgb(image) -> np.ndarray:
    from PySide6.QtGui import QImage
    image = image.convertToFormat(QImage.Format.Format_RGB888)
    w, h, bpl = image.width(), image.height(), image.bytesPerLine()
    raw = np.frombuffer(image.constBits(), dtype=np.uint8, count=h * bpl).reshape(h, bpl)
    return raw[:, :w * 3].reshape(h, w, 3).copy()


def estimate_depth(png_bytes: bytes, url: str = DEPTH_URL, timeout: float = 10.0) -> np.ndarray:
    request = urllib.request.Request(url, data=png_bytes, headers={"Content-Type": "image/png"})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        payload = response.read()
    depth = np.load(io.BytesIO(payload))
    return np.asarray(depth, dtype=np.float32).squeeze()


def colorize(depth: np.ndarray, vmin: float = 0.0, vmax: float | None = None) -> np.ndarray:
    depth = np.asarray(depth, dtype=np.float32)
    finite = depth[np.isfinite(depth)]
    if vmax is None:
        vmax = float(np.percentile(finite, 99)) if finite.size else 1.0
    vmax = max(vmax, vmin + 1e-6)
    index = np.clip((depth - vmin) / (vmax - vmin), 0, 1)
    return TURBO[(index * 255).astype(np.uint8)].copy()


def write_ply(path, depth, rgb, eye, rotation, fx, fy, cx, cy, max_depth=150.0, stride=1):
    """Back-project a depth map (camera z, mm) into a coloured world PLY.

    rotation columns: forward = R[:,0], up = R[:,1], right = R[:,2].
    Image axes are (-right, -up), matching the renderer's projection.
    """
    depth = np.asarray(depth, dtype=np.float32)
    h, w = depth.shape
    if rgb.shape[:2] != (h, w):
        ys = np.linspace(0, rgb.shape[0] - 1, h).astype(int)
        xs = np.linspace(0, rgb.shape[1] - 1, w).astype(int)
        rgb = rgb[ys][:, xs]
    u, v = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))
    forward, up, right = rotation[:, 0], rotation[:, 1], rotation[:, 2]
    direction = (forward[None, None, :]
                 - right[None, None, :] * ((v - cy) / fy)[..., None]
                 - up[None, None, :] * ((u - cx) / fx)[..., None])
    keep = np.isfinite(depth) & (depth > 0.5) & (depth < max_depth)
    if stride > 1:
        sparse = np.zeros_like(keep)
        sparse[::stride, ::stride] = True
        keep &= sparse
    points = eye[None, None, :] + depth[..., None] * direction
    points = points[keep]
    colours = rgb[keep]
    dtype = np.dtype([("x", "<f4"), ("y", "<f4"), ("z", "<f4"),
                      ("red", "u1"), ("green", "u1"), ("blue", "u1")])
    data = np.empty(len(points), dtype=dtype)
    data["x"], data["y"], data["z"] = points.T
    data["red"], data["green"], data["blue"] = colours.T
    header = (f"ply\nformat binary_little_endian 1.0\nelement vertex {len(data)}\n"
              "property float x\nproperty float y\nproperty float z\n"
              "property uchar red\nproperty uchar green\nproperty uchar blue\nend_header\n")
    with open(path, "wb") as fp:
        fp.write(header.encode("ascii"))
        data.tofile(fp)
    return len(data)
