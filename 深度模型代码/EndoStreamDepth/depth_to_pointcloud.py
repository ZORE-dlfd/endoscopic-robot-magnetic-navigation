"""Unproject EndoStreamDepth frame_*.npy depth maps into colored PLY point clouds.

Depth from this model is metric (millimeters, C3VD training convention),
so with pixel intrinsics the clouds come out in mm.

    python depth_to_pointcloud.py \
        --rgb-dir  /path/to/frames \
        --depth-dir outputs/<video>/depth_npy_files \
        --out-dir  pointclouds \
        --fx 500 --fy 500 --cx 480 --cy 270 --stride 10

Without --fx/--fy a rough f = 0.8 * width is used; that is NOT calibrated and
gives a visually plausible but geometrically wrong cloud.
"""
import argparse
from pathlib import Path

import cv2
import numpy as np

PLY_DTYPE = np.dtype([("x", "<f4"), ("y", "<f4"), ("z", "<f4"),
                      ("red", "u1"), ("green", "u1"), ("blue", "u1")])


def unproject(depth, rgb, fx, fy, cx, cy, min_depth, max_depth):
    h, w = depth.shape
    if rgb.shape[:2] != (h, w):
        rgb = cv2.resize(rgb, (w, h), interpolation=cv2.INTER_AREA)
    u, v = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))
    z = depth.astype(np.float32)
    keep = np.isfinite(z) & (z > min_depth) & (z < max_depth)
    pts = np.empty(int(keep.sum()), dtype=PLY_DTYPE)
    pts["x"] = (u[keep] - cx) * z[keep] / fx
    pts["y"] = (v[keep] - cy) * z[keep] / fy
    pts["z"] = z[keep]
    pts["red"], pts["green"], pts["blue"] = rgb[keep, 0], rgb[keep, 1], rgb[keep, 2]
    return pts


def write_ply(path, pts):
    header = (f"ply\nformat binary_little_endian 1.0\nelement vertex {len(pts)}\n"
              "property float x\nproperty float y\nproperty float z\n"
              "property uchar red\nproperty uchar green\nproperty uchar blue\n"
              "end_header\n")
    with open(path, "wb") as f:
        f.write(header.encode("ascii"))
        pts.tofile(f)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rgb-dir", required=True)
    ap.add_argument("--depth-dir", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--fx", type=float)
    ap.add_argument("--fy", type=float)
    ap.add_argument("--cx", type=float)
    ap.add_argument("--cy", type=float)
    ap.add_argument("--min-depth", type=float, default=0.1)
    ap.add_argument("--max-depth", type=float, default=np.inf)
    ap.add_argument("--stride", type=int, default=1, help="keep every Nth frame")
    ap.add_argument("--rgb-offset", type=int, default=0,
                    help="global index of this clip's first frame in --rgb-dir")
    ap.add_argument("--orig-size", help="WxH the intrinsics were calibrated at, "
                    "e.g. 1920x1080; they get scaled to the depth-map size")
    args = ap.parse_args()

    rgb_paths = sorted(p for p in Path(args.rgb_dir).iterdir()
                       if p.suffix.lower() in {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
                       and "depth" not in p.stem.lower())
    depth_paths = sorted(Path(args.depth_dir).glob("frame_*.npy"),
                         key=lambda p: int(p.stem.split("_")[1]))
    n = min(len(rgb_paths) - args.rgb_offset, len(depth_paths))
    print(f"{len(rgb_paths)} rgb frames, {len(depth_paths)} depth maps, offset {args.rgb_offset}; pairing {n}")
    if n < len(depth_paths):
        print("WARNING: fewer rgb frames than depth maps, tail frames skipped")
    if n == 0:
        raise SystemExit("no paired frames found")

    z0 = np.load(depth_paths[0])
    h, w = z0.squeeze().shape
    fx = args.fx or 0.8 * w
    fy = args.fy or fx
    cx = args.cx if args.cx is not None else (w - 1) / 2
    cy = args.cy if args.cy is not None else (h - 1) / 2
    if args.orig_size:
        ow, oh = (int(v) for v in args.orig_size.lower().split("x"))
        fx, fy, cx, cy = fx * w / ow, fy * h / oh, cx * w / ow, cy * h / oh
        print(f"intrinsics scaled from {ow}x{oh} to {w}x{h}")
    if args.fx is None:
        print("WARNING: --fx/--fy not given, using f=0.8*width (uncalibrated)")

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    for i in range(0, n, args.stride):
        bgr = cv2.imread(str(rgb_paths[args.rgb_offset + i]), cv2.IMREAD_COLOR)
        depth = np.load(depth_paths[i]).squeeze()
        if bgr is None or depth.ndim != 2:
            print(f"skip frame {i}: bad rgb or depth shape {None if bgr is None else depth.shape}")
            continue
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        pts = unproject(depth, rgb, fx, fy, cx, cy, args.min_depth, args.max_depth)
        idx = args.rgb_offset + i
        write_ply(out / f"frame_{idx:06d}.ply", pts)
        print(f"frame {idx}: {len(pts)} points -> {out / f'frame_{idx:06d}.ply'}")


if __name__ == "__main__":
    # self-check: a 1000 mm plane at f=500, principal point at center
    d = np.full((4, 4), 1000.0, np.float32)
    img = np.zeros((4, 4, 3), np.uint8)
    p = unproject(d, img, 500, 500, 1.5, 1.5, 0.1, np.inf)
    assert len(p) == 16
    assert abs(p["x"][0] - (0 - 1.5) * 2.0) < 1e-3 and abs(p["y"][0] - (0 - 1.5) * 2.0) < 1e-3
    assert abs(p["z"]).max() == 1000.0
    main()
