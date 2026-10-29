"""Turn EndoStreamDepth 518x518 depth maps of a C3VD scene into world-frame PLYs.

Uses the C3VD omnidirectional (Scaramuzza) camera model and pose.txt
(camera-to-world, row-vector convention), both as used by the official
C3VD exampleDataLoader.

    python c3vd_depth_to_world_pc.py \
        --depth-dir /mnt/data/esd/output/trans_t1_a/depth_npy_files \
        --rgb-dir   /mnt/data/C3VD/cv3d/trans_t1_a \
        --pose      /mnt/data/C3VD/cv3d/trans_t1_a/pose.txt \
        --calib     /mnt/data/C3VD/calib/cfhq190l_omnidirectional_params/cfhq190l_omnidirectional_values.txt \
        --out-dir   /mnt/data/esd/world_pc/trans_t1_a \
        --mesh      /mnt/data/C3VD/cv3d/trans_t1_a/mesh_seen.ply \
        --stride 5
"""
import argparse
from pathlib import Path

import cv2
import numpy as np
from scipy.spatial import cKDTree

from depth_to_pointcloud import PLY_DTYPE, write_ply


def load_calib(path):
    w, h, cx, cy, a0, a2, a3, a4, c, d, e = map(float, Path(path).read_text().split())
    return dict(width=int(w), height=int(h), cx=cx, cy=cy,
                a0=a0, a2=a2, a3=a3, a4=a4, c=c, d=d, e=e)


def ray_map(k, out_w, out_h):
    x, y = np.meshgrid(np.arange(out_w, dtype=np.float64), np.arange(out_h, dtype=np.float64))
    # output pixel centres -> original C3VD pixel coordinates
    u = (x + 0.5) * k["width"] / out_w - 0.5
    v = (y + 0.5) * k["height"] / out_h - 0.5
    # C3VD exampleDataLoader uses (cx - 1, cy - 1)
    uvp = np.stack([u - (k["cx"] - 1), v - (k["cy"] - 1)], axis=-1)
    inv = np.linalg.inv(np.array([[k["c"], k["d"]], [k["e"], 1.0]]))
    uvpp = np.einsum("ij,...j->...i", inv, uvp)
    rho = np.sqrt(uvpp[..., 0] ** 2 + uvpp[..., 1] ** 2)
    z = k["a0"] + k["a2"] * rho**2 + k["a3"] * rho**3 + k["a4"] * rho**4
    rays = np.stack([uvpp[..., 0], uvpp[..., 1], z], axis=-1)
    rays /= np.linalg.norm(rays, axis=-1, keepdims=True)
    return rays


def load_poses(path):
    return [np.fromstring(line, sep=",").reshape(4, 4) for line in Path(path).read_text().splitlines() if line.strip()]


def read_ply_vertices(path):
    """Vertices of a binary_little_endian PLY with float x/y/z properties."""
    with open(path, "rb") as f:
        n_vert = None
        n_props = 0
        while True:
            line = f.readline().decode("ascii", "ignore").strip()
            if line.startswith("element vertex"):
                n_vert = int(line.split()[-1])
            elif line.startswith("property float"):
                n_props += 1
            elif line == "end_header":
                break
        data = np.fromfile(f, dtype="<f4", count=n_vert * n_props)
    return data.reshape(n_vert, n_props)[:, :3]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--depth-dir", required=True)
    ap.add_argument("--rgb-dir", required=True)
    ap.add_argument("--pose", required=True)
    ap.add_argument("--calib", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--mesh", help="optional mesh PLY for a distance sanity check")
    ap.add_argument("--max-depth", type=float, default=99.9)
    ap.add_argument("--stride", type=int, default=1)
    args = ap.parse_args()

    depth_paths = sorted(Path(args.depth_dir).glob("frame_*.npy"),
                         key=lambda p: int(p.stem.split("_")[1]))
    poses = load_poses(args.pose)
    z0 = np.load(depth_paths[0]).squeeze()
    H, W = z0.shape
    rays = ray_map(load_calib(args.calib), W, H)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    mesh_tree = cKDTree(read_ply_vertices(args.mesh)) if args.mesh else None

    for i in range(0, min(len(depth_paths), len(poses)), args.stride):
        depth = np.load(depth_paths[i]).squeeze().astype(np.float32)
        rgb = cv2.cvtColor(cv2.imread(str(Path(args.rgb_dir) / f"{i:04d}_color.png")), cv2.COLOR_BGR2RGB)
        rgb = cv2.resize(rgb, (W, H), interpolation=cv2.INTER_AREA)
        valid = (np.isfinite(depth) & (depth > 0) & (depth < args.max_depth)
                 & (rays[..., 2] > 1e-3))
        d = depth[valid]
        r = rays[valid]
        pts = np.stack([d * r[:, 0] / r[:, 2], d * r[:, 1] / r[:, 2], d], axis=-1)
        world = np.hstack([pts, np.ones((len(pts), 1))]) @ poses[i]
        world = world[:, :3]
        arr = np.empty(len(world), dtype=PLY_DTYPE)
        arr["x"], arr["y"], arr["z"] = world.T
        arr["red"], arr["green"], arr["blue"] = rgb[valid, 0], rgb[valid, 1], rgb[valid, 2]
        path = out / f"frame_{i:06d}_world.ply"
        write_ply(path, arr)
        msg = f"frame {i}: {len(arr)} pts -> {path}"
        if mesh_tree is not None:
            dist, _ = mesh_tree.query(world[::10], workers=-1)
            msg += f" | median mesh dist {np.median(dist):.2f} mm, p90 {np.percentile(dist, 90):.2f} mm"
        print(msg)


if __name__ == "__main__":
    k = load_calib("/mnt/data/C3VD/calib/cfhq190l_omnidirectional_params/cfhq190l_omnidirectional_values.txt")
    r = ray_map(k, 518, 518)
    print("centre ray", r[259, 259], "corner ray", r[0, 0])
    assert abs(r[259, 259, 2] - 1.0) < 0.05
    main()
