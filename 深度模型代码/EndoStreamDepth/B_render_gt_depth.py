"""Ray-cast ground-truth depth for the B simulation export.

Camera: pinhole, 960x720, fx=fy=623.538, cx=480, cy=360, no distortion.
Poses: camera_pose.csv columns cam_x/y/z_mm + forward/up/right axes (world, mm).
Mesh: colon_inner_wall.stl in the same world frame.

The image axes are (-right, -up): this was verified against the renderer's
headlamp shading model (corr 0.907 vs <=0.2 for the other orientations).

    python B_render_gt_depth.py --frames 300 --out-dir /mnt/data/esd/B_gt --size 518
"""
import argparse
import csv
from pathlib import Path

import numpy as np
import trimesh
from trimesh.ray.ray_pyembree import RayMeshIntersector

POSE = "/mnt/data/esd/B相机导出/数据/camera_pose.csv"
MESH = "/mnt/data/esd/B相机导出/模型/B世界坐标/colon_inner_wall.stl"
FX = FY = 623.538
CX, CY = 480.0, 360.0
W0, H0 = 960, 720


def load_poses(path):
    out = []
    for r in csv.DictReader(open(path, encoding="utf-8-sig")):
        out.append(dict(
            p=np.array([float(r["cam_x_mm"]), float(r["cam_y_mm"]), float(r["cam_z_mm"])]),
            f=np.array([float(r["fx"]), float(r["fy"]), float(r["fz"])]),
            u=np.array([float(r["ux"]), float(r["uy"]), float(r["uz"])]),
            rr=np.array([float(r["rx"]), float(r["ry"]), float(r["rz"])]),
        ))
    return out


def ray_dirs(width, height):
    fx, fy = FX * width / W0, FY * height / H0
    cx, cy = CX * width / W0, CY * height / H0
    u, v = np.meshgrid(np.arange(width, dtype=np.float64), np.arange(height, dtype=np.float64))
    # verified against the renderer at frame 100 (shading corr 0.90):
    #   dir = -right*(v-cy)/fy - up*(u-cx)/fx + forward
    return u, v, -(v - cy) / fy, -(u - cx) / fx


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pose", default=POSE)
    ap.add_argument("--mesh", default=MESH)
    ap.add_argument("--frames", type=int, default=300)
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--size", type=int, default=518)
    ap.add_argument("--out-dir", required=True)
    args = ap.parse_args()

    poses = load_poses(args.pose)
    mesh = trimesh.load(args.mesh, process=False)
    inter = RayMeshIntersector(mesh)
    W = H = args.size
    u, v, x, y = ray_dirs(W, H)
    out = Path(args.out_dir) / "depth"
    out.mkdir(parents=True, exist_ok=True)

    for i in range(args.start, min(args.start + args.frames, len(poses))):
        po = poses[i]
        f = po["f"] / np.linalg.norm(po["f"])
        uu = po["u"] / np.linalg.norm(po["u"])
        rr = po["rr"] / np.linalg.norm(po["rr"])
        dirs = rr[None, None, :] * x[..., None] + uu[None, None, :] * y[..., None] + f[None, None, :]
        dirs = (dirs / np.linalg.norm(dirs, axis=-1, keepdims=True)).reshape(-1, 3)
        origins = np.tile(po["p"], (len(dirs), 1))
        loc, idx_ray, _ = inter.intersects_location(origins, dirs, multiple_hits=False)
        depth = np.zeros(len(dirs), np.float32)
        if len(loc):
            t = np.linalg.norm(loc - origins[idx_ray], axis=1)
            depth[idx_ray] = t * (dirs[idx_ray] @ f)
        depth = depth.reshape(H, W)
        np.save(out / f"frame_{i}.npy", depth)
        if i % 50 == 0:
            m = depth > 0
            print(f"frame {i}: hit {m.mean()*100:.1f}%, depth p50 {np.median(depth[m]):.2f} mm")


if __name__ == "__main__":
    main()
