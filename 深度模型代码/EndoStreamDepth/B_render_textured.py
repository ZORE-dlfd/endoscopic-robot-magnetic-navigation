"""Re-render the B simulation with a procedural mucosa texture.

Reproduces the reported shader (headlamp, base colour, gamma) and adds a
world-space 3D value-noise texture, so we can test whether texture restores
monocular depth accuracy without waiting for the original software.

    python B_render_textured.py --frames 300 --size 518 \
        --out-dir /mnt/data/esd/B_textured
"""
import argparse
from pathlib import Path

import cv2
import numpy as np
import trimesh
from trimesh.ray.ray_pyembree import RayMeshIntersector

from B_render_gt_depth import MESH, POSE, load_poses, ray_dirs

BASE = np.array([0.780, 0.498, 0.475], np.float64) ** 2.2
VEIN = np.array([0.55, 0.22, 0.20], np.float64) ** 2.2


def _hash(p):
    return np.modf(np.sin(p @ np.array([12.9898, 78.233, 37.719])) * 43758.5453)[0] % 1.0


def vnoise(p):
    i = np.floor(p)
    f = p - i
    f = f * f * (3.0 - 2.0 * f)
    c = {}
    for dx in (0, 1):
        for dy in (0, 1):
            for dz in (0, 1):
                c[dx, dy, dz] = _hash(i + np.array([dx, dy, dz]))
    x00 = c[0, 0, 0] * (1 - f[:, 0]) + c[1, 0, 0] * f[:, 0]
    x10 = c[0, 1, 0] * (1 - f[:, 0]) + c[1, 1, 0] * f[:, 0]
    x01 = c[0, 0, 1] * (1 - f[:, 0]) + c[1, 0, 1] * f[:, 0]
    x11 = c[0, 1, 1] * (1 - f[:, 0]) + c[1, 1, 1] * f[:, 0]
    y0 = x00 * (1 - f[:, 1]) + x10 * f[:, 1]
    y1 = x01 * (1 - f[:, 1]) + x11 * f[:, 1]
    return y0 * (1 - f[:, 2]) + y1 * f[:, 2]


def fbm(p, octaves=4):
    out = np.zeros(len(p))
    amp, tot = 0.5, 0.0
    for _ in range(octaves):
        out += amp * vnoise(p)
        p = p * 2.03
        tot += amp
        amp *= 0.5
    return out / tot


def shade(points, normals, eye, textured=True):
    view = eye[None, :] - points
    dist = np.linalg.norm(view, axis=1)
    view /= dist[:, None]
    diffuse = np.abs((normals * view).sum(1))
    lamp = 35.0 / np.maximum(dist, 1.0) * np.clip((1 - (dist / 160.0) ** 4) ** 2, 0, 1)

    if textured:
        n1 = fbm(points * 0.12)
        n2 = fbm(points * 0.55 + 11.0)
        vein = (1.0 - np.abs(2.0 * fbm(points * 0.18 + 7.0) - 1.0)) ** 6
        albedo = BASE[None, :] * (0.50 + 0.80 * n1 + 0.20 * n2)[:, None]
        albedo = albedo * (1 - 0.45 * vein[:, None]) + VEIN[None, :] * (0.45 * vein[:, None])
    else:
        albedo = np.tile(BASE, (len(points), 1))
    light = np.array([1.0, 0.86, 0.77]) * (lamp * diffuse)[:, None]
    spec = np.array([0.076, 0.024, 0.020]) * (lamp * diffuse**18)[:, None]
    rgb = np.clip((albedo * light + spec) / np.pi, 0, None) ** (1 / 2.2)
    return np.clip(rgb * 255, 0, 255).astype(np.uint8)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames", type=int, default=300)
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--size", type=int, default=518)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--flat", action="store_true", help="use face normals (faceted shading)")
    ap.add_argument("--no-texture", action="store_true", help="constant albedo, shading only")
    args = ap.parse_args()

    poses = load_poses(POSE)
    mesh = trimesh.load(MESH, process=True)  # merge vertices -> smooth normals
    vnorm = mesh.vertex_normals
    inter = RayMeshIntersector(mesh)
    W = H = args.size
    u, v, x, y = ray_dirs(W, H)
    out = Path(args.out_dir) / "帧"
    out.mkdir(parents=True, exist_ok=True)

    for i in range(args.start, min(args.start + args.frames, len(poses))):
        po = poses[i]
        f = po["f"] / np.linalg.norm(po["f"])
        uu = po["u"] / np.linalg.norm(po["u"])
        rr = po["rr"] / np.linalg.norm(po["rr"])
        d = rr[None, None, :] * x[..., None] + uu[None, None, :] * y[..., None] + f[None, None, :]
        d = (d / np.linalg.norm(d, axis=-1, keepdims=True)).reshape(-1, 3)
        o = np.tile(po["p"], (len(d), 1))
        loc, idx_ray, idx_tri = inter.intersects_location(o, d, multiple_hits=False)
        img = np.zeros((len(d), 3), np.uint8)
        if len(loc):
            if args.flat:
                n = mesh.face_normals[idx_tri]
            else:
                tri = mesh.triangles[idx_tri]
                bary = trimesh.triangles.points_to_barycentric(tri, loc)
                n = (vnorm[mesh.faces[idx_tri]] * bary[:, :, None]).sum(1)
                n /= np.linalg.norm(n, axis=1, keepdims=True)
            img[idx_ray] = shade(loc, n, po["p"], textured=not args.no_texture)
        cv2.imwrite(str(out / f"{i:06d}.jpg"), img.reshape(H, W, 3)[:, :, ::-1], [cv2.IMWRITE_JPEG_QUALITY, 95])
        if i % 50 == 0:
            print(f"frame {i}")


if __name__ == "__main__":
    main()
