"""Compare EndoStreamDepth predictions with ray-cast GT depth from the B export."""
import argparse
import csv
from pathlib import Path

import numpy as np


def metrics(pred, gt, mask):
    p, g = pred[mask], gt[mask]
    err = p - g
    ratio = np.maximum(p / g, g / p)
    return dict(
        abs_rel=float(np.mean(np.abs(err) / g)),
        sq_rel=float(np.mean(err**2 / g)),
        rmse=float(np.sqrt(np.mean(err**2))),
        mae=float(np.mean(np.abs(err))),
        d1=float(np.mean(ratio < 1.25)),
        d2=float(np.mean(ratio < 1.25**2)),
        d3=float(np.mean(ratio < 1.25**3)),
        med_ratio=float(np.median(p / g)),
        corr=float(np.corrcoef(p, g)[0, 1]),
        valid=int(mask.sum()),
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred-dir", required=True)
    ap.add_argument("--gt-dir", required=True)
    ap.add_argument("--frames", type=int, default=300)
    ap.add_argument("--out-csv", required=True)
    args = ap.parse_args()

    rows = []
    for i in range(args.frames):
        pred = np.load(Path(args.pred_dir) / f"frame_{i}.npy").squeeze().astype(np.float32)
        gt = np.load(Path(args.gt_dir) / f"frame_{i}.npy").astype(np.float32)
        if pred.shape != gt.shape:
            raise SystemExit(f"shape mismatch frame {i}: {pred.shape} vs {gt.shape}")
        mask = (gt > 0.5) & (gt < 150) & np.isfinite(pred) & (pred > 0)
        m = metrics(pred, gt, mask)
        m["frame"] = i
        rows.append(m)

    keys = [k for k in rows[0] if k != "frame"]
    print(f"frames {len(rows)}")
    for k in keys:
        if k == "valid":
            continue
        vals = np.array([r[k] for r in rows])
        print(f"{k:10s} mean {vals.mean():.4f}  median {np.median(vals):.4f}")
    with open(args.out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["frame"] + keys)
        w.writeheader()
        w.writerows(rows)
    print("wrote", args.out_csv)


if __name__ == "__main__":
    main()
