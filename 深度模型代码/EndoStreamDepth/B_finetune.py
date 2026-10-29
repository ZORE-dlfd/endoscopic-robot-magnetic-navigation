"""Fine-tune EndoStreamDepth on the B simulation render (ViT frozen, DPT head trained).

    python B_finetune.py --frames 300 --epochs 30 --batch 2 --lr 1e-4

Reads  : /mnt/data/esd/B相机导出/帧/NNNNNN.jpg  (960x720)
         /mnt/data/esd/B_gt/depth/frame_N.npy   (518x518 mm, from B_render_gt_depth.py)
Writes : /mnt/data/esd/B_finetune/best.pth, last.pth, log.csv
"""
import argparse
import csv
import time
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from endostreamdepth.model import EndoStreamDepth
from endostreamdepth.util.loss import GradientEdgeLoss, SiLogLoss

CKPT = "/mnt/data/C3VD/weights/iter_20001.pth"
FRAMES = "/mnt/data/esd/B相机导出/帧"
GT = "/mnt/data/esd/B_gt/depth"
MEAN = np.array([0.485, 0.456, 0.406], np.float32)
STD = np.array([0.229, 0.224, 0.225], np.float32)


def load_pair(i, size=518, flip=False):
    # cv2.imread fails on non-ASCII paths (e.g. B内镜纹理); decode bytes instead
    bgr = cv2.imdecode(np.fromfile(f"{FRAMES}/{i:06d}.jpg", dtype=np.uint8), cv2.IMREAD_COLOR)
    if bgr is None:
        raise FileNotFoundError(f"cannot read {FRAMES}/{i:06d}.jpg")
    rgb = cv2.resize(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB), (size, size), interpolation=cv2.INTER_AREA)
    rgb = (rgb.astype(np.float32) / 255.0 - MEAN) / STD
    gt = np.load(f"{GT}/frame_{i}.npy").astype(np.float32)
    if rgb.shape[:2] != gt.shape:
        gt = cv2.resize(gt, (size, size), interpolation=cv2.INTER_NEAREST)
    if flip:
        rgb, gt = rgb[:, ::-1], gt[:, ::-1]
    img = torch.from_numpy(rgb.transpose(2, 0, 1).copy())
    gt = torch.from_numpy(gt.copy())
    mask = (gt > 0.5) & (gt < 150)
    return img, gt, mask


def batch_to_tensors(idx, flip):
    data = [load_pair(i, flip=flip) for i in idx]
    return (torch.stack([d[0] for d in data]).cuda(),
            torch.stack([d[1] for d in data]).cuda(),
            torch.stack([d[2] for d in data]).cuda())


def evaluate(model, idx):
    model.eval()
    abs_rel, rmse, d1 = [], [], []
    with torch.no_grad():
        for i in idx:
            img, gt, mask = load_pair(i)
            img = img[None].cuda()
            feats = model.get_dpt_features(img)
            ph, pw = img.shape[-2] // model.patch_size, img.shape[-1] // model.patch_size
            pred = model.final_head(feats[0], ph, pw)[0]
            m = mask.cuda()
            p, g = pred[m], gt.cuda()[m]
            abs_rel.append(((p - g).abs() / g).mean().item())
            rmse.append(torch.sqrt(((p - g) ** 2).mean()).item())
            d1.append((torch.maximum(p / g, g / p) < 1.25).float().mean().item())
    return float(np.mean(abs_rel)), float(np.mean(rmse)), float(np.mean(d1))


def main():
    global FRAMES, GT
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames", type=int, default=300)
    ap.add_argument("--train-end", type=int, default=240)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch", type=int, default=2)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--out", default="/mnt/data/esd/B_finetune")
    ap.add_argument("--frames-dir", default=FRAMES)
    ap.add_argument("--gt-dir", default=GT)
    args = ap.parse_args()
    FRAMES = args.frames_dir
    GT = args.gt_dir

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(0)
    np.random.seed(0)

    model = EndoStreamDepth(
        batch_size=args.batch, hybrid_configs=None, training=True,
        vit_size="vitl", patch_size=14, attn_class="MemEffAttention",
        use_mamba=False, mamba_type="add", num_mamba_layers=4,
        downsample_mamba=[0.1] * 4, mamba_pos_embed=None,
        mamba_in_dpt_layer=[0, 1, 2, 3], mamba_d_conv=4, mamba_d_state=256,
        use_hydra=False, use_transformer_rnn=False, use_xlstm=False,
    ).cuda()
    sd = torch.load(CKPT, map_location="cpu", weights_only=False)
    sd = sd["model"] if "model" in sd else sd
    model.load_state_dict(sd, strict=False)
    for p in model.parameters():
        p.requires_grad_(False)
    for p in model.depth_head.parameters():
        p.requires_grad_(True)
    n_train = sum(p.numel() for p in model.depth_head.parameters() if p.requires_grad)
    print(f"trainable DPT params: {n_train/1e6:.2f} M")

    opt = torch.optim.AdamW(model.depth_head.parameters(), lr=args.lr, weight_decay=0.01)
    silog = SiLogLoss()
    grad_loss = GradientEdgeLoss()
    train_idx = list(range(args.train_end))
    val_idx = list(range(args.train_end, args.frames))
    print(f"train {len(train_idx)} frames, val {len(val_idx)} frames")

    best = 1e9
    log = open(out / "log.csv", "w", newline="")
    writer = csv.writer(log)
    writer.writerow(["step", "epoch", "loss", "abs_rel", "rmse", "d1", "sec"])
    step = 0
    t0 = time.time()
    for epoch in range(1, args.epochs + 1):
        model.train()
        perm = np.random.permutation(train_idx)
        for b in range(0, len(perm), args.batch):
            idx = perm[b:b + args.batch]
            flip = np.random.rand() < 0.5
            img, gt, mask = batch_to_tensors(idx, flip)
            feats = model.get_dpt_features(img)
            ph, pw = img.shape[-2] // model.patch_size, img.shape[-1] // model.patch_size
            pred = model.final_head(feats[0], ph, pw)
            pred = pred * mask
            valid = mask.bool()
            loss = (silog(pred, gt, valid & (gt > 0.01) & (gt < 100))
                    + F.l1_loss(torch.log(pred[valid] + 1e-6), torch.log(gt[valid] + 1e-6))
                    + grad_loss(pred, gt, valid)
                    + 0.01 * F.l1_loss(pred[valid], gt[valid]))
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.depth_head.parameters(), 1.0)
            opt.step()
            step += 1
            if step % 20 == 0:
                print(f"epoch {epoch} step {step} loss {loss.item():.4f} ({time.time()-t0:.0f}s)")
        if epoch % 2 == 0 or epoch == args.epochs:
            ar, rm, d1 = evaluate(model, val_idx)
            print(f"  VAL epoch {epoch}: AbsRel {ar:.4f} RMSE {rm:.2f} mm d1 {d1:.4f}")
            writer.writerow([step, epoch, loss.item(), ar, rm, d1, round(time.time() - t0, 1)])
            log.flush()
            torch.save({"model": model.state_dict()}, out / "last.pth")
            if ar < best:
                best = ar
                torch.save({"model": model.state_dict()}, out / "best.pth")
    log.close()
    print("done, best AbsRel", best)


if __name__ == "__main__":
    main()
