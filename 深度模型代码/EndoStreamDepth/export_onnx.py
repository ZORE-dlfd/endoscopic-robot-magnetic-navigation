"""Export the fine-tuned EndoStreamDepth frame model to ONNX (fixed 518x518).

    python export_onnx.py --checkpoint /mnt/data/esd/B_finetune_final/best.pth \
        --out /mnt/data/esd/sim_depth.onnx
"""
import argparse

import numpy as np
import torch
import torch.nn.functional as F

from endostreamdepth.model import EndoStreamDepth
from endostreamdepth.dinov2_layers import attention as attn


def sdpa_forward(self, x, attn_bias=None):
    B, N, C = x.shape
    qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
    q, k, v = qkv[0], qkv[1], qkv[2]
    out = F.scaled_dot_product_attention(q, k, v)
    out = out.transpose(1, 2).reshape(B, N, C)
    return self.proj_drop(self.proj(out))


attn.MemEffAttention.forward = sdpa_forward


class ExportModel(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x):
        feats = self.model.get_dpt_features(x)
        ph, pw = x.shape[-2] // 14, x.shape[-1] // 14
        return self.model.final_head(feats[0], ph, pw).unsqueeze(1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--size", type=int, default=518)
    args = ap.parse_args()

    model = EndoStreamDepth(
        batch_size=1, hybrid_configs=None, training=False,
        vit_size="vitl", patch_size=14, attn_class="MemEffAttention",
        use_mamba=False, mamba_type="add", num_mamba_layers=4,
        downsample_mamba=[0.1] * 4, mamba_pos_embed=None,
        mamba_in_dpt_layer=[0, 1, 2, 3], mamba_d_conv=4, mamba_d_state=256,
        use_hydra=False, use_transformer_rnn=False, use_xlstm=False,
    ).cpu().eval()
    state = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    state = state["model"] if "model" in state else state
    model.load_state_dict(state, strict=False)
    wrapper = ExportModel(model).eval()

    x = torch.randn(1, 3, args.size, args.size)
    with torch.no_grad():
        reference = wrapper(x)
    torch.onnx.export(
        wrapper, (x,), args.out,
        input_names=["image"], output_names=["depth"],
        opset_version=17, do_constant_folding=True,
        dynamic_axes=None,
    )
    print("saved", args.out, "reference", float(reference.min()), float(reference.max()))


if __name__ == "__main__":
    main()
