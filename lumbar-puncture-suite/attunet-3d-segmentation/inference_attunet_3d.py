#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""3D AttUNet 推理（公开版：仅核心前向与 NIfTI 导出，不含多模型与审计）。"""
from __future__ import annotations

import argparse
import gc
import os
import sys

import nibabel as nib
import numpy as np
import torch
from glob import glob
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from utils import load_config

REVERSE_MAPPING_3D = {
    0: 0, 1: 1, 2: 2, 3: 3, 4: 4, 5: 5, 6: 6,
    7: 10, 8: 11, 9: 12, 10: 13, 11: 14, 12: 15, 13: 16, 14: 17, 15: 18, 16: 19,
}


def apply_reverse_mapping(pred_mapped: np.ndarray) -> np.ndarray:
    pred = np.asarray(pred_mapped, dtype=np.int64)
    out = np.zeros_like(pred)
    for src, dst in REVERSE_MAPPING_3D.items():
        out[pred == src] = dst
    return out


def load_model(config_path: str, checkpoint: str, device: torch.device):
    from models.attunet_3d import AttU_Net3d

    cfg = load_config(config_path)
    params = cfg["model"]["params"]
    net = AttU_Net3d(img_ch=params["img_ch"], output_ch=params["output_ch"])
    state = torch.load(checkpoint, map_location="cpu")
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    net.load_state_dict(state, strict=False)
    net.to(device).eval()
    return net, tuple(cfg["dataset"]["input_size"])


def predict_volume(model, image: np.ndarray, input_size, device) -> np.ndarray:
    """滑动窗口推理占位实现：按配置 input_size 裁块、投票融合（简化版）。"""
    d, h, w = image.shape
    pd, ph, pw = input_size
    logits_acc = np.zeros((17, d, h, w), dtype=np.float32)
    counts = np.zeros((d, h, w), dtype=np.float32)
    stride = (max(1, pd // 2), max(1, ph // 2), max(1, pw // 2))
    with torch.no_grad():
        for z in range(0, max(1, d - pd + 1), stride[0]):
            for y in range(0, max(1, h - ph + 1), stride[1]):
                for x in range(0, max(1, w - pw + 1), stride[2]):
                    patch = image[z : z + pd, y : y + ph, x : x + pw].astype(np.float32)
                    pad = (
                        (0, max(0, pd - patch.shape[0])),
                        (0, max(0, ph - patch.shape[1])),
                        (0, max(0, pw - patch.shape[2])),
                    )
                    if any(p[1] for p in pad):
                        patch = np.pad(patch, pad, mode="constant")
                    t = torch.from_numpy(patch).unsqueeze(0).unsqueeze(0).to(device)
                    out = model(t).softmax(1).cpu().numpy()[0]
                    ze, ye, xe = z + pd, y + ph, x + pw
                    logits_acc[:, z:ze, y:ye, x:xe] += out[:, : ze - z, : ye - y, : xe - x]
                    counts[z:ze, y:ye, x:xe] += 1.0
    counts = np.maximum(counts, 1.0)
    pred = np.argmax(logits_acc / counts[np.newaxis, ...], axis=0).astype(np.int64)
    return apply_reverse_mapping(pred)


def main() -> None:
    p = argparse.ArgumentParser(description="3D AttUNet 推理（公开版）")
    p.add_argument("--config", type=str, default="configs/custom_3D/attunet_3d.example.yaml")
    p.add_argument("--checkpoint", type=str, required=True, help="训练得到的权重 .pt")
    p.add_argument("--input", type=str, required=True, help="单文件或图像目录")
    p.add_argument("--output", type=str, required=True, help="输出目录")
    p.add_argument("--gpu", type=int, default=0)
    args = p.parse_args()

    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() and args.gpu >= 0 else "cpu")
    model, input_size = load_model(args.config, args.checkpoint, device)
    os.makedirs(args.output, exist_ok=True)

    paths = []
    if os.path.isfile(args.input):
        paths = [args.input]
    else:
        paths = sorted(glob(os.path.join(args.input, "*.nii.gz")) + glob(os.path.join(args.input, "*.nii")))

    for fp in tqdm(paths, desc="infer"):
        nii = nib.load(fp)
        vol = nii.get_fdata().astype(np.float32)
        pred = predict_volume(model, vol, input_size, device)
        out_path = os.path.join(args.output, os.path.basename(fp))
        nib.save(nib.Nifti1Image(pred.astype(np.float32), nii.affine, nii.header), out_path)
        gc.collect()

    print(f"完成，输出目录: {args.output}")


if __name__ == "__main__":
    main()
