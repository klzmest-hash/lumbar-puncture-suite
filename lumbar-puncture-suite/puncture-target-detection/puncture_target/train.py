#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
训练穿刺靶点偏移预测基线模型。

  python -m puncture_target.train ^
    --config hyperparams.example.yaml ^
    --csv path/to/puncture_endpoints.csv ^
    --ct_root path/to/ct_nifti ^
    --out_dir runs/puncture_baseline

可选：--mask_root 与 CT 同形状的分割掩码目录（第二通道；缺失时该通道填 0）。

推荐：有掩码时使用 --train_center_mode heuristic（或与默认 auto 配合），使裁块中心与 infer 的掩码启发式一致，
左右靶点共享同一块 patch，减少依赖推理后处理修正。

指定 GPU：--gpu 0 使用第 1 块卡；--gpu 1 使用第 2 块；不传则 cuda:0（若可用）；--gpu -1 强制 CPU。
（若已设置 CUDA_VISIBLE_DEVICES，则 --gpu 0 表示可见列表中的第一块。）

原始 CT：必须；掩码：可选，用于强调椎体/椎间盘区域，建议有则加上。

服务器默认：batch_size=8、num_workers=2、persistent_workers 关（与 attunet-fifth-3d 一致）、CUDA 上 AMP；
**cudnn.benchmark 默认关闭**（避免首 batch 长时间卡在 0%）。
需要略快后续 step 可加 --cudnn_benchmark。显存不足请加 --batch_size 4；
首 batch 仍慢可加 --num_workers 0、--prefetch_factor 1、或把数据拷到本地盘；需要可试 --compile_torch（PyTorch 2.x）。

公开版：超参由 --config 指定（见 hyperparams.example.yaml）。
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from contextlib import nullcontext
from typing import Any, Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader


def amp_autocast(device: torch.device, enabled: bool):
    """PyTorch 2.x 用 torch.amp.autocast，避免 cuda.amp 的 FutureWarning。"""
    if not enabled or device.type != "cuda":
        return nullcontext()
    if hasattr(torch, "amp") and hasattr(torch.amp, "autocast"):
        return torch.amp.autocast(device_type="cuda", enabled=True)
    import torch.cuda.amp as cuda_amp

    return cuda_amp.autocast(enabled=True)


def new_grad_scaler() -> Any:
    """优先 torch.amp.GradScaler('cuda')（PyTorch 2.x）。"""
    if hasattr(torch, "amp") and hasattr(torch.amp, "GradScaler"):
        return torch.amp.GradScaler("cuda")
    import torch.cuda.amp as cuda_amp

    return cuda_amp.GradScaler()

try:
    from tqdm import tqdm
except ImportError:
    tqdm = None  # type: ignore[misc, assignment]

from puncture_target.data import (
    PunctureEndpointDataset,
    filter_rows_with_ct,
    load_csv_rows,
    split_by_case,
)
from puncture_target.model import TargetOffsetNet3D
from puncture_target.config_loader import load_hyperparams, default_example_config_path

_PP_ROOT = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
if _PP_ROOT not in sys.path:
    sys.path.insert(0, _PP_ROOT)


def resolve_device(gpu: Optional[int]) -> torch.device:
    """
    gpu is None: 若存在 CUDA 则用 cuda:0，否则 CPU。
    gpu == -1: 强制 CPU。
    gpu >= 0: 使用 cuda:{gpu}。
    """
    if gpu is not None and gpu < 0:
        return torch.device("cpu")
    if not torch.cuda.is_available():
        if gpu is not None and gpu >= 0:
            raise RuntimeError("已指定 --gpu，但当前环境未检测到 CUDA（无法使用 GPU）")
        return torch.device("cpu")
    if gpu is None:
        return torch.device("cuda:0")
    n = torch.cuda.device_count()
    if gpu >= n:
        raise RuntimeError(f"--gpu {gpu} 无效，当前仅有 {n} 块 GPU（编号 0～{n - 1}）")
    return torch.device(f"cuda:{gpu}")


def _dataloader_worker_init(_worker_id: int) -> None:
    """
    每个 DataLoader 子进程内限制 BLAS/OpenMP 线程，避免
    num_workers×多线程 与主进程抢满 CPU（常见为「多进程反而卡、0 反而能动」）。
    """
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")


def dataloader_kwargs(
    num_workers: int,
    pin_memory: bool,
    prefetch_factor: int = 2,
    persistent_workers: bool = False,
) -> Dict[str, Any]:
    """
    与 Awesome-U-Net ``attunet-fifth-3d.py`` 一致：默认 **persistent_workers=False**，
    避免 3D 整卷在 worker 内长期驻留导致内存与 NFS 上「首 batch 极慢」更难排查。
    需要跨 epoch 复用 worker 时可加 ``--persistent_workers``。
    """
    kw: Dict[str, Any] = {
        "num_workers": num_workers,
        "pin_memory": pin_memory,
    }
    if num_workers > 0:
        kw["persistent_workers"] = bool(persistent_workers)
        kw["prefetch_factor"] = max(2, int(prefetch_factor))
        kw["worker_init_fn"] = _dataloader_worker_init
    return kw


def collate_batch(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    return {
        "x": torch.stack([b["x"] for b in batch]),
        "stem_idx": torch.stack([b["stem_idx"] for b in batch]),
        "offset": torch.stack([b["offset"] for b in batch]),
        "case_id": [b["case_id"] for b in batch],
    }


@torch.no_grad()
def eval_epoch(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    criterion: nn.Module,
    desc: str = "val",
    use_amp: bool = False,
    non_blocking: bool = False,
    log_first_batch: bool = False,
) -> Dict[str, float]:
    model.eval()
    total = 0.0
    n = 0
    euclid = []
    it = loader
    if tqdm is not None:
        it = tqdm(loader, desc=desc, leave=False, file=sys.stdout)
    if log_first_batch:
        print(f"[{desc}] 等待首个 batch...", flush=True)
    t0 = time.perf_counter()
    first_batch = True
    for batch in it:
        if first_batch and log_first_batch:
            print(
                f"[{desc}] 首个 batch 等待 {time.perf_counter() - t0:.1f}s",
                flush=True,
            )
            first_batch = False
        x = batch["x"].to(device, non_blocking=non_blocking)
        stem = batch["stem_idx"].to(device, non_blocking=non_blocking)
        off = batch["offset"].to(device, non_blocking=non_blocking)
        with amp_autocast(device, use_amp):
            pred = model(x, stem)
            loss = criterion(pred, off)
        total += loss.item() * x.size(0)
        n += x.size(0)
        err = (pred - off).float().norm(dim=1)
        euclid.extend(err.cpu().numpy().tolist())
    arr = np.array(euclid) if euclid else np.array([])
    return {
        "loss": total / max(n, 1),
        "mean_euclidean_mm": float(np.mean(arr)) if arr.size else 0.0,
        "rmse_euclidean_mm": float(np.sqrt(np.mean(arr**2))) if arr.size else 0.0,
    }


def train_epoch(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    desc: str = "train",
    scaler: Optional[Any] = None,
    use_amp: bool = False,
    non_blocking: bool = False,
    log_first_batch: bool = False,
) -> Dict[str, float]:
    """与 eval_epoch 相同指标：L1、偏移欧氏误差均值与 RMSE（mm）。"""
    model.train()
    total = 0.0
    n = 0
    euclid: List[float] = []
    it = loader
    if tqdm is not None:
        it = tqdm(loader, desc=desc, leave=False, file=sys.stdout)
    if log_first_batch:
        print(
            f"[{desc}] 等待首个 batch（DataLoader 读整卷 NIfTI；慢盘/NFS 可能需数分钟）...",
            flush=True,
        )
    t_wait0 = time.perf_counter()
    first_batch = True
    for batch in it:
        if first_batch:
            if log_first_batch:
                print(
                    f"[{desc}] 首个 batch 已返回，取数等待 {time.perf_counter() - t_wait0:.1f}s，"
                    f"开始 GPU 前向/反传...",
                    flush=True,
                )
            t_step0 = time.perf_counter()
        x = batch["x"].to(device, non_blocking=non_blocking)
        stem = batch["stem_idx"].to(device, non_blocking=non_blocking)
        off = batch["offset"].to(device, non_blocking=non_blocking)
        optimizer.zero_grad(set_to_none=True)
        with amp_autocast(device, use_amp):
            pred = model(x, stem)
            loss = criterion(pred, off)
        err = (pred - off).float().norm(dim=1)
        euclid.extend(err.detach().cpu().numpy().tolist())
        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()
        total += loss.item() * x.size(0)
        n += x.size(0)
        if first_batch:
            if log_first_batch:
                print(
                    f"[{desc}] 首个 batch 训练步耗时 {time.perf_counter() - t_step0:.1f}s "
                    f"（若极长：勿加 --cudnn_benchmark；或 --num_workers 0；或数据放本地盘）",
                    flush=True,
                )
            first_batch = False
    arr = np.array(euclid) if euclid else np.array([])
    return {
        "loss": total / max(n, 1),
        "mean_euclidean_mm": float(np.mean(arr)) if arr.size else 0.0,
        "rmse_euclidean_mm": float(np.sqrt(np.mean(arr**2))) if arr.size else 0.0,
    }


def _train_main_body(args: argparse.Namespace) -> int:
    if args.train_center_mode == "auto":
        resolved_center_mode = "heuristic" if args.mask_root else "target"
    else:
        resolved_center_mode = args.train_center_mode
    if resolved_center_mode == "heuristic" and not args.mask_root:
        print(
            "错误: train_center_mode=heuristic（或 auto 在无 mask 时不可用 heuristic）需要 --mask_root",
            file=sys.stderr,
        )
        return 1

    csv_abs = os.path.abspath(args.csv)
    out_dir_abs = os.path.abspath(args.out_dir)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    df = load_csv_rows(args.csv)
    df = filter_rows_with_ct(df, args.ct_root)
    if len(df) < 4:
        print("有效样本过少（请检查 ct_root 与 CSV 中 case_id 是否对应）", file=sys.stderr)
        return 1

    tr_df, va_df, te_df = split_by_case(
        df, args.val_ratio, args.test_ratio, args.seed
    )
    print(
        f"划分: train {len(tr_df)} 条 ({tr_df['case_id'].nunique()} 例), "
        f"val {len(va_df)}, test {len(te_df)}",
        flush=True,
    )
    split_payload: Dict[str, Any] = {
        "seed": args.seed,
        "val_ratio": float(args.val_ratio),
        "test_ratio": float(args.test_ratio),
        "csv": csv_abs,
        "ct_root": os.path.abspath(args.ct_root),
        "mask_root": os.path.abspath(args.mask_root)
        if args.mask_root
        else None,
        "train_case_ids": sorted(tr_df["case_id"].unique().tolist()),
        "val_case_ids": sorted(va_df["case_id"].unique().tolist()),
        "test_case_ids": sorted(te_df["case_id"].unique().tolist()),
        "n_rows": {"train": len(tr_df), "val": len(va_df), "test": len(te_df)},
    }
    os.makedirs(args.out_dir, exist_ok=True)
    split_path = os.path.join(args.out_dir, "split_cases.json")
    with open(split_path, "w", encoding="utf-8") as f:
        json.dump(split_payload, f, indent=2, ensure_ascii=False)
    print(f"病例划分已写入: {split_path}", flush=True)
    print(
        "提示: 每个样本都会从磁盘读取完整 CT（+掩码）再裁块，首个 epoch 在打印 "
        "Epoch 1 之前可能等待较久，属正常现象；已显示 batch 进度条。",
        flush=True,
    )

    patch_size = tuple(int(x) for x in args.patch_size)
    in_ch = 2 if args.mask_root else 1

    ds_kw = dict(
        patch_size=patch_size,
        center_mode=resolved_center_mode,
        z_percentile_l4l5=args.z_percentile_l4l5,
        z_percentile_l5s1=args.z_percentile_l5s1,
        l5s1_l5_weight=args.l5s1_l5_weight,
        center_jitter_voxels=args.center_jitter_voxels,
        max_cached_cases=args.max_cached_cases,
    )

    ds_tr = PunctureEndpointDataset(
        tr_df,
        args.ct_root,
        args.mask_root,
        train=True,
        **ds_kw,
    )
    ds_va = PunctureEndpointDataset(
        va_df,
        args.ct_root,
        args.mask_root,
        train=False,
        jitter_voxels=0.0,
        **ds_kw,
    )
    ds_te = PunctureEndpointDataset(
        te_df,
        args.ct_root,
        args.mask_root,
        train=False,
        jitter_voxels=0.0,
        **ds_kw,
    )

    print(
        f"裁块中心模式: {resolved_center_mode}  (train_center_mode={args.train_center_mode})",
        flush=True,
    )
    if resolved_center_mode == "heuristic":
        print(
            f"启发式 LRU 缓存: 每 worker 最多 {args.max_cached_cases} 例整幅 CT+mask",
            flush=True,
        )

    try:
        device = resolve_device(args.gpu)
    except RuntimeError as e:
        print(str(e), file=sys.stderr)
        return 1

    pin_mem = device.type == "cuda"
    dl_kw = dataloader_kwargs(
        args.num_workers,
        pin_mem,
        prefetch_factor=args.prefetch_factor,
        persistent_workers=args.persistent_workers,
    )
    dl_tr = DataLoader(
        ds_tr,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_batch,
        **dl_kw,
    )
    dl_va = DataLoader(
        ds_va,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_batch,
        **dl_kw,
    )
    dl_te = DataLoader(
        ds_te,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_batch,
        **dl_kw,
    )

    if device.type == "cuda":
        torch.backends.cudnn.benchmark = bool(args.cudnn_benchmark)
    use_amp = device.type == "cuda" and not args.no_amp
    scaler: Optional[Any] = new_grad_scaler() if use_amp else None

    print(f"设备: {device}", flush=True)
    bench_on = device.type == "cuda" and torch.backends.cudnn.benchmark
    pw = args.persistent_workers if args.num_workers > 0 else False
    print(
        f"加速: batch_size={args.batch_size}  num_workers={args.num_workers}  "
        f"persistent_workers={pw}  "
        f"AMP={'开' if use_amp else '关'}  cudnn.benchmark={bench_on}",
        flush=True,
    )
    if args.num_workers > 0:
        print(
            "提示: DataLoader worker 已设置 OMP/MKL/OPENBLAS=1 线程，"
            "可减轻多进程与 numpy 多线程叠加导致的卡顿。",
            flush=True,
        )
    if device.type == "cuda" and not args.cudnn_benchmark:
        print(
            "提示: cudnn.benchmark 默认关闭，避免首 batch 长时间停在 0%；"
            "稳定后再追求速度可加 --cudnn_benchmark。",
            flush=True,
        )
    print(
        "说明: 穿刺任务标签为 RAS(mm) 偏移，不能像分割 Dataset3D 那样先把整卷 resize 成小体再训；"
        "仍需整卷读入再裁 patch，I/O 与 attunet 类似但每 epoch 样本行数更多(每例 4 stem)。"
        "默认 DataLoader 设置已对齐 attunet-fifth-3d(num_workers=2, persistent_workers=False)。",
        flush=True,
    )

    model = TargetOffsetNet3D(in_channels=in_ch).to(device)
    if args.compile_torch and hasattr(torch, "compile"):
        model = torch.compile(model)  # type: ignore[assignment]
        print("已启用 torch.compile", flush=True)
    elif args.compile_torch:
        print("当前 PyTorch 无 torch.compile，已忽略 --compile_torch", flush=True)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(args.epochs, 1))
    criterion = nn.L1Loss()

    best_val = float("inf")
    best_val_metrics: Optional[Dict[str, float]] = None
    best_epoch: Optional[int] = None
    history: List[Dict[str, float]] = []

    nb = pin_mem
    for ep in range(1, args.epochs + 1):
        tr = train_epoch(
            model,
            dl_tr,
            device,
            opt,
            criterion,
            desc=f"train ep{ep}",
            scaler=scaler,
            use_amp=use_amp,
            non_blocking=nb,
            log_first_batch=(ep == 1 and args.verbose_first_batch),
        )
        va = eval_epoch(
            model,
            dl_va,
            device,
            criterion,
            desc=f"val ep{ep}",
            use_amp=use_amp,
            non_blocking=nb,
            log_first_batch=(ep == 1 and args.verbose_first_batch),
        )
        sch.step()
        history.append(
            {
                "epoch": ep,
                "train_loss": tr["loss"],
                "train_mean_euclidean_mm": tr["mean_euclidean_mm"],
                "train_rmse_euclidean_mm": tr["rmse_euclidean_mm"],
                **{f"val_{k}": v for k, v in va.items()},
            }
        )
        print(
            f"Epoch {ep}/{args.epochs}  train_L1={tr['loss']:.4f}  "
            f"val_L1={va['loss']:.4f}  val_mean_euclid_mm={va['mean_euclidean_mm']:.3f}"
        )
        improved = bool(va["loss"] < best_val)
        if improved:
            best_val = va["loss"]
            best_val_metrics = {k: float(v) for k, v in va.items()}
            best_epoch = ep
            path = os.path.join(args.out_dir, "best.pt")
            torch.save(
                {
                    "model": model.state_dict(),
                    "in_channels": in_ch,
                    "patch_size": list(patch_size),
                    "train_center_mode": resolved_center_mode,
                },
                path,
            )

    best_path = os.path.join(args.out_dir, "best.pt")
    if os.path.isfile(best_path):
        ckpt = torch.load(best_path, map_location=device)
        model.load_state_dict(ckpt["model"])

    if len(te_df) == 0:
        print("\n测试集为空（病例数较少时可减小 --val_ratio / --test_ratio）")
        te = None
    else:
        te = eval_epoch(
            model,
            dl_te,
            device,
            criterion,
            desc="test",
            use_amp=use_amp,
            non_blocking=nb,
        )
    metrics_path = os.path.join(args.out_dir, "metrics.json")
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "split": split_payload,
                "train_center_mode": resolved_center_mode,
                "train_center_arg": args.train_center_mode,
                "z_percentile_l4l5": args.z_percentile_l4l5,
                "z_percentile_l5s1": args.z_percentile_l5s1,
                "l5s1_l5_weight": args.l5s1_l5_weight,
                "center_jitter_voxels": args.center_jitter_voxels,
                "max_cached_cases": args.max_cached_cases,
                "cudnn_benchmark": bool(args.cudnn_benchmark),
                "prefetch_factor": args.prefetch_factor,
                "verbose_first_batch": bool(args.verbose_first_batch),
                "persistent_workers": bool(args.persistent_workers),
                "test": te,
                "best_val": best_val,
                "best_epoch": best_epoch,
                "best_val_metrics": best_val_metrics,
                "history": history[-5:],
            },
            f,
            indent=2,
            ensure_ascii=False,
        )
    if te is not None:
        print(
            f"\n测试集: L1={te['loss']:.4f}  平均欧氏误差(mm)≈{te['mean_euclidean_mm']:.3f}  "
            f"RMSE≈{te['rmse_euclidean_mm']:.3f}"
        )
    print(f"最佳模型: {os.path.join(args.out_dir, 'best.pt')}")
    return 0


def _build_train_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="穿刺靶点 3D CNN 基线训练")
    p.add_argument("--csv", type=str, required=True, help="extract_endpoints_dataset 生成的 CSV")
    p.add_argument("--ct_root", type=str, required=True, help="CT NIfTI 根目录（与抽取脚本查找规则一致）")
    p.add_argument("--mask_root", type=str, default=None, help="可选：分割掩码根目录（与 CT 同网格）")
    p.add_argument(
        "--train_center_mode",
        type=str,
        choices=["auto", "target", "heuristic"],
        default="auto",
        help="裁块中心：target=以真值靶点为中心（旧）；heuristic=与 infer 相同掩码启发式 c4/c5；"
        "auto=有 --mask_root 时用 heuristic，否则 target",
    )
    p.add_argument(
        "--z_percentile_l4l5",
        type=float,
        default=None,
        help="heuristic 模式：与 infer --z_p_l4l5 一致（单标签分位启发式）",
    )
    p.add_argument(
        "--z_percentile_l5s1",
        type=float,
        default=None,
        help="heuristic 模式：与 infer --z_p_l5s1 一致",
    )
    p.add_argument(
        "--l5s1_l5_weight",
        type=float,
        default=None,
        help="heuristic 模式：多标签时 L5-S1 中心加权，与 infer 一致",
    )
    p.add_argument(
        "--center_jitter_voxels",
        type=float,
        default=0.0,
        help="仅 heuristic：对裁块中心体素微抖（同病例同节段组左右共用种子）；0 关闭",
    )
    p.add_argument(
        "--max_cached_cases",
        type=int,
        default=None,
        help="仅 heuristic 生效：每个 DataLoader worker 内 LRU 缓存多少例整幅 CT+掩码；"
        " 防止多 epoch 内存涨满被系统 OOM 杀 worker（报 Killed）。内存紧可改为 4",
    )
    p.add_argument(
        "--config",
        type=str,
        default=None,
        help="超参 YAML；默认使用仓库根目录 hyperparams.example.yaml",
    )
    p.add_argument("--out_dir", type=str, required=True, help="检查点与日志目录")
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument(
        "--batch_size",
        type=int,
        default=8,
        help="默认 8；显存不足可改为 4。",
    )
    p.add_argument("--lr", type=float, default=None)
    p.add_argument("--val_ratio", type=float, default=None)
    p.add_argument("--test_ratio", type=float, default=None)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--patch_size", type=int, nargs=3, default=None)
    p.add_argument(
        "--num_workers",
        type=int,
        default=2,
        help="数据加载进程数；默认 2（与 attunet-fifth-3d 脚本里对 3D 的覆盖一致）。"
        " Windows 若报错可改为 0；仍卡首 batch 可试 0 或把数据拷到本地盘；勿加 --cudnn_benchmark。",
    )
    p.add_argument(
        "--persistent_workers",
        action="store_true",
        help="DataLoader 跨 epoch 保持 worker（略快、多占内存；默认关闭，同 attunet-fifth-3d）",
    )
    p.add_argument(
        "--no_amp",
        action="store_true",
        help="关闭混合精度（默认在 CUDA 上开启 AMP 以加速）",
    )
    p.add_argument(
        "--cudnn_benchmark",
        action="store_true",
        help="开启 cudnn.benchmark（后续 step 可能略快；首次遇到每种卷积尺寸会试算法，首 batch 可能卡很久）",
    )
    p.add_argument(
        "--verbose_first_batch",
        action="store_true",
        help="打印首个 batch 取数/训练步耗时（调试用；默认不打印，避免误以为卡在提示行）",
    )
    p.add_argument(
        "--prefetch_factor",
        type=int,
        default=2,
        help="num_workers>0 时每个 worker 预取批次数（≥2）；慢盘/NFS 可保持 2，仍卡可试减小 num_workers",
    )
    p.add_argument(
        "--compile_torch",
        action="store_true",
        help="使用 torch.compile（需 PyTorch 2.x，部分环境可能不兼容）",
    )
    p.add_argument(
        "--gpu",
        type=int,
        default=None,
        metavar="ID",
        help="使用的 GPU 编号（0 为第一块）；不传则自动用 cuda:0（若可用）；-1 强制 CPU",
    )
    return p


def _apply_hyperparams_config(args: argparse.Namespace) -> None:
    cfg_path = args.config or default_example_config_path()
    hp = load_hyperparams(cfg_path)
    tr, he = hp["training"], hp["heuristic"]
    for key, val in tr.items():
        if getattr(args, key, None) is None:
            setattr(args, key, val)
    if args.z_percentile_l4l5 is None:
        args.z_percentile_l4l5 = he["z_percentile_l4l5"]
    if args.z_percentile_l5s1 is None:
        args.z_percentile_l5s1 = he["z_percentile_l5s1"]
    if args.l5s1_l5_weight is None:
        args.l5s1_l5_weight = he["l5s1_l5_weight"]


def main() -> int:
    args = _build_train_arg_parser().parse_args()
    _apply_hyperparams_config(args)
    return _train_main_body(args)


if __name__ == "__main__":
    # 允许从 path_planning_algorithm 目录运行: python -m puncture_target.train
    sys.exit(main())
