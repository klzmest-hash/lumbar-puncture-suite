#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
真实推理：新 CT（+ 可选掩码）→ 四个靶点 RAS（mm）。

无金标准靶点时，用掩码体素 z 分位数估计 L4-L5 / L5-S1 两组裁块中心（启发式），
再与训练相同方式裁 patch、网络预测偏移，得到 end_point。
CT/掩码读入与训练共用 ``data.load_nib``（mmap 等一致）；``--z_p_*`` / ``--l5s1_l5_weight`` 应与训练时一致。

若你有更准的先验中心，可用 --centers_json 覆盖（见下方示例）。

单病例:
  python -m puncture_target.infer --ct case.nii.gz --mask mask.nii.gz --checkpoint best.pt --out_json pred.json

批量（ct_dir 下每个病例子目录或平铺的 *.nii.gz，与 extract_endpoints 查找规则一致）:
  python -m puncture_target.infer --ct_dir .../images --mask_dir .../masks --checkpoint best.pt --out_dir .../infer_out

仅推理测试集（与 train 相同 seed / val_ratio / test_ratio，或训练产出的 split_cases.json）:
  python -m puncture_target.infer --test_only --csv .../puncture_endpoints.csv --ct_dir .../images ^
    --mask_dir .../masks --gt_csv .../puncture_endpoints.csv --checkpoint best.pt --out_dir .../infer_out
  # 或: --test_only --split_json path/to/split_cases.json --gt_csv ...（终端逐例打印四靶点 L1，与训练 eval 一致）
  # 有 --gt_csv 时默认另写 infer_metrics_report.xlsx；可用 --no_excel 关闭，或 --excel_out 指定路径

除 pred_targets.json 外，默认在同目录生成四个 3D Slicer 可加载的 Fiducial：l4_l5_left.mrk.json 等（坐标为 LPS，mm）。
加 --no_slicer 可只写汇总 JSON。

公开版：推理后处理与启发式参数由 --config 加载（见 hyperparams.example.yaml）。

有金标准靶点 CSV 时（与训练相同列：case_id, stem, end_*_mm），可加 --gt_csv，在 JSON 中写入 gt_evaluation：
  • raw_network_vs_gt：网络直接输出的 RAS 与真值差（与 train/eval 的 offset 误差一致，**不含** infer 后处理）
  • final_output_vs_gt：经左右拉开/外移/椎间孔微调后的点与真值差（与 end_points_ras_mm 一致）
若肉眼觉得「很准」但训练指标偏大，多半是拿 **final** 和真值比，而 metrics 统计的是 **raw**。

centers.json 示例（RAS，mm，可选）:
  {"l4_l5_center": [-10.0, -80.0, -1300.0], "l5_s1_center": [-10.0, -85.0, -1330.0]}
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

from nibabel.orientations import aff2axcodes
import numpy as np
import torch

from puncture_target.data import (
    STEM_TO_IDX,
    build_patch_tensor,
    filter_rows_with_ct,
    find_volume_for_case,
    heuristic_centers_ras_from_mask,
    load_csv_rows,
    load_nib,
    split_by_case,
)
from puncture_target.model import TargetOffsetNet3D
from puncture_target.slicer_mrk import export_fiducials_mrk
from puncture_target.train import amp_autocast, resolve_device
from puncture_target.config_loader import load_hyperparams, default_example_config_path

_PP_ROOT = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
if _PP_ROOT not in sys.path:
    sys.path.insert(0, _PP_ROOT)


L4L5_STEMS: List[str] = ["l4_l5_left", "l4_l5_right"]
L5S1_STEMS: List[str] = ["l5_s1_left", "l5_s1_right"]

ALL_STEMS: List[str] = L4L5_STEMS + L5S1_STEMS

_LR_PAIRS: Tuple[Tuple[str, str], ...] = (
    ("l4_l5_left", "l4_l5_right"),
    ("l5_s1_left", "l5_s1_right"),
)


def enforce_lr_lateral_separation_ras(
    end_points: Dict[str, List[float]],
    collapse_mm: float = 4.0,
    half_width_mm: float = 10.0,
) -> Tuple[Dict[str, List[float]], bool]:
    """
    若同一节段 left/right 在冠状 X 上几乎重合（网络或旧 checkpoint 塌缩），
    在 RAS 下将两点沿 X 拉开：患者左侧为 −X，右侧为 +X（与 Slicer/path_planning 一致）。

    仅改 X，Y/Z 保持网络输出；不改变 patch 中心。
    """
    if collapse_mm <= 0 or half_width_mm <= 0:
        return end_points, False
    applied = False
    out = {k: list(v) for k, v in end_points.items()}
    for lk, rk in _LR_PAIRS:
        if lk not in out or rk not in out:
            continue
        xl = float(out[lk][0])
        xr = float(out[rk][0])
        if abs(xl - xr) >= collapse_mm:
            continue
        mid = 0.5 * (xl + xr)
        out[lk][0] = mid - float(half_width_mm)
        out[rk][0] = mid + float(half_width_mm)
        applied = True
    return out, applied


def lateral_extra_outward_ras(
    end_points: Dict[str, List[float]],
    extra_mm: float,
) -> Dict[str, List[float]]:
    """
    在左右靶点冠状 X 上各再向外移 extra_mm（患者左侧更负、右侧更正），
    使落点更靠近椎间孔外侧缘，而非椎体冠状中线附近。
    """
    if extra_mm <= 0:
        return end_points
    out = {k: list(v) for k, v in end_points.items()}
    for lk, rk in _LR_PAIRS:
        if lk not in out or rk not in out:
            continue
        xl = float(out[lk][0])
        xr = float(out[rk][0])
        if xl > xr:
            lk, rk = rk, lk
            xl, xr = xr, xl
        out[lk][0] = xl - float(extra_mm)
        out[rk][0] = xr + float(extra_mm)
    return out


def _anterior_unit_world(affine: np.ndarray) -> np.ndarray:
    """世界坐标里「解剖前」单位方向（用于椎间孔相对椎管更靠前）。"""
    R = np.asarray(affine, dtype=np.float64)[:3, :3]
    y_dir = R[:, 1]
    n = float(np.linalg.norm(y_dir)) + 1e-9
    y_dir = y_dir / n
    try:
        codes = aff2axcodes(affine)
        y_ax = codes[1] if len(codes) > 1 else "A"
    except Exception:
        y_ax = "A"
    if y_ax == "P":
        anterior = -y_dir
    else:
        anterior = y_dir
    na = float(np.linalg.norm(anterior)) + 1e-9
    return (anterior / na).astype(np.float64)


def anterior_foramen_nudge_ras(
    end_points: Dict[str, List[float]],
    affine: np.ndarray,
    mm: float,
) -> Dict[str, List[float]]:
    """
    沿解剖**前**（腹侧）平移 mm。注意：若靶点已偏椎体前/内缘，**不要用**或与
    ``posterior_foramen_nudge_ras`` 二选一；椎间孔在**后外侧**，通常用**后向**微调。
    """
    if mm <= 0:
        return end_points
    u = _anterior_unit_world(affine)
    out = {k: list(v) for k, v in end_points.items()}
    for k in out:
        for i in range(3):
            out[k][i] = float(out[k][i]) + float(u[i]) * float(mm)
    return out


def posterior_foramen_nudge_ras(
    end_points: Dict[str, List[float]],
    affine: np.ndarray,
    mm: float,
) -> Dict[str, List[float]]:
    """
    沿解剖**后**（背侧）平移 mm，使点更靠近**后方椎间孔**开口，远离椎体前内缘。
    世界方向为「前」向量的反方向（aff2axcodes 判定前后）。
    """
    if mm <= 0:
        return end_points
    u = -_anterior_unit_world(affine)
    out = {k: list(v) for k, v in end_points.items()}
    for k in out:
        for i in range(3):
            out[k][i] = float(out[k][i]) + float(u[i]) * float(mm)
    return out


def _superior_unit_world(affine: np.ndarray) -> np.ndarray:
    """世界坐标中「向患者头侧 / 上一椎体」单位方向（aff2axcodes 第三轴 S/I）。"""
    R = np.asarray(affine, dtype=np.float64)[:3, :3]
    k_dir = R[:, 2]
    n = float(np.linalg.norm(k_dir)) + 1e-9
    k_dir = k_dir / n
    try:
        codes = aff2axcodes(affine)
        third = codes[2] if len(codes) > 2 else "S"
    except Exception:
        third = "S"
    if third == "I":
        k_dir = -k_dir
    return k_dir.astype(np.float64)


def superior_foramen_nudge_for_stems_ras(
    end_points: Dict[str, List[float]],
    affine: np.ndarray,
    mm: float,
    stems: Sequence[str],
) -> Dict[str, List[float]]:
    """
    对指定 stem 沿头侧平移 mm（正值向头侧/上一椎体，负值向尾侧）。
    """
    if mm == 0:
        return end_points
    u = _superior_unit_world(affine)
    out = {k: list(v) for k, v in end_points.items()}
    for stem in stems:
        if stem not in out:
            continue
        for i in range(3):
            out[stem][i] = float(out[stem][i]) + float(u[i]) * float(mm)
    return out


def _nifti_stem(filename: str) -> str:
    fn = filename.lower()
    if fn.endswith(".nii.gz"):
        return filename[: -len(".nii.gz")]
    if fn.endswith(".nii"):
        return filename[: -len(".nii")]
    return ""


def discover_case_ids(ct_root: str) -> List[str]:
    """
    与数据集约定一致：若 ct_root 下有病例子目录则用语录名；否则取根目录下各 .nii/.nii.gz 的文件名（去后缀）为病例 ID。
    """
    root = os.path.abspath(ct_root)
    if not os.path.isdir(root):
        return []
    subdirs = [
        name
        for name in sorted(os.listdir(root))
        if os.path.isdir(os.path.join(root, name))
    ]
    if subdirs:
        return subdirs
    ids: List[str] = []
    for name in sorted(os.listdir(root)):
        stem = _nifti_stem(name)
        if stem:
            ids.append(stem)
    return ids


def load_gt_for_case(csv_path: str, case_id: str) -> Dict[str, List[float]]:
    """从与训练相同格式的 CSV 读取某病例四个靶点 RAS(mm)。"""
    df = load_csv_rows(csv_path)
    df = df[df["case_id"].astype(str) == str(case_id)]
    out: Dict[str, List[float]] = {}
    for _, row in df.iterrows():
        stem = str(row["stem"])
        out[stem] = [
            float(row["end_x_mm"]),
            float(row["end_y_mm"]),
            float(row["end_z_mm"]),
        ]
    return out


def derive_case_id_from_ct(ct_path: str) -> str:
    """从单文件路径推断 case_id（去 .nii/.nii.gz）。"""
    base = os.path.basename(ct_path)
    low = base.lower()
    if low.endswith(".nii.gz"):
        return base[: -len(".nii.gz")]
    if low.endswith(".nii"):
        return base[: -len(".nii")]
    return os.path.splitext(base)[0]


def l1_offset_mm(
    pred_offset: np.ndarray,
    gt_ras: np.ndarray,
    patch_center_ras: np.ndarray,
) -> float:
    """与 train 中 nn.L1Loss 一致：预测 offset 与 (gt_ras - patch_center) 的逐分量绝对误差均值。"""
    gt_off = (gt_ras - patch_center_ras).astype(np.float64)
    pred = np.asarray(pred_offset, dtype=np.float64).reshape(3)
    return float(np.mean(np.abs(pred - gt_off)))


def resolve_test_case_ids(
    *,
    split_json: Optional[str],
    csv_path: Optional[str],
    ct_root: str,
    seed: int,
    val_ratio: float,
    test_ratio: float,
) -> Set[str]:
    """
    得到测试集 case_id 集合。优先 ``split_json`` 的 test_case_ids；
    否则用与 train/export_split 相同的 CSV + ct_root + seed 比例重算。
    """
    if split_json:
        with open(split_json, "r", encoding="utf-8") as f:
            payload = json.load(f)
        ids = payload.get("test_case_ids")
        if not ids:
            raise ValueError(f"{split_json} 中缺少 test_case_ids")
        return {str(x) for x in ids}

    if not csv_path:
        raise ValueError("仅推理测试集需要 --split_json，或同时提供 --csv（与训练相同）")

    df = load_csv_rows(csv_path)
    df = filter_rows_with_ct(df, ct_root)
    if len(df) < 4:
        raise ValueError("有效样本过少，无法划分测试集（请检查 csv 与 ct_dir）")
    _, _, te_df = split_by_case(df, val_ratio, test_ratio, seed)
    return {str(x) for x in te_df["case_id"].unique().tolist()}


def format_duration_sec(sec: float) -> str:
    """将秒数格式化为易读字符串（如 2分5秒）。"""
    sec = max(0.0, float(sec))
    if sec < 60.0:
        return f"{sec:.1f}秒"
    total_s = int(round(sec))
    m, s = divmod(total_s, 60)
    if m < 60:
        return f"{m}分{s}秒"
    h, m = divmod(m, 60)
    return f"{h}小时{m}分{s}秒"


def format_case_timing_line(
    case_sec: float,
    cumulative_sec: float,
    index: int,
    total: int,
    completed_ok: int,
) -> str:
    """
    本例耗时 + 自批量开始累计耗时（第二例从第一例结束后的总时刻继续计）。
    按已完成例数粗估剩余时间。
    """
    eta_part = ""
    if completed_ok > 0 and index < total:
        avg_ok = cumulative_sec / float(completed_ok)
        remaining = total - index
        eta_part = f" | 预估剩余 {format_duration_sec(avg_ok * remaining)}"
    return (
        f"  [计时] 本例 {format_duration_sec(case_sec)} | "
        f"累计 {format_duration_sec(cumulative_sec)} ({cumulative_sec:.1f}s) | "
        f"第 {index}/{total} 例{eta_part}"
    )


def format_per_stem_l1_line(per_stem_l1: Dict[str, float]) -> str:
    parts: List[str] = []
    vals: List[float] = []
    for stem in ALL_STEMS:
        if stem in per_stem_l1:
            v = per_stem_l1[stem]
            parts.append(f"{stem}={v:.4f}")
            vals.append(v)
    mean_s = f" mean={float(np.mean(vals)):.4f}" if vals else ""
    return "  L1(mm) raw网络 vs GT: " + " ".join(parts) + mean_s


def compute_per_stem_l1_means(
    l1_by_stem: Dict[str, List[float]],
) -> Dict[str, Dict[str, Any]]:
    """四种靶点各自在全部病例上的 L1 算术平均。"""
    out: Dict[str, Dict[str, Any]] = {}
    for stem in ALL_STEMS:
        vals = l1_by_stem.get(stem, [])
        if not vals:
            continue
        arr = np.asarray(vals, dtype=np.float64)
        out[stem] = {"mean": float(np.mean(arr)), "n": int(arr.size)}
    return out


def format_aggregate_l1_summary(
    l1_by_stem: Dict[str, List[float]],
    n_cases: int,
) -> str:
    """批量推理结束时：四种靶点分别的总体 L1 均值。"""
    per_stem = compute_per_stem_l1_means(l1_by_stem)
    if not per_stem:
        return ""
    parts = [
        f"{stem}={per_stem[stem]['mean']:.4f}(n={per_stem[stem]['n']})"
        for stem in ALL_STEMS
        if stem in per_stem
    ]
    return (
        "总体 L1(mm) raw网络 vs GT（按靶点类型）: "
        + " ".join(parts)
        + f" （共 {n_cases} 例）"
    )


def compute_ras_errors_vs_gt(
    pred_ras: Dict[str, List[float]],
    gt_ras: Dict[str, List[float]],
) -> Dict[str, Any]:
    """
    逐 stem 计算 ||pred - gt||（RAS mm），及均值 / RMSE（仅对有 GT 的 stem）。
    """
    per_stem: Dict[str, Any] = {}
    errs: List[float] = []
    for stem in ALL_STEMS:
        if stem not in pred_ras or stem not in gt_ras:
            per_stem[stem] = {"missing": True}
            continue
        p = np.asarray(pred_ras[stem], dtype=np.float64)
        g = np.asarray(gt_ras[stem], dtype=np.float64)
        e = float(np.linalg.norm(p - g))
        errs.append(e)
        per_stem[stem] = {
            "euclidean_mm": e,
            "pred_ras_mm": [float(pred_ras[stem][i]) for i in range(3)],
            "gt_ras_mm": [float(gt_ras[stem][i]) for i in range(3)],
        }
    arr = np.asarray(errs, dtype=np.float64) if errs else np.zeros(0)
    summary = {
        "mean_euclidean_mm": float(np.mean(arr)) if arr.size else None,
        "rmse_euclidean_mm": float(np.sqrt(np.mean(arr**2))) if arr.size else None,
        "n_points": int(arr.size),
    }
    return {"per_stem": per_stem, "summary": summary}


def load_centers_json(path: str) -> Tuple[np.ndarray, np.ndarray]:
    with open(path, "r", encoding="utf-8") as f:
        d = json.load(f)
    if "l4_l5_center" not in d or "l5_s1_center" not in d:
        raise ValueError("centers.json 需含 l4_l5_center 与 l5_s1_center，各为 [x,y,z] mm (RAS)")
    a = np.asarray(d["l4_l5_center"], dtype=np.float64).reshape(3)
    b = np.asarray(d["l5_s1_center"], dtype=np.float64).reshape(3)
    return a, b


@torch.no_grad()
def predict_group(
    model: torch.nn.Module,
    ct: np.ndarray,
    mask: Optional[np.ndarray],
    affine: np.ndarray,
    center_ras: np.ndarray,
    stems: List[str],
    device: torch.device,
    patch_size: Tuple[int, int, int],
    use_amp: bool,
    in_ch: int,
    gt_by_stem: Optional[Dict[str, List[float]]] = None,
) -> Tuple[Dict[str, List[float]], Dict[str, float]]:
    x, patch_center_ras = build_patch_tensor(
        ct, mask if in_ch == 2 else None, affine, center_ras, patch_size
    )
    x = x.to(device)
    pc = np.asarray(patch_center_ras, dtype=np.float64)
    out: Dict[str, List[float]] = {}
    l1_map: Dict[str, float] = {}
    for stem in stems:
        stem_idx = torch.tensor([STEM_TO_IDX[stem]], dtype=torch.long, device=device)
        with amp_autocast(device, use_amp):
            offset = model(x, stem_idx)
        off_np = offset.cpu().numpy().reshape(-1)[:3]
        pred = pc + off_np
        out[stem] = [float(pred[0]), float(pred[1]), float(pred[2])]
        if gt_by_stem and stem in gt_by_stem:
            gt = np.asarray(gt_by_stem[stem], dtype=np.float64)
            l1_map[stem] = l1_offset_mm(off_np, gt, pc)
    return out, l1_map


def run_inference(
    ct_path: str,
    mask_path: Optional[str],
    model: torch.nn.Module,
    device: torch.device,
    ckpt_meta: Dict[str, Any],
    use_amp: bool,
    centers_json_path: Optional[str],
    z_p_l4l5: float,
    z_p_l5s1: float,
    l5s1_l5_weight: float,
    lr_spread: bool,
    lr_collapse_mm: float,
    lr_half_width_mm: float,
    lr_extra_lateral_mm: float,
    foramen_anterior_mm: float = 0.0,
    foramen_posterior_mm: float = 0.0,
    foramen_superior_l4l5_mm: float = 0.0,
    foramen_superior_l5s1_mm: float = 0.0,
    gt_by_stem: Optional[Dict[str, List[float]]] = None,
) -> Tuple[Optional[Dict[str, Any]], Optional[str], Dict[str, float]]:
    """
    返回 (payload, None, per_stem_l1) 或 (None, error_message, {}).
    ckpt_meta: in_channels, patch_size 已在主流程解析。
    per_stem_l1: 与 train eval 一致的 offset L1（mm），需 gt_by_stem。
    """
    in_ch = int(ckpt_meta["in_channels"])
    patch_size = tuple(int(x) for x in ckpt_meta["patch_size"])

    if os.path.isdir(ct_path):
        return (
            None,
            "ct_path 是目录。请使用批量模式: --ct_dir / --mask_dir / --out_dir",
            {},
        )
    if mask_path and os.path.isdir(mask_path):
        return None, "mask_path 是目录，请使用批量模式或传入单个 .nii.gz 文件", {}

    ct_img = load_nib(ct_path)
    ct = np.asanyarray(ct_img.dataobj)
    affine = ct_img.affine
    mask_arr: Optional[np.ndarray] = None
    if mask_path:
        m_img = load_nib(mask_path)
        mask_arr = np.asanyarray(m_img.dataobj)
        if mask_arr.shape != ct.shape:
            return None, f"掩码形状 {mask_arr.shape} 与 CT {ct.shape} 不一致", {}

    if centers_json_path:
        c4, c5 = load_centers_json(centers_json_path)
    else:
        if mask_arr is None:
            return None, "无 centers_json 时需要掩码以估计裁块中心", {}
        c4, c5 = heuristic_centers_ras_from_mask(
            mask_arr,
            affine,
            z_percentile_l4l5=z_p_l4l5,
            z_percentile_l5s1=z_p_l5s1,
            l5s1_l5_weight=l5s1_l5_weight,
        )

    results: Dict[str, Any] = {}
    per_stem_l1: Dict[str, float] = {}
    results["l4_l5"], l1_l4 = predict_group(
        model,
        ct,
        mask_arr,
        affine,
        c4,
        L4L5_STEMS,
        device,
        patch_size,
        use_amp,
        in_ch,
        gt_by_stem=gt_by_stem,
    )
    per_stem_l1.update(l1_l4)
    results["l5_s1"], l1_l5 = predict_group(
        model,
        ct,
        mask_arr,
        affine,
        c5,
        L5S1_STEMS,
        device,
        patch_size,
        use_amp,
        in_ch,
        gt_by_stem=gt_by_stem,
    )
    per_stem_l1.update(l1_l5)

    flat: Dict[str, List[float]] = {}
    for k, v in results["l4_l5"].items():
        flat[k] = v
    for k, v in results["l5_s1"].items():
        flat[k] = v

    # 后处理前：与 train/eval 一致的网络 RAS 输出（无 infer 几何后处理）
    flat_raw: Dict[str, List[float]] = {k: list(v) for k, v in flat.items()}

    spread_applied = False
    spread_note = ""
    if lr_spread:
        flat, spread_applied = enforce_lr_lateral_separation_ras(
            flat,
            collapse_mm=lr_collapse_mm,
            half_width_mm=lr_half_width_mm,
        )
        if spread_applied:
            spread_note = (
                f" 左右靶点冠状 X 间距 < {lr_collapse_mm} mm 时已按 RAS 对称拉开 "
                f"(±{lr_half_width_mm} mm)。"
            )

    post_note = ""
    if lr_extra_lateral_mm > 0:
        flat = lateral_extra_outward_ras(flat, lr_extra_lateral_mm)
        post_note += (
            f" 冠状每侧再外移 {lr_extra_lateral_mm} mm（椎间孔方向）。"
        )
    if foramen_anterior_mm > 0:
        flat = anterior_foramen_nudge_ras(flat, affine, foramen_anterior_mm)
        post_note += f" 沿解剖前（腹侧）+{foramen_anterior_mm} mm。"
    if foramen_posterior_mm > 0:
        flat = posterior_foramen_nudge_ras(flat, affine, foramen_posterior_mm)
        post_note += f" 沿解剖后（背侧/椎间孔侧）+{foramen_posterior_mm} mm。"
    if foramen_superior_l4l5_mm != 0:
        flat = superior_foramen_nudge_for_stems_ras(
            flat, affine, foramen_superior_l4l5_mm, L4L5_STEMS
        )
        post_note += (
            f" L4-L5 沿头侧平移 {foramen_superior_l4l5_mm:+.1f} mm（向上一椎体/间盘上缘）。"
        )
    if foramen_superior_l5s1_mm != 0:
        flat = superior_foramen_nudge_for_stems_ras(
            flat, affine, foramen_superior_l5s1_mm, L5S1_STEMS
        )
        post_note += (
            f" L5-S1 沿头侧平移 {foramen_superior_l5s1_mm:+.1f} mm（向 L5 / 间盘上缘）。"
        )

    payload: Dict[str, Any] = {
        "end_points_ras_mm": flat,
        "lr_lateral_spread_applied": spread_applied,
        "lr_extra_lateral_mm": float(lr_extra_lateral_mm),
        "foramen_anterior_mm": float(foramen_anterior_mm),
        "foramen_posterior_mm": float(foramen_posterior_mm),
        "foramen_superior_l4l5_mm": float(foramen_superior_l4l5_mm),
        "foramen_superior_l5s1_mm": float(foramen_superior_l5s1_mm),
        "patch_centers_ras_mm": {
            "l4_l5_group": [float(c4[0]), float(c4[1]), float(c4[2])],
            "l5_s1_group": [float(c5[0]), float(c5[1]), float(c5[2])],
        },
        "note": (
            "坐标为 RAS (mm)，与 path_planning / Slicer 管线一致；"
            "裁块中心来自掩码启发式或 centers_json。真实误差可能大于验证集（未用金标准中心裁块）。"
            + spread_note
            + post_note
        ),
        "ct": os.path.abspath(ct_path),
        "checkpoint": os.path.abspath(str(ckpt_meta.get("_checkpoint_path", ""))),
    }
    if per_stem_l1:
        payload["per_stem_l1_offset_mm"] = {
            k: float(v) for k, v in per_stem_l1.items()
        }
        vals = list(per_stem_l1.values())
        payload["mean_l1_offset_mm"] = float(np.mean(vals)) if vals else None

    if gt_by_stem:
        n_gt = sum(1 for s in ALL_STEMS if s in gt_by_stem)
        if n_gt < 4:
            payload["gt_evaluation_warning"] = (
                f"CSV 中仅找到 {n_gt}/4 个 stem 的真值，部分指标可能不全"
            )
        payload["gt_evaluation"] = {
            "coordinate_system": "RAS_mm",
            "note": (
                "raw_network_vs_gt：与训练时 eval 一致——网络输出经 patch_center+offset 得到的 RAS，"
                "未经过 infer 的左右拉开/外移/椎间孔微调。"
                " final_output_vs_gt：与 end_points_ras_mm 相同管线，含上述后处理。"
            ),
            "raw_network_vs_gt": compute_ras_errors_vs_gt(flat_raw, gt_by_stem),
            "final_output_vs_gt": compute_ras_errors_vs_gt(flat, gt_by_stem),
            "per_stem_l1_offset_mm": payload.get("per_stem_l1_offset_mm"),
            "mean_l1_offset_mm": payload.get("mean_l1_offset_mm"),
        }
    return payload, None, per_stem_l1


def main() -> int:
    p = argparse.ArgumentParser(description="穿刺靶点推理（新 CT / 掩码）")
    p.add_argument(
        "--ct",
        type=str,
        default=None,
        help="单病例：CT NIfTI 文件路径",
    )
    p.add_argument(
        "--mask",
        type=str,
        default=None,
        help="单病例：掩码；训练为 2 通道时必传",
    )
    p.add_argument(
        "--ct_dir",
        type=str,
        default=None,
        help="批量：CT 根目录（病例子目录或平铺 nii.gz）",
    )
    p.add_argument(
        "--mask_dir",
        type=str,
        default=None,
        help="批量：掩码根目录（结构与 ct_dir 一致）",
    )
    p.add_argument(
        "--out_dir",
        type=str,
        default=None,
        help="批量：输出目录，每个病例写入 <case_id>/pred_targets.json",
    )
    p.add_argument(
        "--config",
        type=str,
        default=None,
        help="超参 YAML；默认 hyperparams.example.yaml",
    )
    p.add_argument("--checkpoint", type=str, required=True, help="best.pt")
    p.add_argument(
        "--out_json",
        type=str,
        default=None,
        help="单病例：输出 JSON 文件路径",
    )
    p.add_argument("--gpu", type=int, default=None, help="同 train；默认 cuda:0；-1 为 CPU")
    p.add_argument("--no_amp", action="store_true", help="关闭 AMP")
    p.add_argument(
        "--centers_json",
        type=str,
        default=None,
        help="手动两组裁块中心 RAS(mm)，见脚本顶部示例；不传则用掩码启发式",
    )
    p.add_argument(
        "--z_p_l4l5",
        type=float,
        default=None,
        help="掩码启发式：RAS 第 3 分量(S) 上，取 s>=percentile(s, 该值) 的体素估 L4-L5 中心 (0-100)",
    )
    p.add_argument(
        "--z_p_l5s1",
        type=float,
        default=None,
        help="掩码启发式：RAS 第 3 分量(S) 上，取 s<=percentile(s, 该值) 的体素估 L5-S1 中心 (0-100)",
    )
    p.add_argument(
        "--l5s1_l5_weight",
        type=float,
        default=None,
        help="多标签掩码时 L5-S1 裁块中心 = w*L5质心+(1-w)*S1质心；骶骨大时提高 w(如0.72)避免中心进 S1",
    )
    p.add_argument(
        "--no_lr_spread",
        action="store_true",
        help="关闭「左右靶点冠状 X 过近时对称拉开」的后处理（默认开启）",
    )
    p.add_argument(
        "--lr_collapse_mm",
        type=float,
        default=None,
        help="左右靶点 X 间距小于该值则视为塌缩并拉开（默认 4 mm）",
    )
    p.add_argument(
        "--lr_half_width_mm",
        type=float,
        default=None,
        help="塌缩时对称拉开相对中线 ±X（mm，RAS）；左右间距约 2×该值（默认 14）",
    )
    p.add_argument(
        "--lr_extra_lateral_mm",
        type=float,
        default=None,
        help="无论是否塌缩，左右靶点冠状 X 每侧再向外移该值（mm），更靠椎间孔；默认 3；可设 0 关闭",
    )
    p.add_argument(
        "--foramen_anterior_mm",
        type=float,
        default=0.0,
        help="沿解剖前（腹侧）平移 mm，推向椎体/间盘前缘，与后方椎间孔相反；一般保持 0",
    )
    p.add_argument(
        "--foramen_posterior_mm",
        type=float,
        default=0.0,
        help="沿解剖后（背侧）平移 mm，靠近后方椎间孔、远离椎体前内缘；建议 3～6；勿与 anterior 同时加大",
    )
    p.add_argument(
        "--foramen_superior_l4l5_mm",
        type=float,
        default=0.0,
        help="L4-L5 两点沿头侧(S)平移 mm：正值向 L4/间盘上缘，负值向尾侧；例如 3～6",
    )
    p.add_argument(
        "--foramen_superior_l5s1_mm",
        type=float,
        default=0.0,
        help="L5-S1 两点沿头侧(S)平移 mm：正值向 L5/间盘上缘，负值向尾侧；例如 3～6",
    )
    p.add_argument(
        "--no_slicer",
        action="store_true",
        help="不生成四个 *.mrk.json（默认会生成，供 3D Slicer 打开）",
    )
    p.add_argument(
        "--gt_csv",
        type=str,
        default=None,
        help="可选：与训练相同格式的金标准 CSV，写入 pred JSON 的 gt_evaluation（逐靶点欧氏 mm）",
    )
    p.add_argument(
        "--case_id",
        type=str,
        default=None,
        help="单病例且使用 --gt_csv 时指定病例 ID；省略则从 --ct 文件名推断",
    )
    p.add_argument(
        "--test_only",
        action="store_true",
        help="批量时仅推理测试集病例（需 --split_json 或 --csv + --ct_dir，划分参数与 train 一致）",
    )
    p.add_argument(
        "--split_json",
        type=str,
        default=None,
        help="train 输出的 split_cases.json；与 --test_only 联用，直接读取 test_case_ids",
    )
    p.add_argument(
        "--csv",
        type=str,
        default=None,
        help="与 train 相同的数据 CSV；--test_only 且无 split_json 时用于按 seed 重算测试集",
    )
    p.add_argument("--seed", type=int, default=None)
    p.add_argument(
        "--val_ratio",
        type=float,
        default=None,
        help="与 train 相同的验证集比例",
    )
    p.add_argument(
        "--test_ratio",
        type=float,
        default=None,
        help="与 train 相同的测试集比例",
    )
    args = p.parse_args()
    cfg_path = args.config or default_example_config_path()
    hp = load_hyperparams(cfg_path)
    he, inf = hp["heuristic"], hp["inference"]
    if args.z_p_l4l5 is None:
        args.z_p_l4l5 = he["z_percentile_l4l5"]
    if args.z_p_l5s1 is None:
        args.z_p_l5s1 = he["z_percentile_l5s1"]
    if args.l5s1_l5_weight is None:
        args.l5s1_l5_weight = he["l5s1_l5_weight"]
    for k, v in inf.items():
        if hasattr(args, k) and getattr(args, k) is None:
            setattr(args, k, v)

    batch = args.ct_dir is not None
    single = args.ct is not None
    if batch and single:
        print("错误: 请只使用 --ct/--out_json（单病例）或 --ct_dir/--out_dir（批量）", file=sys.stderr)
        return 1
    if not batch and not single:
        print(
            "错误: 单病例请用 --ct 与 --out_json；批量请用 --ct_dir、--out_dir（2 通道模型还需 --mask_dir）",
            file=sys.stderr,
        )
        return 1
    if single and not args.out_json:
        print("错误: 单病例模式需要 --out_json", file=sys.stderr)
        return 1
    if single and args.ct:
        if os.path.isdir(args.ct):
            print(
                "错误: --ct 指向了文件夹，不是 NIfTI 文件。\n"
                "  • 批量推理请用: --ct_dir <CT根目录> --mask_dir <掩码根目录> --out_dir <输出根目录>\n"
                "  • 单病例请写具体文件，例如: ...\\\\images\\\\病例名.nii.gz",
                file=sys.stderr,
            )
            return 1
        if not os.path.isfile(args.ct):
            print(f"错误: 找不到 CT 文件: {args.ct}", file=sys.stderr)
            return 1
    if single and args.mask:
        if os.path.isdir(args.mask):
            print(
                "错误: --mask 指向了文件夹。批量请用 --mask_dir；单病例请指向某个 .nii.gz",
                file=sys.stderr,
            )
            return 1
        if not os.path.isfile(args.mask):
            print(f"错误: 找不到掩码文件: {args.mask}", file=sys.stderr)
            return 1
    if single and args.out_json and os.path.isdir(args.out_json):
        print(
            "错误: --out_json 应是 JSON 文件路径（如 ...\\\\output\\\\pred.json），不是文件夹",
            file=sys.stderr,
        )
        return 1

    ckpt_path = os.path.abspath(args.checkpoint)
    ckpt = torch.load(args.checkpoint, map_location="cpu")
    in_ch = int(ckpt.get("in_channels", 1))

    if batch:
        if not args.out_dir:
            print("错误: 批量模式需要 --out_dir", file=sys.stderr)
            return 1
        if in_ch == 2 and not args.mask_dir:
            print("错误: 2 通道模型批量推理需要 --mask_dir", file=sys.stderr)
            return 1
    patch_size = tuple(int(x) for x in ckpt.get("patch_size", [96, 96, 96]))

    if in_ch == 2 and not batch and not args.mask:
        print("错误: 检查点为 2 通道（CT+掩码），请提供 --mask", file=sys.stderr)
        return 1

    ckpt_meta: Dict[str, Any] = {
        "in_channels": in_ch,
        "patch_size": list(patch_size),
        "_checkpoint_path": ckpt_path,
    }

    try:
        device = resolve_device(args.gpu)
    except RuntimeError as e:
        print(str(e), file=sys.stderr)
        return 1
    use_amp = device.type == "cuda" and not args.no_amp

    state = ckpt["model"]
    legacy = "stem_emb.weight" in state
    model = TargetOffsetNet3D(in_channels=in_ch, legacy_embedding=legacy).to(device)
    model.load_state_dict(state, strict=True)
    model.eval()
    if legacy:
        print(
            "提示: 检查点为旧版（stem 嵌入 + 单头），左右靶点可能很接近；"
            "建议用当前代码重新训练 best.pt 以启用四 stem 独立头。",
            flush=True,
        )

    gt_csv_path: Optional[str] = (
        os.path.abspath(args.gt_csv) if args.gt_csv else None
    )

    if single:
        cid = args.case_id or derive_case_id_from_ct(args.ct)
        gt_by: Optional[Dict[str, List[float]]] = None
        if gt_csv_path:
            if not os.path.isfile(gt_csv_path):
                print(f"错误: 找不到 --gt_csv: {gt_csv_path}", file=sys.stderr)
                return 1
            gt_by = load_gt_for_case(gt_csv_path, cid)
            if not gt_by:
                print(
                    f"警告: CSV 中无 case_id={cid} 的行，gt_evaluation 将为空",
                    file=sys.stderr,
                )
        payload, err, per_stem_l1 = run_inference(
            args.ct,
            args.mask,
            model,
            device,
            ckpt_meta,
            use_amp,
            args.centers_json,
            args.z_p_l4l5,
            args.z_p_l5s1,
            args.l5s1_l5_weight,
            lr_spread=not args.no_lr_spread,
            lr_collapse_mm=args.lr_collapse_mm,
            lr_half_width_mm=args.lr_half_width_mm,
            lr_extra_lateral_mm=args.lr_extra_lateral_mm,
            foramen_anterior_mm=args.foramen_anterior_mm,
            foramen_posterior_mm=args.foramen_posterior_mm,
            foramen_superior_l4l5_mm=args.foramen_superior_l4l5_mm,
            foramen_superior_l5s1_mm=args.foramen_superior_l5s1_mm,
            gt_by_stem=gt_by,
        )
        if err:
            print(f"错误: {err}", file=sys.stderr)
            return 1
        assert payload is not None
        payload["checkpoint"] = ckpt_path
        if per_stem_l1:
            print(f"[{cid}]{format_per_stem_l1_line(per_stem_l1)}", flush=True)
        elif gt_csv_path:
            print(
                "警告: 未计算出 L1（CSV 中可能缺少该病例四靶点真值）",
                file=sys.stderr,
            )

        out_json = os.path.abspath(args.out_json)
        os.makedirs(os.path.dirname(out_json) or ".", exist_ok=True)
        with open(out_json, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, ensure_ascii=False)
        print(json.dumps(payload, indent=2, ensure_ascii=False))
        print(f"\n已写入: {out_json}", flush=True)
        if not args.no_slicer:
            out_dir_mrk = os.path.dirname(out_json) or "."
            mrk_paths = export_fiducials_mrk(
                out_dir_mrk, payload["end_points_ras_mm"]
            )
            print("3D Slicer Fiducial（LPS, mm）:", flush=True)
            for pth in mrk_paths:
                print(f"  {pth}", flush=True)
        return 0

    # 批量
    ct_dir = os.path.abspath(args.ct_dir)
    out_dir = os.path.abspath(args.out_dir)
    mask_dir = os.path.abspath(args.mask_dir) if args.mask_dir else None

    if in_ch == 2 and not mask_dir:
        print("错误: 2 通道模型需要 --mask_dir", file=sys.stderr)
        return 1

    case_ids = discover_case_ids(ct_dir)
    if not case_ids:
        print(f"错误: 在 {ct_dir} 未找到病例（子目录或 *.nii.gz）", file=sys.stderr)
        return 1

    n_discovered = len(case_ids)
    if args.test_only:
        if not args.split_json and not args.csv:
            print(
                "错误: --test_only 需要 --split_json（train 生成的 split_cases.json）"
                " 或 --csv（与 train 相同，配合 --ct_dir 重算测试集）",
                file=sys.stderr,
            )
            return 1
        try:
            test_ids = resolve_test_case_ids(
                split_json=os.path.abspath(args.split_json)
                if args.split_json
                else None,
                csv_path=os.path.abspath(args.csv) if args.csv else None,
                ct_root=ct_dir,
                seed=args.seed,
                val_ratio=args.val_ratio,
                test_ratio=args.test_ratio,
            )
        except (ValueError, OSError, json.JSONDecodeError) as e:
            print(f"错误: 无法解析测试集划分: {e}", file=sys.stderr)
            return 1
        case_ids = [c for c in case_ids if c in test_ids]
        missing = sorted(test_ids - set(case_ids))
        if missing:
            print(
                f"提示: 测试集中有 {len(missing)} 例在 ct_dir 未找到 CT，将跳过",
                flush=True,
            )
        print(
            f"仅推理测试集: {len(case_ids)} 例（ct_dir 共 {n_discovered} 例，"
            f"划分内测试集 {len(test_ids)} 例，seed={args.seed}）",
            flush=True,
        )
        if not case_ids:
            print("错误: 测试集与 ct_dir 无交集，无病例可推理", file=sys.stderr)
            return 1
    elif args.split_json:
        print(
            "提示: 已传 --split_json 但未加 --test_only，仍将推理 ct_dir 下全部病例",
            flush=True,
        )

    if args.test_only and not gt_csv_path:
        print(
            "警告: --test_only 未配合 --gt_csv，无法在终端打印四靶点 L1；"
            "评估时请加上与训练相同的 CSV",
            file=sys.stderr,
        )

    os.makedirs(out_dir, exist_ok=True)
    ok_n = 0
    batch_gt_summary: List[Dict[str, Any]] = []
    batch_timing: List[Dict[str, Any]] = []
    l1_by_stem: Dict[str, List[float]] = {s: [] for s in ALL_STEMS}
    n_cases_with_l1 = 0
    n_total = len(case_ids)
    batch_t0 = time.perf_counter()
    print(
        f"开始批量推理，共 {n_total} 例（计时从 0 起累计）...",
        flush=True,
    )
    for case_idx, case_id in enumerate(case_ids, start=1):
        case_t0 = time.perf_counter()
        ct_path = find_volume_for_case(ct_dir, case_id)
        if not ct_path:
            case_sec = time.perf_counter() - case_t0
            cumulative_sec = time.perf_counter() - batch_t0
            batch_timing.append(
                {
                    "case_id": case_id,
                    "index": case_idx,
                    "status": "skip_no_ct",
                    "case_elapsed_sec": round(case_sec, 2),
                    "cumulative_elapsed_sec": round(cumulative_sec, 2),
                }
            )
            print(f"[跳过] {case_id}: 未找到 CT", flush=True)
            print(
                format_case_timing_line(
                    case_sec, cumulative_sec, case_idx, n_total, ok_n
                ),
                flush=True,
            )
            continue
        mask_path: Optional[str] = None
        if mask_dir:
            mask_path = find_volume_for_case(mask_dir, case_id)
            if not mask_path:
                case_sec = time.perf_counter() - case_t0
                cumulative_sec = time.perf_counter() - batch_t0
                batch_timing.append(
                    {
                        "case_id": case_id,
                        "index": case_idx,
                        "status": "skip_no_mask",
                        "case_elapsed_sec": round(case_sec, 2),
                        "cumulative_elapsed_sec": round(cumulative_sec, 2),
                    }
                )
                print(f"[跳过] {case_id}: 未找到掩码", flush=True)
                print(
                    format_case_timing_line(
                        case_sec, cumulative_sec, case_idx, n_total, ok_n
                    ),
                    flush=True,
                )
                continue

        gt_by_b: Optional[Dict[str, List[float]]] = None
        if gt_csv_path:
            if not os.path.isfile(gt_csv_path):
                print(f"错误: 找不到 --gt_csv: {gt_csv_path}", file=sys.stderr)
                return 1
            gt_by_b = load_gt_for_case(gt_csv_path, case_id)

        payload, err, per_stem_l1 = run_inference(
            ct_path,
            mask_path,
            model,
            device,
            ckpt_meta,
            use_amp,
            args.centers_json,
            args.z_p_l4l5,
            args.z_p_l5s1,
            args.l5s1_l5_weight,
            lr_spread=not args.no_lr_spread,
            lr_collapse_mm=args.lr_collapse_mm,
            lr_half_width_mm=args.lr_half_width_mm,
            lr_extra_lateral_mm=args.lr_extra_lateral_mm,
            foramen_anterior_mm=args.foramen_anterior_mm,
            foramen_posterior_mm=args.foramen_posterior_mm,
            foramen_superior_l4l5_mm=args.foramen_superior_l4l5_mm,
            foramen_superior_l5s1_mm=args.foramen_superior_l5s1_mm,
            gt_by_stem=gt_by_b,
        )
        if err:
            case_sec = time.perf_counter() - case_t0
            cumulative_sec = time.perf_counter() - batch_t0
            batch_timing.append(
                {
                    "case_id": case_id,
                    "index": case_idx,
                    "status": "skip_infer_error",
                    "case_elapsed_sec": round(case_sec, 2),
                    "cumulative_elapsed_sec": round(cumulative_sec, 2),
                }
            )
            print(f"[跳过] {case_id}: {err}", flush=True)
            print(
                format_case_timing_line(
                    case_sec, cumulative_sec, case_idx, n_total, ok_n
                ),
                flush=True,
            )
            continue
        assert payload is not None
        payload["case_id"] = case_id
        payload["checkpoint"] = ckpt_path
        case_sec = time.perf_counter() - case_t0
        cumulative_sec = time.perf_counter() - batch_t0

        case_out = os.path.join(out_dir, case_id)
        os.makedirs(case_out, exist_ok=True)
        out_path = os.path.join(case_out, "pred_targets.json")
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, ensure_ascii=False)
        if not args.no_slicer:
            export_fiducials_mrk(case_out, payload["end_points_ras_mm"])
        if gt_csv_path and payload.get("gt_evaluation"):
            ge = payload["gt_evaluation"]
            raw_s = ge["raw_network_vs_gt"]["summary"]
            fin_s = ge["final_output_vs_gt"]["summary"]
            batch_gt_summary.append(
                {
                    "case_id": case_id,
                    "index": case_idx,
                    "case_elapsed_sec": round(case_sec, 2),
                    "cumulative_elapsed_sec": round(cumulative_sec, 2),
                    "per_stem_l1_offset_mm": payload.get("per_stem_l1_offset_mm"),
                    "mean_l1_offset_mm": payload.get("mean_l1_offset_mm"),
                    "raw_mean_euclidean_mm": raw_s.get("mean_euclidean_mm"),
                    "raw_rmse_euclidean_mm": raw_s.get("rmse_euclidean_mm"),
                    "final_mean_euclidean_mm": fin_s.get("mean_euclidean_mm"),
                    "final_rmse_euclidean_mm": fin_s.get("rmse_euclidean_mm"),
                    "n_points": raw_s.get("n_points"),
                }
            )
        batch_timing.append(
            {
                "case_id": case_id,
                "index": case_idx,
                "status": "ok",
                "case_elapsed_sec": round(case_sec, 2),
                "cumulative_elapsed_sec": round(cumulative_sec, 2),
            }
        )
        line = (
            f"[完成] {case_id} -> {out_path}"
            + ("" if args.no_slicer else f" + 4x *.mrk.json")
        )
        print(line, flush=True)
        if per_stem_l1:
            print(format_per_stem_l1_line(per_stem_l1), flush=True)
            for stem, v in per_stem_l1.items():
                if stem in l1_by_stem:
                    l1_by_stem[stem].append(float(v))
            n_cases_with_l1 += 1
        elif gt_csv_path:
            print(
                f"  警告: {case_id} 在 CSV 中无完整四靶点真值，未输出 L1",
                flush=True,
            )
        print(
            format_case_timing_line(
                case_sec, cumulative_sec, case_idx, n_total, ok_n + 1
            ),
            flush=True,
        )
        ok_n += 1

    total_elapsed_sec = time.perf_counter() - batch_t0
    per_stem_l1_means = compute_per_stem_l1_means(l1_by_stem)
    timing_doc: Dict[str, Any] = {
        "total_elapsed_sec": round(total_elapsed_sec, 2),
        "total_elapsed_human": format_duration_sec(total_elapsed_sec),
        "n_cases_planned": n_total,
        "n_cases_ok": ok_n,
        "per_case": batch_timing,
        "note": "cumulative_elapsed_sec 为自批量开始起的总耗时（连续计时，便于估算后续耗时）",
    }
    timing_path = os.path.join(out_dir, "infer_timing_summary.json")
    with open(timing_path, "w", encoding="utf-8") as f:
        json.dump(timing_doc, f, indent=2, ensure_ascii=False)

    sum_path: Optional[str] = None
    if gt_csv_path and batch_gt_summary:
        sum_path = os.path.join(out_dir, "gt_infer_metrics_summary.json")
        summary_doc: Dict[str, Any] = {
            "per_case": batch_gt_summary,
            "timing": timing_doc,
            "overall_l1_offset_mm": {
                "per_stem": per_stem_l1_means,
                "n_cases": n_cases_with_l1,
                "note": "四种靶点各自在所有病例上的 L1 算术平均（非四类混在一起）",
            },
        }
        with open(sum_path, "w", encoding="utf-8") as f:
            json.dump(summary_doc, f, indent=2, ensure_ascii=False)
        print(f"金标准对比汇总: {sum_path}", flush=True)

    denom = len(case_ids)
    scope = "测试集" if args.test_only else "待处理"
    print(
        f"\n批量完成：成功 {ok_n} / {denom} 例（{scope}），"
        f"总耗时 {format_duration_sec(total_elapsed_sec)}，"
        f"输出根目录: {out_dir}",
        flush=True,
    )
    print(f"计时汇总: {timing_path}", flush=True)
    agg_line = format_aggregate_l1_summary(l1_by_stem, n_cases_with_l1)
    if agg_line:
        print(agg_line, flush=True)

    summary_batch: Dict[str, Any] = {
        "task": "puncture_target_infer",
        "status": "completed" if ok_n else "partial",
        "mode": "batch",
        "test_only": args.test_only,
        "ct_dir": ct_dir,
        "mask_dir": mask_dir,
        "out_dir": out_dir,
        "n_cases_planned": n_total,
        "n_cases_ok": ok_n,
        "total_elapsed_sec": round(total_elapsed_sec, 2),
        "timing_summary_path": timing_path,
        "overall_l1_per_stem": per_stem_l1_means,
        "n_cases_with_l1": n_cases_with_l1,
    }
    if gt_csv_path and batch_gt_summary:
        summary_batch["gt_metrics_summary_path"] = sum_path
    return 0 if ok_n else 1


if __name__ == "__main__":
    sys.exit(main())
