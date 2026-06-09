#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
手术工作通道路径规划算法

功能：
1. 生成半径4mm、长度150mm的手术工作通道圆柱体
2. 使用随机采样的方式生成多个通道
3. 导入nii.gz格式的分割完的数据
4. 统计通道与分割数据之间各个结构的交集体积
5. 最优路径（--filter_optimal）：默认「同侧髂骨零碰撞」可行约束（准入侧髂骨体积≤数值零容差）+ 字典序（关节突→不含背景的总交集体积→同侧多裂→竖脊→腰大→腰方）；可选用 --iliac_max_intersection_mm3>容差恢复宽松门槛。JSON 的 total_intersection_volume_mm3 仍含标签 0，排序第二键用不含背景之和
6. 靶点：默认使用与脚本同目录 puncture_baseline/best.pt 做 puncture_target 推理（需 --ct/--ct_dir）；或 --endpoints_csv 金标准坐标；或 --legacy_target 几何靶点

公开版：几何/排序等可调项见 planning_config.py（占位值，需本地标定）。
"""

import os
import sys
import numpy as np
import nibabel as nib
from nibabel.orientations import aff2axcodes
from scipy.spatial.distance import cdist
try:
    from scipy import ndimage
    SCIPY_NDIMAGE_AVAILABLE = True
except ImportError:
    SCIPY_NDIMAGE_AVAILABLE = False
from tqdm import tqdm
import argparse
import csv
import json
import re
from collections import defaultdict
from typing import Tuple, List, Dict, Optional, Callable
import warnings
warnings.filterwarnings('ignore')

from planning_config import PLANNING

# 标签名称映射（与 inference_3d_example.py / 数据集一致）
LABEL_NAME_MAPPING = {
    1: 'L1', 2: 'L2', 3: 'L3', 4: 'L4', 5: 'L5', 6: 'S1',
    10: 'PMR', 11: 'QLR', 12: 'ESR', 13: 'MFR', 14: 'PML',
    15: 'QLL', 16: 'ESL', 17: 'MFL', 18: 'Ilium_left', 19: 'Ilium_right'
}

# 椎旁肌：10–13 右侧（PMR/QLR/ESR/MFR），14–17 左侧（PML/QLL/ESL/MFL），与 inference_3d 命名一致
PARASPINAL_LABELS_RIGHT = (10, 11, 12, 13)
PARASPINAL_LABELS_LEFT = (14, 15, 16, 17)

# 最优路径筛选/打印用中文说明
FILTER_OPTIMAL_LABEL_NAMES = {
    4: "L4椎体", 5: "L5椎体", 6: "S1(骶骨)",
    10: "腰大肌-右(PMR)", 11: "腰方肌-右(QLR)", 12: "竖脊肌-右(ESR)", 13: "多裂肌-右(MFR)",
    14: "腰大肌-左(PML)", 15: "腰方肌-左(QLL)", 16: "竖脊肌-左(ESL)", 17: "多裂肌-左(MFL)",
    18: "髂骨-左", 19: "髂骨-右",
}

# 髂骨「零碰撞」可行约束：同侧髂骨交集体积视为 0 的数值容差（mm³）
DEFAULT_ILIAC_ZERO_EPS_MM3 = PLANNING.iliac_zero_eps_mm3

# angle_average / Spearman 工作表中「合计」行：仅对规划关注结构（chart_labels_*）求和，与 CPA-CSA 图总柱一致
ANGLE_AVERAGE_PLANNING_TOTAL_ROW = "合计(规划关注结构)"


def _angle_avg_structure_display_name(lbl: int) -> str:
    """Excel / Spearman 行名：与 FILTER_OPTIMAL_LABEL_NAMES 一致的中文结构名。"""
    return FILTER_OPTIMAL_LABEL_NAMES.get(
        int(lbl), LABEL_NAME_MAPPING.get(int(lbl), f"标签{int(lbl)}")
    )


def label_display_title_for_chart(label: int) -> str:
    """Chart title: same English short names as LABEL_NAME_MAPPING / inference_3d (not FILTER_OPTIMAL Chinese)."""
    return LABEL_NAME_MAPPING.get(label, f"Label {label}")


# Prose names for CPA–CSA bar charts: "Intersection Volume of …" (title / y-axis), aligned with LABEL_NAME_MAPPING semantics.
INTERSECTION_VOLUME_CHART_STRUCTURE_NAME = {
    1: "L1 vertebra",
    2: "L2 vertebra",
    3: "L3 vertebra",
    4: "L4 vertebra",
    5: "L5 vertebra",
    6: "S1 sacrum",
    10: "Psoas major (right)",
    11: "Quadratus lumborum (right)",
    12: "Erector spinae (right)",
    13: "Multifidus (right)",
    14: "Psoas major (left)",
    15: "Quadratus lumborum (left)",
    16: "Erector spinae (left)",
    17: "Multifidus (left)",
    18: "Ilium (left)",
    19: "Ilium (right)",
}


def label_prose_name_for_intersection_chart(label: int) -> str:
    """English structure phrase for plot titles/labels (Intersection Volume of …)."""
    return INTERSECTION_VOLUME_CHART_STRUCTURE_NAME.get(
        int(label), label_display_title_for_chart(int(label))
    )


def _spearman_heatmap_row_label_en(cn_row: str) -> str:
    """Spearman ρ 热图 Y 轴：Excel 中 Structure 为中文，绘图改为英文以避免默认字体乱码。"""
    s = str(cn_row).strip()
    if s == ANGLE_AVERAGE_PLANNING_TOTAL_ROW:
        return "Total"
    for lid, zh in FILTER_OPTIMAL_LABEL_NAMES.items():
        if zh == s:
            return label_prose_name_for_intersection_chart(int(lid))
    if s.startswith("标签"):
        tail = s.replace("标签", "").strip()
        if tail.isdigit():
            return label_prose_name_for_intersection_chart(int(tail))
    return s


def label_safe_filename_stem_for_chart(label: int) -> str:
    """英文结构短名（与 inference_3d_example LABEL_NAME_MAPPING 一致），用于文件名。"""
    base = LABEL_NAME_MAPPING.get(label, f"label_{label}")
    for ch in r'\/:*?"<>|':
        base = base.replace(ch, "_")
    return base


def chart_labels_l4_l5_for_side(facet_run_side: str) -> List[int]:
    """L4-L5 图表/角度汇总：椎体 + 与入路同侧的椎旁肌 + 同侧髂骨（右 19，左 18）。"""
    s = (facet_run_side or 'auto').strip().lower()
    if s == 'right':
        return [4, 5, 10, 11, 12, 13, 19]
    if s == 'left':
        return [4, 5, 14, 15, 16, 17, 18]
    return [4, 5, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19]


def chart_labels_l5_s1_for_side(facet_run_side: str) -> List[int]:
    s = (facet_run_side or 'auto').strip().lower()
    if s == 'right':
        return [5, 6, 10, 11, 12, 13, 19]
    if s == 'left':
        return [5, 6, 14, 15, 16, 17, 18]
    return [5, 6, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19]


def chart_labels_for_planning_task(
    use_facet_joint: bool,
    use_l5_s1_disc: bool,
    facet_side: str,
) -> List[int]:
    """
    与临床/图表一致的关注结构 ID 序列（去重保序）：
    L4-L5：L4、L5、同侧四肌、同侧髂骨（右 19，左 18）；
    L5-S1：L5、S1(骶骨标签)、同侧四肌、同侧髂骨。
    若同次任务同时开 facet 与 disc，取并集。
    """
    s = (facet_side or "auto").strip().lower()
    merged: List[int] = []
    if use_facet_joint:
        merged.extend(chart_labels_l4_l5_for_side(s))
    if use_l5_s1_disc:
        merged.extend(chart_labels_l5_s1_for_side(s))
    seen: set = set()
    out: List[int] = []
    for x in merged:
        if x not in seen:
            seen.add(x)
            out.append(x)
    return out


def _path_planning_algo_dir() -> str:
    return os.path.dirname(os.path.abspath(__file__))


_PP_ROOT = _path_planning_algo_dir()
if _PP_ROOT not in sys.path:
    sys.path.insert(0, _PP_ROOT)


def resolve_puncture_checkpoint_path(checkpoint_user: Optional[str]) -> Optional[str]:
    """puncture_target 权重：用户路径优先；否则优先本包 puncture_baseline/best.pt，再项目 runs/puncture_baseline。"""
    if checkpoint_user and os.path.isfile(checkpoint_user):
        return os.path.abspath(checkpoint_user)
    here = _path_planning_algo_dir()
    for rel in (
        os.path.join(here, "puncture_baseline", "best.pt"),
        os.path.join(os.path.dirname(here), "runs", "puncture_baseline", "best.pt"),
    ):
        if os.path.isfile(rel):
            return os.path.abspath(rel)
    return None


def default_puncture_baseline_checkpoint_path() -> Optional[str]:
    """与脚本同目录下 puncture_baseline/best.pt（存在则返回绝对路径）。"""
    p = os.path.join(_path_planning_algo_dir(), "puncture_baseline", "best.pt")
    return os.path.abspath(p) if os.path.isfile(p) else None


# 与 extract_endpoints_dataset / replan_from_manual_targets 中 stem 列一致
_STEM_ENDPOINTS_CSV_RE = re.compile(r"^(l4_l5|l5_s1)_(left|right|auto)$", re.IGNORECASE)


def parse_stem_endpoints_csv(stem: str) -> Tuple[str, str]:
    """stem 如 l4_l5_left -> (节段 'l4_l5'|'l5_s1', facet 侧 'left'|'right'|'auto')。"""
    m = _STEM_ENDPOINTS_CSV_RE.match((stem or "").strip())
    if not m:
        raise ValueError(
            f"stem 不符合约定 l4_l5_left / l5_s1_right / *_auto: {stem!r}"
        )
    return m.group(1).lower(), m.group(2).lower()


def read_tasks_from_endpoints_csv(csv_path: str) -> List[Tuple[str, str, np.ndarray]]:
    """
    读取 extract_endpoints_dataset 导出的 CSV（end_*_mm 为 RAS mm，与 results.json end_point 一致）。
    返回 [(case_id, stem, target_ras), ...]。
    """
    path = os.path.abspath(csv_path)
    if not os.path.isfile(path):
        raise FileNotFoundError(f"CSV 不存在: {path}")
    out: List[Tuple[str, str, np.ndarray]] = []
    required = {"case_id", "stem", "end_x_mm", "end_y_mm", "end_z_mm"}
    with open(path, "r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames:
            return out
        headers = {h.strip() for h in reader.fieldnames if h}
        missing = required - headers
        if missing:
            raise ValueError(
                f"CSV 缺少列 {missing}，需要 case_id、stem、end_x_mm、end_y_mm、end_z_mm。"
            )
        for lineno, row in enumerate(reader, start=2):
            case_id = (row.get("case_id") or "").strip()
            stem = (row.get("stem") or "").strip()
            if not case_id or not stem:
                print(f"[跳过] 第 {lineno} 行: 空 case_id 或 stem")
                continue
            try:
                x = float(row["end_x_mm"])
                y = float(row["end_y_mm"])
                z = float(row["end_z_mm"])
            except (KeyError, ValueError, TypeError) as e:
                print(f"[跳过] 第 {lineno} 行 ({case_id}/{stem}): 坐标无效 — {e}")
                continue
            try:
                parse_stem_endpoints_csv(stem)
            except ValueError as e:
                print(f"[跳过] 第 {lineno} 行: {e}")
                continue
            ras = np.array([x, y, z], dtype=float)
            out.append((case_id, stem, ras))
    return out


def find_mask_for_case_endpoints_csv(
    masks_dir: str, case_id: str, case_dir: Optional[str] = None
) -> Optional[str]:
    """先在 masks_dir 下找平铺的 {case_id}.nii(.gz)，再在 case_dir 或 masks_dir/case_id/ 内找。"""
    for ext in (".nii.gz", ".nii"):
        p = os.path.join(masks_dir, f"{case_id}{ext}")
        if os.path.isfile(p):
            return p
    nested = os.path.join(masks_dir, case_id)
    if os.path.isdir(nested):
        for ext in (".nii.gz", ".nii"):
            p = os.path.join(nested, f"{case_id}{ext}")
            if os.path.isfile(p):
                return p
    if case_dir:
        for ext in (".nii.gz", ".nii"):
            p = os.path.join(case_dir, f"{case_id}{ext}")
            if os.path.isfile(p):
                return p
    return None


def _nii_stem_for_pair(path: str) -> str:
    b = os.path.basename(path)
    low = b.lower()
    if low.endswith(".nii.gz"):
        return b[: -len(".nii.gz")]
    if low.endswith(".nii"):
        return b[: -len(".nii")]
    return os.path.splitext(b)[0]


def resolve_ct_nifti_for_mask(
    segmentation_file: str,
    ct_file: Optional[str],
    ct_dir: Optional[str],
) -> Optional[str]:
    """为分割掩码解析配对的 CT NIfTI（与 puncture_target 推理一致需同网格）。"""
    if ct_dir and os.path.isdir(ct_dir):
        stem = _nii_stem_for_pair(segmentation_file)
        for name in (f"{stem}.nii.gz", f"{stem}.nii"):
            cand = os.path.join(ct_dir, name)
            if os.path.isfile(cand):
                return cand
        try:
            for fn in sorted(os.listdir(ct_dir)):
                if not (fn.lower().endswith(".nii.gz") or fn.lower().endswith(".nii")):
                    continue
                s2 = _nii_stem_for_pair(fn)
                if s2 == stem:
                    return os.path.join(ct_dir, fn)
        except OSError:
            pass
        return None
    if ct_file and os.path.isfile(ct_file):
        return os.path.abspath(ct_file)
    return None


def _ensure_puncture_target_import_path() -> None:
    d = _path_planning_algo_dir()
    if d not in sys.path:
        sys.path.insert(0, d)


# (ckpt_abs, gpu_int) -> (model, device, ckpt_meta, use_amp)
_PUNCTURE_BUNDLE_CACHE: Dict[Tuple[str, Optional[int]], Tuple] = {}


def _get_cached_puncture_bundle(ckpt_path: str, gpu: Optional[int]):
    ckpt_abs = os.path.abspath(ckpt_path)
    key = (ckpt_abs, gpu)
    if key in _PUNCTURE_BUNDLE_CACHE:
        return _PUNCTURE_BUNDLE_CACHE[key]
    _ensure_puncture_target_import_path()
    import torch
    from puncture_target.model import TargetOffsetNet3D
    from puncture_target.train import resolve_device

    ckpt = torch.load(ckpt_abs, map_location="cpu")
    in_ch = int(ckpt.get("in_channels", 1))
    patch_size = tuple(int(x) for x in ckpt.get("patch_size", [96, 96, 96]))
    ckpt_meta: Dict = {
        "in_channels": in_ch,
        "patch_size": list(patch_size),
        "_checkpoint_path": ckpt_abs,
    }
    device = resolve_device(gpu)
    use_amp = device.type == "cuda"
    state = ckpt["model"]
    legacy = "stem_emb.weight" in state
    model = TargetOffsetNet3D(in_channels=in_ch, legacy_embedding=legacy).to(device)
    model.load_state_dict(state, strict=True)
    model.eval()
    bundle = (model, device, ckpt_meta, use_amp)
    _PUNCTURE_BUNDLE_CACHE[key] = bundle
    return bundle


def intersection_volume_mm3_for_label(volumes_dict: Optional[Dict], label) -> float:
    """Per-label volume from intersection_volumes_mm3; JSON keys are often str."""
    if not volumes_dict:
        return 0.0
    v = volumes_dict.get(label)
    if v is not None:
        return float(v)
    v = volumes_dict.get(str(label))
    if v is not None:
        return float(v)
    if isinstance(label, str) and label.isdigit():
        v = volumes_dict.get(int(label))
        if v is not None:
            return float(v)
    return 0.0


def _intersection_label_ids_in_results(results: List[Dict]) -> set:
    """交集体积字典中出现的整数标签（排除背景 0）。"""
    s = set()
    for result in results:
        vol = result.get("intersection_volumes_mm3") or {}
        for k in vol.keys():
            try:
                ki = int(k) if not isinstance(k, int) else int(k)
            except (TypeError, ValueError):
                continue
            if ki != 0:
                s.add(ki)
    return s


def _path_sum_intersection_mm3_for_labels(path: Dict, labels: List[int]) -> float:
    """对给定标签列表求交集体积之和（用于与 chart_labels_* 一致的总柱图）。"""
    vols = path.get("intersection_volumes_mm3") or {}
    return float(
        sum(intersection_volume_mm3_for_label(vols, lb) for lb in labels)
    )


def _cpa_csa_bar_colors(num_csa: int) -> List:
    """Bar colors: tab10 for first 10 series (blue=first CSA), then tab20."""
    import matplotlib.pyplot as plt

    if num_csa <= 10:
        cmap = plt.cm.get_cmap("tab10")
        return [cmap(i) for i in range(num_csa)]
    import matplotlib.cm as cm

    _cm = cm.get_cmap("tab20")
    return [_cm(i / max(num_csa, 1)) for i in range(num_csa)]


def results_use_spherical_alpha_beta_axes(results: Optional[List[Dict]]) -> bool:
    """True：球面 α×β 网格结果；此时 coronal/transverse 角度字段存的是 α、β，而非与解剖面的派生角。"""
    if not results:
        return False
    return (results[0].get("angle_grid_mode") or "") == "spherical_alpha_beta"


def angle_volume_charts_folder_name(results: Optional[List[Dict]]) -> str:
    """
    交集体积按角度分面的柱状图输出子目录名。
    球面 α×β 与可行域 CPA×CSA 分目录，避免 PNG 与 cpa_csa 混名困扰。
    """
    return (
        "alpha_beta_volume_charts"
        if results_use_spherical_alpha_beta_axes(results)
        else "cpa_csa_volume_charts"
    )


def _apply_cpa_csa_legend(
        ax, fig, csa_angles: List[float], bar_containers: List,
        secondary_angle_name: str = "CSA") -> None:
    """柱状图图例：第二角度轴（默认 CSA；球面模式下传 β）。"""
    num_csa = len(csa_angles)
    if num_csa == 0 or not bar_containers:
        return
    handles = list(bar_containers)
    labels = [rf'{secondary_angle_name} ${float(csa):.0f}^\circ$' for csa in csa_angles]
    ncol = 2 if num_csa > 1 else 1
    ax.legend(
        handles,
        labels,
        loc="upper left",
        bbox_to_anchor=(1.02, 1.0),
        borderaxespad=0,
        fontsize=9,
        ncol=ncol,
        fancybox=False,
        facecolor="white",
        edgecolor="black",
        framealpha=1.0,
        frameon=True,
    )
    fig.tight_layout(rect=[0, 0, 0.76, 1])


# Excel导出相关导入
try:
    import pandas as pd
    PANDAS_AVAILABLE = True
except ImportError:
    PANDAS_AVAILABLE = False

try:
    from scipy.stats import spearmanr
    SPEARMAN_AVAILABLE = True
except ImportError:
    spearmanr = None  # type: ignore
    SPEARMAN_AVAILABLE = False

# 3D可视化相关导入
try:
    import matplotlib.pyplot as plt
    import matplotlib
    MATPLOTLIB_AVAILABLE = True
    
    # 配置中文字体支持
    def setup_chinese_font():
        """配置matplotlib中文字体"""
        import platform
        from matplotlib.font_manager import FontProperties
        
        system = platform.system()
        
        # 常见中文字体列表（按优先级排序）
        chinese_fonts = []
        
        if system == 'Windows':
            chinese_fonts = ['Microsoft YaHei', 'SimHei', 'SimSun', 'KaiTi', 'FangSong']
        elif system == 'Darwin':  # macOS
            chinese_fonts = ['PingFang SC', 'STHeiti', 'Arial Unicode MS', 'Heiti SC']
        else:  # Linux
            chinese_fonts = ['Noto Sans CJK SC', 'WenQuanYi Micro Hei', 'WenQuanYi Zen Hei', 
                           'Droid Sans Fallback', 'AR PL UMing CN']
        
        # 获取系统所有可用字体
        try:
            from matplotlib.font_manager import fontManager
            available_fonts = [f.name for f in fontManager.ttflist]
        except Exception:
            available_fonts = []
        
        # 尝试设置中文字体
        font_set = False
        for font_name in chinese_fonts:
            # 检查字体是否在可用字体列表中
            if font_name in available_fonts or system == 'Windows':
                try:
                    # 设置字体
                    plt.rcParams['font.sans-serif'] = [font_name] + plt.rcParams['font.sans-serif']
                    # 解决负号显示问题
                    plt.rcParams['axes.unicode_minus'] = False
                    font_set = True
                    print(f"已设置中文字体: {font_name}")
                    break
                except Exception as e:
                    continue
        
        if not font_set:
            # 如果所有字体都不可用，尝试使用系统默认字体
            try:
                plt.rcParams['axes.unicode_minus'] = False
                print("警告: 未找到中文字体，中文可能显示为方块")
                print("提示: 可以安装中文字体或手动配置matplotlib字体")
            except Exception:
                pass
    
    # 初始化中文字体（延迟到第一次使用时）
    _chinese_font_initialized = False
    
    def ensure_chinese_font():
        """确保中文字体已初始化"""
        global _chinese_font_initialized
        if not _chinese_font_initialized:
            setup_chinese_font()
            _chinese_font_initialized = True
    
except ImportError:
    MATPLOTLIB_AVAILABLE = False


class SurgicalPathPlanner:
    """手术路径规划器"""
    
    def __init__(self, 
                 channel_radius_mm: float = 4.0,
                 channel_length_mm: float = 150.0,
                 resolution: float = 0.5,
                 min_angle_deg: float = 0.0,
                 max_angle_deg: float = 30.0):
        """
        初始化路径规划器
        
        Args:
            channel_radius_mm: 通道半径（毫米）
            channel_length_mm: 通道长度（毫米）
            resolution: 圆柱体内部采样分辨率（毫米），用于体积计算
            min_angle_deg: 轨迹与冠状面/横断面的最小夹角（度），默认0度
            max_angle_deg: 轨迹与冠状面/横断面的最大夹角（度），默认30度
        """
        self.channel_radius_mm = channel_radius_mm
        self.channel_length_mm = channel_length_mm
        self.resolution = resolution
        self.min_angle_deg = min_angle_deg
        self.max_angle_deg = max_angle_deg
        # 用于存储最后一次规划的数据，以便可视化
        self._last_segmentation = None
        self._last_affine = None
        self._last_voxel_sizes = None
        self._last_segmentation_file = None
        # 缓存关节突区域掩码，避免重复计算
        self._facet_joint_masks_cache = None
        self._cached_segmentation_shape = None
        # 当前使用的 L4 / L5 标签（在 plan_paths 中根据参数更新）
        self._l4_label = 4
        self._l5_label = 5
        
    def load_nii_gz(self, file_path: str) -> Tuple[np.ndarray, np.ndarray, Dict]:
        """
        加载NII.GZ格式的分割数据
        
        Args:
            file_path: NII.GZ文件路径
            
        Returns:
            data: 分割数据数组
            affine: 仿射变换矩阵
            metadata: 元数据字典（包含体素大小等信息）
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"文件不存在: {file_path}")
        
        nii_img = nib.load(file_path)
        data = nii_img.get_fdata()
        affine = nii_img.affine
        
        # 提取体素大小（从affine矩阵的对角线元素）
        voxel_sizes = np.abs(np.diag(affine)[:3])
        
        metadata = {
            'shape': data.shape,
            'voxel_sizes': voxel_sizes,
            'affine': affine,
            'header': nii_img.header
        }
        
        print(f"已加载文件: {file_path}")
        print(f"  数据形状: {data.shape}")
        print(f"  体素大小: {voxel_sizes} mm")
        print(f"  标签值范围: {int(data.min())} - {int(data.max())}")
        print(f"  唯一标签数: {len(np.unique(data))}")
        
        return data, affine, metadata
    
    def get_voxel_coordinates(self, shape: Tuple[int, int, int], 
                             affine: np.ndarray) -> np.ndarray:
        """
        获取体素坐标对应的物理坐标
        
        Args:
            shape: 数据形状 (H, W, D)
            affine: 仿射变换矩阵
            
        Returns:
            physical_coords: 物理坐标数组 (N, 3)，每行是(x, y, z)物理坐标
        """
        h, w, d = shape
        # 生成体素坐标网格
        voxel_coords = np.mgrid[0:h, 0:w, 0:d].reshape(3, -1).T
        
        # 转换为齐次坐标
        ones = np.ones((voxel_coords.shape[0], 1))
        voxel_coords_homogeneous = np.hstack([voxel_coords, ones])
        
        # 应用仿射变换得到物理坐标
        physical_coords = (affine @ voxel_coords_homogeneous.T).T[:, :3]
        
        return physical_coords
    
    def generate_cylinder_points(self, 
                                start_point: np.ndarray,
                                direction: np.ndarray,
                                radius_mm: float,
                                length_mm: float,
                                resolution: float) -> np.ndarray:
        """
        生成圆柱体内的点（物理坐标）
        
        Args:
            start_point: 圆柱体起点 (3,)
            direction: 圆柱体方向向量 (3,)，会被归一化
            radius_mm: 圆柱体半径（毫米）
            length_mm: 圆柱体长度（毫米）
            resolution: 采样分辨率（毫米）
            
        Returns:
            points: 圆柱体内的点 (N, 3)，物理坐标
        """
        # 归一化方向向量
        direction = direction / np.linalg.norm(direction)
        
        # 生成沿轴线的采样点
        num_steps = int(length_mm / resolution) + 1
        t_values = np.linspace(0, length_mm, num_steps)
        
        # 生成每个横截面的点
        # 横截面采样：使用极坐标
        num_radial = max(3, int(radius_mm / resolution))
        num_angular = max(8, int(2 * np.pi * radius_mm / resolution))
        
        # 生成径向和角度采样
        r_values = np.linspace(0, radius_mm, num_radial)
        theta_values = np.linspace(0, 2 * np.pi, num_angular, endpoint=False)
        
        # 构建横截面网格
        R, Theta = np.meshgrid(r_values, theta_values)
        
        # 找到两个垂直于direction的基向量
        # 任意选择一个不平行于direction的向量
        if abs(direction[0]) < 0.9:
            v1 = np.array([1, 0, 0])
        else:
            v1 = np.array([0, 1, 0])
        
        # 使用Gram-Schmidt正交化
        v1 = v1 - np.dot(v1, direction) * direction
        v1 = v1 / np.linalg.norm(v1)
        v2 = np.cross(direction, v1)
        v2 = v2 / np.linalg.norm(v2)
        
        # 生成所有点
        points = []
        for t in t_values:
            center = start_point + t * direction
            for r, theta in zip(R.flatten(), Theta.flatten()):
                if r <= radius_mm:  # 确保在圆内
                    # 在横截面上的偏移
                    offset = r * (np.cos(theta) * v1 + np.sin(theta) * v2)
                    point = center + offset
                    points.append(point)
        
        return np.array(points)
    
    def physical_to_voxel(self, physical_coords: np.ndarray, 
                         affine: np.ndarray) -> np.ndarray:
        """
        将物理坐标转换为体素坐标
        
        Args:
            physical_coords: 物理坐标 (N, 3)
            affine: 仿射变换矩阵
            
        Returns:
            voxel_coords: 体素坐标 (N, 3)
        """
        # 转换为齐次坐标
        ones = np.ones((physical_coords.shape[0], 1))
        physical_homogeneous = np.hstack([physical_coords, ones])
        
        # 应用逆仿射变换
        inv_affine = np.linalg.inv(affine)
        voxel_coords = (inv_affine @ physical_homogeneous.T).T[:, :3]
        
        return voxel_coords
    
    def extract_facet_joint_region(self,
                                  segmentation: np.ndarray,
                                  label: int,
                                  posterior_ratio: float = PLANNING.posterior_ratio,
                                  superior_ratio: float = PLANNING.superior_ratio,
                                  inferior_ratio: float = PLANNING.inferior_ratio) -> np.ndarray:
        """
        提取椎体的关节突区域（上下关节突合并区域）

        参照 `extract_facet_joints.py` 中的实现方式：
        - 上关节突：椎体的后部 + 上 1/3 区域
        - 下关节突：椎体的后部 + 下 1/3 区域

        在路径规划中，我们使用「上下关节突合并区域」与虚拟工作通道
        计算交集体积，用来替代原来 L4/L5 整体椎体的交集体积。
        
        Args:
            segmentation: 分割数据数组 (H, W, D)
            label: 当前椎体标签（例如 L4/L5 的标签值）
            posterior_ratio: 后部区域比例（默认 0.3，表示后 30%）
            superior_ratio: 上部区域比例（默认 0.33，表示上 1/3）
            inferior_ratio: 下部区域比例（默认 0.33，表示下 1/3）
            
        Returns:
            facet_joint_mask: 上下关节突合并区域的布尔掩码 (H, W, D)
        """
        # 获取当前标签的掩码
        current_mask = (segmentation == label)
        
        if not np.any(current_mask):
            return np.zeros_like(current_mask, dtype=bool)
        
        # 获取当前标签的所有体素坐标
        indices = np.where(current_mask)
        coords = np.array([indices[0], indices[1], indices[2]]).T

        # ---------- 后部区域（posterior） ----------
        y_coords = coords[:, 1]
        y_min = np.min(y_coords)
        y_max = np.max(y_coords)
        # 后部区域：Y 方向的后 posterior_ratio 比例
        y_threshold = y_min + (y_max - y_min) * (1 - posterior_ratio)

        # ---------- 上关节突（椎体上 1/3 + 后部） ----------
        z_coords = coords[:, 2]
        z_min = np.min(z_coords)
        z_max = np.max(z_coords)
        # 上部区域：Z 方向的上 superior_ratio 比例
        z_superior_threshold = z_min + (z_max - z_min) * (1 - superior_ratio)

        superior_indices = coords[
            (coords[:, 1] <= y_threshold) &      # 后部
            (coords[:, 2] >= z_superior_threshold)  # 上部
        ]
        superior_mask = np.zeros_like(current_mask, dtype=bool)
        if len(superior_indices) > 0:
            superior_mask[superior_indices[:, 0],
                          superior_indices[:, 1],
                          superior_indices[:, 2]] = True

        # ---------- 下关节突（椎体下 1/3 + 后部） ----------
        z_inferior_threshold = z_min + (z_max - z_min) * inferior_ratio

        inferior_indices = coords[
            (coords[:, 1] <= y_threshold) &      # 后部
            (coords[:, 2] <= z_inferior_threshold)  # 下部
        ]
        inferior_mask = np.zeros_like(current_mask, dtype=bool)
        if len(inferior_indices) > 0:
            inferior_mask[inferior_indices[:, 0],
                          inferior_indices[:, 1],
                          inferior_indices[:, 2]] = True

        # ---------- 合并上下关节突 ----------
        facet_joint_mask = superior_mask | inferior_mask
        return facet_joint_mask
    
    def compute_intersection_volume(self,
                                   segmentation: np.ndarray,
                                   cylinder_points_voxel: np.ndarray,
                                   voxel_sizes: np.ndarray,
                                   label_values: Optional[List[int]] = None,
                                   use_facet_joint_only: bool = True) -> Dict[int, float]:
        """
        计算圆柱体与分割数据的交集体积
        
        Args:
            segmentation: 分割数据数组 (H, W, D)
            cylinder_points_voxel: 圆柱体内的体素坐标点 (N, 3)
            voxel_sizes: 体素大小 (3,)，单位：毫米
            label_values: 要统计的标签值列表，如果为None则统计所有标签
            use_facet_joint_only: 对于关节突标签（默认 1-6，见 facet_joint_labels），是否只计算上下关节突区域（默认True）；其余标签按完整结构统计
            
        Returns:
            intersection_volumes: 字典，键为标签值，值为交集体积（立方毫米）
        """
        h, w, d = segmentation.shape
        
        # 将体素坐标转换为整数索引
        indices = np.round(cylinder_points_voxel).astype(int)
        
        # 过滤掉超出边界的点
        valid_mask = (
            (indices[:, 0] >= 0) & (indices[:, 0] < h) &
            (indices[:, 1] >= 0) & (indices[:, 1] < w) &
            (indices[:, 2] >= 0) & (indices[:, 2] < d)
        )
        
        valid_indices = indices[valid_mask]
        
        if len(valid_indices) == 0:
            return {}
        
        # 对于「关节突标签」，使用缓存的上下关节突区域掩码（如果可用）；其余标签用完整结构
        facet_joint_masks: Dict[int, np.ndarray] = {}
        if use_facet_joint_only:
            # 关节突标签列表（默认由 plan_paths 设为 [1,2,3,4,5,6]，未设置时兼容旧逻辑仅 L4/L5）
            target_labels = getattr(
                self, "_facet_joint_labels",
                [getattr(self, "_l4_label", 4), getattr(self, "_l5_label", 5)]
            )

            # 检查缓存是否有效（分割数据形状是否匹配）
            if (self._facet_joint_masks_cache is not None and
                self._cached_segmentation_shape == segmentation.shape):
                facet_joint_masks = self._facet_joint_masks_cache
            else:
                print("正在计算关节突标签上下关节突区域掩码（仅需计算一次）...")
                for lbl in target_labels:
                    if label_values is None or lbl in label_values:
                        facet_joint_masks[lbl] = self.extract_facet_joint_region(
                            segmentation, lbl
                        )
                self._facet_joint_masks_cache = facet_joint_masks
                self._cached_segmentation_shape = segmentation.shape
                print("关节突掩码已缓存")
        
        # 去重：同一个体素可能被多次采样
        # 使用字典记录每个体素对应的标签
        voxel_label_map = {}
        for idx in valid_indices:
            key = tuple(idx)
            if key not in voxel_label_map:
                voxel_label = segmentation[idx[0], idx[1], idx[2]]
                voxel_label_map[key] = voxel_label
        
        # 统计每个标签的体素数
        label_counts: Dict[int, int] = {}
        for idx_tuple, label in voxel_label_map.items():
            label_int = int(label)

            # 对于 L4 / L5，只统计上下关节突区域的体素
            if use_facet_joint_only and label_int in facet_joint_masks:
                facet_mask = facet_joint_masks[label_int]
                if facet_mask[idx_tuple[0], idx_tuple[1], idx_tuple[2]]:
                    label_counts[label_int] = label_counts.get(label_int, 0) + 1
            else:
                # 其他标签正常统计
                if label_values is None or label_int in label_values:
                    label_counts[label_int] = label_counts.get(label_int, 0) + 1
        
        # 计算体素体积（立方毫米）
        voxel_volume = np.prod(voxel_sizes)
        
        # 计算交集体积
        intersection_volumes = {}
        for label, count in label_counts.items():
            if label_values is None or label in label_values:
                intersection_volumes[label] = count * voxel_volume
        
        return intersection_volumes
    
    def _physical_x_mid_from_volume(
            self, affine: np.ndarray, segmentation_shape: Tuple[int, ...]) -> float:
        corners = np.array([
            [0, 0, 0],
            [segmentation_shape[0] - 1, 0, 0],
            [0, segmentation_shape[1] - 1, 0],
            [0, 0, segmentation_shape[2] - 1],
            [segmentation_shape[0] - 1, segmentation_shape[1] - 1, 0],
            [segmentation_shape[0] - 1, 0, segmentation_shape[2] - 1],
            [0, segmentation_shape[1] - 1, segmentation_shape[2] - 1],
            [segmentation_shape[0] - 1, segmentation_shape[1] - 1, segmentation_shape[2] - 1],
        ], dtype=float)
        pc = self.voxel_to_physical(corners, affine)
        return float(np.mean(pc[:, 0]))
    
    def resolve_paraspinal_side(
            self, facet_side: str, target_point: np.ndarray,
            affine: np.ndarray, segmentation_shape: Tuple[int, ...]) -> str:
        """
        返回 'left' | 'right'，用于同侧椎旁肌(10–13 vs 14–17)与髂骨(19 vs 18)的筛选权重。
        auto：与入路 dX 判定一致（靶点 X 相对体数据包围盒中线）。
        """
        fs = (facet_side or 'auto').strip().lower()
        if fs == 'right':
            return 'right'
        if fs == 'left':
            return 'left'
        if fs == 'both':
            return 'left'
        tp = np.asarray(target_point, dtype=float).reshape(-1)[:3]
        xm = self._physical_x_mid_from_volume(affine, segmentation_shape)
        return 'right' if float(tp[0]) >= xm else 'left'
    
    def _lateral_dx_sign_from_facet_side(
            self,
            facet_side: str,
            target_point: np.ndarray,
            affine: np.ndarray,
            segmentation_shape: Tuple[int, ...]) -> float:
        """
        与 --facet_side 一致，决定横断面内通道方向 X 分量的符号（RAS：+X 为患者右侧）。
        右侧入路：起点应在靶点更靠 +X 的后外侧，故从起点指向靶点的方向 dx<0，返回 -1。
        左侧入路：dx>0，返回 +1。auto：按靶点 X 与扫描体素包围盒中心 X 判定偏左/偏右。
        """
        fs = (facet_side or 'auto').strip().lower()
        if fs == 'right':
            return -1.0
        if fs == 'left':
            return 1.0
        if fs == 'both':
            return 1.0
        if fs != 'auto':
            return 1.0
        x_mid = self._physical_x_mid_from_volume(affine, segmentation_shape)
        return -1.0 if float(target_point[0]) >= x_mid else 1.0
    
    def direction_from_angles(self, 
                             angle_coronal_deg: float,
                             angle_transverse_deg: float,
                             lateral_dx_sign: float = 1.0) -> np.ndarray:
        """
        根据与冠状面和横断面的夹角计算方向向量
        
        在RAS坐标系中：
        - 冠状面法向量：[0, 1, 0]（垂直于A轴，即前后方向）
        - 横断面法向量：[0, 0, 1]（垂直于S轴，即上下方向）
        
        方向向量需要满足：
        - sin(angle_coronal) = |direction · [0, 1, 0]| = |dy|
        - sin(angle_transverse) = |direction · [0, 0, 1]| = |dz|
        - ||direction|| = 1
        
        对于通道方向，通常是从后向前、从上向下，所以：
        - dy > 0（向前，从后到前）
        - dz < 0（向下，从上到下）
        - dx 符号由 lateral_dx_sign 决定：RAS 下 +X 为患者右侧时，右侧入路取 -1（起点在靶点更靠 +X 侧）
        
        Args:
            angle_coronal_deg: 与冠状面的夹角（度）
            angle_transverse_deg: 与横断面的夹角（度）
            lateral_dx_sign: +1 或 -1，与 --facet_side left/right 一致
            
        Returns:
            direction: 归一化的方向向量 (3,)
        """
        # 转换为弧度
        angle_coronal_rad = np.radians(angle_coronal_deg)
        angle_transverse_rad = np.radians(angle_transverse_deg)
        
        # 计算方向向量的分量
        # dy = sin(angle_coronal)（向前，正值）
        # dz = -sin(angle_transverse)（向下，负值）
        # dx = ±sqrt(1 - dy^2 - dz^2)，符号由 lateral_dx_sign 与 facet_side 一致
        
        dy = np.sin(angle_coronal_rad)
        dz = -np.sin(angle_transverse_rad)  # 负值表示向下
        
        # 计算dx，确保向量归一化；sin²(CPA)+sin²(CSA)>1 时无一方向可同时满足 |dy|=sin(CPA)、|dz|=sin(CSA)
        dx_squared = 1.0 - dy**2 - dz**2
        
        if dx_squared < 0:
            raise ValueError(
                "角度组合几何不可行: sin²(CPA)+sin²(CSA)="
                f"{np.sin(angle_coronal_rad) ** 2 + np.sin(angle_transverse_rad) ** 2:.6f} > 1"
            )
        dx = float(lateral_dx_sign) * np.sqrt(dx_squared)
        
        # 构建方向向量
        direction = np.array([dx, dy, dz])
        
        # 归一化（确保单位向量）
        norm = np.linalg.norm(direction)
        if norm > 1e-10:
            direction = direction / norm
        else:
            # 如果向量太小，使用默认方向（向下向前），X 分量与入路侧一致
            direction = np.array([float(lateral_dx_sign), 1.0, -1.0])
            direction = direction / np.linalg.norm(direction)
        
        return direction

    def direction_from_spherical_alpha_beta(
        self,
        alpha_deg: float,
        beta_deg: float,
        lateral_dx_sign: float,
    ) -> np.ndarray:
        """
        球面角参数化（单位方向，RAS）：α 为 XY 平面内从 +X 轴起的方位角；β 为相对 XY 平面的仰角。

            dx = cos(β) cos(α),  dy = cos(β) sin(α),  dz = sin(β)

        再对 dx 乘以 lateral_dx_sign，与入路左/右一致。

        网格扫描时 α、β 由命令行范围与步长驱动；球面模式下结果 JSON 中
        target_angle_coronal_deg / angle_with_coronal_deg 存 **α**，
        target_angle_transverse_deg / angle_with_transverse_deg 存 **β**。
        若需与冠状面/横断面夹角（CPA/CSA），请对返回向量自行调用
        calculate_angle_with_coronal_plane / calculate_angle_with_transverse_plane。

        约定：将用户输入的 β 取相反数代入上式，使常用正角度区间（如 0°–60°）下针向倾向 −Z（足侧），
        与原通道 dz 多为负的习惯一致；论文中可对 β 另行命名说明。
        """
        a = np.radians(float(alpha_deg))
        b = np.radians(-float(beta_deg))
        ca, sa = np.cos(a), np.sin(a)
        cb, sb = np.cos(b), np.sin(b)
        dx = cb * ca
        dy = cb * sa
        dz = sb
        direction = np.array([dx, dy, dz], dtype=float)
        direction[0] *= float(lateral_dx_sign)
        nrm = np.linalg.norm(direction)
        if nrm > 1e-12:
            direction = direction / nrm
        else:
            direction = np.array([float(lateral_dx_sign), 0.0, -1.0], dtype=float)
            direction = direction / np.linalg.norm(direction)
        return direction

    def generate_channel_from_direction(
        self,
        target_point: np.ndarray,
        direction: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """固定靶点，沿单位方向（起点→终点）生成通道。"""
        d = np.asarray(direction, dtype=float).reshape(3)
        d = d / (np.linalg.norm(d) + 1e-12)
        end_point = np.array(target_point, dtype=float).reshape(3)
        start_point = end_point - d * self.channel_length_mm
        return start_point, end_point, d

    def generate_channel_from_angles(self,
                                    target_point: np.ndarray,
                                    angle_coronal_deg: float,
                                    angle_transverse_deg: float,
                                    lateral_dx_sign: float = 1.0) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        根据指定角度生成通道（固定终点，根据角度计算起点和方向）
        
        Args:
            target_point: 固定的穿刺靶点（终点）物理坐标 (3,)
            angle_coronal_deg: 与冠状面的夹角（度）
            angle_transverse_deg: 与横断面的夹角（度）
            
        Returns:
            start_point: 起点（体表端）物理坐标 (3,)
            end_point: 终点（靶点）物理坐标 (3,)
            direction: 方向向量 (3,)，从起点指向终点
        """
        # 根据角度计算方向向量
        direction = self.direction_from_angles(
            angle_coronal_deg, angle_transverse_deg, lateral_dx_sign=lateral_dx_sign)
        
        # 终点是固定的靶点
        end_point = np.array(target_point)
        
        # 从终点反推起点：起点 = 终点 - direction * length
        # 注意：direction是从起点指向终点的，所以起点在终点反方向
        start_point = end_point - direction * self.channel_length_mm
        
        return start_point, end_point, direction
    
    def generate_random_channel(self,
                               target_point: np.ndarray,
                               segmentation_shape: Tuple[int, int, int],
                               affine: np.ndarray,
                               margin_mm: float = 10.0,
                               lateral_dx_sign: Optional[float] = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        随机生成一个通道（固定终点，随机起点和方向）
        
        Args:
            target_point: 固定的穿刺靶点（终点）物理坐标 (3,)
            segmentation_shape: 分割数据形状 (H, W, D)
            affine: 仿射变换矩阵
            margin_mm: 边界留白（毫米），确保通道起点不在边界上
            
        Returns:
            start_point: 起点（体表端）物理坐标 (3,)
            end_point: 终点（靶点）物理坐标 (3,)
            direction: 方向向量 (3,)，从起点指向终点
        """
        # 获取数据边界（物理坐标）
        corners_voxel = np.array([
            [0, 0, 0],
            [segmentation_shape[0]-1, 0, 0],
            [0, segmentation_shape[1]-1, 0],
            [0, 0, segmentation_shape[2]-1],
            [segmentation_shape[0]-1, segmentation_shape[1]-1, 0],
            [segmentation_shape[0]-1, 0, segmentation_shape[2]-1],
            [0, segmentation_shape[1]-1, segmentation_shape[2]-1],
            [segmentation_shape[0]-1, segmentation_shape[1]-1, segmentation_shape[2]-1]
        ])
        
        # 转换为物理坐标
        corners_physical = self.voxel_to_physical(corners_voxel, affine)
        
        # 计算边界框
        min_coords = corners_physical.min(axis=0) + margin_mm
        max_coords = corners_physical.max(axis=0) - margin_mm
        
        # 确保有足够的空间
        if np.any(max_coords <= min_coords):
            raise ValueError("数据空间太小，无法生成通道")
        
        # 终点是固定的靶点
        end_point = np.array(target_point)
        
        # 随机生成方向（单位向量）
        # 使用球面均匀分布生成随机方向
        # 方法：生成标准正态分布然后归一化
        direction = np.random.randn(3)
        direction = direction / (np.linalg.norm(direction) + 1e-12)
        if lateral_dx_sign is not None and abs(float(lateral_dx_sign)) > 1e-6:
            sgn = float(lateral_dx_sign)
            if sgn < 0 and direction[0] > 1e-9:
                direction[0] *= -1.0
            elif sgn > 0 and direction[0] < -1e-9:
                direction[0] *= -1.0
            direction = direction / (np.linalg.norm(direction) + 1e-12)
        
        # 从终点反推起点：起点 = 终点 - direction * length
        # 注意：direction是从起点指向终点的，所以起点在终点反方向
        start_point = end_point - direction * self.channel_length_mm
        
        # 检查起点是否在数据边界内（如果不在，可能需要调整方向或跳过）
        # 这里我们允许起点稍微超出边界，但会在后续处理中检查
        
        return start_point, end_point, direction
    
    def voxel_to_physical(self, voxel_coords: np.ndarray, 
                         affine: np.ndarray) -> np.ndarray:
        """
        将体素坐标转换为物理坐标
        
        Args:
            voxel_coords: 体素坐标 (N, 3)
            affine: 仿射变换矩阵
            
        Returns:
            physical_coords: 物理坐标 (N, 3)
        """
        # 转换为齐次坐标
        if voxel_coords.ndim == 1:
            voxel_coords = voxel_coords.reshape(1, -1)
        
        ones = np.ones((voxel_coords.shape[0], 1))
        voxel_homogeneous = np.hstack([voxel_coords, ones])
        
        # 应用仿射变换
        physical_coords = (affine @ voxel_homogeneous.T).T[:, :3]
        
        if physical_coords.shape[0] == 1:
            return physical_coords[0]
        return physical_coords
    
    @staticmethod
    def _world_y_axis_is_anterior_positive(affine: np.ndarray) -> bool:
        """
        判断世界坐标 +Y 是否指向患者前方（解剖 A）。
        RAS（+Y=A）：True；不少 CT/NIfTI 为 LPS（+Y=P）：False。
        若搞反，会把椎体「前方」误当后缘，靶点落在椎体内部。
        """
        try:
            codes = aff2axcodes(affine)
            if len(codes) > 1:
                if codes[1] == 'A':
                    return True
                if codes[1] == 'P':
                    return False
        except Exception:
            pass
        return True
    
    def _posterior_y_on_mask_strip(self, pts: np.ndarray, mid_x: float,
                                   affine: np.ndarray, fallback_y: float) -> float:
        """椎间盘带内、邻近中线处椎体后缘的 Y（窄冠状窗 + 最靠后分位，减少取到椎体内部）。"""
        if len(pts) == 0:
            return float(fallback_y)
        spread = float(np.std(pts[:, 0]))
        band = max(spread * PLANNING.band_spread_frac, PLANNING.band_min_mm)
        central = pts[np.abs(pts[:, 0] - mid_x) < band]
        if len(central) < 5:
            central = pts
        y_ap = self._world_y_axis_is_anterior_positive(affine)
        yv = central[:, 1]
        if y_ap:
            # 窄带内取偏前分位，对应椎体后壁而非棘突最尖端（避免 min 取到棘突尾）
            return float(np.percentile(yv, 22.0))
        return float(np.percentile(yv, 78.0))
    
    def _filter_physical_posterior_band(
            self, phys: np.ndarray, affine: np.ndarray, frac: float) -> np.ndarray:
        """按世界坐标 Y 保留解剖学后方约 frac 宽度的点（关节突粗筛）。"""
        if len(phys) == 0:
            return phys
        yp = phys[:, 1]
        ymn, ymx = float(np.min(yp)), float(np.max(yp))
        span = max(ymx - ymn, 1e-3)
        if self._world_y_axis_is_anterior_positive(affine):
            thr = ymn + span * (1.0 - frac)
            return phys[phys[:, 1] <= thr]
        thr = ymn + span * frac
        return phys[phys[:, 1] >= thr]
    
    def _posterior_row_index(self, physical_coords: np.ndarray, affine: np.ndarray) -> int:
        """在给定体素集合中取最靠解剖学后方的一行的索引。"""
        y = physical_coords[:, 1]
        if self._world_y_axis_is_anterior_positive(affine):
            return int(np.argmin(y))
        return int(np.argmax(y))
    
    def _posterior_y_quantile_indices(self, y_values: np.ndarray, affine: np.ndarray,
                                      posterior_ratio: float) -> np.ndarray:
        """取 Y 上最靠后的一段体素（比例 posterior_ratio）的索引。"""
        n = len(y_values)
        if n == 0:
            return np.array([], dtype=np.int64)
        k = max(1, int(n * posterior_ratio))
        order = np.argsort(y_values)
        if self._world_y_axis_is_anterior_positive(affine):
            return order[:k]
        return order[-k:]
    
    def _x_from_facet_masks_at_disc(
            self,
            segmentation: np.ndarray,
            affine: np.ndarray,
            vertebral_labels: List[int],
            disc_z_mid: float,
            z_span: float,
            split_x: float,
            facet_side: str,
            posterior_y_ref: float) -> Optional[float]:
        """
        在椎间隙高度附近，用 extract_facet_joint_region 得到的关节突体素估计 X。
        内侧缘与外侧缘之间偏外侧插值，避免靶点落在椎体正中。
        """
        z_tol = max(z_span * PLANNING.z_tol_band_frac, PLANNING.z_tol_min_mm)
        # 关节突在后方，靶点 Y 前移后常与掩码 Y 相差 >14mm；过小会筛空掩码、x_facet=None
        y_tol = 48.0
        chunks: List[np.ndarray] = []
        for lb in vertebral_labels:
            fac = self.extract_facet_joint_region(segmentation, lb)
            if not np.any(fac):
                continue
            ii = np.where(fac)
            vox = np.array([ii[0], ii[1], ii[2]]).T
            phys = self.voxel_to_physical(vox.astype(np.float64), affine)
            phys = self._filter_physical_posterior_band(phys, affine, PLANNING.posterior_band_frac)
            if len(phys) == 0:
                continue
            m = np.abs(phys[:, 2] - disc_z_mid) <= z_tol
            phys = phys[m]
            m2 = np.abs(phys[:, 1] - posterior_y_ref) <= y_tol
            phys = phys[m2]
            if len(phys) > 0:
                chunks.append(phys)
        if not chunks:
            for yw in (56.0, 72.0):
                chunks = []
                for lb in vertebral_labels:
                    fac = self.extract_facet_joint_region(segmentation, lb)
                    if not np.any(fac):
                        continue
                    ii = np.where(fac)
                    vox = np.array([ii[0], ii[1], ii[2]]).T
                    phys = self.voxel_to_physical(vox.astype(np.float64), affine)
                    phys = self._filter_physical_posterior_band(phys, affine, PLANNING.posterior_band_frac)
                    if len(phys) == 0:
                        continue
                    m = np.abs(phys[:, 2] - disc_z_mid) <= z_tol * 1.25
                    phys = phys[m]
                    m2 = np.abs(phys[:, 1] - posterior_y_ref) <= yw
                    phys = phys[m2]
                    if len(phys) > 0:
                        chunks.append(phys)
                if chunks:
                    break
        if not chunks:
            return None
        fc = np.vstack(chunks)
        fs = (facet_side or 'auto').strip().lower()
        med_lat = PLANNING.med_lat
        lat_w = 1.0 - med_lat
        
        def _blend_side(coords: np.ndarray, left_side: bool) -> Optional[float]:
            if len(coords) < 2:
                return float(coords[0, 0]) if len(coords) == 1 else None
            if left_side:
                mx = float(np.max(coords[:, 0]))
                lx = float(np.min(coords[:, 0]))
                return med_lat * mx + lat_w * lx
            mx = float(np.min(coords[:, 0]))
            lx = float(np.max(coords[:, 0]))
            return med_lat * mx + lat_w * lx
        
        if fs == 'left':
            L = fc[fc[:, 0] < split_x]
            v = _blend_side(L, True)
            return v
        if fs == 'right':
            R = fc[fc[:, 0] >= split_x]
            v = _blend_side(R, False)
            return v
        L = fc[fc[:, 0] < split_x]
        R = fc[fc[:, 0] >= split_x]
        if len(L) < 2 or len(R) < 2:
            return None
        xl = _blend_side(L, True)
        xr = _blend_side(R, False)
        if xl is None or xr is None:
            return None
        return xl if abs(xl - split_x) >= abs(xr - split_x) else xr
    
    def _foramen_lateral_shell_x(
            self,
            combined_disc_coords: np.ndarray,
            split_x: float,
            facet_side: str,
            ) -> float:
        """
        无关节突掩码时：在椎间盘壳上取入路侧「靠外缘」的 X 参考，用于椎间孔冠状插值。
        """
        fs = (facet_side or 'auto').strip().lower()
        xs = combined_disc_coords[:, 0]
        if fs == 'left':
            side = combined_disc_coords[xs < split_x]
            if len(side) < 3:
                side = combined_disc_coords
            return float(np.percentile(side[:, 0], 10.0))
        if fs == 'right':
            side = combined_disc_coords[xs >= split_x]
            if len(side) < 3:
                side = combined_disc_coords
            return float(np.percentile(side[:, 0], 90.0))
        L = combined_disc_coords[xs < split_x]
        R = combined_disc_coords[xs >= split_x]
        if len(L) < 3 or len(R) < 3:
            return float(np.median(xs))
        lx = float(np.percentile(L[:, 0], 10.0))
        rx = float(np.percentile(R[:, 0], 90.0))
        return lx if abs(lx - split_x) >= abs(rx - split_x) else rx

    def _disc_level_target_posterior_and_facet(
            self,
            combined_disc_coords: np.ndarray,
            posterior_y_primary_strip: np.ndarray,
            disc_midpoint: np.ndarray,
            disc_z_mid: float,
            disc_z_min: float,
            disc_z_max: float,
            affine: np.ndarray,
            facet_side: str,
            min_primary_pts: int = 8,
            segmentation: Optional[np.ndarray] = None,
            facet_vertebra_labels: Optional[List[int]] = None,
            disc_segment: str = 'l4_l5',
            ) -> Tuple[np.ndarray, Optional[np.ndarray], Optional[np.ndarray], float]:
        """
        椎间隙层面：穿刺靶点定位于 **椎间孔（neural foramen）** 内（解剖近似，RAS mm）。

        由 L4/L5/S1 分割壳体推导，**不**再走「后缘 Y + 多段前移 + 关节突折中 + Z 偏置」长链调参。

        几何含义：
        - **Z**：椎间隙 [z_lo,z_hi] 内固定比例。z 从 z_lo→z_hi 为「L5 上缘侧→L4 下缘侧」。
          L4-L5 取 **偏 z_lo（约 0.38）** 对准椎间孔/间盘层，避免落在 L4 椎体侧缘（过高时像打到 L4 骨）。
          L5-S1 约 0.54。并带边距裁剪。
        - **冠状 X**：split_x 与同侧 x_ref 插值；`foramen_lateral_frac` 略偏小使点更靠孔区内侧（透亮窗）。
        - **矢状 Y**：壳后份与间盘质心混合；**加大质心权重**使点更靠近间盘「开窗」而非椎体后缘骨面。

        facet_side 为 left/right 时插值指向该侧关节突参考；auto 时 x_ref 取与中线距离更大的一侧。
        """
        fs = (facet_side or 'auto').strip().lower()
        if fs not in ('left', 'right', 'auto'):
            fs = 'auto'
        seg_l5s1 = (disc_segment or '').strip().lower() == 'l5_s1'
        centroid_x = float(disc_midpoint[0])
        yc = float(disc_midpoint[1])
        z_lo = float(min(disc_z_min, disc_z_max))
        z_hi = float(max(disc_z_min, disc_z_max))
        z_span = max(z_hi - z_lo, 1e-3)
        margin_z = PLANNING.margin_z_frac * z_span
        # z_lo=靠 L5 上缘/尾侧端，z_hi=靠 L4 下缘/头侧端；较小 z_frac → 更靠 z_lo，离开 L4 骨
        z_frac = PLANNING.z_frac_l5s1 if seg_l5s1 else PLANNING.z_frac_l4l5
        tz = float(z_lo + z_frac * (z_hi - z_lo))
        tz = float(np.clip(tz, z_lo + margin_z, z_hi - margin_z))

        foramen_lateral_frac = PLANNING.foramen_lateral_frac
        y_post_blend = PLANNING.y_post_blend

        if len(combined_disc_coords) == 0:
            return (
                np.array([centroid_x, yc, tz], dtype=np.float64),
                None,
                None,
                yc,
            )

        split_x = float(np.median(combined_disc_coords[:, 0]))
        y_ap = self._world_y_axis_is_anterior_positive(affine)
        ys = combined_disc_coords[:, 1]
        if y_ap:
            y_shell_post = float(np.percentile(ys, 28.0))
            y_shell_ant = float(np.percentile(ys, 72.0))
        else:
            y_shell_post = float(np.percentile(ys, 72.0))
            y_shell_ant = float(np.percentile(ys, 28.0))
        foramen_y = float(
            y_post_blend * y_shell_post + (1.0 - y_post_blend) * yc
        )
        lo, hi = sorted([y_shell_post, y_shell_ant])
        foramen_y = float(np.clip(foramen_y, lo, hi))
        # 再向间盘质心 Y 拉一次，使进针点更贴近「椎间隙透亮区」而非椎弓根/椎体后外侧骨面
        foramen_y = float(PLANNING.foramen_y_blend * foramen_y + (1.0 - PLANNING.foramen_y_blend) * yc)
        foramen_y = float(np.clip(foramen_y, lo, hi))

        y_for_facet = float(PLANNING.facet_y_blend * (foramen_y + y_shell_post))

        x_facet: Optional[float] = None
        if segmentation is not None and facet_vertebra_labels:
            x_facet = self._x_from_facet_masks_at_disc(
                segmentation, affine, list(facet_vertebra_labels),
                disc_z_mid, z_span, split_x, fs, y_for_facet,
            )
        if x_facet is None:
            x_ref = self._foramen_lateral_shell_x(
                combined_disc_coords, split_x, fs)
        else:
            x_ref = float(x_facet)

        tx = float(split_x + foramen_lateral_frac * (x_ref - split_x))
        xs_disc = combined_disc_coords[:, 0]
        x_span_disc = float(
            np.percentile(xs_disc, 93) - np.percentile(xs_disc, 7))
        dx_cap = max(PLANNING.dx_cap_min_mm, PLANNING.dx_cap_span_frac * max(x_span_disc, 1e-3))
        tx = float(np.clip(tx, split_x - dx_cap, split_x + dx_cap))

        posterior_ratio = PLANNING.posterior_ratio_facet
        n_all = len(combined_disc_coords)
        k_post = max(1, min(n_all, max(int(n_all * posterior_ratio), n_all // 4)))
        order_all = np.argsort(combined_disc_coords[:, 1])
        post_idx = order_all[:k_post] if y_ap else order_all[-k_post:]
        combined_posterior_coords = combined_disc_coords[post_idx]
        if len(combined_posterior_coords) < 12:
            combined_posterior_coords = np.array(combined_disc_coords, copy=True)

        left_medial_point: Optional[np.ndarray] = None
        right_medial_point: Optional[np.ndarray] = None
        left_coords = combined_posterior_coords[combined_posterior_coords[:, 0] < split_x]
        right_coords = combined_posterior_coords[combined_posterior_coords[:, 0] >= split_x]
        if len(left_coords) > 0 and len(right_coords) > 0:
            left_medial_point = left_coords[np.argmax(left_coords[:, 0])]
            right_medial_point = right_coords[np.argmin(right_coords[:, 0])]

        target_point = np.array([tx, foramen_y, tz], dtype=np.float64)
        seg_name = "L5-S1" if seg_l5s1 else "L4-L5"
        xfs = f"{float(x_facet):.2f}" if x_facet is not None else "—"
        print(
            f"  [{seg_name}椎间孔靶点] z_frac={z_frac:.2f} (z_lo→z_hi=L5侧→L4侧), "
            f"split_x={split_x:.2f} mm, x_facet={xfs}, x_ref={x_ref:.2f} mm, "
            f"孔区冠状插值={foramen_lateral_frac:.2f}, "
            f"target=({tx:.2f},{foramen_y:.2f},{tz:.2f}) mm (RAS)"
        )
        return target_point, left_medial_point, right_medial_point, float(foramen_y)
    
    def _l5_s1_disc_adjacent_coords(
            self,
            l5_physical_coords: np.ndarray,
            s1_physical_coords: np.ndarray
            ) -> Optional[Tuple[np.ndarray, float, float, float, np.ndarray, np.ndarray, np.ndarray]]:
        """
        L5 下缘 + S1 上缘附近的椎体表面点（椎间盘层面壳）。
        返回 (disc_midpoint, disc_z_mid, l5_inferior_z, s1_superior_z, combined, l5_strip_for_y, s1_superior_band)，
        其中 l5_strip_for_y 优先为椎间隙全 Z 窗内 L5 体素（供后缘窄带统计），失败返回 None。
        """
        l5_center = np.mean(l5_physical_coords, axis=0)
        s1_center = np.mean(s1_physical_coords, axis=0)
        disc_midpoint = (l5_center + s1_center) / 2.0
        l5_inferior_z = float(np.max(l5_physical_coords[:, 2]))
        s1_superior_z = float(np.min(s1_physical_coords[:, 2]))
        disc_z_mid = (l5_inferior_z + s1_superior_z) / 2.0
        z_span = max(l5_inferior_z - s1_superior_z, 1.0)
        z_tolerance = z_span * PLANNING.z_tolerance_frac
        
        l5_inferior_mask = (
            (l5_physical_coords[:, 2] >= l5_inferior_z - z_tolerance) &
            (l5_physical_coords[:, 2] <= l5_inferior_z))
        l5_inferior_coords = l5_physical_coords[l5_inferior_mask]
        if len(l5_inferior_coords) == 0:
            l5_inferior_mask = l5_physical_coords[:, 2] >= l5_inferior_z - z_span * PLANNING.l5_inferior_z_frac
            l5_inferior_coords = l5_physical_coords[l5_inferior_mask]
        
        s1_superior_mask = (
            (s1_physical_coords[:, 2] >= s1_superior_z) &
            (s1_physical_coords[:, 2] <= s1_superior_z + z_tolerance))
        s1_superior_coords_band = s1_physical_coords[s1_superior_mask]
        if len(s1_superior_coords_band) == 0:
            s1_superior_mask = s1_physical_coords[:, 2] <= s1_superior_z + z_span * PLANNING.s1_superior_z_frac
            s1_superior_coords_band = s1_physical_coords[s1_superior_mask]
        
        if len(l5_inferior_coords) == 0 or len(s1_superior_coords_band) == 0:
            return None
        
        combined_disc_coords = np.vstack([l5_inferior_coords, s1_superior_coords_band])
        
        z_int_lo = float(min(s1_superior_z, l5_inferior_z))
        z_int_hi = float(max(s1_superior_z, l5_inferior_z))
        m5w = (l5_physical_coords[:, 2] >= z_int_lo) & (l5_physical_coords[:, 2] <= z_int_hi)
        m6w = (s1_physical_coords[:, 2] >= z_int_lo) & (s1_physical_coords[:, 2] <= z_int_hi)
        combined_wide = np.vstack([l5_physical_coords[m5w], s1_physical_coords[m6w]])
        l5_in_disc = l5_physical_coords[m5w]
        if len(combined_wide) >= 80:
            combined_disc_coords = combined_wide
        
        if len(combined_disc_coords) < 10:
            z_tolerance = z_span * PLANNING.disc_z_tol_relaxed_frac
            l5_inferior_mask = (
                (l5_physical_coords[:, 2] >= l5_inferior_z - z_tolerance) &
                (l5_physical_coords[:, 2] <= l5_inferior_z))
            s1_superior_mask = (
                (s1_physical_coords[:, 2] >= s1_superior_z) &
                (s1_physical_coords[:, 2] <= s1_superior_z + z_tolerance))
            l5_inferior_coords = l5_physical_coords[l5_inferior_mask]
            s1_superior_coords_band = s1_physical_coords[s1_superior_mask]
            combined_disc_coords = np.vstack([l5_inferior_coords, s1_superior_coords_band])
            if len(combined_wide) >= 80:
                combined_disc_coords = combined_wide
        
        l5_strip_for_y = l5_in_disc if len(l5_in_disc) >= 8 else l5_inferior_coords
        
        return (
            disc_midpoint, disc_z_mid, l5_inferior_z, s1_superior_z,
            combined_disc_coords, l5_strip_for_y, s1_superior_coords_band)
    
    def _lateral_edge_centroid(
            self,
            pts: np.ndarray,
            split_x: float,
            facet_side: str,
            lateral_frac: float = PLANNING.lateral_frac,
            ) -> Optional[np.ndarray]:
        """
        在单侧（左：X<split_x；右：X>=split_x）内，取靠「入路侧最外缘」一带体素的质心，
        用于近似「最边上的中心点」而非全半侧质心。
        """
        fs = (facet_side or 'auto').strip().lower()
        if fs == 'left':
            side = pts[pts[:, 0] < split_x]
        elif fs == 'right':
            side = pts[pts[:, 0] >= split_x]
        else:
            return None
        if len(side) < 3:
            side = pts
        if len(side) < 2:
            return None
        x = side[:, 0]
        xmin, xmax = float(np.min(x)), float(np.max(x))
        span = xmax - xmin
        if span < 1e-4:
            return np.mean(side, axis=0)
        # lateral_frac>=1：整侧半区质心（不靠最外缘窄带），避免 X 过度外偏
        if lateral_frac >= 0.999:
            return np.mean(side, axis=0)
        if fs == 'left':
            thr = xmin + lateral_frac * span
            lat = side[side[:, 0] <= thr]
        else:
            thr = xmax - lateral_frac * span
            lat = side[side[:, 0] >= thr]
        if len(lat) < 2:
            lat = side
        return np.mean(lat, axis=0)
    
    def _ipsilateral_disc_edge_midpoint(
            self,
            upper_phys: np.ndarray,
            lower_phys: np.ndarray,
            z_upper_inferior: float,
            z_lower_superior: float,
            facet_side: str,
            split_source: Optional[np.ndarray] = None,
            lateral_frac: float = 1.0,
            ) -> Optional[np.ndarray]:
        """
        椎间隙水平：在上椎体下缘薄壳与下椎体上缘薄壳上，分别取入路侧质心
        （lateral_frac=1 为整半侧；<1 为最外缘一带，易过度外偏）。
        返回点 Z 取上下缘 Z 中点；与 legacy 融合时 Z 会以 legacy 为准。
        """
        fs = (facet_side or 'auto').strip().lower()
        if fs not in ('left', 'right'):
            return None
        z_span = abs(float(z_upper_inferior) - float(z_lower_superior))
        z_tol = max(z_span * 0.30, 4.0)
        m_u = (
            (upper_phys[:, 2] >= z_upper_inferior - z_tol)
            & (upper_phys[:, 2] <= z_upper_inferior))
        m_l = (
            (lower_phys[:, 2] >= z_lower_superior)
            & (lower_phys[:, 2] <= z_lower_superior + z_tol))
        u = upper_phys[m_u]
        v = lower_phys[m_l]
        if len(u) < 5 or len(v) < 5:
            return None
        if split_source is not None and len(split_source) >= 10:
            split_x = float(np.median(split_source[:, 0]))
        else:
            split_x = float(np.median(np.vstack([u, v])[:, 0]))
        p_up = self._lateral_edge_centroid(u, split_x, fs, lateral_frac)
        p_lo = self._lateral_edge_centroid(v, split_x, fs, lateral_frac)
        if p_up is None or p_lo is None:
            return None
        tgt = (p_up + p_lo) / 2.0
        z_mid = (float(z_upper_inferior) + float(z_lower_superior)) / 2.0
        tgt = tgt.astype(np.float64)
        tgt[2] = z_mid
        return tgt
    
    def _blend_disc_edge_with_legacy(
            self,
            edge: np.ndarray,
            legacy: np.ndarray,
            xy_edge_weight: float = PLANNING.xy_edge_weight,
            ) -> np.ndarray:
        """
        薄壳入路侧半侧中点（edge）仅作微调，避免贴骨；冠状距中线、椎间隙 Z 以主靶点为主
        （主靶点 Z 为椎间孔算法结果，与纯几何 z_mid 可能不同）。
        """
        w = float(np.clip(xy_edge_weight, 0.0, 1.0))
        out = np.asarray(legacy, dtype=np.float64).reshape(3).copy()
        out[0] = (1.0 - w) * float(legacy[0]) + w * float(edge[0])
        out[1] = (1.0 - w) * float(legacy[1]) + w * float(edge[1])
        out[2] = float(legacy[2])
        return out
    
    def detect_l4_l5_facet_joint_projection(self,
                                           segmentation: np.ndarray,
                                           affine: np.ndarray,
                                           l4_label: int = 4,
                                           l5_label: int = 5,
                                           facet_side: str = 'auto',
                                           use_disc_edge_midpoint: bool = False,
                                           ) -> Optional[np.ndarray]:
        """
        L4-L5 穿刺靶点：由椎间隙壳 + 关节突掩码估计 **椎间孔（neural foramen）** 内一点（RAS）。

        use_disc_edge_midpoint=True 且 facet_side 为 left/right：在椎间孔靶点基础上与薄壳半侧中点做轻度 XY 融合（Z 不变）。

        Args:
            segmentation: 分割数据数组 (H, W, D)
            affine: 仿射变换矩阵
            l4_label: L4椎体的标签值，默认4
            l5_label: L5椎体的标签值，默认5
            facet_side: 'left' | 'right' | 'auto'。auto 时仅用椎间孔靶点（无薄壳半侧融合）。
            use_disc_edge_midpoint: 是否优先使用椎间隙外侧半上下缘中点模式。

        Returns:
            target_point: 投影点物理坐标 (3,)，如果无法识别则返回None
        """
        facet_side = (facet_side or 'auto').strip().lower()
        if facet_side not in ('left', 'right', 'auto'):
            facet_side = 'auto'
        # 检查L4和L5是否存在
        l4_mask = (segmentation == l4_label)
        l5_mask = (segmentation == l5_label)
        
        if not np.any(l4_mask):
            print(f"警告: 未找到L4椎体（标签值 {l4_label}）")
            return None
        
        if not np.any(l5_mask):
            print(f"警告: 未找到L5椎体（标签值 {l5_label}）")
            return None
        
        # 获取L4和L5的所有体素坐标
        l4_indices = np.where(l4_mask)
        l5_indices = np.where(l5_mask)
        
        # 转换为物理坐标
        l4_voxel_coords = np.array([l4_indices[0], l4_indices[1], l4_indices[2]]).T
        l5_voxel_coords = np.array([l5_indices[0], l5_indices[1], l5_indices[2]]).T
        
        l4_physical_coords = self.voxel_to_physical(l4_voxel_coords, affine)
        l5_physical_coords = self.voxel_to_physical(l5_voxel_coords, affine)
        
        # 计算L4和L5的中心点（质心）
        l4_center = np.mean(l4_physical_coords, axis=0)
        l5_center = np.mean(l5_physical_coords, axis=0)
        
        # 计算L4-L5椎间盘中线（两个椎体中心之间的中点）
        disc_midpoint = (l4_center + l5_center) / 2.0
        
        # 找到L4的下缘和L5的上缘，确定椎间盘区域
        # 在RAS坐标系中，Z轴是上下方向（Superior），Z值越大越靠下
        l4_inferior_z = np.max(l4_physical_coords[:, 2])  # L4下缘（Z最大值）
        l5_superior_z = np.min(l5_physical_coords[:, 2])   # L5上缘（Z最小值）
        
        # 椎间盘区域的Z坐标范围（L4下缘和L5上缘之间）
        disc_z_min = l4_inferior_z
        disc_z_max = l5_superior_z
        disc_z_mid = (disc_z_min + disc_z_max) / 2.0  # 椎间盘中间位置
        
        # 找到L4下缘附近的点和L5上缘附近的点（用于确定小关节位置）
        # 使用Z坐标在椎间盘区域附近的点
        z_tolerance = (disc_z_max - disc_z_min) * PLANNING.disc_z_tol_frac  # 允许30%的容差
        
        # L4下缘附近的点（Z坐标接近L4下缘）
        l4_inferior_mask = (l4_physical_coords[:, 2] >= l4_inferior_z - z_tolerance) & \
                          (l4_physical_coords[:, 2] <= l4_inferior_z)
        
        # L5上缘附近的点（Z坐标接近L5上缘）
        l5_superior_mask = (l5_physical_coords[:, 2] >= l5_superior_z) & \
                          (l5_physical_coords[:, 2] <= l5_superior_z + z_tolerance)
        
        l4_inferior_coords = l4_physical_coords[l4_inferior_mask]
        l5_superior_coords = l5_physical_coords[l5_superior_mask]
        
        # 合并L4下缘和L5上缘附近的点（这些点位于椎间盘区域附近）
        combined_disc_coords = np.vstack([l4_inferior_coords, l5_superior_coords])
        
        # 全椎间隙 Z 窗内 L4+L5 体素：比仅下/上缘薄壳更能反映轴位外形，避免 X/Y 退化为质心
        z_int_lo = float(min(l5_superior_z, l4_inferior_z))
        z_int_hi = float(max(l5_superior_z, l4_inferior_z))
        m4w = (l4_physical_coords[:, 2] >= z_int_lo) & (l4_physical_coords[:, 2] <= z_int_hi)
        m5w = (l5_physical_coords[:, 2] >= z_int_lo) & (l5_physical_coords[:, 2] <= z_int_hi)
        combined_wide = np.vstack([l4_physical_coords[m4w], l5_physical_coords[m5w]])
        l5_in_disc = l5_physical_coords[m5w]
        if len(combined_wide) >= 80:
            combined_disc_coords = combined_wide
        
        # 如果合并后的点太少，扩大搜索范围
        if len(combined_disc_coords) < 10:
            # 扩大Z坐标范围
            z_tolerance = (disc_z_max - disc_z_min) * PLANNING.disc_z_tol_relaxed_frac
            l4_inferior_mask = (l4_physical_coords[:, 2] >= l4_inferior_z - z_tolerance) & \
                              (l4_physical_coords[:, 2] <= l4_inferior_z)
            l5_superior_mask = (l5_physical_coords[:, 2] >= l5_superior_z) & \
                              (l5_physical_coords[:, 2] <= l5_superior_z + z_tolerance)
            l4_inferior_coords = l4_physical_coords[l4_inferior_mask]
            l5_superior_coords = l5_physical_coords[l5_superior_mask]
            combined_disc_coords = np.vstack([l4_inferior_coords, l5_superior_coords])
            if len(combined_wide) >= 80:
                combined_disc_coords = combined_wide
        
        strip_for_y = l5_in_disc if len(l5_in_disc) >= 8 else l5_superior_coords
        
        split_src = combined_wide if len(combined_wide) >= 80 else combined_disc_coords
        target_point, left_medial_point, right_medial_point, posterior_y_ref = \
            self._disc_level_target_posterior_and_facet(
                combined_disc_coords, strip_for_y, disc_midpoint,
                disc_z_mid, disc_z_min, disc_z_max, affine, facet_side,
                segmentation=segmentation,
                facet_vertebra_labels=[l5_label],
                disc_segment='l4_l5')
        
        if use_disc_edge_midpoint and facet_side in ('left', 'right'):
            edge_tgt = self._ipsilateral_disc_edge_midpoint(
                l4_physical_coords, l5_physical_coords,
                l4_inferior_z, l5_superior_z, facet_side, split_source=split_src,
                lateral_frac=1.0)
            if edge_tgt is not None:
                blended = self._blend_disc_edge_with_legacy(edge_tgt, target_point)
                try:
                    _ax = aff2axcodes(affine)
                    _yax = _ax[1] if len(_ax) > 1 else '?'
                except Exception:
                    _yax = '?'
                print(f"\n自动识别L4-L5靶点（薄壳半侧中点 + 后缘/关节突融合，facet_side={facet_side}）:")
                print(f"  NIfTI 世界坐标第2轴: {_yax}（用于判断左右分割 split_x）")
                print(f"  L4中心: [{l4_center[0]:.2f}, {l4_center[1]:.2f}, {l4_center[2]:.2f}] mm")
                print(f"  L5中心: [{l5_center[0]:.2f}, {l5_center[1]:.2f}, {l5_center[2]:.2f}] mm")
                print(f"  L4下缘Z坐标: {l4_inferior_z:.2f} mm")
                print(f"  L5上缘Z坐标: {l5_superior_z:.2f} mm")
                print(f"  椎间盘Z坐标范围: [{disc_z_min:.2f}, {disc_z_max:.2f}] mm")
                print(f"  L4-L5椎间盘中线（质心连线中点，仅供参考）: [{disc_midpoint[0]:.2f}, {disc_midpoint[1]:.2f}, {disc_midpoint[2]:.2f}] mm")
                print(f"  薄壳入路侧半侧中点（参考）: [{edge_tgt[0]:.2f}, {edge_tgt[1]:.2f}, {edge_tgt[2]:.2f}] mm")
                print(f"  椎间孔主靶点（参考）: [{target_point[0]:.2f}, {target_point[1]:.2f}, {target_point[2]:.2f}] mm")
                print(f"  融合后靶点（XY 约18%薄壳，Z=主靶点）: [{blended[0]:.2f}, {blended[1]:.2f}, {blended[2]:.2f}] mm")
                print(f"  靶点Z坐标验证: {blended[2]:.2f} mm (应在 [{disc_z_min:.2f}, {disc_z_max:.2f}] 范围内)")
                if left_medial_point is not None:
                    print(f"  左侧小关节内侧缘点: [{left_medial_point[0]:.2f}, {left_medial_point[1]:.2f}, {left_medial_point[2]:.2f}] mm")
                if right_medial_point is not None:
                    print(f"  右侧小关节内侧缘点: [{right_medial_point[0]:.2f}, {right_medial_point[1]:.2f}, {right_medial_point[2]:.2f}] mm")
                return blended
        
        try:
            _ax = aff2axcodes(affine)
            _yax = _ax[1] if len(_ax) > 1 else '?'
        except Exception:
            _yax = '?'
        print(f"\n自动识别L4-L5靶点（椎间孔几何，facet_side={facet_side}）:")
        print(f"  NIfTI 世界坐标第2轴: {_yax}（用于判断后缘取 Y 最小或最大）")
        print(f"  L4中心: [{l4_center[0]:.2f}, {l4_center[1]:.2f}, {l4_center[2]:.2f}] mm")
        print(f"  L5中心: [{l5_center[0]:.2f}, {l5_center[1]:.2f}, {l5_center[2]:.2f}] mm")
        print(f"  L4下缘Z坐标: {l4_inferior_z:.2f} mm")
        print(f"  L5上缘Z坐标: {l5_superior_z:.2f} mm")
        print(f"  椎间盘Z坐标范围: [{disc_z_min:.2f}, {disc_z_max:.2f}] mm")
        print(f"  L4-L5椎间盘中线: [{disc_midpoint[0]:.2f}, {disc_midpoint[1]:.2f}, {disc_midpoint[2]:.2f}] mm")
        print(f"  矢状对应后缘深度 posterior_y_ref: {posterior_y_ref:.2f} mm（质心Y={disc_midpoint[1]:.2f} 供对比）")
        
        if left_medial_point is not None:
            print(f"  左侧小关节内侧缘点: [{left_medial_point[0]:.2f}, {left_medial_point[1]:.2f}, {left_medial_point[2]:.2f}] mm")
        if right_medial_point is not None:
            print(f"  右侧小关节内侧缘点: [{right_medial_point[0]:.2f}, {right_medial_point[1]:.2f}, {right_medial_point[2]:.2f}] mm")
        
        print(f"  识别到的靶点: [{target_point[0]:.2f}, {target_point[1]:.2f}, {target_point[2]:.2f}] mm")
        print(f"  靶点Z坐标验证: {target_point[2]:.2f} mm (应在 [{disc_z_min:.2f}, {disc_z_max:.2f}] 范围内)")
        
        return target_point
    
    def detect_l5_s1_intersection_point(self,
                                       segmentation: np.ndarray,
                                       affine: np.ndarray,
                                       l5_label: int = 5,
                                       s1_label: int = 6,
                                       facet_side: str = 'auto',
                                       use_disc_edge_midpoint: bool = False,
                                       ) -> Optional[np.ndarray]:
        """
        自动识别 L5-S1 靶点。优先在椎间隙壳上用椎间孔几何（与 L4-L5 同一套 `_disc_level_target_posterior_and_facet`）；
        若无法构建椎间盘层面点云，则回退为全椎体后缘代表点。
        """
        l5_mask = (segmentation == l5_label)
        s1_mask = (segmentation == s1_label)
        
        if not np.any(l5_mask):
            print(f"警告: 未找到L5椎体（标签值 {l5_label}）")
            return None
        
        if not np.any(s1_mask):
            print(f"警告: 未找到S1椎体（标签值 {s1_label}）")
            return None
        
        l5_indices = np.where(l5_mask)
        s1_indices = np.where(s1_mask)
        
        l5_voxel_coords = np.array([l5_indices[0], l5_indices[1], l5_indices[2]]).T
        s1_voxel_coords = np.array([s1_indices[0], s1_indices[1], s1_indices[2]]).T
        
        l5_physical_coords = self.voxel_to_physical(l5_voxel_coords, affine)
        s1_physical_coords = self.voxel_to_physical(s1_voxel_coords, affine)
        
        l5_center = np.mean(l5_physical_coords, axis=0)
        s1_center = np.mean(s1_physical_coords, axis=0)
        disc_midpoint = (l5_center + s1_center) / 2.0
        
        pack = self._l5_s1_disc_adjacent_coords(l5_physical_coords, s1_physical_coords)
        if pack is not None:
            dm, dz_mid, l5_iz, s1_sz, combined, li, ss = pack
            target_point, lm, rm, py = self._disc_level_target_posterior_and_facet(
                combined, li, dm, dz_mid, l5_iz, s1_sz, affine, facet_side,
                segmentation=segmentation,
                facet_vertebra_labels=[l5_label],
                disc_segment='l5_s1')
            if use_disc_edge_midpoint and facet_side in ('left', 'right'):
                edge_tgt = self._ipsilateral_disc_edge_midpoint(
                    l5_physical_coords, s1_physical_coords,
                    l5_iz, s1_sz, facet_side, split_source=combined,
                    lateral_frac=1.0)
                if edge_tgt is not None:
                    blended = self._blend_disc_edge_with_legacy(edge_tgt, target_point)
                    try:
                        _ax = aff2axcodes(affine)
                        _yax = _ax[1] if len(_ax) > 1 else '?'
                    except Exception:
                        _yax = '?'
                    print(f"\n自动识别L5-S1靶点（薄壳半侧中点 + 后缘/关节突融合，facet_side={facet_side}）:")
                    print(f"  NIfTI 世界坐标第2轴: {_yax}")
                    print(f"  L5中心: [{l5_center[0]:.2f}, {l5_center[1]:.2f}, {l5_center[2]:.2f}] mm")
                    print(f"  S1中心: [{s1_center[0]:.2f}, {s1_center[1]:.2f}, {s1_center[2]:.2f}] mm")
                    print(f"  椎间盘中位线（质心连线中点，仅供参考）: [{disc_midpoint[0]:.2f}, {disc_midpoint[1]:.2f}, {disc_midpoint[2]:.2f}] mm")
                    print(f"  薄壳入路侧半侧中点（参考）: [{edge_tgt[0]:.2f}, {edge_tgt[1]:.2f}, {edge_tgt[2]:.2f}] mm")
                    print(f"  椎间孔主靶点（参考）: [{target_point[0]:.2f}, {target_point[1]:.2f}, {target_point[2]:.2f}] mm")
                    print(f"  融合后靶点（XY 约18%薄壳，Z=主靶点）: [{blended[0]:.2f}, {blended[1]:.2f}, {blended[2]:.2f}] mm")
                    if lm is not None:
                        print(f"  左侧小关节内侧缘点: [{lm[0]:.2f}, {lm[1]:.2f}, {lm[2]:.2f}] mm")
                    if rm is not None:
                        print(f"  右侧小关节内侧缘点: [{rm[0]:.2f}, {rm[1]:.2f}, {rm[2]:.2f}] mm")
                    return blended
            try:
                _ax = aff2axcodes(affine)
                _yax = _ax[1] if len(_ax) > 1 else '?'
            except Exception:
                _yax = '?'
            print(f"\n自动识别L5-S1靶点（椎间隙壳 + 关节突掩码/外侧插值 X，facet_side={facet_side}）:")
            print(f"  NIfTI 世界坐标第2轴: {_yax}")
            print(f"  L5中心: [{l5_center[0]:.2f}, {l5_center[1]:.2f}, {l5_center[2]:.2f}] mm")
            print(f"  S1中心: [{s1_center[0]:.2f}, {s1_center[1]:.2f}, {s1_center[2]:.2f}] mm")
            print(f"  椎间盘中位线: [{disc_midpoint[0]:.2f}, {disc_midpoint[1]:.2f}, {disc_midpoint[2]:.2f}] mm")
            print(f"  posterior_y_ref: {py:.2f} mm（质心Y={disc_midpoint[1]:.2f}）")
            if lm is not None:
                print(f"  左侧小关节内侧缘点: [{lm[0]:.2f}, {lm[1]:.2f}, {lm[2]:.2f}] mm")
            if rm is not None:
                print(f"  右侧小关节内侧缘点: [{rm[0]:.2f}, {rm[1]:.2f}, {rm[2]:.2f}] mm")
            print(f"  识别到的靶点: [{target_point[0]:.2f}, {target_point[1]:.2f}, {target_point[2]:.2f}] mm")
            return target_point
        
        l5_posterior_idx = self._posterior_row_index(l5_physical_coords, affine)
        s1_posterior_idx = self._posterior_row_index(s1_physical_coords, affine)
        l5_posterior_point = l5_physical_coords[l5_posterior_idx]
        s1_posterior_point = s1_physical_coords[s1_posterior_idx]
        posterior_midpoint = (l5_posterior_point + s1_posterior_point) / 2.0
        target_point = np.array([
            posterior_midpoint[0], posterior_midpoint[1], posterior_midpoint[2]])
        
        print(f"\n自动识别L5-S1交叉点（回退：全椎体后缘中点，未用椎间隙壳）:")
        print(f"  L5中心: [{l5_center[0]:.2f}, {l5_center[1]:.2f}, {l5_center[2]:.2f}] mm")
        print(f"  S1中心: [{s1_center[0]:.2f}, {s1_center[1]:.2f}, {s1_center[2]:.2f}] mm")
        print(f"  椎间盘中位线: [{disc_midpoint[0]:.2f}, {disc_midpoint[1]:.2f}, {disc_midpoint[2]:.2f}] mm")
        print(f"  L5后缘点: [{l5_posterior_point[0]:.2f}, {l5_posterior_point[1]:.2f}, {l5_posterior_point[2]:.2f}] mm")
        print(f"  S1后缘点: [{s1_posterior_point[0]:.2f}, {s1_posterior_point[1]:.2f}, {s1_posterior_point[2]:.2f}] mm")
        print(f"  识别到的交叉点: [{target_point[0]:.2f}, {target_point[1]:.2f}, {target_point[2]:.2f}] mm")
        
        return target_point
    
    def detect_l5_s1_disc_posterior_intersection(self,
                                                segmentation: np.ndarray,
                                                affine: np.ndarray,
                                                l5_label: int = 5,
                                                s1_label: int = 6,
                                                facet_side: str = 'auto',
                                                use_disc_edge_midpoint: bool = False,
                                                ) -> Optional[np.ndarray]:
        """
        L5-S1 椎间盘水平穿刺靶点：与 L4-L5 相同——椎间隙壳上估计椎间孔内靶点（RAS）。
        """
        l5_mask = (segmentation == l5_label)
        s1_mask = (segmentation == s1_label)
        
        if not np.any(l5_mask):
            print(f"警告: 未找到L5椎体（标签值 {l5_label}）")
            return None
        if not np.any(s1_mask):
            print(f"警告: 未找到S1椎体（标签值 {s1_label}）")
            return None
        
        l5_indices = np.where(l5_mask)
        s1_indices = np.where(s1_mask)
        l5_voxel_coords = np.array([l5_indices[0], l5_indices[1], l5_indices[2]]).T
        s1_voxel_coords = np.array([s1_indices[0], s1_indices[1], s1_indices[2]]).T
        l5_physical_coords = self.voxel_to_physical(l5_voxel_coords, affine)
        s1_physical_coords = self.voxel_to_physical(s1_voxel_coords, affine)
        
        l5_center = np.mean(l5_physical_coords, axis=0)
        s1_center = np.mean(s1_physical_coords, axis=0)
        disc_midpoint = (l5_center + s1_center) / 2.0
        
        pack = self._l5_s1_disc_adjacent_coords(l5_physical_coords, s1_physical_coords)
        if pack is None:
            print("警告: 无法在椎间盘层面找到L5下缘或S1上缘点，回退到L5-S1后缘交叉点")
            return self.detect_l5_s1_intersection_point(
                segmentation, affine, l5_label, s1_label, facet_side=facet_side,
                use_disc_edge_midpoint=use_disc_edge_midpoint)
        
        dm, dz_mid, l5_iz, s1_sz, combined, li, ss = pack
        target_point, lm, rm, py = self._disc_level_target_posterior_and_facet(
            combined, li, dm, dz_mid, l5_iz, s1_sz, affine, facet_side,
            segmentation=segmentation,
            facet_vertebra_labels=[l5_label],
            disc_segment='l5_s1')
        if use_disc_edge_midpoint and facet_side in ('left', 'right'):
            edge_tgt = self._ipsilateral_disc_edge_midpoint(
                l5_physical_coords, s1_physical_coords,
                l5_iz, s1_sz, facet_side, split_source=combined,
                lateral_frac=1.0)
            if edge_tgt is not None:
                blended = self._blend_disc_edge_with_legacy(edge_tgt, target_point)
                try:
                    _ax = aff2axcodes(affine)
                    _yax = _ax[1] if len(_ax) > 1 else '?'
                except Exception:
                    _yax = '?'
                print(f"\n自动识别L5-S1椎间盘水平靶点（薄壳半侧中点 + 后缘/关节突融合，facet_side={facet_side}）:")
                print(f"  NIfTI 世界坐标第2轴: {_yax}")
                print(f"  L5中心: [{l5_center[0]:.2f}, {l5_center[1]:.2f}, {l5_center[2]:.2f}] mm")
                print(f"  S1中心: [{s1_center[0]:.2f}, {s1_center[1]:.2f}, {s1_center[2]:.2f}] mm")
                print(f"  L5下缘Z: {l5_iz:.2f} mm, S1上缘Z: {s1_sz:.2f} mm, 椎间隙中位Z: {dz_mid:.2f} mm")
                print(f"  椎间盘中位线（质心连线中点，仅供参考）: [{disc_midpoint[0]:.2f}, {disc_midpoint[1]:.2f}, {disc_midpoint[2]:.2f}] mm")
                print(f"  薄壳入路侧半侧中点（参考）: [{edge_tgt[0]:.2f}, {edge_tgt[1]:.2f}, {edge_tgt[2]:.2f}] mm")
                print(f"  椎间孔主靶点（参考）: [{target_point[0]:.2f}, {target_point[1]:.2f}, {target_point[2]:.2f}] mm")
                print(f"  融合后靶点（XY 约18%薄壳，Z=主靶点）: [{blended[0]:.2f}, {blended[1]:.2f}, {blended[2]:.2f}] mm")
                if lm is not None:
                    print(f"  左侧小关节内侧缘点: [{lm[0]:.2f}, {lm[1]:.2f}, {lm[2]:.2f}] mm")
                if rm is not None:
                    print(f"  右侧小关节内侧缘点: [{rm[0]:.2f}, {rm[1]:.2f}, {rm[2]:.2f}] mm")
                return blended
        
        try:
            _ax = aff2axcodes(affine)
            _yax = _ax[1] if len(_ax) > 1 else '?'
        except Exception:
            _yax = '?'
        print(f"\n自动识别L5-S1椎间盘水平靶点（后缘深度 + 关节突掩码/外侧插值 X，facet_side={facet_side}）:")
        print(f"  NIfTI 世界坐标第2轴: {_yax}")
        print(f"  L5中心: [{l5_center[0]:.2f}, {l5_center[1]:.2f}, {l5_center[2]:.2f}] mm")
        print(f"  S1中心: [{s1_center[0]:.2f}, {s1_center[1]:.2f}, {s1_center[2]:.2f}] mm")
        print(f"  L5下缘Z: {l5_iz:.2f} mm, S1上缘Z: {s1_sz:.2f} mm, 椎间隙中位Z: {dz_mid:.2f} mm")
        print(f"  椎间盘中位线: [{disc_midpoint[0]:.2f}, {disc_midpoint[1]:.2f}, {disc_midpoint[2]:.2f}] mm")
        print(f"  posterior_y_ref: {py:.2f} mm（质心Y={disc_midpoint[1]:.2f}）")
        if lm is not None:
            print(f"  左侧小关节内侧缘点: [{lm[0]:.2f}, {lm[1]:.2f}, {lm[2]:.2f}] mm")
        if rm is not None:
            print(f"  右侧小关节内侧缘点: [{rm[0]:.2f}, {rm[1]:.2f}, {rm[2]:.2f}] mm")
        print(f"  识别到的靶点: [{target_point[0]:.2f}, {target_point[1]:.2f}, {target_point[2]:.2f}] mm")
        
        return target_point

    @staticmethod
    def _puncture_stem_key(
        detect_facet_joint: bool,
        detect_l5_s1_disc: bool,
        auto_detect_target: bool,
        facet_side: str,
        end_points_ras: Dict,
    ) -> Optional[str]:
        if detect_facet_joint:
            prefix = "l4_l5"
        elif detect_l5_s1_disc or auto_detect_target:
            prefix = "l5_s1"
        else:
            return None
        fs = (facet_side or "auto").strip().lower()
        kL, kR = f"{prefix}_left", f"{prefix}_right"
        if fs == "left":
            return kL
        if fs == "right":
            return kR
        if kL not in end_points_ras or kR not in end_points_ras:
            return kL if kL in end_points_ras else kR
        pL = np.asarray(end_points_ras[kL], dtype=float)
        pR = np.asarray(end_points_ras[kR], dtype=float)
        return kL if abs(pL[0]) >= abs(pR[0]) else kR

    def run_puncture_target_inference(
        self,
        ct_path: str,
        mask_path: str,
        checkpoint_path: str,
        detect_facet_joint: bool,
        detect_l5_s1_disc: bool,
        auto_detect_target: bool,
        facet_side: str,
        puncture_gpu: Optional[int] = None,
        *,
        z_p_l4l5: float = PLANNING.z_p_l4l5,
        z_p_l5s1: float = PLANNING.z_p_l5s1,
        l5s1_l5_weight: float = PLANNING.l5s1_l5_weight,
        lr_spread: bool = True,
        lr_collapse_mm: float = 4.0,
        lr_half_width_mm: float = 0.0,
        lr_extra_lateral_mm: float = 0.0,
        foramen_anterior_mm: float = 0.0,
        foramen_posterior_mm: float = 0.0,
        foramen_superior_l4l5_mm: float = 0.0,
        foramen_superior_l5s1_mm: float = 0.0,
    ) -> Optional[np.ndarray]:
        """
        调用 puncture_target.infer.run_inference，按当前规划模式选取 l4_l5_* / l5_s1_* 之一。
        默认后处理与 ``python -m puncture_target.infer`` 在
        ``--lr_half_width_mm 0 --lr_extra_lateral_mm 0 --foramen_posterior_mm 0
        --foramen_superior_l4l5_mm 0 --foramen_superior_l5s1_mm 0`` 时一致（接近网络 raw + 裁块中心流程）。
        需已安装 torch；失败时返回 None。
        """
        try:
            from puncture_target.infer import run_inference
        except ImportError as e:
            print(f"警告: 无法导入 puncture_target（{e}），请安装 requirements_puncture_target.txt")
            return None
        try:
            model, device, ckpt_meta, use_amp = _get_cached_puncture_bundle(
                checkpoint_path, puncture_gpu
            )
        except Exception as e:
            print(f"警告: 加载 puncture 模型失败: {e}")
            return None
        print(
            "  puncture 后处理（与 infer 一致）: "
            f"lr_spread={lr_spread}, lr_collapse_mm={lr_collapse_mm}, lr_half_width_mm={lr_half_width_mm}, "
            f"lr_extra_lateral_mm={lr_extra_lateral_mm}, "
            f"foramen_posterior_mm={foramen_posterior_mm}, "
            f"foramen_superior_l4l5_mm={foramen_superior_l4l5_mm}, "
            f"foramen_superior_l5s1_mm={foramen_superior_l5s1_mm}"
        )
        payload, err = run_inference(
            ct_path,
            mask_path,
            model,
            device,
            ckpt_meta,
            use_amp,
            centers_json_path=None,
            z_p_l4l5=z_p_l4l5,
            z_p_l5s1=z_p_l5s1,
            l5s1_l5_weight=l5s1_l5_weight,
            lr_spread=lr_spread,
            lr_collapse_mm=lr_collapse_mm,
            lr_half_width_mm=lr_half_width_mm,
            lr_extra_lateral_mm=lr_extra_lateral_mm,
            foramen_anterior_mm=foramen_anterior_mm,
            foramen_posterior_mm=foramen_posterior_mm,
            foramen_superior_l4l5_mm=foramen_superior_l4l5_mm,
            foramen_superior_l5s1_mm=foramen_superior_l5s1_mm,
            gt_by_stem=None,
        )
        if err or not payload:
            print(f"警告: puncture_target 推理失败: {err}")
            return None
        end_pts = payload.get("end_points_ras_mm") or {}
        stem = self._puncture_stem_key(
            detect_facet_joint,
            detect_l5_s1_disc,
            auto_detect_target,
            facet_side,
            end_pts,
        )
        if not stem or stem not in end_pts:
            print(f"警告: puncture 输出中无 stem={stem}")
            return None
        return np.asarray(end_pts[stem], dtype=float).reshape(3)
    
    def plan_paths(self,
                  segmentation_file: str,
                  num_channels: int = 100,
                  label_values: Optional[List[int]] = None,
                  margin_mm: float = 10.0,
                  seed: Optional[int] = None,
                  target_point: Optional[np.ndarray] = None,
                  auto_detect_target: bool = False,
                  detect_facet_joint: bool = False,
                  detect_l5_s1_disc: bool = False,
                  l4_label: int = 4,
                  l5_label: int = 5,
                  s1_label: int = 6,
                  facet_joint_labels: Optional[List[int]] = None,
                  facet_side: str = 'auto',
                  disc_edge_midpoint_target: bool = False,
                  use_angle_grid: bool = False,
                  angle_step_deg: float = 5.0,
                  cpa_min_deg: Optional[float] = None,
                  cpa_max_deg: Optional[float] = None,
                  cpa_step_deg: Optional[float] = None,
                  csa_min_deg: Optional[float] = None,
                  csa_max_deg: Optional[float] = None,
                  csa_step_deg: Optional[float] = None,
                  ct_file: Optional[str] = None,
                  ct_dir: Optional[str] = None,
                  puncture_checkpoint: Optional[str] = None,
                  legacy_target_only: bool = False,
                  puncture_gpu: Optional[int] = None,
                  puncture_z_p_l4l5: float = PLANNING.z_p_l4l5,
                  puncture_z_p_l5s1: float = PLANNING.z_p_l5s1,
                  puncture_l5s1_l5_weight: float = PLANNING.l5s1_l5_weight,
                  puncture_lr_spread: bool = True,
                  puncture_lr_collapse_mm: float = 4.0,
                  puncture_lr_half_width_mm: float = 0.0,
                  puncture_lr_extra_lateral_mm: float = 0.0,
                  puncture_foramen_anterior_mm: float = 0.0,
                  puncture_foramen_posterior_mm: float = 0.0,
                  puncture_foramen_superior_l4l5_mm: float = 0.0,
                  puncture_foramen_superior_l5s1_mm: float = 0.0,
                  angle_grid_spherical: bool = True) -> List[Dict]:
        """
        规划多个通道并计算交集体积
        
        Args:
            segmentation_file: 分割数据文件路径（NII.GZ格式）
            num_channels: 要生成的通道数量（仅在随机模式下使用）
            label_values: 要统计的标签值列表，如果为None则统计所有标签
            margin_mm: 边界留白（毫米）
            seed: 随机种子（仅在随机模式下使用）
            target_point: 固定的穿刺靶点（终点）物理坐标 (3,)，如果为None则使用默认值
            auto_detect_target: 是否自动识别L5-S1交叉点作为靶点，默认False
            detect_facet_joint: 是否自动识别 L4-L5 靶点（椎间孔几何：间盘壳 + 关节突柱），默认 False
            detect_l5_s1_disc: 是否以L5-S1椎体上下后缘与椎间盘中位线交叉点作为靶点，默认False
            l4_label: L4椎体的标签值，默认4
            l5_label: L5椎体的标签值，默认5
            s1_label: S1椎体的标签值，默认6
            facet_joint_labels: 交集体积计算时仅使用关节突区域的标签列表，默认[1,2,3,4,5,6]（L1-S1）；其余标签使用完整结构
            facet_side: L4-L5 与 L5-S1 靶点取左/右/自动关节突内侧缘 X；通道入路侧与之相同（RAS 下 +X 为右）
            disc_edge_midpoint_target: True 时在 left/right 下启用薄壳半侧与椎间孔主靶点的 XY 融合；False（默认）仅用椎间孔主靶点
            use_angle_grid: 是否使用角度网格模式（每5度生成一个角度），默认False（随机模式）
            angle_step_deg: 角度间隔（度），默认5.0
            angle_grid_spherical: True（默认）时球面角 α×β 全矩形网格；结果中的 coronal/transverse 角度字段存 α、β。
            False 为旧版可行域 CPA×CSA（sin² 约束），此时该二字段为名义 CPA/CSA。
            ct_file: 与掩码配对的 CT NIfTI（单病例）；启用 puncture_target 网络靶点时推荐提供
            ct_dir: 批量时 CT 所在目录，按文件名 stem 与掩码配对
            puncture_checkpoint: puncture_target 的 best.pt；None 时先找本包 puncture_baseline/best.pt，再找项目 runs/puncture_baseline/best.pt
            legacy_target_only: True 时强制使用原有几何靶点，不调用网络
            puncture_gpu: 网络推理 GPU 编号（同 infer --gpu）；None 为默认 cuda:0
            puncture_* : 与 ``puncture_target.infer.run_inference`` 一致的几何后处理；默认 lr_half_width/extra/foramen/superior 均为 0（与命令行 infer 零修正一致）。若需脚本默认的左右拉开+外移，可设 puncture_lr_half_width_mm=14、puncture_lr_extra_lateral_mm=3 等。
            
        Returns:
            results: 结果列表，每个元素包含通道信息和交集体积
        """
        if seed is not None:
            np.random.seed(seed)
        
        # 加载分割数据
        segmentation, affine, metadata = self.load_nii_gz(segmentation_file)
        
        # 获取体素大小用于体积计算
        voxel_sizes = metadata['voxel_sizes']
        voxel_volume = np.prod(voxel_sizes)  # 立方毫米
        
        # 更新当前使用的 L4 / L5 / S1 标签（用于靶点检测与最优路径排序等）
        self._l4_label = l4_label
        self._l5_label = l5_label
        self._s1_label = s1_label

        # 记录当前规划的节段类型，供后续统计/权重使用
        # - L4-L5：使用双侧小关节靶点
        # - L5-S1：使用椎间盘靶点
        if detect_facet_joint:
            self._current_segment = "l4_l5"
        elif detect_l5_s1_disc:
            self._current_segment = "l5_s1"
        else:
            self._current_segment = "other"

        # 如果用户显式指定了需要统计的标签列表，确保包含通用的统计标签：
        # - 髂骨左 18、髂骨右 19（L4-L5 / L5-S1 均需；圆柱穿过哪侧即计入对应体积）
        # - L5-S1 节段：确保包含 S1 椎体标签（s1_label，默认6）
        if label_values is not None:
            label_set = set(label_values)
            label_set.add(18)
            label_set.add(19)
            # L5-S1：确保统计 S1
            if self._current_segment == "l5_s1":
                label_set.add(s1_label)
            label_values = sorted(label_set)

        # 交集体积计算时仅使用关节突的标签（默认 L1-S1：1,2,3,4,5,6），其余标签用完整结构
        if facet_joint_labels is None:
            facet_joint_labels = [1, 2, 3, 4, 5, 6]
        self._facet_joint_labels = list(facet_joint_labels)

        # 预先计算并缓存「关节突标签」的上下关节突区域掩码
        print(f"\n正在预先计算标签 {self._facet_joint_labels} 的上下关节突区域掩码（交集体积仅统计这些标签的关节突）...")
        facet_joint_masks: Dict[int, np.ndarray] = {}
        for lbl in self._facet_joint_labels:
            if (label_values is None or lbl in label_values) and np.any(segmentation == lbl):
                facet_joint_masks[lbl] = self.extract_facet_joint_region(segmentation, lbl)
        self._facet_joint_masks_cache = facet_joint_masks
        self._cached_segmentation_shape = segmentation.shape
        print(f"已缓存 {len(facet_joint_masks)} 个标签的关节突掩码（{list(facet_joint_masks.keys())}），其余标签按完整结构统计交集体积。")
        
        # 确定靶点（优先 puncture_target 网络，需 CT + 权重；否则几何方法）
        resolved_ct = resolve_ct_nifti_for_mask(segmentation_file, ct_file, ct_dir)
        ckpt_resolved = None if legacy_target_only else resolve_puncture_checkpoint_path(puncture_checkpoint)
        wants_auto_target = (
            target_point is None
            and (detect_facet_joint or detect_l5_s1_disc or auto_detect_target)
        )
        neural_target: Optional[np.ndarray] = None
        if wants_auto_target and ckpt_resolved and resolved_ct:
            print(
                f"\n尝试 puncture_target 网络靶点: CT={resolved_ct}\n"
                f"  掩码={segmentation_file}\n  权重={ckpt_resolved}"
            )
            neural_target = self.run_puncture_target_inference(
                resolved_ct,
                segmentation_file,
                ckpt_resolved,
                detect_facet_joint,
                detect_l5_s1_disc,
                auto_detect_target,
                facet_side,
                puncture_gpu=puncture_gpu,
                z_p_l4l5=puncture_z_p_l4l5,
                z_p_l5s1=puncture_z_p_l5s1,
                l5s1_l5_weight=puncture_l5s1_l5_weight,
                lr_spread=puncture_lr_spread,
                lr_collapse_mm=puncture_lr_collapse_mm,
                lr_half_width_mm=puncture_lr_half_width_mm,
                lr_extra_lateral_mm=puncture_lr_extra_lateral_mm,
                foramen_anterior_mm=puncture_foramen_anterior_mm,
                foramen_posterior_mm=puncture_foramen_posterior_mm,
                foramen_superior_l4l5_mm=puncture_foramen_superior_l4l5_mm,
                foramen_superior_l5s1_mm=puncture_foramen_superior_l5s1_mm,
            )
            if neural_target is not None:
                print(
                    f"  网络靶点（RAS mm）: [{neural_target[0]:.2f}, {neural_target[1]:.2f}, {neural_target[2]:.2f}]"
                )
        elif wants_auto_target and not legacy_target_only:
            if not resolved_ct:
                print(
                    "\n提示: 未提供可用的 CT（--ct / --ct_dir），自动穿刺靶点需要与掩码同形状的配对 CT。"
                )
            elif not ckpt_resolved:
                print(
                    "\n提示: 未找到 puncture 权重（runs/puncture_baseline/best.pt "
                    "或 path_planning_algorithm/puncture_baseline/best.pt）。"
                )

        # 自动靶点且非 legacy：必须以 puncture_target 网络偏移为准，禁止静默退回椎体几何或体数据中心
        if wants_auto_target and not legacy_target_only:
            if not ckpt_resolved or not resolved_ct:
                raise RuntimeError(
                    "自动穿刺靶点需要配对 CT（--ct 或 --ct_dir）以及 puncture 权重；"
                    "无法在缺少网络预测的条件下继续使用椎体几何近似或体素中心。"
                    "请配置 CT、权重后再运行；若必须使用纯几何靶点请另行指定 --legacy_target。"
                )
            if neural_target is None:
                raise RuntimeError(
                    "puncture_target 未返回有效穿刺靶点（推理失败、输出缺 stem、或与当前节段侧别不匹配）。"
                    "禁止在未获得网络偏移的情况下改用椎间孔几何或数据中心。"
                    "请查看上方警告日志并检查 CT/掩码对齐与数据质量；或使用 --legacy_target 明确启用几何靶点。"
                )

        # 确定靶点（调用方已传入 target_point 时优先使用：外部 mrk / --target_point，不得再被几何检测覆盖）
        if neural_target is not None:
            target_point = neural_target
            print(f"\n使用 puncture_target 网络靶点（facet_side={facet_side}）")
        elif target_point is not None:
            target_point = np.asarray(target_point, dtype=float).reshape(-1)[:3]
            print(f"\n使用指定靶点（外部 mrk / --target_point 等）: [{target_point[0]:.2f}, {target_point[1]:.2f}, {target_point[2]:.2f}] mm（RAS）")
        elif detect_facet_joint:
            # 自动识别L4-L5双侧小关节内侧缘投影点
            detected_target = self.detect_l4_l5_facet_joint_projection(
                segmentation, affine, l4_label, l5_label, facet_side=facet_side,
                use_disc_edge_midpoint=disc_edge_midpoint_target,
            )
            if detected_target is not None:
                target_point = detected_target
                print(f"\n使用自动识别的L4-L5靶点（facet_side={facet_side}）")
            else:
                # 如果自动识别失败，回退到数据中心
                h, w, d = segmentation.shape
                center_voxel = np.array([h/2, w/2, d/2])
                target_point = self.voxel_to_physical(center_voxel, affine)
                print(f"\n自动识别失败，使用默认靶点（数据中心）: {target_point}")
        elif detect_l5_s1_disc:
            # L5-S1 椎体上下后缘与椎间盘中位线交叉点
            detected_target = self.detect_l5_s1_disc_posterior_intersection(
                segmentation, affine, l5_label, s1_label, facet_side=facet_side,
                use_disc_edge_midpoint=disc_edge_midpoint_target,
            )
            if detected_target is not None:
                target_point = detected_target
                print(f"\n使用L5-S1椎间盘水平靶点（facet_side={facet_side}）")
            else:
                h, w, d = segmentation.shape
                center_voxel = np.array([h/2, w/2, d/2])
                target_point = self.voxel_to_physical(center_voxel, affine)
                print(f"\n自动识别失败，使用默认靶点（数据中心）: {target_point}")
        elif auto_detect_target:
            # 自动识别L5-S1交叉点
            detected_target = self.detect_l5_s1_intersection_point(
                segmentation, affine, l5_label, s1_label, facet_side=facet_side,
                use_disc_edge_midpoint=disc_edge_midpoint_target,
            )
            if detected_target is not None:
                target_point = detected_target
                print(f"\n使用自动识别的L5-S1靶点（facet_side={facet_side}）")
            else:
                # 如果自动识别失败，回退到数据中心
                h, w, d = segmentation.shape
                center_voxel = np.array([h/2, w/2, d/2])
                target_point = self.voxel_to_physical(center_voxel, affine)
                print(f"\n自动识别失败，使用默认靶点（数据中心）: {target_point}")
        else:
            # 使用数据中心的默认值
            h, w, d = segmentation.shape
            center_voxel = np.array([h/2, w/2, d/2])
            target_point = self.voxel_to_physical(center_voxel, affine)
            print(f"\n使用默认靶点（数据中心）: {target_point}")
        
        tp_arr = np.asarray(target_point, dtype=float).reshape(-1)[:3]
        self._paraspinal_side = self.resolve_paraspinal_side(
            facet_side, tp_arr, affine, segmentation.shape)
        
        lateral_dx_sign = self._lateral_dx_sign_from_facet_side(
            facet_side, tp_arr, affine, segmentation.shape)
        
        # 根据模式生成通道
        if use_angle_grid:
            # CPA/CSA 可分别设定范围与步长；未指定时使用统一 min/max/step
            cpa_min = cpa_min_deg if cpa_min_deg is not None else self.min_angle_deg
            cpa_max = cpa_max_deg if cpa_max_deg is not None else self.max_angle_deg
            cpa_step = cpa_step_deg if cpa_step_deg is not None else angle_step_deg
            csa_min = csa_min_deg if csa_min_deg is not None else self.min_angle_deg
            csa_max = csa_max_deg if csa_max_deg is not None else self.max_angle_deg
            csa_step = csa_step_deg if csa_step_deg is not None else angle_step_deg

            print(f"\n使用角度网格模式生成通道...")
            if angle_grid_spherical:
                print(f"球面角 α（XY 方位，对应 CPA 轴范围）: [{cpa_min}°, {cpa_max}°], 步长 {cpa_step}°")
                print(f"球面角 β（相对 XY 平面仰角，对应 CSA 轴范围）: [{csa_min}°, {csa_max}°], 步长 {csa_step}°")
            else:
                print(f"CPA（冠状面名义）: [{cpa_min}°, {cpa_max}°], 步长 {cpa_step}°")
                print(f"CSA（横断面名义）: [{csa_min}°, {csa_max}°], 步长 {csa_step}°")

            coronal_angles = np.arange(cpa_min, cpa_max + cpa_step/2, cpa_step)
            transverse_angles = np.arange(csa_min, csa_max + csa_step/2, csa_step)

            if angle_grid_spherical:
                # 球面角 α×β：完整矩形网格；β 在代码中对用户值取负再代入 sin/cos，见 direction_from_spherical_alpha_beta
                angle_pairs = [
                    (float(a), float(b))
                    for a in coronal_angles
                    for b in transverse_angles
                ]
                n_rect = len(angle_pairs)
                print(
                    f"球面角网格（XY 方位 α × 仰角 β）: {n_rect} 对；"
                    f"target_angle_coronal_deg / angle_with_coronal_deg = α，"
                    f"target_angle_transverse_deg / angle_with_transverse_deg = β（不再写入派生 CPA/CSA）"
                )
            else:
                # 可行域内 CPA×CSA（sin²(CPA)+sin²(CSA)≤1），与 direction_from_angles 名义一致
                angle_pairs = []
                for angle_coronal in coronal_angles:
                    sin_c = np.sin(np.radians(float(angle_coronal)))
                    sin_t_cap = float(np.sqrt(max(0.0, 1.0 - sin_c * sin_c)))
                    csa_geom_cap = float(
                        np.degrees(np.arcsin(np.clip(sin_t_cap, 0.0, 1.0)))
                    )
                    csa_cap = min(csa_max, csa_geom_cap)
                    for angle_transverse in transverse_angles:
                        if float(angle_transverse) > csa_cap + 1e-9:
                            continue
                        angle_pairs.append((float(angle_coronal), float(angle_transverse)))

                n_rect = len(coronal_angles) * len(transverse_angles)
                print(
                    f"CPA×CSA 可行域网格: 完整矩形约 {n_rect} 对；几何可行 {len(angle_pairs)} 对"
                )

            _side_cn = '右' if lateral_dx_sign < 0 else '左'
            print(f"入路与 --facet_side 一致：通道自患者{_side_cn}侧后外侧指向靶点（横断面 dX 符号 {lateral_dx_sign:+.0f}）")

            results = []
            channel_id = 0

            for pa, pb in tqdm(angle_pairs, desc="生成角度网格通道"):
                try:
                    if angle_grid_spherical:
                        direction = self.direction_from_spherical_alpha_beta(
                            pa, pb, lateral_dx_sign=lateral_dx_sign)
                        start_point, end_point, direction = (
                            self.generate_channel_from_direction(target_point, direction)
                        )
                        actual_angle_coronal = float(pa)
                        actual_angle_transverse = float(pb)
                        tgt_cpa = float(pa)
                        tgt_csa = float(pb)
                    else:
                        start_point, end_point, direction = self.generate_channel_from_angles(
                            target_point, pa, pb, lateral_dx_sign=lateral_dx_sign)
                        actual_angle_coronal = self.calculate_angle_with_coronal_plane(direction)
                        actual_angle_transverse = self.calculate_angle_with_transverse_plane(direction)
                        if (abs(actual_angle_coronal - pa) > 0.05 or
                                abs(actual_angle_transverse - pb) > 0.05):
                            print(
                                f"\n警告: 角度不匹配 - 目标: ({pa:.1f}°, {pb:.1f}°), "
                                f"实际: ({actual_angle_coronal:.1f}°, {actual_angle_transverse:.1f}°)"
                            )
                        tgt_cpa = float(pa)
                        tgt_csa = float(pb)

                    channel_id += 1

                    # 生成圆柱体内的点
                    cylinder_points = self.generate_cylinder_points(
                        start_point, direction,
                        self.channel_radius_mm, self.channel_length_mm,
                        self.resolution
                    )

                    # 转换为体素坐标
                    cylinder_points_voxel = self.physical_to_voxel(cylinder_points, affine)

                    # 计算交集体积
                    intersection_volumes = self.compute_intersection_volume(
                        segmentation, cylinder_points_voxel, voxel_sizes, label_values
                    )

                    # 计算通道总体积（用于参考）
                    channel_volume = np.pi * (self.channel_radius_mm ** 2) * self.channel_length_mm

                    result = {
                        'channel_id': channel_id,
                        'start_point': start_point.tolist(),
                        'end_point': end_point.tolist(),
                        'direction': direction.tolist(),
                        'angle_with_coronal_deg': float(actual_angle_coronal),
                        'angle_with_transverse_deg': float(actual_angle_transverse),
                        'target_angle_coronal_deg': tgt_cpa,
                        'target_angle_transverse_deg': tgt_csa,
                        'channel_volume_mm3': channel_volume,
                        'intersection_volumes_mm3': intersection_volumes,
                        'total_intersection_volume_mm3': sum(intersection_volumes.values()),
                        'angle_grid_mode': (
                            'spherical_alpha_beta' if angle_grid_spherical else 'feasible_cpa_csa'
                        ),
                    }
                    if angle_grid_spherical:
                        result['grid_spherical_alpha_deg'] = float(pa)
                        result['grid_spherical_beta_deg'] = float(pb)

                    results.append(result)

                except Exception as e:
                    print(f"\n警告: 生成网格通道 ({pa:.1f}°, {pb:.1f}°) 时出错: {str(e)}")
                    continue

            print(f"\n成功生成 {len(results)} 个通道（角度网格模式）")
        else:
            # 随机模式（原有逻辑）
            print(f"\n开始生成 {num_channels} 个随机通道...")
            _rs = '右' if lateral_dx_sign < 0 else '左'
            print(f"入路与 --facet_side 一致：随机通道方向 X 分量偏向患者{_rs}侧后外侧（dX 符号 {lateral_dx_sign:+.0f}）")
            print(f"通道参数: 半径={self.channel_radius_mm}mm, 长度={self.channel_length_mm}mm")
            print(f"采样分辨率: {self.resolution}mm")
            print(f"固定靶点（终点）: [{target_point[0]:.2f}, {target_point[1]:.2f}, {target_point[2]:.2f}] mm")
            print(f"角度约束: 与冠状面/横断面的夹角范围 [{self.min_angle_deg}°, {self.max_angle_deg}°]")
            
            results = []
            rejected_count = 0  # 记录被拒绝的通道数量
            max_attempts_per_channel = 1000  # 每个通道最多尝试次数，避免无限循环
            
            for i in tqdm(range(num_channels), desc="生成通道"):
                attempts = 0
                channel_generated = False
                
                while attempts < max_attempts_per_channel and not channel_generated:
                    try:
                        # 生成随机通道（固定终点，随机起点和方向）
                        start_point, end_point, direction = self.generate_random_channel(
                            target_point, segmentation.shape, affine, margin_mm,
                            lateral_dx_sign=lateral_dx_sign)
                        
                        # 检查角度约束
                        if not self.check_angle_constraints(direction):
                            rejected_count += 1
                            attempts += 1
                            continue  # 不满足角度约束，重新生成
                        
                        channel_generated = True
                        
                        # 生成圆柱体内的点
                        cylinder_points = self.generate_cylinder_points(
                            start_point, direction,
                            self.channel_radius_mm, self.channel_length_mm,
                            self.resolution
                        )
                        
                        # 转换为体素坐标
                        cylinder_points_voxel = self.physical_to_voxel(cylinder_points, affine)
                        
                        # 计算交集体积
                        intersection_volumes = self.compute_intersection_volume(
                            segmentation, cylinder_points_voxel, voxel_sizes, label_values
                        )
                        
                        # 计算通道总体积（用于参考）
                        channel_volume = np.pi * (self.channel_radius_mm ** 2) * self.channel_length_mm
                        
                        # 计算角度信息（用于记录）
                        angle_coronal = self.calculate_angle_with_coronal_plane(direction)
                        angle_transverse = self.calculate_angle_with_transverse_plane(direction)
                        
                        # 保存结果
                        result = {
                            'channel_id': i + 1,
                            'start_point': start_point.tolist(),  # 起点（体表端）
                            'end_point': end_point.tolist(),      # 终点（靶点，固定）
                            'direction': direction.tolist(),      # 方向向量（从起点指向终点）
                            'angle_with_coronal_deg': float(angle_coronal),      # 与冠状面的夹角
                            'angle_with_transverse_deg': float(angle_transverse), # 与横断面的夹角
                            'channel_volume_mm3': channel_volume,
                            'intersection_volumes_mm3': intersection_volumes,
                            'total_intersection_volume_mm3': sum(intersection_volumes.values())
                        }
                        
                        results.append(result)
                        break  # 成功生成通道，退出while循环
                        
                    except Exception as e:
                        attempts += 1
                        if attempts >= max_attempts_per_channel:
                            print(f"\n警告: 生成第 {i+1} 个通道时出错: {str(e)}")
                        continue
                
                # 如果尝试了max_attempts_per_channel次仍未生成有效通道，跳过
                if not channel_generated:
                    print(f"\n警告: 无法生成满足角度约束的第 {i+1} 个通道（已尝试 {max_attempts_per_channel} 次）")
            
            print(f"\n成功生成 {len(results)} 个通道")
            if rejected_count > 0:
                print(f"因角度约束被拒绝的通道数: {rejected_count}")
        
        # 保存数据以便后续可视化
        self._last_segmentation = segmentation
        self._last_affine = affine
        self._last_voxel_sizes = voxel_sizes
        self._last_segmentation_file = segmentation_file
        
        return results
    
    def save_results(self, results: List[Dict], output_file: str):
        """
        保存结果到JSON文件
        
        Args:
            results: 结果列表
            output_file: 输出文件路径
        """
        output_dir = os.path.dirname(output_file)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        print(f"\n结果已保存到: {output_file}")
    
    def ras_to_lps(self, coords: np.ndarray) -> np.ndarray:
        """
        将RAS坐标系转换为LPS坐标系
        
        RAS (Right-Anterior-Superior) -> LPS (Left-Posterior-Superior)
        转换公式：
        - L = -R (Left = -Right)
        - P = -A (Posterior = -Anterior)
        - S = S  (Superior = Superior)
        
        Args:
            coords: RAS坐标系下的坐标 (N, 3) 或 (3,)
            
        Returns:
            lps_coords: LPS坐标系下的坐标 (N, 3) 或 (3,)
        """
        coords = np.array(coords)
        original_shape = coords.shape
        
        # 确保是2D数组
        if coords.ndim == 1:
            coords = coords.reshape(1, -1)
        
        # 转换：L=-R, P=-A, S=S
        lps_coords = np.array([
            -coords[:, 0],  # L = -R
            -coords[:, 1],  # P = -A
            coords[:, 2]    # S = S
        ]).T
        
        # 恢复原始形状
        if len(original_shape) == 1:
            return lps_coords[0]
        return lps_coords
    
    def _print_slicer_ras_lps_hint(
            self,
            paths_to_export: List[Dict],
            convert_coordinates: bool,
            ) -> None:
        """导出 Slicer 后打印 RAS↔LPS 对照，避免控制台与 mrk.json 数值不一致被误认为错误。"""
        if not convert_coordinates or not paths_to_export:
            return
        ep = np.asarray(paths_to_export[0]['end_point'], dtype=float).reshape(-1)
        if ep.size != 3:
            return
        lps = self.ras_to_lps(ep)
        print(
            "  坐标系说明: 控制台与 JSON 中 X、Y 符号相反通常正常——"
            "程序内部为 RAS，mrk.json 为 LPS（L=-R, P=-A, S 不变）。"
            f"首条路径终点 RAS [{ep[0]:.2f}, {ep[1]:.2f}, {ep[2]:.2f}] "
            f"→ LPS [{lps[0]:.2f}, {lps[1]:.2f}, {lps[2]:.2f}]"
        )
    
    def export_to_slicer_markups(self, results: List[Dict], output_file: str, 
                                 max_paths: Optional[int] = None,
                                 convert_coordinates: bool = True):
        """
        导出路径为3D Slicer Markups格式（.mrk.json）
        
        3D Slicer可以识别此格式，用于显示路径的起点和终点
        
        Args:
            results: 通道结果列表
            output_file: 输出文件路径（.mrk.json格式）
            max_paths: 最多导出的路径数量，None表示导出所有
            convert_coordinates: 是否进行坐标系转换（RAS->LPS），默认True
        """
        if not results:
            print("警告: 没有路径数据可导出")
            return
        
        # 限制导出的路径数量
        paths_to_export = results[:max_paths] if max_paths else results
        
        # 构建Markups JSON格式（3D Slicer Markups Line格式）
        # 每个路径作为一条Line，包含起点和终点两个控制点
        markups_list = []
        
        for i, result in enumerate(paths_to_export):
            start_point = np.array(result['start_point'])
            end_point = np.array(result['end_point'])
            
            # 3D Slicer使用LPS坐标系（Left-Posterior-Superior）
            # NIfTI文件通常使用RAS坐标系，需要转换
            # RAS to LPS: L=-R, P=-A, S=S
            if convert_coordinates:
                start_point = self.ras_to_lps(start_point)
                end_point = self.ras_to_lps(end_point)
            
            # 为每个路径创建一条Line
            # 设置为不可交互（locked），避免在3D Slicer中被意外修改
            # 不显示标签和描述，但保持路径线条可见
            line_markup = {
                "type": "Line",
                "coordinateSystem": "LPS",
                "locked": True,  # 锁定整个路径，不可编辑
                "labelFormat": "%N-%d",
                "controlPoints": [
                    {
                        "id": f"{i+1}-start",
                        "label": "",  # 空标签，不显示
                        "description": "",  # 空描述，不显示
                        "associatedNodeID": "",
                        "position": [float(start_point[0]), float(start_point[1]), float(start_point[2])],
                        "orientation": [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
                        "selected": False,  # 默认不选中
                        "locked": True,  # 锁定控制点，不可移动
                        "visibility": True,  # 控制点需要可见才能显示线条（但标记会被隐藏）
                        "positionStatus": "defined"
                    },
                    {
                        "id": f"{i+1}-end",
                        "label": "",  # 空标签，不显示
                        "description": "",  # 空描述，不显示
                        "associatedNodeID": "",
                        "position": [float(end_point[0]), float(end_point[1]), float(end_point[2])],
                        "orientation": [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
                        "selected": False,  # 默认不选中
                        "locked": True,  # 锁定控制点，不可移动
                        "visibility": True,  # 控制点需要可见才能显示线条（但标记会被隐藏）
                        "positionStatus": "defined"
                    }
                ],
                "display": {
                    "visibility": True,  # 线条可见
                    "opacity": 1.0,
                    "color": [0.0, 1.0, 0.0],  # 绿色
                    "selectedColor": [1.0, 0.0, 0.0],  # 红色
                    "activeColor": [0.0, 0.0, 1.0],  # 蓝色
                    "propertiesLabelVisibility": False,  # 不显示属性标签
                    "pointLabelsVisibility": False,  # 不显示点标签
                    "textScale": 3.0,
                    "glyphType": "Sphere3D",
                    "glyphScale": 2.0,
                    "glyphSize": 5.0,
                    "useGlyphScale": True,
                    "sliceProjection": False,
                    "sliceProjectionUseFiducialColor": True,
                    "sliceProjectionOutlinedBehindSlicePlane": False,
                    "lineThickness": 2.0,  # 线条粗细
                    "lineColorFadingStart": 1.0,
                    "lineColorFadingEnd": 10.0,
                    "lineColorFadingSaturation": 1.0,
                    "lineColorFadingHueOffset": 0.0,
                    "handlesInteractive": False,  # 禁用交互式手柄
                    "snapMode": "toVisibleSurface"
                }
            }
            markups_list.append(line_markup)
        
        # 构建完整的Markups数据
        markups_data = {
            "@schema": "https://raw.githubusercontent.com/slicer/slicer/master/Modules/Loadable/Markups/Resources/Schema/markups-schema-v1.0.0.json#",
            "markups": markups_list
        }
        
        # 保存文件
        output_dir = os.path.dirname(output_file)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(markups_data, f, indent=2, ensure_ascii=False)
        
        print(f"\n3D Slicer Markups文件已保存到: {output_file}")
        print(f"  导出了 {len(paths_to_export)} 个路径（共 {len(results)} 个）")
        self._print_slicer_ras_lps_hint(paths_to_export, convert_coordinates)
        print(f"  在3D Slicer中：File -> Add Data -> 选择此文件")
    
    def export_to_slicer_fcsv(self, results: List[Dict], output_file: str,
                              max_paths: Optional[int] = None,
                              convert_coordinates: bool = True):
        """
        导出路径为3D Slicer Fiducial CSV格式（.fcsv）
        
        这是3D Slicer的经典格式，兼容性更好
        
        Args:
            results: 通道结果列表
            output_file: 输出文件路径（.fcsv格式）
            max_paths: 最多导出的路径数量，None表示导出所有
            convert_coordinates: 是否进行坐标系转换（RAS->LPS），默认True
        """
        if not results:
            print("警告: 没有路径数据可导出")
            return
        
        # 限制导出的路径数量
        paths_to_export = results[:max_paths] if max_paths else results
        
        # 构建FCSV文件内容
        lines = []
        
        # FCSV文件头
        lines.append("# Markups fiducial file version = 4.11")
        lines.append("# CoordinateSystem = 0")  # 0=LPS
        lines.append("# columns = id,x,y,z,ow,ox,oy,oz,vis,sel,lock,label,desc,associatedNodeID")
        
        # 添加每个路径的起点和终点
        for i, result in enumerate(paths_to_export):
            start_point = np.array(result['start_point'])
            end_point = np.array(result['end_point'])
            
            # 坐标系转换（RAS -> LPS）
            if convert_coordinates:
                start_point = self.ras_to_lps(start_point)
                end_point = self.ras_to_lps(end_point)
            
            # 起点
            # FCSV格式：vis,sel,lock (visible, selected, locked)
            # 设置为：可见=1, 不选中=0, 锁定=1（不可交互）
            lines.append(
                f"vtkMRMLMarkupsFiducialNode_{i*2+1},"
                f"{start_point[0]:.6f},{start_point[1]:.6f},{start_point[2]:.6f},"
                f"0.0,0.0,0.0,1.0,"  # 四元数（无旋转）
                f"1,0,1,"  # visible=1, selected=0, locked=1（锁定，不可交互）
                f"Channel_{result['channel_id']}_Start,"
                f"Path {result['channel_id']} start point,"
                f""
            )
            
            # 终点
            lines.append(
                f"vtkMRMLMarkupsFiducialNode_{i*2+2},"
                f"{end_point[0]:.6f},{end_point[1]:.6f},{end_point[2]:.6f},"
                f"0.0,0.0,0.0,1.0,"  # 四元数（无旋转）
                f"1,0,1,"  # visible=1, selected=0, locked=1（锁定，不可交互）
                f"Channel_{result['channel_id']}_End,"
                f"Path {result['channel_id']} end point (target),"
                f""
            )
        
        # 保存文件
        output_dir = os.path.dirname(output_file)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write('\n'.join(lines))
        
        print(f"\n3D Slicer FCSV文件已保存到: {output_file}")
        print(f"  导出了 {len(paths_to_export)} 个路径（共 {len(results)} 个）")
        self._print_slicer_ras_lps_hint(paths_to_export, convert_coordinates)
        print(f"  在3D Slicer中：File -> Add Data -> 选择此文件")
        print(f"  注意：FCSV格式只包含起点和终点，需要在Slicer中手动连接成线")
    
    def export_to_slicer(self, results: List[Dict], output_file: str,
                        format: str = 'mrk.json', max_paths: Optional[int] = None,
                        convert_coordinates: bool = True):
        """
        导出路径为3D Slicer格式（统一接口）
        
        Args:
            results: 通道结果列表
            output_file: 输出文件路径
            format: 导出格式，'mrk.json' 或 'fcsv'
            max_paths: 最多导出的路径数量，None表示导出所有
            convert_coordinates: 是否进行坐标系转换（RAS->LPS），默认True
                                如果路径位置不匹配，可以尝试设置为False
        """
        if format.lower() == 'fcsv':
            self.export_to_slicer_fcsv(results, output_file, max_paths, convert_coordinates)
        elif format.lower() in ['mrk.json', 'json']:
            # 确保文件扩展名正确
            if not output_file.endswith('.mrk.json'):
                output_file = output_file.replace('.json', '.mrk.json')
            self.export_to_slicer_markups(results, output_file, max_paths, convert_coordinates)
        else:
            raise ValueError(f"不支持的格式: {format}，支持 'mrk.json' 或 'fcsv'")
    
    def print_statistics(self, results: List[Dict]):
        """
        打印统计信息
        
        Args:
            results: 结果列表
        """
        if not results:
            print("没有结果可统计")
            return
        
        # 收集所有标签
        all_labels = set()
        for result in results:
            all_labels.update(result['intersection_volumes_mm3'].keys())
        
        all_labels = sorted(all_labels)
        
        print("\n" + "="*80)
        print("统计结果")
        print("="*80)
        print(f"总通道数: {len(results)}")
        print(f"唯一标签数: {len(all_labels)}")
        print(f"\n各标签的平均交集体积 (mm³):")
        print("-"*80)
        
        for label in all_labels:
            volumes = [
                intersection_volume_mm3_for_label(r['intersection_volumes_mm3'], label)
                for r in results
            ]
            mean_vol = np.mean(volumes)
            std_vol = np.std(volumes)
            min_vol = np.min(volumes)
            max_vol = np.max(volumes)
            
            print(f"标签 {label:3d}: 平均={mean_vol:10.2f}, "
                  f"标准差={std_vol:10.2f}, "
                  f"最小={min_vol:10.2f}, "
                  f"最大={max_vol:10.2f}")
        
        # 总体积统计（JSON 内 total 含背景；另给出不含标签 0 的统计）
        total_volumes = [r['total_intersection_volume_mm3'] for r in results]
        total_no_bg = [_path_total_non_background_volume_mm3(r) for r in results]
        print(f"\n总交集体积统计 (mm³)，与 JSON total_intersection_volume_mm3 一致（含标签 0 背景）:")
        print(f"  平均: {np.mean(total_volumes):.2f}")
        print(f"  标准差: {np.std(total_volumes):.2f}")
        print(f"  最小: {np.min(total_volumes):.2f}")
        print(f"  最大: {np.max(total_volumes):.2f}")
        print(f"\n不含标签 0（背景）的交集体积之和 (mm³):")
        print(f"  平均: {np.mean(total_no_bg):.2f}")
        print(f"  标准差: {np.std(total_no_bg):.2f}")
        print(f"  最小: {np.min(total_no_bg):.2f}")
        print(f"  最大: {np.max(total_no_bg):.2f}")
        
        print("="*80)
    
    
    def visualize_facet_joint_intersection(self,
                                          results: List[Dict],
                                          output_file: str,
                                          channel_index: int = 0,
                                          labels_to_show: List[int] = [1, 2, 3, 4, 5, 6],
                                          downsample_factor: int = 2):
        """
        可视化关节突区域与虚拟工作通道的交集体积（用于验证）
        
        此函数专门用于验证代码确实只计算了关节突区域与通道的交集体积，
        而不是整个椎体与通道的交集体积。
        
        Args:
            results: 通道结果列表
            output_file: 输出图片文件路径
            channel_index: 要可视化的通道索引（默认0，即第一个通道）
            labels_to_show: 要显示的标签列表（默认标签1-6，即L1-S1）
            downsample_factor: 下采样因子，用于减少数据量
        """
        if not MATPLOTLIB_AVAILABLE:
            print("警告: matplotlib未安装，无法生成可视化图片")
            print("请运行: pip install matplotlib")
            return
        
        if self._last_segmentation is None:
            print("警告: 没有可用的分割数据，无法生成可视化")
            return
        
        if not results or channel_index >= len(results):
            print(f"警告: 通道索引 {channel_index} 超出范围（共有 {len(results)} 个通道）")
            return
        
        ensure_chinese_font()
        
        print(f"\n开始生成关节突区域交集体积验证图...")
        print(f"  使用通道 {channel_index + 1}/{len(results)}")
        print(f"  显示标签: {labels_to_show}")
        
        segmentation = self._last_segmentation
        affine = self._last_affine
        voxel_sizes = self._last_voxel_sizes
        
        # 获取选中的通道信息
        channel = results[channel_index]
        start_point = np.array(channel['start_point'])
        end_point = np.array(channel['end_point'])
        direction = np.array(channel['direction'])
        intersection_volumes = channel.get('intersection_volumes_mm3', {})
        
        # 生成通道圆柱体的体素点
        cylinder_points = self.generate_cylinder_points(
            start_point, end_point,
            self.channel_radius_mm, self.channel_length_mm,
            self.resolution
        )
        cylinder_points_voxel = self.physical_to_voxel(cylinder_points, affine)
        
        # 下采样分割数据以加快可视化
        if downsample_factor > 1:
            segmentation_vis = segmentation[::downsample_factor, ::downsample_factor, ::downsample_factor]
            voxel_sizes_vis = voxel_sizes * downsample_factor
        else:
            segmentation_vis = segmentation
            voxel_sizes_vis = voxel_sizes
        
        # 创建图形（多个子图）
        fig = plt.figure(figsize=(20, 12))
        
        # 子图1：3D视图 - 显示完整椎体和关节突区域
        ax1 = fig.add_subplot(2, 2, 1, projection='3d')
        ax1.set_title('3D视图：完整椎体 vs 关节突区域', fontsize=14, fontweight='bold', pad=20)
        
        # 子图2：3D视图 - 显示通道与关节突区域的交集
        ax2 = fig.add_subplot(2, 2, 2, projection='3d')
        ax2.set_title('3D视图：通道与关节突区域交集', fontsize=14, fontweight='bold', pad=20)
        
        # 子图3：冠状面切片视图
        ax3 = fig.add_subplot(2, 2, 3)
        ax3.set_title('冠状面切片：通道与关节突区域交集', fontsize=14, fontweight='bold')
        
        # 子图4：交集体积对比柱状图
        ax4 = fig.add_subplot(2, 2, 4)
        ax4.set_title('交集体积对比（关节突区域 vs 完整椎体估算）', fontsize=14, fontweight='bold')
        
        # 准备数据：获取关节突区域掩码
        facet_joint_masks = {}
        if self._facet_joint_masks_cache is not None:
            facet_joint_masks = self._facet_joint_masks_cache
        
        # 计算通道与完整椎体的交集体积（用于对比）
        h, w, d = segmentation.shape
        indices = np.round(cylinder_points_voxel).astype(int)
        valid_mask = (
            (indices[:, 0] >= 0) & (indices[:, 0] < h) &
            (indices[:, 1] >= 0) & (indices[:, 1] < w) &
            (indices[:, 2] >= 0) & (indices[:, 2] < d)
        )
        valid_indices = indices[valid_mask]
        
        # 统计完整椎体的交集体积（用于对比）
        full_vertebrae_volumes = {}
        voxel_volume = np.prod(voxel_sizes)
        for idx in valid_indices:
            label = int(segmentation[idx[0], idx[1], idx[2]])
            if label in labels_to_show:
                full_vertebrae_volumes[label] = full_vertebrae_volumes.get(label, 0) + voxel_volume
        
        # 绘制3D视图
        for ax_idx, (ax, show_full) in enumerate([(ax1, True), (ax2, False)]):
            # 显示完整椎体（半透明）
            if show_full:
                for label in labels_to_show:
                    label_mask = (segmentation == label)
                    if np.any(label_mask):
                        # 下采样
                        if downsample_factor > 1:
                            label_mask = label_mask[::downsample_factor, ::downsample_factor, ::downsample_factor]
                        
                        # 获取表面点
                        indices_vis = np.where(label_mask)
                        if len(indices_vis[0]) > 0:
                            coords_vis = np.array([indices_vis[0], indices_vis[1], indices_vis[2]]).T
                            coords_vis = coords_vis[::max(1, len(coords_vis)//5000)]  # 限制点数
                            coords_physical_vis = self.voxel_to_physical(
                                coords_vis * downsample_factor, affine
                            )
                            ax.scatter(coords_physical_vis[:, 0], 
                                     coords_physical_vis[:, 1], 
                                     coords_physical_vis[:, 2],
                                     c=f'C{label}', alpha=0.1, s=1, 
                                     label=f'标签{label}（完整椎体）')
            
            # 显示关节突区域（更明显）
            for label in labels_to_show:
                if label in facet_joint_masks:
                    facet_mask = facet_joint_masks[label]
                    if np.any(facet_mask):
                        # 下采样
                        if downsample_factor > 1:
                            facet_mask = facet_mask[::downsample_factor, ::downsample_factor, ::downsample_factor]
                        
                        indices_facet = np.where(facet_mask)
                        if len(indices_facet[0]) > 0:
                            coords_facet = np.array([indices_facet[0], indices_facet[1], indices_facet[2]]).T
                            coords_facet = coords_facet[::max(1, len(coords_facet)//2000)]  # 限制点数
                            coords_facet_physical = self.voxel_to_physical(
                                coords_facet * downsample_factor, affine
                            )
                            ax.scatter(coords_facet_physical[:, 0],
                                     coords_facet_physical[:, 1],
                                     coords_facet_physical[:, 2],
                                     c=f'C{label}', alpha=0.5, s=3,
                                     label=f'标签{label}（关节突区域）')
            
            # 显示通道与关节突区域的交集（最明显）
            for label in labels_to_show:
                if label in facet_joint_masks:
                    facet_mask = facet_joint_masks[label]
                    intersection_points = []
                    for idx in valid_indices:
                        if (0 <= idx[0] < h and 0 <= idx[1] < w and 0 <= idx[2] < d):
                            if facet_mask[idx[0], idx[1], idx[2]]:
                                intersection_points.append(idx)
                    
                    if intersection_points:
                        intersection_points = np.array(intersection_points)
                        intersection_physical = self.voxel_to_physical(intersection_points, affine)
                        ax.scatter(intersection_physical[:, 0],
                                 intersection_physical[:, 1],
                                 intersection_physical[:, 2],
                                 c=f'C{label}', alpha=1.0, s=10, edgecolors='black', linewidths=0.5,
                                 label=f'标签{label}（交集）')
            
            # 绘制通道圆柱体
            cylinder_points_vis = cylinder_points[::max(1, len(cylinder_points)//1000)]
            ax.scatter(cylinder_points_vis[:, 0],
                      cylinder_points_vis[:, 1],
                      cylinder_points_vis[:, 2],
                      c='red', alpha=0.3, s=1, label='虚拟工作通道')
            
            ax.set_xlabel('X (mm)', fontsize=10)
            ax.set_ylabel('Y (mm)', fontsize=10)
            ax.set_zlabel('Z (mm)', fontsize=10)
            ax.legend(loc='upper left', fontsize=8, ncol=2)
        
        # 绘制冠状面切片视图
        # 找到通道中心位置
        channel_center = (start_point + end_point) / 2
        channel_center_voxel = self.physical_to_voxel(channel_center.reshape(1, -1), affine)[0]
        slice_idx = int(channel_center_voxel[1])  # Y轴方向
        
        if 0 <= slice_idx < w:
            # 显示完整椎体
            for label in labels_to_show:
                label_slice = (segmentation[:, slice_idx, :] == label)
                if np.any(label_slice):
                    # 使用标签索引作为颜色
                    color_map = np.zeros_like(label_slice, dtype=float)
                    color_map[label_slice] = label % 10  # 使用标签值模10作为颜色索引
                    ax3.imshow(color_map, alpha=0.2, 
                             cmap='tab10', 
                             vmin=0, vmax=10, aspect='auto', origin='lower')
            
            # 显示关节突区域
            for label in labels_to_show:
                if label in facet_joint_masks:
                    facet_slice = facet_joint_masks[label][:, slice_idx, :]
                    if np.any(facet_slice):
                        color_map = np.zeros_like(facet_slice, dtype=float)
                        color_map[facet_slice] = label % 10
                        ax3.imshow(color_map, alpha=0.6,
                                 cmap='tab10',
                                 vmin=0, vmax=10, aspect='auto', origin='lower')
            
            # 显示通道与关节突区域的交集
            for label in labels_to_show:
                if label in facet_joint_masks:
                    facet_slice = facet_joint_masks[label][:, slice_idx, :]
                    intersection_slice = np.zeros_like(facet_slice, dtype=bool)
                    for idx in valid_indices:
                        if idx[1] == slice_idx and 0 <= idx[0] < h and 0 <= idx[2] < d:
                            if facet_slice[idx[0], idx[2]]:
                                intersection_slice[idx[0], idx[2]] = True
                    
                    if np.any(intersection_slice):
                        ax3.imshow(intersection_slice.astype(float), alpha=1.0,
                                 cmap='hot', aspect='auto', origin='lower')
            
            ax3.set_xlabel('Z (切片索引)', fontsize=10)
            ax3.set_ylabel('X (切片索引)', fontsize=10)
            ax3.set_title(f'冠状面切片 Y={slice_idx}（通道中心位置）', fontsize=12)
        
        # 绘制交集体积对比柱状图
        labels = sorted([l for l in labels_to_show if l in intersection_volumes])
        if labels:
            facet_volumes = [intersection_volumes.get(l, 0) for l in labels]
            full_volumes = [full_vertebrae_volumes.get(l, 0) for l in labels]
            
            x = np.arange(len(labels))
            width = 0.35
            
            bars1 = ax4.bar(x - width/2, facet_volumes, width, 
                           label='关节突区域交集体积（实际计算）', 
                           color='steelblue', alpha=0.8)
            bars2 = ax4.bar(x + width/2, full_volumes, width,
                           label='完整椎体交集体积（估算对比）',
                           color='lightcoral', alpha=0.8)
            
            ax4.set_xlabel('标签', fontsize=12, fontweight='bold')
            ax4.set_ylabel('交集体积 (mm³)', fontsize=12, fontweight='bold')
            ax4.set_xticks(x)
            ax4.set_xticklabels([f'标签{l}\n(L{"1" if l==1 else "2" if l==2 else "3" if l==3 else "4" if l==4 else "5" if l==5 else "S1"})' for l in labels])
            ax4.legend(fontsize=10)
            ax4.grid(True, alpha=0.3, axis='y')
            
            # 添加数值标签
            for bars in [bars1, bars2]:
                for bar in bars:
                    height = bar.get_height()
                    if height > 0:
                        ax4.text(bar.get_x() + bar.get_width()/2., height,
                                f'{height:.1f}',
                                ha='center', va='bottom', fontsize=8)
        
        plt.tight_layout()
        plt.savefig(output_file, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"  关节突区域交集体积验证图已保存到: {output_file}")
        plt.close()
    
    
    def calculate_angle_from_direction(self, direction: np.ndarray, 
                                      reference_direction: Optional[np.ndarray] = None) -> float:
        """
        计算方向向量与参考方向的夹角（度）
        
        Args:
            direction: 方向向量 (3,)
            reference_direction: 参考方向向量，默认为垂直向下 [0, 0, -1]
            
        Returns:
            angle: 夹角（度）
        """
        if reference_direction is None:
            reference_direction = np.array([0, 0, -1])  # 垂直向下
        
        # 归一化
        direction = direction / np.linalg.norm(direction)
        reference_direction = reference_direction / np.linalg.norm(reference_direction)
        
        # 计算夹角（使用点积）
        cos_angle = np.clip(np.dot(direction, reference_direction), -1.0, 1.0)
        angle_rad = np.arccos(cos_angle)
        angle_deg = np.degrees(angle_rad)
        
        return angle_deg
    
    def calculate_angle_with_coronal_plane(self, direction: np.ndarray) -> float:
        """
        计算轨迹方向与冠状面的夹角（度）
        
        在RAS坐标系中，冠状面垂直于A轴（Anterior），法向量是 [0, 1, 0]
        轨迹与平面的夹角 = arcsin(|direction · normal|)
        
        Args:
            direction: 轨迹方向向量 (3,)
            
        Returns:
            angle: 轨迹与冠状面的夹角（度），范围0-90度
        """
        # 归一化方向向量
        direction = direction / np.linalg.norm(direction)
        
        # 冠状面法向量（垂直于A轴，即前后方向）
        # 在RAS坐标系中，冠状面法向量是 [0, 1, 0] 或 [0, -1, 0]
        coronal_normal = np.array([0, 1, 0])
        
        # 计算方向向量在法向量上的投影
        dot_product = abs(np.dot(direction, coronal_normal))
        
        # 轨迹与平面的夹角 = arcsin(|direction · normal|)
        # 因为方向向量和法向量都是单位向量，dot_product的范围是[0, 1]
        sin_angle = np.clip(dot_product, 0.0, 1.0)
        angle_rad = np.arcsin(sin_angle)
        angle_deg = np.degrees(angle_rad)
        
        return angle_deg
    
    def calculate_angle_with_transverse_plane(self, direction: np.ndarray) -> float:
        """
        计算轨迹方向与横断面的夹角（度）
        
        在RAS坐标系中，横断面垂直于S轴（Superior），法向量是 [0, 0, 1]
        轨迹与平面的夹角 = arcsin(|direction · normal|)
        
        Args:
            direction: 轨迹方向向量 (3,)
            
        Returns:
            angle: 轨迹与横断面的夹角（度），范围0-90度
        """
        # 归一化方向向量
        direction = direction / np.linalg.norm(direction)
        
        # 横断面法向量（垂直于S轴，即上下方向）
        # 在RAS坐标系中，横断面法向量是 [0, 0, 1] 或 [0, 0, -1]
        transverse_normal = np.array([0, 0, 1])
        
        # 计算方向向量在法向量上的投影
        dot_product = abs(np.dot(direction, transverse_normal))
        
        # 轨迹与平面的夹角 = arcsin(|direction · normal|)
        sin_angle = np.clip(dot_product, 0.0, 1.0)
        angle_rad = np.arcsin(sin_angle)
        angle_deg = np.degrees(angle_rad)
        
        return angle_deg
    
    def check_angle_constraints(self, direction: np.ndarray) -> bool:
        """
        检查轨迹方向是否满足角度约束
        
        检查轨迹与冠状面和横断面的夹角是否都在指定范围内
        
        Args:
            direction: 轨迹方向向量 (3,)
            
        Returns:
            valid: 如果两个夹角都在[min_angle_deg, max_angle_deg]范围内返回True，否则返回False
        """
        # 计算与冠状面的夹角
        angle_coronal = self.calculate_angle_with_coronal_plane(direction)
        
        # 计算与横断面的夹角
        angle_transverse = self.calculate_angle_with_transverse_plane(direction)
        
        # 检查两个夹角是否都在允许范围内
        coronal_valid = self.min_angle_deg <= angle_coronal <= self.max_angle_deg
        transverse_valid = self.min_angle_deg <= angle_transverse <= self.max_angle_deg
        
        return coronal_valid and transverse_valid
    
    
    def plot_cpa_csa_volume_chart(self,
                                  results: List[Dict],
                                  output_dir: str,
                                  angle_step_deg: float = 5.0,
                                  segmentation: Optional[np.ndarray] = None,
                                  all_labels: Optional[List[int]] = None,
                                  use_structure_names: bool = True):
        """
        为每个标签绘制CPA-CSA交集体积柱状图
        
        图表格式：
        - X轴：CPA角度（冠状面角度）
        - 每个X轴位置：多个柱状图，代表不同的CSA值（横断面角度）
        - Y轴：交集体积（mm³）
        - 每个标签生成一张图
        
        Args:
            results: 通道结果列表
            output_dir: 输出目录
            angle_step_deg: 角度间隔（度），用于分组，默认5.0
            segmentation: 分割数据数组（可选），用于获取所有可能的标签
            all_labels: 要生成的标签列表（可选），如果指定则只生成这些标签的图表
            use_structure_names: 为 True 时，标题与文件名使用解剖结构名（与 inference_3d_example 一致），
                为 False 时沿用旧版 ``Label {id}`` / ``label_{id}_...`` 命名。
        """
        if not MATPLOTLIB_AVAILABLE:
            print("警告: matplotlib未安装，无法生成CPA-CSA交集体积图")
            print("请运行: pip install matplotlib")
            return
        
        # 确保中文字体已初始化
        ensure_chinese_font()
        
        if not results:
            print("警告: 没有结果数据，无法生成图表")
            return
        
        print(f"\n开始生成交集体积柱状图（按角度分面）...")
        
        use_ab = results_use_spherical_alpha_beta_axes(results)
        x_axis_label = 'α (°)' if use_ab else 'CPA (°)'
        legend_secondary = 'β' if use_ab else 'CSA'
        angle_term_a = 'α' if use_ab else 'CPA'
        angle_term_b = 'β' if use_ab else 'CSA'
        vol_png_suffix = 'alpha_beta_volume' if use_ab else 'cpa_csa_volume'
        
        # 是否与 angle_average / chart_labels_* 一致：显式传入标签列表时，总图只对列表内结构求和（不含标签 0）
        explicit_chart_label_filter = all_labels is not None
        labels_present = _intersection_label_ids_in_results(results)
        
        # 收集要出图的结构标签（不含背景 0）
        # 显式列表（如 chart_labels_*）：必须为列表中每一项各出一张图，即使该结构在所有路径上交集体积均为 0
        # （否则腰大肌等键常缺失会被误删，出现「少图」）
        if all_labels is not None:
            all_labels = sorted({int(l) for l in all_labels if int(l) != 0})
            if not all_labels:
                print("  警告: 指定标签列表为空（或仅含 0），跳过交集体积角度图")
                return
        elif segmentation is not None:
            unique_labels = np.unique(segmentation)
            unique_labels = unique_labels[unique_labels > 0]
            all_labels = sorted([int(l) for l in unique_labels])
            print(f"  从分割数据中获取到 {len(all_labels)} 个标签（已排除 0）: {all_labels}")
        else:
            all_labels = sorted(labels_present)
            print(f"  注意: 只生成在结果中出现的非背景标签（共 {len(all_labels)} 个）")
            print(f"  提示: 与 angle_average_from_results 一致时，请传入 chart_labels_l4_l5_for_side / chart_labels_l5_s1_for_side")
        
        if len(all_labels) == 0:
            print("警告: 没有找到任何标签数据")
            return
        
        if use_structure_names:
            print(
                f"  将生成 {len(all_labels)} 个结构的图表: "
                f"{[label_prose_name_for_intersection_chart(int(l)) for l in all_labels]}"
            )
        else:
            print(f"  将生成 {len(all_labels)} 个标签的图表: {all_labels}")
        
        # 收集所有CPA和CSA角度
        cpa_angles = set()
        csa_angles = set()
        
        for result in results:
            # 优先使用目标角度，如果没有则使用实际角度
            cpa = result.get('target_angle_coronal_deg', result.get('angle_with_coronal_deg', 0.0))
            csa = result.get('target_angle_transverse_deg', result.get('angle_with_transverse_deg', 0.0))
            
            # 四舍五入到最近的angle_step_deg
            cpa_rounded = round(cpa / angle_step_deg) * angle_step_deg
            csa_rounded = round(csa / angle_step_deg) * angle_step_deg
            
            cpa_angles.add(cpa_rounded)
            csa_angles.add(csa_rounded)
        
        cpa_angles = sorted(cpa_angles, key=float)
        csa_angles = sorted(csa_angles, key=float)
        
        print(f"  {angle_term_a}角度范围: {min(cpa_angles):.1f}° - {max(cpa_angles):.1f}°")
        print(f"  {angle_term_b}角度范围: {min(csa_angles):.1f}° - {max(csa_angles):.1f}°")
        print(f"  {angle_term_a}角度列表: {cpa_angles}")
        print(f"  {angle_term_b}角度列表: {csa_angles}")
        
        num_csa = len(csa_angles)
        # 每个 CPA 刻度在 x 轴上相距 1.0；组内柱体总宽度须 < 1，否则与相邻 CPA 组重叠。
        # 旧版固定 bar_width=0.12 时，若 CSA 条数≥9 则 9*0.12>1 会明显叠在一起。
        _cpa_slot_margin = 0.06
        bar_width = (1.0 - _cpa_slot_margin) / max(num_csa, 1)
        chart_colors = _cpa_csa_bar_colors(num_csa)
        
        # 创建输出目录
        os.makedirs(output_dir, exist_ok=True)
        
        # 为每个标签准备数据并绘制图表
        for label in all_labels:
            # 构建数据矩阵：data[cpa_index][csa_index] = volume
            data_matrix = {}
            
            for result in results:
                # 获取角度
                cpa = result.get('target_angle_coronal_deg', result.get('angle_with_coronal_deg', 0.0))
                csa = result.get('target_angle_transverse_deg', result.get('angle_with_transverse_deg', 0.0))
                
                # 四舍五入到最近的angle_step_deg
                cpa_rounded = round(cpa / angle_step_deg) * angle_step_deg
                csa_rounded = round(csa / angle_step_deg) * angle_step_deg
                
                # 获取该标签的交集体积（JSON 加载后键常为 str）
                volume = intersection_volume_mm3_for_label(result['intersection_volumes_mm3'], label)
                
                # 存储数据
                if cpa_rounded not in data_matrix:
                    data_matrix[cpa_rounded] = {}
                if csa_rounded not in data_matrix[cpa_rounded]:
                    data_matrix[cpa_rounded][csa_rounded] = []
                
                data_matrix[cpa_rounded][csa_rounded].append(volume)
            
            # 计算平均值（如果有多个相同角度的数据点）
            for cpa in data_matrix:
                for csa in data_matrix[cpa]:
                    volumes = data_matrix[cpa][csa]
                    data_matrix[cpa][csa] = np.mean(volumes) if volumes else 0.0
            
            # 创建图表
            fig, ax = plt.subplots(figsize=(14, 8))
            
            # 设置柱状图参数（bar_width / chart_colors 已在上面按 num_csa 统一计算）
            x_positions = np.arange(len(cpa_angles))
            
            # 为每个CSA角度绘制柱状图（图例由 _apply_cpa_csa_legend 统一生成）
            bar_containers: List = []
            for csa_idx, csa in enumerate(csa_angles):
                volumes = []
                for cpa in cpa_angles:
                    if cpa in data_matrix and csa in data_matrix[cpa]:
                        volumes.append(data_matrix[cpa][csa])
                    else:
                        volumes.append(0.0)
                
                # 计算每个柱子的x位置（偏移以形成分组）
                offset = (csa_idx - num_csa / 2 + 0.5) * bar_width
                x_pos = x_positions + offset
                
                color = chart_colors[csa_idx % len(chart_colors)]
                cont = ax.bar(
                    x_pos,
                    volumes,
                    bar_width,
                    color=color,
                    alpha=0.8,
                    edgecolor='black',
                    linewidth=0.5,
                )
                bar_containers.append(cont)
            
            # 设置X轴（与论文图风格一致：CPA (°)）
            ax.set_xlabel(x_axis_label, fontsize=12, fontweight='bold')
            ax.set_xticks(x_positions)
            ax.set_xticklabels([rf'${cpa:.0f}^\circ$' for cpa in cpa_angles])
            
            # 设置标题/Y 轴：Intersection Volume of <structure> … ($mm^{3}$)
            if use_structure_names:
                _struct = label_prose_name_for_intersection_chart(int(label))
            else:
                _struct = f"Label {label}"
            ax.set_ylabel(
                rf'Intersection Volume of {_struct} ($mm^{3}$)',
                fontsize=12,
                fontweight='bold',
            )
            ax.set_title(
                f'Intersection Volume of {_struct}',
                fontsize=14,
                fontweight='bold',
                pad=20,
            )
            
            ax.grid(True, alpha=0.3, linestyle='--', axis='y')
            ax.set_axisbelow(True)
            _apply_cpa_csa_legend(ax, fig, csa_angles, bar_containers, secondary_angle_name=legend_secondary)
            
            # 保存图片（默认以结构英文名命名，避免仅用数字标签）
            if use_structure_names:
                stem = label_safe_filename_stem_for_chart(int(label))
                out_name = f'{stem}_{vol_png_suffix}.png'
            else:
                out_name = f'label_{label}_{vol_png_suffix}.png'
            output_file = os.path.join(output_dir, out_name)
            plt.savefig(output_file, dpi=300, bbox_inches='tight', facecolor='white')
            plt.close()
            
            print(f"  已保存: {out_name}")
        
        # 生成总交集体积图（所有标签的总和）
        print(f"\n生成总交集体积图...")
        fig, ax = plt.subplots(figsize=(14, 8))
        
        # 构建总交集体积数据
        total_data_matrix = {}
        
        for result in results:
            cpa = result.get('target_angle_coronal_deg', result.get('angle_with_coronal_deg', 0.0))
            csa = result.get('target_angle_transverse_deg', result.get('angle_with_transverse_deg', 0.0))
            
            cpa_rounded = round(cpa / angle_step_deg) * angle_step_deg
            csa_rounded = round(csa / angle_step_deg) * angle_step_deg
            
            if explicit_chart_label_filter:
                total_volume = _path_sum_intersection_mm3_for_labels(result, all_labels)
            else:
                total_volume = _path_total_non_background_volume_mm3(result)
            
            if cpa_rounded not in total_data_matrix:
                total_data_matrix[cpa_rounded] = {}
            if csa_rounded not in total_data_matrix[cpa_rounded]:
                total_data_matrix[cpa_rounded][csa_rounded] = []
            
            total_data_matrix[cpa_rounded][csa_rounded].append(total_volume)
        
        # 计算平均值
        for cpa in total_data_matrix:
            for csa in total_data_matrix[cpa]:
                volumes = total_data_matrix[cpa][csa]
                total_data_matrix[cpa][csa] = np.mean(volumes) if volumes else 0.0
        
        # 绘制总交集体积图（与上面共用 bar_width / chart_colors，避免组内柱重叠）
        x_positions = np.arange(len(cpa_angles))
        
        bar_containers_total: List = []
        for csa_idx, csa in enumerate(csa_angles):
            volumes = []
            for cpa in cpa_angles:
                if cpa in total_data_matrix and csa in total_data_matrix[cpa]:
                    volumes.append(total_data_matrix[cpa][csa])
                else:
                    volumes.append(0.0)
            
            offset = (csa_idx - num_csa / 2 + 0.5) * bar_width
            x_pos = x_positions + offset
            
            color = chart_colors[csa_idx % len(chart_colors)]
            cont = ax.bar(
                x_pos,
                volumes,
                bar_width,
                color=color,
                alpha=0.8,
                edgecolor='black',
                linewidth=0.5,
            )
            bar_containers_total.append(cont)
        
        ax.set_xlabel(x_axis_label, fontsize=12, fontweight='bold')
        ax.set_xticks(x_positions)
        ax.set_xticklabels([rf'${cpa:.0f}^\circ$' for cpa in cpa_angles])
        if explicit_chart_label_filter:
            _yt = r'Total Intersection Volume ($mm^{3}$)'
            _tt = "Total Intersection Volume"
        else:
            _yt = r'Intersection Volume of All Structures ($mm^{3}$)'
            _tt = "Intersection Volume of All Structures (excl. background)"
        ax.set_ylabel(_yt, fontsize=12, fontweight='bold')
        ax.set_title(_tt, fontsize=14, fontweight='bold', pad=20)
        ax.grid(True, alpha=0.3, linestyle='--', axis='y')
        ax.set_axisbelow(True)
        _apply_cpa_csa_legend(ax, fig, csa_angles, bar_containers_total, secondary_angle_name=legend_secondary)
        total_base = 'total_alpha_beta_volume' if use_ab else 'total_intersection_volume'
        total_output_file = os.path.join(output_dir, f'{total_base}.png')
        plt.savefig(total_output_file, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        
        print(
            f"  已保存: {total_base}.png"
            + (" (total — planned structures)" if explicit_chart_label_filter else "")
        )
        print(f"\n所有交集体积图（{angle_term_a}×{angle_term_b}）已保存到: {output_dir}")
        print(f"共生成 {len(all_labels) + 1} 张图表（{len(all_labels)} 个结构 + 1 张总交集体积）")
    
    def export_all_paths_intersection_excel(
        self,
        results: List[Dict],
        results_json_path: str,
        *,
        facet_side: str = "auto",
        use_facet_joint: bool = False,
        use_l5_s1_disc: bool = False,
    ) -> None:
        """
        导出本任务全部通道的交集体积（不做最优排名筛选；按通道 ID 升序列出）。
        文件名：与 results.json 同 stem，后缀 _all_paths_intersections.xlsx

        仅输出与规划任务一致的关注结构列（L4-L5：L4/L5+同侧四肌+同侧髂骨；L5-S1：L5/骶骨+同侧四肌+同侧髂骨），
        不再把 results 中出现的 L1–L3、对侧肌等全部标签展开为列。
        「总交集体积」= 上述关注结构体积之和（不含其它分割标签）。
        """
        if not results:
            return
        if not PANDAS_AVAILABLE:
            print("\n警告: pandas 未安装，跳过「全部路径交集体积」Excel。请安装: pip install pandas openpyxl")
            return
        label_list = chart_labels_for_planning_task(
            use_facet_joint, use_l5_s1_disc, facet_side
        )
        if not label_list:
            all_labels: set = set()
            for r in results:
                iv = r.get('intersection_volumes_mm3') or {}
                for k in iv.keys():
                    try:
                        ki = int(k) if not isinstance(k, int) else int(k)
                    except (TypeError, ValueError):
                        continue
                    if ki == 0:
                        continue
                    all_labels.add(ki)
            label_list = sorted(all_labels)
        label_names = FILTER_OPTIMAL_LABEL_NAMES
        use_ab = results_use_spherical_alpha_beta_axes(results)
        col_a = 'α(°)' if use_ab else 'CPA角度'
        col_b = 'β(°)' if use_ab else 'CSA角度'
        rows: List[Dict] = []
        for path in sorted(results, key=lambda x: int(x.get('channel_id', 0))):
            cpa = path.get('target_angle_coronal_deg', path.get('angle_with_coronal_deg', 0.0))
            csa = path.get('target_angle_transverse_deg', path.get('angle_with_transverse_deg', 0.0))
            intersection_volumes = path.get('intersection_volumes_mm3', {}) or {}
            row: Dict = {
                '通道ID': int(path.get('channel_id', 0)),
                col_a: round(float(cpa), 2),
                col_b: round(float(csa), 2),
            }
            total_planned = 0.0
            for label in label_list:
                vol = intersection_volumes.get(str(label), intersection_volumes.get(label, 0.0))
                fv = float(vol) if vol is not None else 0.0
                total_planned += fv
                name = label_names.get(label, f'标签{label}')
                row[f'{name}(mm3)'] = round(fv, 4)
            row['总交集体积'] = round(total_planned, 4)
            rows.append(row)
        try:
            df = pd.DataFrame(rows)
            base = os.path.splitext(os.path.abspath(results_json_path))[0]
            excel_filename = f"{base}_all_paths_intersections.xlsx"
            d = os.path.dirname(excel_filename)
            if d and not os.path.exists(d):
                os.makedirs(d, exist_ok=True)
            df.to_excel(excel_filename, index=False, engine='openpyxl')
            print(f"\n[OK] 已导出全部路径交集体积 Excel: {excel_filename}")
        except Exception as e:
            print(f"\n警告: 导出全部路径交集体积 Excel 失败: {e}")

    def _optimal_facet_label_ids_for_segment(self) -> List[int]:
        """当前节段对应的椎体标签（交集体积已在关节突掩码内统计）。"""
        segment_type = getattr(self, "_current_segment", "other")
        l4 = int(getattr(self, "_l4_label", 4))
        l5 = int(getattr(self, "_l5_label", 5))
        s1 = int(getattr(self, "_s1_label", 6))
        if segment_type == "l5_s1":
            return [l5, s1]
        if segment_type == "l4_l5":
            return [l4, l5]
        return [l4, l5]

    def _iliac_aggregate_mm3_for_gate(
            self,
            vols: Dict,
            mode: str,
            side: str,
    ) -> float:
        """髂骨门槛用聚合体积（mm³）：sum / max / ipsilateral。"""
        v18 = intersection_volume_mm3_for_label(vols, 18)
        v19 = intersection_volume_mm3_for_label(vols, 19)
        m = (mode or "sum").strip().lower()
        if m == "max":
            return float(max(v18, v19))
        if m == "ipsilateral":
            if side == "right":
                return float(v19)
            if side == "left":
                return float(v18)
            return float(v18 + v19)
        return float(v18 + v19)

    def _ipsilateral_paraspinal_rank_volumes(self, vols: Dict, side: str) -> Tuple[float, float, float, float]:
        """同侧：多裂肌、竖脊肌、腰大肌、腰方肌 的交集体积（mm³）。"""
        if side == "right":
            return (
                intersection_volume_mm3_for_label(vols, 13),
                intersection_volume_mm3_for_label(vols, 12),
                intersection_volume_mm3_for_label(vols, 10),
                intersection_volume_mm3_for_label(vols, 11),
            )
        return (
            intersection_volume_mm3_for_label(vols, 17),
            intersection_volume_mm3_for_label(vols, 16),
            intersection_volume_mm3_for_label(vols, 14),
            intersection_volume_mm3_for_label(vols, 15),
        )

    def filter_optimal_paths_weighted_legacy(self,
                            results: List[Dict],
                            label_weights: Optional[Dict[int, float]] = None,
                            total_volume_weight: float = 0.3,
                            top_k: int = 10) -> List[Dict]:
        """
        旧版：按 Σ(体积×权重)+总交集体积×权重 得分筛选（得分越低越优）。
        需显式传入 use_legacy_weighted_ranking / --filter_optimal_legacy_weights 时使用。
        """
        if not results:
            print("警告: 没有结果数据，无法筛选最优路径")
            return []
        
        segment_type = getattr(self, "_current_segment", "other")
        side = getattr(self, "_paraspinal_side", None)
        if side not in ("left", "right"):
            side = "left"
        if side == "right":
            muscle_paraspinal = {10: 0.3, 11: 0.3, 12: 0.3, 13: 0.4}
            muscle_paraspinal_ilium = {**muscle_paraspinal, 19: 0.3}
        else:
            muscle_paraspinal = {14: 0.3, 15: 0.3, 16: 0.3, 17: 0.4}
            muscle_paraspinal_ilium = {**muscle_paraspinal, 18: 0.3}
        if segment_type == "l5_s1":
            default_weights = {5: 0.4, 6: 0.4, **muscle_paraspinal_ilium}
        elif segment_type == "l4_l5":
            default_weights = {4: 0.4, 5: 0.4, **muscle_paraspinal_ilium}
        else:
            default_weights = {4: 0.4, 5: 0.4, **muscle_paraspinal}
        
        if label_weights is None:
            label_weights = default_weights
        else:
            final_weights = default_weights.copy()
            final_weights.update(label_weights)
            label_weights = final_weights
        
        print(f"\n开始筛选最优路径（旧版加权得分）...")
        print(f"标签权重分配（入路侧={getattr(self, '_paraspinal_side', '?')}）:")
        for label, weight in sorted(label_weights.items()):
            label_name = FILTER_OPTIMAL_LABEL_NAMES.get(label, f"标签{label}")
            print(f"  {label_name} (标签{label}): {weight:.2f}")
        print(f"总交集体积权重: {total_volume_weight:.2f}")
        
        scored_paths = []
        
        for result in results:
            weighted_volume = 0.0
            intersection_volumes = result.get('intersection_volumes_mm3', {})
            
            for label, weight in label_weights.items():
                label_str = str(label)
                label_int = int(label)
                volume = intersection_volumes.get(label_str, 
                                                 intersection_volumes.get(label_int, 0.0))
                weighted_volume += volume * weight
            
            total_volume = float(_path_total_non_background_volume_mm3(result))
            weighted_volume += total_volume * total_volume_weight
            
            scored_path = result.copy()
            scored_path['weighted_score'] = weighted_volume
            scored_path['label_intersection_sum'] = sum(
                intersection_volumes.get(str(label), 
                                        intersection_volumes.get(int(label), 0.0))
                for label in label_weights.keys()
            )
            scored_paths.append(scored_path)
        
        scored_paths.sort(key=lambda x: x['weighted_score'])
        optimal_paths = scored_paths[:top_k]

        use_ab = results_use_spherical_alpha_beta_axes(results)
        ang1_name = "α(°)" if use_ab else "CPA角度"
        ang2_name = "β(°)" if use_ab else "CSA角度"
        
        print(f"\n筛选完成，找到 {len(optimal_paths)} 个最优路径（得分从低到高）:")
        print("-" * 100)
        print(
            f"{'排名':<6} {'通道ID':<10} {'加权得分':<15} {'标签交集体积':<20} {'总交集体积':<20} "
            f"{ang1_name:<12} {ang2_name:<12}"
        )
        print("-" * 100)
        
        excel_data = []
        label_names = dict(FILTER_OPTIMAL_LABEL_NAMES)
        
        for idx, path in enumerate(optimal_paths, 1):
            if use_ab:
                cpa = float(
                    path.get(
                        "grid_spherical_alpha_deg",
                        path.get(
                            "target_angle_coronal_deg",
                            path.get("angle_with_coronal_deg", 0.0),
                        ),
                    )
                )
                csa = float(
                    path.get(
                        "grid_spherical_beta_deg",
                        path.get(
                            "target_angle_transverse_deg",
                            path.get("angle_with_transverse_deg", 0.0),
                        ),
                    )
                )
            else:
                cpa = float(path.get('target_angle_coronal_deg', path.get('angle_with_coronal_deg', 0.0)))
                csa = float(path.get('target_angle_transverse_deg', path.get('angle_with_transverse_deg', 0.0)))
            
            print(f"{idx:<6} {path['channel_id']:<10} {path['weighted_score']:<15.2f} "
                  f"{path['label_intersection_sum']:<20.2f} {path['total_intersection_volume_mm3']:<20.2f} "
                  f"{cpa:<12.1f} {csa:<12.1f}")
            
            intersection_volumes = path.get('intersection_volumes_mm3', {})
            row_data = {
                '排名': idx,
                '通道ID': path['channel_id'],
                '加权得分': round(path['weighted_score'], 2),
                '总交集体积': round(path['total_intersection_volume_mm3'], 2),
                ang1_name: round(cpa, 1),
                ang2_name: round(csa, 1),
            }
            for label in sorted(label_weights.keys()):
                label_str = str(label)
                label_int = int(label)
                volume = intersection_volumes.get(label_str, 
                                                 intersection_volumes.get(label_int, 0.0))
                label_name = label_names.get(label, f"标签{label}")
                row_data[label_name] = round(volume, 2)
            excel_data.append(row_data)
        
        print("-" * 100)
        
        if PANDAS_AVAILABLE and excel_data:
            try:
                df = pd.DataFrame(excel_data)
                if hasattr(self, '_last_output_path') and self._last_output_path:
                    base_name = os.path.splitext(self._last_output_path)[0]
                    excel_filename = f"{base_name}_最优路径排名.xlsx"
                else:
                    excel_filename = "最优路径排名.xlsx"
                excel_dir = os.path.dirname(excel_filename) if os.path.dirname(excel_filename) else '.'
                if excel_dir and not os.path.exists(excel_dir):
                    os.makedirs(excel_dir, exist_ok=True)
                df.to_excel(excel_filename, index=False, engine='openpyxl')
                print(f"\n[OK] 数据已导出到Excel表格: {excel_filename}")
            except Exception as e:
                print(f"\n警告: 导出Excel表格时出错: {str(e)}")
                print("   提示: 请确保已安装pandas和openpyxl库: pip install pandas openpyxl")
        elif not PANDAS_AVAILABLE:
            print(f"\n警告: pandas未安装，无法导出Excel表格")
        
        if optimal_paths:
            print(f"\n最优路径（第1名）:")
            best_path = optimal_paths[0]
            print(f"  通道ID: {best_path['channel_id']}")
            print(f"  加权得分: {best_path['weighted_score']:.2f}")
            print(f"  起点: [{best_path['start_point'][0]:.2f}, {best_path['start_point'][1]:.2f}, {best_path['start_point'][2]:.2f}] mm")
            print(f"  终点: [{best_path['end_point'][0]:.2f}, {best_path['end_point'][1]:.2f}, {best_path['end_point'][2]:.2f}] mm")
            if use_ab:
                a1 = float(
                    best_path.get(
                        "grid_spherical_alpha_deg",
                        best_path.get(
                            "target_angle_coronal_deg",
                            best_path.get("angle_with_coronal_deg", 0.0),
                        ),
                    )
                )
                a2 = float(
                    best_path.get(
                        "grid_spherical_beta_deg",
                        best_path.get(
                            "target_angle_transverse_deg",
                            best_path.get("angle_with_transverse_deg", 0.0),
                        ),
                    )
                )
                print(f"  α: {a1:.1f}°  β: {a2:.1f}°（球面角网格）")
            else:
                print(
                    f"  CPA角度: {best_path.get('target_angle_coronal_deg', best_path.get('angle_with_coronal_deg', 0.0)):.1f}°"
                )
                print(
                    f"  CSA角度: {best_path.get('target_angle_transverse_deg', best_path.get('angle_with_transverse_deg', 0.0)):.1f}°"
                )
            print(f"  各标签交集体积:")
            intersection_volumes = best_path.get('intersection_volumes_mm3', {})
            for label in sorted(label_weights.keys()):
                label_str = str(label)
                label_int = int(label)
                volume = intersection_volumes.get(label_str, intersection_volumes.get(label_int, 0.0))
                label_name = FILTER_OPTIMAL_LABEL_NAMES.get(label, f"标签{label}")
                print(f"    {label_name} (标签{label}): {volume:.2f} mm³")
        
        return optimal_paths

    def filter_optimal_paths(self,
                            results: List[Dict],
                            label_weights: Optional[Dict[int, float]] = None,
                            total_volume_weight: float = 0.3,
                            top_k: int = 10,
                            iliac_max_intersection_mm3: float = 0.0,
                            iliac_gate_mode: str = "ipsilateral",
                            iliac_zero_eps_mm3: float = DEFAULT_ILIAC_ZERO_EPS_MM3,
                            use_legacy_weighted_ranking: bool = False,
                            ) -> List[Dict]:
        """
        筛选最优路径（默认策略）

        采用「髂骨无碰撞可行约束 + 多指标字典序排序」（非旧版加权时）：
        1) **可行性**：默认仅考察 **入路侧髂骨**（左入路标签 18，右入路标签 19）交集体积。
           当 ``iliac_max_intersection_mm3 <= 0`` 时，启用 **严格零碰撞**：仅
           ``同侧髂骨体积 ≤ iliac_zero_eps_mm3`` 的候选路径进入排序（数值上将微小体积视为未命中髂骨）。
           若 ``iliac_max_intersection_mm3 > 0``，则退化为宽松门槛：
           ``髂骨聚合值 < iliac_max_intersection_mm3``（聚合方式由 ``iliac_gate_mode`` 决定）。
           严格零碰撞模式下 ``iliac_gate_mode`` 强制为 ``ipsilateral``。
        2) **字典序（一律越小越优）**：相关椎体关节突交集体积和 → 不含标签 0 的总交集体积 →
           同侧多裂肌 → 竖脊肌 → 腰大肌 → 腰方肌。
           （第二键与 JSON ``total_intersection_volume_mm3`` 区分：后者含背景；排序用不含标签 0 之和。）

        旧版加权：``use_legacy_weighted_ranking=True`` 时改用 ``filter_optimal_paths_weighted_legacy``
        （不设髂骨可行性筛选；行为与旧版一致）。

        Args:
            results: 通道结果列表
            label_weights: 仅旧版加权生效；默认策略下若传入会提示忽略
            total_volume_weight: 仅旧版加权生效
            top_k: 返回前 K 条通过髂骨门槛后的最优路径
            iliac_max_intersection_mm3: 宽松模式下髂骨聚合须小于该正数（mm³）；≤0 时表示严格同侧髂骨零碰撞（见 iliac_zero_eps_mm3）
            iliac_gate_mode: sum / max / ipsilateral（严格零碰撞时强制 ipsilateral）
            iliac_zero_eps_mm3: 严格模式下判定「髂骨体积为 0」的数值容差（mm³）
            use_legacy_weighted_ranking: 使用旧版 Σ(体积×权重) 得分

        Returns:
            optimal_paths: 最优路径列表（默认按字典序）
        """
        if use_legacy_weighted_ranking:
            return self.filter_optimal_paths_weighted_legacy(
                results, label_weights=label_weights,
                total_volume_weight=total_volume_weight, top_k=top_k,
            )

        if not results:
            print("警告: 没有结果数据，无法筛选最优路径")
            return []

        use_ab = results_use_spherical_alpha_beta_axes(results)
        la = 'α' if use_ab else 'CPA'
        lb = 'β' if use_ab else 'CSA'
        excel_ca = 'α(°)' if use_ab else 'CPA角度'
        excel_cb = 'β(°)' if use_ab else 'CSA角度'

        if label_weights is not None:
            print(
                "\n提示: 当前默认最优路径策略已改为「同侧髂骨零碰撞可行约束 + 多指标字典序」；"
                "已忽略 --label_weights。如需旧版请加 --filter_optimal_legacy_weights。"
            )

        side = getattr(self, "_paraspinal_side", None)
        if side not in ("left", "right"):
            side = "left"

        facet_ids = self._optimal_facet_label_ids_for_segment()
        gate_mode_req = (iliac_gate_mode or "ipsilateral").strip().lower()
        eps = float(iliac_zero_eps_mm3)
        cap = float(iliac_max_intersection_mm3)
        strict_ipsilateral_zero = cap <= 0.0

        if strict_ipsilateral_zero:
            gate_mode_eff = "ipsilateral"
            if gate_mode_req != "ipsilateral":
                print(
                    "\n提示: 髂骨可行约束为「同侧髂骨数值零」（iliac_max_intersection_mm3≤0），"
                    f"聚合模式已强制 ipsilateral（忽略所请求的 {gate_mode_req}）。"
                )
        else:
            gate_mode_eff = gate_mode_req

        mode_expl = (
            f"同侧髂骨须≤{eps:g} mm³（严格零碰撞）"
            if strict_ipsilateral_zero
            else f"聚合模式={gate_mode_eff}，须 < {cap:.6f} mm³（宽松门槛）"
        )

        print(f"\n开始筛选最优路径（髂骨可行约束 + 多指标字典序）…")
        print(
            f"  节段={getattr(self, '_current_segment', '?')} | 入路侧={side} | "
            f"关节突相关标签(交集体积已用关节突掩码)={facet_ids}"
        )
        print(f"  髂骨可行约束: {mode_expl}（左髂=18，右髂=19）")
        print(
            "  排序优先级（小者优先）: 关节突椎体交集体积和 → 不含背景的总交集体积 → "
            "同侧多裂肌 → 竖脊肌 → 腰大肌 → 腰方肌"
        )

        scored_paths: List[Dict] = []
        for result in results:
            vols = result.get("intersection_volumes_mm3") or {}
            iliac_g = self._iliac_aggregate_mm3_for_gate(vols, gate_mode_eff, side)
            facet_sum = float(sum(
                intersection_volume_mm3_for_label(vols, lb) for lb in facet_ids
            ))
            total_v = float(_path_total_non_background_volume_mm3(result))
            mf, es, pm, ql = self._ipsilateral_paraspinal_rank_volumes(vols, side)
            rank_tuple = (facet_sum, total_v, mf, es, pm, ql)

            if strict_ipsilateral_zero:
                passes = iliac_g <= eps
            else:
                passes = iliac_g < cap

            out = result.copy()
            out["iliac_aggregate_mm3"] = iliac_g
            out["iliac_gate_mode"] = gate_mode_eff
            out["iliac_collision_free_strict"] = strict_ipsilateral_zero
            out["iliac_zero_eps_mm3"] = eps
            out["passes_iliac_gate"] = passes
            out["facet_intersection_sum_mm3"] = facet_sum
            out["optimal_rank_tuple"] = list(rank_tuple)
            out["weighted_score"] = float(facet_sum)
            out["label_intersection_sum"] = float(facet_sum)
            scored_paths.append(out)

        eligible = [p for p in scored_paths if p.get("passes_iliac_gate")]
        ineligible_n = len(scored_paths) - len(eligible)

        if ineligible_n:
            print(
                f"\n  未过髂骨门槛的路径: {ineligible_n}/{len(scored_paths)} "
                f"（不参与最优排名）"
            )
        if not eligible:
            print(
            "\n警告: 无任何路径满足髂骨可行约束；返回列表为空。"
            "\n  若为严格「同侧髂骨零碰撞」，请检查网格是否过粗或对侧掩码；"
            "\n  若需放宽，可设 --iliac_max_intersection_mm3 为正数（如 5），并按需调整 --iliac_gate。"
            )
            return []

        eligible.sort(key=lambda x: tuple(x["optimal_rank_tuple"]))
        optimal_paths = eligible[:top_k]

        print(f"\n筛选完成，最优路径 {len(optimal_paths)} 条（髂骨达标，字典序升序）:")
        print("-" * 120)
        hdr = (
            f"{'排名':<6} {'通道ID':<10} {'髂骨聚合mm³':<14} {'关节突和':<12} {'非背景总体积':<12} "
            f"{'多裂':<10} {'竖脊':<10} {'腰大':<10} {'腰方':<10} {f'{la}':<8} {f'{lb}':<8}"
        )
        print(hdr)
        print("-" * 120)

        excel_data = []
        for idx, path in enumerate(optimal_paths, 1):
            cpa = path.get('target_angle_coronal_deg', path.get('angle_with_coronal_deg', 0.0))
            csa = path.get('target_angle_transverse_deg', path.get('angle_with_transverse_deg', 0.0))
            rt = path["optimal_rank_tuple"]
            print(
                f"{idx:<6} {path['channel_id']:<10} {path['iliac_aggregate_mm3']:<14.4f} "
                f"{rt[0]:<12.2f} {rt[1]:<12.2f} {rt[2]:<10.2f} {rt[3]:<10.2f} {rt[4]:<10.2f} {rt[5]:<10.2f} "
                f"{cpa:<8.1f} {csa:<8.1f}"
            )
            row_data = {
                '排名': idx,
                '通道ID': path['channel_id'],
                '髂骨聚合_mm3': round(path['iliac_aggregate_mm3'], 4),
                '通过髂骨门槛': path['passes_iliac_gate'],
                '关节突椎体交集体积和_mm3': round(rt[0], 2),
                '不含背景交集体积和_mm3': round(rt[1], 2),
                '同侧多裂肌_mm3': round(rt[2], 2),
                '同侧竖脊肌_mm3': round(rt[3], 2),
                '同侧腰大肌_mm3': round(rt[4], 2),
                '同侧腰方肌_mm3': round(rt[5], 2),
                '字典序_key': [round(x, 4) for x in rt],
                excel_ca: round(cpa, 1),
                excel_cb: round(csa, 1),
            }
            excel_data.append(row_data)

        print("-" * 120)

        if PANDAS_AVAILABLE and excel_data:
            try:
                if hasattr(self, '_last_output_path') and self._last_output_path:
                    base_name = os.path.splitext(self._last_output_path)[0]
                    excel_filename = f"{base_name}_最优路径排名.xlsx"
                else:
                    excel_filename = "最优路径排名.xlsx"
                excel_dir = os.path.dirname(excel_filename) if os.path.dirname(excel_filename) else '.'
                if excel_dir and not os.path.exists(excel_dir):
                    os.makedirs(excel_dir, exist_ok=True)
                pd.DataFrame(excel_data).to_excel(excel_filename, index=False, engine='openpyxl')
                print(f"\n[OK] 数据已导出到Excel表格: {excel_filename}")
            except Exception as e:
                print(f"\n警告: 导出Excel表格时出错: {str(e)}")
        elif not PANDAS_AVAILABLE:
            print(f"\n警告: pandas未安装，无法导出Excel表格")

        if optimal_paths:
            best_path = optimal_paths[0]
            rt = best_path["optimal_rank_tuple"]
            print(f"\n最优路径（第1名）:")
            print(f"  通道ID: {best_path['channel_id']}")
            if strict_ipsilateral_zero:
                print(
                    f"  同侧髂骨交集体积: {best_path['iliac_aggregate_mm3']:.4f} mm³ "
                    f"（严格零碰撞：须≤{eps:g} mm³）"
                )
            else:
                print(
                    f"  髂骨聚合（{gate_mode_eff}）: {best_path['iliac_aggregate_mm3']:.4f} mm³ "
                    f"（宽松门槛：须<{cap:.4f} mm³）"
                )
            print(f"  排序分量: 关节突和={rt[0]:.2f}, 不含背景总交集体积={rt[1]:.2f}, "
                  f"多裂/竖脊/腰大/腰方={rt[2]:.2f}/{rt[3]:.2f}/{rt[4]:.2f}/{rt[5]:.2f} mm³")
            print(f"  起点: [{best_path['start_point'][0]:.2f}, {best_path['start_point'][1]:.2f}, {best_path['start_point'][2]:.2f}] mm")
            print(f"  终点: [{best_path['end_point'][0]:.2f}, {best_path['end_point'][1]:.2f}, {best_path['end_point'][2]:.2f}] mm")
            print(f"  {la}角: {best_path.get('target_angle_coronal_deg', best_path.get('angle_with_coronal_deg', 0.0)):.1f}°")
            print(f"  {lb}角: {best_path.get('target_angle_transverse_deg', best_path.get('angle_with_transverse_deg', 0.0)):.1f}°")

        return optimal_paths


ANGLE_AVERAGE_METHODS_NOTE_ZH = (
    "【方法学说明 / 与论文统计表述对应】\n"
    "0) Total 行：为 intersection_volumes_mm3 中除标签 0（背景）外所有标签体积之和；"
    "与 JSON 字段 total_intersection_volume_mm3（含背景）不同，便于与解剖结构列对照。\n"
    "1) mean_mm3：各 (CPA, CSA) 分桶内，对所有路径的交集体积取算术平均（mm³）。\n"
    "2) std_mm3：同一分桶内样本标准差（ddof=1，n<2 时为 0），单位 mm³。\n"
    "3) mean_pm_SD：以「均值 ± 标准差」形式便于直接写入正文或表格（工作表名 ASCII，避免 Excel 编码问题）。\n"
    "4) 分层 Spearman（与「固定 CSA 看 CPA」「固定 CPA 看 CSA」叙述一致）：\n"
    "   - spearman_CPA_given_CSA：在每个固定的 CSA 水平（0°,5°,…，以数据中出现的分桶为准）下，"
    "仅使用该 CSA 上的全部路径，将 CPA（分桶角度）与交集体积做 Spearman 相关，得到一组 ρ 与 p。\n"
    "   - spearman_CSA_given_CPA：在每个固定的 CPA 水平下，将 CSA 与体积做 Spearman 相关。\n"
    "   - rho_CPA_wide / rho_CSA_wide：将 ρ 制成宽表，便于画热图（行为 Structure，列为固定角度）。\n"
    "   - spearman_rho_range：各结构在分层相关中 ρ 的最小值与最大值，便于正文报告「相关系数范围」。\n"
    "ρ 即秩相关系数（论文中常记为 r）；p 为双侧近似 p 值（scipy.stats.spearmanr）。"
    "若某一层内 CPA（或 CSA）无变异或样本过少，ρ/p 可能为 NaN。\n"
    "\n"
    "【球面 α×β 网格】若规划使用 angle_grid_mode=spherical_alpha_beta，则 mean_mm3 / std_mm3 / Spearman 分桶的\n"
    "「第一角度轴」对应 α（°）、「第二角度轴」对应 β（°）；mean 等表中「CPA」列名为 α、「CSA」列名为 β。\n"
    "Spearman 等工作表文件名仍含 CPA/CSA 字样以便旧脚本解析，数值含义同上（α/β）。\n"
)


def _path_label_volume_mm3(path: Dict, lbl: int) -> float:
    vol = path.get('intersection_volumes_mm3', {}) or {}
    lbl_str = str(lbl)
    v = vol.get(lbl_str, vol.get(lbl, 0.0))
    return float(v) if v is not None else 0.0


def _path_total_non_background_volume_mm3(path: Dict) -> float:
    """
    Sum of intersection_volumes_mm3 over labels other than 0 (background).
    Used for angle-average Excel / Spearman / total chart so Total matches anatomical content.
    """
    vol = path.get('intersection_volumes_mm3') or {}
    s = 0.0
    for k, v in vol.items():
        try:
            ki = int(k) if not isinstance(k, int) else int(k)
        except (ValueError, TypeError):
            continue
        if ki == 0:
            continue
        try:
            s += float(v)
        except (TypeError, ValueError):
            pass
    return s


def _bucket_cpa_deg(path: Dict, angle_step_deg: float) -> float:
    cpa = float(path.get('target_angle_coronal_deg', path.get('angle_with_coronal_deg', 0.0)))
    return round(cpa / angle_step_deg) * angle_step_deg


def _bucket_csa_deg(path: Dict, angle_step_deg: float) -> float:
    csa = float(path.get('target_angle_transverse_deg', path.get('angle_with_transverse_deg', 0.0)))
    return round(csa / angle_step_deg) * angle_step_deg


def _volume_metrics(ordered_labels: List[int]) -> List[Tuple[str, Callable[[Dict], float]]]:
    labs = list(ordered_labels)
    out: List[Tuple[str, Callable[[Dict], float]]] = [
        (
            ANGLE_AVERAGE_PLANNING_TOTAL_ROW,
            lambda p, _labs=labs: float(
                sum(_path_label_volume_mm3(p, lb) for lb in _labs)
            ),
        ),
    ]
    for lbl in ordered_labels:
        name = _angle_avg_structure_display_name(int(lbl))
        out.append((name, lambda p, lb=lbl: _path_label_volume_mm3(p, lb)))
    return out


def _spearman_stratified_cpa_given_csa(
    paths: List[Dict],
    angle_step_deg: float,
    ordered_labels: List[int],
    csa_levels: List[float],
) -> List[Dict]:
    """固定 CSA 分层：CPA vs 体积。"""
    rows: List[Dict] = []
    if not paths:
        return rows
    metrics = _volume_metrics(ordered_labels)
    for csa_fixed in csa_levels:
        sub = [p for p in paths if _bucket_csa_deg(p, angle_step_deg) == csa_fixed]
        for mname, vol_fn in metrics:
            cpas = [_bucket_cpa_deg(p, angle_step_deg) for p in sub]
            vols = [vol_fn(p) for p in sub]
            n = len(sub)
            rho = pval = np.nan
            if n >= 2 and SPEARMAN_AVAILABLE and spearmanr is not None:
                try:
                    rho, pval = spearmanr(cpas, vols)
                    rho, pval = float(rho), float(pval)
                except Exception:
                    pass
            rows.append({
                'Structure': mname,
                'CSA_fixed_deg': int(csa_fixed),
                'n': n,
                'Spearman_rho_CPA': rho,
                'p_value': pval,
            })
    return rows


def _spearman_stratified_csa_given_cpa(
    paths: List[Dict],
    angle_step_deg: float,
    ordered_labels: List[int],
    cpa_levels: List[float],
) -> List[Dict]:
    """固定 CPA 分层：CSA vs 体积。"""
    rows: List[Dict] = []
    if not paths:
        return rows
    metrics = _volume_metrics(ordered_labels)
    for cpa_fixed in cpa_levels:
        sub = [p for p in paths if _bucket_cpa_deg(p, angle_step_deg) == cpa_fixed]
        for mname, vol_fn in metrics:
            csas = [_bucket_csa_deg(p, angle_step_deg) for p in sub]
            vols = [vol_fn(p) for p in sub]
            n = len(sub)
            rho = pval = np.nan
            if n >= 2 and SPEARMAN_AVAILABLE and spearmanr is not None:
                try:
                    rho, pval = spearmanr(csas, vols)
                    rho, pval = float(rho), float(pval)
                except Exception:
                    pass
            rows.append({
                'Structure': mname,
                'CPA_fixed_deg': int(cpa_fixed),
                'n': n,
                'Spearman_rho_CSA': rho,
                'p_value': pval,
            })
    return rows


def _rho_range_summary(
    df_cpa_given_csa: "pd.DataFrame",
    df_csa_given_cpa: "pd.DataFrame",
) -> "pd.DataFrame":
    """各结构在分层 ρ 上的 min/max（忽略 NaN）。"""
    if df_cpa_given_csa.empty and df_csa_given_cpa.empty:
        return pd.DataFrame()
    rows = []
    structures = set()
    if not df_cpa_given_csa.empty and 'Structure' in df_cpa_given_csa.columns:
        structures.update(df_cpa_given_csa['Structure'].unique())
    if not df_csa_given_cpa.empty and 'Structure' in df_csa_given_cpa.columns:
        structures.update(df_csa_given_cpa['Structure'].unique())
    for s in sorted(structures, key=lambda x: str(x)):
        r_cpa = df_cpa_given_csa.loc[df_cpa_given_csa['Structure'] == s, 'Spearman_rho_CPA'] if not df_cpa_given_csa.empty else pd.Series(dtype=float)
        r_csa = df_csa_given_cpa.loc[df_csa_given_cpa['Structure'] == s, 'Spearman_rho_CSA'] if not df_csa_given_cpa.empty else pd.Series(dtype=float)
        r_cpa = pd.to_numeric(r_cpa, errors='coerce').dropna()
        r_csa = pd.to_numeric(r_csa, errors='coerce').dropna()
        rows.append({
            'Structure': s,
            'min_rho_CPA_given_CSA': float(r_cpa.min()) if len(r_cpa) else np.nan,
            'max_rho_CPA_given_CSA': float(r_cpa.max()) if len(r_cpa) else np.nan,
            'min_rho_CSA_given_CPA': float(r_csa.min()) if len(r_csa) else np.nan,
            'max_rho_CSA_given_CPA': float(r_csa.max()) if len(r_csa) else np.nan,
        })
    return pd.DataFrame(rows)


def _reorder_spearman_rho_wide_total_first(df: "pd.DataFrame") -> "pd.DataFrame":
    """热图 Y 轴：合计行置顶（与论文总交集体积习惯一致）。"""
    if df is None or df.empty:
        return df
    total_key = ANGLE_AVERAGE_PLANNING_TOTAL_ROW
    # index 可能是 object，与常量逐元素比较
    match = None
    for i in df.index:
        if str(i).strip() == total_key:
            match = i
            break
    if match is None:
        return df
    rest = [i for i in df.index if i != match]
    return df.reindex([match] + rest)


def _try_save_stratified_spearman_heatmaps(
    excel_path: str,
    df_rho_cpa_wide: "pd.DataFrame",
    df_rho_csa_wide: "pd.DataFrame",
    *,
    use_alpha_beta_axes: bool = False,
) -> None:
    """在 xlsx 同目录输出 ρ 热图（需 matplotlib）。"""
    if not MATPLOTLIB_AVAILABLE:
        return
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return
    base = os.path.splitext(excel_path)[0]
    if use_alpha_beta_axes:
        meta = (
            (
                df_rho_cpa_wide,
                'Fixed β (deg)',
                r'Spearman $\rho$ ($\alpha$ | fixed $\beta$)',
                'alpha_given_beta',
            ),
            (
                df_rho_csa_wide,
                'Fixed α (deg)',
                r'Spearman $\rho$ ($\beta$ | fixed $\alpha$)',
                'beta_given_alpha',
            ),
        )
    else:
        meta = (
            (
                df_rho_cpa_wide,
                'Fixed CSA (deg)',
                'Spearman rho (CPA given fixed CSA)',
                'CPA_given_CSA',
            ),
            (
                df_rho_csa_wide,
                'Fixed CPA (deg)',
                'Spearman rho (CSA given fixed CPA)',
                'CSA_given_CPA',
            ),
        )
    for df, xlab, title, fname in meta:
        if df is None or df.empty:
            continue
        try:
            df = _reorder_spearman_rho_wide_total_first(df)
            arr = df.astype(float).values
            arr_plot = np.ma.masked_invalid(arr)
            fig, ax = plt.subplots(figsize=(max(6, 0.35 * arr.shape[1] + 4), max(4, 0.35 * arr.shape[0] + 2)))
            im = ax.imshow(arr_plot, aspect='auto', cmap='RdBu_r', vmin=-1, vmax=1)
            ax.set_xticks(np.arange(arr.shape[1]))
            ax.set_yticks(np.arange(arr.shape[0]))
            ax.set_xticklabels([str(c) for c in df.columns], rotation=45, ha='right', fontsize=8)
            y_en = [_spearman_heatmap_row_label_en(str(i)) for i in df.index]
            ax.set_yticklabels(y_en, fontsize=8)
            ax.set_xlabel(xlab)
            ax.set_ylabel('Structure')
            ax.set_title(title)
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label='rho')
            outp = f'{base}_spearman_rho_{fname}_heatmap.png'
            fig.savefig(outp, dpi=150, bbox_inches='tight', pad_inches=0.35)
            plt.close(fig)
        except Exception:
            try:
                plt.close('all')
            except Exception:
                pass


def _write_angle_average_sheet(paths: List[Dict],
                               output_path: str,
                               angle_step_deg: float,
                               segment_name: str,
                               allowed_labels: List[int]) -> bool:
    """
    将指定路径按 (CPA, CSA) 分桶，写入 Excel：均值、标准差、mean±SD、Spearman、方法说明。
    allowed_labels: 仅输出这些标签（与临床关注结构 chart_labels_* 用法类似）
    """
    if not paths or not PANDAS_AVAILABLE:
        return False
    use_ab = results_use_spherical_alpha_beta_axes(paths)
    col_row_cpa = 'α' if use_ab else 'CPA'
    col_hdr_prefix = 'β' if use_ab else 'CSA'
    buckets: Dict[Tuple[float, float], List[Dict]] = defaultdict(list)
    for path in paths:
        cpa = path.get('target_angle_coronal_deg', path.get('angle_with_coronal_deg', 0.0))
        csa = path.get('target_angle_transverse_deg', path.get('angle_with_transverse_deg', 0.0))
        cpa_bucket = round(float(cpa) / angle_step_deg) * angle_step_deg
        csa_bucket = round(float(csa) / angle_step_deg) * angle_step_deg
        buckets[(cpa_bucket, csa_bucket)].append(path)
    cpa_angles = sorted({p[0] for p in buckets.keys()})
    csa_angles = sorted({p[1] for p in buckets.keys()})
    if not cpa_angles or not csa_angles:
        return False
    ordered = [int(l) for l in allowed_labels]

    def cell_mean_std(plist: List[Dict], get_vol: Callable[[Dict], float]) -> Tuple[float, float]:
        vals = [get_vol(p) for p in plist]
        n = len(vals)
        if n == 0:
            return 0.0, 0.0
        m = float(np.mean(vals))
        s = float(np.std(vals, ddof=1)) if n > 1 else 0.0
        return m, s

    table_mean: Dict[Tuple[float, str], Dict[float, float]] = defaultdict(lambda: defaultdict(float))
    table_std: Dict[Tuple[float, str], Dict[float, float]] = defaultdict(lambda: defaultdict(float))

    for (cpa, csa), plist in buckets.items():
        m, s = cell_mean_std(
            plist,
            lambda p, labs=ordered: float(
                sum(_path_label_volume_mm3(p, lb) for lb in labs)
            ),
        )
        table_mean[(cpa, ANGLE_AVERAGE_PLANNING_TOTAL_ROW)][csa] = m
        table_std[(cpa, ANGLE_AVERAGE_PLANNING_TOTAL_ROW)][csa] = s
        for lbl in ordered:
            lname = _angle_avg_structure_display_name(lbl)
            m, s = cell_mean_std(plist, lambda p, lb=lbl: _path_label_volume_mm3(p, lb))
            table_mean[(cpa, lname)][csa] = m
            table_std[(cpa, lname)][csa] = s

    def build_rows(table: Dict[Tuple[float, str], Dict[float, float]], rnd: bool) -> pd.DataFrame:
        out_rows = []
        for cpa in cpa_angles:
            for metric in [ANGLE_AVERAGE_PLANNING_TOTAL_ROW] + [
                _angle_avg_structure_display_name(l) for l in ordered
            ]:
                row = {
                    col_row_cpa: f'{col_row_cpa} {int(cpa)}°' if metric == ANGLE_AVERAGE_PLANNING_TOTAL_ROW else '',
                    'Structure': metric,
                }
                for csa in csa_angles:
                    v = table[(cpa, metric)][csa]
                    row[f'{col_hdr_prefix} {int(csa)}°'] = round(v, 3) if rnd else v
                out_rows.append(row)
        return pd.DataFrame(out_rows)

    df_mean = build_rows(table_mean, rnd=True)
    df_std = build_rows(table_std, rnd=True)

    pm_rows = []
    for cpa in cpa_angles:
        for metric in [ANGLE_AVERAGE_PLANNING_TOTAL_ROW] + [
            _angle_avg_structure_display_name(l) for l in ordered
        ]:
            row = {
                col_row_cpa: f'{col_row_cpa} {int(cpa)}°' if metric == ANGLE_AVERAGE_PLANNING_TOTAL_ROW else '',
                'Structure': metric,
            }
            for csa in csa_angles:
                m = table_mean[(cpa, metric)][csa]
                s = table_std[(cpa, metric)][csa]
                row[f'{col_hdr_prefix} {int(csa)}°'] = f"{m:.3f} ± {s:.3f}"
            pm_rows.append(row)
    df_pm = pd.DataFrame(pm_rows)

    spear_rows_cpa = _spearman_stratified_cpa_given_csa(paths, angle_step_deg, ordered, csa_angles)
    spear_rows_csa = _spearman_stratified_csa_given_cpa(paths, angle_step_deg, ordered, cpa_angles)
    df_cpa_given_csa = pd.DataFrame(spear_rows_cpa)
    df_csa_given_cpa = pd.DataFrame(spear_rows_csa)

    if not df_cpa_given_csa.empty:
        df_rho_cpa_wide = df_cpa_given_csa.pivot_table(
            index='Structure',
            columns='CSA_fixed_deg',
            values='Spearman_rho_CPA',
            aggfunc='first',
        )
        df_rho_cpa_wide = df_rho_cpa_wide.reindex(sorted(df_rho_cpa_wide.columns), axis=1)
    else:
        df_rho_cpa_wide = pd.DataFrame()
    if not df_csa_given_cpa.empty:
        df_rho_csa_wide = df_csa_given_cpa.pivot_table(
            index='Structure',
            columns='CPA_fixed_deg',
            values='Spearman_rho_CSA',
            aggfunc='first',
        )
        df_rho_csa_wide = df_rho_csa_wide.reindex(sorted(df_rho_csa_wide.columns), axis=1)
    else:
        df_rho_csa_wide = pd.DataFrame()

    df_range = _rho_range_summary(df_cpa_given_csa, df_csa_given_cpa)

    methods_df = pd.DataFrame({'说明': [ANGLE_AVERAGE_METHODS_NOTE_ZH]})
    if not SPEARMAN_AVAILABLE:
        methods_df = pd.DataFrame({
            '说明': [ANGLE_AVERAGE_METHODS_NOTE_ZH + '\n（当前环境未安装 scipy.stats.spearmanr，分层相关 ρ 为 NaN。）']
        })

    try:
        with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
            df_mean.to_excel(writer, sheet_name='mean_mm3', index=False)
            df_std.to_excel(writer, sheet_name='std_mm3', index=False)
            df_pm.to_excel(writer, sheet_name='mean_pm_SD', index=False)
            df_cpa_given_csa.to_excel(writer, sheet_name='spearman_CPA_given_CSA', index=False)
            df_csa_given_cpa.to_excel(writer, sheet_name='spearman_CSA_given_CPA', index=False)
            df_rho_cpa_wide.to_excel(writer, sheet_name='rho_CPA_wide')
            df_rho_csa_wide.to_excel(writer, sheet_name='rho_CSA_wide')
            df_range.to_excel(writer, sheet_name='spearman_rho_range', index=False)
            methods_df.to_excel(writer, sheet_name='methods_note', index=False)
        _try_save_stratified_spearman_heatmaps(
            output_path, df_rho_cpa_wide, df_rho_csa_wide, use_alpha_beta_axes=use_ab
        )
        return True
    except Exception:
        return False


def export_angle_average_excel(all_planning_results: List[Tuple],
                               output_dir: str,
                               angle_step_deg: float = 5.0) -> Optional[str]:
    """
    分别生成 L4-L5 与 L5-S1 两种规划的交集体积 Excel（左/右分侧时各一份）。
    每份工作簿按 (CPA, CSA) 分桶，含：mean_mm3、std_mm3、mean_pm_SD、分层 spearman（spearman_CPA_given_CSA /
    spearman_CSA_given_CPA）、rho_CPA_wide / rho_CSA_wide、spearman_rho_range、methods_note；并尝试输出 ρ 热图 PNG。

    Args:
        all_planning_results: 每项为 (results, output_file, use_facet_joint, use_l5_s1_disc)
            或 5 元组末尾加 facet_side: 'left'|'right'|'auto'
        output_dir: 输出目录
        angle_step_deg: 角度步长（度）
    """
    if not PANDAS_AVAILABLE:
        print("\n警告: 未安装 pandas，无法导出按角度的平均交集体积 Excel。请安装: pip install pandas openpyxl")
        return None
    groups: Dict[Tuple[str, str], List[Dict]] = defaultdict(list)
    for item in all_planning_results:
        if len(item) >= 5:
            results, _, use_facet, use_l5_s1, fside = item[0], item[1], item[2], item[3], str(item[4])
        else:
            results, _, use_facet, use_l5_s1 = item[0], item[1], item[2], item[3]
            fside = 'auto'
        if fside not in ('left', 'right', 'auto'):
            fside = 'auto'
        if use_facet:
            groups[('L4-L5', fside)].extend(results)
        if use_l5_s1:
            groups[('L5-S1', fside)].extend(results)
    os.makedirs(output_dir, exist_ok=True)
    written = []
    for seg_label in ('L4-L5', 'L5-S1'):
        for fside in ('left', 'right', 'auto'):
            paths = groups.get((seg_label, fside), [])
            if not paths:
                continue
            if fside in ('left', 'right'):
                fname = f'{seg_label}_{fside}_intersection_volume_average_by_angle.xlsx'
                sheet_title = f'{seg_label} ({fside})'
            else:
                fname = f'{seg_label}_intersection_volume_average_by_angle.xlsx'
                sheet_title = seg_label
            p = os.path.join(output_dir, fname)
            lbls = (chart_labels_l4_l5_for_side(fside) if seg_label == 'L4-L5'
                    else chart_labels_l5_s1_for_side(fside))
            if _write_angle_average_sheet(paths, p, angle_step_deg, sheet_title, lbls):
                written.append(p)
                print(f"\n[OK] {sheet_title} 规划：已导出 mean/std/mean±SD/Spearman 至: {p}")
    if not written:
        print("\n警告: 没有路径数据或角度分桶为空，未生成 Excel")
        return None
    return written[0] if written else None


def _parse_plan_subset_arg(s: str):
    """将 --plan_subset 解析为 (detect_facet, detect_l5s1, side) 列表；失败返回 None。"""
    if not s or not str(s).strip():
        return None
    token_map = {
        'l4_l5_left': (True, False, 'left'),
        'l4_l5_right': (True, False, 'right'),
        'l5_s1_left': (False, True, 'left'),
        'l5_s1_right': (False, True, 'right'),
    }
    specs = []
    seen_keys = set()
    for part in s.split(','):
        key = part.strip().lower()
        if not key:
            continue
        if key not in token_map:
            print(f"错误: --plan_subset 含无效项 {part!r}，允许: {', '.join(token_map)}")
            return None
        if key in seen_keys:
            continue
        seen_keys.add(key)
        specs.append(token_map[key])
    if not specs:
        print("错误: --plan_subset 未解析到任何有效任务")
        return None
    return specs


def _stem_key_for_plan_task(use_facet_joint: bool, use_l5_s1_disc: bool, facet_for_plan: str) -> str:
    seg = 'l4_l5' if use_facet_joint else 'l5_s1'
    return f'{seg}_{facet_for_plan}'.lower()


def _external_mrk_filename_candidates(stem_key: str) -> List[str]:
    """stem_key 如 l4_l5_left → 优先 Slicer 命名 L4-L5_left.mrk.json，兼容 l4_l5_left.mrk.json。"""
    sk = stem_key.lower()
    names = [f'{sk}.mrk.json']
    # 前缀 l4_l5_ / l5_s1_ 均为 6 个字符，应用 [6:]；误用 [7:] 会把 left→eft、right→ight
    if sk.startswith('l4_l5_'):
        names.insert(0, f'L4-L5_{sk[6:]}.mrk.json')
        # 兼容旧版错误的 off-by-one 导出文件名
        if sk == 'l4_l5_left':
            names.append('L4-L5_eft.mrk.json')
        elif sk == 'l4_l5_right':
            names.append('L4-L5_ight.mrk.json')
    elif sk.startswith('l5_s1_'):
        names.insert(0, f'L5-S1_{sk[6:]}.mrk.json')
        if sk == 'l5_s1_left':
            names.append('L5-S1_eft.mrk.json')
        elif sk == 'l5_s1_right':
            names.append('L5-S1_ight.mrk.json')
    return names


def _read_mrk_lps_position(path: str) -> np.ndarray:
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    markups = data.get('markups') or []
    for m in markups:
        mtype = (m.get('type') or '').strip()
        cps = m.get('controlPoints') or []
        if mtype == 'Fiducial' and cps:
            pos = cps[0].get('position')
            if pos is not None and len(pos) >= 3:
                return np.array([float(pos[0]), float(pos[1]), float(pos[2])], dtype=float)
        if mtype == 'Line' and len(cps) >= 2:
            pos = cps[1].get('position')
            if pos is not None and len(pos) >= 3:
                return np.array([float(pos[0]), float(pos[1]), float(pos[2])], dtype=float)
    raise ValueError(f'未在文件中找到 Fiducial 或 Line 的有效 position: {path}')


def _lps_mm_to_ras_mm(lps: np.ndarray) -> np.ndarray:
    z = np.asarray(lps, dtype=float).reshape(-1)[:3]
    return np.array([-z[0], -z[1], z[2]], dtype=float)


def _resolve_external_mrk_path(search_dir: str, stem_key: str) -> Optional[str]:
    for name in _external_mrk_filename_candidates(stem_key):
        p = os.path.join(search_dir, name)
        if os.path.isfile(p):
            return p
    return None


def _load_target_ras_from_external_dir(
    external_root: str,
    patient_key: str,
    multi_patient: bool,
    use_facet_joint: bool,
    use_l5_s1_disc: bool,
    facet_for_plan: str,
) -> Optional[np.ndarray]:
    stem_key = _stem_key_for_plan_task(use_facet_joint, use_l5_s1_disc, facet_for_plan)
    search_dir = external_root
    if multi_patient:
        sub = os.path.join(external_root, patient_key)
        if os.path.isdir(sub):
            search_dir = sub
    path = _resolve_external_mrk_path(search_dir, stem_key)
    if not path:
        tried = ', '.join(_external_mrk_filename_candidates(stem_key))
        print(f"错误: 在 {search_dir} 未找到 {stem_key} 对应文件（尝试过: {tried}）")
        return None
    try:
        lps = _read_mrk_lps_position(path)
        ras = _lps_mm_to_ras_mm(lps)
        print(f"  外部靶点: {path} → RAS [{ras[0]:.2f}, {ras[1]:.2f}, {ras[2]:.2f}] mm")
        return ras
    except Exception as e:
        print(f"错误: 读取外部靶点失败 {path}: {e}")
        return None



def all_paths_intersections_excel_path(results_json_path: str) -> str:
    """与 export_all_paths_intersection_excel 一致的 Excel 路径。"""
    base = os.path.splitext(os.path.abspath(results_json_path))[0]
    return f"{base}_all_paths_intersections.xlsx"


def main():
    """主函数"""
    parser = argparse.ArgumentParser(
        description='手术工作通道路径规划算法',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用示例:
  # 基本用法：生成100个随机通道
  python path_planning_algorithm.py --input segmentation.nii.gz --output results.json
  
  # 指定通道参数
  python path_planning_algorithm.py --input segmentation.nii.gz --output results.json \\
      --radius 4 --length 150 --num_channels 200
  
  # 只统计特定标签
  python path_planning_algorithm.py --input segmentation.nii.gz --output results.json \\
      --labels 1 2 3
  
  # 设置随机种子以便复现
  python path_planning_algorithm.py --input segmentation.nii.gz --output results.json \\
      --seed 42
  
  # 指定固定的穿刺靶点（终点）
  python path_planning_algorithm.py --input segmentation.nii.gz --output results.json \\
      --target_point 100.0 150.0 200.0
  
  # 自动识别L5-S1交叉点作为终点（推荐）
  python path_planning_algorithm.py --input segmentation.nii.gz --output results.json \\
      --auto_detect_target
  
  # 自动识别并指定L5和S1的标签值
  python path_planning_algorithm.py --input segmentation.nii.gz --output results.json \\
      --auto_detect_target --l5_label 5 --s1_label 6
  
  # 自动识别L4-L5双侧小关节内侧缘投影点
  python path_planning_algorithm.py --input segmentation.nii.gz --output results.json \\
      --detect_facet_joint --l4_label 4 --l5_label 5
  
  # L5-S1椎体上下后缘与椎间盘中位线交叉点作为靶点
  python path_planning_algorithm.py --input segmentation.nii.gz --output results.json \\
      --detect_l5_s1_disc --l5_label 5 --s1_label 6
  
  # 使用角度网格模式（每5度生成一个角度）
  python path_planning_algorithm.py --input segmentation.nii.gz --output results.json \\
      --use_angle_grid
  
  # 使用角度网格模式，指定角度间隔为5度，角度范围为0-10度
  python path_planning_algorithm.py --input segmentation.nii.gz --output results.json \\
      --use_angle_grid --angle_step 5.0 --min_angle 0.0 --max_angle 10.0
  
  # 筛选最优路径（默认：同侧髂骨数值零碰撞 + 字典序；宽松时用 --iliac_max_intersection_mm3 为正）
  python path_planning_algorithm.py --input segmentation.nii.gz --output results.json \\
      --use_angle_grid --filter_optimal --top_k 10
  
  # 旧版：按 Σ(体积×权重) 筛选时需加 --filter_optimal_legacy_weights
  python path_planning_algorithm.py --input segmentation.nii.gz --output results.json \\
      --use_angle_grid --filter_optimal --filter_optimal_legacy_weights --top_k 10 \\
      --label_weights '{"4":0.4,"5":0.4,"14":0.3,"15":0.3,"16":0.3,"17":0.4}' \\
      --total_volume_weight 0.2
  
  # 导出为3D Slicer格式（Markups JSON，推荐）
  python path_planning_algorithm.py --input segmentation.nii.gz --output results.json \\
      --export_slicer
  
  # 导出为3D Slicer FCSV格式
  python path_planning_algorithm.py --input segmentation.nii.gz --output results.json \\
      --export_slicer --slicer_format fcsv
  
  # 指定3D Slicer导出文件路径
  python path_planning_algorithm.py --input segmentation.nii.gz --output results.json \\
      --export_slicer --slicer_output paths.mrk.json
  
  # 只导出前10个路径
  python path_planning_algorithm.py --input segmentation.nii.gz --output results.json \\
      --export_slicer --max_export_paths 10
        """
    )
    
    parser.add_argument('--input', type=str, required=True,
                       help='输入的分割数据文件路径（NII.GZ格式），或包含多个NII.GZ mask文件的文件夹路径')
    parser.add_argument('--output', type=str, required=True,
                       help='输出结果文件路径（JSON格式）；当输入为文件夹时，此为输出根目录，每个病例输出到该目录下的子文件夹')
    parser.add_argument('--radius', type=float, default=4.0,
                       help='通道半径（毫米），默认: 4.0')
    parser.add_argument('--length', type=float, default=150.0,
                       help='通道长度（毫米），默认: 150.0')
    parser.add_argument('--num_channels', type=int, default=100,
                       help='要生成的通道数量，默认: 100')
    parser.add_argument('--resolution', type=float, default=0.5,
                       help='圆柱体内部采样分辨率（毫米），默认: 0.5')
    parser.add_argument('--labels', type=int, nargs='+', default=None,
                       help='要统计的标签值列表（可选），默认统计所有标签')
    parser.add_argument('--margin', type=float, default=10.0,
                       help='边界留白（毫米），默认: 10.0')
    parser.add_argument('--seed', type=int, default=None,
                       help='随机种子（用于复现结果）')
    parser.add_argument('--target_point', type=float, nargs=3, default=None,
                       help='固定的穿刺靶点（终点）物理坐标 [x, y, z]（毫米），如果不指定则使用数据中心')
    parser.add_argument(
        '--ct', type=str, default=None,
        help='与 --input 掩码配对的 CT NIfTI（单文件）；与 puncture_target 联用时需与掩码同形状')
    parser.add_argument(
        '--ct_dir', type=str, default=None,
        help='批量：CT 所在目录，按文件名 stem 与每个掩码配对（puncture_target 需要）')
    parser.add_argument(
        '--endpoints_csv', type=str, default=None,
        help='extract_endpoints_dataset 等生成的 CSV（列 case_id、stem、end_x_mm、end_y_mm、end_z_mm，RAS mm）；'
             '指定后按各行靶点规划，--input 须为掩码目录。与 --target_point / --external_targets_dir / --plan_subset 互斥')
    parser.add_argument(
        '--puncture_checkpoint', type=str, default=None,
        help='puncture_target 权重 best.pt；省略时自动使用与脚本同目录下 puncture_baseline/best.pt（若存在），'
             '否则仍查找项目 runs/puncture_baseline/best.pt')
    parser.add_argument(
        '--legacy_target', action='store_true',
        help='强制使用原有几何靶点，不调用 puncture_target 网络')
    parser.add_argument(
        '--external_targets_dir', type=str, default=None,
        help='使用已有 Slicer 靶点（LPS 的 .mrk.json）：支持 L4-L5_left.mrk.json 或 l4_l5_left.mrk.json 等；'
             '多病例且输入为掩码目录时，可在该目录下按病例名建子目录并放入对应 mrk')
    parser.add_argument(
        '--puncture_gpu', type=int, default=None,
        help='puncture 推理 GPU（同 infer --gpu）；默认自动选 CUDA')
    parser.add_argument(
        '--puncture_no_lr_spread', action='store_true',
        help='与 infer --no_lr_spread 一致：关闭「左右冠状过近时对称拉开」')
    parser.add_argument(
        '--puncture_lr_collapse_mm', type=float, default=4.0,
        help='infer --lr_collapse_mm，默认 4')
    parser.add_argument(
        '--puncture_lr_half_width_mm', type=float, default=0.0,
        help='infer --lr_half_width_mm；默认 0（不拉开，与常用零后处理一致）；脚本原版多为 14')
    parser.add_argument(
        '--puncture_lr_extra_lateral_mm', type=float, default=0.0,
        help='infer --lr_extra_lateral_mm；默认 0；脚本原版多为 3')
    parser.add_argument(
        '--puncture_foramen_anterior_mm', type=float, default=0.0,
        help='infer --foramen_anterior_mm，默认 0')
    parser.add_argument(
        '--puncture_foramen_posterior_mm', type=float, default=0.0,
        help='infer --foramen_posterior_mm，默认 0')
    parser.add_argument(
        '--puncture_foramen_superior_l4l5_mm', type=float, default=0.0,
        help='infer --foramen_superior_l4l5_mm，默认 0')
    parser.add_argument(
        '--puncture_foramen_superior_l5s1_mm', type=float, default=0.0,
        help='infer --foramen_superior_l5s1_mm，默认 0')
    parser.add_argument(
        '--puncture_z_p_l4l5', type=float, default=PLANNING.z_p_l4l5, help='infer --z_p_l4l5（占位，请标定）')
    parser.add_argument(
        '--puncture_z_p_l5s1', type=float, default=PLANNING.z_p_l5s1, help='infer --z_p_l5s1（占位，请标定）')
    parser.add_argument(
        '--puncture_l5s1_l5_weight', type=float, default=PLANNING.l5s1_l5_weight,
        help='infer --l5s1_l5_weight')
    parser.add_argument(
        '--puncture_infer_script_defaults', action='store_true',
        help='使用 puncture_target.infer 脚本默认几何后处理（lr_half_width=14, lr_extra_lateral=3 等），'
             '而非当前默认的零外移/零椎间孔微调')
    parser.add_argument('--auto_detect_target', action='store_true',
                       help='自动识别L5和S1椎体后缘与椎间盘中位线交叉点作为终点')
    parser.add_argument('--detect_facet_joint', action='store_true',
                       help='自动识别L4-L5靶点：椎间盘层面椎体后缘深度 + 单侧小关节内侧缘X（见 --facet_side）')
    parser.add_argument('--facet_side', type=str, default='both',
                       choices=['auto', 'left', 'right', 'both'],
                       help='both（默认）= 左、右各规划一套靶点+入路；left/right= 单侧；auto= 自动选一侧（靶点与入路同侧）。'
                            'RAS 下 +X 为患者右侧。')
    parser.add_argument('--detect_l5_s1_disc', action='store_true',
                       help='以L5到S1椎体的上下后缘与椎间盘中位线交叉点作为穿刺靶点')
    parser.add_argument(
        '--plan_subset',
        type=str,
        default=None,
        help='逗号分隔，仅执行所列规划：l4_l5_left,l4_l5_right,l5_s1_left,l5_s1_right。'
             '指定时按勾选组合生成子目录，不再使用 --facet_side 与 detect 的任务组合逻辑。',
    )
    parser.add_argument('--l4_label', type=int, default=4,
                       help='L4椎体的标签值，默认: 4')
    parser.add_argument('--l5_label', type=int, default=5,
                       help='L5椎体的标签值，默认: 5')
    parser.add_argument('--s1_label', type=int, default=6,
                       help='S1椎体的标签值，默认: 6')
    parser.add_argument('--facet_joint_labels', type=int, nargs='+', default=None,
                       help='交集体积计算时仅使用关节突区域的标签列表，默认: 1 2 3 4 5 6（L1-S1）；其余标签按完整结构统计')
    parser.add_argument('--disc_edge_midpoint', action='store_true',
                       help='在 left/right 下将「上/下缘薄壳入路侧半侧中点」与椎间孔主靶点在 XY 上小幅融合（默认关闭）')
    parser.add_argument('--use_angle_grid', action='store_true',
                       help='使用角度网格模式（每5度生成一个角度），而不是随机采样')
    parser.add_argument(
        '--no_angle_grid_spherical',
        action='store_true',
        help='与 --use_angle_grid 联用时禁用球面角网格，改用旧版可行域 CPA×CSA；'
        '默认已启用球面角 α×β 矩形网格',
    )
    # 兼容旧命令行（球面网格已为默认，此开关无效果）
    parser.add_argument('--angle_grid_spherical', action='store_true', help=argparse.SUPPRESS)
    parser.add_argument('--angle_step', type=float, default=5.0,
                       help='角度间隔（度），默认: 5.0（仅在角度网格模式下使用）')
    parser.add_argument('--plot_cpa_csa_chart', dest='plot_cpa_csa_chart', action='store_true',
                       help='显式开启 CPA-CSA 柱状图（默认已开启，可不写）')
    parser.add_argument('--no_plot_cpa_csa_chart', dest='plot_cpa_csa_chart', action='store_false',
                       help='不生成 CPA-CSA 交集体积柱状图（默认会生成）')
    parser.add_argument('--plot_labels', type=int, nargs='+', default=None,
                       help='指定要生成图表的标签列表（可选），如果不指定则只生成有交集的标签')
    parser.add_argument('--filter_optimal', action='store_true',
                       help='筛选最优路径：默认同侧髂骨严格零碰撞（数值容差）+ 字典序；宽松门槛见 --iliac_max_intersection_mm3')
    parser.add_argument('--top_k', type=int, default=10,
                       help='返回前K个最优路径，默认: 10')
    parser.add_argument('--filter_optimal_legacy_weights', action='store_true',
                       help='使用旧版加权得分筛选（与 --label_weights / --total_volume_weight 配合）')
    parser.add_argument('--iliac_max_intersection_mm3', type=float, default=0.0,
                       help='髂骨可行约束：≤0 时为严格「同侧髂骨零碰撞」（≤ --iliac_zero_eps_mm3）；'
                            '>0 时为宽松模式，髂骨聚合须小于该值（mm³），配合 --iliac_gate')
    parser.add_argument('--iliac_gate', type=str, default='ipsilateral',
                       choices=['sum', 'max', 'ipsilateral'],
                       help='髂骨聚合定义（宽松模式下生效；严格零碰撞时强制 ipsilateral）：'
                            'sum=左18+右19；max=较大一侧；ipsilateral=仅入路侧髂骨，默认 ipsilateral')
    parser.add_argument('--iliac_zero_eps_mm3', type=float, default=DEFAULT_ILIAC_ZERO_EPS_MM3,
                       help='严格零碰撞模式下判定髂骨体积为 0 的数值容差（mm³），默认 1e-9')
    parser.add_argument('--label_weights', type=str, default=None,
                       help='仅与 --filter_optimal_legacy_weights 联用：标签权重 JSON')
    parser.add_argument('--total_volume_weight', type=float, default=0.3,
                       help='仅旧版加权：总交集体积权重，默认: 0.3')
    parser.add_argument('--optimal_output', type=str, default=None,
                       help='最优路径输出文件路径（JSON格式），默认: 在输出JSON同目录下生成optimal_paths.json')
    parser.add_argument('--angle_curves_output', type=str, default=None,
                       help='CPA-CSA 柱状图输出目录（默认: 在输出JSON同目录下生成 cpa_csa_volume_charts/）')
    parser.add_argument('--export_slicer', action='store_true',
                       help='导出为3D Slicer格式')
    parser.add_argument('--slicer_output', type=str, default=None,
                       help='3D Slicer导出文件路径（默认: 在输出JSON同目录下生成）')
    parser.add_argument('--slicer_format', type=str, default='mrk.json',
                       choices=['mrk.json', 'fcsv'],
                       help='3D Slicer导出格式：mrk.json（推荐）或 fcsv，默认: mrk.json')
    parser.add_argument('--max_export_paths', type=int, default=None,
                       help='最多导出的路径数量（默认: 导出所有）')
    parser.add_argument('--no_coordinate_convert', action='store_true',
                       help='禁用 RAS→LPS 转换（mrk.json 将与控制台同号）。默认必须转换：Slicer 标记点用 LPS，与控制台 RAS 相比常为 X、Y 变号。仅当与影像严重错位时再试本选项')
    parser.add_argument('--min_angle', type=float, default=0.0,
                       help='轨迹与冠状面/横断面的最小夹角（度），默认: 0.0（CPA/CSA 未单独指定时共用）')
    parser.add_argument('--max_angle', type=float, default=30.0,
                       help='轨迹与冠状面/横断面的最大夹角（度），默认: 30.0（CPA/CSA 未单独指定时共用）')
    parser.add_argument('--cpa_min', type=float, default=None,
                       help='CPA（冠状面）最小角度（度），不指定则用 --min_angle')
    parser.add_argument('--cpa_max', type=float, default=None,
                       help='CPA（冠状面）最大角度（度），不指定则用 --max_angle')
    parser.add_argument('--cpa_step', type=float, default=None,
                       help='CPA（冠状面）角度步长（度），不指定则用 --angle_step')
    parser.add_argument('--csa_min', type=float, default=None,
                       help='CSA（横断面）最小角度（度），不指定则用 --min_angle')
    parser.add_argument('--csa_max', type=float, default=None,
                       help='CSA（横断面）最大角度（度），不指定则用 --max_angle')
    parser.add_argument('--csa_step', type=float, default=None,
                       help='CSA（横断面）角度步长（度），不指定则用 --angle_step')
    
    parser.set_defaults(plot_cpa_csa_chart=True)
    args = parser.parse_args()
    if args.puncture_checkpoint is None or not str(args.puncture_checkpoint).strip():
        args.puncture_checkpoint = default_puncture_baseline_checkpoint_path()
    else:
        args.puncture_checkpoint = str(args.puncture_checkpoint).strip()
    _path_planning_run(args)


def _path_planning_run(args) -> None:
        # 检查输入路径并构建任务列表：(segmentation_file, output_json_path)
        if not os.path.exists(args.input):
            print(f"错误: 输入路径不存在: {args.input}")
            return

        if args.external_targets_dir:
            ed = str(args.external_targets_dir).strip()
            if not os.path.isdir(ed):
                print(f"错误: --external_targets_dir 不是有效目录: {ed}")
                return
            args.external_targets_dir = os.path.abspath(ed)
        if args.external_targets_dir and args.target_point is not None:
            print("错误: --external_targets_dir 与 --target_point 不能同时使用")
            return
    
        plan_specs = _parse_plan_subset_arg(args.plan_subset) if args.plan_subset else None
        if args.plan_subset and plan_specs is None:
            return
        if plan_specs is not None and args.target_point is not None:
            print("错误: --plan_subset 与 --target_point 不能同时使用")
            return

        use_endpoints_csv = bool(args.endpoints_csv and str(args.endpoints_csv).strip())
        if use_endpoints_csv and args.target_point is not None:
            print("错误: --endpoints_csv 与 --target_point 不能同时使用")
            return
        if use_endpoints_csv and args.external_targets_dir:
            print("错误: --endpoints_csv 与 --external_targets_dir 不能同时使用")
            return
        if use_endpoints_csv and args.plan_subset:
            print("错误: --endpoints_csv 与 --plan_subset 不能同时使用")
            return

        tasks: List[Tuple] = []
        n_input_files = 0
        multi_patient_mrk = False

        if use_endpoints_csv:
            if not os.path.isdir(args.input):
                print("错误: 使用 --endpoints_csv 时 --input 须为包含掩码的目录（每病例 <case_id>.nii.gz）")
                return
            masks_dir = os.path.abspath(args.input)
            out_root = os.path.abspath(args.output)
            os.makedirs(out_root, exist_ok=True)
            csv_abs = os.path.abspath(str(args.endpoints_csv).strip())
            try:
                csv_rows = read_tasks_from_endpoints_csv(csv_abs)
            except (FileNotFoundError, ValueError) as e:
                print(f"错误: {e}")
                return
            if args.detect_facet_joint or args.detect_l5_s1_disc or args.auto_detect_target:
                print(
                    "提示: 已指定 --endpoints_csv，靶点以 CSV 为准，忽略 --detect_facet_joint / "
                    "--detect_l5_s1_disc / --auto_detect_target。"
                )
            for case_id, stem, ras in csv_rows:
                mask_path = find_mask_for_case_endpoints_csv(masks_dir, case_id)
                if not mask_path:
                    print(
                        "[跳过] 未找到掩码 %s.nii(.gz)（已查 %s 及 %s）"
                        % (case_id, masks_dir, os.path.join(masks_dir, case_id))
                    )
                    continue
                try:
                    seg_type, fside = parse_stem_endpoints_csv(stem)
                except ValueError as e:
                    print("[跳过] %s/%s: %s" % (case_id, stem, e))
                    continue
                plan_segment = "l4_l5" if seg_type == "l4_l5" else "l5_s1"
                use_facet = seg_type == "l4_l5"
                use_l5s1 = seg_type == "l5_s1"
                out_dir = os.path.join(out_root, case_id, stem)
                os.makedirs(out_dir, exist_ok=True)
                out_json = os.path.join(out_dir, "results.json")
                tasks.append((mask_path, out_json, use_facet, use_l5s1, fside, ras, plan_segment))
            if not tasks:
                print("错误: --endpoints_csv 未产生任何有效任务（请检查 case_id 与掩码文件是否一致）")
                return
            n_input_files = len({os.path.abspath(t[0]) for t in tasks})
            multi_patient_mrk = n_input_files > 1
            print(
                "已从 --endpoints_csv 生成 %d 个规划任务（掩码目录: %s，输出根: %s）。"
                % (len(tasks), masks_dir, out_root)
            )
            # 与非 CSV 分支一致，供末尾「全部完成」汇总使用
            both_targets = any(t[2] for t in tasks) and any(t[3] for t in tasks)
            multi_side = len({t[4] for t in tasks if t[4] in ('left', 'right')}) > 1
        else:
            if plan_specs is None:
                # 是否同时跑 L4-L5 与 L5-S1 两种靶点（每个病例各生成两套结果）
                both_targets = args.detect_facet_joint and args.detect_l5_s1_disc
    
                fs_arg = (args.facet_side or 'both').strip().lower()
                if fs_arg not in ('auto', 'left', 'right', 'both'):
                    print("错误: --facet_side 必须是 auto、left、right 或 both")
                    return
                facet_side_runs = ['left', 'right'] if fs_arg == 'both' else [fs_arg]
                multi_side = len(facet_side_runs) > 1
            else:
                both_targets = False
                facet_side_runs = []
                multi_side = False
    
            n_input_files = 0
        
            if os.path.isdir(args.input):
                # 输入为文件夹：收集所有 mask 文件（.nii.gz 或 .nii）
                mask_files = []
                for f in sorted(os.listdir(args.input)):
                    full = os.path.join(args.input, f)
                    if not os.path.isfile(full):
                        continue
                    if f.endswith('.nii.gz'):
                        mask_files.append(full)
                    elif f.endswith('.nii') and not f.endswith('.nii.gz'):
                        mask_files.append(full)
                if not mask_files:
                    print(f"错误: 文件夹中未找到 NII.GZ 或 NII 格式的 mask 文件: {args.input}")
                    return
                n_input_files = len(mask_files)
                base_out = args.output
                os.makedirs(base_out, exist_ok=True)
                tasks = []
                for m in mask_files:
                    name = os.path.basename(m)
                    if name.endswith('.nii.gz'):
                        name = name[:-7]
                    elif name.endswith('.nii'):
                        name = name[:-4]
                    patient_dir = os.path.join(base_out, name)
                    os.makedirs(patient_dir, exist_ok=True)
                    if plan_specs is not None:
                        for use_facet, use_l5s1, fside in plan_specs:
                            seg_sub = f'l4_l5_{fside}' if use_facet else f'l5_s1_{fside}'
                            sub = os.path.join(patient_dir, seg_sub)
                            os.makedirs(sub, exist_ok=True)
                            tasks.append((m, os.path.join(sub, 'results.json'), use_facet, use_l5s1, fside))
                    elif both_targets:
                        for fside in facet_side_runs:
                            suf = f'_{fside}' if multi_side else ''
                            l4_l5_dir = os.path.join(patient_dir, f'l4_l5{suf}')
                            l5_s1_dir = os.path.join(patient_dir, f'l5_s1{suf}')
                            os.makedirs(l4_l5_dir, exist_ok=True)
                            os.makedirs(l5_s1_dir, exist_ok=True)
                            tasks.append((m, os.path.join(l4_l5_dir, 'results.json'), True, False, fside))
                            tasks.append((m, os.path.join(l5_s1_dir, 'results.json'), False, True, fside))
                    else:
                        for fside in facet_side_runs:
                            if args.detect_facet_joint and not args.detect_l5_s1_disc:
                                seg_sub = f'l4_l5_{fside}'
                            elif args.detect_l5_s1_disc and not args.detect_facet_joint:
                                seg_sub = f'l5_s1_{fside}'
                            else:
                                seg_sub = None
                            if multi_side:
                                if seg_sub is not None:
                                    sub = os.path.join(patient_dir, seg_sub)
                                else:
                                    sub = os.path.join(patient_dir, f'plan_{fside}')
                                os.makedirs(sub, exist_ok=True)
                                out_json = os.path.join(sub, 'results.json')
                            else:
                                if seg_sub is not None:
                                    sub = os.path.join(patient_dir, seg_sub)
                                    os.makedirs(sub, exist_ok=True)
                                    out_json = os.path.join(sub, 'results.json')
                                else:
                                    out_json = os.path.join(patient_dir, 'results.json')
                            tasks.append((m, out_json, args.detect_facet_joint, args.detect_l5_s1_disc, fside))
                if plan_specs is not None:
                    print(f"已从文件夹读取 {n_input_files} 个 mask 文件（plan_subset 共 {len(plan_specs)} 类任务），共 {len(tasks)} 次规划。")
                else:
                    seg_desc = "L4-L5 与 L5-S1" if both_targets else "单节段靶点"
                    side_desc = "左、右两侧各一套" if multi_side else f"facet_side={facet_side_runs[0]}"
                    print(f"已从文件夹读取 {n_input_files} 个 mask 文件（{seg_desc}，{side_desc}），共 {len(tasks)} 次规划。")
            else:
                n_input_files = 1
                tasks = []
                if plan_specs is not None:
                    base_dir = os.path.dirname(args.output) or '.'
                    bn = os.path.basename(args.output)
                    for use_facet, use_l5s1, fside in plan_specs:
                        seg_sub = f'l4_l5_{fside}' if use_facet else f'l5_s1_{fside}'
                        sub = os.path.join(base_dir, seg_sub)
                        os.makedirs(sub, exist_ok=True)
                        outp = os.path.join(sub, bn)
                        tasks.append((args.input, outp, use_facet, use_l5s1, fside))
                    print(f"plan_subset：{len(plan_specs)} 类任务（单文件输入），输出至各 l4_l5_*/l5_s1_* 子目录。")
                elif both_targets:
                    base_dir = os.path.dirname(args.output) or '.'
                    for fside in facet_side_runs:
                        suf = f'_{fside}' if multi_side else ''
                        l4_l5_dir = os.path.join(base_dir, f'l4_l5{suf}')
                        l5_s1_dir = os.path.join(base_dir, f'l5_s1{suf}')
                        os.makedirs(l4_l5_dir, exist_ok=True)
                        os.makedirs(l5_s1_dir, exist_ok=True)
                        tasks.append((args.input, os.path.join(l4_l5_dir, 'results.json'), True, False, fside))
                        tasks.append((args.input, os.path.join(l5_s1_dir, 'results.json'), False, True, fside))
                    sd = "左、右各一套" if multi_side else facet_side_runs[0]
                    print(f"将分别生成 L4-L5 与 L5-S1（{sd}）。")
                else:
                    for fside in facet_side_runs:
                        if args.detect_facet_joint and not args.detect_l5_s1_disc:
                            seg_sub = f'l4_l5_{fside}'
                        elif args.detect_l5_s1_disc and not args.detect_facet_joint:
                            seg_sub = f'l5_s1_{fside}'
                        else:
                            seg_sub = None
                        if multi_side:
                            d = os.path.dirname(args.output) or '.'
                            bn = os.path.basename(args.output)
                            if seg_sub is not None:
                                sub = os.path.join(d, seg_sub)
                            else:
                                sub = os.path.join(d, fside)
                            os.makedirs(sub, exist_ok=True)
                            outp = os.path.join(sub, bn)
                        else:
                            if seg_sub is not None:
                                d = os.path.dirname(args.output) or '.'
                                if os.path.basename(os.path.normpath(d)) == seg_sub:
                                    outp = args.output
                                else:
                                    sub = os.path.join(d, seg_sub)
                                    os.makedirs(sub, exist_ok=True)
                                    bn = os.path.basename(args.output)
                                    outp = os.path.join(sub, bn)
                            else:
                                outp = args.output
                        tasks.append((args.input, outp, args.detect_facet_joint, args.detect_l5_s1_disc, fside))
        
        # 验证角度参数
        if args.min_angle < 0 or args.max_angle > 90:
            print(f"错误: 角度范围必须在 [0, 90] 度之间")
            return
        if args.min_angle >= args.max_angle:
            print(f"错误: 最小角度 ({args.min_angle}°) 必须小于最大角度 ({args.max_angle}°)")
            return
    
        # 创建路径规划器
        planner = SurgicalPathPlanner(
            channel_radius_mm=args.radius,
            channel_length_mm=args.length,
            resolution=args.resolution,
            min_angle_deg=args.min_angle,
            max_angle_deg=args.max_angle
        )
    
        # 解析靶点坐标
        target_point = None
        if args.target_point is not None:
            target_point = np.array(args.target_point)
    
        # 收集所有病例的路径结果，用于最后按 CPA/CSA 角度汇总输出 Excel（含任务类型与 facet_side）
        all_planning_results: List[Tuple] = []

        if not use_endpoints_csv:
            multi_patient_mrk = n_input_files > 1 if os.path.isdir(args.input) else False
            if (
                target_point is None
                and not args.external_targets_dir
                and (args.detect_facet_joint or args.detect_l5_s1_disc or args.auto_detect_target)
                and not args.legacy_target
            ):
                ck_try = resolve_puncture_checkpoint_path(args.puncture_checkpoint)
                if ck_try and not args.ct and not args.ct_dir:
                    print(
                        f"\n[靶点] 将尝试 puncture_target 网络（权重: {ck_try}）。"
                        "当前未指定 --ct 或 --ct_dir；在非 --legacy_target 且需自动靶点时，"
                        "缺少配对 CT 将导致规划报错而非退回几何/数据中心。"
                    )
    
        # 对每个病例（及每种靶点、左/右侧）执行规划与输出
        for seg_idx, task in enumerate(tasks, 1):
            optimal_paths_for_audit = None
            if len(task) == 7:
                segmentation_file, output_file, use_facet_joint, use_l5_s1_disc, facet_for_plan, csv_ras, forced_segment = task
                target_override = csv_ras
            else:
                segmentation_file, output_file, use_facet_joint, use_l5_s1_disc, facet_for_plan = task
                target_override = None
                forced_segment = None
            if len(tasks) > 1:
                target_label = ""
                if use_facet_joint:
                    target_label = " [L4-L5 小关节]"
                elif use_l5_s1_disc:
                    target_label = " [L5-S1 椎间盘]"
                side_note = ""
                if facet_for_plan == 'left':
                    side_note = " [左侧]"
                elif facet_for_plan == 'right':
                    side_note = " [右侧]"
                elif facet_for_plan == 'auto':
                    side_note = " [auto]"
                print(f"\n{'='*60}")
                print(f"正在处理 [{seg_idx}/{len(tasks)}]: {os.path.basename(segmentation_file)}{target_label}{side_note}")
                print(f"输出目录: {os.path.dirname(output_file)}")
                print('='*60)
        
            tp_task = target_point
            if target_override is not None:
                tp_task = np.asarray(target_override, dtype=float).reshape(-1)[:3]
            elif args.external_targets_dir:
                patient_key = _nii_stem_for_pair(segmentation_file)
                ext = _load_target_ras_from_external_dir(
                    args.external_targets_dir,
                    patient_key,
                    multi_patient_mrk,
                    use_facet_joint,
                    use_l5_s1_disc,
                    facet_for_plan,
                )
                if ext is None:
                    print("错误: 无法从 external_targets_dir 加载本任务靶点，中止（退出码 1）。", flush=True)
                    sys.exit(1)
                tp_task = ext

            # 规划路径（按当前任务选用靶点）
            results = planner.plan_paths(
                segmentation_file=segmentation_file,
                num_channels=args.num_channels,
                label_values=args.labels,
                margin_mm=args.margin,
                seed=args.seed,
                target_point=tp_task,
                auto_detect_target=False if target_override is not None else args.auto_detect_target,
                detect_facet_joint=False if target_override is not None else use_facet_joint,
                detect_l5_s1_disc=False if target_override is not None else use_l5_s1_disc,
                l4_label=args.l4_label,
                l5_label=args.l5_label,
                s1_label=args.s1_label,
                facet_joint_labels=args.facet_joint_labels,
                facet_side=facet_for_plan,
                disc_edge_midpoint_target=args.disc_edge_midpoint,
                use_angle_grid=args.use_angle_grid,
                angle_grid_spherical=(
                    bool(args.use_angle_grid) and not bool(args.no_angle_grid_spherical)
                ),
                angle_step_deg=args.angle_step,
                cpa_min_deg=args.cpa_min,
                cpa_max_deg=args.cpa_max,
                cpa_step_deg=args.cpa_step,
                csa_min_deg=args.csa_min,
                csa_max_deg=args.csa_max,
                csa_step_deg=args.csa_step,
                ct_file=args.ct,
                ct_dir=args.ct_dir,
                puncture_checkpoint=args.puncture_checkpoint,
                legacy_target_only=args.legacy_target,
                puncture_gpu=args.puncture_gpu,
                puncture_z_p_l4l5=args.puncture_z_p_l4l5,
                puncture_z_p_l5s1=args.puncture_z_p_l5s1,
                puncture_l5s1_l5_weight=args.puncture_l5s1_l5_weight,
                puncture_lr_spread=not args.puncture_no_lr_spread,
                puncture_lr_collapse_mm=args.puncture_lr_collapse_mm,
                puncture_lr_half_width_mm=args.puncture_lr_half_width_mm,
                puncture_lr_extra_lateral_mm=args.puncture_lr_extra_lateral_mm,
                puncture_foramen_anterior_mm=args.puncture_foramen_anterior_mm,
                puncture_foramen_posterior_mm=args.puncture_foramen_posterior_mm,
                puncture_foramen_superior_l4l5_mm=args.puncture_foramen_superior_l4l5_mm,
                puncture_foramen_superior_l5s1_mm=args.puncture_foramen_superior_l5s1_mm,
            )
        
            if forced_segment is not None:
                planner._current_segment = forced_segment
        
            # 打印统计信息
            planner.print_statistics(results)
        
            # 保存结果（当前病例的 output 路径）
            planner.save_results(results, output_file)
            if results:
                planner.export_all_paths_intersection_excel(
                    results,
                    output_file,
                    facet_side=facet_for_plan,
                    use_facet_joint=use_facet_joint,
                    use_l5_s1_disc=use_l5_s1_disc,
                )
        
            # 收集本病例路径，用于最后按角度汇总 Excel
            all_planning_results.append(
                (results, output_file, use_facet_joint, use_l5_s1_disc, facet_for_plan))
        
            # 生成CPA-CSA交集体积柱状图（如果启用）
            if args.plot_cpa_csa_chart:
                if not results:
                    print("\n警告: 没有结果数据，无法生成CPA-CSA交集体积图")
                else:
                    if args.angle_curves_output is None:
                        base_dir = os.path.dirname(output_file) or '.'
                        cpa_csa_dir = os.path.join(
                            base_dir, angle_volume_charts_folder_name(results)
                        )
                    else:
                        _av = angle_volume_charts_folder_name(results)
                        cpa_csa_dir = (
                            args.angle_curves_output
                            if len(tasks) == 1
                            else os.path.join(os.path.dirname(output_file), _av)
                        )
                    # 与 angle_average_from_results / export_angle_average_excel 一致：按任务节段与 facet 侧选用 chart_labels_*
                    if args.plot_labels is not None:
                        chart_label_ids = args.plot_labels
                    elif use_facet_joint and not use_l5_s1_disc:
                        chart_label_ids = chart_labels_l4_l5_for_side(facet_for_plan)
                    elif use_l5_s1_disc and not use_facet_joint:
                        chart_label_ids = chart_labels_l5_s1_for_side(facet_for_plan)
                    else:
                        chart_label_ids = None
                    planner.plot_cpa_csa_volume_chart(
                        results,
                        cpa_csa_dir,
                        args.angle_step,
                        segmentation=None,
                        all_labels=chart_label_ids,
                        use_structure_names=True,
                    )
        
            # 导出为3D Slicer格式（如果启用）
            if args.export_slicer:
                if not results:
                    print("\n警告: 没有结果数据，无法导出到3D Slicer")
                else:
                    if args.slicer_output is None:
                        base_name = os.path.splitext(output_file)[0]
                        if args.slicer_format == 'fcsv':
                            slicer_output = f"{base_name}_paths.fcsv"
                        else:
                            slicer_output = f"{base_name}_paths.mrk.json"
                    else:
                        slicer_output = args.slicer_output if len(tasks) == 1 else os.path.join(os.path.dirname(output_file), "paths." + ("fcsv" if args.slicer_format == 'fcsv' else "mrk.json"))
                
                    planner.export_to_slicer(
                        results=results,
                        output_file=slicer_output,
                        format=args.slicer_format,
                        max_paths=args.max_export_paths,
                        convert_coordinates=not args.no_coordinate_convert
                    )
        
            optimal_output_path: Optional[str] = None
            # 筛选最优路径（如果启用）
            if args.filter_optimal:
                if not results:
                    print("\n警告: 没有结果数据，无法筛选最优路径")
                else:
                    label_weights = None
                    if args.label_weights:
                        try:
                            label_weights = json.loads(args.label_weights)
                            label_weights = {int(k): float(v) for k, v in label_weights.items()}
                        except json.JSONDecodeError:
                            print(f"\n警告: 标签权重格式错误，使用默认权重")
                            label_weights = None
                
                    planner._last_output_path = output_file
                
                    optimal_paths = planner.filter_optimal_paths(
                        results=results,
                        label_weights=label_weights,
                        total_volume_weight=args.total_volume_weight,
                        top_k=args.top_k,
                        iliac_max_intersection_mm3=args.iliac_max_intersection_mm3,
                        iliac_gate_mode=args.iliac_gate,
                        iliac_zero_eps_mm3=args.iliac_zero_eps_mm3,
                        use_legacy_weighted_ranking=args.filter_optimal_legacy_weights,
                    )
                    optimal_paths_for_audit = optimal_paths
                
                    if optimal_paths:
                        if args.optimal_output is None:
                            base_name = os.path.splitext(output_file)[0]
                            optimal_output = f"{base_name}_optimal_paths.json"
                        else:
                            optimal_output = args.optimal_output if len(tasks) == 1 else os.path.join(os.path.dirname(output_file), "optimal_paths.json")
                        optimal_output_path = optimal_output
                        planner.save_results(optimal_paths, optimal_output)
                    
                        if args.export_slicer:
                            if args.slicer_output is None:
                                base_name = os.path.splitext(output_file)[0]
                                if args.slicer_format == 'fcsv':
                                    optimal_slicer_output = f"{base_name}_optimal_paths.fcsv"
                                else:
                                    optimal_slicer_output = f"{base_name}_optimal_paths.mrk.json"
                            else:
                                base_name = os.path.splitext(args.slicer_output)[0] if len(tasks) == 1 else os.path.splitext(output_file)[0]
                                if args.slicer_format == 'fcsv':
                                    optimal_slicer_output = f"{base_name}_optimal_paths.fcsv"
                                else:
                                    optimal_slicer_output = f"{base_name}_optimal_paths.mrk.json"
                        
                            planner.export_to_slicer(
                                results=optimal_paths,
                                output_file=optimal_slicer_output,
                                format=args.slicer_format,
                                max_paths=None,
                                convert_coordinates=not args.no_coordinate_convert
                            )
                            print(f"\n最优路径已导出到3D Slicer格式: {optimal_slicer_output}")
        if len(tasks) > 1:
            parts = [f"{n_input_files} 个输入"]
            if plan_specs is not None:
                parts.append(f"× plan_subset（{len(plan_specs)} 类）")
            else:
                if both_targets:
                    parts.append("× L4-L5 与 L5-S1")
                if multi_side:
                    parts.append("× 左/右两侧")
            print(f"\n全部完成：{' '.join(parts)}，共 {len(tasks)} 次规划，结果已写入对应输出目录。")
    
        # 按 CPA/CSA 角度汇总所有病人的交集体积平均值，输出 Excel 和 CPA-CSA 平均图
        if all_planning_results:
            base_out = args.output if os.path.isdir(args.input) else (os.path.dirname(args.output) or '.')
            angle_avg_dir = os.path.join(base_out, "angle_average")
            export_angle_average_excel(all_planning_results, angle_avg_dir, args.angle_step)
            # 分别在 angle_average 下输出 L4-L5、L5-S1 的 CPA-CSA 交集体积平均图（与筛选最优路径所用结构一致）
            for use_facet_f, use_l5_s1_f, seg_tag in [
                (True, False, 'L4-L5'),
                (False, True, 'L5-S1'),
            ]:
                for fside in ('left', 'right', 'auto'):
                    chunk: List[Dict] = []
                    for item in all_planning_results:
                        if len(item) >= 5:
                            results, _, uf, us1, fs = item[0], item[1], item[2], item[3], item[4]
                        else:
                            results, _, uf, us1 = item[0], item[1], item[2], item[3]
                            fs = 'auto'
                        if uf == use_facet_f and us1 == use_l5_s1_f and fs == fside:
                            chunk.extend(results)
                    if not chunk or not args.plot_cpa_csa_chart:
                        continue
                    _ab_sub = (
                        'alpha_beta_volume_charts'
                        if results_use_spherical_alpha_beta_axes(chunk)
                        else 'cpa_csa_volume_charts'
                    )
                    if fside in ('left', 'right'):
                        subdir = f"{seg_tag}_{fside}_{_ab_sub}"
                    else:
                        subdir = f"{seg_tag}_{_ab_sub}"
                    charts_dir = os.path.join(angle_avg_dir, subdir)
                    label_ids = (chart_labels_l4_l5_for_side(fside) if seg_tag == 'L4-L5'
                                 else chart_labels_l5_s1_for_side(fside))
                    _ab_txt = 'α-β' if _ab_sub == 'alpha_beta_volume_charts' else 'CPA-CSA'
                    print(f"\n生成 {seg_tag}（{fside}）平均 {_ab_txt} 交集体积图...")
                    planner.plot_cpa_csa_volume_chart(
                        chunk,
                        charts_dir,
                        args.angle_step,
                        segmentation=None,
                        all_labels=label_ids,
                        use_structure_names=True,
                    )



if __name__ == "__main__":
    main()

