"""CSV + NIfTI 读取，RAS 靶点 → patch 与偏移标签。"""

from __future__ import annotations

import hashlib
import os
import warnings
from collections import OrderedDict
from typing import Dict, List, Literal, Optional, Sequence, Tuple

import numpy as np
import nibabel as nib
import pandas as pd
import torch
from nibabel.affines import apply_affine
from torch.utils.data import Dataset


STEM_TO_IDX: Dict[str, int] = {
    "l4_l5_left": 0,
    "l4_l5_right": 1,
    "l5_s1_left": 2,
    "l5_s1_right": 3,
}


def load_nib(path: str):
    """
    加载 NIfTI；优先 mmap（部分环境对 NFS 更友好），失败则普通读。
    后续仍会 asanyarray 整卷进内存，mmap 不能消除「整卷读」耗时，仅略改善读盘。
    """
    try:
        return nib.load(path, mmap=True)
    except (TypeError, OSError, ValueError):
        return nib.load(path)


def find_volume_for_case(root: str, case_id: str) -> Optional[str]:
    """与 extract_endpoints_dataset.find_ct_for_case 相同逻辑。"""
    if not root or not os.path.isdir(root):
        return None
    sub = os.path.join(root, case_id)
    for ext in (".nii.gz", ".nii"):
        for p in (
            os.path.join(sub, f"{case_id}{ext}"),
            os.path.join(root, f"{case_id}{ext}"),
        ):
            if os.path.isfile(p):
                return os.path.abspath(p)
    if os.path.isdir(sub):
        for fn in sorted(os.listdir(sub)):
            low = fn.lower()
            if low.endswith(".nii.gz") or low.endswith(".nii"):
                return os.path.abspath(os.path.join(sub, fn))
    return None


def _pad_min_shape(
    vol: np.ndarray, min_shape: Sequence[int], fill: float
) -> Tuple[np.ndarray, np.ndarray]:
    """各维 pad 到至少 min_shape。返回 (padded, pad_before) 供 affine 还原。"""
    d, h, w = vol.shape
    pd, ph, pw = min_shape
    pad_d = max(0, pd - d)
    pad_h = max(0, ph - h)
    pad_w = max(0, pw - w)
    if pad_d == pad_h == pad_w == 0:
        return vol, np.zeros(3, dtype=np.float64)
    before = (pad_d // 2, pad_h // 2, pad_w // 2)
    after = (pad_d - before[0], pad_h - before[1], pad_w - before[2])
    padded = np.pad(
        vol,
        ((before[0], after[0]), (before[1], after[1]), (before[2], after[2])),
        mode="constant",
        constant_values=fill,
    )
    return padded, np.array(before, dtype=np.float64)


def _crop_patch(
    vol: np.ndarray,
    center_ijk: np.ndarray,
    patch_size: Tuple[int, int, int],
    fill: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    以 center_ijk 为中心取 patch；越界部分用 fill 填充。
    返回 patch 与 **patch 几何中心** 对应的体素坐标（用于物理中心）。
    """
    pd, ph, pw = patch_size
    cz, cy, cx = float(center_ijk[0]), float(center_ijk[1]), float(center_ijk[2])
    half_d, half_h, half_w = pd // 2, ph // 2, pw // 2
    out = np.full(patch_size, fill, dtype=np.float32)
    zs, ys, xs = int(np.floor(cz)) - half_d, int(np.floor(cy)) - half_h, int(np.floor(cx)) - half_w
    ze, ye, xe = zs + pd, ys + ph, xs + pw
    D, H, W = vol.shape
    # 源与目标索引
    src_z0, src_y0, src_x0 = max(0, zs), max(0, ys), max(0, xs)
    src_z1 = min(D, ze)
    src_y1 = min(H, ye)
    src_x1 = min(W, xe)
    dst_z0 = src_z0 - zs
    dst_y0 = src_y0 - ys
    dst_x0 = src_x0 - xs
    dst_z1 = dst_z0 + (src_z1 - src_z0)
    dst_y1 = dst_y0 + (src_y1 - src_y0)
    dst_x1 = dst_x0 + (src_x1 - src_x0)
    if src_z0 < src_z1 and src_y0 < src_y1 and src_x0 < src_x1:
        out[dst_z0:dst_z1, dst_y0:dst_y1, dst_x0:dst_x1] = vol[
            src_z0:src_z1, src_y0:src_y1, src_x0:src_x1
        ].astype(np.float32)
    # patch 体素中心（连续坐标）
    center_patch_vox = np.array(
        [zs + (pd - 1) / 2.0, ys + (ph - 1) / 2.0, xs + (pw - 1) / 2.0],
        dtype=np.float64,
    )
    return out, center_patch_vox


def _normalize_ct(hu: np.ndarray, hu_min: float, hu_max: float) -> np.ndarray:
    x = np.clip(hu.astype(np.float32), hu_min, hu_max)
    return (x - hu_min) / (hu_max - hu_min + 1e-6)


def build_patch_tensor(
    ct: np.ndarray,
    mask: Optional[np.ndarray],
    affine: np.ndarray,
    center_ras: np.ndarray,
    patch_size: Tuple[int, int, int],
    hu_min: float,
    hu_max: float,
) -> Tuple[torch.Tensor, np.ndarray]:
    """
    以 center_ras（mm，RAS）为裁块中心，与训练时一致的归一化与双通道逻辑。
    返回 (x, patch_center_ras)，x 形状 (1, C, D, H, W)；patch_center_ras 用于 pred_ras = center + offset。
    """
    inv = np.linalg.inv(affine)
    ijk = apply_affine(inv, np.asarray(center_ras, dtype=np.float64).reshape(3))
    pd, ph, pw = patch_size
    ct_pad, pad_before = _pad_min_shape(ct, (pd, ph, pw), hu_min)
    ijk_pad = ijk + pad_before
    ct_patch, center_vox = _crop_patch(
        ct_pad, ijk_pad, patch_size, fill=hu_min
    )
    ct_ch = _normalize_ct(ct_patch, hu_min, hu_max)
    channels: List[np.ndarray] = [ct_ch]
    if mask is not None:
        if mask.shape != ct.shape:
            raise ValueError(f"掩码与 CT 形状不一致: {mask.shape} vs {ct.shape}")
        m_pad, _ = _pad_min_shape(mask, (pd, ph, pw), 0.0)
        if m_pad.shape != ct_pad.shape:
            raise ValueError("掩码 pad 后与 CT 仍不一致")
        m_patch, _ = _crop_patch(
            m_pad.astype(np.float32), ijk_pad, patch_size, fill=0.0
        )
        channels.append((m_patch > 0.5).astype(np.float32))
    x = np.stack(channels, axis=0)
    patch_center_ras = apply_affine(affine, center_vox - pad_before)
    t = torch.from_numpy(x).float().unsqueeze(0)
    return t, patch_center_ras.astype(np.float64)


def _heuristic_centers_from_vertebra_labels(
    mask: np.ndarray,
    affine: np.ndarray,
    l5s1_l5_weight: float,
) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """
    多标签分割（与 path_planning 一致：L4=4, L5=5, S1=6）时，用椎体体素估两组裁块中心。

    - L4-L5：**L4 与 L5 各自质心的中点**（0.5*(c_L4+c_L5)）。并集体素均值会在 L5 更大时
      把中心拉向 L5 椎体内部；两 centroid 中点更贴近椎间隙层。
    - L5-S1：**L5 与 S1 质心的加权**，默认更信 L5（骶骨体素多，mean(L5∪S1) 会把中心拖进 S1，
      裁块落在骶骨内）。l5s1_l5_weight 越大越靠近 L5 / 椎间隙上缘侧。
    """
    m = mask.astype(np.int32)
    u = np.unique(m)
    if 4 not in u or 5 not in u:
        return None
    R = affine[:3, :3].astype(np.float64)
    t = affine[:3, 3].astype(np.float64)

    idx4 = np.argwhere(m == 4)
    idx5 = np.argwhere(m == 5)
    if len(idx4) < 40 or len(idx5) < 40:
        return None
    ras4 = (R @ idx4.astype(np.float64).T).T + t
    ras5 = (R @ idx5.astype(np.float64).T).T + t
    c4 = np.mean(ras4, axis=0)
    c5 = np.mean(ras5, axis=0)
    c45 = 0.5 * (c4 + c5)

    if 6 not in u:
        return None
    idx6 = np.argwhere(m == 6)
    if len(idx6) < 40:
        return None
    ras6 = (R @ idx6.astype(np.float64).T).T + t
    c6 = np.mean(ras6, axis=0)
    w = float(np.clip(l5s1_l5_weight, 0.0, 1.0))
    c56 = w * c5 + (1.0 - w) * c6

    # 正常腰椎：L4-L5 组应比 L5-S1 组更靠头侧（RAS 第 3 分量更大）。
    if float(c45[2]) < float(c56[2]) - 2.0:
        return None

    return c45.astype(np.float64), c56.astype(np.float64)


def heuristic_centers_ras_from_mask(
    mask: np.ndarray,
    affine: np.ndarray,
    z_percentile_l4l5: float,
    z_percentile_l5s1: float,
    l5s1_l5_weight: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    无真值靶点时，用掩码估计 L4-L5 与 L5-S1 两组裁块中心（RAS mm）。

    **优先**：若掩码为多标签且含 L4/L5/S1（4/5/6），则 L4-L5 中心 = **L4 与 L5 质心的中点**，
    L5-S1 中心 = **L5 与 S1 质心的加权**（默认偏 L5，避免骶骨过大把中心拖进 S1 内）。

    **否则**：在 RAS 第 3 分量上用分位数划分（见 z_percentile_*）。

    若仍不合理请用 --centers_json。
    """
    labeled = _heuristic_centers_from_vertebra_labels(
        mask, affine, l5s1_l5_weight=l5s1_l5_weight
    )
    if labeled is not None:
        return labeled

    idx = np.argwhere(mask > 0.5)
    if len(idx) < 10:
        raise ValueError("掩码有效体素过少，无法估计中心")

    R = affine[:3, :3].astype(np.float64)
    t = affine[:3, 3].astype(np.float64)
    ras = (R @ idx.astype(np.float64).T).T + t
    s = ras[:, 2]

    p_hi = float(np.percentile(s, z_percentile_l4l5))
    p_lo = float(np.percentile(s, z_percentile_l5s1))

    sel_l4 = ras[s >= p_hi]
    sel_l5 = ras[s <= p_lo]

    if len(sel_l4) < 5:
        p_hi = float(np.percentile(s, max(35.0, z_percentile_l4l5 - 10.0)))
        sel_l4 = ras[s >= p_hi]
    if len(sel_l5) < 5:
        p_lo = float(np.percentile(s, min(65.0, z_percentile_l5s1 + 10.0)))
        sel_l5 = ras[s <= p_lo]

    def _mean_or_global(sel: np.ndarray) -> np.ndarray:
        if len(sel) < 1:
            return np.mean(ras, axis=0)
        return np.mean(sel, axis=0)

    center_l4 = _mean_or_global(sel_l4)
    center_l5 = _mean_or_global(sel_l5)

    if center_l4[2] <= center_l5[2] + 0.5:
        order = np.argsort(s)
        n = len(order)
        mid = n // 2
        sel_l4 = ras[order[mid:]]
        sel_l5 = ras[order[:mid]]
        center_l4 = _mean_or_global(sel_l4)
        center_l5 = _mean_or_global(sel_l5)

    return center_l4.astype(np.float64), center_l5.astype(np.float64)


def _stem_group_key(stem: str) -> str:
    """与推理时 predict_group 分组一致：l4_l5* 共用 c4，l5_s1* 共用 c5。"""
    if stem.startswith("l4_l5"):
        return "l4_l5"
    if stem.startswith("l5_s1"):
        return "l5_s1"
    raise ValueError(f"未知 stem: {stem}")


def _det_rng(case_id: str, group: str) -> np.random.Generator:
    """同一病例、同一节段组内左右 stem 使用相同种子（与推理同 patch 一致）。"""
    h = hashlib.md5(f"{case_id}\0{group}".encode()).digest()
    seed = int.from_bytes(h[:4], "little")
    return np.random.default_rng(seed)


class PunctureEndpointDataset(Dataset):
    """
    每行：case_id, stem, end_x_mm, end_y_mm, end_z_mm；CT 由 ct_root 查找；可选 mask_root。

    center_mode:
    - ``target``（默认）：以真值靶点为中心裁块（训练时体素抖动），与旧版一致。
    - ``heuristic``：与 ``puncture_target.infer`` 相同，用掩码估计 c4/c5，按 stem 选中心裁块，
      左右靶点共享同一块 patch，偏移标签仍为 ``真值 - patch 几何中心``。

    ``max_cached_cases``：启发式模式下每个 DataLoader worker 内最多缓存多少例的**整幅** CT+掩码；
    不设上限时多 epoch 后易占满内存，被系统 OOM killer 杀掉子进程（报 Killed）。
    """

    def __init__(
        self,
        df: pd.DataFrame,
        ct_root: str,
        mask_root: Optional[str],
        patch_size: Tuple[int, int, int],
        hu_min: float,
        hu_max: float,
        jitter_voxels: float,
        train: bool = True,
        center_mode: Literal["target", "heuristic"] = "target",
        z_percentile_l4l5: float,
        z_percentile_l5s1: float,
        l5s1_l5_weight: float,
        center_jitter_voxels: float = 0.0,
        max_cached_cases: int = 8,
    ) -> None:
        self.rows = df.reset_index(drop=True)
        self.ct_root = ct_root
        self.mask_root = mask_root
        self.patch_size = patch_size
        self.hu_min = hu_min
        self.hu_max = hu_max
        self.jitter_voxels = jitter_voxels
        self.train = train
        self.center_mode = center_mode
        self.z_percentile_l4l5 = z_percentile_l4l5
        self.z_percentile_l5s1 = z_percentile_l5s1
        self.l5s1_l5_weight = l5s1_l5_weight
        self.center_jitter_voxels = float(center_jitter_voxels)
        self.max_cached_cases = max(1, int(max_cached_cases))
        self._case_vol_cache: OrderedDict[
            str, Tuple[np.ndarray, Optional[np.ndarray], np.ndarray]
        ] = OrderedDict()
        self._case_center_cache: Dict[str, Optional[Tuple[np.ndarray, np.ndarray]]] = {}
        self._warned_heuristic_fallback: set[str] = set()

    def __len__(self) -> int:
        return len(self.rows)

    def _load_case_ct_mask_affine(
        self, case_id: str
    ) -> Tuple[np.ndarray, Optional[np.ndarray], np.ndarray]:
        cd = self._case_vol_cache
        if case_id in cd:
            cd.move_to_end(case_id)
            ct, m, aff = cd[case_id]
            return ct, m, aff

        ct_path = find_volume_for_case(self.ct_root, case_id)
        if not ct_path:
            raise FileNotFoundError(f"未找到 CT: case_id={case_id}, ct_root={self.ct_root}")

        ct_img = load_nib(ct_path)
        ct = np.asanyarray(ct_img.dataobj)
        affine = ct_img.affine

        mask_arr: Optional[np.ndarray] = None
        if self.mask_root:
            mp = find_volume_for_case(self.mask_root, case_id)
            if mp:
                m_img = load_nib(mp)
                mask_arr = np.asanyarray(m_img.dataobj)
                if mask_arr.shape != ct.shape:
                    raise ValueError(
                        f"掩码与 CT 原始形状不一致 case={case_id}: "
                        f"{mask_arr.shape} vs {ct.shape}"
                    )

        cd[case_id] = (ct, mask_arr, affine)
        while len(cd) > self.max_cached_cases:
            evicted, _ = cd.popitem(last=False)
            self._case_center_cache.pop(evicted, None)

        return ct, mask_arr, affine

    def _get_heuristic_centers(
        self, case_id: str, mask_arr: np.ndarray, affine: np.ndarray
    ) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        if case_id in self._case_center_cache:
            return self._case_center_cache[case_id]
        try:
            c4, c5 = heuristic_centers_ras_from_mask(
                mask_arr,
                affine,
                z_percentile_l4l5=self.z_percentile_l4l5,
                z_percentile_l5s1=self.z_percentile_l5s1,
                l5s1_l5_weight=self.l5s1_l5_weight,
            )
            out: Optional[Tuple[np.ndarray, np.ndarray]] = (
                c4.astype(np.float64),
                c5.astype(np.float64),
            )
        except (ValueError, Exception):
            out = None
        self._case_center_cache[case_id] = out
        return out

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        row = self.rows.iloc[idx]
        case_id = str(row["case_id"])
        stem = str(row["stem"])
        if stem not in STEM_TO_IDX:
            raise KeyError(f"未知 stem: {stem}")
        target_ras = np.array(
            [row["end_x_mm"], row["end_y_mm"], row["end_z_mm"]], dtype=np.float64
        )

        if self.center_mode == "heuristic":
            if not self.mask_root:
                raise RuntimeError("center_mode=heuristic 需要 mask_root")
            ct, mask_arr, affine = self._load_case_ct_mask_affine(case_id)
            if mask_arr is None:
                if case_id not in self._warned_heuristic_fallback:
                    self._warned_heuristic_fallback.add(case_id)
                    warnings.warn(
                        f"病例 {case_id} 无掩码文件，heuristic 模式回退为 target 裁块",
                        stacklevel=2,
                    )
                return self._getitem_target_centered(case_id, stem, target_ras)

            centers = self._get_heuristic_centers(case_id, mask_arr, affine)
            if centers is None:
                if case_id not in self._warned_heuristic_fallback:
                    self._warned_heuristic_fallback.add(case_id)
                    warnings.warn(
                        f"病例 {case_id} 掩码启发式失败，heuristic 模式回退为 target 裁块",
                        stacklevel=2,
                    )
                return self._getitem_target_centered(case_id, stem, target_ras)

            c4, c5 = centers
            center_ras = c4 if stem.startswith("l4_l5") else c5

            if self.center_jitter_voxels > 0 and self.train:
                grp = _stem_group_key(stem)
                rng = _det_rng(case_id, grp)
                inv = np.linalg.inv(affine)
                ijk = apply_affine(inv, center_ras.reshape(3)) + rng.uniform(
                    -self.center_jitter_voxels,
                    self.center_jitter_voxels,
                    size=3,
                ).astype(np.float64)
                center_ras = apply_affine(affine, ijk).reshape(3).astype(np.float64)

            mask_ch = mask_arr if self.mask_root else None
            t, patch_center_ras = build_patch_tensor(
                ct,
                mask_ch,
                affine,
                center_ras,
                self.patch_size,
                self.hu_min,
                self.hu_max,
            )
            x = t.squeeze(0)
            offset_mm = (target_ras - patch_center_ras).astype(np.float32)

            return {
                "x": x,
                "stem_idx": torch.tensor(STEM_TO_IDX[stem], dtype=torch.long),
                "offset": torch.from_numpy(offset_mm),
                "case_id": case_id,
            }

        return self._getitem_target_centered(case_id, stem, target_ras)

    def _getitem_target_centered(
        self,
        case_id: str,
        stem: str,
        target_ras: np.ndarray,
    ) -> Dict[str, torch.Tensor]:
        """以真值靶点为中心裁块（旧逻辑）。"""
        ct_path = find_volume_for_case(self.ct_root, case_id)
        if not ct_path:
            raise FileNotFoundError(f"未找到 CT: case_id={case_id}, ct_root={self.ct_root}")

        ct_img = load_nib(ct_path)
        ct = np.asanyarray(ct_img.dataobj)
        affine = ct_img.affine
        inv = np.linalg.inv(affine)
        ijk = apply_affine(inv, target_ras)
        if self.train and self.jitter_voxels > 0:
            ijk = ijk + np.random.uniform(
                -self.jitter_voxels, self.jitter_voxels, size=3
            ).astype(np.float64)

        pd, ph, pw = self.patch_size
        ct_pad, pad_before = _pad_min_shape(ct, (pd, ph, pw), self.hu_min)
        ijk_pad = ijk + pad_before

        ct_patch, center_vox = _crop_patch(
            ct_pad, ijk_pad, self.patch_size, fill=self.hu_min
        )
        ct_ch = _normalize_ct(ct_patch, self.hu_min, self.hu_max)

        channels: List[np.ndarray] = [ct_ch]
        use_mask_ch = self.mask_root is not None
        if use_mask_ch:
            mp = find_volume_for_case(self.mask_root, case_id)
            if mp:
                m_img = load_nib(mp)
                m = np.asanyarray(m_img.dataobj)
                if m.shape != ct.shape:
                    raise ValueError(
                        f"掩码与 CT 原始形状不一致 case={case_id}: mask {m.shape} vs ct {ct.shape}"
                    )
                m_pad, _ = _pad_min_shape(m, (pd, ph, pw), 0.0)
                if m_pad.shape != ct_pad.shape:
                    raise ValueError(
                        f"掩码 pad 后形状仍与 CT 不一致 case={case_id}"
                    )
                m_patch, _ = _crop_patch(
                    m_pad.astype(np.float32), ijk_pad, self.patch_size, fill=0.0
                )
                channels.append((m_patch > 0.5).astype(np.float32))
            else:
                channels.append(np.zeros_like(ct_ch, dtype=np.float32))

        x = np.stack(channels, axis=0)

        center_ras = apply_affine(affine, center_vox - pad_before)
        offset_mm = (target_ras - center_ras).astype(np.float32)

        return {
            "x": torch.from_numpy(x),
            "stem_idx": torch.tensor(STEM_TO_IDX[stem], dtype=torch.long),
            "offset": torch.from_numpy(offset_mm),
            "case_id": case_id,
        }


def load_csv_rows(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    need = {"case_id", "stem", "end_x_mm", "end_y_mm", "end_z_mm"}
    miss = need - set(df.columns)
    if miss:
        raise ValueError(f"CSV 缺少列: {miss}")
    df = df[df["stem"].isin(STEM_TO_IDX.keys())].copy()
    return df


def filter_rows_with_ct(df: pd.DataFrame, ct_root: str) -> pd.DataFrame:
    """只保留能解析到 CT 文件的行。"""
    keep: List[bool] = []
    for _, row in df.iterrows():
        p = find_volume_for_case(ct_root, str(row["case_id"]))
        keep.append(p is not None)
    out = df[np.array(keep)].reset_index(drop=True)
    return out


def split_by_case(
    df: pd.DataFrame,
    val_ratio: float,
    test_ratio: float,
    seed: int,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """按 case_id 划分，避免同一病人泄漏。"""
    rng = np.random.default_rng(seed)
    cases = df["case_id"].unique()
    rng.shuffle(cases)
    n = len(cases)
    n_test = int(round(n * test_ratio))
    n_val = int(round(n * val_ratio))
    n_train = n - n_val - n_test
    if n_train <= 0:
        raise ValueError("病例数过少，请减小 val/test 比例")
    c_train = set(cases[:n_train])
    c_val = set(cases[n_train : n_train + n_val])
    c_test = set(cases[n_train + n_val :])
    tr = df[df["case_id"].isin(c_train)].reset_index(drop=True)
    va = df[df["case_id"].isin(c_val)].reset_index(drop=True)
    te = df[df["case_id"].isin(c_test)].reset_index(drop=True)
    return tr, va, te
