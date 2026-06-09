"""
公开版占位超参数（非论文最终调优值）。

本文件仅说明算法中可调项的结构与量级；具体数值需在本地验证集上重新标定。
公开代码展示方法流程，不保证与论文报告数值一一复现。
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class PlanningHyperparams:
    # 髂骨零碰撞容差 (mm³)
    iliac_zero_eps_mm3: float = 1e-6

    # 关节突 / 椎间孔几何（占位，需标定）
    z_frac_l4l5: float = 0.40
    z_frac_l5s1: float = 0.50
    foramen_lateral_frac: float = 0.35
    y_post_blend: float = 0.40
    margin_z_frac: float = 0.10
    posterior_band_frac: float = 0.40
    med_lat: float = 0.50
    posterior_ratio_facet: float = 0.33
    z_tolerance_frac: float = 0.30
    lateral_frac: float = 0.33
    xy_edge_weight: float = 0.15
    band_spread_frac: float = 0.20
    band_min_mm: float = 3.0
    z_tol_band_frac: float = 0.35
    z_tol_min_mm: float = 7.0
    foramen_y_blend: float = 0.70
    facet_y_blend: float = 0.50
    dx_cap_min_mm: float = 8.0
    dx_cap_span_frac: float = 0.40
    disc_z_tol_frac: float = 0.30
    disc_z_tol_relaxed_frac: float = 0.50
    l5_inferior_z_frac: float = 0.50
    s1_superior_z_frac: float = 0.50

    # 与 puncture_target 联用时的启发式占位（建议通过 CLI 覆盖）
    z_p_l4l5: float = 50.0
    z_p_l5s1: float = 50.0
    l5s1_l5_weight: float = 0.50

    # legacy 几何靶点区域比例
    posterior_ratio: float = 0.33
    superior_ratio: float = 0.33
    inferior_ratio: float = 0.33


PLANNING = PlanningHyperparams()
