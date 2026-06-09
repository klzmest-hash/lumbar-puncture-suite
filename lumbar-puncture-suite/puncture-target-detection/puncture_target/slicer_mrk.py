"""将预测靶点（RAS mm）导出为 3D Slicer 可加载的 Fiducial markups（LPS，每靶点一个 .mrk.json）。"""

from __future__ import annotations

import json
import os
from typing import Dict, List, Mapping

# 与 replan_from_manual_targets.lps_to_ras 互逆
def ras_mm_to_lps_mm(ras: List[float]) -> List[float]:
    x, y, z = float(ras[0]), float(ras[1]), float(ras[2])
    return [-x, -y, z]


SCHEMA = (
    "https://raw.githubusercontent.com/slicer/slicer/master/"
    "Modules/Loadable/Markups/Resources/Schema/markups-schema-v1.0.3.json#"
)


def _one_fiducial_markup(lps: List[float], label: str) -> Dict:
    """单点 Fiducial，结构与仓库内 Slicer 导出一致，便于直接拖入 Slicer。"""
    return {
        "@schema": SCHEMA,
        "markups": [
            {
                "type": "Fiducial",
                "coordinateSystem": "LPS",
                "coordinateUnits": "mm",
                "locked": False,
                "fixedNumberOfControlPoints": True,
                "labelFormat": "%N-%d",
                "lastUsedControlPointNumber": 1,
                "controlPoints": [
                    {
                        "id": "1",
                        "label": label,
                        "description": "",
                        "position": [lps[0], lps[1], lps[2]],
                        "orientation": [
                            -1.0,
                            -0.0,
                            -0.0,
                            -0.0,
                            -1.0,
                            -0.0,
                            0.0,
                            0.0,
                            1.0,
                        ],
                        "selected": True,
                        "locked": False,
                        "visibility": True,
                        "positionStatus": "defined",
                    }
                ],
                "measurements": [],
                "display": {
                    "visibility": True,
                    "opacity": 1.0,
                    "color": [0.4, 1.0, 1.0],
                    "selectedColor": [1.0, 0.5, 0.5],
                    "activeColor": [0.4, 1.0, 0.0],
                    "propertiesLabelVisibility": False,
                    "pointLabelsVisibility": True,
                    "textScale": 3.0,
                    "glyphType": "Sphere3D",
                    "glyphScale": 3.0,
                    "glyphSize": 5.0,
                    "useGlyphScale": True,
                },
            }
        ],
    }


STEM_ORDER = [
    "l4_l5_left",
    "l4_l5_right",
    "l5_s1_left",
    "l5_s1_right",
]


def export_fiducials_mrk(
    output_dir: str,
    end_points_ras_mm: Mapping[str, List[float]],
) -> List[str]:
    """
    在 output_dir 下写入 l4_l5_left.mrk.json 等四个文件（LPS，mm）。
    返回已写入文件路径列表。
    """
    os.makedirs(output_dir, exist_ok=True)
    written: List[str] = []
    for stem in STEM_ORDER:
        if stem not in end_points_ras_mm:
            continue
        ras = end_points_ras_mm[stem]
        lps = ras_mm_to_lps_mm(list(ras))
        label = stem.replace("_", "-")
        data = _one_fiducial_markup(lps, label)
        path = os.path.join(output_dir, f"{stem}.mrk.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=4, ensure_ascii=False)
        written.append(path)
    return written


def export_single_fiducial_mrk_json(
    out_path: str, ras_mm: List[float], label: str = "target"
) -> str:
    """
    将单个靶点 (RAS mm) 写为 3D Slicer 可拖入的 Fiducial markups (``*.mrk.json``，LPS)。"""
    lps = ras_mm_to_lps_mm([float(ras_mm[0]), float(ras_mm[1]), float(ras_mm[2])])
    data = _one_fiducial_markup(lps, label)
    p = os.path.abspath(out_path)
    d = os.path.dirname(p)
    if d:
        os.makedirs(d, exist_ok=True)
    with open(p, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4, ensure_ascii=False)
    return p
