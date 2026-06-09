# 手术工作通道路径规划（公开版）

## 核心代码

| 文件 | 说明 |
|------|------|
| `path_planning_algorithm.py` | 通道采样、交集体积、字典序最优路径 |
| `planning_config.py` | 几何/启发式**占位**常数（非论文最终值） |

## 用法

```bash
pip install -r requirements.txt
python path_planning_algorithm.py --input mask.nii.gz --ct ct.nii.gz --output results/out.json --filter_optimal
```

几何比例、椎间孔定位、靶点启发式等见 `planning_config.py`，请按本地数据调节。

公开版已移除：实验审计、内置论文级后处理默认（如 infer 脚本专用的左右拉开 14 mm 等）。
