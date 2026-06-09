# lumbar-puncture-suite（公开版）

CT 辅助腰椎穿刺：**3D 分割 → 靶点识别 → 工作通道路径规划**。本仓库为论文/附件用**方法参考实现**，展示算法与代码结构。

## 重要说明（公开版）

- **不含** 训练权重、数据集、论文最终超参数。
- 超参见各子目录 `*.example.yaml` 与 `planning_config.py`，数值为**占位**，需在本地验证集上重新标定。
- **不保证** 与论文报告指标/结果完全一致；完整复现需作者授权的配置与权重（若提供）。

| 目录 | 内容 |
|------|------|
| [attunet-3d-segmentation](./attunet-3d-segmentation/) | 3D Attention U-Net 分割 |
| [puncture-target-detection](./puncture-target-detection/) | L4-L5 / L5-S1 穿刺靶点 |
| [surgical-path-planning](./surgical-path-planning/) | 工作通道路径规划 |

## 推荐流水线

1. 分割 → `mask.nii.gz`
2. 靶点 → `pred_targets.json`（需自训练 `best.pt`）
3. 规划 → `results.json`（`--puncture_checkpoint` 或 CSV 靶点）

各子目录 `README.md` 有独立用法。
