# 穿刺靶点识别（公开版）

## 核心代码

| 文件 | 说明 |
|------|------|
| `puncture_target/model.py` | `TargetOffsetNet3D` |
| `puncture_target/data.py` | 数据与启发式裁块中心 |
| `puncture_target/train.py` / `infer.py` | 训练 / 推理 |
| `hyperparams.example.yaml` | **示例**超参（复制为 `hyperparams.yaml` 后标定） |

## 训练

```bash
pip install -r requirements.txt
copy hyperparams.example.yaml hyperparams.yaml
python -m puncture_target.train --config hyperparams.yaml --csv ... --ct_root ... --out_dir runs/baseline
```

## 推理

```bash
python -m puncture_target.infer --config hyperparams.yaml --ct case.nii.gz --mask mask.nii.gz --checkpoint best.pt --out_json pred.json
```

公开版已移除：审计链、Excel 导出工具；命令行不再内置论文调优默认超参。
