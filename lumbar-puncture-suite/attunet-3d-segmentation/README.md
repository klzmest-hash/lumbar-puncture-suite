# 3D Attention U-Net 分割（公开版）

## 核心代码

| 文件 | 说明 |
|------|------|
| `models/attunet_3d.py` | 网络结构 |
| `train_and_test/custom_3D/attunet-fifth-3d.py` | 训练入口 |
| `inference_attunet_3d.py` | 推理入口 |
| `configs/custom_3D/attunet_3d.example.yaml` | **示例**超参（需复制后标定） |

## 训练

```bash
pip install -r requirements.txt
python train_and_test/custom_3D/attunet-fifth-3d.py --config custom_3D/attunet_3d.example.yaml
```

## 推理

```bash
python inference_attunet_3d.py --checkpoint path/to/best.pt --input <图像目录> --output <输出目录>
```

公开版已移除：多模型推理、实验审计、论文专用 `fifth_attunet_3d.yaml` 配置。
