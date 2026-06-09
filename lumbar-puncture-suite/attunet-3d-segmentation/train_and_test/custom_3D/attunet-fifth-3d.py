#!/usr/bin/env python
# coding: utf-8
# AttUNet 3D - fifth（模仿 transunet-fifth-3d.py）
# 第一部分：导入、配置、数据集与 DataLoader
#
# 公开版：超参见 configs/custom_3D/attunet_3d.example.yaml

from __future__ import print_function, division

import os
import sys

current_working_dir = os.getcwd()
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(script_dir))

if project_root not in sys.path:
    sys.path.insert(0, project_root)
if current_working_dir not in sys.path and os.path.exists(os.path.join(current_working_dir, 'losses.py')):
    sys.path.insert(0, current_working_dir)

import argparse
import json
from glob import glob
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

import torch
try:
    import pandas as pd
    HAS_PANDAS = True
except ImportError:
    HAS_PANDAS = False
try:
    import openpyxl
    HAS_OPENPYXL = True
except ImportError:
    HAS_OPENPYXL = False
import torch.optim as optim
import torchmetrics

from losses import DiceLossWithLogtis
from datasets.dataset_3d import Dataset3D
from torch.nn import CrossEntropyLoss

from utils import load_config, _print

import warnings
warnings.filterwarnings("ignore")

torch.manual_seed(0)
np.random.seed(0)
torch.cuda.manual_seed(0)
import random
random.seed(0)


def _run_attunet_3d_training(config_name: str):
    CONFIG_NAME = config_name
    CONFIG_FILE_PATH = os.path.join("configs", CONFIG_NAME)

    config = load_config(CONFIG_FILE_PATH)
    _print("Config:", "info_underline")
    print(json.dumps(config, indent=2))
    print(20 * "~-", "\n")

    _print("训练参数（来自 yaml）", "info_underline")
    print(f"  input_size (D,H,W): {config['dataset']['input_size']}")
    print(f"  epochs: {config['training']['epochs']}")
    print(f"  batch_size (train): {config['data_loader']['train']['batch_size']}\n")

    save_dir_abs = os.path.abspath(config['model']['save_dir'])
    config_abs = os.path.abspath(CONFIG_FILE_PATH)

    gpu_id = config['run'].get('gpu', config['run'].get('device', '0'))
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    from torch.utils.data import DataLoader, ConcatDataset

    INPUT_SIZE = config['dataset']['input_size']
    if isinstance(INPUT_SIZE, int):
        INPUT_SIZE = (INPUT_SIZE, INPUT_SIZE, INPUT_SIZE)
    elif isinstance(INPUT_SIZE, (list, tuple)):
        if len(INPUT_SIZE) == 2:
            INPUT_SIZE = (INPUT_SIZE[0], INPUT_SIZE[0], INPUT_SIZE[1])
        elif len(INPUT_SIZE) == 3:
            INPUT_SIZE = tuple(INPUT_SIZE)
    print(f"Input size (3D): {INPUT_SIZE}")

    data_root = config['dataset'].get('data_root', 'inputs')
    if not os.path.isabs(data_root):
        data_root = os.path.abspath(data_root)
    img_ext = config['dataset']['img_ext']

    print(f"\n{'='*80}")
    print("🔍 数据集路径诊断信息")
    print(f"{'='*80}")
    print(f"📁 数据集根目录: {data_root}")

    # 支持两种布局，通过 config['dataset']['use_multi_center'] 控制：
    # 1) 单中心：inputs/3D/train/images
    # 2) 多中心：inputs/3D/kh/train/images, inputs/3D/zs/train/images, ...
    use_multi_center = config['dataset'].get('use_multi_center', False)
    centers_cfg = config['dataset'].get('centers', []) or []
    if not use_multi_center:
        print("布局: 单中心 (train/val/test 直接位于 data_root 下)")
        centers = [""]
    else:
        multi_center_roots = []
        if centers_cfg:
            # 显式指定中心名称
            for name in centers_cfg:
                p = os.path.join(data_root, name)
                if os.path.isdir(os.path.join(p, "train", "images")):
                    multi_center_roots.append(name)
        else:
            # 自动扫描 data_root 下含 train/images 的子目录
            for name in sorted(os.listdir(data_root)):
                p = os.path.join(data_root, name)
                if not os.path.isdir(p):
                    continue
                if os.path.isdir(os.path.join(p, "train", "images")):
                    multi_center_roots.append(name)
        centers = multi_center_roots
        print(f"布局: 多中心 ({len(centers)} 个中心): {centers}")

    def _glob_split(split):
        """汇总所有中心的某个划分（train/val/test）的图像 ID 列表。"""
        img_ids = []
        for c in centers:
            base = data_root if c == "" else os.path.join(data_root, c)
            img_dir = os.path.join(base, split, "images")
            if img_ext == ".nii.gz":
                paths = glob(os.path.join(img_dir, "*" + img_ext))
                ids = [os.path.basename(p).replace(".nii.gz", "") for p in paths]
            else:
                paths = glob(os.path.join(img_dir, "*" + img_ext))
                ids = [os.path.splitext(os.path.basename(p))[0] for p in paths]
            img_ids.extend(ids)
        return sorted(set(img_ids))

    img_train = _glob_split("train")
    img_val = _glob_split("val")
    img_test = _glob_split("test")

    print(f"{'='*80}")
    print(f"   训练图像总数: {len(img_train)}")
    print(f"   验证图像总数: {len(img_val)}")
    print(f"   测试图像总数: {len(img_test)}")
    print(f"{'='*80}\n")

    train_img_ids, val_img_ids, test_img_ids = img_train, img_val, img_test
    print(f"训练集: {len(train_img_ids)} 个 | 验证集: {len(val_img_ids)} 个 | 测试集: {len(test_img_ids)} 个\n")

    if len(train_img_ids) == 0 and len(val_img_ids) == 0 and len(test_img_ids) == 0:
        raise ValueError("数据集为空")
    if len(train_img_ids) == 0:
        raise ValueError("训练集为空")

    train_transform = val_transform = None
    class_mapping = {0: 0, 1: 1, 2: 2, 3: 3, 4: 4, 5: 5, 6: 6, 10: 7, 11: 8, 12: 9, 13: 10, 14: 11, 15: 12, 16: 13, 17: 14, 18: 15, 19: 16}
    shared_class_mapping = class_mapping

    # 各结构名称（与 inference_3d_example.py 中 LABEL_NAME_MAPPING 一致）
    # 真实标签为 0-6、10-19；训练时 class_mapping 映射为模型索引 0~16，此处按模型索引 0~16 对应名称
    STRUCTURE_NAMES = {
        0: 'Background',   # 真实标签 0
        1: 'L1', 2: 'L2', 3: 'L3', 4: 'L4', 5: 'L5', 6: 'S1',   # 真实标签 1-6
        7: 'PMR', 8: 'QLR', 9: 'ESR', 10: 'MFR', 11: 'PML', 12: 'QLL', 13: 'ESL', 14: 'MFL',   # 真实标签 10-17
        15: 'Ilium_left', 16: 'Ilium_right',   # 真实标签 18, 19
    }
    # 模型索引 -> 真实标签（与 class_mapping 反向，便于表格中展示）
    MODEL_INDEX_TO_REAL_LABEL = {
        0: 0, 1: 1, 2: 2, 3: 3, 4: 4, 5: 5, 6: 6,
        7: 10, 8: 11, 9: 12, 10: 13, 11: 14, 12: 15, 13: 16, 14: 17, 15: 18, 16: 19,
    }

    print("创建训练/验证/测试数据集...")

    def _build_split_dataset(split, ids, transform, mapping):
        """根据当前中心布局构建指定划分的 Dataset 或 ConcatDataset。"""
        datasets = []
        for c in centers:
            base = data_root if c == "" else os.path.join(data_root, c)
            img_dir = os.path.join(base, split, "images")
            mask_dir = os.path.join(base, split, "masks")
            # 当前中心下实际存在的 ID
            if img_ext == ".nii.gz":
                paths = glob(os.path.join(img_dir, "*" + img_ext))
                local_ids = {os.path.basename(p).replace(".nii.gz", "") for p in paths}
            else:
                paths = glob(os.path.join(img_dir, "*" + img_ext))
                local_ids = {os.path.splitext(os.path.basename(p))[0] for p in paths}
            split_ids = [i for i in ids if i in local_ids]
            if len(split_ids) == 0:
                continue
            ds = Dataset3D(
                img_ids=split_ids,
                img_dir=img_dir,
                mask_dir=mask_dir,
                img_ext=config["dataset"]["img_ext"],
                mask_ext=config["dataset"]["mask_ext"],
                num_classes=config["dataset"]["num_classes"],
                transform=transform,
                input_size=INPUT_SIZE,
                class_mapping=mapping,
            )
            datasets.append(ds)
        if len(datasets) == 0:
            return None
        if len(datasets) == 1:
            return datasets[0]
        return ConcatDataset(datasets)

    tr_dataset = _build_split_dataset("train", train_img_ids, train_transform, class_mapping)
    vl_dataset = _build_split_dataset("val", val_img_ids, val_transform, shared_class_mapping)
    te_dataset = _build_split_dataset("test", test_img_ids, val_transform, shared_class_mapping)

    print(f"训练: {len(tr_dataset)} | 验证: {len(vl_dataset)} | 测试: {len(te_dataset)}")

    import platform
    if platform.system() == 'Windows' or True:
        for k in ['train', 'validation', 'test']:
            if config['data_loader'][k].get('num_workers', 0) > 0:
                config['data_loader'][k]['num_workers'] = 2
            config['data_loader'][k]['persistent_workers'] = False
    print("ℹ️  3D数据加载：禁用persistent_workers")

    tr_dataloader = DataLoader(tr_dataset, **config['data_loader']['train'])
    vl_dataloader = DataLoader(vl_dataset, **config['data_loader']['validation'])
    te_dataloader = DataLoader(te_dataset, **config['data_loader']['test'])
    print("✅ DataLoader 创建完成\n")
    print("⚠️  跳过数据加载测试（3D数据resize较慢）\n")

    # ---------- 第二部分：device、AMP、指标、validate、train ----------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Torch device: {device}, GPU: {gpu_id}")

    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False
        print("✅ 已启用 cudnn.benchmark")

    USE_AMP = config.get('training', {}).get('use_amp', True)
    if USE_AMP and torch.cuda.is_available():
        scaler = torch.cuda.amp.GradScaler()
        print("✅ 已启用混合精度训练 (AMP)")
    else:
        scaler = None
        print("⚠️  未启用 AMP")

    num_classes = config['dataset']['num_classes']

    class DiceMetric3D(torchmetrics.Metric):
        def __init__(self, num_classes):
            super().__init__()
            self.num_classes = num_classes
            self.add_state("intersection", default=torch.zeros(num_classes), dist_reduce_fx="sum")
            self.add_state("pred_sum", default=torch.zeros(num_classes), dist_reduce_fx="sum")
            self.add_state("target_sum", default=torch.zeros(num_classes), dist_reduce_fx="sum")

        def update(self, preds, target):
            preds, target = preds.long(), target.long()
            for c in range(self.num_classes):
                pred_c, target_c = (preds == c), (target == c)
                self.intersection[c] += (pred_c & target_c).sum()
                self.pred_sum[c] += pred_c.sum()
                self.target_sum[c] += target_c.sum()

        def compute(self):
            dice = 2 * self.intersection / (self.pred_sum + self.target_sum + 1e-8)
            return dice.mean()

    def evaluate_per_class_metrics(model, dataloader, num_cls, use_amp=True):
        """在指定 DataLoader 上计算每个类别的平均 Dice、Precision、Recall。返回三个 numpy 数组，索引为类别 0..num_cls-1。"""
        model.eval()
        inter = torch.zeros(num_cls, device=device, dtype=torch.float64)
        pred_sum = torch.zeros(num_cls, device=device, dtype=torch.float64)
        target_sum = torch.zeros(num_cls, device=device, dtype=torch.float64)
        with torch.no_grad():
            for batch_data in tqdm(dataloader, desc="Per-class metrics"):
                imgs = batch_data['image'].to(device, non_blocking=True)
                msks = batch_data['mask'].to(device, non_blocking=True)
                if use_amp and scaler is not None:
                    with torch.cuda.amp.autocast():
                        preds = model(imgs)
                else:
                    preds = model(imgs)
                preds_ = torch.argmax(preds, 1, keepdim=False).long()
                msks_ = torch.argmax(msks, 1, keepdim=False).long()
                for c in range(num_cls):
                    pred_c = (preds_ == c)
                    target_c = (msks_ == c)
                    inter[c] += (pred_c & target_c).sum()
                    pred_sum[c] += pred_c.sum()
                    target_sum[c] += target_c.sum()
        dice = (2 * inter / (pred_sum + target_sum + 1e-8)).cpu().numpy()
        precision = (inter / (pred_sum + 1e-8)).cpu().numpy()
        recall = (inter / (target_sum + 1e-8)).cpu().numpy()
        return dice, precision, recall

    metric_list = [DiceMetric3D(num_classes=num_classes)]
    metrics = torchmetrics.MetricCollection(metric_list, prefix='train_metrics/')
    train_metrics = metrics.clone(prefix='train_metrics/').to(device)
    valid_metrics = metrics.clone(prefix='valid_metrics/').to(device)
    test_metrics = metrics.clone(prefix='test_metrics/').to(device)

    def make_serializeable_metrics(computed_metrics):
        return {k: float(v.cpu().detach().numpy()) for k, v in computed_metrics.items()}

    def validate(model, criterion, vl_dataloader):
        model.eval()
        with torch.no_grad():
            evaluator = valid_metrics.clone().to(device)
            losses, cnt = [], 0.0
            for batch, batch_data in enumerate(vl_dataloader):
                imgs = batch_data['image'].to(device, non_blocking=True)
                msks = batch_data['mask'].to(device, non_blocking=True)
                cnt += msks.shape[0]
                if scaler is not None:
                    with torch.cuda.amp.autocast():
                        preds = model(imgs)
                        loss = criterion(preds, msks)
                else:
                    preds = model(imgs)
                    loss = criterion(preds, msks)
                losses.append(loss.item())
                preds_ = torch.argmax(preds, 1, keepdim=False)
                msks_ = torch.argmax(msks, 1, keepdim=False)
                evaluator.update(preds_, msks_)
            loss = np.sum(losses) / cnt
        return evaluator, loss

    tr_prms = config['training']

    def train(model, device, tr_dataloader, vl_dataloader, config, criterion, optimizer, scheduler, save_dir='./', save_file_id=None):
        EPOCHS = tr_prms['epochs']
        torch.cuda.empty_cache()
        model = model.to(device)
        evaluator = train_metrics.clone().to(device)
        epochs_info, best_model, best_result, best_vl_loss = [], None, {}, np.Inf

        for epoch in range(EPOCHS):
            model.train()
            evaluator.reset()
            tr_iterator = tqdm(enumerate(tr_dataloader), total=len(tr_dataloader))
            tr_losses, cnt = [], 0
            for batch, batch_data in tr_iterator:
                imgs = batch_data['image'].to(device, non_blocking=True)
                msks = batch_data['mask'].to(device, non_blocking=True)
                optimizer.zero_grad()
                if scaler is not None:
                    with torch.cuda.amp.autocast():
                        preds = model(imgs)
                        loss = criterion(preds, msks)
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    preds = model(imgs)
                    loss = criterion(preds, msks)
                    loss.backward()
                    optimizer.step()
                preds_ = torch.argmax(preds, 1, keepdim=False)
                msks_ = torch.argmax(msks, 1, keepdim=False)
                evaluator.update(preds_, msks_)
                cnt += imgs.shape[0]
                tr_losses.append(loss.item())
                tr_iterator.set_description(f"Training) ep:{epoch:03d}, batch:{batch+1:04d} -> loss:{np.sum(tr_losses)/cnt:.5f}")

            tr_loss = np.sum(tr_losses) / cnt
            vl_metrics, vl_loss = validate(model, criterion, vl_dataloader)
            tr_metrics_computed = evaluator.compute()
            vl_metrics_computed = vl_metrics.compute()

            improved = vl_loss < best_vl_loss
            if improved:
                best_model, best_vl_loss = model, vl_loss
                best_result = {'tr_loss': tr_loss, 'vl_loss': vl_loss, 'tr_metrics': make_serializeable_metrics(tr_metrics_computed), 'vl_metrics': make_serializeable_metrics(vl_metrics_computed)}

            epochs_info.append({'tr_loss': tr_loss, 'vl_loss': vl_loss, 'tr_metrics': make_serializeable_metrics(tr_metrics_computed), 'vl_metrics': make_serializeable_metrics(vl_metrics_computed)})
            tr_dice = tr_metrics_computed.get('train_metrics/DiceMetric3D', torch.tensor(0.0))
            vl_dice = vl_metrics_computed.get('valid_metrics/DiceMetric3D', torch.tensor(0.0))
            tr_dice = tr_dice.item() if isinstance(tr_dice, torch.Tensor) else tr_dice
            vl_dice = vl_dice.item() if isinstance(vl_dice, torch.Tensor) else vl_dice
            print(f"\nEpoch {epoch+1}/{EPOCHS} 完成 - 训练 Loss: {tr_loss:.6f}, Dice: {tr_dice:.6f} | 验证 Loss: {vl_loss:.6f}, Dice: {vl_dice:.6f}")
            evaluator.reset()
            scheduler.step(vl_loss)

        os.makedirs(config['model']['save_dir'], exist_ok=True)
        fn = f"{save_file_id + '_' if save_file_id else ''}result.json"
        result_json_path = os.path.join(config['model']['save_dir'], fn)
        with open(result_json_path, "w") as f:
            json.dump({'id': save_file_id, 'config': config, 'epochs_info': epochs_info, 'best_result': best_result}, f, indent=4)
        last_pt = os.path.join(config['model']['save_dir'], "last_model_state_dict.pt")
        best_pt = os.path.join(config['model']['save_dir'], "best_model_state_dict.pt")
        torch.save(model.state_dict(), last_pt)
        torch.save(best_model.state_dict(), best_pt)
        return best_model, model, {'epochs_info': epochs_info, 'best_result': best_result}

    # ---------- 第三部分：test、模型、criterion/optimizer/scheduler、训练与测试、绘图 ----------
    import nibabel as nib

    def test(model, te_dataloader):
        save_outputs_dir = os.path.join(config['model']['save_dir'], "outputs")
        os.makedirs(save_outputs_dir, exist_ok=True)
        model.eval()
        with torch.no_grad():
            evaluator = test_metrics.clone().to(device)
            for batch_data in tqdm(te_dataloader):
                imgs = batch_data['image'].to(device)
                msks = batch_data['mask'].to(device)
                ids = batch_data['id']
                preds = model(imgs)
                preds_ = torch.argmax(preds, 1).cpu().numpy()
                msks_ = torch.argmax(msks, 1).cpu().numpy()
                evaluator.update(torch.from_numpy(preds_), torch.from_numpy(msks_))
                for i in range(len(preds_)):
                    fid = ids[i]
                    pred_volume = preds_[i].astype(np.uint8)
                    nii_img = nib.Nifti1Image(pred_volume, affine=np.eye(4))
                    nib.save(nii_img, os.path.join(save_outputs_dir, fid + '.nii.gz'))
        return evaluator

    from models.attunet_3d import AttU_Net3d as Net

    model = Net(**config['model']['params'])
    torch.cuda.empty_cache()
    model = model.to(device)
    print(f"模型参数量: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

    os.makedirs(config['model']['save_dir'], exist_ok=True)
    if config['model']['load_weights']:
        model.load_state_dict(torch.load(os.path.join(config['model']['save_dir'], "model_state_dict.pt")))
        print("Loaded pre-trained weights...")

    criterion_dice = DiceLossWithLogtis()
    criterion_ce = CrossEntropyLoss()

    def criterion(preds, masks):
        c_dice = criterion_dice(preds, masks)
        masks_long = masks.argmax(dim=1)
        c_ce = criterion_ce(preds, masks_long)
        return 0.5 * c_dice + 0.5 * c_ce

    optimizer = optim.Adam(model.parameters(), **tr_prms['optimizer']['params'])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', **tr_prms['scheduler'])

    best_model, model, res = train(
        model, device, tr_dataloader, vl_dataloader, config,
        criterion, optimizer, scheduler,
        save_dir=config['model']['save_dir'], save_file_id=None,
    )

    # 训练结束后：各结构在训练集/验证集上的平均 Dice、Precision、Recall，导出 Excel/CSV
    _print("各结构指标（训练集/验证集）", "info_underline")
    tr_dice_pc, tr_prec_pc, tr_rec_pc = evaluate_per_class_metrics(best_model, tr_dataloader, num_classes, use_amp=USE_AMP)
    vl_dice_pc, vl_prec_pc, vl_rec_pc = evaluate_per_class_metrics(best_model, vl_dataloader, num_classes, use_amp=USE_AMP)
    metrics_export_path = None
    if HAS_PANDAS and HAS_OPENPYXL:
        rows = []
        for c in range(num_classes):
            name = STRUCTURE_NAMES.get(c, f"Class_{c}")
            real_label = MODEL_INDEX_TO_REAL_LABEL.get(c, c)
            rows.append({
                '结构': name,
                '真实标签': real_label,
                '训练集_Dice': round(float(tr_dice_pc[c]), 6),
                '训练集_Precision': round(float(tr_prec_pc[c]), 6),
                '训练集_Recall': round(float(tr_rec_pc[c]), 6),
                '验证集_Dice': round(float(vl_dice_pc[c]), 6),
                '验证集_Precision': round(float(vl_prec_pc[c]), 6),
                '验证集_Recall': round(float(vl_rec_pc[c]), 6),
            })
        df = pd.DataFrame(rows)
        excel_path = os.path.join(config['model']['save_dir'], "per_structure_metrics.xlsx")
        df.to_excel(excel_path, index=False, sheet_name="per_structure")
        metrics_export_path = excel_path
        print(f"✅ 各结构训练集/验证集 Dice、Precision、Recall 已导出: {excel_path}")
    else:
        if not HAS_PANDAS:
            print("⚠️ 未安装 pandas，无法导出 Excel。请运行: pip install pandas openpyxl")
        elif not HAS_OPENPYXL:
            print("⚠️ 未安装 openpyxl，无法导出 .xlsx。请运行: pip install openpyxl")
        csv_path = os.path.join(config['model']['save_dir'], "per_structure_metrics.csv")
        with open(csv_path, 'w', encoding='utf-8-sig') as f:
            f.write("结构,真实标签,训练集_Dice,训练集_Precision,训练集_Recall,验证集_Dice,验证集_Precision,验证集_Recall\n")
            for c in range(num_classes):
                name = STRUCTURE_NAMES.get(c, f"Class_{c}")
                real_label = MODEL_INDEX_TO_REAL_LABEL.get(c, c)
                f.write(f"{name},{real_label},{tr_dice_pc[c]:.6f},{tr_prec_pc[c]:.6f},{tr_rec_pc[c]:.6f},{vl_dice_pc[c]:.6f},{vl_prec_pc[c]:.6f},{vl_rec_pc[c]:.6f}\n")
        metrics_export_path = csv_path
        print(f"✅ 各结构指标已导出 CSV: {csv_path}")

    per_class_val_dice = {
        STRUCTURE_NAMES.get(c, f"Class_{c}"): float(vl_dice_pc[c])
        for c in range(num_classes)
    }

    te_metrics = test(best_model, te_dataloader)
    te_metrics_computed = te_metrics.compute()
    test_metrics_dict = make_serializeable_metrics(te_metrics_computed)
    with open(os.path.join(config['model']['save_dir'], "test_metrics.txt"), "w") as f:
        for key, value in test_metrics_dict.items():
            f.write(f"{key}: {value}\n")
    print(f"Test metrics saved to {config['model']['save_dir']}/test_metrics.txt")

    with open(os.path.join(config['model']['save_dir'], "result.json"), 'r') as f:
        results = json.load(f)
    epochs_info = results['epochs_info']
    tr_losses = [d['tr_loss'] for d in epochs_info]
    vl_losses = [d['vl_loss'] for d in epochs_info]
    tr_dice = [d['tr_metrics'].get('train_metrics/DiceMetric3D', 0.0) for d in epochs_info]
    vl_dice = [d['vl_metrics'].get('valid_metrics/DiceMetric3D', 0.0) for d in epochs_info]

    _, axs = plt.subplots(1, 2, figsize=[12, 3])
    axs[0].set_title("Loss")
    axs[0].plot(tr_losses, 'r-', label="train loss")
    axs[0].plot(vl_losses, 'b-', label="validation loss")
    axs[0].legend()
    axs[1].set_title("Dice score")
    axs[1].plot(tr_dice, 'r-', label="train dice")
    axs[1].plot(vl_dice, 'b-', label="validation dice")
    axs[1].legend()
    plt.savefig(os.path.join(config['model']['save_dir'], "result.png"))
    print(f"Training curves saved to {config['model']['save_dir']}/result.png")

    best = res.get('best_result') or {}
    best_vl_metrics = best.get('vl_metrics') or {}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='AttUNet 3D 训练与验证（公开版）')
    parser.add_argument(
        '--config',
        type=str,
        default='custom_3D/attunet_3d.example.yaml',
        help='YAML 配置（相对 configs/）；请复制 example 后自行标定',
    )
    args = parser.parse_args()
    os.chdir(project_root)
    _run_attunet_3d_training(args.config)
