#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
工具函数模块
包含配置加载、打印、可视化等功能
"""

import os
import yaml
import json
import torch
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, Any


def load_config(config_path: str) -> Dict[str, Any]:
    """
    加载YAML配置文件
    
    Args:
        config_path: 配置文件路径
        
    Returns:
        配置字典
    """
    # 如果路径是相对路径，尝试从不同位置查找
    if not os.path.isabs(config_path):
        # 尝试从当前目录
        if os.path.exists(config_path):
            pass
        # 尝试从Awesome-U-Net根目录
        elif os.path.exists(os.path.join('configs', os.path.basename(config_path))):
            config_path = os.path.join('configs', os.path.basename(config_path))
        # 尝试从configs目录
        elif os.path.exists(os.path.join(os.path.dirname(__file__), '..', 'configs', os.path.basename(config_path))):
            config_path = os.path.join(os.path.dirname(__file__), '..', 'configs', os.path.basename(config_path))
        else:
            # 尝试完整路径
            possible_paths = [
                config_path,
                os.path.join('configs', config_path),
                os.path.join('configs', os.path.basename(config_path)),
            ]
            for path in possible_paths:
                if os.path.exists(path):
                    config_path = path
                    break
    
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"配置文件不存在: {config_path}")
    
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    return config


def _print(message: str, style: str = "info"):
    """
    格式化打印信息
    
    Args:
        message: 要打印的消息
        style: 样式 ('info', 'info_underline', 'warning', 'error', 'success')
    """
    styles = {
        'info': '',
        'info_underline': '\033[4m',  # 下划线
        'warning': '\033[93m',  # 黄色
        'error': '\033[91m',  # 红色
        'success': '\033[92m',  # 绿色
    }
    
    reset = '\033[0m'  # 重置样式
    
    if style in styles:
        print(f"{styles[style]}{message}{reset}")
    else:
        print(message)


def show_sbs(image, mask, title="Image and Mask"):
    """
    并排显示图像和掩码
    
    Args:
        image: 图像 (C, H, W) 或 (H, W, C) 的tensor或numpy数组
        mask: 掩码 (H, W) 或 (C, H, W) 的tensor或numpy数组
        title: 图像标题
    """
    # 转换为numpy数组
    if isinstance(image, torch.Tensor):
        image = image.cpu().numpy()
    if isinstance(mask, torch.Tensor):
        mask = mask.cpu().numpy()
    
    # 处理图像维度
    if image.ndim == 3:
        if image.shape[0] == 3 or image.shape[0] == 1:  # (C, H, W)
            image = np.transpose(image, (1, 2, 0))
        if image.shape[2] == 1:  # 单通道
            image = image.squeeze(2)
    
    # 处理掩码维度
    if mask.ndim == 3:
        if mask.shape[0] > 1:  # one-hot编码
            mask = np.argmax(mask, axis=0)
        else:
            mask = mask.squeeze(0)
    
    # 归一化到[0, 1]
    if image.max() > 1.0:
        image = image / 255.0
    
    # 创建图形
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    
    # 显示图像
    if image.ndim == 2:
        axes[0].imshow(image, cmap='gray')
    else:
        axes[0].imshow(image)
    axes[0].set_title('Image')
    axes[0].axis('off')
    
    # 显示掩码
    axes[1].imshow(mask, cmap='jet')
    axes[1].set_title('Mask')
    axes[1].axis('off')
    
    plt.suptitle(title)
    plt.tight_layout()
    plt.show()


def save_config(config: Dict[str, Any], save_path: str):
    """
    保存配置到YAML文件
    
    Args:
        config: 配置字典
        save_path: 保存路径
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, 'w', encoding='utf-8') as f:
        yaml.dump(config, f, default_flow_style=False, allow_unicode=True)


def count_parameters(model):
    """
    统计模型参数数量
    
    Args:
        model: PyTorch模型
        
    Returns:
        总参数数量，可训练参数数量
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total_params, trainable_params


def set_seed(seed: int = 42):
    """
    设置随机种子，保证可重复性
    
    Args:
        seed: 随机种子
    """
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

