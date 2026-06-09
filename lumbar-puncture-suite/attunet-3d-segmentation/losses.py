#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
损失函数模块
包含Dice Loss等分割任务常用的损失函数
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class DiceLoss(nn.Module):
    """Dice Loss（用于二分类或多分类）"""
    
    def __init__(self, smooth=1.0):
        """
        Args:
            smooth: 平滑系数，避免分母为0
        """
        super(DiceLoss, self).__init__()
        self.smooth = smooth
    
    def forward(self, pred, target):
        """
        Args:
            pred: 预测值 (B, C, H, W) 或 (B, H, W)
            target: 真实值 (B, C, H, W) 或 (B, H, W)，one-hot编码
        """
        # 如果target是one-hot编码，需要转换为类别索引
        if target.dim() == 4 and target.size(1) > 1:
            target = torch.argmax(target, dim=1)
        
        # 如果pred是logits，应用softmax
        if pred.dim() == 4:
            pred = F.softmax(pred, dim=1)
            # 转换为类别概率
            pred = torch.argmax(pred, dim=1)
        
        # 展平
        pred = pred.contiguous().view(-1)
        target = target.contiguous().view(-1)
        
        # 计算Dice系数
        intersection = (pred * target).sum()
        dice = (2. * intersection + self.smooth) / (pred.sum() + target.sum() + self.smooth)
        
        return 1 - dice


class DiceLossWithLogtis(nn.Module):
    """
    Dice Loss with Logits（直接使用logits，内部应用softmax）
    支持 2D (B, C, H, W) 和 3D (B, C, D, H, W) 输入。
    """
    
    def __init__(self, smooth=1.0):
        """
        Args:
            smooth: 平滑系数，避免分母为0
        """
        super(DiceLossWithLogtis, self).__init__()
        self.smooth = smooth
    
    def forward(self, pred, target):
        """
        Args:
            pred: 预测logits，shape 可以是 (B, C, H, W) 或 (B, C, D, H, W)
            target: 真实值，one-hot 编码 (B, C, ...) 或类别索引 (B, ...)
        """
        # 应用softmax得到概率（在类别维度上）
        pred = F.softmax(pred, dim=1)
        
        # 如果 target 是 one-hot，转换为类别索引
        if target.dim() == pred.dim() and target.size(1) > 1:
            target = torch.argmax(target, dim=1)
        
        num_classes = pred.size(1)
        dice_loss = 0.0
        
        # 对每个类别计算 Dice Loss
        for c in range(num_classes):
            # 取出类别 c 的预测，展平除 batch 之外的所有维度
            pred_c = pred[:, c, ...].contiguous().view(-1)
            target_c = (target == c).float().contiguous().view(-1)
            
            intersection = (pred_c * target_c).sum()
            dice = (2. * intersection + self.smooth) / (pred_c.sum() + target_c.sum() + self.smooth)
            dice_loss += (1 - dice)
        
        # 平均所有类别的损失
        return dice_loss / num_classes


# 为了兼容性，也可以提供其他损失函数
class BCEDiceLoss(nn.Module):
    """BCE + Dice Loss组合"""
    
    def __init__(self, bce_weight=0.5, dice_weight=0.5, smooth=1.0):
        super(BCEDiceLoss, self).__init__()
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight
        self.bce = nn.BCEWithLogitsLoss()
        self.dice = DiceLossWithLogtis(smooth=smooth)
    
    def forward(self, pred, target):
        bce_loss = self.bce(pred, target)
        dice_loss = self.dice(pred, target)
        return self.bce_weight * bce_loss + self.dice_weight * dice_loss

