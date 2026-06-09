"""轻量 3D CNN：体素块 → 相对块中心的靶点偏移 (mm)。"""

from __future__ import annotations

import torch
import torch.nn as nn


class TargetOffsetNet3D(nn.Module):
    """
    输入: (B, C, D, H, W)，C=1 仅 CT，C=2 为 CT + mask。
    输出: (B, 3) 为相对 patch 中心的偏移，单位 mm（与标签一致）。
    stem_idx: (B,) 整型 0..3，区分 l4_l5_left / right, l5_s1_left / right。

    默认 **legacy_embedding=False**：共享 encoder，**四个 stem 各独立 3 维输出**（最后一层 12 维），
    避免 stem 小嵌入 + 单头导致左右预测塌缩。

    legacy_embedding=True：兼容旧 checkpoint（Embedding + 单 MLP 头）。
    """

    def __init__(
        self,
        in_channels: int = 1,
        num_stems: int = 4,
        stem_emb_dim: int = 16,
        base_ch: int = 32,
        legacy_embedding: bool = False,
    ) -> None:
        super().__init__()
        if in_channels not in (1, 2):
            raise ValueError("in_channels 应为 1 或 2")
        self.legacy_embedding = bool(legacy_embedding)
        self.num_stems = int(num_stems)
        ch = base_ch
        self.enc = nn.Sequential(
            self._block(in_channels, ch),
            nn.MaxPool3d(2),
            self._block(ch, ch * 2),
            nn.MaxPool3d(2),
            self._block(ch * 2, ch * 4),
            nn.MaxPool3d(2),
            self._block(ch * 4, ch * 4),
            nn.MaxPool3d(2),
        )
        self.ch_last = ch * 4
        self.pool = nn.AdaptiveAvgPool3d(1)
        if self.legacy_embedding:
            self.stem_emb = nn.Embedding(num_stems, stem_emb_dim)
            hid = 256
            self.head = nn.Sequential(
                nn.Linear(self.ch_last + stem_emb_dim, hid),
                nn.ReLU(inplace=True),
                nn.Dropout(0.2),
                nn.Linear(hid, 3),
            )
        else:
            hid = 256
            self.pre = nn.Sequential(
                nn.Linear(self.ch_last, hid),
                nn.ReLU(inplace=True),
                nn.Dropout(0.2),
            )
            self.fc_stems = nn.Linear(hid, self.num_stems * 3)

    @staticmethod
    def _block(in_c: int, out_c: int) -> nn.Sequential:
        return nn.Sequential(
            nn.Conv3d(in_c, out_c, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm3d(out_c),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_c, out_c, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm3d(out_c),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor, stem_idx: torch.Tensor) -> torch.Tensor:
        z = self.pool(self.enc(x)).flatten(1)
        if self.legacy_embedding:
            e = self.stem_emb(stem_idx)
            return self.head(torch.cat([z, e], dim=1))
        h = self.pre(z)
        out = self.fc_stems(h).view(-1, self.num_stems, 3)
        b = out.size(0)
        device = out.device
        return out[torch.arange(b, device=device), stem_idx.long()]
