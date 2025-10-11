"""Segmented ganglia baselines."""

import torch
import torch.nn as nn

from ..utils import D_AUDIT, D_OLFACT, D_PROP, D_SOMATO, D_VISION, MLP

__all__ = ["SegmentedGanglia", "SegmentedGangliaRestricted"]


# 2) Segmented ganglia: somatoを体節に分割し局所制御+全身座標
class SegmentedGanglia(nn.Module):
    def __init__(self, segments=6, motor_per_seg=2):
        super().__init__()
        assert D_SOMATO % segments == 0
        self.segments = segments
        self.local_dim = D_SOMATO // segments
        self.coord = MLP(D_VISION + D_OLFACT + D_AUDIT + D_PROP, 64, 16, depth=2)
        self.controllers = nn.ModuleList(
            [MLP(self.local_dim + 16, 64, motor_per_seg, depth=2) for _ in range(segments)]
        )

    def forward(self, x):  # type: ignore[override]
        B = x["somatosensory"].size(0)
        s = x["somatosensory"].view(B, self.segments, self.local_dim)
        c = self.coord(
            torch.cat(
                [x["vision"], x["olfaction"], x["auditory"], x["proprioception"]], -1
            )
        )
        outs = [self.controllers[i](torch.cat([s[:, i, :], c], -1)) for i in range(self.segments)]
        motor = torch.tanh(torch.cat(outs, -1))  # [B, segments*motor_per_seg]
        return {"motor": motor}


# 2) Segmented ganglia (修正版): 各体節が somato の局所情報だけで制御
class SegmentedGangliaRestricted(nn.Module):
    """
    修正版:
    - 各体節は自分の somatosensory 入力だけで motor を生成
    - グローバル情報 (vision, olfact, auditory, proprio) は利用できない
    - 本来の生物的 segmental ganglia に近く、Detour のような空間推論はできないはず
    """

    def __init__(self, segments=6, motor_per_seg=2):
        super().__init__()
        assert D_SOMATO % segments == 0
        self.segments = segments
        self.local_dim = D_SOMATO // segments
        self.controllers = nn.ModuleList(
            [MLP(self.local_dim, 32, motor_per_seg, depth=2) for _ in range(segments)]
        )

    def forward(self, x):  # type: ignore[override]
        B = x["somatosensory"].size(0)
        s = x["somatosensory"].view(B, self.segments, self.local_dim)
        outs = [self.controllers[i](s[:, i, :]) for i in range(self.segments)]
        motor = torch.tanh(torch.cat(outs, -1))  # [B, segments*motor_per_seg]
        return {"motor": motor}
