"""Cnidarian nerve net baseline model."""

import math

import torch
import torch.nn as nn

from ..utils import INPUT_DIM, N_ABSTRACT_ACTIONS

__all__ = ["CnidarianNerveNet"]


# 1) Cnidarian: fixed random features (+small plastic head)
class CnidarianNerveNet(nn.Module):
    """
    生物の拡散神経網の素朴近似：入力→固定ランダム射影→tanh→小さな可塑ヘッド。
    ユニバーサル近似器になりにくい容量に制限。
    """

    def __init__(self, motor_dim=8, feat_dim=48, abstract_dim=None):
        super().__init__()
        with torch.no_grad():
            W = torch.randn(INPUT_DIM, feat_dim) / math.sqrt(INPUT_DIM)
            b = torch.zeros(feat_dim)
        self.register_buffer("W", W)
        self.register_buffer("b", b)
        self.head = nn.Linear(feat_dim, motor_dim)
        self.motor_dim = motor_dim
        self.abstract_dim = abstract_dim or N_ABSTRACT_ACTIONS
        self.abstract_head = nn.Linear(feat_dim, self.abstract_dim)

    def forward(self, x):  # type: ignore[override]
        z = torch.cat(
            [
                x["vision"],
                x["olfaction"],
                x["somatosensory"],
                x["auditory"],
                x["proprioception"],
            ],
            -1,
        )
        h = torch.tanh(z @ self.W + self.b)  # [B,feat]
        motor = torch.tanh(self.head(h))
        abstract_logits = self.abstract_head(h)
        return {"motor": motor, "abstract_logits": abstract_logits}
