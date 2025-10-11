"""Common neural circuit components shared across models."""

import torch
import torch.nn as nn

from ..utils import MLP

__all__ = [
    "BasalGanglia",
    "Cerebellum",
    "EIBlock",
    "ThalamicRelay",
    "BasalGangliaGate",
]


class BasalGanglia(nn.Module):
    def __init__(self, in_dim, motor_dim):
        super().__init__()
        self.pi = MLP(in_dim, 128, motor_dim, depth=2)
        self.g = MLP(in_dim, 64, motor_dim, depth=2)

    def forward(self, z):  # type: ignore[override]
        return self.pi(z) * torch.softmax(self.g(z), -1)


class Cerebellum(nn.Module):
    def __init__(self, sensory_dim, motor_dim):
        super().__init__()
        self.fm = MLP(sensory_dim + motor_dim, 128, motor_dim, depth=2)

    def forward(self, sensory, intended):  # type: ignore[override]
        return 0.2 * torch.tanh(self.fm(torch.cat([sensory, intended], -1)))


class EIBlock(nn.Module):
    """皮質カラムの極小モデル: 興奮Eと抑制Iの相互作用（1ステップ離散化）"""

    def __init__(self, d):
        super().__init__()
        self.W_e = nn.Linear(d, d, bias=False)  # E←E
        self.W_i = nn.Linear(d, d, bias=False)  # I←E
        self.U_e = nn.Linear(d, d, bias=False)  # E←I（抑制）
        self.alpha = nn.Parameter(torch.tensor(0.7))  # Eの慣性
        self.beta = nn.Parameter(torch.tensor(0.5))  # Iの慣性
        self.ln_e = nn.LayerNorm(d)
        self.ln_i = nn.LayerNorm(d)

    def forward(self, e, i, inp=None):  # type: ignore[override]
        # inpは外部入力（視覚・体性感覚などの投射を想定）
        if inp is None:
            inp = 0.0
        e_new = self.ln_e(self.alpha * e + self.W_e(e) - self.U_e(i) + inp)
        i_new = self.ln_i(self.beta * i + self.W_i(e))
        return torch.relu(e_new), torch.relu(i_new)


class ThalamicRelay(nn.Module):
    """視床リレー核: 皮質E活動を受け、選択的に再入力（可変ゲイン）"""

    def __init__(self, d):
        super().__init__()
        self.relay = nn.Linear(d, d, bias=False)
        self.gain = nn.Parameter(torch.tensor(0.8))

    def forward(self, cortical_e):  # type: ignore[override]
        return self.gain * self.relay(cortical_e)


class BasalGangliaGate(nn.Module):
    """BGゲート: コンテキストにより視床出力を選択的に通す（抑制的出力の近似）"""

    def __init__(self, d_ctx, d):
        super().__init__()
        self.policy = nn.Sequential(
            nn.Linear(d_ctx, d),
            nn.Tanh(),
            nn.Linear(d, d),
            nn.Sigmoid(),
        )

    def forward(self, ctx, thalamus_out):  # type: ignore[override]
        g = self.policy(ctx)  # [B,d]
        return g * thalamus_out, g  # 出力とゲインマップ
