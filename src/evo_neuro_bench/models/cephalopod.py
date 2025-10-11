"""Cephalopod model with recurrent loops and arm ganglia."""

import torch
import torch.nn as nn

from ..utils import D_SOMATO, D_VISION

__all__ = ["ReflexArc", "BrachialGanglion", "CephalopodBrainV3"]


# ---------------------------------------------
# ReflexArc: 局所反射弓（somatosensory → motor）
# ---------------------------------------------
class ReflexArc(nn.Module):
    def __init__(self, sensory_dim, motor_dim, hidden=16):
        super().__init__()
        self.afferent = nn.Linear(sensory_dim, hidden)
        self.interneuron = nn.ReLU()
        self.efferent = nn.Linear(hidden, motor_dim)

    def forward(self, sensory_input):  # type: ignore[override]
        h = self.interneuron(self.afferent(sensory_input))
        motor = torch.tanh(self.efferent(h))
        return motor


# ---------------------------------------------
# BrachialGanglion: 腕ガングリオン
# ReflexArc と中央脳入力を学習可能パラメータで統合
# ---------------------------------------------
class BrachialGanglion(nn.Module):
    def __init__(self, sensory_dim, central_dim, motor_dim):
        super().__init__()
        self.reflex_arc = ReflexArc(sensory_dim, motor_dim)
        self.reflex_weight = nn.Parameter(torch.tensor(1.0))  # reflexシナプス強度
        self.central_fc = nn.Linear(central_dim, motor_dim)
        self.central_weight = nn.Parameter(torch.tensor(1.0))  # centralシナプス強度

    def forward(self, sensory_input, central_cmd):  # type: ignore[override]
        reflex_out = self.reflex_arc(sensory_input)
        central_out = torch.tanh(self.central_fc(central_cmd))
        # 学習可能な重みで加算統合
        motor = self.reflex_weight * reflex_out + self.central_weight * central_out
        return motor


class CephalopodBrainV3(nn.Module):
    """
    頭足類モデル（再入ループ付き）
    - optic lobe（視葉）: 視覚入力と垂直葉出力との再帰相互作用
    - vertical lobe（垂直葉）: 視覚＋体性感覚の連合、抑制競合付きHebbian学習
    - peduncle lobe（柄葉）: 小脳様前向きモデル（誤差補正）
    - brachial ganglia（腕神経節）: 反射経路＋中央コマンド統合
    """

    def __init__(self, arms=8, central_dim=64, motor_dim=4, re_loops=3):
        super().__init__()
        self.arms = arms
        self.central_dim = central_dim
        self.motor_dim = motor_dim
        self.re_loops = re_loops
        self.sensory_dim = D_SOMATO
        self.visual_dim = D_VISION

        # --- 視葉 ---
        self.optic_in = nn.Linear(self.visual_dim, central_dim)
        self.optic_rec = nn.Linear(central_dim, central_dim, bias=False)  # 再入ループ入力
        self.optic_act = nn.Tanh()

        # --- 垂直葉 ---
        self.vert_in = nn.Linear(central_dim + self.sensory_dim, central_dim)
        self.vert_self = nn.Linear(central_dim, central_dim, bias=False)  # 再帰
        self.vert_inhib = nn.Linear(central_dim, central_dim, bias=False)  # 抑制項
        self.beta = nn.Parameter(torch.tensor(0.3))  # Hebbian強度
        self.gamma = nn.Parameter(torch.tensor(0.5))  # 抑制強度
        self.vert_norm = nn.LayerNorm(central_dim)

        # --- 柄葉（小脳様） ---
        self.peduncle_lobe = nn.Sequential(
            nn.Linear(central_dim, central_dim),
            nn.Tanh(),
        )

        # --- 腕ガングリオン ---
        self.brachial_ganglia = nn.ModuleList(
            [BrachialGanglion(self.sensory_dim, central_dim, motor_dim) for _ in range(self.arms)]
        )

        self.out_act = nn.Tanh()

    def forward(self, x):  # type: ignore[override]
        """
        x: {"vision": [B, Dv], "somatosensory": [B, Ds]}
        return: {"motor": [B, arms*motor_dim], "central_command": [B, Dc]}
        """

        B = x["vision"].size(0)

        # 初期視葉出力
        optic = torch.tanh(self.optic_in(x["vision"]))
        vert = torch.zeros_like(optic)

        # --- 再入ループ (optic <-> vertical) ---
        for _ in range(self.re_loops):
            # 垂直葉更新：視覚＋体性感覚＋再帰
            vert_in = torch.cat([optic, x["somatosensory"]], dim=-1)
            h = torch.tanh(self.vert_in(vert_in))
            inhib = self.vert_inhib(vert)  # 抑制信号
            vert = torch.tanh(self.vert_self(vert) + h - self.gamma * inhib)
            # Hebbian項（自己相関強調）
            vert = vert + self.beta * (vert * h)
            vert = self.vert_norm(vert)

            # 視葉更新（再入）
            optic = self.optic_act(self.optic_in(x["vision"]) + self.optic_rec(vert))

        # 柄葉（小脳様）
        central_cmd = self.peduncle_lobe(vert)

        # 腕反射経路（局所統合）
        motors = []
        for i in range(self.arms):
            mi = self.brachial_ganglia[i](x["somatosensory"], central_cmd)
            motors.append(mi)
        motor = self.out_act(torch.cat(motors, dim=-1))
        return {"motor": motor, "central_command": central_cmd}
