"""Fish brain model with reflex/volitional mixing."""

import torch
import torch.nn as nn

from ..utils import D_OLFACT, D_PROP, D_SOMATO, D_VISION, MLP
from .common import Cerebellum

__all__ = ["SpinalMixer", "FishBrainV3"]


class SpinalMixer(nn.Module):
    """
    脊髄での反射/随意の調停を抽象化:
      gate ∈ [0,1]^{motor_dim} を学習し、motor = tanh( gate*reflex + (1-gate)*volitional )
    gate は（文脈依存）= g(somato, proprio, reflex, volitional)
    """

    def __init__(self, motor_dim: int, ctx_dim: int):
        super().__init__()
        # gateを出す小さなMLP（文脈: somato+proprio と両ドライブを適度に見る）
        hidden = 128
        self.ctx_proj = MLP(ctx_dim, hidden, hidden, depth=2)
        self.g_head = nn.Linear(hidden, motor_dim)
        # 初期は随意寄り(=gate小さめ)にしたい場合はbiasを負に初期化しても良い
        nn.init.constant_(self.g_head.bias, 0.0)

    def forward(self, reflex_drive, volitional_drive, ctx):  # type: ignore[override]
        h = self.ctx_proj(ctx)
        gate = torch.sigmoid(self.g_head(h))  # [B, motor_dim] in [0,1]
        motor = torch.tanh(gate * reflex_drive + (1.0 - gate) * volitional_drive)
        return motor, gate


class FishBrainV3(nn.Module):
    """
    解剖学マッピング:
      - optic_tectum (視蓋): 視覚統合 + ventral_telencephalonからの再入
      - ventral_telencephalon (前脳腹側部): 行動選択・ゲーティング
      - cerebellum (小脳様): 誤差補正・協調
      - spinal_interneurons: 反射ドライブ
      - spinal_mixer: 反射と随意の学習的調停
    改修点:
      - optic_tectum と ventral_telencephalon 間に再入ループを導入
      - ventral_telencephalon 出力を視蓋入力へ再投射（期待・文脈バイアス）
      - 反射経路は従来通り維持
    """

    def __init__(self, motor_dim=12, loops=2, hidden_dim=64):
        super().__init__()
        self.loops = loops
        self.hidden_dim = hidden_dim

        # --- 視蓋 (optic tectum) ---
        self.optic_in = nn.Linear(D_VISION, hidden_dim)
        self.optic_rec = nn.Linear(motor_dim, hidden_dim, bias=False)  # 再入投射 (BG出力)
        self.optic_act = nn.Tanh()

        # --- 前脳腹側部 (ventral telencephalon; BG様) ---
        self.bg_in = nn.Linear(hidden_dim + D_OLFACT + D_PROP, motor_dim)
        self.bg_gate = nn.Linear(motor_dim, motor_dim, bias=False)
        self.bg_act = nn.Tanh()

        # --- 小脳様補正 ---
        self.cerebellum = Cerebellum(D_SOMATO + D_PROP, motor_dim)

        # --- 反射経路 ---
        self.spinal_interneurons = MLP(D_SOMATO + D_PROP, 128, motor_dim, depth=2)

        # --- 脊髄統合 ---
        ctx_dim = D_SOMATO + D_PROP
        self.spinal_mixer = SpinalMixer(motor_dim=motor_dim, ctx_dim=ctx_dim)

    def forward(self, x):  # type: ignore[override]
        B = x["vision"].size(0)
        # 初期視蓋表象
        optic = self.optic_act(self.optic_in(x["vision"]))
        bg_out = torch.zeros(B, self.hidden_dim, device=optic.device)

        # --- optic–BG再入ループ ---
        for _ in range(self.loops):
            bg_in = torch.cat([optic, x["olfaction"], x["proprioception"]], dim=-1)
            bg_drive = self.bg_act(self.bg_in(bg_in))
            optic = self.optic_act(self.optic_in(x["vision"]) + self.optic_rec(bg_drive))

        # --- 小脳経路 ---
        z_cb = torch.cat([x["somatosensory"], x["proprioception"]], -1)
        volitional = bg_drive + self.cerebellum(z_cb, bg_drive)

        # --- 反射経路 ---
        reflex = self.spinal_interneurons(torch.cat([x["somatosensory"], x["proprioception"]], -1))

        # --- 脊髄統合 ---
        ctx = torch.cat([x["somatosensory"], x["proprioception"]], -1)
        motor, gate = self.spinal_mixer(reflex, volitional, ctx)

        return {"motor": motor, "gate": gate}
