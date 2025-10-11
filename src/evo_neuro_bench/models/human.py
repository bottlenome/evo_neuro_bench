"""Human cortex inspired model."""

import torch
import torch.nn as nn

from ..utils import D_AUDIT, D_PROP, D_SOMATO, D_VISION, MLP
from .common import Cerebellum
from .fish import SpinalMixer

__all__ = ["HumanCortexV4"]


class HumanCortexV4(nn.Module):
    """
    解剖学マッピング:
      - 感覚統合皮質 (V/S/P/A)
      - PFC: 再帰的作業表象 + Thalamic再入 + BGゲート
      - 海馬様統合: 文脈保持
      - 小脳: 予測誤差補正
      - 脊髄反射＋統合: Reflex/Volitionalの調停
    改修:
      - PFC ↔ Thalamus の再入ループを導入
      - Thalamus出力はBGゲートを介してPFCへ再投射
      - Descending制御ゲインをSpinalMixerに追加
    """

    def __init__(self, motor_dim=20, d_emb=64, wm_dim=128, loops=2):
        super().__init__()
        self.motor_dim = motor_dim
        self.wm_dim = wm_dim
        self.d_emb = d_emb
        self.loops = loops

        # 感覚埋め込み
        self.tv = MLP(D_VISION, 128, d_emb, depth=2)
        self.ts = MLP(D_SOMATO, 128, d_emb, depth=2)
        self.tp = MLP(D_PROP, 64, d_emb, depth=2)
        self.ta = MLP(D_AUDIT, 64, d_emb, depth=2)

        # PFC (再入対象)
        self.pfc_in = nn.Linear(d_emb * 4, wm_dim)
        self.pfc_rec = nn.Linear(wm_dim, wm_dim, bias=False)
        self.pfc_norm = nn.LayerNorm(wm_dim)

        # Thalamus + BG Gate
        self.thalamus = nn.Linear(wm_dim, wm_dim, bias=False)
        self.bg_gate = nn.Sequential(
            nn.Linear(D_SOMATO + D_PROP, wm_dim),
            nn.Tanh(),
            nn.Linear(wm_dim, wm_dim),
            nn.Sigmoid(),
        )

        # 海馬様統合（既存）
        enc_layer = nn.TransformerEncoderLayer(
            d_model=wm_dim, nhead=4, dim_feedforward=256
        )
        self.hippocampal = nn.TransformerEncoder(enc_layer, num_layers=1)
        self.h_norm = nn.LayerNorm(wm_dim)

        # 小脳様補正
        self.cerebellum = Cerebellum(D_SOMATO + D_PROP, motor_dim)

        # 反射系
        self.spinal_reflex = MLP(D_SOMATO + D_PROP, 128, motor_dim, depth=2)
        self.spinal_mixer = SpinalMixer(motor_dim, D_SOMATO + D_PROP)

        # 下行性ゲイン
        self.desc_gain = nn.Sequential(nn.Linear(wm_dim, motor_dim), nn.Tanh())

    def forward(self, x):  # type: ignore[override]
        B = x["vision"].size(0)

        # 感覚統合
        s = torch.cat(
            [
                self.tv(x["vision"]),
                self.ts(x["somatosensory"]),
                self.tp(x["proprioception"]),
                self.ta(x["auditory"]),
            ],
            dim=-1,
        )

        # PFC初期化
        pfc = torch.tanh(self.pfc_in(s))
        ctx_bg = torch.cat([x["somatosensory"], x["proprioception"]], -1)

        # --- 再入ループ (PFC↔Thalamus with BG gate) ---
        for _ in range(self.loops):
            th = self.thalamus(pfc)
            g = self.bg_gate(ctx_bg)
            pfc = torch.tanh(self.pfc_in(s) + self.pfc_rec(th * g))
            pfc = self.pfc_norm(pfc)

        # 海馬様統合
        mem = self.hippocampal(pfc.unsqueeze(0)).squeeze(0)
        mem = self.h_norm(mem)

        # 小脳予測補正
        cb_inp = torch.cat([x["somatosensory"], x["proprioception"]], -1)
        volitional = self.cerebellum(cb_inp, self.desc_gain(mem))

        # 反射ルート
        reflex = self.spinal_reflex(torch.cat([x["somatosensory"], x["proprioception"]], -1))

        # 脊髄統合
        ctx = torch.cat([x["somatosensory"], x["proprioception"]], -1)
        motor, gate = self.spinal_mixer(reflex, volitional, ctx)

        return {"motor": motor, "gate": gate, "wm": mem}
