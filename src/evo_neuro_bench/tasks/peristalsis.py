"""Peristalsis regression task."""

import math
import random

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..utils import D_AUDIT, D_OLFACT, D_PROP, D_SOMATO, D_VISION, make_obs

__all__ = ["PeristalsisDataset", "phase_corr", "train_peristalsis"]


# ===============================================================
# Task: Peristalsis (蠕動波出力) — 回帰（MSE）
# 目標：各体節のモータ出力が位相差φのある正弦波を形成
# 文献背景：体節CPG/体節間結合（例：Friesen & Pearce 2007, leech locomotor circuits）
# ===============================================================
class PeristalsisDataset(torch.utils.data.Dataset):
    """
    回帰タスク：motor_target ∈ R^(motor_dim)
    入力obsは somato と proprio に時間・目標速度などを符号化（単純ノイズでもOK）。
    """

    def __init__(
        self,
        motor_dim,
        n_samples=6000,
        A=1.0,
        omega=0.4,
        phi=0.8,
        noise=0.05,
        device="cpu",
    ):
        self.device = device
        self.motor_dim = motor_dim
        self.A, self.omega, self.phi = A, omega, phi
        self.noise = noise
        self.T = []
        self.targets = []
        Xv, Xo, Xa, Xp, Xs = [], [], [], [], []
        for _ in range(n_samples):
            t = random.uniform(0, 2 * math.pi)
            target = torch.tensor(
                [A * math.sin(omega * t + phi * i) for i in range(motor_dim)],
                device=device,
                dtype=torch.float32,
            )
            # 観測は最小限：somato/proprioに t, omega, phi を雑に埋める（モデル間で公平）
            v = torch.randn(D_VISION, device=device) * self.noise
            o = torch.randn(D_OLFACT, device=device) * self.noise
            a = torch.randn(D_AUDIT, device=device) * self.noise
            p = torch.randn(D_PROP, device=device) * self.noise
            s = torch.randn(D_SOMATO, device=device) * self.noise
            # t, omega, phi を少数の次元に書き込む
            p[:3] = torch.tensor([t, omega, phi], device=device)
            Xv.append(v)
            Xo.append(o)
            Xa.append(a)
            Xp.append(p)
            Xs.append(s)
            self.targets.append(target)
        self.V = torch.stack(Xv)
        self.O = torch.stack(Xo)
        self.Au = torch.stack(Xa)
        self.P = torch.stack(Xp)
        self.S = torch.stack(Xs)
        self.Y = torch.stack(self.targets)

    def __len__(self):
        return self.V.size(0)

    def __getitem__(self, idx):
        obs = {
            "vision": self.V[idx],
            "olfaction": self.O[idx],
            "auditory": self.Au[idx],
            "proprioception": self.P[idx],
            "somatosensory": self.S[idx],
        }
        return obs, self.Y[idx]


@torch.no_grad()
def phase_corr(y_pred, y_true):
    # 各体節系列の位相整合をざっくり測る（相関係数の平均）
    num = (y_pred * y_true).sum(-1)
    den = y_pred.pow(2).sum(-1).sqrt() * y_true.pow(2).sum(-1).sqrt() + 1e-9
    return (num / den).mean().item()


def train_peristalsis(base_model: nn.Module, device="cpu", epochs=5, batch_size=128, lr=1e-3):
    dummy = make_obs(batch=2, device=device)
    motor_dim = base_model(dummy)["motor"].size(-1)
    ds_tr = PeristalsisDataset(motor_dim, n_samples=6000, device=device)
    ds_ev = PeristalsisDataset(motor_dim, n_samples=800, device=device)
    dl = torch.utils.data.DataLoader(ds_tr, batch_size=batch_size, shuffle=True, drop_last=True)

    opt = torch.optim.Adam(base_model.parameters(), lr=lr)
    curve = []  # (trial_idx, "accuracy") の形式に近づける
    trial_counter = 0

    for ep in range(1, epochs + 1):
        for obs, target in dl:
            trial_counter += len(target)
            for k in obs:
                obs[k] = obs[k].to(device)
            target = target.to(device)
            pred = base_model(obs)["motor"]
            loss = F.mse_loss(pred, target)
            opt.zero_grad()
            loss.backward()
            opt.step()

            # phaseCorr を「精度指標」とみなして curve に記録
            acc_batch = phase_corr(pred, target)  # 0〜1 に近い値
            curve.append((trial_counter, acc_batch))
        print(f"[Peristalsis] epoch={ep} loss={loss.item():.4f}")

    # 評価
    base_model.eval()
    mse_tot, corr_tot, N = 0.0, 0.0, 0
    with torch.no_grad():
        for obs, target in ds_ev:
            for k in obs:
                obs[k] = obs[k].unsqueeze(0).to(device)
            target = target.unsqueeze(0).to(device)
            pred = base_model(obs)["motor"]
            mse_tot += F.mse_loss(pred, target).item()
            corr_tot += phase_corr(pred.squeeze(0), target.squeeze(0))
            N += 1
    mse_eval = mse_tot / max(1, N)
    corr_eval = corr_tot / max(1, N)
    print(f"[Peristalsis] eval MSE={mse_eval:.4f}, phaseCorr={corr_eval:.3f}")

    return mse_eval, corr_eval, curve
