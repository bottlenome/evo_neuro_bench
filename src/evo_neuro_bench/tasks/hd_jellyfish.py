"""High-dimensional jellyfish avoidance task."""

import random
from collections import deque

import torch
import torch.nn.functional as F

from ..adapters import ModelAdapter
from ..utils import (
    D_AUDIT,
    D_OLFACT,
    D_PROP,
    D_SOMATO,
    D_VISION,
    FrozenObsMixer,
    S_BASE,
    V_BASE,
)

__all__ = ["HDJellyfishDataset", "train_hd_jellyfish"]


# ===============================================================
# Task A: HD-Jellyfish (high-dim cue; supervised 3-way)
# ===============================================================
class HDJellyfishDataset(torch.utils.data.Dataset):
    """
    label: 0=左へ回避, 1=直進, 2=右へ回避
    - create low-dim cues on V_BASE and S_BASE, then lift to full dims.
    - other modalities are noise (same for all models).
    """

    def __init__(self, n_samples=8000, p_none=0.3, amp=(0.6, 1.3), noise=0.1, device="cpu", mixer=None):
        self.device = device
        self.noise = noise
        self.mixer = mixer or FrozenObsMixer()
        Xv, Xs, Xo, Xa, Xp, Y = [], [], [], [], [], []
        for _ in range(n_samples):
            r = random.random()
            left, right = 0.0, 0.0
            if r < p_none:
                y = 1
            else:
                if random.random() < 0.5:
                    left = random.uniform(*amp)
                    y = 2  # 右回避
                else:
                    right = random.uniform(*amp)
                    y = 0  # 左回避
            v = torch.randn(V_BASE, device=device) * (noise * 0.3)
            half = V_BASE // 2
            if left > 0:
                v[:half] += left
            if right > 0:
                v[half:] += right
            s = torch.randn(S_BASE, device=device) * (noise * 0.3)
            halfs = S_BASE // 2
            if left > 0:
                s[:halfs] += 0.3 * left
            if right > 0:
                s[halfs:] += 0.3 * right
            V_full, S_full = self.mixer(v.unsqueeze(0), s.unsqueeze(0))
            V_full = V_full.squeeze(0) + noise * torch.randn(D_VISION, device=device)
            S_full = S_full.squeeze(0) + noise * torch.randn(D_SOMATO, device=device)
            O = noise * torch.randn(D_OLFACT, device=device)
            A = noise * torch.randn(D_AUDIT, device=device)
            P = noise * torch.randn(D_PROP, device=device)
            Xv.append(V_full)
            Xs.append(S_full)
            Xo.append(O)
            Xa.append(A)
            Xp.append(P)
            Y.append(y)
        self.V = torch.stack(Xv)
        self.S = torch.stack(Xs)
        self.O = torch.stack(Xo)
        self.A = torch.stack(Xa)
        self.P = torch.stack(Xp)
        self.Y = torch.tensor(Y, dtype=torch.long, device=device)

    def __len__(self):
        return self.V.size(0)

    def __getitem__(self, idx):
        obs = {
            "vision": self.V[idx],
            "olfaction": self.O[idx],
            "somatosensory": self.S[idx],
            "auditory": self.A[idx],
            "proprioception": self.P[idx],
        }
        return obs, self.Y[idx]


def train_hd_jellyfish(adapter: ModelAdapter, device="cpu", epochs=3, batch_size=128, p_none=0.3, noise=0.1):
    mixer = FrozenObsMixer().to(device)
    ds_tr = HDJellyfishDataset(8000, p_none=p_none, noise=noise, device=device, mixer=mixer)
    ds_ev = HDJellyfishDataset(1000, p_none=p_none, noise=noise, device=device, mixer=mixer)
    dl = torch.utils.data.DataLoader(ds_tr, batch_size=batch_size, shuffle=True, drop_last=True)

    opt = torch.optim.Adam(adapter.parameters(), lr=1e-3)
    adapter.train()

    curve = []  # (step, moving_avg_loss)
    step = 0
    window = 100
    loss_window = deque(maxlen=window)

    for ep in range(1, epochs + 1):
        total, cnt = 0.0, 0
        for obs, y in dl:
            step += 1
            for k in obs:
                obs[k] = obs[k].to(device)
            y = y.to(device)

            logits, _ = adapter(obs)
            loss = F.cross_entropy(logits, y)

            opt.zero_grad()
            loss.backward()
            opt.step()

            total += loss.item() * y.size(0)
            cnt += y.size(0)

            # 移動平均でカーブを記録
            loss_window.append(loss.item())
            avg_loss = sum(loss_window) / len(loss_window)
            curve.append((step, avg_loss))

        print(f"[HD-Jellyfish] epoch={ep} loss={total / max(1, cnt):.3f}")

    # 評価
    adapter.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for i in range(len(ds_ev)):
            obs, y = ds_ev[i]
            for k in obs:
                obs[k] = obs[k].unsqueeze(0).to(device)
            y = int(y.item())
            logits, _ = adapter(obs)
            pred = logits.argmax(-1).item()
            correct += int(pred == y)
            total += 1
    acc = correct / max(1, total)
    print(f"[HD-Jellyfish] eval accuracy = {acc:.3f}")

    return acc, curve
