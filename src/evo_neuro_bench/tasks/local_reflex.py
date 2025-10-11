"""Local reflex classification task."""

import random

import torch
import torch.nn.functional as F

from ..adapters import ModelAdapter
from ..utils import D_AUDIT, D_OLFACT, D_PROP, D_SOMATO, D_VISION

__all__ = ["LocalReflexDataset", "train_local_reflex"]


# ===============================================================
# Task: Local Reflex（体節接触反射）— 3クラス分類
# 文献背景：Bässler (1986) 他、stick insect などの局所反射制御
# ラベル: 0=左回避, 1=直進維持, 2=右回避（簡略化）
# ===============================================================
class LocalReflexDataset(torch.utils.data.Dataset):
    def __init__(self, n_samples=6000, segments=6, contact_p=0.4, noise=0.05, device="cpu"):
        self.device = device
        self.segments = segments
        assert D_SOMATO % segments == 0
        self.local_dim = D_SOMATO // segments
        Xv, Xo, Xa, Xp, Xs, Y = [], [], [], [], [], []
        for _ in range(n_samples):
            # どの体節に接触が入るか（複数もありうる）
            contacts = [1 if random.random() < contact_p else 0 for _ in range(segments)]
            # 左群（前半の体節）に接触が偏れば左回避、右群（後半）なら右回避、どちらもなければ直進
            left_sum = sum(contacts[: segments // 2])
            right_sum = sum(contacts[segments // 2 :])
            if left_sum > right_sum and left_sum > 0:
                y = 2  # 左接触多 → 右回避
            elif right_sum > left_sum and right_sum > 0:
                y = 0  # 右接触多 → 左回避
            else:
                y = 1  # 直進

            # somato 符号化：各体節ブロックの最初の1次元に接触強度、それ以外はノイズ
            s = torch.randn(D_SOMATO, device=device) * noise
            for i, c in enumerate(contacts):
                if c:
                    start = i * self.local_dim
                    s[start] = 1.0  # 接触フラグ

            obs = {
                "vision": torch.randn(D_VISION, device=device) * noise,  # 使わない
                "olfaction": torch.randn(D_OLFACT, device=device) * noise,
                "auditory": torch.randn(D_AUDIT, device=device) * noise,
                "proprioception": torch.randn(D_PROP, device=device) * noise,
                "somatosensory": s,
            }
            Xv.append(obs["vision"])
            Xo.append(obs["olfaction"])
            Xa.append(obs["auditory"])
            Xp.append(obs["proprioception"])
            Xs.append(s)
            Y.append(y)
        self.V = torch.stack(Xv)
        self.O = torch.stack(Xo)
        self.Au = torch.stack(Xa)
        self.P = torch.stack(Xp)
        self.S = torch.stack(Xs)
        self.Y = torch.tensor(Y, dtype=torch.long, device=device)

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


def train_local_reflex(adapter: ModelAdapter, device="cpu", epochs=4, batch_size=128, lr=1e-3):
    ds_tr = LocalReflexDataset(n_samples=6000, device=device)
    ds_ev = LocalReflexDataset(n_samples=1000, device=device)
    dl = torch.utils.data.DataLoader(ds_tr, batch_size=batch_size, shuffle=True, drop_last=True)

    opt = torch.optim.Adam(adapter.parameters(), lr=lr)
    curve = []
    trial_counter = 0

    for ep in range(1, epochs + 1):
        for obs, y in dl:
            trial_counter += len(y)
            for k in obs:
                obs[k] = obs[k].to(device)
            y = y.to(device)
            logits, _ = adapter(obs)
            loss = F.cross_entropy(logits, y)
            opt.zero_grad()
            loss.backward()
            opt.step()

            with torch.no_grad():
                pred = logits.argmax(-1)
                acc_batch = (pred == y).float().mean().item()
                curve.append((trial_counter, acc_batch))

        print(f"[LocalReflex] epoch={ep} loss={loss.item():.3f}")

    # 評価
    adapter.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for obs, y in ds_ev:
            for k in obs:
                obs[k] = obs[k].unsqueeze(0).to(device)
            y = int(y.item())
            logits, _ = adapter(obs)
            pred = logits.argmax(-1).item()
            correct += int(pred == y)
            total += 1
    acc = correct / max(1, total)
    print(f"[LocalReflex] eval accuracy = {acc:.3f}")
    return acc, curve
