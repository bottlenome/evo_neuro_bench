"""Detour navigation classification task."""

import random

import torch
import torch.nn.functional as F

from ..adapters import ModelAdapter
from ..utils import D_AUDIT, D_OLFACT, D_PROP, D_SOMATO, D_VISION

__all__ = ["HardDetourDataset", "train_detour"]


# ===============================================================
# Task: Detour (魚類空間認知タスクの再現)
# ===============================================================
# ===============================================================
# Hard Detour 課題（Cnidarianでは解けない設計）
# ===============================================================
class HardDetourDataset(torch.utils.data.Dataset):
    """
    ゴールと障害物を2次元座標に配置し、相対関係から行動を決める課題。
    label: 0=左, 1=直進, 2=右
    - Cnidarian のような線形モデルでは非線形規則を表現できず失敗するはず。
    """

    def __init__(self, n_samples=5000, field_size=5, noise=0.1, device="cpu"):
        self.device = device
        self.noise = noise
        self.samples = []
        for _ in range(n_samples):
            # ゴール座標（前方に配置）
            gx, gy = random.randint(-field_size, field_size), field_size
            # 障害物座標（ゴール手前にラダム配置）
            ox, oy = random.randint(-field_size, field_size), random.randint(1, field_size)

            # Visionベクトル初期化
            v = torch.zeros(D_VISION, device=device)

            # ゴールの位置を符号化（インデックスをmodで割り当て）
            v[(gx + field_size) % D_VISION] = 1.0
            # 障害物の位置を符号化（ゴールと異なるインデックスに負の信号）
            v[(ox + oy + field_size) % D_VISION] = -1.0

            # 正解ラベルの決定（非線形ルール）
            if abs(ox - gx) < 2 and oy < gy:
                # ゴール手前に障害物あり → 回り込みが必要
                if ox <= 0:
                    y = 2  # 障害物が左寄り → 右回り
                else:
                    y = 0  # 障害物が右寄り → 左回り
            else:
                # 障害物が進路にかかっていない → ゴール方向へ直進
                if gx < -1:
                    y = 0
                elif gx > 1:
                    y = 2
                else:
                    y = 1

            # ノイズを加える
            v += noise * torch.randn_like(v)

            obs = {
                "vision": v,
                "olfaction": torch.randn(D_OLFACT, device=device) * noise,
                "auditory": torch.randn(D_AUDIT, device=device) * noise,
                "proprioception": torch.randn(D_PROP, device=device) * noise,
                "somatosensory": torch.randn(D_SOMATO, device=device) * noise,
            }
            self.samples.append((obs, y))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


def train_detour(adapter: ModelAdapter, device="cpu", epochs=5, batch_size=128):
    ds_tr = HardDetourDataset(4000, device=device)
    ds_ev = HardDetourDataset(1000, device=device)
    dl = torch.utils.data.DataLoader(ds_tr, batch_size=batch_size, shuffle=True, drop_last=True)

    opt = torch.optim.Adam(adapter.parameters(), lr=1e-3)
    adapter.train()

    curve = []
    trial_counter = 0

    for ep in range(1, epochs + 1):
        total, cnt = 0.0, 0
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

            total += loss.item() * y.size(0)
            cnt += y.size(0)

        print(f"[HardDetour] epoch={ep} loss={total / max(1, cnt):.3f}")

    # 評価
    adapter.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for i in range(len(ds_ev)):
            obs, y = ds_ev[i]
            for k in obs:
                obs[k] = obs[k].unsqueeze(0).to(device)
            logits, _ = adapter(obs)
            pred = logits.argmax(-1).item()
            correct += int(pred == y)
            total += 1

    acc = correct / max(1, total)
    print(f"[HardDetour] eval accuracy = {acc:.3f}")
    return acc, curve
