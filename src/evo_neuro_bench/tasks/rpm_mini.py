"""Raven's Progressive Matrices mini task."""

import random

import torch
import torch.nn.functional as F

from ..adapters import ModelAdapter
from ..utils import D_AUDIT, D_OLFACT, D_PROP, D_SOMATO, D_VISION, FrozenObsMixer, S_BASE, V_BASE

__all__ = ["RPMMiniDataset", "train_rpm_mini"]


# ================== RPM-Mini (2x2) ==================
class RPMMiniDataset(torch.utils.data.Dataset):
    """
    2x2のRaven最小版（行:ルール→合成）。6x6グリッドを128次元に射影してvisionへ。
    ルール: XNOR / XOR / COUNT のいずれか。選択肢は3つ。
    """

    def __init__(self, n=6000, grid=6, device="cpu", noise=0.05, mixer=None, seed=0):
        super().__init__()
        self.device = device
        self.noise = noise
        self.grid = grid
        torch.manual_seed(seed)
        random.seed(seed)
        self.mixer = mixer or FrozenObsMixer()
        self.samples = []
        for _ in range(n):
            rule = random.choice(["XNOR", "XOR", "COUNT"])
            # タイルをバイナリ画像(6x6)で3つ作る（T00, T01, T10）。T11が欠損。
            T00 = (torch.rand(grid, grid, device=device) > 0.65).float()
            # 次元ごとのトグル/シフト
            mask = (torch.rand_like(T00) > 0.5).float()
            T01 = T00 if rule == "COUNT" else torch.clamp(T00 + (1 - mask), 0, 1)
            T10 = torch.clamp(T00 * mask, 0, 1) if rule != "COUNT" else (torch.rand_like(T00) > 0.65).float()

            if rule == "XNOR":
                T11 = 1.0 - torch.logical_xor(T01.bool(), T10.bool()).float()  # 一致＝1
            elif rule == "XOR":
                T11 = torch.logical_xor(T01.bool(), T10.bool()).float()
            else:  # COUNT: 個数保存（T11の1の数＝T01の1の数）
                ones_target = int(T01.sum().item())
                flat = torch.zeros(grid * grid, device=device)
                idx = torch.randperm(grid * grid, device=device)[:ones_target]
                flat[idx] = 1.0
                T11 = flat.view(grid, grid)

            def vec(img):
                base = img.flatten().float()  # 36次元
                # 36→V_BASE(=8)へ圧縮 → 128へ固定射影（FrozenObsMixerに合せる）
                # ここは単純に36→128へパディング&線形でも良いが、ノイズで多様性付与
                v = torch.zeros(V_BASE, device=device)
                take = min(V_BASE, base.numel())
                v[:take] = base[:take]
                V_full, _ = self.mixer(v.unsqueeze(0), torch.zeros(1, S_BASE, device=device))
                return V_full.squeeze(0) + self.noise * torch.randn(D_VISION, device=device)

            # ビネット：T00,T01 / T10,  ?(T11)
            panel = torch.stack([vec(T00), vec(T01), vec(T10)], 0)  # (3,128)
            vision = panel.mean(0)  # 簡単化：まとめて1ベクトルに符号化

            # 候補の生成（正解1つ＋ダミー2つ）
            correct_vec = vec(T11)
            wrong1 = vec(T11.roll(shifts=1, dims=0))  # 適当な擾乱
            wrong2 = vec(1.0 - T11)  # 反転
            choices = torch.stack([correct_vec, wrong1, wrong2], 0)
            order = torch.randperm(3)
            y = int((order == 0).nonzero()[0])
            choices = choices[order]
            vision = vision + 0.25 * choices.mean(0)

            obs = {
                "vision": vision,
                "somatosensory": torch.randn(D_SOMATO, device=device) * self.noise,
                "olfaction": torch.randn(D_OLFACT, device=device) * self.noise,
                "auditory": torch.randn(D_AUDIT, device=device) * self.noise,
                "proprioception": torch.randn(D_PROP, device=device) * self.noise,
            }
            self.samples.append((obs, y))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, i):
        return self.samples[i]


def train_rpm_mini(adapter: ModelAdapter, device="cpu", epochs=30, batch_size=128, noise=0.05):
    ds_tr = RPMMiniDataset(6000, device=device, noise=noise)
    ds_ev = RPMMiniDataset(1000, device=device, noise=noise, seed=13)
    dl = torch.utils.data.DataLoader(ds_tr, batch_size=batch_size, shuffle=True, drop_last=True)
    opt = torch.optim.Adam(adapter.parameters(), lr=1e-3)
    curve = []
    trials = 0
    adapter.train()
    for ep in range(1, epochs + 1):
        total = cnt = 0
        for obs, y in dl:
            trials += len(y)
            for k in obs:
                obs[k] = obs[k].to(device)
            y = y.to(device)
            logits, _ = adapter(obs)
            loss = F.cross_entropy(logits, y)
            opt.zero_grad()
            loss.backward()
            opt.step()
            with torch.no_grad():
                acc = (logits.argmax(-1) == y).float().mean().item()
                curve.append((trials, acc))
            total += loss.item() * y.size(0)
            cnt += y.size(0)
        print(f"[RPM-Mini] epoch={ep} loss={total / max(1, cnt):.3f}")
    # eval
    adapter.eval()
    correct = total = 0
    with torch.no_grad():
        for obs, y in ds_ev:
            for k in obs:
                obs[k] = obs[k].unsqueeze(0).to(device)
            y = int(y)
            logits, _ = adapter(obs)
            pred = logits.argmax(-1).item()
            correct += int(pred == y)
            total += 1
    acc = correct / max(1, total)
    print(f"[RPM-Mini] eval accuracy = {acc:.3f}")
    return acc, curve
