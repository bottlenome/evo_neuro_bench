"""ARC-like pattern completion mini task."""

import random

import torch
import torch.nn.functional as F

from ..adapters import ModelAdapter
from ..utils import D_AUDIT, D_OLFACT, D_PROP, D_SOMATO, D_VISION, FrozenObsMixer, S_BASE, V_BASE

__all__ = ["ARCMiniDataset", "train_arc_mini"]


class ARCMiniDataset(torch.utils.data.Dataset):
    def __init__(self, n=6000, grid=6, k=3, device="cpu", noise=0.05, mixer=None, seed=7):
        super().__init__()
        self.device = device
        self.noise = noise
        self.k = k
        self.grid = grid
        random.seed(seed)
        torch.manual_seed(seed)
        self.mixer = mixer or FrozenObsMixer()
        self.samples = []
        for _ in range(n):
            rule = random.choice(["MAJ", "CCNT", "PARITY"])

            def rand_grid():
                return torch.randint(0, self.k, (grid, grid), device=device)

            def apply(g):
                if rule == "MAJ":
                    # 全体で最多色に塗りつぶし
                    vals, counts = torch.unique(g, return_counts=True)
                    c = vals[torch.argmax(counts)].item()
                    return torch.full_like(g, int(c))
                elif rule == "CCNT":
                    # 1の連結成分数を保つ(粗い：1色以外は0)
                    bin = (g == 1).float()
                    # ここでは近似として総数をそのまま別の場所に散布
                    ones = int(bin.sum().item())
                    out = torch.zeros_like(g)
                    idx = torch.randperm(g.numel(), device=device)[:ones]
                    out.view(-1)[idx] = 1
                    return out.long()
                else:  # PARITY
                    # 偶数行:そのまま、奇数行:色を反転(mod k)
                    out = g.clone()
                    out[1::2] = (self.k - 1 - out[1::2]) % self.k
                    return out

            # 2つの例 (in_i -> out_i)
            in1, in2 = rand_grid(), rand_grid()
            out1, out2 = apply(in1), apply(in2)
            # クエリ入力
            in3 = rand_grid()
            out3 = apply(in3)

            def enc(g):
                # 色をone-hot→平均→V_BASEへ落としてFrozenObsMixerで128へ
                oh = F.one_hot(g, num_classes=self.k).float().mean(dim=(0, 1))  # (k,)
                v = torch.zeros(V_BASE, device=device)
                take = min(V_BASE, oh.numel())
                v[:take] = oh[:take]
                V_full, _ = self.mixer(v.unsqueeze(0), torch.zeros(1, S_BASE, device=device))
                return V_full.squeeze(0) + self.noise * torch.randn(D_VISION, device=device)

            # 例をvisionに埋める（in,outを平均加算）
            vision = enc(in1) + enc(out1) + enc(in2) + enc(out2) + 0.5 * enc(in3)

            # 候補作成（正解＋2ダミー）
            correct = enc(out3)
            wrong1 = enc((out3 + 1) % self.k)  # 色シフト
            wrong2 = enc(out3.roll(shifts=1, dims=0))  # 粗擾乱（意味なしでもOK）
            choices = torch.stack([correct, wrong1, wrong2], 0)
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


def train_arc_mini(adapter: ModelAdapter, device="cpu", epochs=30, batch_size=128, noise=0.05):
    ds_tr = ARCMiniDataset(6000, device=device, noise=noise)
    ds_ev = ARCMiniDataset(1000, device=device, noise=noise, seed=13)
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
        print(f"[ARC-Mini] epoch={ep} loss={total / max(1, cnt):.3f}")
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
    print(f"[ARC-Mini] eval accuracy = {acc:.3f}")
    return acc, curve
