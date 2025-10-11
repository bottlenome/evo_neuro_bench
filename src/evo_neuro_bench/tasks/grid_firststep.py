"""Grid navigation first-step classification task."""

import random
from collections import deque as _dq

import torch
import torch.nn.functional as F

from ..adapters import ModelAdapter
from ..utils import D_AUDIT, D_OLFACT, D_PROP, D_SOMATO, D_VISION, FrozenObsMixer, S_BASE, V_BASE

__all__ = ["GridPathFirstStep", "train_grid_firststep"]


# ================== Grid-Path-FirstStep ==================
class GridPathFirstStep(torch.utils.data.Dataset):
    """
    7x7グリッド: S(スタート), G(ゴール), #(障害物)
    ルール: 最短路の最初の一手 (L/F/R) を3択で答える。
    """

    def __init__(self, n=8000, size=7, device="cpu", noise=0.05, mixer=None, seed=2):
        super().__init__()
        self.device = device
        self.noise = noise
        self.N = size
        random.seed(seed)
        torch.manual_seed(seed)
        self.mixer = mixer or FrozenObsMixer()
        self.samples = []
        for _ in range(n):
            grid = torch.zeros(size, size, dtype=torch.long, device=device)
            # S,G配置
            sx, sy = random.randint(0, size - 1), random.randint(0, size - 1)
            gx, gy = random.randint(0, size - 1), random.randint(0, size - 1)
            while (gx, gy) == (sx, sy):
                gx, gy = random.randint(0, size - 1), random.randint(0, size - 1)
            grid[sy, sx] = 1
            grid[gy, gx] = 2
            # 障害物
            for _ in range(random.randint(1, size)):
                ox, oy = random.randint(0, size - 1), random.randint(0, size - 1)
                if (ox, oy) not in [(sx, sy), (gx, gy)]:
                    grid[oy, ox] = 3  # '#'

            # 最短路をBFSで探索し、一手目を決める
            dirs = [(1, 0), (0, 1), (-1, 0), (0, -1)]  # E,S,W,N（右手系）
            prev = {}
            q = _dq([(sx, sy)])
            visited = {(sx, sy)}
            found = False
            while q:
                x, y = q.popleft()
                if (x, y) == (gx, gy):
                    found = True
                    break
                for dx, dy in dirs:
                    nx, ny = x + dx, y + dy
                    if 0 <= nx < size and 0 <= ny < size and (nx, ny) not in visited and grid[ny, nx] != 3:
                        visited.add((nx, ny))
                        prev[(nx, ny)] = (x, y)
                        q.append((nx, ny))
            if not found:
                # 経路なし→ランダム方角（学習上はノイズだがロバスト性テストになる）
                step_label = random.randint(0, 2)
            else:
                # ゴールから戻って1手目を抽出
                path = []
                cur = (gx, gy)
                while cur != (sx, sy):
                    path.append(cur)
                    cur = prev[cur]
                path = path[::-1]
                first = path[0] if path else (gx, gy)
                dx, dy = first[0] - sx, first[1] - sy
                forward = (1, 0)  # 基準
                # dir to label: 左=0, 前=1, 右=2
                if (dx, dy) == forward:
                    step_label = 1
                elif (dx, dy) == (0, -1):
                    step_label = 0  # 上は左に相当（基準から見て）
                elif (dx, dy) == (0, 1):
                    step_label = 2  # 下は右
                else:
                    step_label = 0 if random.random() < 0.5 else 2

            # 視覚エンコード（one-hot平均→FrozenObsMixer→128）
            # 0:空,1:S,2:G,3:# のone-hot平均で荒い地図表現
            oh = F.one_hot(grid, num_classes=4).float().mean(dim=(0, 1))  # (4,)
            v = torch.zeros(V_BASE, device=device)
            take = min(V_BASE, oh.numel())
            v[:take] = oh[:take]
            V_full, _ = self.mixer(v.unsqueeze(0), torch.zeros(1, S_BASE, device=device))
            vision = V_full.squeeze(0) + self.noise * torch.randn(D_VISION, device=device)

            obs = {
                "vision": vision,
                "somatosensory": torch.randn(D_SOMATO, device=device) * self.noise,
                "olfaction": torch.randn(D_OLFACT, device=device) * self.noise,
                "auditory": torch.randn(D_AUDIT, device=device) * self.noise,
                "proprioception": torch.randn(D_PROP, device=device) * self.noise,
            }
            self.samples.append((obs, step_label))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, i):
        return self.samples[i]


def train_grid_firststep(adapter: ModelAdapter, device="cpu", epochs=20, batch_size=128, noise=0.05):
    ds_tr = GridPathFirstStep(8000, device=device, noise=noise)
    ds_ev = GridPathFirstStep(1200, device=device, noise=noise, seed=11)
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
        print(f"[Grid-FirstStep] epoch={ep} loss={total / max(1, cnt):.3f}")
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
    print(f"[Grid-FirstStep] eval accuracy = {acc:.3f}")
    return acc, curve
