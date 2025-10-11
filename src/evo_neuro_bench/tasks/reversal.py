"""Reversal learning environment and trainer."""

import random
from collections import deque

import torch
from torch.distributions import Categorical

from ..adapters import ModelAdapter
from ..utils import D_VISION, make_obs

__all__ = ["ReversalEnv", "train_reversal"]


# ===============================================================
# Task B: Reversal Learning (2AFC RL)
# ===============================================================
class ReversalEnv:
    def __init__(self, dim=D_VISION, noise=0.8, rev_at=1500, device="cpu"):
        self.dim, self.noise, self.rev_at, self.device = dim, noise, rev_at, device
        self.mu = torch.stack([torch.randn(dim), torch.randn(dim)], 0).to(device)
        self.t = 0

    def step(self):
        cls = random.randint(0, 1)
        v = self.mu[cls] + self.noise * torch.randn(self.dim, device=self.device)
        obs = make_obs(vision=v.unsqueeze(0), device=self.device)
        correct = cls if self.t < self.rev_at else 1 - cls
        self.t += 1
        return obs, correct


def train_reversal(adapter: ModelAdapter, steps=2000, lr=1e-3, device="cpu"):
    env = ReversalEnv(device=device)
    opt = torch.optim.Adam(adapter.parameters(), lr=lr)
    baseline = 0.0
    beta = 0.02

    pre_acc, post_acc = [], []
    curve = []  # (step, moving_avg_acc)

    adapter.train()
    adapter.reset_memory(1, device)

    window = 100  # 移動平均用
    acc_window = deque(maxlen=window)

    for i in range(1, steps + 1):
        obs, correct = env.step()
        logits, _ = adapter(obs)
        dist = Categorical(logits=logits)  # 温度調整するなら logits/τ
        act = dist.sample()
        rew = 1.0 if int(act) == correct else 0.0
        adv = rew - baseline

        loss = -(dist.log_prob(act) * adv)
        opt.zero_grad()
        loss.backward()
        opt.step()

        baseline = (1 - beta) * baseline + beta * rew

        # 精度記録
        acc = 1.0 if int(act) == correct else 0.0
        (pre_acc if i < env.rev_at else post_acc).append(acc)
        acc_window.append(acc)
        avg_acc = sum(acc_window) / len(acc_window)
        curve.append((i, avg_acc))

        if i % 500 == 0:
            print(
                f"[Reversal] step={i:4d} "
                f"pre_acc={sum(pre_acc[-500:]) / max(1, len(pre_acc[-500:])):.2f} "
                f"post_acc={sum(post_acc[-500:]) / max(1, len(post_acc[-500:])):.2f}"
            )

    pa = sum(pre_acc[-500:]) / max(1, len(pre_acc[-500:]))
    po = sum(post_acc[-500:]) / max(1, len(post_acc[-500:]))

    return float(pa), float(po), curve
